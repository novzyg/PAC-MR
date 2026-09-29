"""Multi-GPU queue, one independent two-stage run per slot; validation CSV only."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(dim=256, graph_layers=2, attention_heads=4, dropout=.2, batch_size=4, eval_batch_size=8,
                base_epochs=30, adjust_epochs=20, patience=8, base_lr=.0003,
                adjust_lr=.001, ddi_weight=1., seed=42, threshold=.5)
METRICS = ['jaccard','ddi','f1','prauc','precision','recall','avg_med','conflicts_per_visit']


def dump(path, obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
    temp.replace(path)


def expand(path):
    cfg=json.loads(path.read_text())
    if set(cfg)-{'fixed','grid'}: raise ValueError('Only fixed/grid sections are supported')
    fixed,grid=cfg.get('fixed',{}),cfg.get('grid',{})
    if (set(fixed)|set(grid))-set(DEFAULTS): raise ValueError('Unknown hyperparameter')
    if set(fixed)&set(grid): raise ValueError('Parameter appears in both fixed and grid')
    if any(not isinstance(v,list) or not v for v in grid.values()): raise ValueError('Grid values must be nonempty lists')
    configs=[]
    for values in itertools.product(*grid.values()):
        p={**DEFAULTS,**fixed,**dict(zip(grid,values))}
        for k in ['graph_layers','attention_heads','dim','batch_size','eval_batch_size','base_epochs','adjust_epochs','patience']:
            if not isinstance(p[k],int) or isinstance(p[k],bool) or p[k]<1: raise ValueError(k)
        if p['dim'] % p['attention_heads']: raise ValueError('dim must be divisible by attention_heads')
        if not isinstance(p['seed'],int): raise ValueError('seed must be integer')
        if not 0<=p['dropout']<1 or not 0<p['threshold']<1: raise ValueError('dropout/threshold')
        if p['base_lr']<=0 or p['adjust_lr']<=0 or p['ddi_weight']<0: raise ValueError('lr/ddi_weight')
        if p not in configs: configs.append(p)
    return configs


def command(job, phase, a):
    p=job['params']; folder=Path(job['directory'])
    cmd=[sys.executable,str(ROOT/'src/main.py'),'train',
         '--data',str(a.data),'--split',str(folder/'split.json'),
         '--split-mode','ordered','--split-seed','2026','--device','cuda:0',
         '--dim',str(p['dim']),'--graph-layers',str(p['graph_layers']),
         '--attention-heads',str(p['attention_heads']),'--dropout',str(p['dropout']),
         '--batch-size',str(p['batch_size']),'--eval-batch-size',str(p['eval_batch_size']),
         '--seed',str(p['seed']),'--threshold-mode','fixed','--threshold',str(p['threshold']),
         '--patience',str(p['patience']),'--aux-weight','0','--resume',
         '--output',str(folder/phase)]
    if phase=='base':
        cmd+=['--variant','base','--epochs',str(p['base_epochs']),'--lr',str(p['base_lr']),'--ddi-weight','0']
    else:
        cmd+=['--variant','full','--init',str(folder/'base/best.pt'),
              '--epochs',str(p['adjust_epochs']),'--lr',str(p['adjust_lr']),
              '--ddi-weight',str(p['ddi_weight']),'--weight-decay','0']
    if a.smoke: cmd+=['--smoke']
    return cmd


def summary(out,jobs):
    rows=[]
    for job in jobs:
        row={k:job[k] for k in ['run_id','status','gpu','directory']}
        row.update(job['params'])
        row['error']=job.get('error','')
        for phase in ['base','full']:
            path=Path(job['directory'])/phase/'DONE.json'
            if path.exists():
                done=json.loads(path.read_text());row[phase+'_best_epoch']=done['best_epoch']
                for k in METRICS: row[phase+'_val_'+k]=done['validation'][k]
        rows.append(row)
    fields=['run_id','status','gpu','directory',*DEFAULTS,'error']
    for phase in ['base','full']: fields += [phase+'_best_epoch']+[phase+'_val_'+k for k in METRICS]
    tmp=out/'results.csv.tmp'
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(out/'results.csv')
    dump(out/'status.json',jobs)


def main(a):
    configs=expand(a.grid)
    a.data=a.data.resolve();a.output=a.output.resolve()
    jobs=[]
    for p in configs:
        digest=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12]
        jobs.append(dict(run_id='run_'+digest,params=p,status='pending',gpu='',directory=str(a.output/('run_'+digest))))
    print(f'{len(jobs)} configurations; {2*len(jobs)} training stages. GPUs: {a.gpus}',flush=True)
    if a.dry_run:
        for j in jobs: print(j['run_id'],json.dumps(j['params']))
        return
    import torch
    gpus=[g.strip() for g in a.gpus.split(',')]
    if not gpus or len(set(gpus))!=len(gpus) or any(not g.isdigit() for g in gpus): raise ValueError('Use unique integer GPU IDs, e.g. 0,1')
    if os.environ.get('CUDA_VISIBLE_DEVICES'): raise ValueError('Unset CUDA_VISIBLE_DEVICES; specify physical indices using --gpus')
    if any(int(g)>=torch.cuda.device_count() for g in gpus): raise ValueError('Requested GPU is unavailable')
    for n in ['records_final.pkl','voc_final.pkl','ddi_A_final.pkl']:
        if not (a.data/n).is_file(): raise FileNotFoundError(a.data/n)
    a.output.mkdir(parents=True,exist_ok=True)
    lock=a.output/'GRID_RUNNING.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    active={}
    offsets={}
    try:
        # Pin source and data for safe reruns; GPUs may change across resumes.
        def sha(p):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            return h.hexdigest()
        manifest=dict(configurations=configs,data=str(a.data),smoke=a.smoke,
                      sources={p.name:sha(p) for p in sorted((ROOT/'src').glob('*.py'))},
                      fingerprints={n:sha(a.data/n) for n in ['records_final.pkl','voc_final.pkl','ddi_A_final.pkl']},
                      evaluation='validation only, fixed threshold, ordered patient split')
        mp=a.output/'manifest.json'
        if mp.exists() and json.loads(mp.read_text())!=manifest: raise ValueError('Grid/data/source changed; choose a new output directory')
        dump(mp,manifest);summary(a.output,jobs)
        pending=list(jobs)
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending: continue
                j=pending.pop(0);folder=Path(j['directory']);folder.mkdir(exist_ok=True)
                j['gpu']=gpu;j['status']='running_base'
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
                # Every worker sees its assigned physical GPU as cuda:0.
                env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
                log=(folder/'train.log').open('a',buffering=1)
                offsets[j['run_id']]=(folder/'train.log').stat().st_size
                cmd=command(j,'base',a);log.write('\nCOMMAND '+json.dumps(cmd)+'\n')
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(j,'base',proc,log,env)
                print('START',j['run_id'],'GPU',gpu,flush=True);summary(a.output,jobs)
            for gpu,(j,phase,proc,log,env) in list(active.items()):
                code=proc.poll()
                # Read complete new lines only; prefix GPU/run for interleaved output.
                with (Path(j['directory'])/'train.log').open() as reader:
                    reader.seek(offsets[j['run_id']])
                    while True:
                        pos=reader.tell();line=reader.readline()
                        if not line or (not line.endswith('\n') and code is None):
                            reader.seek(pos);break
                        if line.strip() and not line.startswith('COMMAND '):
                            print(f"[GPU {gpu} | {j['run_id']}] {line.rstrip()}",flush=True)
                    offsets[j['run_id']]=reader.tell()
                if code is None:continue
                if code==0 and phase=='base':
                    cmd=command(j,'full',a);log.write('\nCOMMAND '+json.dumps(cmd)+'\n')
                    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    j['status']='running_full';active[gpu]=(j,'full',child,log,env)
                else:
                    j['status']='complete' if code==0 else 'failed'
                    if code: j['error']=f'{phase} exit {code}; see train.log'
                    log.close();del active[gpu]
                    print(j['status'].upper(),j['run_id'],flush=True)
                summary(a.output,jobs)
            if active:time.sleep(1)
        print('Results:',a.output/'results.csv',flush=True)
        if any(j['status']=='failed' for j in jobs):raise RuntimeError('Some runs failed; inspect CSV and per-run logs')
    finally:
        # Graceful interrupt gives training code a chance to release its lock.
        for j,phase,proc,log,env in active.values():
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
        for j,phase,proc,log,env in active.values():
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
            log.close();j['status']='interrupted'
        if active:summary(a.output,jobs)
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--grid',type=Path,default=ROOT/'configs/grid.json')
    p.add_argument('--gpus',default='0')
    p.add_argument('--data',type=Path,default=ROOT/'data/mimic-iv_all_all')
    p.add_argument('--output',type=Path,default=ROOT/'saved/grid_v1')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main(args)
