"""One-factor sweeps with one shared base per seed; train and test are separate."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import sha256


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def write_csv(path, rows):
    if rows:
        temp = path.with_suffix(path.suffix + '.tmp')
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)


def expand(cfg):
    expected = {'dim', 'graph_layers', 'attention_heads', 'dropout', 'batch_size',
                'eval_batch_size', 'base_epochs', 'adjust_epochs', 'patience',
                'base_lr', 'adjust_lr', 'ddi_weight', 'displacement_cap', 'cost_ratio'}
    if set(cfg) != {'fixed', 'sweeps', 'seeds', 'thresholds'} or set(cfg['fixed']) != expected:
        raise ValueError('Configuration keys differ from sensitivity.json schema')
    if set(cfg['sweeps']) != {'ddi_weight', 'displacement_cap', 'cost_ratio'}:
        raise ValueError('Expected the three supported single-factor sweeps')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']) or any(type(s) is not int for s in cfg['seeds']):
        raise ValueError('Seeds must be unique integers')
    if not cfg['thresholds'] or len(set(cfg['thresholds'])) != len(cfg['thresholds']) or any(not 0 < t < 1 for t in cfg['thresholds']):
        raise ValueError('Thresholds must be unique and between 0 and 1')
    configs = [dict(cfg['fixed'])]
    for name, values in cfg['sweeps'].items():
        if not values:
            raise ValueError('Empty sweep')
        for value in values:
            p = {**cfg['fixed'], name: value}
            if p not in configs:
                configs.append(p)
    for p in configs:
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in p.values()):
            raise ValueError('All hyperparameters must be finite numbers')
        for k in ['dim', 'graph_layers', 'attention_heads', 'batch_size', 'eval_batch_size', 'base_epochs', 'adjust_epochs', 'patience']:
            if type(p[k]) is not int or p[k] < 1:
                raise ValueError(k)
        if p['dim'] % p['attention_heads'] or not 0 <= p['dropout'] < 1:
            raise ValueError('Invalid attention dimensions or dropout')
        if min(p['base_lr'], p['adjust_lr'], p['displacement_cap']) <= 0 or p['ddi_weight'] < 0 or p['cost_ratio'] < 1:
            raise ValueError('Invalid learning rate, DDI weight, cap or cost ratio')
    return configs


def command(a, p, seed, folder, base):
    cmd = [sys.executable, str(ROOT/'src/main.py'), 'train',
           '--data', str(a.data), '--split', str(base.parent/'split.json'),
           '--split-mode', 'ordered', '--split-seed', '2026', '--seed', str(seed),
           '--device', a.device, '--threshold-mode', 'fixed', '--threshold', '0.5',
           '--selection', 'accuracy', '--aux-weight', '0', '--resume', '--output', str(folder)]
    for name in ['dim', 'graph_layers', 'attention_heads', 'dropout', 'batch_size', 'eval_batch_size', 'patience', 'displacement_cap', 'cost_ratio']:
        cmd += ['--'+name.replace('_', '-'), str(p[name])]
    if folder == base:
        cmd += ['--variant', 'base', '--epochs', str(p['base_epochs']), '--lr', str(p['base_lr']),
                '--weight-decay', '0.00001', '--ddi-weight', '0']
    else:
        cmd += ['--variant', 'full', '--init', str(base/'best.pt'), '--epochs', str(p['adjust_epochs']),
                '--lr', str(p['adjust_lr']), '--weight-decay', '0', '--ddi-weight', str(p['ddi_weight'])]
    if a.smoke:
        cmd += ['--smoke', '--epochs', '1']
    return cmd


def run(cmd, log):
    print('COMMAND', ' '.join(cmd), flush=True)
    with log.open('a') as f:
        f.write('\nCOMMAND '+json.dumps(cmd)+'\n'); f.flush()
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            try:
                for line in proc.stdout:
                    print(line, end='', flush=True); f.write(line); f.flush()
                code = proc.wait()
                if code:
                    raise RuntimeError(f'Command failed ({code}); see {log}')
            except BaseException:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                raise


def report(a, jobs, split, adjacency):
    import numpy as np
    from metrics import metrics, ranking_metrics
    rows = []
    for seed, name, p, folder in jobs:
        path = folder/f'{split}_predictions.npz'
        if not path.exists():
            continue
        with np.load(path) as prediction:
            ranking = ranking_metrics(prediction['y'], prediction['probability'], prediction['patient'])
            for threshold in a.cfg['thresholds']:
                result = metrics(prediction['y'], prediction['probability'], prediction['patient'], adjacency, threshold, ranking=ranking)
                rows.append(dict(seed=seed, configuration=name, ddi_weight=p['ddi_weight'],
                                 displacement_cap=p['displacement_cap'], cost_ratio=p['cost_ratio'], **result))
    write_csv(a.output/f'{split}_thresholds.csv', rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row['configuration'], row['threshold']), []).append(row)
    summaries = []
    for (name, threshold), values in grouped.items():
        entry = dict(configuration=name, threshold=threshold, seeds=len(values))
        for k in ['ddi_weight', 'displacement_cap', 'cost_ratio']:
            entry[k] = values[0][k]
        for k in ['jaccard', 'ddi', 'f1', 'prauc', 'precision', 'recall', 'avg_med', 'conflicts_per_visit']:
            numbers = [v[k] for v in values]
            entry[k+'_mean'] = float(np.mean(numbers))
            entry[k+'_std'] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else ''
        summaries.append(entry)
    write_csv(a.output/f'{split}_summary.csv', summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['train', 'test'])
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sensitivity.json')
    parser.add_argument('--output', type=Path, default=ROOT/'saved/sensitivity_v1')
    parser.add_argument('--data', type=Path, default=ROOT/'data/mimic-iv_all_all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', help='Override seeds, e.g. 1023,2024,42; keep identical for train/test')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    a = parser.parse_args()
    a.cfg = json.loads(a.config.read_text())
    if a.seeds:
        a.cfg['seeds'] = [int(s) for s in a.seeds.split(',')]
    configs = expand(a.cfg)
    a.output = a.output.resolve(); a.data = a.data.resolve()
    jobs = []
    for seed in a.cfg['seeds']:
        for i, p in enumerate(configs):
            name = f'config_{i:02d}'
            jobs.append((seed, name, p, a.output/f'seed_{seed}'/name/'full'))
    print(f"{len(a.cfg['seeds'])} base trainings + {len(jobs)} adjustment trainings; "
          f"{len(a.cfg['thresholds'])} evaluation thresholds. Sequential on {a.device}.", flush=True)
    if a.dry_run:
        for i, p in enumerate(configs):
            print(f'config_{i:02d}', json.dumps(p))
        print('Thresholds:', a.cfg['thresholds'])
        return
    import torch
    from data import load_data
    torch.set_num_threads(4)
    manifest = dict(config=a.cfg, data=str(a.data), smoke=a.smoke,
                    sources={str(p.relative_to(ROOT)): sha256(p) for p in [*sorted((ROOT/'src').glob('*.py')), Path(__file__).resolve()]},
                    fingerprints={n: sha256(a.data/n) for n in ['records_final.pkl', 'voc_final.pkl', 'ddi_A_final.pkl']},
                    selection='Per-stage max validation Jaccard at fixed threshold 0.5; test thresholds predeclared')
    mp = a.output/'manifest.json'
    if a.phase == 'test' and not mp.exists():
        raise FileNotFoundError('Train first; no experiment manifest found')
    a.output.mkdir(parents=True, exist_ok=True)
    lock = a.output/'SENSITIVITY_RUNNING.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        if mp.exists() and json.loads(mp.read_text()) != manifest:
            raise ValueError('Config/source/data changed; use a new output directory')
        dump(mp, manifest)
        dump(a.output/'configurations.json', [dict(configuration=f'config_{i:02d}', **p) for i, p in enumerate(configs)])
        if a.phase == 'test':
            for _, _, _, folder in jobs:
                if not (folder/'DONE.json').exists():
                    raise FileNotFoundError(f'Incomplete training: {folder}')
        if a.phase == 'train':
            for seed in a.cfg['seeds']:
                base = a.output/f'seed_{seed}'/'base'; base.mkdir(parents=True, exist_ok=True)
                if not (base/'DONE.json').exists():
                    run(command(a, a.cfg['fixed'], seed, base, base), base/'train.log')
                for s, name, p, folder in jobs:
                    if s != seed:
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    if not (folder/'DONE.json').exists():
                        run(command(a, p, seed, folder, base), folder/'train.log')
        else:
            for seed, name, p, folder in jobs:
                if not (folder/'test_metrics.json').exists():
                    run([sys.executable, str(ROOT/'src/main.py'), 'evaluate',
                         '--run', str(folder), '--device', a.device], folder/'evaluate.log')
                elif not (folder/'test_predictions.npz').exists():
                    raise FileNotFoundError(f'Missing test predictions: {folder}')
        _, _, adjacency, _ = load_data(a.data)
        report(a, jobs, 'val' if a.phase == 'train' else 'test', adjacency)
        print('COMPLETE:', a.output, flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
