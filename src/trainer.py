import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import Visits, collate, load_data, make_split, sha256, training_cooccurrence
from metrics import choose, metrics, ranking_metrics
from models import ConflictModel


def build_model(config, sizes, adjacency, cooccurrence=None):
    """Single construction path for training, validation and test."""
    return ConflictModel(sizes, adjacency, config['dim'], config['dropout'],
                         config['variant'], config['layers'], cooccurrence=cooccurrence,
                         graph_layers=config['graph_layers'], attention_heads=config['attention_heads'],
                         displacement_cap=config.get('displacement_cap', .1),
                         cost_ratio=config.get('cost_ratio', 4.))


def validate_checkpoint_config(config):
    # Old training ignored these CLI values; do not silently change its test function.
    if config.get('parameter_wiring_version', 0) < 1 and config['variant'] != 'base':
        if config.get('displacement_cap', .1) != .1 or config.get('cost_ratio', 4.) != 4.:
            raise ValueError('Legacy checkpoint has non-default cap/cost settings but no verified '
                             'parameter wiring. Retrain in a new output directory; do not test '
                             'with parameters that may differ from training.')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def save_csv(path, rows):
    if rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def move(batch, device):
    return {k: v.to(device) if k not in ['patient', 'visit'] else v for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    output = {k: [] for k in ['y', 'probability', 'patient', 'visit']}
    stats, n = {}, 0
    for batch_idx, batch in enumerate(loader, 1):
        logits, _, diagnostic = model.forward_details(move(batch, device), diagnostics=True)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite evaluation logits')
        output['probability'].append(logits.sigmoid().cpu().numpy())
        for key in ['y', 'patient', 'visit']:
            output[key].append(batch[key].numpy())
        if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == len(loader):
            print(f'EVAL batch {batch_idx}/{len(loader)} ({100*batch_idx/len(loader):.1f}%)',flush=True)
        count = len(logits)
        n += count
        for key, value in diagnostic.items():
            stats[key] = stats.get(key, 0.) + float(value) * count
    return {key: np.concatenate(value) for key, value in output.items()}, {k:v/n for k,v in stats.items()}


def curve(prediction, adjacency):
    ranking = ranking_metrics(prediction['y'], prediction['probability'], prediction['patient'])
    return [metrics(prediction['y'], prediction['probability'], prediction['patient'], adjacency,
                    threshold=round(float(t), 3), ranking=ranking) for t in np.arange(.05, .951, .025)]


def fixed_k(prediction, adjacency):
    ranking = ranking_metrics(prediction['y'], prediction['probability'], prediction['patient'])
    order = np.argsort(-prediction['probability'], axis=1, kind='stable')
    rows = []
    for k in [10, 15, 20, 25, 30]:
        pred = np.zeros_like(prediction['probability'])
        pred[np.arange(len(pred))[:, None], order[:, :k]] = 1
        row = metrics(prediction['y'], pred, prediction['patient'], adjacency, .5, ranking=ranking)
        row.pop('threshold')
        rows.append({'k': k, **row})
    return rows


def active_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def calibrate(model, batches, device):
    model.eval()
    total,count=0.,0
    active=torch.zeros(model.backbone.drug.num_embeddings,dtype=torch.bool,device=device)
    active[model.ei]=True;active[model.ej]=True
    for batch in batches:
        h,u=model.backbone(move(batch,device));q=model.backbone.predict(h,u).sigmoid()
        x=.5*q[:,model.ei]*q[:,model.ej]
        b=torch.zeros_like(q);b.index_add_(1,model.ei,x);b.index_add_(1,model.ej,x)
        v=(b/model.scale)[:,active]
        total+=float(v.sum());count+=v.numel()
    model.exposure_scale.fill_(max(total/count,1e-6) if count else 1.)
    return {'kappa':float(model.exposure_scale),'train_entries':count}


def loader(records, ids, nmed, batch, workers=0, shuffle=False):
    return DataLoader(Visits(records, ids, nmed), batch_size=batch, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate)


def train(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    lock = root/'RUNNING.lock'
    # Fail instead of allowing concurrent writers to corrupt checkpoints.
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        _train(args, root)
    finally:
        lock.unlink(missing_ok=True)


def _train(args, root):
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu only for smoke tests')
    records, sizes, adjacency, info = load_data(args.data)
    if args.smoke:
        records=records[:6]
    split = make_split(records, info, args.split, args.split_seed, args.split_mode)
    config = {k:v for k,v in vars(args).items() if k not in ['func', 'resume']}
    config.update({'sizes': sizes, 'fingerprints': info['fingerprints'], 'split_sha256':sha256(args.split),
                   'parameter_wiring_version':1,
                   'init_sha256':sha256(args.init) if args.init else None,
                   'source_sha256':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}})
    if (root/'config.json').exists():
        if json.loads((root/'config.json').read_text()) != config:
            raise ValueError('Run configuration or source differs; choose a new output directory')
        if (root/'DONE.json').exists():
            print('Already complete:', root, flush=True)
            return
        if not args.resume:
            raise ValueError('Incomplete run exists; pass --resume')
    else:
        save_json(root/'config.json', config)
    cooccurrence = training_cooccurrence(records, split['train'], sizes[2])
    model = build_model(config, sizes, adjacency, cooccurrence).to(device)
    print(f"MODEL enhanced | total_parameters={sum(p.numel() for p in model.parameters()):,} "
          f"trainable_parameters={active_parameters(model):,} "
          f"training_cooccurrence_edges={int(cooccurrence.sum()/2)}", flush=True)
    if args.init:
        initial = torch.load(args.init, map_location='cpu', weights_only=False)
        if initial['config']['variant'] != 'base':
            raise ValueError('--init must be a first-stage base checkpoint')
        for key in ['dim', 'graph_layers', 'attention_heads', 'dropout']:
            if initial['config'].get(key) != config[key]:
                raise ValueError(f'Base/full architecture mismatch: {key}')
        if initial['config']['fingerprints'] != info['fingerprints'] or initial['config']['split_sha256'] != config['split_sha256']:
            raise ValueError('Warm start data/split mismatch')
        state = {k.removeprefix('backbone.'):v for k,v in initial['model'].items() if k.startswith('backbone.')}
        model.backbone.load_state_dict(state, strict=True)
        state_init = {k:v.clone() for k,v in state.items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    start, best, stale = 0, None, 0
    history = []
    if args.resume and (root/'last.pt').exists():
        state = torch.load(root/'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, best, stale = state['epoch'] + 1, state['best'], state['stale']
        history = state['history']
        # Repair an interrupted metadata write from authoritative last checkpoint.
        with (root/'history.jsonl').open('w') as f:
            for row in history:
                f.write(json.dumps(row, allow_nan=False) + '\n')
    train_loader = loader(records, split['train'], sizes[2], args.batch_size, args.workers, True)
    val_loader = loader(records, split['val'], sizes[2], args.eval_batch_size, args.workers)
    if args.variant != 'base' and not args.init:
        raise ValueError('Adjustment stage requires --init from the base stage')
    if args.variant != 'base' and start == 0:
        save_json(root/'calibration.json',calibrate(model,loader(records,split['train'],sizes[2],args.eval_batch_size),device))
    ei, ej = model.edge_i, model.edge_j
    environment = {'python':platform.python_version(), 'torch':torch.__version__,
                   'numpy':np.__version__, 'device':str(device),
                   'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                   'active_parameters':active_parameters(model),
                   'total_parameters':sum(p.numel() for p in model.parameters()),
                   'training_cooccurrence_edges':int(cooccurrence.sum()/2), 'dataset':info}
    save_json(root/'environment.json', environment)
    wall = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    # A crash after early stopping must only redo final artifact export.
    epoch_end = start if stale >= args.patience else args.epochs
    for epoch in range(start, epoch_end):
        print(f"EPOCH {epoch}/{args.epochs-1} | stage={args.variant} | TRAIN",flush=True)
        # Epoch seeding also makes a resumed run reproduce shuffle/dropout streams.
        seed_all(args.seed + 1009 * epoch)
        freeze = args.variant != 'base'
        for p in model.backbone.parameters():
            p.requires_grad_(not freeze)
        model.train()
        if freeze:
            model.backbone.eval()
        totals, seen = np.zeros(3), 0
        tic = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader, 1):
            batch = move(batch, device)
            logits, raw, _ = model.forward_details(batch)
            bce = F.binary_cross_entropy_with_logits(logits, batch['y'])
            aux = F.binary_cross_entropy_with_logits(raw, batch['y'])
            p = logits.sigmoid()
            ddi = (p[:,ei] * p[:,ej]).mean() if len(ei) else logits.sum()*0
            loss = bce + args.aux_weight*aux + args.ddi_weight*ddi
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            n = len(logits)
            totals += np.array([loss.item(), bce.item(), ddi.item()]) * n
            seen += n
            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == len(train_loader):
                elapsed=time.perf_counter()-tic
                eta=elapsed/batch_idx*(len(train_loader)-batch_idx)
                print(f'TRAIN epoch={epoch} batch={batch_idx}/{len(train_loader)} '
                      f'({100*batch_idx/len(train_loader):.1f}%) loss={totals[0]/seen:.6f} '
                      f'BCE={totals[1]/seen:.6f} soft_DDI={totals[2]/seen:.6f} '
                      f'elapsed={elapsed:.1f}s ETA={eta:.1f}s',flush=True)
            if args.smoke: break
        print(f'VALIDATION epoch={epoch} started',flush=True)
        prediction, diagnostic = predict(model, val_loader, device)
        rows = ([metrics(prediction['y'], prediction['probability'], prediction['patient'], adjacency, args.threshold)]
                if args.threshold_mode == 'fixed' else curve(prediction, adjacency))
        selected = (dict(max(rows, key=lambda r: (r['jaccard'], -r['ddi'], r['recall'])), feasible=True)
                    if args.selection == 'accuracy' else choose(rows, args.target_ddi))
        selected['ddi_target_met'] = selected['ddi'] <= args.target_ddi
        # Persist every validated epoch, independent of best-checkpoint selection.
        epoch_name = f"Epoch_{epoch:03d}_JA_{selected['jaccard']:.6f}_DDI_{selected['ddi']:.6f}.pt"
        epoch_path = root / epoch_name
        epoch_tmp = root / (epoch_name + '.tmp')
        torch.save({'model':model.state_dict(), 'config':config, 'epoch':epoch,
                    'validation':selected}, epoch_tmp)
        epoch_tmp.replace(epoch_path)
        # Feasibility first, then Jaccard; if infeasible prefer lower DDI.
        key = (1, selected['jaccard'], -selected['ddi']) if selected['feasible'] else (0, -selected['ddi'], selected['jaccard'])
        improved = best is None or tuple(key) > tuple(best['key'])
        if improved:
            best = {'key':list(key), 'epoch':epoch, 'validation':selected}
            stale = 0
            torch.save({'model':model.state_dict(), 'config':config, 'epoch':epoch, 'validation':selected}, root/'best.pt.tmp')
            (root/'best.pt.tmp').replace(root/'best.pt')
        else:
            stale += 1
        row = {'epoch':epoch, 'loss':float(totals[0]/seen), 'bce':float(totals[1]/seen),
               'soft_ddi':float(totals[2]/seen), 'epoch_seconds':time.perf_counter()-tic,
               'selected_validation':selected, 'diagnostics':diagnostic, 'improved':improved}
        history.append(row)
        torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch,
                    'best':best, 'stale':stale, 'history':history}, root/'last.pt.tmp')
        (root/'last.pt.tmp').replace(root/'last.pt')
        with (root/'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps({'epoch':epoch, 'loss':round(row['loss'],5), 'val_jaccard':round(selected['jaccard'],5),
                          'val_ddi':round(selected['ddi'],5), 'threshold':selected['threshold'],
                          'seconds':round(row['epoch_seconds'],1)}, ensure_ascii=False), flush=True)
        print(f"VALIDATION epoch={epoch} Jaccard={selected['jaccard']:.6f} "
              f"DDI={selected['ddi']:.6f} F1={selected['f1']:.6f} "
              f"PRAUC={selected['prauc']:.6f} AVG_MED={selected['avg_med']:.2f} "
              f"best_epoch={best['epoch']} stale={stale}/{args.patience} "
              f"saved={epoch_name}",flush=True)
        if stale >= args.patience:
            break
    state = torch.load(root/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    prediction, diagnostic = predict(model, val_loader, device)
    np.savez_compressed(root/'val_predictions.npz', **prediction)
    save_csv(root/'val_curve.csv', curve(prediction, adjacency))
    save_csv(root/'val_fixed_k.csv', fixed_k(prediction, adjacency))
    save_json(root/'diagnostics.json', diagnostic)
    if args.variant != 'base':
        assert all(torch.equal(v.cpu(),state_init[k]) for k,v in model.backbone.state_dict().items())
    save_json(root/'DONE.json', {'best_epoch':state['epoch'], 'validation':state['validation'],
              'wall_seconds_this_invocation':time.perf_counter()-wall,
              'peak_cuda_mb':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0,
              'active_parameters':active_parameters(model), 'test_evaluated':False})


def evaluate(args):
    root = Path(args.run)
    with (root/'EVALUATING.lock').open('x') as f:
        f.write(str(os.getpid()))
    try:
        _evaluate(args)
    finally:
        (root/'EVALUATING.lock').unlink(missing_ok=True)


def _evaluate(args):
    torch.set_num_threads(args.threads)
    seed_all(42)
    root = Path(args.run)
    output = root/'test_metrics.json'
    if output.exists():
        raise FileExistsError('Test already evaluated for this run; refusing to overwrite')
    state = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    config = state['config']
    validate_checkpoint_config(config)
    records, sizes, adjacency, info = load_data(config['data'])
    if info['fingerprints'] != config['fingerprints'] or sha256(config['split']) != config['split_sha256']:
        raise ValueError('Data or split changed')
    if config.get('smoke'): records=records[:6]
    split = json.loads(Path(config['split']).read_text())
    model = build_model(config, sizes, adjacency)
    model.load_state_dict(state['model'], strict=True)
    model.to(args.device)
    prediction, diagnostic = predict(model, loader(records, split['test'], sizes[2], config['eval_batch_size']), args.device)
    threshold = state['validation']['threshold']
    result = metrics(prediction['y'], prediction['probability'], prediction['patient'], adjacency, threshold)
    result.update({'threshold_source':'validation checkpoint selection', 'validation':state['validation'],
                   'test_target_met':result['ddi'] <= config['target_ddi']})
    np.savez_compressed(root/'test_predictions.npz', **prediction)
    save_json(root/'test_diagnostics.json', diagnostic)
    save_csv(root/'test_fixed_k.csv',fixed_k(prediction,adjacency))
    # Write the completion indicator last so interrupted exports are retryable.
    save_json(output, result)
    print(json.dumps(result, indent=2))


def prepare(args):
    records, sizes, adjacency, info = load_data(args.data)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    split = make_split(records, info, root/'split.json', args.split_seed, args.split_mode)
    # Test labels are not summarized here; reserve test evaluation for finalization.
    for part in ['train', 'val']:
        dataset = Visits(records, split[part], sizes[2])
        y = np.stack([dataset[k][1].numpy() for k in range(len(dataset))])
        ids = np.array([x[0] for x in dataset.index])
        info[part] = metrics(y, y, ids, adjacency, .5, ranking={'prauc':1.,'visit_prauc':1.})
    info['split_patient_counts'] = {k:len(split[k]) for k in ['train','val','test']}
    save_json(root/'data_summary.json', info)
    print(json.dumps(info, indent=2))
