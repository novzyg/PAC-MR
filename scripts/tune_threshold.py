"""Select a checkpoint and threshold using saved validation curves only."""
import argparse
import csv
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-jaccard', type=float, default=0.33)
    parser.add_argument('--margin', type=float, default=0.005)
    args = parser.parse_args()
    if not 0 < args.min_jaccard < 1 or not 0 <= args.margin < 1 - args.min_jaccard:
        parser.error('Invalid Jaccard target or margin')
    candidates = []
    for curve in sorted(args.source.glob('*/full/val_curve.csv')):
        if not (curve.parent / 'DONE.json').exists():
            continue
        for row in csv.DictReader(curve.open()):
            row = {key: float(value) for key, value in row.items()}
            candidates.append({'checkpoint': str((curve.parent / 'best.pt').resolve()), **row})
    feasible = [row for row in candidates if row['jaccard'] > args.min_jaccard
                and row['jaccard'] >= args.min_jaccard + args.margin]
    if not feasible:
        raise RuntimeError('No validation candidate meets the target and margin')
    selected = min(feasible, key=lambda row: (row['ddi'], -row['jaccard'], row['checkpoint'], row['threshold']))
    state = torch.load(selected['checkpoint'], map_location='cpu', weights_only=False)
    original_validation = state['validation']
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'validation_candidates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    state['validation'] = {key: value for key, value in selected.items() if key != 'checkpoint'}
    state['config'] = dict(state['config'], threshold=selected['threshold'], threshold_mode='validation')
    provenance = {
        'source_checkpoint': selected['checkpoint'],
        'original_validation': original_validation,
        'selection_rule': 'Minimum validation DDI subject to Jaccard > target and Jaccard >= target + margin',
        'min_jaccard': args.min_jaccard, 'margin': args.margin,
        'candidate_count': len(candidates), 'feasible_count': len(feasible),
        'selected_validation': state['validation'],
        'weights_changed': False, 'test_used_for_selection': False,
    }
    state['threshold_tuning'] = provenance
    torch.save(state, args.output / 'best.pt')
    for name, value in [('selection.json', provenance), ('config.json', state['config'])]:
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
