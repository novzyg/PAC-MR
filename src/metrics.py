import numpy as np
from sklearn.metrics import average_precision_score


def patient_mean(values, ids):
    _, groups = np.unique(ids, return_inverse=True)
    return float((np.bincount(groups, weights=values) / np.bincount(groups)).mean())


def ranking_metrics(y, probability, patient):
    # Average precision per visit, then equal-weight patients, matching common MR practice.
    ap = np.array([average_precision_score(a, b) for a, b in zip(y, probability)])
    return {'prauc': patient_mean(ap, patient), 'visit_prauc': float(ap.mean())}


def metrics(y, probability, patient, adjacency, threshold=0.5, ranking=None):
    pred = (probability >= threshold).astype(np.float64)
    truth = y.astype(np.float64)
    tp = (pred * truth).sum(1)
    count = pred.sum(1)
    actual = truth.sum(1)
    precision = tp / np.maximum(count, 1)
    recall = tp / np.maximum(actual, 1)
    jac = tp / np.maximum(count + actual - tp, 1)
    f1 = 2 * tp / np.maximum(count + actual, 1)
    conflicts = ((pred @ adjacency) * pred).sum(1) / 2
    pairs = count * (count - 1) / 2
    row = {'threshold': float(threshold), 'ddi': float(conflicts.sum()/max(pairs.sum(), 1)),
           'avg_med': float(count.mean()), 'conflicts_per_visit': float(conflicts.mean()),
           'empty_rate': float((count == 0).mean()), 'visits': len(y),
           'patients': len(np.unique(patient)), 'pair_count': float(pairs.sum())}
    for name, value in [('jaccard', jac), ('f1', f1), ('precision', precision), ('recall', recall)]:
        row[name] = patient_mean(value, patient)
        row['visit_' + name] = float(value.mean())
    row.update(ranking if ranking is not None else ranking_metrics(y, probability, patient))
    return row


def choose(rows, target):
    feasible = [r for r in rows if r['ddi'] <= target]
    if feasible:
        best = max(feasible, key=lambda r: (r['jaccard'], -r['ddi'], r['recall']))
    else:
        best = min(rows, key=lambda r: (r['ddi'], -r['jaccard']))
    return {**best, 'feasible': bool(feasible)}
