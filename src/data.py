"""Read trusted local SafeDrug-style pickles; never use current/future drugs as inputs."""
import hashlib
import json
from pathlib import Path
import dill
import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_data(root):
    root = Path(root)
    names = ['records_final.pkl', 'voc_final.pkl', 'ddi_A_final.pkl']
    objects = []
    for name in names:
        with (root/name).open('rb') as f:
            objects.append(dill.load(f))
    records, voc, raw_adj = objects
    sizes = [len(voc[k].idx2word) for k in ['diag_voc', 'pro_voc', 'med_voc']]
    adj = np.asarray(raw_adj)
    if adj.shape != (sizes[2], sizes[2]) or not np.isfinite(adj).all():
        raise ValueError('DDI shape or values invalid')
    if not np.array_equal(adj, adj.T) or np.any(np.diag(adj) != 0):
        raise ValueError('Expected symmetric DDI matrix with zero diagonal; no silent repair')
    if not np.isin(adj, [0, 1]).all():
        raise ValueError('This implementation requires binary DDI adjacency')
    clean = []
    for patient in records:
        if not patient:
            raise ValueError('Empty patient record')
        visits = []
        for visit in patient:
            if len(visit) < 3:
                raise ValueError('Expected [diagnoses, procedures, medications, ...]')
            row = []
            for ids, size in zip(visit[:3], sizes):
                ids = sorted(set(int(x) for x in ids))
                if any(x < 0 or x >= size for x in ids):
                    raise ValueError('Medical code outside vocabulary')
                row.append(ids)
            if not row[2]:
                raise ValueError('Empty target medication set')
            visits.append(row)
        clean.append(visits)
    fingerprints = {name: sha256(root/name) for name in names}
    summary = {'patients': len(clean), 'visits': sum(map(len, clean)), 'sizes': sizes,
               'ddi_edges': int(np.triu(adj, 1).sum()), 'fingerprints': fingerprints,
               'input_note': 'First two fields only; extra visit fields ignored; original visit order retained'}
    return clean, sizes, adj.astype(np.float32), summary


def make_split(records, summary, output, seed=2026, mode='random'):
    output = Path(output)
    if output.exists():
        split = json.loads(output.read_text())
        if split['fingerprints'] != summary['fingerprints']:
            raise ValueError('Dataset changed since split creation')
        if split['seed'] != seed or split['mode'] != mode:
            raise ValueError('Split seed/mode mismatch; use another output directory')
    else:
        ids = np.arange(len(records))
        if mode == 'random':
            np.random.default_rng(seed).shuffle(ids)
        elif mode != 'ordered':
            raise ValueError(mode)
        ntrain = int(len(ids) * 2 / 3)
        ntest = (len(ids) - ntrain) // 2
        # SafeDrug-style slice ordering: train, test, validation.
        split = {'seed': seed, 'mode': mode, 'fingerprints': summary['fingerprints'],
                 'train': ids[:ntrain].tolist(), 'test': ids[ntrain:ntrain+ntest].tolist(),
                 'val': ids[ntrain+ntest:].tolist()}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(split, indent=2))
    all_ids = split['train'] + split['val'] + split['test']
    if sorted(all_ids) != list(range(len(records))) or len(set(all_ids)) != len(records):
        raise ValueError('Invalid patient split')
    return split


class Visits(Dataset):
    def __init__(self, records, patient_ids, nmed):
        self.records, self.nmed = records, nmed
        self.index = [(p, t) for p in patient_ids for t in range(len(records[p]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        p, t = self.index[index]
        # No future visits and no medication codes in model input.
        prefix = [(v[0], v[1]) for v in self.records[p][:t+1]]
        y = torch.zeros(self.nmed)
        y[self.records[p][t][2]] = 1
        return prefix, y, p, t


def collate(rows):
    b, tmax = len(rows), max(len(r[0]) for r in rows)
    batch = {}
    for field, key in enumerate(['diag', 'proc']):
        cmax = max(1, max(len(v[field]) for row in rows for v in row[0]))
        ids = torch.zeros(b, tmax, cmax, dtype=torch.long)
        for n, row in enumerate(rows):
            for t, visit in enumerate(row[0]):
                if visit[field]:
                    ids[n, t, :len(visit[field])] = torch.tensor(visit[field]) + 1
        batch[key] = ids
    batch['length'] = torch.tensor([len(r[0]) for r in rows])
    batch['y'] = torch.stack([r[1] for r in rows])
    batch['patient'] = torch.tensor([r[2] for r in rows])
    batch['visit'] = torch.tensor([r[3] for r in rows])
    return batch


def training_cooccurrence(records, patient_ids, nmed):
    """Binary within-visit graph; only explicitly supplied training patients contribute."""
    adjacency = np.zeros((nmed, nmed), dtype=np.float32)
    for patient_id in patient_ids:
        for visit in records[patient_id]:
            meds = visit[2]
            adjacency[np.ix_(meds, meds)] = 1
    np.fill_diagonal(adjacency, 0)
    return adjacency
