import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


VARIANTS = ['base','uniform','edge_only','allocation_only','full']


def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


def allocate(strength, ri, rj):
    """Minimize .5*ri*di^2 + .5*rj*dj^2 subject to di+dj=strength."""
    return strength * rj / (ri + rj), strength * ri / (ri + rj)


class ResidualFusion(nn.Module):
    def __init__(self, n_in, dim, dropout):
        super().__init__()
        self.project = nn.Linear(n_in, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2*dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.project(x)
        return self.norm(x + self.ff(x))


class RelationEncoder(nn.Module):
    def __init__(self, dim, layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, graph):
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + self.dropout(torch.relu(layer(graph @ x))))
        return x


def normalized_graph(adjacency):
    a = torch.as_tensor(adjacency, dtype=torch.float32).clone()
    a.fill_diagonal_(1)
    inv = a.sum(-1).clamp_min(1).rsqrt()
    return inv[:, None] * a * inv[None, :]


class Backbone(nn.Module):
    def __init__(self, sizes, dim, dropout, adjacency, cooccurrence,
                 graph_layers=2, attention_heads=4):
        super().__init__()
        if attention_heads < 1 or dim % attention_heads or graph_layers < 1:
            raise ValueError('dim must be divisible by positive attention_heads; graph_layers >= 1')
        self.diag = nn.Embedding(sizes[0] + 1, dim, padding_idx=0)
        self.proc = nn.Embedding(sizes[1] + 1, dim, padding_idx=0)
        self.diag_score = nn.Linear(dim, 1)
        self.proc_score = nn.Linear(dim, 1)
        self.visit = ResidualFusion(2*dim, dim, dropout)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.history_query = nn.Linear(dim, dim)
        self.history_key = nn.Linear(dim, dim)
        self.patient = ResidualFusion(3*dim, dim, dropout)
        self.drug = nn.Embedding(sizes[2], dim)
        self.register_buffer('ddi_graph', normalized_graph(adjacency))
        if cooccurrence is None:
            cooccurrence = torch.zeros(sizes[2], sizes[2])
        self.register_buffer('ehr_graph', normalized_graph(cooccurrence))
        self.ddi_encoder = RelationEncoder(dim, graph_layers, dropout)
        self.ehr_encoder = RelationEncoder(dim, graph_layers, dropout)
        self.drug_fusion = ResidualFusion(3*dim, dim, dropout)
        self.code_type = nn.Embedding(2, dim)
        self.cross_attention = nn.MultiheadAttention(dim, attention_heads, dropout=dropout,
                                                     batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.condition = ResidualFusion(4*dim, dim, dropout)
        # Keep this head structure: conflict descent uses its exact analytic gradient.
        self.head = mlp(2*dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def pool(embedding, score, ids):
        x = embedding(ids)
        valid = ids != 0
        scores = score(x).squeeze(-1).masked_fill(~valid, -1e9)
        weights = scores.softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return (weights[..., None] * x).sum(-2)

    def forward(self, batch):
        x = self.visit(torch.cat([self.pool(self.diag, self.diag_score, batch['diag']),
                                 self.pool(self.proc, self.proc_score, batch['proc'])], -1))
        x = self.dropout(x)
        packed = pack_padded_sequence(x, batch['length'].cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_states, last = self.gru(packed)
        states, _ = pad_packed_sequence(packed_states, batch_first=True, total_length=x.shape[1])
        row = torch.arange(len(x), device=x.device)
        current = x[row, batch['length']-1]
        valid = torch.arange(x.shape[1], device=x.device)[None] < batch['length'][:, None]
        scores = (self.history_key(states)*self.history_query(current)[:, None]).sum(-1) / x.shape[-1]**.5
        weights = scores.masked_fill(~valid, -1e9).softmax(-1)
        history = (weights[..., None]*states).sum(1)
        h = self.patient(torch.cat([current, last[0], history], -1))
        base = self.drug.weight
        drug = self.drug_fusion(torch.cat([base, self.ehr_encoder(base, self.ehr_graph),
                                          self.ddi_encoder(base, self.ddi_graph)], -1))
        e = drug[None].expand(len(h), -1, -1)
        # Attend to current diagnosis/procedure tokens; h is an always-valid history token.
        diag_ids = batch['diag'][row, batch['length']-1]
        proc_ids = batch['proc'][row, batch['length']-1]
        tokens = torch.cat([h[:, None], self.diag(diag_ids)+self.code_type.weight[0],
                            self.proc(proc_ids)+self.code_type.weight[1]], 1)
        padding = torch.cat([torch.zeros(len(h), 1, dtype=torch.bool, device=h.device),
                             diag_ids == 0, proc_ids == 0], 1)
        context, _ = self.cross_attention(e, tokens, tokens, key_padding_mask=padding,
                                          need_weights=False)
        context = self.cross_norm(e + context)
        hx = h[:, None].expand_as(e)
        u = self.condition(torch.cat([hx, e, hx*e, context], -1))
        return h, u

    def predict(self, h, u):
        return self.head(torch.cat([h[:, None].expand_as(u), u], -1)).squeeze(-1)


class ConflictModel(nn.Module):
    def __init__(self, sizes, adjacency, dim=256, dropout=.2, variant='full', layers=1,
                 cooccurrence=None, graph_layers=2, attention_heads=4, displacement_cap=0.1, cost_ratio=4.0):
        super().__init__()
        if not 0 < displacement_cap < float("inf") or not 1 <= cost_ratio < float("inf"):
            raise ValueError("displacement_cap must be positive and cost_ratio >= 1; both finite")
        self.displacement_cap = float(displacement_cap)
        self.cost_ratio = float(cost_ratio)
        self.variant = variant
        if variant not in ['base','uniform','edge_only','allocation_only','full'] or layers != 1:
            raise ValueError('Supported variants: base/uniform/edge_only/allocation_only/full; layers=1')
        self.backbone = Backbone(sizes, dim, dropout, adjacency, cooccurrence, graph_layers, attention_heads)
        self.backbone.requires_grad_(variant == 'base')
        edges = torch.triu(torch.as_tensor(adjacency).bool(), 1).nonzero()
        self.register_buffer('ei', edges[:, 0])
        self.register_buffer('ej', edges[:, 1])
        self.edge = mlp(4*dim, dim, 1)
        nn.init.zeros_(self.edge[-1].weight)
        nn.init.zeros_(self.edge[-1].bias)
        self.edge.requires_grad_(variant in ['full','edge_only'])
        degree = torch.as_tensor(adjacency).sum(1).clamp_min(1).sqrt()
        self.register_buffer('scale', degree)
        self.register_buffer('exposure_scale', torch.tensor(1.))
        self.cost = mlp(2 * dim, 32, 1)
        nn.init.zeros_(self.cost[-1].weight)
        nn.init.zeros_(self.cost[-1].bias)
        self.cost.requires_grad_(variant in ['full','allocation_only'])
        self.alpha = nn.Parameter(torch.tensor(-2.944439))  # 2*sigmoid = .1

    def train(self, mode=True):
        super().train(mode)
        if self.variant != 'base': self.backbone.eval()
        return self

    @property
    def edge_i(self): return self.ei

    @property
    def edge_j(self): return self.ej

    def forward(self, patient_visits):
        """SafeDrug-style API: (final logits [B,M], scalar DDI loss).

        A single patient's visit prefix uses [diagnoses, procedures, medications]
        records; medications are ignored as model inputs. Batched dict also works.
        """
        logits, _, _ = self.forward_details(patient_visits)
        probability = logits.sigmoid()
        ddi_loss = ((probability[:, self.ei] * probability[:, self.ej]).mean()
                    if self.ei.numel() else logits.sum() * 0)
        return logits, ddi_loss

    def forward_details(self, batch, diagnostics=False, mode='original', multiplier=1.):
        # SafeDrug-style single-patient visit prefix is also accepted.
        if not isinstance(batch, dict):
            from data import collate
            prefix = [(v[0],v[1]) for v in batch]
            batch = collate([(prefix, torch.zeros(self.backbone.drug.num_embeddings),0,0)])
            batch = {k:v.to(self.ei.device) for k,v in batch.items()}
        if self.variant == 'base':
            h,u=self.backbone(batch)
            raw=self.backbone.predict(h,u)
            return raw,raw,{}
        if not self.ei.numel():
            with torch.no_grad():
                h,u=self.backbone(batch);raw=self.backbone.predict(h,u)
            return raw+self.alpha*0,raw,{}

        with torch.no_grad():
            h, u = self.backbone(batch)
            raw = self.backbone.predict(h, u)
            # Exact local gradient of the existing Linear-ReLU-Linear head wrt u.
            first, _, last = self.backbone.head
            x = torch.cat([h[:, None].expand_as(u), u], -1)
            mask = (first(x) > 0).to(u.dtype)
            g = (mask * last.weight[0]) @ first.weight[:, u.shape[-1]:]
            direction = g / (g.square().sum(-1, keepdim=True) + 1e-8)
            q = raw.sigmoid()
            # Expected co-prescription exposure, NOT learned patient edge importance.
            exposure = q[:, self.ei] * q[:, self.ej]
        strength = torch.ones_like(exposure)
        if self.variant in ['full','edge_only'] and mode != 'constant_edge':
            ui,uj=u[:,self.ei],u[:,self.ej]
            features=torch.cat([h[:,None].expand_as(ui),ui+uj,(ui-uj).abs(),ui*uj],-1)
            strength=2*self.edge(features).squeeze(-1).sigmoid()
        exposure=exposure*strength
        c = torch.ones_like(q)
        if self.variant in ['full','allocation_only'] and mode != 'equal':
            c = .5 + (.5 * (self.cost_ratio - 1)) * self.cost(torch.cat([h[:, None].expand_as(u), u], -1)).squeeze(-1).sigmoid()
        frac = c[:, self.ej] / (c[:, self.ei] + c[:, self.ej])
        if mode == 'swap':
            frac = 1 - frac
        burden = torch.zeros_like(q)
        burden.index_add_(1, self.ei, exposure * frac)
        burden.index_add_(1, self.ej, exposure * (1 - frac))
        budget = 2 * self.alpha.sigmoid() * burden / self.scale / self.exposure_scale * multiplier
        if mode == 'bypass':
            budget = budget * 0
        # Bound representation displacement by the configured fraction (norm floor 1).
        cap = self.displacement_cap * u.norm(dim=-1).clamp_min(1) / direction.norm(dim=-1).clamp_min(1e-8)
        step = torch.minimum(budget, cap)
        # Detached Armijo selection: gradient flows through accepted budget, no Hessian.
        with torch.no_grad():
            factor = torch.ones_like(step)
            descent = (g * direction).sum(-1)
            for _ in range(8):
                z = self.backbone.predict(h, u - (step * factor)[..., None] * direction)
                ok = z <= raw - .1 * step * factor * descent
                factor = torch.where(ok, factor, factor * .5)
            z = self.backbone.predict(h, u - (step * factor)[..., None] * direction)
            factor = torch.where(z <= raw - .1 * step * factor * descent, factor, torch.zeros_like(factor))
        adjusted = u - (step * factor)[..., None] * direction
        logits = self.backbone.predict(h, adjusted)
        stats = {'edge_mean':strength.mean(), 'edge_std':strength.std(unbiased=False), 'alpha': 2*self.alpha.sigmoid(), 'cost_std': c.std(unbiased=False),
                 'allocation_asymmetry': (2*frac-1).abs().mean(),
                 'budget_mean': budget.mean(), 'logit_drop': (raw-logits).mean(),
                 'prob_drop': (q-logits.sigmoid()).mean(),
                 'increase_rate': (logits > raw + 1e-6).float().mean(),
                 'backtrack_rate': (factor < 1).float().mean(),
                 'rejected_rate': (factor == 0).float().mean(),
                 'cap_rate': (budget > cap).float().mean(),
                 'representation_change': (adjusted-u).norm(dim=-1).mean()}
        return logits, raw, stats if diagnostics else {}
