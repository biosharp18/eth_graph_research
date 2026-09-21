"""Two-headed TGN (Rossi et al., 2020) over daily aggregated edges.

The day is the memory timestep: a node's memory updates at most once per day,
with one GRU step on the mean of that day's messages. Predictions for day T
must use memory reflecting days < T only — enforced by the training loop
(tgn.train) and the streaming scorer (tgn.streaming), not here.
"""
import numpy as np
import torch
import torch.nn as nn

from tgat.model import TemporalAttention, TimeEncoder


class TGN(nn.Module):
    def __init__(self, edge_feat_dim: int, raw_feat_dim: int = 6,
                 dim: int = 100, n_heads: int = 2, dropout: float = 0.1,
                 pair_feat_dim: int = 0, head: str = "mlp",
                 n_layers: int = 1, n_stack: int = 0, k_inner: int = 0):
        super().__init__()
        self.dim = dim
        self.pair_feat_dim = pair_feat_dim
        self.head = head
        self.n_layers = n_layers
        self.n_stack = n_stack
        self.k_inner = k_inner  # 0 = inner hops use the store's k
        self.time_enc = TimeEncoder(dim)
        # raw message: [own_mem | other_mem | timeenc(dt) | edge_feat | dir_flag]
        self.gru = nn.GRUCell(2 * dim + dim + raw_feat_dim + 1, dim)
        self.attn = TemporalAttention(dim, edge_feat_dim, dim, n_heads, dropout)
        # inner hops (attn2 = one hop in, attn3 = two hops in, ...): a
        # neighbor is itself embedded at its edge's day (TGAT convention)
        # by the next-deeper attention; the deepest hop reads raw memory
        for j in range(2, n_layers + 1):
            setattr(self, f"attn{j}", TemporalAttention(
                dim, edge_feat_dim, dim, n_heads, dropout))
        # n_stack extra residual attention layers over the SAME one-hop
        # neighbor set: depth of processing without widening the receptive
        # field (the query is refined, neighbor states stay raw memory)
        self.stack = nn.ModuleList([
            TemporalAttention(dim, edge_feat_dim, dim, n_heads, dropout)
            for _ in range(n_stack)])
        # pair_feat_dim > 0 appends streaming pair-recency features
        # (tgn.recency) to the link head input; the amount head is unchanged
        if head == "bilinear":
            # multiplicative source-candidate interaction: sharper cross-
            # source calibration than the concat-MLP, feature path additive
            self.bilin_u = nn.Linear(dim, dim, bias=False)
            self.bilin_v = nn.Linear(dim, dim, bias=False)
            if pair_feat_dim:
                self.feat_mlp = nn.Sequential(
                    nn.Linear(pair_feat_dim, dim // 2), nn.ReLU(),
                    nn.Linear(dim // 2, 1))
        else:
            self.link_head = nn.Sequential(
                nn.Linear(2 * dim + pair_feat_dim, dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(dim, 1))
        self.amt_head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, 1))

    def link_logit(self, zs, zd, pair_feat=None):
        if self.head == "bilinear":
            logit = (self.bilin_u(zs) * self.bilin_v(zd)).sum(-1) \
                / self.dim ** 0.5
            if self.pair_feat_dim:
                logit = logit + self.feat_mlp(pair_feat).squeeze(-1)
            return logit
        pair = torch.cat([zs, zd], dim=-1)
        if self.pair_feat_dim:
            pair = torch.cat([pair, pair_feat], dim=-1)
        return self.link_head(pair).squeeze(-1)

    def init_memory(self, n_nodes: int, device):
        return (torch.zeros(n_nodes, self.dim, device=device),
                torch.zeros(n_nodes, device=device))

    def apply_messages(self, mem, last_update, src, dst, day: int, edge_feat):
        """One memory step for the events of a single day (numpy inputs)."""
        device = mem.device
        E = len(src)
        node = torch.from_numpy(np.concatenate([src, dst])).to(device)
        other = torch.from_numpy(np.concatenate([dst, src])).to(device)
        feat = torch.from_numpy(np.concatenate([edge_feat, edge_feat])).to(device)
        flag = torch.cat([torch.ones(E, device=device),
                          torch.zeros(E, device=device)]).unsqueeze(1)
        dt = float(day) - last_update[node]
        msgs = torch.cat([mem[node], mem[other], self.time_enc(dt), feat, flag],
                         dim=1)
        active, inv = torch.unique(node, return_inverse=True)
        agg = torch.zeros(len(active), msgs.shape[1], device=device)
        agg.index_add_(0, inv, msgs)
        counts = torch.zeros(len(active), device=device)
        counts.index_add_(0, inv, torch.ones(len(node), device=device))
        agg = agg / counts.unsqueeze(1)
        mem = mem.clone()
        mem[active] = self.gru(agg, mem[active])
        last_update = last_update.clone()
        last_update[active] = float(day)
        return mem, last_update

    def _embed(self, nodes, days, level, mem, store, device):
        """Embed `nodes` at query `days`; level 0 is the outermost hop."""
        k = self.k_inner if (level > 0 and self.k_inner) else None
        nbr, dt, feat, mask = store.sample(nodes, days, k=k)
        B, kk = nbr.shape
        if level + 1 < self.n_layers:
            # embed each distinct valid (neighbor, edge-day) once with the
            # next-deeper hop; a neighbor's state at its edge day only sees
            # days < edge day < query day, preserving strictly-past order
            nbr_days = days[:, None] - dt.astype(np.int64)
            flat_valid = mask.reshape(-1)
            pairs = np.stack([nbr.reshape(-1)[flat_valid],
                              nbr_days.reshape(-1)[flat_valid]], axis=1)
            nbr_h = torch.zeros(B * kk, self.dim, device=device)
            if len(pairs):
                uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
                u_h = self._embed(uniq[:, 0], uniq[:, 1], level + 1, mem,
                                  store, device)
                nbr_h[torch.from_numpy(np.flatnonzero(flat_valid)).to(
                    device)] = u_h[torch.from_numpy(inv.reshape(-1)).to(device)]
            nbr_h = nbr_h.reshape(B, kk, self.dim)
        else:
            nbr_h = mem[torch.from_numpy(nbr.reshape(-1)).to(device)].reshape(
                B, kk, self.dim)
        node_h = mem[torch.from_numpy(nodes).to(device)]
        feat_t = torch.from_numpy(feat).to(device)
        tenc = self.time_enc(torch.from_numpy(dt).to(device))
        mask_t = torch.from_numpy(mask).to(device)
        attn = self.attn if level == 0 else getattr(self, f"attn{level + 1}")
        h = attn(node_h, nbr_h, feat_t, tenc, mask_t)
        if level == 0:
            for layer in self.stack:
                h = h + layer(h, nbr_h, feat_t, tenc, mask_t)
        return h

    def embed(self, nodes, days, mem, store, device):
        return self._embed(np.asarray(nodes), np.asarray(days), 0, mem,
                           store, device)

    def forward(self, src_nodes, dst_nodes, days, mem, store, device,
                pair_feat=None):
        zs = self.embed(src_nodes, days, mem, store, device)
        zd = self.embed(dst_nodes, days, mem, store, device)
        pair = torch.cat([zs, zd], dim=-1)
        return (self.link_logit(zs, zd, pair_feat),
                self.amt_head(pair).squeeze(-1))


def build_from_checkpoint(path, edge_feat_dim: int, raw_feat_dim: int,
                          dim: int, device, k_inner: int = 0) -> TGN:
    """Load a checkpoint, inferring head type, pair_feat_dim, hop depth and
    stack depth from keys. k_inner is not recorded in the checkpoint (like
    --k / --nbr-mode) and must be passed by the caller when non-default."""
    sd = torch.load(path, map_location=device)
    if "bilin_u.weight" in sd:
        head = "bilinear"
        pair_feat_dim = (int(sd["feat_mlp.0.weight"].shape[1])
                         if "feat_mlp.0.weight" in sd else 0)
    else:
        head = "mlp"
        pair_feat_dim = int(sd["link_head.0.weight"].shape[1]) - 2 * dim
    n_layers = 1 + len({k.split(".")[0] for k in sd
                        if k.startswith("attn") and k[4:5].isdigit()})
    n_stack = len({k.split(".")[1] for k in sd if k.startswith("stack.")})
    model = TGN(edge_feat_dim=edge_feat_dim, raw_feat_dim=raw_feat_dim,
                dim=dim, pair_feat_dim=pair_feat_dim, head=head,
                n_layers=n_layers, n_stack=n_stack,
                k_inner=k_inner).to(device)
    model.load_state_dict(sd)
    return model
