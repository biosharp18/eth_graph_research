"""Two-headed TGAT (Xu et al., ICLR 2020) over daily aggregated edges."""
import numpy as np
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """phi(dt) = cos(dt * w + b), learnable."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(
            1.0 / 10 ** np.linspace(0, 4, dim, dtype=np.float32)))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:  # [B,k] -> [B,k,dim]
        return torch.cos(dt.unsqueeze(-1) * self.w + self.b)


class TemporalAttention(nn.Module):
    """One multi-head attention layer over a node's past edges."""

    def __init__(self, dim, edge_feat_dim, time_dim, n_heads, dropout):
        super().__init__()
        kv_dim = dim + edge_feat_dim + time_dim
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          kdim=kv_dim, vdim=kv_dim,
                                          batch_first=True)
        self.merge = nn.Sequential(
            nn.Linear(dim + dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim))

    def forward(self, node_h, nbr_h, nbr_feat, nbr_tenc, mask):
        # node_h [B,d]; nbr_h [B,k,d]; nbr_feat [B,k,Fe]; nbr_tenc [B,k,dt]; mask [B,k]
        kv = torch.cat([nbr_h, nbr_feat, nbr_tenc], dim=-1)
        q = node_h.unsqueeze(1)
        no_nbr = ~mask.any(dim=1)
        pad = ~mask
        pad[no_nbr, 0] = False  # avoid all-masked NaN; output overwritten below
        out, _ = self.attn(q, kv, kv, key_padding_mask=pad)
        out = out.squeeze(1)
        out = torch.where(no_nbr.unsqueeze(1), torch.zeros_like(out), out)
        return self.merge(torch.cat([out, node_h], dim=-1))


class TGAT(nn.Module):
    def __init__(self, edge_feat_dim: int, dim: int = 100, n_layers: int = 2,
                 n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim, self.n_layers = dim, n_layers
        self.time_enc = TimeEncoder(dim)
        self.h0 = nn.Parameter(torch.zeros(dim))  # shared "zero feature" embedding
        self.layers = nn.ModuleList([
            TemporalAttention(dim, edge_feat_dim, dim, n_heads, dropout)
            for _ in range(n_layers)])
        self.link_head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, 1))
        self.amt_head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, 1))

    def _embed(self, nodes, days, store, device, depth):
        B = len(nodes)
        if depth == 0:
            return self.h0.expand(B, self.dim)
        nbr, dt, feat, mask = store.sample(nodes, days)
        k = nbr.shape[1]
        # recurse: neighbor embeddings at one layer shallower, at the edge's day
        nbr_days = (days[:, None] - dt.astype(np.int64)).reshape(-1)
        nbr_h = self._embed(nbr.reshape(-1), nbr_days, store, device,
                            depth - 1).reshape(B, k, self.dim)
        node_h = self._embed(nodes, days, store, device, 0)
        t_dt = torch.from_numpy(dt).to(device)
        return self.layers[depth - 1](
            node_h,
            nbr_h,
            torch.from_numpy(feat).to(device),
            self.time_enc(t_dt),
            torch.from_numpy(mask).to(device),
        )

    def embed(self, nodes, days, store, device):
        return self._embed(np.asarray(nodes), np.asarray(days), store, device,
                           self.n_layers)

    def forward(self, src_nodes, dst_nodes, days, store, device):
        zs = self.embed(src_nodes, days, store, device)
        zd = self.embed(dst_nodes, days, store, device)
        pair = torch.cat([zs, zd], dim=-1)
        return self.link_head(pair).squeeze(-1), self.amt_head(pair).squeeze(-1)
