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
                 dim: int = 100, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.time_enc = TimeEncoder(dim)
        # raw message: [own_mem | other_mem | timeenc(dt) | edge_feat | dir_flag]
        self.gru = nn.GRUCell(2 * dim + dim + raw_feat_dim + 1, dim)
        self.attn = TemporalAttention(dim, edge_feat_dim, dim, n_heads, dropout)
        self.link_head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, 1))
        self.amt_head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, 1))

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

    def embed(self, nodes, days, mem, store, device):
        nodes = np.asarray(nodes)
        days = np.asarray(days)
        nbr, dt, feat, mask = store.sample(nodes, days)
        B, k = nbr.shape
        node_h = mem[torch.from_numpy(nodes).to(device)]
        nbr_h = mem[torch.from_numpy(nbr.reshape(-1)).to(device)].reshape(
            B, k, self.dim)
        return self.attn(node_h, nbr_h,
                         torch.from_numpy(feat).to(device),
                         self.time_enc(torch.from_numpy(dt).to(device)),
                         torch.from_numpy(mask).to(device))

    def forward(self, src_nodes, dst_nodes, days, mem, store, device):
        zs = self.embed(src_nodes, days, mem, store, device)
        zd = self.embed(dst_nodes, days, mem, store, device)
        pair = torch.cat([zs, zd], dim=-1)
        return self.link_head(pair).squeeze(-1), self.amt_head(pair).squeeze(-1)
