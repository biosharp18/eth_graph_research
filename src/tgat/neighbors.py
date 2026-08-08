"""Strictly-past, recent-k temporal neighbor lookup over daily edges.

Internals use a flat CSR-style layout keyed by a composite (node, day) integer
so that both construction and sampling avoid per-node Python objects and
per-row Python loops. See docs/superpowers/sdd/2026-08-07-tgat-daily-flow for
the design this implements.
"""
import numpy as np


class NeighborStore:
    def __init__(self, src, dst, day, edge_feat, n_nodes: int, k: int = 20):
        self.k = k
        self.n_nodes = n_nodes
        F = edge_feat.shape[1]
        self.F = F + 1  # + direction flag

        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        day = np.asarray(day, dtype=np.int64)
        E = len(src)

        # Duplicate each edge into two incidences (one per endpoint), interleaved
        # so that, for edge i, the source-role incidence lands at position 2*i and
        # the destination-role incidence at position 2*i+1. This preserves the old
        # per-edge iteration order (source appended, then destination) including
        # for self-loops, so a stable sort on (node, day) reproduces the exact old
        # per-node append order for same-day ties.
        inc_node = np.empty(2 * E, dtype=np.int64)
        inc_other = np.empty(2 * E, dtype=np.int64)
        inc_day = np.empty(2 * E, dtype=np.int64)
        inc_feat = np.zeros((2 * E, self.F), dtype=np.float32)

        inc_node[0::2] = src
        inc_node[1::2] = dst
        inc_other[0::2] = dst
        inc_other[1::2] = src
        inc_day[0::2] = day
        inc_day[1::2] = day
        feat32 = np.asarray(edge_feat, dtype=np.float32)
        inc_feat[0::2, :F] = feat32
        inc_feat[0::2, F] = 1.0  # source-side: this node sent
        inc_feat[1::2, :F] = feat32
        inc_feat[1::2, F] = 0.0  # destination-side: this node received

        D = int(day.max()) + 2 if E > 0 else 1
        self.D = D

        keys = inc_node * D + inc_day
        order = np.argsort(keys, kind="stable")

        self.keys = keys[order]
        self.inc_other = inc_other[order]
        self.inc_day = inc_day[order]
        self.inc_feat = inc_feat[order]
        inc_node_sorted = inc_node[order]
        self.node_start = np.searchsorted(inc_node_sorted, np.arange(n_nodes), side="left")
        self.node_end = np.searchsorted(inc_node_sorted, np.arange(n_nodes), side="right")

    def sample(self, nodes: np.ndarray, qdays: np.ndarray):
        nodes = np.asarray(nodes, dtype=np.int64)
        qdays = np.asarray(qdays, dtype=np.int64)
        B, k = len(nodes), self.k

        if len(self.keys) == 0:
            return (np.zeros((B, k), np.int64), np.zeros((B, k), np.float32),
                    np.zeros((B, k, self.F), np.float32), np.zeros((B, k), bool))

        cut = np.searchsorted(self.keys, nodes * self.D + qdays, side="left")
        cut = np.minimum(cut, self.node_end[nodes])
        lo = np.maximum(cut - k, self.node_start[nodes])

        idx = lo[:, None] + np.arange(k)[None, :]
        valid = idx < cut[:, None]
        idx_clipped = np.minimum(idx, len(self.keys) - 1)

        gathered_other = self.inc_other[idx_clipped]
        gathered_day = self.inc_day[idx_clipped]
        gathered_feat = self.inc_feat[idx_clipped]

        out_nbr = np.where(valid, gathered_other, 0).astype(np.int64)
        out_dt = ((qdays[:, None] - gathered_day) * valid).astype(np.float32)
        out_feat = gathered_feat * valid[:, :, None]
        out_mask = valid

        return out_nbr, out_dt, out_feat, out_mask
