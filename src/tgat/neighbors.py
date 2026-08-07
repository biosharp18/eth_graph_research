"""Strictly-past, recent-k temporal neighbor lookup over daily edges."""
import numpy as np


class NeighborStore:
    def __init__(self, src, dst, day, edge_feat, n_nodes: int, k: int = 20):
        self.k = k
        F = edge_feat.shape[1]
        self.F = F + 1  # + direction flag
        # per-node arrays sorted by day (input is already day-sorted; concat keeps order)
        nbr, days, feats = [[] for _ in range(n_nodes)], [[] for _ in range(n_nodes)], [[] for _ in range(n_nodes)]
        for i in range(len(src)):
            s, d = int(src[i]), int(dst[i])
            f_out = np.append(edge_feat[i], 1.0).astype(np.float32)
            f_in = np.append(edge_feat[i], 0.0).astype(np.float32)
            nbr[s].append(d); days[s].append(day[i]); feats[s].append(f_out)
            nbr[d].append(s); days[d].append(day[i]); feats[d].append(f_in)
        self.nbr = [np.array(x, dtype=np.int64) for x in nbr]
        self.days = [np.array(x, dtype=np.int64) for x in days]
        self.feats = [np.stack(x) if x else np.zeros((0, self.F), np.float32) for x in feats]

    def sample(self, nodes: np.ndarray, qdays: np.ndarray):
        B, k = len(nodes), self.k
        out_nbr = np.zeros((B, k), np.int64)
        out_dt = np.zeros((B, k), np.float32)
        out_feat = np.zeros((B, k, self.F), np.float32)
        out_mask = np.zeros((B, k), bool)
        for b in range(B):
            n, qd = int(nodes[b]), int(qdays[b])
            cut = int(np.searchsorted(self.days[n], qd, side="left"))
            lo = max(0, cut - k)
            m = cut - lo
            if m == 0:
                continue
            out_nbr[b, :m] = self.nbr[n][lo:cut]
            out_dt[b, :m] = qd - self.days[n][lo:cut]
            out_feat[b, :m] = self.feats[n][lo:cut]
            out_mask[b, :m] = True
        return out_nbr, out_dt, out_feat, out_mask
