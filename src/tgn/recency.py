"""Streaming pair-recency features for the link head.

Diagnostics (2026-08-14) showed the TGN's deployment-ranking failure is not
partner confusion: when the true destination is a past partner, ~98% of the
candidates outranking it are nodes the source never paid. The embeddings
cannot carry "c is *s's* partner" at full-ranking precision, so we hand the
head that information directly, computed with the exact streaming discipline
memory uses: features for day T reflect days < T only.

FEAT_DIM = 4: [pair seen, log1p(days since pair), dst seen, log1p(days since
dst was last paid)]. dt terms are zero when the corresponding flag is 0.
"""
from collections import defaultdict

import numpy as np

FEAT_DIM = 4


class PairRecency:
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.pair_last = defaultdict(dict)  # s -> {d: last day}
        self.dst_last = np.full(n_nodes, -1, np.int64)

    def observe_day(self, src, dst, day: int):
        for s, d in zip(np.asarray(src).tolist(), np.asarray(dst).tolist()):
            self.pair_last[s][d] = day
        self.dst_last[np.asarray(dst)] = day

    def _dst_cols(self, d_arr, day):
        ld = self.dst_last[d_arr]
        seen = ld >= 0
        dt = np.where(seen, np.log1p(day - ld), 0.0)
        return seen.astype(np.float32), dt.astype(np.float32)

    def features(self, s_arr, d_arr, day: int) -> np.ndarray:
        """(B, FEAT_DIM) float32 for pairs (s_arr[i], d_arr[i]) queried at day."""
        s_arr = np.asarray(s_arr)
        d_arr = np.asarray(d_arr)
        f = np.zeros((len(s_arr), FEAT_DIM), np.float32)
        for i, (s, d) in enumerate(zip(s_arr.tolist(), d_arr.tolist())):
            last = self.pair_last[s].get(d)
            if last is not None:
                f[i, 0] = 1.0
                f[i, 1] = np.log1p(day - last)
        f[:, 2], f[:, 3] = self._dst_cols(d_arr, day)
        return f

    def features_cross(self, s_arr, d_arr, day: int) -> np.ndarray:
        """(B, B, FEAT_DIM): features of (s_arr[i], d_arr[j]) for all i, j."""
        s_arr = np.asarray(s_arr)
        d_arr = np.asarray(d_arr)
        B = len(s_arr)
        f = np.zeros((B, B, FEAT_DIM), np.float32)
        dl = d_arr.tolist()
        for i, s in enumerate(s_arr.tolist()):
            cand = self.pair_last[s]
            if cand:
                for j, d in enumerate(dl):
                    last = cand.get(d)
                    if last is not None:
                        f[i, j, 0] = 1.0
                        f[i, j, 1] = np.log1p(day - last)
        seen, dt = self._dst_cols(d_arr, day)
        f[:, :, 2] = seen[None, :]
        f[:, :, 3] = dt[None, :]
        return f

    def features_all(self, s: int, day: int) -> np.ndarray:
        """(n_nodes, FEAT_DIM): features of (s, c) for every candidate c."""
        f = np.zeros((self.n_nodes, FEAT_DIM), np.float32)
        cand = self.pair_last[int(s)]
        if cand:
            cs = np.fromiter(cand.keys(), np.int64, len(cand))
            ld = np.fromiter(cand.values(), np.int64, len(cand))
            f[cs, 0] = 1.0
            f[cs, 1] = np.log1p(day - ld)
        f[:, 2], f[:, 3] = self._dst_cols(np.arange(self.n_nodes), day)
        return f
