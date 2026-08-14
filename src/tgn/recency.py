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
from collections import defaultdict, deque

import numpy as np

FEAT_DIM = 4
FEAT_DIM_BUCKETS = 7  # + pair-dt indicators (==1, ==2, <=7) for sharp recency
# + global-in-time node activity: src seen/recency, src/dst total frequency,
# dst 30-day frequency, undirected degrees. Phase-0 diagnosis (tgn_global/)
# measured these as the information the inductive protocol needs and the
# 20-edge attention + GRU memory cannot count.
FEAT_DIM_GLOBAL = 11
# global + the sharp pair-dt indicators (==1, ==2, <=7) at cols 11-13
FEAT_DIM_GLOBAL_BUCKETS = 14
WINDOW = 30  # days, for the rolling dst-frequency feature


class PairRecency:
    def __init__(self, n_nodes: int, feat_dim: int = FEAT_DIM):
        assert feat_dim in (FEAT_DIM, FEAT_DIM_BUCKETS, FEAT_DIM_GLOBAL,
                            FEAT_DIM_GLOBAL_BUCKETS)
        self.n_nodes = n_nodes
        self.feat_dim = feat_dim
        self.pair_last = defaultdict(dict)  # s -> {d: last day}
        self.dst_last = np.full(n_nodes, -1, np.int64)
        # global-activity state (maintained always; used when feat_dim == 11)
        self.src_last = np.full(n_nodes, -1, np.int64)
        self.src_count = np.zeros(n_nodes, np.int64)
        self.dst_count = np.zeros(n_nodes, np.int64)
        self.deg = np.zeros(n_nodes, np.int64)  # undirected distinct degree
        self._nbrs = defaultdict(set)
        # rolling window of past events for dst_freq30; queries must be
        # day-monotonic (all callers stream days in order)
        self._win = deque()
        self._cnt30 = np.zeros(n_nodes, np.int64)

    def _pair_cols(self, f, i, day, last):
        """Fill row/cell i of feature array f for a seen pair last seen at
        `last` (f indexed [i, col] or [i, j, col] via tuple i)."""
        dt = day - last
        f[i + (0,)] = 1.0
        f[i + (1,)] = np.log1p(dt)
        if self.feat_dim == FEAT_DIM_BUCKETS:
            f[i + (4,)] = 1.0 if dt <= 1 else 0.0
            f[i + (5,)] = 1.0 if dt == 2 else 0.0
            f[i + (6,)] = 1.0 if dt <= 7 else 0.0
        elif self.feat_dim == FEAT_DIM_GLOBAL_BUCKETS:
            f[i + (11,)] = 1.0 if dt <= 1 else 0.0
            f[i + (12,)] = 1.0 if dt == 2 else 0.0
            f[i + (13,)] = 1.0 if dt <= 7 else 0.0

    def observe_day(self, src, dst, day: int):
        src = np.asarray(src)
        dst = np.asarray(dst)
        for s, d in zip(src.tolist(), dst.tolist()):
            self.pair_last[s][d] = day
            self._win.append((day, d))
            self._cnt30[d] += 1
            for a, b in ((s, d), (d, s)):
                if b not in self._nbrs[a]:
                    self._nbrs[a].add(b)
                    self.deg[a] += 1
        self.dst_last[dst] = day
        self.src_last[src] = day
        np.add.at(self.src_count, src, 1)
        np.add.at(self.dst_count, dst, 1)

    def _advance_window(self, day: int):
        """Drop events older than WINDOW days before the query day. Queries
        must be day-monotonic (all callers stream days in order)."""
        while self._win and self._win[0][0] < day - WINDOW:
            _, d = self._win.popleft()
            self._cnt30[d] -= 1

    def _dst_cols(self, d_arr, day):
        ld = self.dst_last[d_arr]
        seen = ld >= 0
        dt = np.where(seen, np.log1p(day - ld), 0.0)
        return seen.astype(np.float32), dt.astype(np.float32)

    def _global_cols(self, s_arr, d_arr, day):
        """(B, 7) global-activity block for feat_dim == FEAT_DIM_GLOBAL:
        [src_seen, log1p(dt_src), log1p(src_freq), log1p(dst_freq),
         log1p(dst_freq_30d), log1p(deg_s), log1p(deg_d)]."""
        self._advance_window(day)
        ls = self.src_last[s_arr]
        s_seen = ls >= 0
        out = np.zeros((len(s_arr), 7), np.float32)
        out[:, 0] = s_seen
        out[:, 1] = np.where(s_seen, np.log1p(day - ls), 0.0)
        out[:, 2] = np.log1p(self.src_count[s_arr])
        out[:, 3] = np.log1p(self.dst_count[d_arr])
        out[:, 4] = np.log1p(self._cnt30[d_arr])
        out[:, 5] = np.log1p(self.deg[s_arr])
        out[:, 6] = np.log1p(self.deg[d_arr])
        return out

    def features(self, s_arr, d_arr, day: int) -> np.ndarray:
        """(B, feat_dim) float32 for pairs (s_arr[i], d_arr[i]) queried at day."""
        s_arr = np.asarray(s_arr)
        d_arr = np.asarray(d_arr)
        f = np.zeros((len(s_arr), self.feat_dim), np.float32)
        for i, (s, d) in enumerate(zip(s_arr.tolist(), d_arr.tolist())):
            last = self.pair_last[s].get(d)
            if last is not None:
                self._pair_cols(f, (i,), day, last)
        f[:, 2], f[:, 3] = self._dst_cols(d_arr, day)
        if self.feat_dim >= FEAT_DIM_GLOBAL:
            f[:, 4:11] = self._global_cols(s_arr, d_arr, day)
        return f

    def features_cross(self, s_arr, d_arr, day: int) -> np.ndarray:
        """(B, B, feat_dim): features of (s_arr[i], d_arr[j]) for all i, j."""
        s_arr = np.asarray(s_arr)
        d_arr = np.asarray(d_arr)
        B = len(s_arr)
        f = np.zeros((B, B, self.feat_dim), np.float32)
        dl = d_arr.tolist()
        for i, s in enumerate(s_arr.tolist()):
            cand = self.pair_last[s]
            if cand:
                for j, d in enumerate(dl):
                    last = cand.get(d)
                    if last is not None:
                        self._pair_cols(f, (i, j), day, last)
        seen, dt = self._dst_cols(d_arr, day)
        f[:, :, 2] = seen[None, :]
        f[:, :, 3] = dt[None, :]
        if self.feat_dim >= FEAT_DIM_GLOBAL:
            # src-side cols broadcast over j, dst-side cols over i
            g_s = self._global_cols(s_arr, s_arr, day)  # cols 0-2, 5 valid
            g_d = self._global_cols(d_arr, d_arr, day)  # cols 3, 4, 6 valid
            for col in (0, 1, 2, 5):
                f[:, :, 4 + col] = g_s[:, col][:, None]
            for col in (3, 4, 6):
                f[:, :, 4 + col] = g_d[:, col][None, :]
        return f

    def features_all(self, s: int, day: int) -> np.ndarray:
        """(n_nodes, feat_dim): features of (s, c) for every candidate c."""
        f = np.zeros((self.n_nodes, self.feat_dim), np.float32)
        cand = self.pair_last[int(s)]
        if cand:
            cs = np.fromiter(cand.keys(), np.int64, len(cand))
            ld = np.fromiter(cand.values(), np.int64, len(cand))
            dt = day - ld
            f[cs, 0] = 1.0
            f[cs, 1] = np.log1p(dt)
            if self.feat_dim == FEAT_DIM_BUCKETS:
                f[cs, 4] = (dt <= 1).astype(np.float32)
                f[cs, 5] = (dt == 2).astype(np.float32)
                f[cs, 6] = (dt <= 7).astype(np.float32)
            elif self.feat_dim == FEAT_DIM_GLOBAL_BUCKETS:
                f[cs, 11] = (dt <= 1).astype(np.float32)
                f[cs, 12] = (dt == 2).astype(np.float32)
                f[cs, 13] = (dt <= 7).astype(np.float32)
        all_nodes = np.arange(self.n_nodes)
        f[:, 2], f[:, 3] = self._dst_cols(all_nodes, day)
        if self.feat_dim >= FEAT_DIM_GLOBAL:
            g = self._global_cols(np.full(self.n_nodes, int(s)), all_nodes,
                                  day)
            f[:, 4:11] = g
        return f
