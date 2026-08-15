"""Campaign-2 Phase 0: does WIDENING the receptive field add information?

Extends the Phase-0 signal audit with time-deep and day-context features and
dumps per-row features on the frozen inductive pools for day-split oracle
ablations. Streaming discipline identical to diag_inductive (day T sees
days < T only).
"""
import argparse
import json
from bisect import bisect_left
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from tgat.data import load_daily_graph
from tgat.evaluate import build_negative_pools, _pair_sets
from tgat.train import auroc

NEG_INF = -1e9

SIGNALS = [
    # campaign-1 model features (reference block, = FEAT_DIM_GLOBAL content)
    "pair_seen", "pair_recency", "dst_recency", "src_recency",
    "src_freq", "dst_freq", "dst_freq30", "deg_s", "deg_c",
    # T1 time-deep
    "pair_freq", "pair_age", "dst_freq7", "dst_freq90", "src_freq30",
    "out_deg_s", "in_deg_c", "rec_rank", "n_partners_s",
    # G3 day-context (identical for pos/neg of the same day; interactions only)
    "day_edges_prev", "day_edges_ema7",
]


class WideState:
    def __init__(self, n_nodes):
        self.n = n_nodes
        self.pair_last = defaultdict(dict)   # s -> {c: last day}
        self.pair_count = defaultdict(dict)  # s -> {c: n events}
        self.pair_first = defaultdict(dict)  # s -> {c: first day}
        self.dst_last = np.full(n_nodes, -1, np.int64)
        self.src_last = np.full(n_nodes, -1, np.int64)
        self.dst_count = np.zeros(n_nodes, np.int64)
        self.src_count = np.zeros(n_nodes, np.int64)
        self.deg = np.zeros(n_nodes, np.int64)
        self._nbrs = defaultdict(set)
        self.out_deg = np.zeros(n_nodes, np.int64)   # distinct partners as src
        self.in_deg = np.zeros(n_nodes, np.int64)    # distinct sources as dst
        self.dst_days = defaultdict(list)            # dst -> sorted day list
        self.src_days = defaultdict(list)
        self.daily_edges = {}                        # day -> count
        self.ema7 = 0.0

    def observe_day(self, src, dst, day):
        self.daily_edges[day] = len(src)
        self.ema7 = 0.75 * self.ema7 + 0.25 * len(src)
        for s, d in zip(src.tolist(), dst.tolist()):
            pc = self.pair_count[s]
            if d not in pc:
                pc[d] = 0
                self.pair_first[s][d] = day
                self.out_deg[s] += 1
                self.in_deg[d] += 1
            pc[d] += 1
            self.pair_last[s][d] = day
            for a, b in ((s, d), (d, s)):
                if b not in self._nbrs[a]:
                    self._nbrs[a].add(b)
                    self.deg[a] += 1
            self.dst_days[d].append(day)
            self.src_days[s].append(day)
        self.dst_last[dst] = day
        self.src_last[src] = day
        np.add.at(self.src_count, src, 1)
        np.add.at(self.dst_count, dst, 1)

    def _win(self, days, day, w):
        return len(days) - bisect_left(days, day - w)

    def features(self, s, c, day, prev_day):
        f = {}
        last = self.pair_last[s].get(c)
        f["pair_seen"] = 1.0 if last is not None else 0.0
        f["pair_recency"] = -np.log1p(day - last) if last is not None else NEG_INF
        f["dst_recency"] = (-np.log1p(day - self.dst_last[c])
                            if self.dst_last[c] >= 0 else NEG_INF)
        f["src_recency"] = (-np.log1p(day - self.src_last[s])
                            if self.src_last[s] >= 0 else NEG_INF)
        f["src_freq"] = np.log1p(self.src_count[s])
        f["dst_freq"] = np.log1p(self.dst_count[c])
        f["dst_freq30"] = np.log1p(self._win(self.dst_days[c], day, 30))
        f["deg_s"] = np.log1p(self.deg[s])
        f["deg_c"] = np.log1p(self.deg[c])
        f["pair_freq"] = np.log1p(self.pair_count[s].get(c, 0))
        first = self.pair_first[s].get(c)
        f["pair_age"] = np.log1p(day - first) if first is not None else 0.0
        f["dst_freq7"] = np.log1p(self._win(self.dst_days[c], day, 7))
        f["dst_freq90"] = np.log1p(self._win(self.dst_days[c], day, 90))
        f["src_freq30"] = np.log1p(self._win(self.src_days[s], day, 30))
        f["out_deg_s"] = np.log1p(self.out_deg[s])
        f["in_deg_c"] = np.log1p(self.in_deg[c])
        # rank of c by recency among s's partners: 0 = most recent
        if last is not None:
            better = sum(1 for lt in self.pair_last[s].values() if lt > last)
            f["rec_rank"] = np.log1p(better)
        else:
            f["rec_rank"] = np.log1p(max(len(self.pair_last[s]), 1)) + 1.0
        f["n_partners_s"] = np.log1p(len(self.pair_last[s]))
        f["day_edges_prev"] = np.log1p(self.daily_edges.get(prev_day, 0))
        f["day_edges_ema7"] = np.log1p(self.ema7)
        return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap.add_argument("--out", default="tgn_global/diag_widen.json")
    ap.add_argument("--npz", default="tgn_global/diag_widen.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    g = load_daily_graph(args.parquet)
    test = slice(g.val_end, len(g.src))
    ts, td, tt = g.src[test], g.dst[test], g.day[test]
    n_test = len(ts)
    _, before, _ = _pair_sets(g)
    pos_seen = np.array([(int(a), int(b)) in before for a, b in zip(ts, td)])
    pools = {s: build_negative_pools(g, "inductive", s) for s in args.seeds}

    state = WideState(g.n_nodes)
    pos_feats = {k: np.zeros(n_test) for k in SIGNALS}
    neg_feats = {s: {k: np.zeros(n_test) for k in SIGNALS} for s in args.seeds}

    all_days = np.unique(g.day)
    t0 = int(tt.min())
    prev_day = -1
    for T in all_days.tolist():
        lo = np.searchsorted(g.day, T, "left")
        hi = np.searchsorted(g.day, T, "right")
        if T >= t0:
            for i in np.where(tt == T)[0]:
                f = state.features(int(ts[i]), int(td[i]), T, prev_day)
                for k in SIGNALS:
                    pos_feats[k][i] = f[k]
                for s in args.seeds:
                    a, b = int(pools[s][i][0]), int(pools[s][i][1])
                    fn = state.features(a, b, T, prev_day)
                    for k in SIGNALS:
                        neg_feats[s][k][i] = fn[k]
        state.observe_day(g.src[lo:hi], g.dst[lo:hi], T)
        prev_day = T

    np.savez_compressed(
        args.npz, signals=np.array(SIGNALS),
        pos=np.stack([pos_feats[k] for k in SIGNALS], 1),
        pos_seen=pos_seen, test_day=tt,
        **{f"neg_{s}": np.stack([neg_feats[s][k] for k in SIGNALS], 1)
           for s in args.seeds})

    out = {"auroc": {}}
    y = np.r_[np.ones(n_test), np.zeros(n_test)]
    for k in SIGNALS:
        vals = [auroc(y, np.r_[pos_feats[k], neg_feats[s][k]])
                for s in args.seeds]
        out["auroc"][k] = round(float(np.mean(vals)), 4)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
