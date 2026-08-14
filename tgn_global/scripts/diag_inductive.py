"""Phase-0 diagnosis: can streaming-safe GLOBAL/structural signals separate
the frozen inductive pools at all?

For every test event (s, d, T) and its inductive negative (ns, nd, T) (seeds
0-4, exact frozen pools via tgat.evaluate.build_negative_pools), compute
heuristic scores from days < T only, streaming day by day through the whole
edge stream. Report tie-aware AU-ROC per signal, overall and stratified by
(positive seen/unseen) x (negative pair already active before T).

No training, no model — this measures information content, not capacity.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tgat.data import load_daily_graph
from tgat.evaluate import build_negative_pools, _pair_sets
from tgat.train import auroc

NEG_INF = -1e9

SIGNALS = ["pair_seen", "pair_recency", "dst_recency", "src_recency",
           "dst_freq", "src_freq", "cn", "aa", "cn_recent30", "two_hop",
           "pref_attach", "dst_freq30"]


class GlobalState:
    """Streaming graph state; features for day T reflect days < T only."""

    def __init__(self, n_nodes):
        self.n = n_nodes
        self.nbr_last = [dict() for _ in range(n_nodes)]  # undirected: x -> last day
        self.out_last = [dict() for _ in range(n_nodes)]  # directed s -> {d: last day}
        self.pair_last = [dict() for _ in range(n_nodes)]
        self.node_last = np.full(n_nodes, -1, np.int64)
        self.dst_last = np.full(n_nodes, -1, np.int64)
        self.src_last = np.full(n_nodes, -1, np.int64)
        self.dst_count = np.zeros(n_nodes, np.int64)
        self.src_count = np.zeros(n_nodes, np.int64)
        self.dst_days = [list() for _ in range(n_nodes)]  # event days per dst

    def observe_day(self, src, dst, day):
        for s, d in zip(src.tolist(), dst.tolist()):
            self.nbr_last[s][d] = day
            self.nbr_last[d][s] = day
            self.out_last[s][d] = day
            self.pair_last[s][d] = day
            self.node_last[s] = day
            self.node_last[d] = day
            self.dst_last[d] = day
            self.src_last[s] = day
            self.dst_count[d] += 1
            self.src_count[s] += 1
            self.dst_days[d].append(day)

    def features(self, s, c, day):
        f = {}
        last = self.pair_last[s].get(c)
        f["pair_seen"] = 1.0 if last is not None else 0.0
        f["pair_recency"] = -np.log1p(day - last) if last is not None else NEG_INF
        f["dst_recency"] = (-np.log1p(day - self.dst_last[c])
                            if self.dst_last[c] >= 0 else NEG_INF)
        f["src_recency"] = (-np.log1p(day - self.src_last[s])
                            if self.src_last[s] >= 0 else NEG_INF)
        f["dst_freq"] = float(np.log1p(self.dst_count[c]))
        f["src_freq"] = float(np.log1p(self.src_count[s]))
        f["dst_freq30"] = float(np.log1p(
            sum(1 for t in self.dst_days[c] if t >= day - 30)))
        ns_, nc_ = self.nbr_last[s], self.nbr_last[c]
        if len(ns_) > len(nc_):
            ns_, nc_ = nc_, ns_
        cn = aa = cn30 = 0.0
        for x, ts in ns_.items():
            tc = nc_.get(x)
            if tc is not None:
                cn += 1.0
                aa += 1.0 / np.log1p(len(self.nbr_last[x]))
                if ts >= day - 30 and tc >= day - 30:
                    cn30 += 1.0
        f["cn"] = cn
        f["aa"] = aa
        f["cn_recent30"] = cn30
        hop2 = 0.0
        for x in self.out_last[s]:
            if c in self.out_last[x]:
                hop2 += 1.0
        f["two_hop"] = np.log1p(hop2)
        f["pref_attach"] = float(np.log1p(len(self.nbr_last[s]))
                                 + np.log1p(len(self.nbr_last[c])))
        return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap.add_argument("--out", default="tgn_global/diag_inductive.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--strategy", default="inductive")
    ap.add_argument("--dump-npz", default="",
                    help="also save raw per-row features for oracle fits")
    args = ap.parse_args()

    g = load_daily_graph(args.parquet)
    _, before, _ = _pair_sets(g)
    test = slice(g.val_end, len(g.src))
    ts, td, tt = g.src[test], g.dst[test], g.day[test]
    n_test = len(ts)
    pos_seen = np.array([(int(a), int(b)) in before for a, b in zip(ts, td)])

    pools = {seed: build_negative_pools(g, args.strategy, seed)
             for seed in args.seeds}

    # one streaming pass: at each test day, featurize that day's positives and
    # all seeds' negatives with state reflecting days < T
    state = GlobalState(g.n_nodes)
    day_of_row = g.day
    pos_feats = {k: np.zeros(n_test) for k in SIGNALS}
    neg_feats = {seed: {k: np.zeros(n_test) for k in SIGNALS}
                 for seed in args.seeds}
    neg_active = {seed: np.zeros(n_test, bool) for seed in args.seeds}

    all_days = np.unique(g.day)
    test_start_day = int(tt.min())
    row = 0
    for T in all_days.tolist():
        lo = np.searchsorted(day_of_row, T, "left")
        hi = np.searchsorted(day_of_row, T, "right")
        if T >= test_start_day:
            idx = np.where(tt == T)[0]
            for i in idx:
                f = state.features(int(ts[i]), int(td[i]), T)
                for k in SIGNALS:
                    pos_feats[k][i] = f[k]
                for seed in args.seeds:
                    a, b = int(pools[seed][i][0]), int(pools[seed][i][1])
                    fn = state.features(a, b, T)
                    for k in SIGNALS:
                        neg_feats[seed][k][i] = fn[k]
                    neg_active[seed][i] = state.pair_last[a].get(b) is not None
        state.observe_day(g.src[lo:hi], g.dst[lo:hi], T)
        row = hi

    out = {"strategy": args.strategy, "n_test": int(n_test),
           "pos_seen_frac": float(pos_seen.mean()),
           "neg_pair_active_frac": {
               str(s): float(neg_active[s].mean()) for s in args.seeds},
           "auroc": {}, "auroc_strata": {}}

    y = np.r_[np.ones(n_test), np.zeros(n_test)]
    for k in SIGNALS:
        vals = [auroc(y, np.r_[pos_feats[k], neg_feats[seed][k]])
                for seed in args.seeds]
        out["auroc"][k] = {"mean": float(np.mean(vals)),
                           "std": float(np.std(vals)), "per_seed": vals}

    # strata: newpos_vs_newneg (unseen positive vs never-active negative pair),
    # allpos_vs_activeneg (any positive vs negative already active before T)
    for name, pmask_fn, nmask_fn in (
            ("newpos_vs_newneg", lambda s: ~pos_seen, lambda s: ~neg_active[s]),
            ("seenpos_vs_newneg", lambda s: pos_seen, lambda s: ~neg_active[s]),
            ("allpos_vs_activeneg", lambda s: np.ones(n_test, bool),
             lambda s: neg_active[s])):
        out["auroc_strata"][name] = {}
        for k in SIGNALS:
            vals = []
            for seed in args.seeds:
                pm, nm = pmask_fn(seed), nmask_fn(seed)
                yy = np.r_[np.ones(pm.sum()), np.zeros(nm.sum())]
                sc = np.r_[pos_feats[k][pm], neg_feats[seed][k][nm]]
                vals.append(auroc(yy, sc))
            out["auroc_strata"][name][k] = {"mean": float(np.mean(vals)),
                                            "std": float(np.std(vals))}

    if args.dump_npz:
        np.savez_compressed(
            args.dump_npz,
            signals=np.array(SIGNALS),
            pos=np.stack([pos_feats[k] for k in SIGNALS], 1),
            pos_seen=pos_seen, test_day=tt,
            **{f"neg_{s}": np.stack([neg_feats[s][k] for k in SIGNALS], 1)
               for s in args.seeds},
            **{f"neg_active_{s}": neg_active[s] for s in args.seeds})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["auroc"], indent=2))
    print("strata:", json.dumps(
        {k: {s: round(v["mean"], 4) for s, v in d.items()}
         for k, d in out["auroc_strata"].items()}, indent=2))
    print("neg active frac:", out["neg_pair_active_frac"])


if __name__ == "__main__":
    main()
