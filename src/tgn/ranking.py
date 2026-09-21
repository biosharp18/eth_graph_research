"""Deployment-style ranking: each test event's true destination ranked against
all nodes, filtered (same-source same-day positives other than the target are
excluded). Tie-aware rank = 1 + #strictly-better + 0.5 * #tied-others.

Recency heuristic: score(s, c, T) = -(T - last_day(s, c)) if (s, c) appeared
before day T (anywhere in the stream, streaming forward), else -inf.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tgat.data import DailyGraph, load_daily_graph
from tgat.evaluate import _pair_sets
from tgat.neighbors import NeighborStore

from .model import build_from_checkpoint
from .recency import PairRecency
from .streaming import iter_day_embeddings

HITS = (1, 10, 100)


def filtered_rank(scores, target, exclude):
    s = np.asarray(scores, dtype=np.float64)
    keep = np.ones(len(s), bool)
    keep[exclude] = False
    keep[target] = True
    st = s[target]
    better = int(((s > st) & keep).sum())
    tied = int(((s == st) & keep).sum()) - 1
    return 1.0 + better + 0.5 * tied


def _accumulate(ranks, seen_flags):
    ranks = np.asarray(ranks, dtype=np.float64)
    seen = np.asarray(seen_flags, dtype=bool)
    strata = {"all": np.ones(len(ranks), bool), "seen": seen, "unseen": ~seen}
    out = {"mrr": {k: float((1.0 / ranks[m]).mean()) if m.any() else float("nan")
                   for k, m in strata.items()}}
    for h in HITS:
        out[f"hits{h}"] = {k: float((ranks[m] <= h).mean()) if m.any()
                           else float("nan")
                           for k, m in strata.items()}
    return out


def _test_day_positives(g: DailyGraph):
    """day -> source -> list of (test-slice index, destination)."""
    by_day = defaultdict(lambda: defaultdict(list))
    for j in range(g.val_end, len(g.src)):
        by_day[int(g.day[j])][int(g.src[j])].append((j - g.val_end,
                                                     int(g.dst[j])))
    return by_day


def _seen_flags(g: DailyGraph):
    _, before, _ = _pair_sets(g)
    return [(int(s), int(d)) in before
            for s, d in zip(g.src[g.val_end:], g.dst[g.val_end:])]


def recency_ranking(g: DailyGraph) -> dict:
    by_day = _test_day_positives(g)
    n_test = len(g.src) - g.val_end
    ranks = np.empty(n_test, np.float64)
    src_last = defaultdict(dict)  # s -> {c: last day the pair (s, c) occurred}
    ev_i, E = 0, len(g.src)
    for T in range(g.n_days):
        if T in by_day:
            for s, entries in by_day[T].items():
                cand = src_last[s]
                dsts = [d for _, d in entries]
                for j, d in entries:
                    exclude = [c for c in dsts if c != d]
                    n_excl_seen = sum(1 for c in exclude if c in cand)
                    if d in cand:
                        sd = cand[d]
                        better = sum(1 for c, lt in cand.items()
                                     if lt > sd and c != d and c not in exclude)
                        tied = sum(1 for c, lt in cand.items()
                                   if lt == sd and c != d and c not in exclude)
                        ranks[j] = 1.0 + better + 0.5 * tied
                    else:
                        n_unseen = g.n_nodes - len(cand)
                        tied_others = (n_unseen - 1) - (len(exclude) - n_excl_seen)
                        better = len(cand) - n_excl_seen
                        ranks[j] = 1.0 + better + 0.5 * tied_others
        while ev_i < E and g.day[ev_i] == T:
            src_last[int(g.src[ev_i])][int(g.dst[ev_i])] = T
            ev_i += 1
    out = _accumulate(ranks, _seen_flags(g))
    out["n_test"] = n_test
    out["n_candidates"] = g.n_nodes
    return out


@torch.no_grad()
def tgn_ranking(model, g: DailyGraph, store, device, chunk=4096) -> dict:
    model.eval()
    by_day = _test_day_positives(g)
    n_test = len(g.src) - g.val_end
    ranks = np.empty(n_test, np.float64)
    rec = (PairRecency(g.n_nodes, model.pair_feat_dim)
           if model.pair_feat_dim else None)
    ev_i, E = 0, len(g.src)
    for T, Z in iter_day_embeddings(model, g, store, device, by_day.keys()):
        if rec is not None:
            while ev_i < E and g.day[ev_i] < T:
                rec.observe_day(g.src[ev_i:ev_i + 1], g.dst[ev_i:ev_i + 1],
                                int(g.day[ev_i]))
                ev_i += 1
        for s, entries in by_day[T].items():
            zs = Z[s].unsqueeze(0)
            pf = None
            if rec is not None:
                pf = torch.from_numpy(rec.features_all(s, T)).to(device)
            scores = []
            for lo in range(0, g.n_nodes, chunk):
                zc = Z[lo:lo + chunk]
                scores.append(model.link_logit(
                    zs.expand(len(zc), -1), zc,
                    None if pf is None else pf[lo:lo + chunk]))
            scores = torch.cat(scores).cpu().numpy()
            dsts = [d for _, d in entries]
            for j, d in entries:
                exclude = np.array([c for c in dsts if c != d], np.int64)
                ranks[j] = filtered_rank(scores, d, exclude)
    out = _accumulate(ranks, _seen_flags(g))
    out["n_test"] = n_test
    out["n_candidates"] = g.n_nodes
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet",
                     default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap_.add_argument("--models", default="figures/tgn_models")
    ap_.add_argument("--out", default="figures/tgn_ranking.json")
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_.add_argument("--dim", type=int, default=100)
    ap_.add_argument("--k", type=int, default=20)
    ap_.add_argument("--nbr-mode", choices=["recent", "strat"],
                     default="recent")
    ap_.add_argument("--k-inner", type=int, default=0,
                     help="must match training (not in the checkpoint)")
    args = ap_.parse_args()
    device = torch.device(args.device)

    g = load_daily_graph(args.parquet)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes,
                          k=args.k, mode=args.nbr_mode)

    rec = recency_ranking(g)
    res = {"recency": {k: rec[k] for k in rec if k.startswith(("mrr", "hits"))},
           "tgn": {}, "n_test": rec["n_test"],
           "n_candidates": rec["n_candidates"]}
    for seed in args.seeds:
        model = build_from_checkpoint(
            Path(args.models) / f"tgn_seed{seed}.pt", edge_feat_dim=store.F,
            raw_feat_dim=g.edge_feat.shape[1], dim=args.dim, device=device)
        r = tgn_ranking(model, g, store, device)
        for metric, strata in r.items():
            if not metric.startswith(("mrr", "hits")):
                continue
            for stratum, v in strata.items():
                res["tgn"].setdefault(metric, {}).setdefault(stratum, []).append(v)
        print(f"seed {seed}: MRR {r['mrr']['all']:.4f} "
              f"hits@10 {r['hits10']['all']:.4f}", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
