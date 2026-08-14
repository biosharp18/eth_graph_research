"""Score-level ensemble of two TGN checkpoints (seed-paired) on the frozen
paired evaluation and the deployment ranking.

Diagnostic question (2026-08-14 revisit): the historical-AUROC-best and
MRR-best configs sit at opposite ends of the mixture frontier — are their
strengths complementary (ensemble excels at both => one model should be able
to learn both) or shared (ensemble interpolates)?

Scores are z-normalized per model before averaging: over the pooled query
set for the paired eval, per candidate row for ranking (only relative order
within a comparison matters for both metrics).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tgat.data import load_daily_graph
from tgat.evaluate import build_negative_pools
from tgat.neighbors import NeighborStore
from tgat.train import auroc, average_precision

from .evaluate import tgn_scores
from .model import build_from_checkpoint
from .ranking import (_accumulate, _seen_flags, _test_day_positives,
                      filtered_rank, recency_ranking)
from .recency import PairRecency
from .streaming import iter_day_embeddings


def _z(x):
    x = np.asarray(x, np.float64)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def _candidate_scores(model, Z, rec, s, T, n_nodes, chunk=4096):
    zs = Z[s].unsqueeze(0)
    pf = None
    if rec is not None:
        pf = torch.from_numpy(rec.features_all(s, T)).to(Z.device)
    out = []
    for lo in range(0, n_nodes, chunk):
        zc = Z[lo:lo + chunk]
        out.append(model.link_logit(
            zs.expand(len(zc), -1), zc,
            None if pf is None else pf[lo:lo + chunk]))
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def ensemble_ranking(models, g, store, device):
    by_day = _test_day_positives(g)
    n_test = len(g.src) - g.val_end
    ranks = np.empty(n_test, np.float64)
    recs = [PairRecency(g.n_nodes, m.pair_feat_dim) if m.pair_feat_dim
            else None for m in models]
    iters = [iter_day_embeddings(m, g, store, device, by_day.keys())
             for m in models]
    ev_i, E = 0, len(g.src)
    for steps in zip(*iters):
        T = steps[0][0]
        Zs = [Z for _, Z in steps]
        while ev_i < E and g.day[ev_i] < T:
            for rec in recs:
                if rec is not None:
                    rec.observe_day(g.src[ev_i:ev_i + 1],
                                    g.dst[ev_i:ev_i + 1], int(g.day[ev_i]))
            ev_i += 1
        for s, entries in by_day[T].items():
            per_model = [_z(_candidate_scores(m, Z, rec, s, T, g.n_nodes))
                         for m, Z, rec in zip(models, Zs, recs)]
            scores = np.mean(per_model, axis=0)
            dsts = [d for _, d in entries]
            for j, d in entries:
                exclude = np.array([c for c in dsts if c != d], np.int64)
                ranks[j] = filtered_rank(scores, d, exclude)
    out = _accumulate(ranks, _seen_flags(g))
    out["n_test"] = n_test
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet", required=True)
    ap_.add_argument("--models-a", required=True)
    ap_.add_argument("--models-b", required=True)
    ap_.add_argument("--prefix", default="tgn")
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_.add_argument("--dim", type=int, default=100)
    ap_.add_argument("--k", type=int, default=20)
    args = ap_.parse_args()
    device = torch.device(args.device)

    g = load_daily_graph(args.parquet)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=args.k)

    res = {"paired": {}, "ranking": {}, "recency_mrr":
           recency_ranking(g)["mrr"]["all"]}
    for seed in args.seeds:
        models = [build_from_checkpoint(
            Path(d) / f"{args.prefix}_seed{seed}.pt", edge_feat_dim=store.F,
            raw_feat_dim=g.edge_feat.shape[1], dim=args.dim, device=device)
            for d in (args.models_a, args.models_b)]
        for strategy in ("random", "historical", "inductive"):
            negs = build_negative_pools(g, strategy, seed)
            y = np.r_[np.ones(negs.shape[0]), np.zeros(negs.shape[0])]
            zs = []
            for m in models:
                ps, ns, _ = tgn_scores(m, g, store, negs, device)
                zs.append(_z(np.r_[ps, ns]))
            sc = np.mean(zs, axis=0)
            st = res["paired"].setdefault(strategy, {"auroc": [], "ap": []})
            st["auroc"].append(auroc(y, sc))
            st["ap"].append(average_precision(y, sc))
        r = ensemble_ranking(models, g, store, device)
        for metric, strata in r.items():
            if not metric.startswith(("mrr", "hits")):
                continue
            for stratum, v in strata.items():
                res["ranking"].setdefault(metric, {}).setdefault(
                    stratum, []).append(v)
        print(f"seed {seed}: hist auroc "
              f"{res['paired']['historical']['auroc'][-1]:.4f} "
              f"MRR {r['mrr']['all']:.4f}", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
