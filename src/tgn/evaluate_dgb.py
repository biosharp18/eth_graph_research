"""Paper-faithful DGB evaluation (Poursafaei et al., NeurIPS 2022).

Differences from the legacy protocol in tgat.evaluate/tgn.evaluate (which is
kept unchanged for internal comparability):

- AUROC is computed PER BATCH of 200 consecutive test edges (1:1 pos/neg)
  and the reported number is the MEAN over batches, not one global AUROC.
- random NS: the negative keeps the positive's source and timestamp and
  swaps in a destination drawn uniformly from the dataset's destination
  nodes; the only exclusion is colliding with a current-batch positive.
- historical NS: negatives drawn (without replacement when possible) from
  pairs seen in ANY edge strictly before the current batch — the history
  ACCUMULATES THROUGH TEST — minus current-batch positives; short pools are
  padded with random pairs.
- inductive NS: same, but the pool is pairs seen ONLY during test so far
  (never in train/val), minus current-batch positives; padded with random
  pairs (early test batches are mostly padding by construction).

Padding fraction is reported alongside the metric because the historical/
inductive numbers depend on it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tgat.data import load_daily_graph
from tgat.neighbors import NeighborStore
from tgat.train import auroc

from .model import build_from_checkpoint
from .streaming import score_pairs_streaming


def batch_bounds(n_test: int, batch_size: int):
    return [(lo, min(lo + batch_size, n_test))
            for lo in range(0, n_test, batch_size)]


def dgb_negatives(g, strategy: str, seed: int, batch_size: int = 200,
                  span=None, n_neg: int = 1):
    """Per-batch DGB negatives for an evaluation span (default: test).

    span = (lo, hi) event indices; "inductive" pairs are those never seen
    BEFORE the span that have been observed within it so far — for the val
    span that means val-only pairs, mirroring the test-span construction.

    n_neg negatives per positive (paper: 1). Historical/inductive pools are
    sampled without replacement per batch up to npos*n_neg, then padded with
    random pairs — so raising n_neg raises pad_frac wherever the pool is
    smaller than the demand.

    Returns (neg_src, neg_dst, pad_flag) arrays of length span*n_neg,
    ordered batch-major (a batch's negatives are contiguous); negatives
    share the batch's timestamps. pad_flag marks random-padded rows; it is
    always False for the random strategy."""
    rng = np.random.default_rng(seed)
    lo0, hi0 = span if span is not None else (g.val_end, len(g.src))
    ts, td = g.src[lo0:hi0], g.dst[lo0:hi0]
    n_test = len(ts)
    uniq_src = np.unique(g.src)
    uniq_dst = np.unique(g.dst)
    train_val_pairs = set(zip(g.src[:lo0].tolist(), g.dst[:lo0].tolist()))
    hist = set(train_val_pairs)
    test_seen = set()

    ns = np.empty(n_test * n_neg, np.int64)
    nd = np.empty(n_test * n_neg, np.int64)
    pad = np.zeros(n_test * n_neg, bool)

    def rand_pair(batch_pairs):
        while True:
            s = int(uniq_src[rng.integers(len(uniq_src))])
            d = int(uniq_dst[rng.integers(len(uniq_dst))])
            if (s, d) not in batch_pairs:
                return s, d

    for lo, hi in batch_bounds(n_test, batch_size):
        bs, bd = ts[lo:hi], td[lo:hi]
        batch_pairs = set(zip(bs.tolist(), bd.tolist()))
        npos = hi - lo
        need = npos * n_neg
        olo = lo * n_neg  # output offset (batch-major)
        if strategy == "random":
            for i in range(need):
                s = int(bs[i % npos])
                while True:
                    d = int(uniq_dst[rng.integers(len(uniq_dst))])
                    if (s, d) not in batch_pairs:
                        break
                ns[olo + i], nd[olo + i] = s, d
        else:
            pool_set = (hist if strategy == "historical" else test_seen)
            pool = sorted(pool_set - batch_pairs)
            take = min(need, len(pool))
            chosen = ([pool[j] for j in
                       rng.choice(len(pool), size=take, replace=False)]
                      if take else [])
            for i in range(need):
                if i < take:
                    ns[olo + i], nd[olo + i] = chosen[i]
                else:
                    ns[olo + i], nd[olo + i] = rand_pair(batch_pairs)
                    pad[olo + i] = True
        hist |= batch_pairs
        test_seen |= {p for p in batch_pairs if p not in train_val_pairs}
    return ns, nd, pad


def neg_days(g, batch_size: int = 200, span=None, n_neg: int = 1):
    """Timestamps for dgb_negatives' batch-major rows: negative i belongs to
    positive (i // n_neg) ... within its batch, cycled i % npos."""
    lo0, hi0 = span if span is not None else (g.val_end, len(g.src))
    tt = g.day[lo0:hi0]
    out = np.empty(len(tt) * n_neg, np.int64)
    for lo, hi in batch_bounds(len(tt), batch_size):
        npos = hi - lo
        for i in range(npos * n_neg):
            out[lo * n_neg + i] = tt[lo + i % npos]
    return out


def per_batch_auroc(pos_sc, neg_sc, batch_size: int = 200, n_neg: int = 1):
    vals = []
    for lo, hi in batch_bounds(len(pos_sc), batch_size):
        npos = hi - lo
        y = np.r_[np.ones(npos), np.zeros(npos * n_neg)]
        vals.append(auroc(y, np.r_[pos_sc[lo:hi],
                                   neg_sc[lo * n_neg:hi * n_neg]]))
    return float(np.mean(vals)), vals


def edgebank_scores_dgb(g, variant, ns, nd, batch_size: int = 200,
                        span=None, n_neg: int = 1, freeze_memory=False):
    """EdgeBank under the DGB protocol: memory = all edges strictly before
    the current batch (advanced per batch, so earlier in-span batches
    count). span = (lo, hi) event indices; default is the test span.
    ns/nd are batch-major with n_neg negatives per positive.
    freeze_memory=True stops memory updates at the span start (train+val
    only) — the ablation separating memory accumulation from the sampler's
    pool accumulation."""
    lo0, hi0 = span if span is not None else (g.val_end, len(g.src))
    ts, td, tt = g.src[lo0:hi0], g.dst[lo0:hi0], g.day[lo0:hi0]
    w = int(tt.max() - tt.min()) + 1
    last = {}
    for s, d, t in zip(g.src[:lo0], g.dst[:lo0], g.day[:lo0]):
        last[(int(s), int(d))] = int(t)
    pos_sc = np.zeros(len(ts))
    neg_sc = np.zeros(len(ts) * n_neg)

    def hit(pair, day):
        if pair not in last:
            return 0.0
        return 1.0 if variant == "inf" or last[pair] > day - w else 0.0

    for lo, hi in batch_bounds(len(ts), batch_size):
        npos = hi - lo
        for i in range(lo, hi):
            pos_sc[i] = hit((int(ts[i]), int(td[i])), int(tt[i]))
        for j in range(npos * n_neg):
            o = lo * n_neg + j
            day = int(tt[lo + j % npos])
            neg_sc[o] = hit((int(ns[o]), int(nd[o])), day)
        if not freeze_memory:
            for i in range(lo, hi):
                last[(int(ts[i]), int(td[i]))] = int(tt[i])
    return pos_sc, neg_sc


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet",
                     default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap_.add_argument("--models", default="")
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap_.add_argument("--batch-size", type=int, default=200)
    ap_.add_argument("--k", type=int, default=20)
    ap_.add_argument("--nbr-mode", choices=["recent", "strat"],
                     default="recent")
    ap_.add_argument("--dim", type=int, default=100)
    ap_.add_argument("--edgebank", action="store_true",
                     help="also report EdgeBank inf/tw under this protocol")
    ap_.add_argument("--device",
                     default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap_.parse_args()
    device = torch.device(args.device)

    g = load_daily_graph(args.parquet)
    test = slice(g.val_end, len(g.src))
    ts, td, tt = g.src[test], g.dst[test], g.day[test]
    store = (NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes,
                           k=args.k, mode=args.nbr_mode)
             if args.models else None)

    res = {"protocol": "dgb_per_batch", "batch_size": args.batch_size,
           "n_test": int(len(ts)), "models": args.models, "tgn": {},
           "edgebank": {}}
    for strategy in ("random", "historical", "inductive"):
        if args.models:
            res["tgn"][strategy] = {"auroc_batch_mean": [],
                                    "auroc_global": [], "pad_frac": []}
        for seed in args.seeds:
            ns, nd, pad = dgb_negatives(g, strategy, seed, args.batch_size)
            if args.edgebank:
                for variant in ("inf", "tw"):
                    ps, nsc = edgebank_scores_dgb(g, variant, ns, nd,
                                                  args.batch_size)
                    d = res["edgebank"].setdefault(variant, {}).setdefault(
                        strategy, {"auroc_batch_mean": [], "pad_frac": []})
                    d["auroc_batch_mean"].append(
                        per_batch_auroc(ps, nsc, args.batch_size)[0])
                    d["pad_frac"].append(float(pad.mean()))
            if not args.models:
                continue
            model = build_from_checkpoint(
                Path(args.models) / f"tgn_seed{seed}.pt",
                edge_feat_dim=store.F, raw_feat_dim=g.edge_feat.shape[1],
                dim=args.dim, device=device)
            qs = np.concatenate([ts, ns])
            qd = np.concatenate([td, nd])
            qt = np.concatenate([tt, tt])
            lg, _ = score_pairs_streaming(model, g, store, device, qs, qd, qt)
            n = len(ts)
            mean_b, _ = per_batch_auroc(lg[:n], lg[n:], args.batch_size)
            y = np.r_[np.ones(n), np.zeros(n)]
            res["tgn"][strategy]["auroc_batch_mean"].append(mean_b)
            res["tgn"][strategy]["auroc_global"].append(
                auroc(y, np.r_[lg[:n], lg[n:]]))
            res["tgn"][strategy]["pad_frac"].append(float(pad.mean()))
        if args.models:
            print(f"{strategy}: batch-mean auroc "
                  f"{np.mean(res['tgn'][strategy]['auroc_batch_mean']):.4f} "
                  f"(pad {np.mean(res['tgn'][strategy]['pad_frac']):.3f})",
                  flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
