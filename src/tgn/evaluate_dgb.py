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


def dgb_negatives(g, strategy: str, seed: int, batch_size: int = 200):
    """Per-batch DGB negatives for the test span.

    Returns (neg_src, neg_dst, pad_flag) arrays of length n_test; negatives
    share their positive's timestamp. pad_flag marks rows where a short
    historical/inductive pool fell back to a random pair; it is always
    False for the random strategy."""
    rng = np.random.default_rng(seed)
    test = slice(g.val_end, len(g.src))
    ts, td = g.src[test], g.dst[test]
    n_test = len(ts)
    uniq_src = np.unique(g.src)
    uniq_dst = np.unique(g.dst)
    train_val_pairs = set(zip(g.src[:g.val_end].tolist(),
                              g.dst[:g.val_end].tolist()))
    hist = set(train_val_pairs)
    test_seen = set()

    ns = np.empty(n_test, np.int64)
    nd = np.empty(n_test, np.int64)
    pad = np.zeros(n_test, bool)

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
        if strategy == "random":
            for i in range(npos):
                s = int(bs[i])
                while True:
                    d = int(uniq_dst[rng.integers(len(uniq_dst))])
                    if (s, d) not in batch_pairs:
                        break
                ns[lo + i], nd[lo + i] = s, d
        else:
            pool_set = (hist if strategy == "historical" else test_seen)
            pool = sorted(pool_set - batch_pairs)
            take = min(npos, len(pool))
            chosen = ([pool[j] for j in
                       rng.choice(len(pool), size=take, replace=False)]
                      if take else [])
            for i in range(npos):
                if i < take:
                    ns[lo + i], nd[lo + i] = chosen[i]
                else:
                    ns[lo + i], nd[lo + i] = rand_pair(batch_pairs)
                    pad[lo + i] = True
        hist |= batch_pairs
        test_seen |= {p for p in batch_pairs if p not in train_val_pairs}
    return ns, nd, pad


def per_batch_auroc(pos_sc, neg_sc, batch_size: int = 200):
    vals = []
    for lo, hi in batch_bounds(len(pos_sc), batch_size):
        y = np.r_[np.ones(hi - lo), np.zeros(hi - lo)]
        vals.append(auroc(y, np.r_[pos_sc[lo:hi], neg_sc[lo:hi]]))
    return float(np.mean(vals)), vals


def edgebank_scores_dgb(g, variant, ns, nd, batch_size: int = 200):
    """EdgeBank under the DGB protocol: memory = all edges strictly before
    the current batch (advanced per batch, so earlier test batches count)."""
    test = slice(g.val_end, len(g.src))
    ts, td, tt = g.src[test], g.dst[test], g.day[test]
    w = int(tt.max() - tt.min()) + 1
    last = {}
    for s, d, t in zip(g.src[:g.val_end], g.dst[:g.val_end],
                       g.day[:g.val_end]):
        last[(int(s), int(d))] = int(t)
    pos_sc = np.zeros(len(ts))
    neg_sc = np.zeros(len(ts))

    def hit(pair, day):
        if pair not in last:
            return 0.0
        return 1.0 if variant == "inf" or last[pair] > day - w else 0.0

    for lo, hi in batch_bounds(len(ts), batch_size):
        for i in range(lo, hi):
            pos_sc[i] = hit((int(ts[i]), int(td[i])), int(tt[i]))
            neg_sc[i] = hit((int(ns[i]), int(nd[i])), int(tt[i]))
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
