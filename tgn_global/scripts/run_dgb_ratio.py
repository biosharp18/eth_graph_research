"""DGB protocol at a 3:1 negative:positive ratio (75%/25% split).

AUROC is prevalence-invariant, so any movement vs the 1:1 numbers isolates
the pool-exhaustion mechanism: historical/inductive pools are finite, and
3x demand forces more random padding. pad_frac is reported per cell.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tgat.data import load_daily_graph
from tgat.neighbors import NeighborStore
from tgat.train import auroc

from tgn.evaluate_dgb import (dgb_negatives, edgebank_scores_dgb, neg_days,
                              per_batch_auroc)
from tgn.model import build_from_checkpoint
from tgn.streaming import score_pairs_streaming

MODELS = {
    "tgn_hard": "figures/tgn_hard_models",
    "pfpop_mrr": "figures/tgn_pfpop_mrr_models",
    "g2_twohop_w1": "tgn_global/figures/g2_twohop_w1",
    "w2_wide_ind": "tgn_global/figures/w2_wide_ind",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-neg", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load_daily_graph(args.parquet)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=20)
    test = slice(g.val_end, len(g.src))
    ts, td, tt = g.src[test], g.dst[test], g.day[test]
    n = len(ts)

    res = {"n_neg": args.n_neg, "edgebank": {}, "models": {}}
    for strategy in ("random", "historical", "inductive"):
        negs = {}
        for seed in args.seeds:
            negs[seed] = dgb_negatives(g, strategy, seed, args.batch_size,
                                       n_neg=args.n_neg)
        qt_neg = neg_days(g, args.batch_size, n_neg=args.n_neg)
        for variant in ("inf", "tw"):
            cell = res["edgebank"].setdefault(variant, {}).setdefault(
                strategy, {"auroc": [], "pad_frac": []})
            for seed in args.seeds:
                ns, nd, pad = negs[seed]
                ps, nsc = edgebank_scores_dgb(g, variant, ns, nd,
                                              args.batch_size,
                                              n_neg=args.n_neg)
                cell["auroc"].append(per_batch_auroc(
                    ps, nsc, args.batch_size, args.n_neg)[0])
                cell["pad_frac"].append(float(pad.mean()))
        for name, mdir in MODELS.items():
            cell = res["models"].setdefault(name, {}).setdefault(
                strategy, {"auroc": [], "pad_frac": []})
            for seed in args.seeds:
                ns, nd, pad = negs[seed]
                model = build_from_checkpoint(
                    Path(mdir) / f"tgn_seed{seed}.pt", edge_feat_dim=store.F,
                    raw_feat_dim=g.edge_feat.shape[1], dim=100, device=device)
                qs = np.concatenate([ts, ns])
                qd = np.concatenate([td, nd])
                qt = np.concatenate([tt, qt_neg])
                lg, _ = score_pairs_streaming(model, g, store, device,
                                              qs, qd, qt)
                cell["auroc"].append(per_batch_auroc(
                    lg[:n], lg[n:], args.batch_size, args.n_neg)[0])
                cell["pad_frac"].append(float(pad.mean()))
            print(f"{strategy} {name}: "
                  f"{np.mean(cell['auroc']):.4f} "
                  f"(pad {np.mean(cell['pad_frac']):.3f})", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
