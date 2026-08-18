"""Two additional DGB test modes (user directive, 2026-08-16):

- inductive_sym: negatives sampled exactly as the standard inductive
  setting; POSITIVES restricted to test events whose pair was never seen
  before the test span (symmetric both-sides-new evaluation). Metric =
  per-batch mean AUROC over kept positives vs all batch negatives.
- test_recurring: negatives drawn from ALL pairs active earlier in the
  test span (the never-in-train/val filter dropped); positives unchanged.

Everything else (batching, accumulation, padding, collision checks, 1:1
sampling before masking) identical to the standard protocol.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tgat.data import load_daily_graph
from tgat.neighbors import NeighborStore

from tgn.evaluate_dgb import (dgb_negatives, edgebank_scores_dgb,
                              per_batch_auroc)
from tgn.model import build_from_checkpoint
from tgn.streaming import score_pairs_streaming

MODELS = {
    "tgn_hard": "figures/tgn_hard_models",
    "pfpop_mrr": "figures/tgn_pfpop_mrr_models",
    "g2_twohop_w1": "tgn_global/figures/g2_twohop_w1",
    "w2_wide_ind": "tgn_global/figures/w2_wide_ind",
    "w2_fixed50_last": "tgn_global/figures/w2_fixed50_last",
}
MODES = (("inductive_sym", "inductive", True),
         ("test_recurring", "test_recurring", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load_daily_graph(args.parquet)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=20)
    test = slice(g.val_end, len(g.src))
    ts, td, tt = g.src[test], g.dst[test], g.day[test]
    n = len(ts)
    tv = set(zip(g.src[:g.val_end].tolist(), g.dst[:g.val_end].tolist()))
    unseen_pos = np.array([(int(a), int(b)) not in tv
                           for a, b in zip(ts, td)])

    res = {"unseen_pos_frac": float(unseen_pos.mean()),
           "modes": {m: {"edgebank": {}, "models": {}} for m, _, _ in MODES}}
    for mode, strategy, mask_pos in MODES:
        mask = unseen_pos if mask_pos else None
        out = res["modes"][mode]
        negs = {s: dgb_negatives(g, strategy, s, args.batch_size)
                for s in args.seeds}
        for label, kw in (("inf", {}), ("tw", {}),
                          ("inf_frozen", {"freeze_memory": True})):
            variant = "inf" if label.startswith("inf") else "tw"
            cell = out["edgebank"].setdefault(label,
                                              {"auroc": [], "pad_frac": []})
            for s in args.seeds:
                ns, nd, pad = negs[s]
                ps, nsc = edgebank_scores_dgb(g, variant, ns, nd,
                                              args.batch_size, **kw)
                cell["auroc"].append(per_batch_auroc(
                    ps, nsc, args.batch_size, pos_mask=mask)[0])
                cell["pad_frac"].append(float(pad.mean()))
        for name, mdir in MODELS.items():
            cell = out["models"].setdefault(name, {"auroc": []})
            for s in args.seeds:
                ns, nd, pad = negs[s]
                model = build_from_checkpoint(
                    Path(mdir) / f"tgn_seed{s}.pt", edge_feat_dim=store.F,
                    raw_feat_dim=g.edge_feat.shape[1], dim=100, device=device)
                qs = np.concatenate([ts, ns])
                qd = np.concatenate([td, nd])
                qt = np.concatenate([tt, tt])
                lg, _ = score_pairs_streaming(model, g, store, device,
                                              qs, qd, qt)
                cell["auroc"].append(per_batch_auroc(
                    lg[:n], lg[n:], args.batch_size, pos_mask=mask)[0])
            print(f"{mode} {name}: {np.mean(cell['auroc']):.4f}", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print("done")


if __name__ == "__main__":
    main()
