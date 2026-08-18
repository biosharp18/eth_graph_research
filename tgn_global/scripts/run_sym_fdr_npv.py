"""FDR / NPV / AUROC on the inductive_sym set, with honest thresholding.

For each model and seed: build the SAME inductive_sym construction on the
VALIDATION span (positives = val events whose pair was never seen in
train; negatives = val-span inductive sampling), pick the single threshold
maximizing Youden's J there, freeze it, then compute FDR = FP/(FP+TP) and
NPV = TN/(TN+FN) on the test-span inductive_sym pools at that threshold.
AUROC is reported for the same test pools. EdgeBank has one non-trivial
threshold (flag = seen); its FDR/NPV are computed at that rule directly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tgat.data import load_daily_graph
from tgat.neighbors import NeighborStore
from tgat.model import TGAT
from tgat.train import auroc

from tgn.evaluate_dgb import dgb_negatives, edgebank_scores_dgb, per_batch_auroc
from tgn.model import build_from_checkpoint
from tgn.streaming import score_pairs_streaming

MODELS = {
    "tgn_hard": "figures/tgn_hard_models",
    "pfpop_mrr": "figures/tgn_pfpop_mrr_models",
    "g2_twohop_w1": "tgn_global/figures/g2_twohop_w1",
    "w2_wide_ind": "tgn_global/figures/w2_wide_ind",
    "w2_fixed50_last": "tgn_global/figures/w2_fixed50_last",
}


def youden_threshold(pos_sc, neg_sc):
    sc = np.concatenate([pos_sc, neg_sc])
    y = np.r_[np.ones(len(pos_sc)), np.zeros(len(neg_sc))]
    order = np.argsort(-sc)
    y = y[order]
    tpr = np.cumsum(y) / max(y.sum(), 1)
    fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(sc[order][k])  # flag score >= t

def fdr_npv(pos_sc, neg_sc, t):
    tp = int((pos_sc >= t).sum()); fn = len(pos_sc) - tp
    fp = int((neg_sc >= t).sum()); tn = len(neg_sc) - fp
    fdr = fp / max(fp + tp, 1)
    npv = tn / max(tn + fn, 1)
    return fdr, npv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load_daily_graph(args.parquet)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=20)
    spans = {"val": (g.train_end, g.val_end), "test": (g.val_end, len(g.src))}
    ev, masks = {}, {}
    for name, (lo, hi) in spans.items():
        s, d, t = g.src[lo:hi], g.dst[lo:hi], g.day[lo:hi]
        prior = set(zip(g.src[:lo].tolist(), g.dst[:lo].tolist()))
        masks[name] = np.array([(int(a), int(b)) not in prior
                                for a, b in zip(s, d)])
        ev[name] = (s, d, t)

    def scores(model_kind, path, seed, span):
        (s, d, t) = ev[span]
        ns, nd, _ = dgb_negatives(g, "inductive", seed, 200, span=spans[span])
        qs, qd, qt = (np.concatenate([s, ns]), np.concatenate([d, nd]),
                      np.concatenate([t, t]))
        if model_kind == "tgat":
            model = TGAT(edge_feat_dim=store.F, dim=100).to(device)
            model.load_state_dict(torch.load(
                Path(path) / f"tgat_seed{seed}.pt", map_location=device))
            model.eval()
            out = []
            with torch.no_grad():
                for lo2 in range(0, len(qs), 500):
                    lg, _ = model(qs[lo2:lo2+500], qd[lo2:lo2+500],
                                  qt[lo2:lo2+500], store, device)
                    out.append(lg.cpu().numpy())
            lg = np.concatenate(out)
        else:
            model = build_from_checkpoint(
                Path(path) / f"tgn_seed{seed}.pt", edge_feat_dim=store.F,
                raw_feat_dim=g.edge_feat.shape[1], dim=100, device=device)
            lg, _ = score_pairs_streaming(model, g, store, device, qs, qd, qt)
        n = len(s)
        m = masks[span]
        return lg[:n][m], lg[n:]  # masked positives, all negatives

    res = {"prevalence_test": None, "models": {}}
    all_models = [("tgat", "tgat", "figures/tgat_models")] + \
                 [("tgn", k, v) for k, v in MODELS.items()]
    for kind, name, path in all_models:
        cell = {"auroc": [], "fdr": [], "npv": [], "threshold": []}
        for seed in args.seeds:
            vp, vn = scores(kind, path, seed, "val")
            t_star = youden_threshold(vp, vn)
            tp_, tn_ = scores(kind, path, seed, "test")
            y = np.r_[np.ones(len(tp_)), np.zeros(len(tn_))]
            cell["auroc"].append(auroc(y, np.r_[tp_, tn_]))
            f, v = fdr_npv(tp_, tn_, t_star)
            cell["fdr"].append(f)
            cell["npv"].append(v)
            cell["threshold"].append(t_star)
            if res["prevalence_test"] is None:
                res["prevalence_test"] = len(tp_) / (len(tp_) + len(tn_))
        res["models"][name] = cell
        print(f"{name}: auroc {np.mean(cell['auroc']):.4f} "
              f"fdr {np.mean(cell['fdr']):.4f} npv {np.mean(cell['npv']):.4f}",
              flush=True)

    # EdgeBank: one non-trivial threshold (flag = pair seen)
    for label, kw in (("edgebank_inf", {}),
                      ("edgebank_inf_frozen", {"freeze_memory": True})):
        cell = {"auroc": [], "fdr": [], "npv": []}
        for seed in args.seeds:
            ns, nd, _ = dgb_negatives(g, "inductive", seed, 200)
            ps, nsc = edgebank_scores_dgb(g, "inf", ns, nd, 200, **kw)
            m = masks["test"]
            pos, neg = ps[m], nsc
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            cell["auroc"].append(auroc(y, np.r_[pos, neg]))
            f, v = fdr_npv(pos, neg, 0.5)
            cell["fdr"].append(f)
            cell["npv"].append(v)
        res["models"][label] = cell
        print(f"{label}: auroc {np.mean(cell['auroc']):.4f} "
              f"fdr {np.mean(cell['fdr']):.4f} npv {np.mean(cell['npv']):.4f}",
              flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print("done")


if __name__ == "__main__":
    main()
