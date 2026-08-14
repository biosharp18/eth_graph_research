"""TGN evaluation CLI — identical protocol and JSON schema to tgat.evaluate.

NS pools, EdgeBank recompute, amount baselines and all metric conventions are
imported from tgat (never forked). Only the model scoring differs: TGN scores
stream day by day, memory advancing with observed positives (tgn.streaming).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tgat.data import load_daily_graph
from tgat.evaluate import (_pair_sets, amount_baselines, build_negative_pools,
                           edgebank_scores, mae, rmse)
from tgat.neighbors import NeighborStore
from tgat.train import auroc, average_precision

from .model import TGN
from .streaming import score_pairs_streaming


def tgn_scores(model, g, store, negs, device, batch=500):
    test = slice(g.val_end, len(g.src))
    s, d, t = g.src[test], g.dst[test], g.day[test]
    # one replay for positives and negatives: queries never advance memory,
    # so scoring them together is identical to scoring them separately
    qs = np.concatenate([s, negs[:, 0]])
    qd = np.concatenate([d, negs[:, 1]])
    qt = np.concatenate([t, t])
    lg, am = score_pairs_streaming(model, g, store, device, qs, qd, qt, batch)
    n = len(s)
    return lg[:n], lg[n:], am[:n]


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet",
                     default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap_.add_argument("--models", default="figures/tgn_models")
    ap_.add_argument("--out", default="figures/tgn_results.json")
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_.add_argument("--dim", type=int, default=100)
    ap_.add_argument("--k", type=int, default=20)
    args = ap_.parse_args()
    device = torch.device(args.device)

    g = load_daily_graph(args.parquet)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=args.k)
    _, before, _ = _pair_sets(g)
    test = slice(g.val_end, len(g.src))
    seen = np.array([(int(s), int(d)) in before
                     for s, d in zip(g.src[test], g.dst[test])])
    test_days = g.day[test]

    res = {"edgebank": {}, "tgn": {}, "amount": {},
           "split": {"train_end": g.train_end, "val_end": g.val_end,
                     "n_edges": len(g.src), "seen_frac": float(seen.mean()),
                     "train_days": [int(g.day[0]), int(g.day[g.train_end - 1])],
                     "val_days": [int(g.day[g.train_end]), int(g.day[g.val_end - 1])],
                     "test_days": [int(g.day[g.val_end]), int(g.day[-1])]}}

    for strategy in ("random", "historical", "inductive"):
        res["edgebank"].setdefault("inf", {})[strategy] = {"auroc": [], "ap": []}
        res["edgebank"].setdefault("tw", {})[strategy] = {"auroc": [], "ap": []}
        res["tgn"][strategy] = {"auroc": [], "ap": [], "auroc_seen": [],
                                "auroc_unseen": [], "per_day": []}
        for seed in args.seeds:
            negs = build_negative_pools(g, strategy, seed)
            y = np.r_[np.ones(negs.shape[0]), np.zeros(negs.shape[0])]
            for variant in ("inf", "tw"):
                ps, nsc = edgebank_scores(g, variant, negs)
                res["edgebank"][variant][strategy]["auroc"].append(
                    auroc(y, np.r_[ps, nsc]))
                res["edgebank"][variant][strategy]["ap"].append(
                    average_precision(y, np.r_[ps, nsc]))
            model = TGN(edge_feat_dim=store.F, raw_feat_dim=g.edge_feat.shape[1],
                        dim=args.dim).to(device)
            model.load_state_dict(torch.load(
                Path(args.models) / f"tgn_seed{seed}.pt", map_location=device))
            ps, nsc, pos_amt = tgn_scores(model, g, store, negs, device)
            sc = np.r_[ps, nsc]
            res["tgn"][strategy]["auroc"].append(auroc(y, sc))
            res["tgn"][strategy]["ap"].append(average_precision(y, sc))
            for name, m in (("auroc_seen", seen), ("auroc_unseen", ~seen)):
                mm = np.r_[m, m]
                res["tgn"][strategy][name].append(auroc(y[mm], sc[mm]))
            per_day = {int(dy): auroc(np.r_[np.ones((test_days == dy).sum()),
                                            np.zeros((test_days == dy).sum())],
                                      np.r_[ps[test_days == dy],
                                            nsc[test_days == dy]])
                       for dy in np.unique(test_days)}
            res["tgn"][strategy]["per_day"].append(per_day)
            if strategy == "random":
                pred_log = pos_amt * g.amt_std + g.amt_mean
                actual_log = g.y_amt[test] * g.amt_std + g.amt_mean
                pp, pm, act, seen_amt = amount_baselines(g)
                a = res["amount"]
                a.setdefault("tgn_rmse", []).append(rmse(pred_log, actual_log))
                a.setdefault("tgn_mae", []).append(mae(pred_log, actual_log))
                a.setdefault("tgn_rmse_seen", []).append(
                    rmse(pred_log[seen_amt], actual_log[seen_amt]))
                a.setdefault("tgn_rmse_unseen", []).append(
                    rmse(pred_log[~seen_amt], actual_log[~seen_amt]))
                a["persistence_rmse"] = rmse(pp, act)
                a["persistence_rmse_seen"] = rmse(pp[seen_amt], act[seen_amt])
                a["median_rmse"] = rmse(pm, act)
                a["persistence_mae"] = mae(pp, act)
                a["median_mae"] = mae(pm, act)
        print(f"{strategy}: tgn auroc mean "
              f"{np.mean(res['tgn'][strategy]['auroc']):.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "tgn"} |
                     {"tgn_auroc_means": {s: float(np.mean(v["auroc"]))
                                          for s, v in res["tgn"].items()}},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
