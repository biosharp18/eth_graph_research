"""Evaluation: NS pools, streaming EdgeBank on daily edges, metrics, JSON."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import DailyGraph, load_daily_graph
from .model import TGAT
from .neighbors import NeighborStore
from .train import average_precision


def auroc(y, score):
    y = np.asarray(y); score = np.asarray(score, dtype=np.float64)
    ranks = pd.Series(score).rank(method="average").to_numpy()
    npos = int(y.sum()); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _pair_sets(g: DailyGraph):
    all_pairs = set(zip(g.src.tolist(), g.dst.tolist()))
    before = set(zip(g.src[:g.val_end].tolist(), g.dst[:g.val_end].tolist()))
    test_pairs = set(zip(g.src[g.val_end:].tolist(), g.dst[g.val_end:].tolist()))
    return all_pairs, before, test_pairs - before


def _pos_by_day(g):
    out = {}
    for s, d, t in zip(g.src, g.dst, g.day):
        out.setdefault(int(t), set()).add((int(s), int(d)))
    return out


def build_negative_pools(g: DailyGraph, strategy: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_pairs, before, induct = _pair_sets(g)
    pos_by_day = _pos_by_day(g)
    negs = []
    test_days = g.day[g.val_end:]
    for day in np.unique(test_days):
        idx = np.where(test_days == day)[0]
        npos = len(idx)
        e_t = pos_by_day[int(day)]
        chosen = []
        if strategy in ("historical", "inductive"):
            pool = list((before if strategy == "historical" else induct) - e_t)
            # sample with replacement: the qualifying pool (pairs seen
            # before test / only-in-test pairs, minus today's positives) is
            # often much smaller than the day's positive count, so capping
            # at len(pool) without replacement would push most negatives
            # into the random-top-up branch below. Only fall through to
            # random top-up when the pool is genuinely empty.
            if pool:
                for i in rng.choice(len(pool), size=npos, replace=True):
                    chosen.append(pool[i])
        while len(chosen) < npos:  # random NS and top-up
            s = int(rng.integers(0, g.n_nodes)); d = int(rng.integers(0, g.n_nodes))
            if s != d and (s, d) not in all_pairs:
                chosen.append((s, d))
        negs.extend(chosen)
    return np.array(negs, dtype=np.int64)  # [n_test, 2]


def edgebank_scores(g: DailyGraph, variant: str, negs: np.ndarray):
    test_days = g.day[g.val_end:]
    w = int(test_days.max() - test_days.min()) + 1
    last = {}
    for s, d, t in zip(g.src[:g.val_end], g.dst[:g.val_end], g.day[:g.val_end]):
        last[(int(s), int(d))] = int(t)
    pos_sc = np.zeros(len(test_days)); neg_sc = np.zeros(len(test_days))

    def hit(pair, day):
        if pair not in last:
            return 0.0
        return 1.0 if variant == "inf" or last[pair] > day - w else 0.0

    i = 0
    for day in np.unique(test_days):
        idx = np.where(test_days == day)[0] + g.val_end
        for j in idx:
            pair = (int(g.src[j]), int(g.dst[j]))
            pos_sc[i] = hit(pair, int(day))
            neg_sc[i] = hit((int(negs[i][0]), int(negs[i][1])), int(day))
            i += 1
        for j in idx:  # update after scoring the day
            last[(int(g.src[j]), int(g.dst[j]))] = int(day)
    return pos_sc, neg_sc


def tgat_scores(model, g, store, negs, device, batch=500):
    test = slice(g.val_end, len(g.src))
    s, d, t = g.src[test], g.dst[test], g.day[test]
    ns, nd = negs[:, 0], negs[:, 1]
    outs = []
    model.eval()
    with torch.no_grad():
        for arr_s, arr_d in ((s, d), (ns, nd)):
            sc, am = [], []
            for lo in range(0, len(arr_s), batch):
                logit, amt = model(arr_s[lo:lo + batch], arr_d[lo:lo + batch],
                                   t[lo:lo + batch], store, device)
                sc.append(logit.cpu().numpy()); am.append(amt.cpu().numpy())
            outs.append((np.concatenate(sc), np.concatenate(am)))
    (pos_sc, pos_amt), (neg_sc, _) = outs
    return pos_sc, neg_sc, pos_amt


def amount_baselines(g: DailyGraph):
    """persistence + global-median RMSE/MAE in log10-USD on test positives."""
    log_usd = np.log10(g.usd_sum)
    med = float(np.median(log_usd[:g.train_end]))
    last_amt = {}
    for s, d, la in zip(g.src[:g.val_end], g.dst[:g.val_end], log_usd[:g.val_end]):
        last_amt[(int(s), int(d))] = la
    preds_p, preds_m, actual, seen_mask = [], [], [], []
    for j in range(g.val_end, len(g.src)):
        pair = (int(g.src[j]), int(g.dst[j]))
        seen_mask.append(pair in last_amt)
        preds_p.append(last_amt.get(pair, med))
        preds_m.append(med)
        actual.append(log_usd[j])
        last_amt[pair] = log_usd[j]  # persistence streams forward
    return (np.array(preds_p), np.array(preds_m), np.array(actual),
            np.array(seen_mask))


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b): return float(np.mean(np.abs(a - b)))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet",
                     default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap_.add_argument("--models", default="figures/tgat_models")
    ap_.add_argument("--out", default="figures/tgat_results.json")
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

    res = {"edgebank": {}, "tgat": {}, "amount": {},
           "split": {"train_end": g.train_end, "val_end": g.val_end,
                     "n_edges": len(g.src), "seen_frac": float(seen.mean())}}

    for strategy in ("random", "historical", "inductive"):
        res["edgebank"].setdefault("inf", {})[strategy] = {"auroc": [], "ap": []}
        res["edgebank"].setdefault("tw", {})[strategy] = {"auroc": [], "ap": []}
        res["tgat"][strategy] = {"auroc": [], "ap": [], "auroc_seen": [],
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
            model = TGAT(edge_feat_dim=store.F, dim=args.dim).to(device)
            model.load_state_dict(torch.load(
                Path(args.models) / f"tgat_seed{seed}.pt", map_location=device))
            ps, nsc, pos_amt = tgat_scores(model, g, store, negs, device)
            sc = np.r_[ps, nsc]
            res["tgat"][strategy]["auroc"].append(auroc(y, sc))
            res["tgat"][strategy]["ap"].append(average_precision(y, sc))
            for name, m in (("auroc_seen", seen), ("auroc_unseen", ~seen)):
                mm = np.r_[m, m]
                res["tgat"][strategy][name].append(auroc(y[mm], sc[mm]))
            per_day = {int(dy): auroc(np.r_[np.ones((test_days == dy).sum()),
                                            np.zeros((test_days == dy).sum())],
                                      np.r_[ps[test_days == dy],
                                            nsc[test_days == dy]])
                       for dy in np.unique(test_days)}
            res["tgat"][strategy]["per_day"].append(per_day)
            if strategy == "random":
                actual_z = g.y_amt[test]
                pred_log = pos_amt * g.amt_std + g.amt_mean
                actual_log = actual_z * g.amt_std + g.amt_mean
                pp, pm, act, seen_amt = amount_baselines(g)
                a = res["amount"]
                a.setdefault("tgat_rmse", []).append(rmse(pred_log, actual_log))
                a.setdefault("tgat_mae", []).append(mae(pred_log, actual_log))
                a.setdefault("tgat_rmse_seen", []).append(
                    rmse(pred_log[seen_amt], actual_log[seen_amt]))
                a.setdefault("tgat_rmse_unseen", []).append(
                    rmse(pred_log[~seen_amt], actual_log[~seen_amt]))
                a["persistence_rmse"] = rmse(pp, act)
                a["persistence_rmse_seen"] = rmse(pp[seen_amt], act[seen_amt])
                a["median_rmse"] = rmse(pm, act)
                a["persistence_mae"] = mae(pp, act)
                a["median_mae"] = mae(pm, act)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "tgat"} |
                     {"tgat_auroc_means": {s: float(np.mean(v["auroc"]))
                                           for s, v in res["tgat"].items()}},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
