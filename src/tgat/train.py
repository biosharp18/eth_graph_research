"""Training loop + CLI for the two-headed TGAT."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import DailyGraph, load_daily_graph
from .model import TGAT
from .neighbors import NeighborStore


def sample_training_negatives(rng, src, dst, day, n_nodes, pos_pairs_by_day):
    neg = np.empty(len(src), np.int64)
    for i, (s, t) in enumerate(zip(src, day)):
        pos = pos_pairs_by_day[int(t)]
        while True:
            c = int(rng.integers(0, n_nodes))
            if c != int(s) and (int(s), c) not in pos:
                neg[i] = c
                break
    return neg


def average_precision(y, score):
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="stable")
    y_s, s_s = y[order], score[order]
    tp = np.cumsum(y_s)
    n = np.arange(1, len(y_s) + 1)
    last = np.r_[np.nonzero(np.diff(s_s))[0], len(s_s) - 1]  # end of each tied block
    prec = tp[last] / n[last]
    rec = tp[last] / tp[-1]
    drec = np.diff(np.r_[0.0, rec])
    return float(np.sum(prec * drec))


def auroc(y, score):
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    ranks = pd.Series(score).rank(method="average").to_numpy()
    npos = int(y.sum())
    nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _pos_pairs_by_day(g: DailyGraph):
    out = {}
    for s, d, t in zip(g.src, g.dst, g.day):
        out.setdefault(int(t), set()).add((int(s), int(d)))
    return out


def _score_pairs(model, store, device, s, d, t, batch=500):
    """Batched eval-mode link logits (avoids OOM at large k)."""
    outs = []
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(s), batch):
            hi = lo + batch
            logit, _ = model(s[lo:hi], d[lo:hi], t[lo:hi], store, device)
            outs.append(logit.cpu().numpy())
    return np.concatenate(outs)


def _link_metrics(model, store, device, s, d, neg_d, t):
    lp = _score_pairs(model, store, device, s, d, t)
    ln = _score_pairs(model, store, device, s, neg_d, t)
    y = np.r_[np.ones(len(lp)), np.zeros(len(ln))]
    score = np.r_[lp, ln]
    return average_precision(y, score), auroc(y, score)


def train_one(g: DailyGraph, seed: int, device, epochs=50, patience=5,
              batch=200, lam=1.0, lr=1e-4, dim=100, k=20,
              train_metric_sample=3000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=k)
    model = TGAT(edge_feat_dim=store.F, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_by_day = _pos_pairs_by_day(g)

    val_sl = slice(g.train_end, g.val_end)
    val_neg = sample_training_negatives(
        rng, g.src[val_sl], g.dst[val_sl], g.day[val_sl], g.n_nodes, pos_by_day)

    # fixed train-metric subset with fixed negatives, so the per-epoch train
    # curve measures the same thing every epoch
    rng_eval = np.random.default_rng(seed + 10_000)
    n_tm = min(train_metric_sample, g.train_end)
    tm_idx = np.sort(rng_eval.choice(g.train_end, size=n_tm, replace=False))
    tm_s, tm_d, tm_t = g.src[tm_idx], g.dst[tm_idx], g.day[tm_idx]
    tm_neg = sample_training_negatives(rng_eval, tm_s, tm_d, tm_t,
                                       g.n_nodes, pos_by_day)

    best_ap, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        bces, hubers, totals = [], [], []
        for lo in range(0, g.train_end, batch):
            hi = min(lo + batch, g.train_end)
            s, d, t = g.src[lo:hi], g.dst[lo:hi], g.day[lo:hi]
            nd = sample_training_negatives(rng, s, d, t, g.n_nodes, pos_by_day)
            opt.zero_grad()
            # split backward: free the positive-side graph before the negative
            # forward pass to halve peak activation memory at large k
            logit_p, amt_p = model(s, d, t, store, device)
            y_amt = torch.from_numpy(g.y_amt[lo:hi]).to(device)
            bce_p = F.binary_cross_entropy_with_logits(
                logit_p, torch.ones_like(logit_p)) / 2
            huber = lam * F.huber_loss(amt_p, y_amt, delta=1.0)
            (bce_p + huber).backward()
            logit_n, _ = model(s, nd, t, store, device)
            bce_n = F.binary_cross_entropy_with_logits(
                logit_n, torch.zeros_like(logit_n)) / 2
            bce_n.backward()
            opt.step()
            b = float(bce_p.detach()) + float(bce_n.detach())
            h = float(huber.detach())
            bces.append(b); hubers.append(h); totals.append(b + h)

        train_ap, train_auroc = _link_metrics(
            model, store, device, tm_s, tm_d, tm_neg, tm_t)
        val_ap, val_auroc = _link_metrics(
            model, store, device, g.src[val_sl], g.dst[val_sl], val_neg,
            g.day[val_sl])
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "train_bce": float(np.mean(bces)),
            "train_huber": float(np.mean(hubers)),
            "train_ap": train_ap, "train_auroc": train_auroc,
            "val_ap": val_ap, "val_auroc": val_auroc,
        })
        if val_ap > best_ap:
            best_ap, best_state, bad = val_ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"best_val_ap": best_ap, "epochs_run": len(history),
                   "history": history}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet",
                     default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap_.add_argument("--out", default="figures/tgat_models")
    ap_.add_argument("--epochs", type=int, default=50)
    ap_.add_argument("--k", type=int, default=20)
    ap_.add_argument("--batch", type=int, default=200)
    ap_.add_argument("--dim", type=int, default=100)
    ap_.add_argument("--lr", type=float, default=1e-4)
    ap_.add_argument("--lam", type=float, default=1.0)
    ap_.add_argument("--patience", type=int, default=5)
    ap_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap_.parse_args()

    g = load_daily_graph(args.parquet)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = {}
    for seed in args.seeds:
        model, info = train_one(g, seed, torch.device(args.device),
                                epochs=args.epochs, k=args.k, batch=args.batch,
                                dim=args.dim, lr=args.lr, lam=args.lam,
                                patience=args.patience)
        torch.save(model.state_dict(), out / f"tgat_seed{seed}.pt")
        log[seed] = info
        print(f"seed {seed}: best val AP {info['best_val_ap']:.4f} "
              f"({info['epochs_run']} epochs)", flush=True)
        # flush after every seed so an interrupted run keeps finished histories
        (out / "train_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
