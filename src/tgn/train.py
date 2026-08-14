"""Training loop + CLI for the two-headed TGN.

Memory during training uses the official TGN one-step-lag scheme at day
granularity: the persistent memory is detached and reflects days <= T-2 while
day T trains; day T-1's messages are re-applied differentiably inside each
batch's forward pass (so the GRU and time encoder get gradients), and the
persistent memory advances permanently (no_grad) once day T's batches finish.
Batches never cross day boundaries: every prediction for day T uses memory
reflecting days < T only, and same-day events never see each other.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tgat.data import DailyGraph, load_daily_graph
from tgat.neighbors import NeighborStore
from tgat.train import (_pos_pairs_by_day, auroc, average_precision,
                        sample_training_negatives)

from .model import TGN
from .streaming import day_ranges, score_pairs_streaming


def train_one(g: DailyGraph, seed: int, device, epochs=50, patience=5,
              batch=200, lam=1.0, lr=1e-4, dim=100, k=20,
              train_metric_sample=3000, log_every=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=k)
    model = TGN(edge_feat_dim=store.F, raw_feat_dim=g.edge_feat.shape[1],
                dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_by_day = _pos_pairs_by_day(g)
    off = day_ranges(g.day, g.n_days)

    val_sl = slice(g.train_end, g.val_end)
    val_neg = sample_training_negatives(
        rng, g.src[val_sl], g.dst[val_sl], g.day[val_sl], g.n_nodes, pos_by_day)

    # fixed train-metric subset with fixed negatives (same convention as TGAT)
    rng_eval = np.random.default_rng(seed + 10_000)
    n_tm = min(train_metric_sample, g.train_end)
    tm_idx = np.sort(rng_eval.choice(g.train_end, size=n_tm, replace=False))
    tm_s, tm_d, tm_t = g.src[tm_idx], g.dst[tm_idx], g.day[tm_idx]
    tm_neg = sample_training_negatives(rng_eval, tm_s, tm_d, tm_t,
                                       g.n_nodes, pos_by_day)

    train_days = np.unique(g.day[:g.train_end])

    def epoch_metrics():
        """One streaming replay scoring the fixed train subset + validation."""
        qs = np.concatenate([tm_s, tm_s, g.src[val_sl], g.src[val_sl]])
        qd = np.concatenate([tm_d, tm_neg, g.dst[val_sl], val_neg])
        qt = np.concatenate([tm_t, tm_t, g.day[val_sl], g.day[val_sl]])
        lg, _ = score_pairs_streaming(model, g, store, device, qs, qd, qt)
        n, v = n_tm, g.val_end - g.train_end
        tr_y = np.r_[np.ones(n), np.zeros(n)]
        va_y = np.r_[np.ones(v), np.zeros(v)]
        tr_sc, va_sc = lg[:2 * n], lg[2 * n:]
        return (average_precision(tr_y, tr_sc), auroc(tr_y, tr_sc),
                average_precision(va_y, va_sc), auroc(va_y, va_sc))

    best_ap, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        mem, last = model.init_memory(g.n_nodes, device)
        pending = None  # (lo, hi, day) of the most recently completed day
        bces, hubers, totals = [], [], []
        for T in train_days:
            # clamp to the split boundary: real data is day-snapped, but toy
            # graphs may have a day straddling train_end
            dlo, dhi = int(off[T]), min(int(off[T + 1]), g.train_end)
            for blo in range(dlo, dhi, batch):
                bhi = min(blo + batch, dhi)
                s, d, t = g.src[blo:bhi], g.dst[blo:bhi], g.day[blo:bhi]
                nd = sample_training_negatives(rng, s, d, t, g.n_nodes,
                                               pos_by_day)
                if pending is not None:
                    plo, phi, pday = pending
                    mem_eff, _ = model.apply_messages(
                        mem, last, g.src[plo:phi], g.dst[plo:phi], pday,
                        g.edge_feat[plo:phi])
                else:
                    mem_eff = mem
                opt.zero_grad()
                logit_p, amt_p = model(s, d, t, mem_eff, store, device)
                logit_n, _ = model(s, nd, t, mem_eff, store, device)
                y_amt = torch.from_numpy(g.y_amt[blo:bhi]).to(device)
                bce = (F.binary_cross_entropy_with_logits(
                           logit_p, torch.ones_like(logit_p))
                       + F.binary_cross_entropy_with_logits(
                           logit_n, torch.zeros_like(logit_n))) / 2
                huber = lam * F.huber_loss(amt_p, y_amt, delta=1.0)
                (bce + huber).backward()
                opt.step()
                b, h = float(bce.detach()), float(huber.detach())
                bces.append(b); hubers.append(h); totals.append(b + h)
            with torch.no_grad():
                if pending is not None:
                    plo, phi, pday = pending
                    mem, last = model.apply_messages(
                        mem, last, g.src[plo:phi], g.dst[plo:phi], pday,
                        g.edge_feat[plo:phi])
            pending = (dlo, dhi, int(T))

        train_ap, train_auroc, val_ap, val_auroc = epoch_metrics()
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "train_bce": float(np.mean(bces)),
            "train_huber": float(np.mean(hubers)),
            "train_ap": train_ap, "train_auroc": train_auroc,
            "val_ap": val_ap, "val_auroc": val_auroc,
        })
        if log_every and epoch % log_every == 0:
            print(f"[seed {seed}] epoch {epoch} loss {np.mean(totals):.4f} "
                  f"val_ap {val_ap:.4f}", flush=True)
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
    ap_.add_argument("--out", default="figures/tgn_models")
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
                                patience=args.patience, log_every=1)
        torch.save(model.state_dict(), out / f"tgn_seed{seed}.pt")
        log[seed] = info
        print(f"seed {seed}: best val AP {info['best_val_ap']:.4f} "
              f"({info['epochs_run']} epochs)", flush=True)
        # flush after every seed so an interrupted run keeps finished histories
        (out / "train_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
