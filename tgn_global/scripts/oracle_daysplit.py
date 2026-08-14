"""Honest (day-split) generalization value of the Phase-0 feature family.

Fit logistic / MLP on the inductive-pool rows of the FIRST half of test days,
evaluate on the SECOND half. Positives and negatives disjoint from training.
This is what a scorer trained on these features could plausibly reach on
late-test inductive discrimination. Also prints full-fit logistic weights.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tgat.train import auroc


def fit_eval(Xtr, ytr, Xev, yev, hidden=0, epochs=4000, lr=1e-2, seed=0,
             wd=1e-4):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-9
    tt = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32, device=dev)
    Xt, Xe = tt(Xtr), tt(Xev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    if hidden:
        net = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)).to(dev)
    else:
        net = torch.nn.Linear(Xtr.shape[1], 1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            net(Xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc = net(Xe).squeeze(-1).cpu().numpy()
    return auroc(yev, sc), net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="tgn_global/diag_features.npz")
    ap.add_argument("--out", default="tgn_global/oracle_daysplit.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--parquet",
                    default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    args = ap.parse_args()

    from tgat.data import load_daily_graph
    from tgat.evaluate import build_negative_pools
    g = load_daily_graph(args.parquet)
    tsl = slice(g.val_end, len(g.src))
    pos_pairs = np.stack([g.src[tsl], g.dst[tsl]], 1)
    neg_pairs = {s: build_negative_pools(g, "inductive", s)
                 for s in args.seeds}

    z = np.load(args.npz, allow_pickle=True)
    pos = np.nan_to_num(np.clip(z["pos"], -50, 50))
    days = z["test_day"]
    mid = np.median(np.unique(days))
    early = days <= mid
    signals = z["signals"].tolist()

    res = {"split_day": float(mid), "signals": signals,
           "n_train_events": int(early.sum()),
           "n_eval_events": int((~early).sum()), "per_seed": {}}
    for seed in args.seeds:
        neg = np.nan_to_num(np.clip(z[f"neg_{seed}"], -50, 50))
        Xtr = np.concatenate([pos[early], neg[early]])
        ytr = np.r_[np.ones(early.sum()), np.zeros(early.sum())]
        Xev = np.concatenate([pos[~early], neg[~early]])
        yev = np.r_[np.ones((~early).sum()), np.zeros((~early).sum())]
        lo, netl = fit_eval(Xtr, ytr, Xev, yev, hidden=0)
        ml, netm = fit_eval(Xtr, ytr, Xev, yev, hidden=64)

        # leakage control: drop late rows whose (s,d) pair occurred anywhere
        # in the training rows (as positive or negative) — a fingerprinting
        # memorizer gets no credit on what remains
        train_pair_set = set(map(tuple, pos_pairs[early].tolist())) \
            | set(map(tuple, neg_pairs[seed][early].tolist()))
        pos_novel = np.array([tuple(p) not in train_pair_set
                              for p in pos_pairs[~early].tolist()])
        neg_novel = np.array([tuple(p) not in train_pair_set
                              for p in neg_pairs[seed][~early].tolist()])
        keep = np.r_[pos_novel, neg_novel]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mu = Xtr.mean(0, keepdims=True)
        sd = Xtr.std(0, keepdims=True) + 1e-9
        with torch.no_grad():
            Xe = torch.tensor((Xev - mu) / sd, dtype=torch.float32,
                              device=dev)
            sc = netm(Xe).squeeze(-1).cpu().numpy()
        ml_novel = auroc(yev[keep], sc[keep])
        res["per_seed"][seed] = {
            "logistic": lo, "mlp64": ml, "mlp64_novel_pairs_only": ml_novel,
            "novel_frac_pos": float(pos_novel.mean()),
            "novel_frac_neg": float(neg_novel.mean())}
        if seed == args.seeds[0]:
            w = netl.weight.detach().cpu().numpy().ravel()
            res["logistic_weights_seed0"] = {
                s: round(float(v), 3) for s, v in zip(signals, w)}
        print(f"seed {seed}: day-split logistic {lo:.4f} mlp64 {ml:.4f}",
              flush=True)
    for k in ("logistic", "mlp64"):
        res[f"mean_{k}"] = float(np.mean(
            [res["per_seed"][s][k] for s in args.seeds]))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items()
                      if k.startswith("mean") or k.startswith("logistic_w")},
                     indent=2))


if __name__ == "__main__":
    main()
