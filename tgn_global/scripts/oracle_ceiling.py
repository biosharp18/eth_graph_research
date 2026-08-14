"""Oracle-combination ceiling for the Phase-0 signal family.

Fits a logistic regression and a small MLP DIRECTLY ON the frozen inductive
eval rows (train == eval, deliberately) — an upper bound on what any model
could extract from this feature family, NOT a result. Reported per seed pool.
Also fits per-stratum oracles and a "day-conditional" oracle (features +
one-hot-free day interactions are not included; this bounds static combiners).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tgat.train import auroc


def fit_oracle(X, y, hidden=0, epochs=3000, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-9
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    if hidden:
        net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)).to(dev)
    else:
        net = torch.nn.Linear(X.shape[1], 1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        out = net(Xt).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc = net(Xt).squeeze(-1).cpu().numpy()
    return auroc(y, sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="tgn_global/diag_features.npz")
    ap.add_argument("--out", default="tgn_global/oracle_ceiling.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    pos = np.nan_to_num(np.clip(z["pos"], -50, 50))
    res = {"note": "oracle: fit and evaluated on the same eval rows — "
                   "upper bound of the feature family, not a result",
           "signals": z["signals"].tolist(), "overall": {}, "strata": {}}
    for seed in args.seeds:
        neg = np.nan_to_num(np.clip(z[f"neg_{seed}"], -50, 50))
        X = np.concatenate([pos, neg])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        res["overall"][seed] = {
            "logistic": fit_oracle(X, y, hidden=0),
            "mlp64": fit_oracle(X, y, hidden=64)}
        act = z[f"neg_active_{seed}"]
        seen = z["pos_seen"]
        strata = {"allpos_vs_activeneg": (np.ones(len(pos), bool), act),
                  "newpos_vs_newneg": (~seen, ~act)}
        for name, (pm, nm) in strata.items():
            Xs = np.concatenate([pos[pm], neg[nm]])
            ys = np.r_[np.ones(pm.sum()), np.zeros(nm.sum())]
            res["strata"].setdefault(name, {})[seed] = {
                "logistic": fit_oracle(Xs, ys, hidden=0),
                "mlp64": fit_oracle(Xs, ys, hidden=64)}
        print(f"seed {seed}: overall mlp64 "
              f"{res['overall'][seed]['mlp64']:.4f}", flush=True)

    for k in ("logistic", "mlp64"):
        res[f"overall_mean_{k}"] = float(np.mean(
            [res["overall"][s][k] for s in args.seeds]))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items()
                      if k.startswith("overall_mean")}, indent=2))


if __name__ == "__main__":
    main()
