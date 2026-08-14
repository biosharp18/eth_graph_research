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
from tgat.train import _pos_pairs_by_day, auroc, average_precision

from .model import TGN
from .streaming import day_ranges, score_pairs_streaming


class StreamingNegatives:
    """Streaming training-negative sampler over full (source, dest) pairs.

    With probability hard_frac a negative is "hard": a same-source strictly-
    past partner (probability src_frac within hard — the deployment-ranking
    hard case) or a pair from the global strictly-past pool (the historical-NS
    distribution). Otherwise, and whenever a hard pool is empty, a uniform
    random destination for the same source. Negatives never collide with the
    current day's positives; uniform draws also exclude s == c, matching
    tgat.train.sample_training_negatives.

    Pools must be advanced by the caller via observe_day AFTER a day's events
    are processed, so day-T sampling only ever sees days < T.
    """

    def __init__(self, n_nodes, pos_pairs_by_day, rng, hard_frac=0.0,
                 src_frac=0.5):
        self.n = n_nodes
        self.pos_by_day = pos_pairs_by_day
        self.rng = rng
        self.hard_frac = hard_frac
        self.src_frac = src_frac
        self.pair_set = set()
        self.pair_list = []      # distinct (s, d), first-seen order
        self.partners = {}       # s -> list of distinct past partners

    def observe_day(self, src, dst):
        for s, d in zip(np.asarray(src).tolist(), np.asarray(dst).tolist()):
            if (s, d) not in self.pair_set:
                self.pair_set.add((s, d))
                self.pair_list.append((s, d))
                self.partners.setdefault(s, []).append(d)

    def _uniform(self, s, pos):
        while True:
            c = int(self.rng.integers(0, self.n))
            if c != s and (s, c) not in pos:
                return s, c

    def sample(self, s_arr, d_arr, day, n_neg=1):
        pos = self.pos_by_day.get(int(day), set())
        B = len(s_arr)
        ns = np.empty((B, n_neg), np.int64)
        nd = np.empty((B, n_neg), np.int64)
        for i in range(B):
            s = int(s_arr[i])
            for k in range(n_neg):
                pair = None
                if self.rng.random() < self.hard_frac:
                    if self.rng.random() < self.src_frac:
                        cands = self.partners.get(s)
                        if cands:
                            for _ in range(10):  # reject today's positives
                                c = cands[int(self.rng.integers(0, len(cands)))]
                                if (s, c) not in pos:
                                    pair = (s, c)
                                    break
                    elif self.pair_list:
                        for _ in range(10):
                            p = self.pair_list[
                                int(self.rng.integers(0, len(self.pair_list)))]
                            if p not in pos:
                                pair = p
                                break
                if pair is None:
                    pair = self._uniform(s, pos)
                ns[i, k], nd[i, k] = pair
        return ns, nd


def sample_negatives_streaming(g: DailyGraph, rng, hard_frac, src_frac,
                               lo, hi, n_neg=1):
    """Fixed negatives for events in [lo, hi), pools strictly pre-day."""
    pos_by_day = _pos_pairs_by_day(g)
    sampler = StreamingNegatives(g.n_nodes, pos_by_day, rng,
                                 hard_frac, src_frac)
    off = day_ranges(g.day, g.n_days)
    ns = np.empty((hi - lo, n_neg), np.int64)
    nd = np.empty((hi - lo, n_neg), np.int64)
    for T in range(g.n_days):
        elo, ehi = int(off[T]), int(off[T + 1])
        qlo, qhi = max(elo, lo), min(ehi, hi)
        if qlo < qhi:
            a, b = sampler.sample(g.src[qlo:qhi], g.dst[qlo:qhi], T, n_neg)
            ns[qlo - lo:qhi - lo] = a
            nd[qlo - lo:qhi - lo] = b
        if ehi > elo:
            sampler.observe_day(g.src[elo:ehi], g.dst[elo:ehi])
        if ehi >= hi:
            break
    return ns, nd


def softmax_ce_loss(pos_logit, neg_logit):
    """Cross-entropy of each positive against its K sampled negatives."""
    logits = torch.cat([pos_logit.unsqueeze(1), neg_logit], dim=1)
    return (torch.logsumexp(logits, dim=1) - pos_logit).mean()


def train_one(g: DailyGraph, seed: int, device, epochs=50, patience=5,
              batch=200, lam=1.0, lr=1e-4, dim=100, k=20,
              train_metric_sample=3000, log_every=0,
              loss="bce", n_neg=1, hard_frac=0.0, src_frac=0.5):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=k)
    model = TGN(edge_feat_dim=store.F, raw_feat_dim=g.edge_feat.shape[1],
                dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_by_day = _pos_pairs_by_day(g)
    off = day_ranges(g.day, g.n_days)

    # fixed metric negatives drawn from the SAME distribution as training, so
    # early stopping selects for the intended objective (with hard_frac > 0,
    # val_ap is measured against hard negatives and is NOT comparable to
    # random-negative val_ap from other configs)
    rng_eval = np.random.default_rng(seed + 10_000)
    val_sl = slice(g.train_end, g.val_end)
    val_ns, val_nd = sample_negatives_streaming(
        g, rng_eval, hard_frac, src_frac, g.train_end, g.val_end, n_neg=1)

    n_tm = min(train_metric_sample, g.train_end)
    tm_idx = np.sort(rng_eval.choice(g.train_end, size=n_tm, replace=False))
    tm_s, tm_d, tm_t = g.src[tm_idx], g.dst[tm_idx], g.day[tm_idx]
    all_ns, all_nd = sample_negatives_streaming(
        g, rng_eval, hard_frac, src_frac, 0, g.train_end, n_neg=1)
    tm_ns, tm_nd = all_ns[tm_idx, 0], all_nd[tm_idx, 0]

    train_days = np.unique(g.day[:g.train_end])

    def epoch_metrics():
        """One streaming replay scoring the fixed train subset + validation."""
        qs = np.concatenate([tm_s, tm_ns, g.src[val_sl], val_ns[:, 0]])
        qd = np.concatenate([tm_d, tm_nd, g.dst[val_sl], val_nd[:, 0]])
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
        # negative pools rebuilt each epoch so day-T sampling is strictly-past
        sampler = StreamingNegatives(g.n_nodes, pos_by_day, rng,
                                     hard_frac, src_frac)
        bces, hubers, totals = [], [], []
        for T in train_days:
            # clamp to the split boundary: real data is day-snapped, but toy
            # graphs may have a day straddling train_end
            dlo, dhi = int(off[T]), min(int(off[T + 1]), g.train_end)
            for blo in range(dlo, dhi, batch):
                bhi = min(blo + batch, dhi)
                s, d = g.src[blo:bhi], g.dst[blo:bhi]
                B = bhi - blo
                ns, nds = sampler.sample(s, d, int(T), n_neg=n_neg)
                if pending is not None:
                    plo, phi, pday = pending
                    mem_eff, _ = model.apply_messages(
                        mem, last, g.src[plo:phi], g.dst[plo:phi], pday,
                        g.edge_feat[plo:phi])
                else:
                    mem_eff = mem
                opt.zero_grad()
                # embed each unique node once (all queries share day T)
                nodes = np.concatenate([s, d, ns.ravel(), nds.ravel()])
                uniq, inv = np.unique(nodes, return_inverse=True)
                Z = model.embed(uniq, np.full(len(uniq), int(T)), mem_eff,
                                store, device)[inv]
                zs, zd = Z[:B], Z[B:2 * B]
                zns = Z[2 * B:2 * B + B * n_neg]
                znd = Z[2 * B + B * n_neg:]
                logit_p = model.link_head(
                    torch.cat([zs, zd], dim=-1)).squeeze(-1)
                amt_p = model.amt_head(
                    torch.cat([zs, zd], dim=-1)).squeeze(-1)
                logit_n = model.link_head(
                    torch.cat([zns, znd], dim=-1)).squeeze(-1).view(B, n_neg)
                if loss == "ce":
                    link_loss = softmax_ce_loss(logit_p, logit_n)
                else:
                    link_loss = (F.binary_cross_entropy_with_logits(
                                     logit_p, torch.ones_like(logit_p))
                                 + F.binary_cross_entropy_with_logits(
                                     logit_n, torch.zeros_like(logit_n))) / 2
                y_amt = torch.from_numpy(g.y_amt[blo:bhi]).to(device)
                huber = lam * F.huber_loss(amt_p, y_amt, delta=1.0)
                (link_loss + huber).backward()
                opt.step()
                b, h = float(link_loss.detach()), float(huber.detach())
                bces.append(b); hubers.append(h); totals.append(b + h)
            with torch.no_grad():
                if pending is not None:
                    plo, phi, pday = pending
                    mem, last = model.apply_messages(
                        mem, last, g.src[plo:phi], g.dst[plo:phi], pday,
                        g.edge_feat[plo:phi])
            pending = (dlo, dhi, int(T))
            sampler.observe_day(g.src[dlo:dhi], g.dst[dlo:dhi])

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
    ap_.add_argument("--loss", choices=["bce", "ce"], default="bce")
    ap_.add_argument("--n-neg", type=int, default=1)
    ap_.add_argument("--hard-frac", type=float, default=0.0)
    ap_.add_argument("--src-frac", type=float, default=0.5)
    ap_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap_.parse_args()

    g = load_daily_graph(args.parquet)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = {}
    for seed in args.seeds:
        model, info = train_one(g, seed, torch.device(args.device),
                                epochs=args.epochs, k=args.k, batch=args.batch,
                                dim=args.dim, lr=args.lr, lam=args.lam,
                                patience=args.patience, log_every=1,
                                loss=args.loss, n_neg=args.n_neg,
                                hard_frac=args.hard_frac,
                                src_frac=args.src_frac)
        torch.save(model.state_dict(), out / f"tgn_seed{seed}.pt")
        log[seed] = info
        print(f"seed {seed}: best val AP {info['best_val_ap']:.4f} "
              f"({info['epochs_run']} epochs)", flush=True)
        # flush after every seed so an interrupted run keeps finished histories
        (out / "train_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
