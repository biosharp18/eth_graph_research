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
import bisect
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
from .recency import (FEAT_DIM, FEAT_DIM_BUCKETS, FEAT_DIM_GLOBAL,
                      FEAT_DIM_GLOBAL_BUCKETS, FEAT_DIM_WIDE, PairRecency)
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
                 src_frac=0.5, pop_frac=0.0, pop_window=0, nov_frac=0.0,
                 nov_window=30):
        self.n = n_nodes
        self.pos_by_day = pos_pairs_by_day
        self.rng = rng
        self.hard_frac = hard_frac
        self.src_frac = src_frac
        self.pop_frac = pop_frac
        self.pop_window = pop_window  # 0 = all history; else last N dst tokens
        self.nov_frac = nov_frac      # novelty negatives: recently-FIRST-seen
        self.nov_window = nov_window  # pairs (mimics the inductive NS pool)
        self.pair_set = set()
        self.pair_list = []      # distinct (s, d), first-seen order
        self.first_seen = []     # first-seen day, parallel to pair_list
        self.partners = {}       # s -> list of distinct past partners
        self.dst_tokens = []     # every past dst with repetition (freq-weighted)

    def observe_day(self, src, dst, day=0):
        for s, d in zip(np.asarray(src).tolist(), np.asarray(dst).tolist()):
            if (s, d) not in self.pair_set:
                self.pair_set.add((s, d))
                self.pair_list.append((s, d))
                self.first_seen.append(int(day))
                self.partners.setdefault(s, []).append(d)
            self.dst_tokens.append(d)

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
                u = self.rng.random()
                hp = self.hard_frac + self.pop_frac
                if hp <= u < hp + self.nov_frac and self.first_seen:
                    # novelty negative: a pair whose FIRST appearance is
                    # within nov_window days — the streaming composition of
                    # the inductive NS pool (pair_list is first-seen ordered,
                    # so the eligible pairs are a contiguous tail)
                    lo_i = bisect.bisect_left(self.first_seen,
                                              day - self.nov_window)
                    if lo_i < len(self.pair_list):
                        for _ in range(10):
                            p = self.pair_list[int(self.rng.integers(
                                lo_i, len(self.pair_list)))]
                            if p not in pos:
                                pair = p
                                break
                elif self.hard_frac <= u < self.hard_frac + self.pop_frac \
                        and self.dst_tokens:
                    # popularity negative: dst ~ past-destination frequency —
                    # pushes down globally active hubs the source never paid
                    lo_tok = (max(0, len(self.dst_tokens) - self.pop_window)
                              if self.pop_window else 0)
                    for _ in range(10):
                        c = self.dst_tokens[
                            int(self.rng.integers(lo_tok,
                                                  len(self.dst_tokens)))]
                        if c != s and (s, c) not in pos:
                            pair = (s, c)
                            break
                elif u < self.hard_frac:
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
                               lo, hi, n_neg=1, pop_frac=0.0, nov_frac=0.0,
                               nov_window=30):
    """Fixed negatives for events in [lo, hi), pools strictly pre-day."""
    pos_by_day = _pos_pairs_by_day(g)
    sampler = StreamingNegatives(g.n_nodes, pos_by_day, rng,
                                 hard_frac, src_frac, pop_frac,
                                 nov_frac=nov_frac, nov_window=nov_window)
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
            sampler.observe_day(g.src[elo:ehi], g.dst[elo:ehi], T)
        if ehi >= hi:
            break
    return ns, nd


def softmax_ce_loss(pos_logit, neg_logit):
    """Cross-entropy of each positive against its K sampled negatives."""
    logits = torch.cat([pos_logit.unsqueeze(1), neg_logit], dim=1)
    return (torch.logsumexp(logits, dim=1) - pos_logit).mean()


def full_softmax_mask(s_arr, d_arr, pos_today, n_nodes: int) -> np.ndarray:
    """(B, n_nodes) bool: candidates that must NOT compete with query i —
    the source itself and the source's OTHER positives of the same day
    (the sampler's today-positive rejection, applied to the full vocabulary).
    The true destination d_arr[i] is never masked."""
    B = len(s_arr)
    m = np.zeros((B, n_nodes), bool)
    m[np.arange(B), np.asarray(s_arr)] = True
    by_src = {}
    for s, d in pos_today:
        by_src.setdefault(s, []).append(d)
    for i, (s, d) in enumerate(zip(np.asarray(s_arr).tolist(),
                                   np.asarray(d_arr).tolist())):
        for c in by_src.get(s, ()):
            if c != d:
                m[i, c] = True
    return m


def full_softmax_ce_loss(logits, d_idx, invalid):
    """Cross-entropy of each positive against EVERY candidate destination:
    logits (B, n_nodes), d_idx (B,) true destinations, invalid (B, n_nodes)
    bool mask of candidates excluded from the partition function."""
    pos = logits.gather(1, d_idx.unsqueeze(1)).squeeze(1)
    masked = logits.masked_fill(invalid, float("-inf"))
    return (torch.logsumexp(masked, dim=1) - pos).mean()


def train_one(g: DailyGraph, seed: int, device, epochs=50, patience=5,
              batch=200, lam=1.0, lr=1e-4, dim=100, k=20,
              train_metric_sample=3000, log_every=0,
              loss="bce", n_neg=1, hard_frac=0.0, src_frac=0.5,
              n_neg_hard=0, beta_hard=1.0, pair_feat=False,
              hard_hinge=0.0, select="ap", val_mrr_events=500,
              val_mrr_cands=100, in_batch=False, pop_frac=0.0,
              pair_feat_dim=FEAT_DIM, pop_window=0, head="mlp",
              nov_frac=0.0, nov_window=30, n_layers=1, nbr_mode="recent",
              keep_last=False, n_stack=0, k_inner=0, save_every=0,
              save_dir=None):
    """n_neg_hard > 0 enables the two-term ranking loss: CE against n_neg
    uniform negatives plus beta_hard * CE against n_neg_hard all-hard
    negatives (src_frac splits hard between same-source partners and global
    historical pairs). Keeping the two softmaxes separate stops the hard
    negatives from drowning out the global-calibration gradient (measured:
    single mixed softmax trades MRR for historical AU-ROC).

    hard_hinge > 0 replaces the hard CE term with a margin hinge
    mean(relu(m - (pos - hard_neg))): once the positive clears every hard
    negative by m the gradient stops, so hard negatives that are recent
    partners are not pushed below the unseen-candidate mass (the measured
    failure mode of the mixed softmax at full-ranking).

    select="mrr" early-stops on a sampled validation MRR (val_mrr_events
    events, positive ranked among val_mrr_cands uniform candidates) instead
    of matched-negative val AP — the deployment-shaped selection metric.

    in_batch=True (loss="ce" only) extends each positive's softmax with the
    other same-day-batch destinations as negatives: "active today but not
    yours", the candidate type measured to outrank true destinations at
    full ranking. Same-destination and same-day-positive pairs are masked."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=k,
                          mode=nbr_mode)
    model = TGN(edge_feat_dim=store.F, raw_feat_dim=g.edge_feat.shape[1],
                dim=dim, head=head, n_layers=n_layers, n_stack=n_stack,
                k_inner=k_inner,
                pair_feat_dim=pair_feat_dim if pair_feat else 0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_by_day = _pos_pairs_by_day(g)
    off = day_ranges(g.day, g.n_days)

    # fixed metric negatives drawn from the SAME distribution as training, so
    # early stopping selects for the intended objective (with hard_frac > 0,
    # val_ap is measured against hard negatives and is NOT comparable to
    # random-negative val_ap from other configs)
    rng_eval = np.random.default_rng(seed + 10_000)
    val_sl = slice(g.train_end, g.val_end)
    # two-term mode balances val negatives 50/50 random/hard
    val_hard_frac = 0.5 if n_neg_hard > 0 else hard_frac
    val_ns, val_nd = sample_negatives_streaming(
        g, rng_eval, val_hard_frac, src_frac, g.train_end, g.val_end, n_neg=1,
        pop_frac=pop_frac, nov_frac=nov_frac, nov_window=nov_window)

    n_tm = min(train_metric_sample, g.train_end)
    tm_idx = np.sort(rng_eval.choice(g.train_end, size=n_tm, replace=False))
    tm_s, tm_d, tm_t = g.src[tm_idx], g.dst[tm_idx], g.day[tm_idx]
    all_ns, all_nd = sample_negatives_streaming(
        g, rng_eval, val_hard_frac, src_frac, 0, g.train_end, n_neg=1,
        pop_frac=pop_frac, nov_frac=nov_frac, nov_window=nov_window)
    tm_ns, tm_nd = all_ns[tm_idx, 0], all_nd[tm_idx, 0]

    # fixed sampled-MRR probe: rank each chosen val positive among C uniform
    # candidates (selection metric when select="mrr")
    n_val = g.val_end - g.train_end
    mrr_ev = np.sort(rng_eval.choice(
        n_val, size=min(val_mrr_events, n_val), replace=False)) + g.train_end
    mrr_cand = rng_eval.integers(
        0, g.n_nodes, size=(len(mrr_ev), val_mrr_cands)).astype(np.int64)

    def sampled_val_mrr(scorer_out):
        C = val_mrr_cands
        pos_sc = scorer_out[:len(mrr_ev)]
        cand_sc = scorer_out[len(mrr_ev):].reshape(len(mrr_ev), C)
        better = (cand_sc > pos_sc[:, None]).sum(1)
        tied = (cand_sc == pos_sc[:, None]).sum(1)
        return float((1.0 / (1.0 + better + 0.5 * tied)).mean())

    train_days = np.unique(g.day[:g.train_end])

    def epoch_metrics():
        """One streaming replay scoring the fixed train subset + validation."""
        qs = [tm_s, tm_ns, g.src[val_sl], val_ns[:, 0]]
        qd = [tm_d, tm_nd, g.dst[val_sl], val_nd[:, 0]]
        qt = [tm_t, tm_t, g.day[val_sl], g.day[val_sl]]
        if select in ("mrr", "combo"):
            C = val_mrr_cands
            ms = np.repeat(g.src[mrr_ev], C)
            qs += [g.src[mrr_ev], ms]
            qd += [g.dst[mrr_ev], mrr_cand.ravel()]
            qt += [g.day[mrr_ev], np.repeat(g.day[mrr_ev], C)]
        lg, _ = score_pairs_streaming(
            model, g, store, device, np.concatenate(qs), np.concatenate(qd),
            np.concatenate(qt))
        n, v = n_tm, g.val_end - g.train_end
        tr_y = np.r_[np.ones(n), np.zeros(n)]
        va_y = np.r_[np.ones(v), np.zeros(v)]
        tr_sc, va_sc = lg[:2 * n], lg[2 * n:2 * n + 2 * v]
        val_mrr = (sampled_val_mrr(lg[2 * n + 2 * v:])
                   if select in ("mrr", "combo") else None)
        return (average_precision(tr_y, tr_sc), auroc(tr_y, tr_sc),
                average_precision(va_y, va_sc), auroc(va_y, va_sc), val_mrr)

    best_ap, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        mem, last = model.init_memory(g.n_nodes, device)
        pending = None  # (lo, hi, day) of the most recently completed day
        # negative pools rebuilt each epoch so day-T sampling is strictly-past
        sampler = StreamingNegatives(g.n_nodes, pos_by_day, rng,
                                     hard_frac, src_frac, pop_frac, pop_window,
                                     nov_frac=nov_frac, nov_window=nov_window)
        hard_sampler = StreamingNegatives(g.n_nodes, pos_by_day, rng,
                                          1.0, src_frac)
        # recency features advance with the samplers: day T sees days <= T-1,
        # matching the streaming scorer's discipline
        rec = PairRecency(g.n_nodes, pair_feat_dim) if pair_feat else None
        bces, hubers, totals = [], [], []
        for T in train_days:
            # clamp to the split boundary: real data is day-snapped, but toy
            # graphs may have a day straddling train_end
            dlo, dhi = int(off[T]), min(int(off[T + 1]), g.train_end)
            for blo in range(dlo, dhi, batch):
                bhi = min(blo + batch, dhi)
                s, d = g.src[blo:bhi], g.dst[blo:bhi]
                B = bhi - blo
                if loss != "full":
                    ns, nds = sampler.sample(s, d, int(T), n_neg=n_neg)
                    if n_neg_hard > 0:
                        hs, hds = hard_sampler.sample(s, d, int(T),
                                                      n_neg=n_neg_hard)
                        ns = np.concatenate([ns, hs], axis=1)
                        nds = np.concatenate([nds, hds], axis=1)
                if pending is not None:
                    plo, phi, pday = pending
                    mem_eff, _ = model.apply_messages(
                        mem, last, g.src[plo:phi], g.dst[plo:phi], pday,
                        g.edge_feat[plo:phi])
                else:
                    mem_eff = mem
                opt.zero_grad()
                if loss == "full":
                    # every node is a candidate: embed the whole vocabulary
                    # once at day T, score B x n_nodes pairs, softmax over
                    # all of them (self + other same-day positives masked)
                    all_nodes = np.arange(g.n_nodes)
                    Z_all = model.embed(all_nodes, np.full(g.n_nodes, int(T)),
                                        mem_eff, store, device)
                    zs, zd = Z_all[s], Z_all[d]
                    n = g.n_nodes
                    zs_rep = zs.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
                    zc_rep = Z_all.unsqueeze(0).expand(B, n, -1).reshape(B * n, -1)
                    pf_all = None
                    if rec is not None:
                        cache = {}
                        rows = []
                        for s_ in s.tolist():
                            if s_ not in cache:
                                cache[s_] = rec.features_all(s_, int(T))
                            rows.append(cache[s_])
                        pf_all = torch.from_numpy(
                            np.stack(rows).reshape(B * n, -1)).to(device)
                    logits_all = model.link_logit(zs_rep, zc_rep, pf_all).view(B, n)
                    invalid = torch.from_numpy(full_softmax_mask(
                        s, d, pos_by_day.get(int(T), set()), n)).to(device)
                    link_loss = full_softmax_ce_loss(
                        logits_all, torch.from_numpy(d).to(device), invalid)
                    amt_p = model.amt_head(
                        torch.cat([zs, zd], dim=-1)).squeeze(-1)
                else:
                    # embed each unique node once (all queries share day T)
                    K = ns.shape[1]  # n_neg (+ n_neg_hard in two-term mode)
                    nodes = np.concatenate([s, d, ns.ravel(), nds.ravel()])
                    uniq, inv = np.unique(nodes, return_inverse=True)
                    Z = model.embed(uniq, np.full(len(uniq), int(T)), mem_eff,
                                    store, device)[inv]
                    zs, zd = Z[:B], Z[B:2 * B]
                    zns = Z[2 * B:2 * B + B * K]
                    znd = Z[2 * B + B * K:]
                    pf_p = pf_n = None
                    if rec is not None:
                        pf_p = torch.from_numpy(
                            rec.features(s, d, int(T))).to(device)
                        pf_n = torch.from_numpy(
                            rec.features(ns.ravel(), nds.ravel(),
                                         int(T))).to(device)
                    logit_p = model.link_logit(zs, zd, pf_p)
                    amt_p = model.amt_head(
                        torch.cat([zs, zd], dim=-1)).squeeze(-1)
                    logit_n = model.link_logit(zns, znd, pf_n).view(B, K)
                    if n_neg_hard > 0 and hard_hinge > 0:
                        link_loss = (softmax_ce_loss(logit_p, logit_n[:, :n_neg])
                                     + beta_hard * F.relu(
                                         hard_hinge - (logit_p.unsqueeze(1)
                                                       - logit_n[:, n_neg:])).mean())
                    elif n_neg_hard > 0:
                        link_loss = (softmax_ce_loss(logit_p, logit_n[:, :n_neg])
                                     + beta_hard * softmax_ce_loss(
                                         logit_p, logit_n[:, n_neg:]))
                    elif loss == "ce" and in_batch:
                        zsr = zs.unsqueeze(1).expand(B, B, -1).reshape(B * B, -1)
                        zdr = zd.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)
                        ib_f = None
                        if rec is not None:
                            ib_f = torch.from_numpy(
                                rec.features_cross(s, d, int(T))).to(device)
                            ib_f = ib_f.reshape(B * B, -1)
                        ib = model.link_logit(zsr, zdr, ib_f).view(B, B)
                        pos_t = pos_by_day.get(int(T), set())
                        invalid = d[None, :] == d[:, None]  # own/duplicate dst
                        for i_, s_ in enumerate(s.tolist()):
                            for j_, d_ in enumerate(d.tolist()):
                                if (s_, d_) in pos_t:
                                    invalid[i_, j_] = True
                        ib = ib.masked_fill(
                            torch.from_numpy(invalid).to(device), float("-inf"))
                        logits = torch.cat([logit_p.unsqueeze(1), logit_n, ib], 1)
                        link_loss = (torch.logsumexp(logits, 1) - logit_p).mean()
                    elif loss == "ce":
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
            sampler.observe_day(g.src[dlo:dhi], g.dst[dlo:dhi], int(T))
            # advancing hard_sampler was missing before 2026-08-14: two-term
            # runs (L5, pf_2t, hinge) drew from empty pools and silently fell
            # back to uniform — those results never tested their hypothesis
            hard_sampler.observe_day(g.src[dlo:dhi], g.dst[dlo:dhi], int(T))
            if rec is not None:
                rec.observe_day(g.src[dlo:dhi], g.dst[dlo:dhi], int(T))

        train_ap, train_auroc, val_ap, val_auroc, val_mrr = epoch_metrics()
        h = {
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "train_bce": float(np.mean(bces)),
            "train_huber": float(np.mean(hubers)),
            "train_ap": train_ap, "train_auroc": train_auroc,
            "val_ap": val_ap, "val_auroc": val_auroc,
        }
        if val_mrr is not None:
            h["val_mrr"] = val_mrr
        history.append(h)
        if select == "mrr":
            sel_metric = val_mrr
        elif select == "combo":
            # geometric mean: excel at BOTH the matched-negative paired
            # metric and the sampled ranking metric, penalizing either collapse
            sel_metric = (max(val_mrr, 0.0) * max(val_ap, 0.0)) ** 0.5
        else:
            sel_metric = val_ap
        if log_every and epoch % log_every == 0:
            print(f"[seed {seed}] epoch {epoch} loss {np.mean(totals):.4f} "
                  f"val_ap {val_ap:.4f}"
                  + (f" val_mrr {val_mrr:.4f}" if val_mrr is not None else ""),
                  flush=True)
        if save_every and save_dir is not None \
                and (epoch + 1) % save_every == 0:
            # periodic snapshots for long fixed-epoch runs (epoch dial study)
            torch.save(model.state_dict(),
                       Path(save_dir) / f"tgn_seed{seed}_ep{epoch + 1}.pt")
        if sel_metric > best_ap:
            best_ap, best_state, bad = (sel_metric,
                                        copy.deepcopy(model.state_dict()), 0)
        else:
            bad += 1
            if bad >= patience:
                break

    last_state = copy.deepcopy(model.state_dict()) if keep_last else None
    model.load_state_dict(best_state)
    info = {"best_val_ap": best_ap, "epochs_run": len(history),
            "history": history}
    if keep_last:
        # popped by the caller before JSON serialization
        info["_last_state"] = last_state
    return model, info


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
    ap_.add_argument("--loss", choices=["bce", "ce", "full"], default="bce",
                 help="full = softmax over ALL nodes as candidates "
                      "(no sampled negatives in the loss)")
    ap_.add_argument("--n-neg", type=int, default=1)
    ap_.add_argument("--hard-frac", type=float, default=0.0)
    ap_.add_argument("--src-frac", type=float, default=0.5)
    ap_.add_argument("--n-neg-hard", type=int, default=0)
    ap_.add_argument("--beta-hard", type=float, default=1.0)
    ap_.add_argument("--pair-feat", action="store_true")
    ap_.add_argument("--hard-hinge", type=float, default=0.0)
    ap_.add_argument("--select", choices=["ap", "mrr", "combo"], default="ap")
    ap_.add_argument("--head", choices=["mlp", "bilinear"], default="mlp")
    ap_.add_argument("--in-batch", action="store_true")
    ap_.add_argument("--pop-frac", type=float, default=0.0)
    ap_.add_argument("--pop-window", type=int, default=0)
    ap_.add_argument("--nov-frac", type=float, default=0.0)
    ap_.add_argument("--nov-window", type=int, default=30)
    ap_.add_argument("--n-layers", type=int, default=1,
                     help="attention hops (1 = memory only at the leaves)")
    ap_.add_argument("--n-stack", type=int, default=0,
                     help="extra residual attention layers over the same "
                          "one-hop neighbors (depth without wider field)")
    ap_.add_argument("--k-inner", type=int, default=0,
                     help="neighbors per inner hop (0 = --k); NOT recorded "
                          "in the checkpoint, pass at eval too")
    ap_.add_argument("--nbr-mode", choices=["recent", "strat"],
                     default="recent")
    ap_.add_argument("--save-last", action="store_true",
                     help="also save the final-epoch weights as "
                          "tgn_seed{seed}_last.pt (for fixed-epoch runs)")
    ap_.add_argument("--save-every", type=int, default=0,
                     help="also save tgn_seed{seed}_ep{N}.pt every N epochs")
    ap_.add_argument("--pair-feat-dim", type=int, default=FEAT_DIM,
                     choices=[FEAT_DIM, FEAT_DIM_BUCKETS, FEAT_DIM_GLOBAL,
                              FEAT_DIM_GLOBAL_BUCKETS, FEAT_DIM_WIDE])
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
                                src_frac=args.src_frac,
                                n_neg_hard=args.n_neg_hard,
                                beta_hard=args.beta_hard,
                                pair_feat=args.pair_feat,
                                hard_hinge=args.hard_hinge,
                                select=args.select, in_batch=args.in_batch,
                                pop_frac=args.pop_frac,
                                pop_window=args.pop_window,
                                pair_feat_dim=args.pair_feat_dim,
                                head=args.head, nov_frac=args.nov_frac,
                                nov_window=args.nov_window,
                                n_layers=args.n_layers,
                                nbr_mode=args.nbr_mode,
                                keep_last=args.save_last,
                                n_stack=args.n_stack, k_inner=args.k_inner,
                                save_every=args.save_every, save_dir=out)
        torch.save(model.state_dict(), out / f"tgn_seed{seed}.pt")
        if args.save_last:
            torch.save(info.pop("_last_state"),
                       out / f"tgn_seed{seed}_last.pt")
        log[seed] = info
        print(f"seed {seed}: best val AP {info['best_val_ap']:.4f} "
              f"({info['epochs_run']} epochs)", flush=True)
        # flush after every seed so an interrupted run keeps finished histories
        (out / "train_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
