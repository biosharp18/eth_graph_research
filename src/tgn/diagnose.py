"""Deployment-ranking forensics (no training) — the diagnostic behind the
2026-08-14 pair-recency campaign (tgn_improvement/06).

For a given TGN checkpoint, streams the test days once and reports:
  1. model-only filtered ranks (sanity: must reproduce ranking.json numbers);
  2. lexicographic combo: recency primary, model score as tiebreak
     (the no-training ceiling of "recency + model re-ranking");
  3. for events whose target IS a past partner of the source: composition of
     the candidates that outrank it under the model
     (partner-of-s / globally-seen-node / never-seen-node);
  4. within-source rank correlation between model score and -dt(s,c) over the
     source's past partners (does the model encode pair recency at all?);
  5. recency-only ranks recomputed for cross-checking.

Usage: python diag_ranking.py --parquet P --model path/to/tgn_seedX.pt --out out.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tgat.data import load_daily_graph
from tgat.evaluate import _pair_sets
from tgat.neighbors import NeighborStore
from tgn.model import TGN, build_from_checkpoint
from tgn.ranking import HITS, _accumulate, _seen_flags, _test_day_positives, filtered_rank
from tgn.streaming import iter_day_embeddings


def lex_rank(last_day, scores, target, exclude):
    """Tie-aware rank with primary key last_day (higher=better, -inf if unseen)
    and secondary key model score."""
    keep = np.ones(len(scores), bool)
    keep[exclude] = False
    keep[target] = True
    lt, st = last_day[target], scores[target]
    better = ((last_day > lt) | ((last_day == lt) & (scores > st))) & keep
    tied = (last_day == lt) & (scores == st) & keep
    return 1.0 + int(better.sum()) + 0.5 * (int(tied.sum()) - 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    g = load_daily_graph(args.parquet)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=args.k)
    model = build_from_checkpoint(args.model, edge_feat_dim=store.F,
                                  raw_feat_dim=g.edge_feat.shape[1],
                                  dim=args.dim, device=device)
    model.eval()
    from tgn.recency import PairRecency
    rec = PairRecency(g.n_nodes) if model.pair_feat_dim else None

    by_day = _test_day_positives(g)
    n_test = len(g.src) - g.val_end
    seen_flags = _seen_flags(g)

    model_ranks = np.empty(n_test)
    lex_ranks = np.empty(n_test)
    rec_ranks = np.empty(n_test)
    partner_flags = np.zeros(n_test, bool)   # target is past partner of s at T
    n_partners = np.zeros(n_test, np.int64)  # source's partner count at T
    target_recency_pos = np.full(n_test, -1.0)  # recency rank among partners
    # composition of model-outrankers for partner-target events
    comp = {"partner": 0, "global_seen": 0, "never_seen": 0, "events": 0}
    corr_list = []  # within-source spearman(score, -dt) where >=5 partners

    src_last = defaultdict(dict)
    seen_dst = set()
    ev_i, E = 0, len(g.src)

    # advance recency state through train+val first
    day_iter = iter_day_embeddings(model, g, store, device, by_day.keys())
    for T, Z in day_iter:
        while ev_i < E and g.day[ev_i] < T:
            src_last[int(g.src[ev_i])][int(g.dst[ev_i])] = int(g.day[ev_i])
            seen_dst.add(int(g.dst[ev_i]))
            if rec is not None:
                rec.observe_day(g.src[ev_i:ev_i + 1], g.dst[ev_i:ev_i + 1],
                                int(g.day[ev_i]))
            ev_i += 1
        Zc = Z  # (n_nodes, dim) on device
        for s, entries in by_day[T].items():
            cand = src_last[s]
            zs = Zc[s].unsqueeze(0)
            pf = None
            if rec is not None:
                pf = torch.from_numpy(rec.features_all(s, T)).to(device)
            scores = []
            for lo in range(0, g.n_nodes, 4096):
                zc = Zc[lo:lo + 4096]
                scores.append(model.link_logit(
                    zs.expand(len(zc), -1), zc,
                    None if pf is None else pf[lo:lo + 4096]))
            scores = torch.cat(scores).cpu().numpy().astype(np.float64)
            last_day = np.full(g.n_nodes, -np.inf)
            for c, lt in cand.items():
                last_day[c] = lt
            dsts = [d for _, d in entries]
            # within-source recency/score correlation
            if len(cand) >= 5:
                cs = np.array(list(cand.keys()))
                lts = np.array([cand[c] for c in cs], np.float64)
                sc = scores[cs]
                r1 = np.argsort(np.argsort(lts)).astype(np.float64)
                r2 = np.argsort(np.argsort(sc)).astype(np.float64)
                r1 -= r1.mean(); r2 -= r2.mean()
                denom = np.sqrt((r1 ** 2).sum() * (r2 ** 2).sum())
                if denom > 0:
                    corr_list.append(float((r1 * r2).sum() / denom))
            for j, d in entries:
                exclude = np.array([c for c in dsts if c != d], np.int64)
                model_ranks[j] = filtered_rank(scores, d, exclude)
                lex_ranks[j] = lex_rank(last_day, scores, d, exclude)
                rec_ranks[j] = lex_rank(last_day, np.zeros_like(scores), d,
                                        exclude)
                n_partners[j] = len(cand)
                if d in cand:
                    partner_flags[j] = True
                    target_recency_pos[j] = sum(
                        1 for c, lt in cand.items() if lt > cand[d]) + 1
                    keep = np.ones(g.n_nodes, bool)
                    keep[exclude] = False
                    out = np.where((scores > scores[d]) & keep)[0]
                    comp["events"] += 1
                    for c in out:
                        c = int(c)
                        if c in cand:
                            comp["partner"] += 1
                        elif c in seen_dst:
                            comp["global_seen"] += 1
                        else:
                            comp["never_seen"] += 1

    res = {
        "model": args.model,
        "model_only": _accumulate(model_ranks, seen_flags),
        "lex_recency_then_model": _accumulate(lex_ranks, seen_flags),
        "recency_recheck": _accumulate(rec_ranks, seen_flags),
        "partner_stratum": {
            "n_events": int(partner_flags.sum()),
            "frac_of_test": float(partner_flags.mean()),
            "model_mrr": float((1 / model_ranks[partner_flags]).mean()),
            "lex_mrr": float((1 / lex_ranks[partner_flags]).mean()),
            "recency_mrr": float((1 / rec_ranks[partner_flags]).mean()),
            "target_is_most_recent_frac": float(
                (target_recency_pos[partner_flags] == 1).mean()),
            "median_model_rank": float(np.median(model_ranks[partner_flags])),
            "median_n_partners": float(np.median(n_partners[partner_flags])),
        },
        "nonpartner_stratum": {
            "n_events": int((~partner_flags).sum()),
            "model_mrr": float((1 / model_ranks[~partner_flags]).mean()),
            "lex_mrr": float((1 / lex_ranks[~partner_flags]).mean()),
            "recency_mrr": float((1 / rec_ranks[~partner_flags]).mean()),
            "median_model_rank": float(np.median(model_ranks[~partner_flags])),
        },
        "outranker_composition_partner_events": {
            **comp,
            "mean_outrankers": comp["partner"] + comp["global_seen"]
            + comp["never_seen"],
        },
        "within_source_spearman_score_vs_recency": {
            "mean": float(np.mean(corr_list)),
            "median": float(np.median(corr_list)),
            "n": len(corr_list),
        },
    }
    if comp["events"]:
        tot = comp["partner"] + comp["global_seen"] + comp["never_seen"]
        res["outranker_composition_partner_events"]["mean_outrankers"] = \
            tot / comp["events"]
        for k in ("partner", "global_seen", "never_seen"):
            res["outranker_composition_partner_events"][f"frac_{k}"] = \
                comp[k] / max(tot, 1)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
