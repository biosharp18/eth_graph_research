"""Deployment-style full-candidate ranking for TGAT checkpoints.

Same protocol as tgn.ranking (filtered, tie-aware, seen/unseen strata,
recency baseline recomputed as a cross-check); only the embedding differs:
TGAT is stateless, so each test day's all-node embeddings come straight
from the neighbor store at that day — no memory to stream.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tgat.data import load_daily_graph
from tgat.model import TGAT
from tgat.neighbors import NeighborStore
from tgn.ranking import (_accumulate, _seen_flags, _test_day_positives,
                         filtered_rank, recency_ranking)


@torch.no_grad()
def tgat_ranking(model, g, store, device, embed_chunk=1024,
                 score_chunk=4096) -> dict:
    model.eval()
    by_day = _test_day_positives(g)
    n_test = len(g.src) - g.val_end
    ranks = np.empty(n_test, np.float64)
    for T in sorted(by_day):
        Z = torch.cat([
            model.embed(np.arange(lo, min(lo + embed_chunk, g.n_nodes)),
                        np.full(min(embed_chunk, g.n_nodes - lo), T),
                        store, device)
            for lo in range(0, g.n_nodes, embed_chunk)])
        for s, entries in by_day[T].items():
            zs = Z[s].unsqueeze(0)
            scores = []
            for lo in range(0, g.n_nodes, score_chunk):
                zc = Z[lo:lo + score_chunk]
                pair = torch.cat([zs.expand(len(zc), -1), zc], dim=1)
                scores.append(model.link_head(pair).squeeze(-1))
            scores = torch.cat(scores).cpu().numpy()
            dsts = [d for _, d in entries]
            for j, d in entries:
                exclude = np.array([c for c in dsts if c != d], np.int64)
                ranks[j] = filtered_rank(scores, d, exclude)
    out = _accumulate(ranks, _seen_flags(g))
    out["n_test"] = n_test
    out["n_candidates"] = g.n_nodes
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet",
                     default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap_.add_argument("--models", default="figures/tgat_models")
    ap_.add_argument("--out", default="figures/tgat_ranking.json")
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_.add_argument("--dim", type=int, default=100)
    ap_.add_argument("--k", type=int, default=20)
    args = ap_.parse_args()
    device = torch.device(args.device)

    g = load_daily_graph(args.parquet)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=args.k)

    rec = recency_ranking(g)
    res = {"recency": {k: rec[k] for k in rec if k.startswith(("mrr", "hits"))},
           "tgat": {}, "n_test": rec["n_test"],
           "n_candidates": rec["n_candidates"]}
    for seed in args.seeds:
        model = TGAT(edge_feat_dim=store.F, dim=args.dim).to(device)
        model.load_state_dict(torch.load(
            Path(args.models) / f"tgat_seed{seed}.pt", map_location=device))
        r = tgat_ranking(model, g, store, device)
        for metric, strata in r.items():
            if not metric.startswith(("mrr", "hits")):
                continue
            for stratum, v in strata.items():
                res["tgat"].setdefault(metric, {}).setdefault(stratum, []).append(v)
        print(f"seed {seed}: MRR {r['mrr']['all']:.4f} "
              f"hits@10 {r['hits10']['all']:.4f}", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
