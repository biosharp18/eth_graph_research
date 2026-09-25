"""Per-seed comparison of deep-MP variants against W2@50 (same seeds).

Reads the JSONs written by eval_deep.sh and the W2 fixed-50 references.
Usage: compare_deep.py <variant_dir> [<variant_dir> ...]  (e.g. deep/d_hop2_last)
"""
import json
import sys
from pathlib import Path

import numpy as np

F = Path("tgn_global/figures")
REF = {"dgb": F / "dgb/w2_fixed50_last_dgb.json",
       "modes": F / "dgb/modes_dgb.json",
       "results": F / "w2_fixed50_last_results.json",
       "ranking": F / "w2_fixed50_last_ranking.json"}
REF_BEST = {"dgb": F / "dgb/w2_fixed50_dgb.json",
            "results": F / "w2_fixed50_results.json",
            "ranking": F / "w2_fixed50_ranking.json"}


def metrics(dgb, modes, results, ranking, modes_key=None):
    m = {}
    for s in ("random", "historical", "inductive"):
        m[f"dgb_{s[:4]}"] = dgb["tgn"][s]["auroc_batch_mean"]
    if modes is not None:
        for mode in ("inductive_sym", "test_recurring"):
            key = modes_key or next(iter(modes["modes"][mode]["models"]))
            m[mode[:8]] = modes["modes"][mode]["models"][key]["auroc"]
    for s in ("historical", "inductive"):
        m[f"leg_{s[:4]}"] = results["tgn"][s]["auroc"]
    for k in ("mrr", "hits1", "hits10", "hits100"):
        m[k] = ranking["tgn"][k]["all"]
    return m


def load(d):
    j = lambda n: json.load(open(d / n))
    return metrics(j("dgb.json"), j("modes.json"), j("results.json"),
                   j("ranking.json"))


def main():
    ref_last = metrics(*(json.load(open(REF[k]))
                         for k in ("dgb", "modes", "results", "ranking")),
                       modes_key="w2_fixed50_last")
    ref_best = metrics(json.load(open(REF_BEST["dgb"])), None,
                       json.load(open(REF_BEST["results"])),
                       json.load(open(REF_BEST["ranking"])))
    rows = []
    for arg in sys.argv[1:]:
        d = F / arg
        ref = ref_best if arg.endswith("_best") else ref_last
        seeds = sorted(int(p.stem[8:]) for p in d.glob("tgn_seed*.pt"))
        v = load(d)
        rows.append((arg, seeds, v, ref))
    keys = list(rows[0][2].keys()) if rows else []
    print("metric".ljust(10) + "".join(
        f"{'W2@50':>9}{r[0].replace('deep/', '')[:14]:>16}" for r in rows))
    for k in keys:
        line = k.ljust(10)
        for name, seeds, v, ref in rows:
            if k not in ref:
                line += f"{'-':>9}{np.mean(v[k]):>16.4f}"
                continue
            rv = np.mean([ref[k][s] for s in seeds])
            vv = np.mean(v[k])
            line += f"{rv:>9.4f}{vv:>9.4f} ({vv - rv:+.3f})"
        print(line)
    print("seeds:", {r[0]: r[1] for r in rows})


if __name__ == "__main__":
    main()
