"""Two-panel summary figure for the tgn_global campaign.

Left: AU-ROC under the three NS strategies for the four 5-seed models, with
the day-split feature-oracle level marked on the inductive group.
Right: deployment ranking metrics vs the recency heuristic.
Repo plotting conventions: fixed palette, legends, no dual axes, 170 dpi.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"


def m(path, *keys):
    d = json.load(open(path))
    for k in keys:
        d = d[k]
    return float(np.mean(d)), float(np.std(d))


def main():
    models = [
        ("TGN hard-CE\n(old headline)", GRAY, "figures/tgn_hard_results.json",
         "figures/tgn_hard_ranking.json"),
        ("pair-recency\n(pfpop-mrr)", BLUE, "figures/tgn_pfpop_mrr_results.json",
         "figures/tgn_pfpop_mrr_ranking.json"),
        ("global feats\n(f14, MRR arm)", GREEN,
         "tgn_global/figures/g_f14_mrr_results.json",
         "tgn_global/figures/g_f14_mrr_ranking.json"),
        ("global+novelty\n(inductive arm)", ORANGE,
         "tgn_global/figures/g_headline_ind_results.json",
         "tgn_global/figures/g_headline_ind_ranking.json"),
    ]
    strategies = ["random", "historical", "inductive"]
    rank_metrics = [("mrr", "MRR"), ("hits1", "hits@1"),
                    ("hits10", "hits@10"), ("hits100", "hits@100")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    W = 0.19
    x = np.arange(len(strategies))
    for i, (name, color, rpath, _) in enumerate(models):
        mu = [m(rpath, "tgn", s, "auroc") for s in strategies]
        ax1.bar(x + (i - 1.5) * W, [a for a, _ in mu], W * 0.92, color=color,
                label=name.replace("\n", " "),
                yerr=[sd for _, sd in mu], capsize=2,
                error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    lo, hi = 2 - 1.5 * W - W / 2, 2 + 1.5 * W + W / 2
    ax1.hlines(0.688, lo, hi, color="#333333", linestyle="--", linewidth=1.2)
    ax1.text(2, 0.700, "feature-family oracle 0.69", ha="center", fontsize=8,
             color="#333333")
    ax1.axhline(0.6, color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax1.text(-0.48, 0.612, "old roof 0.60", fontsize=7.5, color="#666666")
    ax1.set_xticks(x, [s.capitalize() for s in strategies])
    ax1.set_ylim(0.4, 1.02)
    ax1.set_ylabel("test AU-ROC (5 seeds)")
    ax1.set_title("Link prediction by NS strategy")
    ax1.legend(fontsize=7.5, loc="upper right")

    x2 = np.arange(len(rank_metrics))
    rec = {"mrr": 0.3540, "hits1": 0.2777, "hits10": 0.4734, "hits100": 0.5266}
    for i, (name, color, _, kpath) in enumerate(models[1:]):
        mu = [m(kpath, "tgn", k, "all") for k, _ in rank_metrics]
        ax2.bar(x2 + (i - 1) * W, [a for a, _ in mu], W * 0.92, color=color,
                yerr=[sd for _, sd in mu], capsize=2,
                error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
                label=name.replace("\n", " "))
    for j, (k, _) in enumerate(rank_metrics):
        ax2.hlines(rec[k], x2[j] - 0.38, x2[j] + 0.38, color="#333333",
                   linewidth=1.4, linestyle="--",
                   label="recency heuristic" if j == 0 else None)
    ax2.set_xticks(x2, [lbl for _, lbl in rank_metrics])
    ax2.set_ylabel("deployment ranking (5 seeds)")
    ax2.set_title("Full-candidate ranking vs recency")
    ax2.legend(fontsize=7.5, loc="upper left")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.7)
        ax.set_axisbelow(True)

    fig.suptitle("tgn_global: global-in-time awareness breaks the inductive "
                 "roof; ranking arm holds the MRR frontier", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("tgn_global/figures/tgn_global_frontier.png", dpi=170)
    print("saved tgn_global/figures/tgn_global_frontier.png")


if __name__ == "__main__":
    main()
