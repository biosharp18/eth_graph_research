"""Campaign-2 summary figure: receptive-field widening results.

Same layout and conventions as plot_frontier.py; models updated to the
campaign-2 arms.
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
        ("campaign-1 inductive arm", GRAY,
         "tgn_global/figures/g_headline_ind_results.json",
         "tgn_global/figures/g_headline_ind_ranking.json"),
        ("wide feats (W1)", BLUE,
         "tgn_global/figures/w1_wide_mrr_results.json",
         "tgn_global/figures/w1_wide_mrr_ranking.json"),
        ("wide + two-hop (G2)", GREEN,
         "tgn_global/figures/g2_twohop_w1_results.json",
         "tgn_global/figures/g2_twohop_w1_ranking.json"),
        ("wide + novelty (W2)", ORANGE,
         "tgn_global/figures/w2_wide_ind_results.json",
         "tgn_global/figures/w2_wide_ind_ranking.json"),
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
                label=name, yerr=[sd for _, sd in mu], capsize=2,
                error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    lo, hi = 2 - 1.5 * W - W / 2, 2 + 1.5 * W + W / 2
    ax1.hlines(0.689, lo, hi, color="#333333", linestyle="--", linewidth=1.2)
    ax1.text(2, 0.701, "feature-family oracle 0.69", ha="center", fontsize=8,
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
    for i, (name, color, _, kpath) in enumerate(models):
        mu = [m(kpath, "tgn", k, "all") for k, _ in rank_metrics]
        ax2.bar(x2 + (i - 1.5) * W, [a for a, _ in mu], W * 0.92, color=color,
                yerr=[sd for _, sd in mu], capsize=2,
                error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
                label=name)
    for j, (k, _) in enumerate(rank_metrics):
        ax2.hlines(rec[k], x2[j] - 0.44, x2[j] + 0.44, color="#333333",
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

    fig.suptitle("Campaign 2: time-deep features beat the recency heuristic; "
                 "two-hop now adds on top; inductive reaches 0.66",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("tgn_global/figures/tgn_global_frontier2.png", dpi=170)
    print("saved tgn_global/figures/tgn_global_frontier2.png")


if __name__ == "__main__":
    main()
