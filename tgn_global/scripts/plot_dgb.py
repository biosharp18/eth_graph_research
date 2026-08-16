"""Protocol-correction figure: paper-faithful DGB AUROC vs the legacy
protocol, for the models in the campaign narrative.

Left: corrected (per-batch mean) AUROC by NS strategy.
Right: legacy vs corrected INDUCTIVE side by side — the panel that shows
which conclusions were protocol artifacts.
Repo conventions: fixed palette, legends, no dual axes, 170 dpi.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"

MODELS = [  # (label, color, dgb json, legacy results json)
    ("TGN hard-CE", GRAY, "tgn_hard", "figures/tgn_hard_results.json"),
    ("pfpop-mrr", BLUE, "pfpop_mrr", "figures/tgn_pfpop_mrr_results.json"),
    ("wide + two-hop (G2)", GREEN, "g2_twohop_w1",
     "tgn_global/figures/g2_twohop_w1_results.json"),
    ("wide + novelty (W2)", ORANGE, "w2_wide_ind",
     "tgn_global/figures/w2_wide_ind_results.json"),
]


def dgb(name, strategy):
    d = json.load(open(f"tgn_global/figures/dgb/{name}_dgb.json"))
    a = d["tgn"][strategy]["auroc_batch_mean"]
    return float(np.mean(a)), float(np.std(a))


def legacy(path, strategy):
    d = json.load(open(path))["tgn"][strategy]["auroc"]
    return float(np.mean(d)), float(np.std(d))


def main():
    strategies = ["random", "historical", "inductive"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                   width_ratios=[1.15, 1])
    W = 0.19
    x = np.arange(len(strategies))
    for i, (name, color, key, _) in enumerate(MODELS):
        mu = [dgb(key, s) for s in strategies]
        ax1.bar(x + (i - 1.5) * W, [a for a, _ in mu], W * 0.92, color=color,
                label=name, yerr=[sd for _, sd in mu], capsize=2,
                error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax1.set_xticks(x, [s.capitalize() for s in strategies])
    ax1.set_ylim(0.2, 1.0)
    ax1.set_ylabel("test AU-ROC, DGB per-batch mean (5 seeds)")
    ax1.set_title("Corrected (paper-faithful) protocol")
    ax1.legend(fontsize=7.5, loc="lower left")

    x2 = np.arange(len(MODELS))
    W2b = 0.34
    for j, (proto, hatch) in enumerate((("legacy", "//"), ("DGB", None))):
        vals = []
        errs = []
        for name, color, key, lpath in MODELS:
            a, sd = (legacy(lpath, "inductive") if proto == "legacy"
                     else dgb(key, "inductive"))
            vals.append(a)
            errs.append(sd)
        ax2.bar(x2 + (j - 0.5) * W2b, vals, W2b * 0.9,
                color=[c for _, c, _, _ in MODELS],
                hatch=hatch, edgecolor="white" if hatch else "none",
                yerr=errs, capsize=2,
                error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax2.set_xticks(x2, [n.replace(" (", "\n(") for n, _, _, _ in MODELS],
                   fontsize=8)
    ax2.set_ylim(0.4, 0.8)
    ax2.set_ylabel("inductive AU-ROC")
    ax2.set_title("Inductive: legacy (hatched) vs corrected")
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(facecolor="#bbbbbb", hatch="//", edgecolor="white",
              label="legacy protocol"),
        Patch(facecolor="#bbbbbb", label="DGB corrected")],
        fontsize=7.5, loc="upper left")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.7)
        ax.set_axisbelow(True)

    fig.suptitle("DGB protocol correction: novelty arms lead inductive at "
                 "0.69–0.73; the legacy '0.60 roof' was protocol-dependent",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("tgn_global/figures/tgn_dgb_correction.png", dpi=170)
    print("saved tgn_global/figures/tgn_dgb_correction.png")


if __name__ == "__main__":
    main()
