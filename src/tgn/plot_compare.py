"""Bar comparison of TGN vs TGAT vs EdgeBank_tw AU-ROC by NS strategy.

Usage: python -m tgn.plot_compare figures/tgat_results.json \
    figures/tgn_results.json figures/tgn_vs_tgat_bar.png
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
STRATEGIES = ["random", "historical", "inductive"]


def main(tgat_path, tgn_path, out_path):
    tgat = json.load(open(tgat_path))
    tgn = json.load(open(tgn_path))
    series = [
        ("EdgeBank_tw", GRAY,
         [tgat["edgebank"]["tw"][s]["auroc"] for s in STRATEGIES]),
        ("TGAT", BLUE, [tgat["tgat"][s]["auroc"] for s in STRATEGIES]),
        ("TGN", ORANGE, [tgn["tgn"][s]["auroc"] for s in STRATEGIES]),
    ]
    x = np.arange(len(STRATEGIES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (label, color, vals) in enumerate(series):
        means = [np.mean(v) for v in vals]
        stds = [np.std(v) for v in vals]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3,
               color=color, label=label)
    ax.axhline(0.5, color="#e1e0d9", linewidth=1, zorder=0)
    ax.set_xticks(x, [s.capitalize() for s in STRATEGIES])
    ax.set_ylabel("AU-ROC (test, 1 neg/pos, 5 seeds)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(framealpha=0.9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
