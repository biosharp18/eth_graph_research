"""Fixed-50-epoch W2 run: training curves INCLUDING the stopping criterion.

Three panels: loss terms; validation AP and sampled validation MRR; the
selection criterion combo = sqrt(val_ap * val_mrr) with each seed's argmax
marked and the span where the original adaptive run stopped (epochs 9-14)
shaded for reference. Repo conventions: fixed palette, one axis per panel,
legends, 170 dpi.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"


def main():
    log = json.load(open("tgn_global/figures/w2_fixed50/train_log.json"))
    seeds = sorted(log)
    hists = {s: log[s]["history"] for s in seeds}
    combo = {s: np.array([np.sqrt(max(e["val_mrr"], 0) * max(e["val_ap"], 0))
                          for e in hists[s]]) for s in seeds}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.4))

    def series(ax, key, color, label):
        allv = [np.array([e[key] for e in hists[s]]) for s in seeds]
        for v in allv:
            ax.plot(np.arange(len(v)), v, color=color, alpha=0.25, lw=0.8)
        n = min(len(v) for v in allv)
        mean = np.mean([v[:n] for v in allv], axis=0)
        ax.plot(np.arange(n), mean, color=color, lw=2.2, label=label)

    series(ax1, "train_loss", BLUE, "total")
    series(ax1, "train_bce", ORANGE, "CE (link)")
    series(ax1, "train_huber", GREEN, "Huber (amount)")
    ax1.set_ylabel("training loss")
    ax1.set_title("Loss terms")

    series(ax2, "val_ap", BLUE, "val AP (matched negatives)")
    series(ax2, "val_mrr", ORANGE, "val MRR (sampled)")
    ax2.set_ylabel("validation metric")
    ax2.set_title("Validation AP vs sampled MRR")

    for s in seeds:
        c = combo[s]
        ax3.plot(np.arange(len(c)), c, color=GREEN, alpha=0.25, lw=0.8)
        b = int(np.argmax(c))
        ax3.plot(b, c[b], "o", color="#333333", ms=4)
    n = min(len(combo[s]) for s in seeds)
    mean = np.mean([combo[s][:n] for s in seeds], axis=0)
    ax3.plot(np.arange(n), mean, color=GREEN, lw=2.2,
             label="combo = sqrt(val_ap x val_mrr)")
    ax3.axvspan(9, 14, color=GRAY, alpha=0.15,
                label="adaptive run stopped here")
    ax3.plot([], [], "o", color="#333333", ms=4, label="per-seed best epoch")
    ax3.set_ylabel("selection criterion")
    ax3.set_title("Stopping criterion over 50 fixed epochs")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("epoch")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)

    fig.suptitle("W2 (wide + novelty) at a fixed 50 epochs — the ranking "
                 "criterion peaks by ~epoch 8 while matched AP keeps rising: "
                 "epoch count is a calibration dial (see results.md)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("tgn_global/figures/w2_fixed50_curves.png", dpi=170)
    print("saved tgn_global/figures/w2_fixed50_curves.png")


if __name__ == "__main__":
    main()
