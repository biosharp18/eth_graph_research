"""Plot per-epoch training curves from a train_log.json.

Usage: python -m tgat.plot_training <train_log.json> <out.png> [title]

Three panels: (1) total loss with BCE / Huber decomposition, (2) link AP on
the fixed train subset vs validation, (3) link AU-ROC likewise. Per-seed
curves are drawn faint; the bold line is the mean over seeds still running
at each epoch (seeds early-stop at different epochs).
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#898781"


def _series(histories, key):
    """Per-seed arrays plus ragged mean over seeds alive at each epoch."""
    per_seed = [np.array([h[key] for h in hist]) for hist in histories]
    n = max(len(a) for a in per_seed)
    stack = np.full((len(per_seed), n), np.nan)
    for i, a in enumerate(per_seed):
        stack[i, : len(a)] = a
    return per_seed, np.nanmean(stack, axis=0)


def plot_training(log_path: str, out_path: str, title: str = "") -> None:
    log = json.load(open(log_path))
    histories = [info["history"] for info in log.values()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    panels = [
        (axes[0], "loss", [("train_loss", BLUE, "total"),
                           ("train_bce", ORANGE, "BCE (link)"),
                           ("train_huber", AQUA, "Huber (amount)")]),
        (axes[1], "link AP", [("train_ap", BLUE, "train (3k subset)"),
                              ("val_ap", ORANGE, "validation")]),
        (axes[2], "link AU-ROC", [("train_auroc", BLUE, "train (3k subset)"),
                                  ("val_auroc", ORANGE, "validation")]),
    ]
    for ax, ylab, series in panels:
        for key, color, label in series:
            per_seed, mean = _series(histories, key)
            for a in per_seed:
                ax.plot(np.arange(len(a)), a, color=color, alpha=0.18,
                        linewidth=0.9)
            ax.plot(np.arange(len(mean)), mean, color=color, linewidth=2,
                    label=label)
        ax.set_xlabel("Epoch", fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.tick_params(labelsize=8.5)
        ax.legend(fontsize=8.5, framealpha=0.9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.7)
        ax.set_axisbelow(True)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    plot_training(sys.argv[1], sys.argv[2],
                  sys.argv[3] if len(sys.argv) > 3 else "")
