# 07 — Revisit (2026-08-14 PM): is the hist/MRR split fundamental?

Premise (user): the historical-AUROC-best and MRR-best models being different
configs is suspicious — a genuinely performant model should excel at both.
This round probes axes *other than* loss and featurization, with no favored
direction. All artifacts under `tgn_improvement/figures/`.

## Probes and results

Reference points (5 seeds): MRR-max 0.8266/0.3314, balanced 0.8639/0.3121,
hist-max 0.8767/0.2965 (format: historical AU-ROC / deployment MRR).

| probe | hist | MRR | verdict |
|---|---|---|---|
| score ensemble of hist-max + MRR-max (5 seeds, z-avg) | 0.8569 | 0.3318 | interpolates hist, keeps best MRR (+0.006 hits@1) |
| weight soup of the same pair (5 seeds) | 0.8271 | 0.3313 | collapses onto MRR-max parent; no merge |
| bilinear head (2 seeds, h.2/p.4) | 0.8687 | 0.2827 | trades, like a mixture shift; best-ever random 0.9825 |
| dim 200 (2 seeds, h.2/p.4) | 0.8399 | 0.2484 | capacity hurts (echoes TGAT dim-200 null) |
| combo selection = √(val-MRR · val-AP) (5 seeds, h.2/p.4) | 0.8645 | 0.3186 | weakly dominant over MRR-selection (+0.007 MRR, hist equal) |
| **combo selection, h.3/p.4 (2 seeds)** | **0.8957** | **0.2784** | old headline's historical (0.899) at 6× its MRR |

## What this establishes

1. **The split is not a knowledge split.** Both fusion methods show the two
   configs share one representation: the ensemble's historical is strictly
   between the parents, and weight averaging lands on one parent rather than
   breaking (the endpoints are linearly connected — same per-seed init,
   different negative mixtures, one basin).
2. **The hard-negative historical edge is a score-scale property, not extra
   information.** It survives neither score averaging nor weight averaging —
   consistent with 06's mechanism: the negative mixture chooses which
   *calibration* one scalar score expresses, and a single scalar cannot be
   calibrated to two reference distributions at once. The tradeoff is a
   property of scoring-rule geometry, not of the network.
3. **Architecture and capacity are not the bottleneck** (bilinear trades,
   dim-200 hurts). Selection criterion gives a real but small free
   improvement; `--select combo` is now the recommended default.
4. Practical answer to "excel at both": `h.3/p.4 + combo` is the closest
   single model — historical statistically indistinguishable from the old
   hard-CE headline (0.896 vs 0.899) with MRR 0.278 vs its 0.046. The
   remaining daylight to (0.94-hist, 0.35-MRR) simultaneously would need a
   *two-output* design (e.g., a shared trunk with two calibration heads, one
   per reference distribution) — which is arguably just honest bookkeeping of
   the fact that the two benchmarks ask differently-calibrated questions.

## Repro

```
... python -m tgn.train --loss ce --n-neg 5 --hard-frac 0.3 --pop-frac 0.4 \
      --pair-feat --select combo --seeds 0 1 --out tgn_improvement/figures/imp_combo_h30
... python -m tgn.ensemble --models-a figures/tgn_pfpop_hist_models \
      --models-b figures/tgn_pfpop_mrr_models --out .../tgn_ensemble_hist_mrr.json
```

Code added: `--head bilinear`, `--select combo` (`src/tgn/train.py`,
`src/tgn/model.py`), score-ensemble diagnostic (`src/tgn/ensemble.py`).
Suite: 73 passing.
