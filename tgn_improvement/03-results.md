# 03 — Headline results (TGN hard-CE, 5 seeds) and artifact map

Winner config: `tgn.train --loss ce --n-neg 5 --hard-frac 0.5` (src_frac 0.5),
everything else at protocol defaults (Adam 1e-4, batch 200 within-day, ≤50
epochs, patience 5 on matched-negative val AP, k=20 neighbors, dim 100).
Early stopping fired for all 5 seeds at 11–20 epochs (vs 33–50 for the
baseline loss) — hard-CE converges much faster.

## Link prediction (test, 1 neg/pos, frozen pools, mean ± std over 5 seeds)

| NS strategy | AU-ROC | AP | AU-ROC seen | AU-ROC unseen |
|---|---|---|---|---|
| random | 0.9679 ± 0.0018 | 0.9563 ± 0.0026 | 0.9651 ± 0.0018 | 0.9698 ± 0.0029 |
| historical | 0.8990 ± 0.0027 | 0.8938 ± 0.0035 | 0.9009 ± 0.0034 | 0.8979 ± 0.0029 |
| inductive | 0.5994 ± 0.0072 | 0.5628 ± 0.0070 | 0.5797 ± 0.0078 | 0.6120 ± 0.0076 |

Deltas that matter:
- historical vs baseline TGN +0.2485; vs TGAT +0.1945; vs EdgeBank_tw +0.2494.
- historical **unseen** stratum 0.5821 → 0.8979 — the old loss's worst
  failure mode eliminated.
- inductive +0.046 over baseline (both TGAT and old TGN were ≈0.55).
- random +0.004 (no cost; slight gain).

## Amount head (unchanged by design; log10-USD)

RMSE 1.6329 ± 0.0171 (seen 0.6565, unseen 2.0501), MAE 0.9946 ± 0.0288.
Statistically the same as baseline TGN (1.6255) and TGAT (1.6361); still
beats persistence (1.9457) and median (2.1116). Link-loss changes did not
disturb the amount head.

## Deployment ranking (6,176 events × 11,812 candidates, filtered, tie-aware)

| metric | TGN hard-CE | TGN baseline | recency heuristic |
|---|---|---|---|
| MRR | 0.0456 (seen 0.0643 / unseen 0.0328) | 0.0657 | 0.3540 (0.6404 / 0.1572) |
| hits@1 | 0.0174 | 0.0281 | 0.2777 |
| hits@10 | 0.0925 | 0.1302 | 0.4734 |
| hits@100 | 0.3397 | 0.4418 | 0.5266 |

Honest cost of the winner: −0.02 MRR vs baseline. If deployment ranking is
the target, the L2 loss (CE, 5 uniform negatives) is the right choice
(MRR 0.0801, hits@10 0.1562) — at the price of historical staying at 0.60.
No loss config approaches recency.

## Where everything lives

| artifact | path |
|---|---|
| winner checkpoints + train log | `figures/tgn_hard_models/` (gitignored; `train_log.json` force-added) |
| winner results / ranking JSON | `figures/tgn_hard_results.json`, `figures/tgn_hard_ranking.json` |
| winner training curves | `figures/tgn_hard_training_curves.png` (loss panel's "BCE (link)" label = the CE term; shared plotting schema) |
| 4-way comparison bar | `figures/tgn_loss_bar.png` (EB_tw / TGAT / TGN-BCE / TGN-hard-CE × 3 strategies) |
| scoping runs L1–L5 | `figures/tgn_loss_l{1..5}/` (results/ranking/train_log JSONs committed) |
| baseline TGN artifacts | `figures/tgn_results.json`, `figures/tgn_ranking.json`, `figures/tgn_models/`, `figures/tgn_training_curves.png`, `figures/tgn_vs_tgat_bar.png` |
| TGAT reference | `figures/tgat_results.json`, `figures/tgat_models/` |
| write-ups | `TASK_RESULTS.md` (baseline + loss addendum) |

## Commit trail (feat/bigquery-exploration, newest last)

- `30b8674` feat(tgn): memory module with daily GRU updates and attention embedding
- `c59d197` feat(tgn): streaming day-replay scorer with positive-only memory advance
- `109ab48` feat(tgn): day-granular training loop with one-day-lag differentiable memory
- `f234fdb` feat(tgn): evaluation CLI sharing tgat NS pools and EdgeBank recompute
- `1cb3e87` feat(tgn): deployment ranking with tie-aware filtered MRR and recency baseline
- `deb994e` feat(tgn): 5-seed experiment results, ranking, and comparison figures
- `ba3c8ac` docs(tgn): results write-up — TGN vs TGAT vs EdgeBank vs recency
- `6440dfc` docs(tgn): implementation plan for the TGN daily-flow baseline
- (loss campaign) feat(tgn): hard-negative mixture sampler and softmax ranking loss;
  feat(tgn): two-term ranking loss — separate random and hard CE;
  `c510205` feat(tgn): loss-ablation scoping results (L1–L5, 2 seeds each);
  `286d433` feat(tgn): hard-negative CE loss — 5-seed headline run and figures;
  `65a9367` docs(tgn): loss-redesign addendum

Test suite at time of writing: **59 passed** (`uv run --group ml pytest tests/ -q`,
~15–30 s warm).
