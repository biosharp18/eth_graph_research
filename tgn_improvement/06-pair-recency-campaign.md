# 06 — Pair-recency campaign (2026-08-14): closing the deployment gap

Objective (user directive): explain *why* no learned config approached the
recency heuristic's deployment MRR (0.354 vs ≤0.08), and improve both
historical AU-ROC and deployment MRR concurrently. Both done. Evaluation
protocol frozen and unchanged throughout, as always.

## Step 1 — Diagnosis (no training)

New per-event forensics over the 6,176 test events × 11,812 candidates
(scratch script; results summarized here):

- **The failure was never partner confusion.** For events whose target is a
  past partner of the source (53% of test), only ~2% of the candidates that
  outrank the true destination are other partners of that source; **70–82%
  are globally active nodes the source never paid** (hard-CE and L2 models
  alike). Within-source, model score vs pair recency already correlated
  (Spearman ≈ 0.56). The `[z_s ‖ z_c]` head simply cannot recover "c is
  *this source's* partner" at full-ranking precision — confirming 04's
  representational hypothesis, but locating it precisely: *cross-source
  calibration*, not within-source ordering.
- **Recency-primary + model-tiebreak (lexicographic, no training) scored
  MRR 0.374 > 0.354** — the model adds real value ordering candidates that
  recency ties (esp. unseen: 0.178 vs 0.157). So "recency signal + learned
  re-ranking" had headroom; the model just needed the signal as an input.

## Step 2 — Streaming pair-recency features (`src/tgn/recency.py`)

4 link-head-only features per (s, c, T), advanced with the exact days-<-T
discipline memory uses (train, streaming eval, and ranking all share it):
`[pair_seen, log1p(Δt_pair), dst_seen, log1p(Δt_dst_activity)]`.
`pair_feat_dim` is inferred from checkpoints (`build_from_checkpoint`), so
all old models load unchanged. Amount head untouched. Optional
`--pair-feat-dim 7` adds sharp Δt indicators (==1, ==2, ≤7) — measured
worth ≈ +0.004 MRR, −0.02 historical; not used in the headline.

## Step 3 — What training signals do with the feature (2-seed scoping)

| config (all CE, pair-feat) | hist | MRR | lesson |
|---|---|---|---|
| 5 uniform (P1) | 0.255 | 0.249 | feature dominates → EdgeBank_inf failure mode; MRR 3× best-ever |
| 5, hard .5 (P2) | **0.938** | 0.019 | best historical in project; recent partners deflated below unseen mass |
| hard .1 | 0.837 | 0.170 | mixture interpolates smoothly… |
| hard .25 | 0.909 | 0.094 | …but no mixture point wins both |
| two-term 5+5 (β .3/1.0) | 0.61 | 0.06 | L5 failure persists with features |
| hinge hard term (m=2, β .5/1) | 0.27 | 0.249 | margin saturates; no historical discrimination |
| **hard .1 + val-MRR selection** | 0.859 | 0.273 | selection metric alone: +0.10 MRR *and* +0.02 hist |
| in-batch negatives (h .1/.2) | 0.27/0.30 | 0.24/0.25 | 199 active-today negs swamp softmax like uniform |
| src_frac 1.0 | 0.640 | 0.192 | all-same-source hard negs hurt both |

**Mechanism (the "why" of the whole gap):** sampled-softmax CE calibrates the
score against its *training negative distribution*. Uniform ≈ deployment
candidates → ranking-calibrated but no within-seen discrimination (historical
collapses to EdgeBank_inf's 0.26 because unseen positives score below any
seen pair). Hard ≈ historical pool → superb paired discrimination, but
sampling recent partners as negatives deflates their base rate, and full
ranking demotes exactly the candidates that are usually correct. The metrics
conflict only through the negative distribution, not intrinsically.

## Step 4 — Popularity negatives (the unlock)

Post-feature forensics on the 0.273-MRR model: median rank of a true partner
fell 45 → 3, and 85% of remaining outrankers were *popular hubs* — uniform
sampling draws each with p 1/11,812, so their scores are never pushed down
for sources they don't serve. `--pop-frac q`: negative destinations drawn
from past-destination frequency (word2vec-style). Result (2 seeds):
hard .1 + pop .5 → hist 0.836, MRR **0.326**; hub outrankers 85% → 18%,
median partner rank 3 → **1**. Windowed popularity (last 10k tokens): worse.

## Headline: 5-seed frontier (all `--loss ce --n-neg 5 --pair-feat --select mrr`)

| config | random | historical | inductive | MRR | hits@1 | hits@10 | hits@100 |
|---|---|---|---|---|---|---|---|
| hard .1 pop .5 (MRR-max) | 0.9798±.003 | 0.8266±.013 | 0.5767±.005 | **0.3314±.009** | 0.2582 | 0.4622 | **0.5993** |
| hard .2 pop .4 (balanced) | 0.9773±.003 | 0.8639±.010 | 0.5730±.010 | 0.3121±.016 | 0.2445 | 0.4345 | 0.5834 |
| hard .3 pop .4 (hist-max) | 0.9690±.004 | **0.8767±.013** | 0.5684±.011 | 0.2965±.018 | 0.2378 | 0.4014 | 0.5522 |
| recency heuristic | — | — | — | 0.3540 | 0.2777 | 0.4734 | 0.5266 |
| old headline (hard-CE) | 0.9679 | 0.8990 | 0.5994 | 0.0456 | 0.0174 | 0.0925 | 0.3397 |
| old TGN baseline (BCE) | 0.9640 | 0.6505 | 0.5534 | 0.0657 | 0.0281 | 0.1302 | 0.4418 |

- Deployment MRR ×5–7 over any pre-feature config; 94% of recency, beats it
  on hits@100 (+0.07) and nearly matches hits@10; unseen-stratum MRR
  0.1597 ≥ recency's 0.1572.
- Historical stays 0.83–0.88 (vs 0.65 baseline; old headline 0.899 is still
  +0.02–0.07 above, entirely from the *unseen* stratum — seen-stratum
  historical is 0.915–0.917, ≥ the old headline's 0.901).
- Random AU-ROC best ever (0.98). Amount head unchanged-to-slightly-better
  (RMSE 1.6165 vs 1.633). Lexicographic recency+model combo with the
  MRR-max model: 0.368 (a valid deployment scorer if beating 0.354 outright
  is required today).
- seed-0 forensics: partner-stratum MRR 0.6265 vs recency 0.6681; remaining
  outrankers are 73% never-seen candidates (genuine model bets), 18% hubs.

## Artifacts

| what | where |
|---|---|
| headline models (3 configs × 5 seeds) | `figures/tgn_pfpop_{mrr,bal,hist}_models/` (gitignored; train_log force-added) |
| headline results/rankings | `figures/tgn_pfpop_{mrr,bal,hist}_{results,ranking}.json` |
| frontier figure | `figures/tgn_frontier.png` |
| scoping runs (2 seeds each) | `figures/tgn_pf_*/` (results/ranking/train_log JSONs force-added) |
| code | `src/tgn/recency.py` (features), `train.py` (`--pair-feat[-dim]`, `--pop-frac`, `--pop-window`, `--select mrr`, `--hard-hinge`, `--in-batch`), `model.py` (`link_logit`, `build_from_checkpoint`) |
| tests | `tests/test_tgn_recency.py` (13 tests; suite total 72) |

Reproduction of the MRR-max headline:

```
... python -m tgn.train --parquet $P --seeds 0 1 2 3 4 \
      --out figures/tgn_pfpop_mrr_models \
      --loss ce --n-neg 5 --hard-frac 0.1 --pop-frac 0.5 \
      --pair-feat --select mrr
... python -m tgn.evaluate --models figures/tgn_pfpop_mrr_models --out figures/tgn_pfpop_mrr_results.json
... python -m tgn.ranking  --models figures/tgn_pfpop_mrr_models --out figures/tgn_pfpop_mrr_ranking.json
```

## Dead ends (do not re-try as configured)

- Two-term CE with pair features (β 0.3/1.0): dominated, like L5.
- Hinge hard term (m=2): saturates, no historical discrimination.
- Full in-batch negatives: swamp the softmax; subsampled variant untried.
- src_frac 1.0, pop_window, n_neg 10/20: no gains over the simple recipe.
- Note `best_val_ap` in train logs is the sampled val-MRR when
  `--select mrr` (field name kept for schema compatibility).

## Open next steps (ordered)

1. Recover the last ~0.02 MRR (hits@1 0.258 vs 0.278): the residual seen-pair
   loss is unseen candidates outranking true partners — plausibly *right*
   bets sometimes; per-day analysis of whether those unseen picks were the
   day's actual unseen positives would tell if the model is being penalized
   for correct risk-taking on the wrong event.
2. Historical-unseen stratum (0.767 in MRR-max config) is where the old
   headline still wins; a small auxiliary hard-CE on *unseen-positive days
   only*, or pop-negatives restricted to seen pairs, might recover it
   without re-deflating partners.
3. TGB-style multi-negative evaluation (unchanged from 04's list).
4. Pair-level learned memory (04's idea 2) is now lower value: the static
   Δt features already deliver most of what it promised.
