# 02 — The loss campaign: every iteration, every number

Design principle: change ONLY the training signal (negative distribution +
loss shape + matching model selection); architecture and the frozen evaluation
stay byte-identical, so every delta is attributable to the loss.

## Machinery built (in `src/tgn/train.py`, tests in `tests/test_tgn_negatives.py`)

- **`StreamingNegatives`** — training-negative sampler over full (s,d) pairs.
  Per positive, with prob `hard_frac` draws a "hard" negative: with prob
  `src_frac` a same-source strictly-past partner `(s, c)` (the
  deployment-ranking hard case), else a pair from the global pool of
  previously-seen pairs (the historical-NS distribution). Fallback and the
  remaining prob mass: uniform random destination (baseline behavior). Pools
  are advanced with a day's events only AFTER that day is processed → no
  leakage by construction; same-day-positive collisions always excluded.
- **`softmax_ce_loss(pos, negs)`** — sampled-softmax ranking CE:
  `logsumexp([pos, neg_1..neg_K]) − pos`, mean over batch.
- **Matched model selection** — validation negatives are drawn from the SAME
  distribution as training (fixed per seed, via `sample_negatives_streaming`).
  Critical: selecting on random-negative val AP would silently undo
  hard-negative training. Consequence: `best_val_ap` values are NOT comparable
  across configs with different negative distributions.
- Unique-node batch embedding (each node embedded once per batch) to make
  K=5 affordable (~1.75× baseline epoch cost instead of ~3×).
- Two-term variant: `--n-neg-hard K2 --beta-hard β` adds a separate all-hard
  CE term (see L5).

## The grid (scoping = 2 seeds each; AU-ROC / ranking on the frozen protocol)

| id | loss | negatives per positive | random | historical | inductive | MRR | hits@10 | best val AP (own negs) / epochs |
|---|---|---|---|---|---|---|---|---|
| L0 | BCE | 1 uniform (baseline, 5 seeds) | 0.9640 | 0.6505 | 0.5534 | 0.0657 | 0.1302 | 0.942–0.947 / 33–50 |
| L1 | BCE | 1, hard 50% | 0.9805 | 0.8798 | 0.5927 | 0.0559 | 0.1069 | 0.873, 0.861 / 32, 23 |
| L2 | softmax CE | 5 uniform | 0.9494 | 0.6030 | 0.5567 | **0.0801** | **0.1562** | 0.941, 0.946 / 44, 40 |
| L3 | softmax CE | 5, hard 50% | 0.9675 | 0.9024 | 0.6067 | 0.0494 | 0.0976 | 0.841, 0.832 / 20, 19 |
| L4 | softmax CE | 5, hard 80%, src_frac 0.625 | 0.9636 | **0.9114** | 0.5890 | 0.0476 | 0.0935 | 0.822, 0.809 / 26, 12 |
| L5 | CE + 1.0·CE (two-term) | 5 uniform + 5 all-hard | 0.9338 | 0.7302 | 0.5947 | 0.0511 | 0.1002 | 0.796, 0.790 / 18, 13 |
| **L3 @ 5 seeds (headline)** | softmax CE | 5, hard 50% | **0.9679 ± 0.0018** | **0.8990 ± 0.0027** | 0.5994 ± 0.0072 | 0.0456 | 0.0925 | 0.831–0.846 / 11–20 |

Scoping artifacts: `figures/tgn_loss_l{1..5}/{results.json,ranking.json,train_log.json}`
(committed); headline: `figures/tgn_hard_*`.

## What each iteration taught

**L1 (hard negatives alone, plain BCE):** the single biggest surprise —
historical jumps 0.65 → 0.88 AND random improves to 0.9805 (best random of
any config). The model was always capable of separating "active today" from
"stale pair"; 1-uniform-negative BCE just never asked. Hard negatives cost
nothing on easy negatives.

**L2 (multi-negative CE alone):** the ONLY config that improves deployment
ranking (MRR +22%, hits@10 +20% over baseline) — multi-candidate softmax
pressure calibrates scores across candidates. But historical stays at
baseline (0.603): random negatives, however many, don't teach pair-level
recurrence discrimination. Also the only config whose early stopping runs
long (40–44 epochs), like the baseline.

**L3 (both in one softmax):** best balanced config — dominates baseline on all
three NS strategies. But the MRR benefit of L2 vanishes (0.049 < 0.066):
hard negatives dominate the softmax gradient (they're the high-loss terms),
crowding out the global-calibration signal. Chosen as headline for the 5-seed
protocol because it improves everything the benchmark measures with one knob.

**L4 (harder + src-heavy: hard 80%, of which 62.5% same-source partners):**
built to rescue MRR by emphasizing the deployment-relevant contrast
(rank the true destination above the source's OTHER past partners). Historical
rose further (0.9114, best observed) but MRR did NOT move (0.0476). Even
negatives drawn exactly from the deployment hard case don't teach full-ranking
— strong evidence the MRR ceiling is not about negative choice.

**L5 (two-term: separate CE over 5 uniform + β·CE over 5 all-hard, β=1):**
the principled "keep both gradients" design — and it underperformed on BOTH
fronts (historical 0.73, MRR 0.051, random 0.934, the worst random of any
config). Best guess at mechanism: the 50/50-mixture validation metric peaks
very early (stops at 13–18 epochs) and favors neither objective; the two
terms may also fight over the score scale (the uniform term saturates fast,
the hard term then reshapes scores it relied on). Not retried with other β —
diminishing returns given L3/L4's pattern.

## Dead ends / do-not-retry list

- **Two-term CE (L5) as configured** (β=1, 5+5, mixture val): dominated.
  If revisited, fix model selection first (e.g., select on hard-only val AP,
  or val MRR), and sweep β — but expect the L3/L4 pattern to persist.
- **Pushing hard_frac higher for MRR** (L4): moves historical, not MRR.
- **Expecting a loss change to close the recency gap in deployment MRR**:
  five diverse configs land in 0.046–0.080 vs recency 0.354. The information
  "which exact partner was most recent" appears not to survive compression
  into a 100-d node memory + 20-edge attention at full-ranking precision.
  This is the architecture/feature frontier — see `04-open-ideas.md`.

## Bugs and pitfalls hit during the campaign (so you don't re-hit them)

- **Model-selection leak-by-mismatch:** early stopping on random-negative val
  AP while training on hard negatives silently reverts the model toward the
  random-NS optimum. Always match the val negative distribution (or use a
  deliberate metric like val MRR).
- The bursty toy graph (`tests/test_tgn_negatives.py:bursty_graph`) exists
  because the planted-pair toys have NO recency signal — a hard-negative loss
  is unlearnable on them. Smoke thresholds were calibrated on measured curves
  (hard-CE crosses val AP 0.6 at ~epoch 6 on the 247-event bursty stream).
- In the bursty toy, sources have one partner → the src-hard branch falls
  back to uniform; tests that assert "hard negatives are past pairs" must pin
  `src_frac=0.0`.
- TGN generally needs 2–3 warm-up epochs before val metrics move (memory
  starts as noise); don't judge a config on epoch-0/1 numbers.
