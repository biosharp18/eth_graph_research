# TGN results — daily flow prediction on the OFAC k=4 graph

**Date:** 2026-08-13. Companion to `TASK.md`; protocol and artifacts per its §2–§5.
Artifacts: `figures/tgn_results.json`, `figures/tgn_ranking.json`,
`figures/tgn_models/train_log.json`, `figures/tgn_training_curves.png`,
`figures/tgn_vs_tgat_bar.png`. TGAT/EdgeBank references from
`figures/tgat_results.json` (recomputed EdgeBank values matched exactly,
confirming the identical evaluation pipeline).

## Headline: the hypothesis did not hold

TASK.md §1 asked whether a *learned* per-node memory (TGN) closes the gap
between TGAT and the recency heuristic, especially under historical negative
sampling. **It does not.** TGN improves on TGAT only where negatives are easy
(random NS), is *worse than TGAT and statistically indistinguishable from
EdgeBank_tw* under historical NS, and closes almost none of the deployment-
ranking gap to the recency heuristic (recency MRR 0.354 vs TGN 0.066 — still
a ~5.4× advantage for the heuristic, barely changed from TGAT's ~6×).

## Link prediction (test AU-ROC / AP, mean ± std over 5 seeds, 1 neg/pos)

| NS strategy | EdgeBank_inf | EdgeBank_tw | TGAT | TGN |
|---|---|---|---|---|
| random AU-ROC | 0.7649 | 0.7221 | 0.9559 ± 0.0032 | **0.9640 ± 0.0018** |
| random AP | 0.7649 | 0.7221 | 0.9407 ± 0.0059 | **0.9518 ± 0.0049** |
| historical AU-ROC | 0.2649 | 0.6496 | **0.7045 ± 0.0090** | 0.6505 ± 0.0154 |
| historical AP | 0.4186 | 0.6128 | **0.6529 ± 0.0086** | 0.5941 ± 0.0183 |
| inductive AU-ROC | 0.5128 | 0.4700 | 0.5513 ± 0.0065 | 0.5534 ± 0.0079 |
| inductive AP | 0.5066 | 0.4860 | 0.5436 ± 0.0095 | 0.5346 ± 0.0133 |

Seen/unseen-pair AU-ROC strata (TGN, with TGAT in parentheses):

| NS strategy | seen | unseen |
|---|---|---|
| random | 0.9790 ± 0.0020 (0.9731) | 0.9541 ± 0.0026 (0.9446) |
| historical | 0.7497 ± 0.0131 (0.7854) | 0.5821 ± 0.0174 (0.6491) |
| inductive | 0.6603 ± 0.0073 (0.6475) | 0.4790 ± 0.0079 (0.4842) |

TGN loses to TGAT on *both* strata under historical NS — the memory does not
even specialize on recurring pairs when the negatives are themselves
previously-seen pairs.

## Amount head (log10-USD on test positives)

| model | RMSE | RMSE seen | RMSE unseen | MAE |
|---|---|---|---|---|
| TGN | 1.6255 ± 0.0067 | 0.6526 ± 0.0050 | 2.0411 ± 0.0101 | 0.9818 ± 0.0064 |
| TGAT | 1.6361 ± 0.0076 | 0.6212 ± 0.0022 | 2.0620 ± 0.0102 | 0.9654 ± 0.0038 |
| persistence | 1.9457 | 0.7519 | — | 1.1706 |
| train median | 2.1116 | — | — | 1.3765 |

Same picture as TGAT: both models beat persistence and the median overall and
on seen pairs; TGN ≈ TGAT (slightly better unseen, slightly worse seen).
Amounts are not where the architectures differ.

## Deployment-style ranking (6,176 test events × 11,812 candidates, filtered, tie-aware)

| metric | TGN (5 seeds) | recency heuristic |
|---|---|---|
| MRR | 0.0657 (seen 0.1071 / unseen 0.0373) | **0.3540** (seen 0.6404 / unseen 0.1572) |
| hits@1 | 0.0281 (0.0513 / 0.0122) | **0.2777** (0.5016 / 0.1238) |
| hits@10 | 0.1302 (0.2085 / 0.0764) | **0.4734** (0.8665 / 0.2033) |
| hits@100 | 0.4418 (0.5963 / 0.3356) | **0.5266** (0.9921 / 0.2066) |

Recency = score each candidate destination by how recently the (source,
candidate) pair transacted, unseen pairs tied at the bottom. On seen pairs it
is devastatingly strong (hits@100 = 0.99: almost every recurring destination
is in the source's recent partner list). TGN's only relative bright spot is
unseen-pair hits@100 (0.34 vs 0.21) — embedding generalization helps in the
tail, but far too little to matter for MRR.

## Training behavior

Best val AP per seed: 0.9472 / 0.9420 / 0.9454 / 0.9449 / 0.9472 at
38 / 33 / 50 / 42 / 50 epochs. Early stopping (patience 5) fired for 3 of 5
seeds, and the two 50-epoch seeds were flat over their last ~5 epochs
(`figures/tgn_training_curves.png`), so no longer run was needed — unlike
TGAT, which was still climbing at 50 epochs. Every TGN seed's val AP exceeds
every TGAT seed's (0.9401–0.9446): **on the validation objective (random
negatives) the learned memory genuinely helps** — which is exactly why it
fails to help where it matters (below).

## Interpretation

1. **The memory learned recurrence, but recurrence-blind training can't rank
   recurrences.** Training and validation both use random negatives, so the
   easiest signal a memory can encode is "this node is active / these two
   nodes have history at all" — enough to push random-NS AU-ROC from 0.956 to
   0.964. Under historical NS, both the positive and the negative are
   previously-seen pairs, and that coarse activity signal cancels out: TGN
   drops to EdgeBank_tw's level (0.6505 vs 0.6496), *below* TGAT (0.7045).
   TGAT's attention over the 20 most recent neighbor edges apparently
   preserves finer pair-recency detail than a single GRU-compressed node
   vector: compressing each node's history into one 100-d state loses
   exactly the who-and-when specificity that historical negatives probe.
2. **Deployment ranking is still owned by pair recency.** The heuristic's 0.354
   MRR vs TGN's 0.066 (and TGAT's ~0.06 from the earlier glimpse) means
   neither architecture, trained with 1 random negative per positive, learns
   anything close to "rank the destination this source paid most recently at
   the top of 11,812 candidates."
3. **Verdict on §1's hypothesis: rejected.** A learned per-node memory does
   not close the TGAT↔recency gap; on the hard metrics it is neutral-to-worse.
   EdgeBank's degenerate memory is a better recurrence detector than the GRU
   memory precisely because it stores pair-level, not node-level, state.
4. **What this points to (deferred per TASK.md §7):** the consistent failure
   mode across TGAT and TGN is the *training signal*, not model capacity —
   hard-negative (historical) training and/or explicit pair-recency features
   are the deferred items this result most urgently motivates. A pair-level
   memory (edge-conditioned rather than node-conditioned) is the
   architectural variant the EdgeBank comparison suggests.

## Reproduction

```
UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml python -m tgn.train \
  --parquet /data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet \
  --seeds 0 1 2 3 4 --out figures/tgn_models          # ~40 min/seed on one A100
... python -m tgn.evaluate --models figures/tgn_models --out figures/tgn_results.json
... python -m tgn.ranking  --models figures/tgn_models --out figures/tgn_ranking.json
... python -m tgat.plot_training figures/tgn_models/train_log.json figures/tgn_training_curves.png
... python -m tgn.plot_compare figures/tgat_results.json figures/tgn_results.json figures/tgn_vs_tgat_bar.png
```

Implementation notes (details in `docs/superpowers/plans/2026-08-13-tgn-daily-flow.md`):
day-as-timestep memory (one GRU step per node per day on mean-aggregated
messages, identity message function + direction flag), one temporal-attention
layer reusing the TGAT store/encoder, one-day-lag differentiable memory during
training (official-TGN scheme at day granularity), streaming eval that scores
each day with strictly-pre-day memory and advances on observed positives only.
Leakage property tests in `tests/test_tgn_streaming.py`; suite: 53 passed.

---

# Addendum (2026-08-13/14): loss redesign — the training signal WAS the bottleneck

Follow-up to the interpretation above (§"Interpretation" pt. 4): we redesigned
the loss, holding the TGN architecture and the entire frozen evaluation
protocol fixed. **Result: the same TGN, trained with a ranking loss over
partly-hard negatives, goes from losing to TGAT under historical NS to beating
everything by a wide margin: historical AU-ROC 0.6505 → 0.8990 ± 0.0027.**
Figure: `figures/tgn_loss_bar.png`; artifacts: `figures/tgn_hard_results.json`,
`figures/tgn_hard_ranking.json`, `figures/tgn_hard_models/train_log.json`,
scoping JSONs under `figures/tgn_loss_l{1..5}/`.

## Loss designs tested

All share the unchanged Huber amount term (λ=1) and the frozen eval. Negatives
are sampled streaming-safe (pools reflect strictly-earlier days only; same-day
positive collisions excluded — `StreamingNegatives` in `src/tgn/train.py`,
property tests in `tests/test_tgn_negatives.py`). "Hard" negatives are 50%
same-source past partners / 50% previously-seen global pairs unless noted.

| config | link loss | negatives per positive | random | historical | inductive | MRR | hits@10 |
|---|---|---|---|---|---|---|---|
| L0 baseline (5 seeds) | BCE | 1 uniform | 0.9640 | 0.6505 | 0.5534 | 0.0657 | 0.1302 |
| L1 | BCE | 1, hard 50% | 0.9805 | 0.8798 | 0.5927 | 0.0559 | 0.1069 |
| L2 | softmax CE | 5 uniform | 0.9494 | 0.6030 | 0.5567 | **0.0801** | **0.1562** |
| L3 | softmax CE | 5, hard 50% | 0.9675 | 0.9024 | 0.6067 | 0.0494 | 0.0976 |
| L4 | softmax CE | 5, hard 80% (src-heavy) | 0.9636 | **0.9114** | 0.5890 | 0.0476 | 0.0935 |
| L5 | CE + CE (two-term) | 5 uniform + 5 hard | 0.9338 | 0.7302 | 0.5947 | 0.0511 | 0.1002 |
| **L3 headline (5 seeds)** | softmax CE | 5, hard 50% | **0.9679 ± 0.0018** | **0.8990 ± 0.0027** | **0.5994 ± 0.0072** | 0.0456 | 0.0925 |

(L1–L5 rows are 2-seed scoping runs; the headline row is the full 5-seed
protocol. Early stopping selects on validation AP against negatives drawn from
the *training* distribution — with hard negatives this fires at 11–20 epochs.)

## Headline (L3, 5 seeds) vs the field, AU-ROC

| NS strategy | EdgeBank_tw | TGAT | TGN (old loss) | TGN (hard-CE loss) |
|---|---|---|---|---|
| random | 0.7221 | 0.9559 | 0.9640 | **0.9679 ± 0.0018** |
| historical | 0.6496 | 0.7045 | 0.6505 | **0.8990 ± 0.0027** |
| inductive | 0.4700 | 0.5513 | 0.5534 | **0.5994 ± 0.0072** |

Strata: the historical unseen-pair AU-ROC jumps 0.5821 → 0.8979 (±0.003) —
the old loss's biggest failure mode is simply gone; seen 0.7497 → 0.9009.
AP moves likewise (historical 0.5941 → 0.8938). Amount is unchanged
(RMSE 1.6329 vs 1.6255; still beats persistence 1.9457) — loss changes on the
link head did not disturb the amount head.

## What each ingredient does (the ablation's actual finding)

1. **Hard negatives fix pairwise discrimination, at zero random-NS cost.**
   Adding 50% hard negatives to plain BCE (L1) already lifts historical
   0.65 → 0.88 *and* random 0.964 → 0.981. The model was always capable of
   telling "active today" from "stale pair" — it had just never been asked.
2. **Multi-negative softmax alone shifts a different metric.** L2 (5 uniform
   negatives, CE) is the only config that improves deployment MRR
   (0.0657 → 0.0801, +22%; hits@10 +20%) but leaves historical at baseline.
   Global-calibration pressure and pair-discrimination pressure are distinct
   signals.
3. **They do not currently combine.** In a single softmax the hard negatives
   dominate the gradient (L3/L4 ≈ L1 on MRR); an explicit two-term split (L5)
   underperformed on both fronts — likely because the balanced-mixture early
   stopping favors neither objective and stops very early (11–18 epochs).
4. **Deployment full-candidate ranking remains recency's turf** (0.354 MRR vs
   ≤0.08 for every learned config). Loss shaping moves it ±30%, not 5×. This
   now looks like a feature/architecture limit, not a training-signal limit —
   pair-recency features or a pair-level memory (both TASK.md §7 deferred)
   are the motivated next step, and an L2-style multi-negative CE is the
   right training loss to pair them with.

## Verdict

The §1 hypothesis fails against the *architecture* but the diagnosis behind it
was right: with a task-appropriate loss — sampled-softmax ranking CE with half
the negatives drawn from the evaluation-relevant hard distributions — the
learned memory is worth +0.25 historical AU-ROC over its own random-negative
baseline and +0.19 over TGAT, making TGN (hard-CE) the strongest model in this
project on every DGB negative-sampling strategy. The improvement is verified
on the unmodified evaluation pipeline (EdgeBank recomputation bit-identical
across runs), across 5 seeds, with seed-level std an order of magnitude
smaller than the gains. Honest cost: deployment MRR drops 0.066 → 0.046
(recency-gap unchanged); use the L2 loss variant if deployment ranking is the
target metric.

Reproduction (env prefix and parquet path as above):

```
... python -m tgn.train --seeds 0 1 2 3 4 --out figures/tgn_hard_models \
      --loss ce --n-neg 5 --hard-frac 0.5
... python -m tgn.evaluate --models figures/tgn_hard_models --out figures/tgn_hard_results.json
... python -m tgn.ranking  --models figures/tgn_hard_models --out figures/tgn_hard_ranking.json
... python -m tgn.plot_loss_compare figures/tgat_results.json \
      figures/tgn_results.json figures/tgn_hard_results.json figures/tgn_loss_bar.png
```

(Note: in `figures/tgn_hard_training_curves.png` the loss panel's
"BCE (link)" label is the shared plotting schema's name for the link-loss
term; for this run it is the softmax CE. The AP/AU-ROC panels are measured
against hard-mixture negatives — not comparable to the baseline's
random-negative curves.)
