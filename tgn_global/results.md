# tgn_global — results (2026-08-14; protocol correction 2026-08-16)

> **Campaign 2 (same day, later) superseded parts of this file — see
> "Campaign 2" section at the bottom. Headline updates: recency heuristic
> beaten on all four ranking metrics (G2: MRR 0.3689, h@1 0.3059,
> h@10 0.4787, h@100 0.6011), inductive raised to 0.6601 (W2).
> (G2 ranking numbers corrected 2026-08-16: first report read the ranking
> JSON mid-write at 3/5 seeds — tgn.ranking flushes per seed.)**

> **2026-08-16 PROTOCOL CORRECTION: every AU-ROC in this file uses the
> LEGACY protocol, which deviates from the DGB paper. The paper-faithful
> reruns are in the "DGB protocol correction" section at the very bottom;
> under the corrected metric W2's inductive is 0.7298 and the "0.60 roof"
> narrative changes materially. Deployment-ranking numbers (MRR/hits@k)
> are unaffected. Legacy AU-ROCs remain valid as internal comparisons
> (same frozen pools across all models).**

**TL;DR.** The ~0.60 inductive AU-ROC roof is broken: **0.6163 ± 0.0056**
(5 seeds, frozen protocol) by giving the TGN *global-in-time* awareness —
explicit node-activity features plus training negatives shaped like the
inductive pool. The user's architectural hypothesis was half right: the
20-recent-edge attention does blind the model to global structure, but the
missing information is global **in time** (total degree, long-horizon
frequency), not global **in hop-space** — a literal 2-hop message-passing
extension changed nothing (clean null). On hits@k, a second arm sets project
bests (hits@1 0.2645, unseen-MRR 0.1648 > recency) but recency's overall
hits@1/hits@10 still stand.

## Headline numbers (5 seeds, frozen evaluation + deployment ranking)

| model | random | historical | inductive (seen/uns) | MRR | h@1 | h@10 | h@100 |
|---|---|---|---|---|---|---|---|
| **inductive arm** `g_headline_ind` (h.1 p.4 **nov.2**, f11, combo) | 0.9373 | 0.8331 | **0.6163 ± .006** (.637/.600) | 0.2381 | 0.1792 | 0.3467 | 0.5255 |
| **MRR arm** `g_f14_mrr` (h.1 p.5, f14, mrr) | 0.9767 | 0.8200 | 0.5766 | 0.3313 | **0.2645** | 0.4558 | 0.5933 |
| score ensemble of the two (z-avg) | 0.9648 | 0.8382 | 0.5953 | 0.3266 | 0.2649 | 0.4399 | 0.5890 |
| old headline (hard-CE) | 0.9679 | **0.8990** | 0.5994 | 0.0456 | 0.0174 | 0.0925 | 0.3397 |
| prev. deploy champion (pfpop_mrr) | 0.9798 | 0.8266 | 0.5767 | **0.3314** | 0.2582 | **0.4622** | **0.5993** |
| recency heuristic | — | — | — | 0.3540 | 0.2777 | 0.4734 | 0.5266 |

Figure: `figures/tgn_global_frontier.png`.

## What produced the break (chain of evidence)

1. **Signal audit** (`scripts/diag_inductive.py`): no single streaming
   signal separates the frozen inductive pools (best 0.59); 50% of
   inductive negatives are already-active-in-test pairs on which pair
   signals invert.
2. **Honest oracle** (`scripts/oracle_daysplit.py`): an MLP over 12 cheap
   streaming features, trained on the FIRST half of test days, scores
   **0.688** on the second half (0.722 with every train-set pair excluded
   from eval). The roof was never informational. Ablation: node
   frequency/recency + degrees carry it (0.676 of it); common-neighbor /
   2-hop structure adds +0.011 — global-in-time, not global-in-space.
3. **Features alone failed** (A1: ind 0.5585): the CE loss never asks the
   inductive question, so the head spends the features on historical/MRR.
4. **Novelty negatives** (`--nov-frac`: negatives drawn from pairs whose
   FIRST appearance is within 30 days — the streaming analogue of the
   inductive pool) make the loss ask it: ind 0.603–0.618 across nov .1–.3,
   with BOTH strata ≥ 0.60 for the first time in the project (every prior
   config mirror-traded seen vs unseen inductive).
5. **The residual gap to 0.69 is single-scalar calibration, not capacity**:
   pushing novel-active pairs down necessarily pushes novel POSITIVES down
   (random −0.04, MRR −0.09 at nov .2), and the 5-seed ensemble of the two
   arms interpolates inductive (0.595) instead of keeping the best — same
   scoring-rule geometry as `tgn_improvement/07`.

## Negative results (reported with equal weight)

- **Two-hop temporal attention (`--n-layers 2`): null.** On the exact
  pfpop_mrr recipe, every metric within 2-seed noise (ind 0.5725 vs 0.5767,
  MRR 0.3237 vs 0.3314). The "longer message passing" form of the
  hypothesis is rejected for this graph — as the ablation predicted.
- **hits@1/hits@10 vs recency: still open.** f14 (global + sharp dt-bucket
  features) sets the learned bests (h@1 0.2645 ± 0.003, unseen-MRR
  0.1648 ± 0.001 > recency's 0.1572, h@100 0.5933 ≫ 0.5266) but overall
  h@1 0.2645 < 0.2777 and h@10 0.4558 < 0.4734. The residual is the
  partner-ordering precision recency gets for free. Lexicographic
  recency-then-model remains the only scorer beating recency outright
  (0.368, `tgn_improvement/06`).
- **Common-neighbor / Adamic-Adar / 2-hop-path features: not worth the
  plumbing** (+0.011 oracle) — deliberately not implemented in the model.

## Bug found in prior work (affects tgn_improvement conclusions)

`train_one`'s `hard_sampler` (two-term loss path) was never advanced with
`observe_day`: its pools stayed empty and all "all-hard" negatives silently
fell back to uniform. Every two-term/hinge result (L5 in 02; pf_2t and
hinge rows in 06) tested nothing — their "dominated" verdicts are void.
Fixed in this campaign (one line); not re-run since the mixture recipe
superseded the two-term design, but any revisit of 07's "two-calibration
heads" idea should start from the fixed code.

## Artifacts

| what | where |
|---|---|
| rationale / log / this write-up | `tgn_global/{rationale,experiment_log,results}.md` |
| diagnosis + oracle scripts & JSONs | `tgn_global/scripts/`, `tgn_global/{diag_inductive,oracle_ceiling,oracle_daysplit}.json`, `diag_features.npz` |
| headline models (5 seeds × 2 arms) | `tgn_global/figures/g_headline_ind/`, `g_f14_mrr/` (checkpoints gitignored, train logs committed) |
| headline results/rankings | `tgn_global/figures/g_{headline_ind,f14_mrr}_{results,ranking}.json` |
| ensemble diagnostic | `tgn_global/figures/g_ensemble_mrr_ind.json` |
| scoping runs (2 seeds each) | `tgn_global/figures/g_feat11*`, `g_f11_nov*`, `g_twohop*` |
| summary figure | `tgn_global/figures/tgn_global_frontier.png` |
| code | `src/tgn/recency.py` (FEAT_DIM_GLOBAL=11 / +buckets=14), `train.py` (`--nov-frac/--nov-window`, `--n-layers`), `model.py` (two-hop embed) |
| tests | `tests/test_tgn_global_feats.py` (7), `tests/test_tgn_twohop.py` (5); suite 85 |

## Reproduction

```bash
UVP='UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv'  # see tgn_improvement/05
P=/data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet

# inductive arm (roof-breaker)
$UVP uv run --group ml python -m tgn.train --parquet $P --seeds 0 1 2 3 4 \
  --out tgn_global/figures/g_headline_ind --loss ce --n-neg 5 \
  --hard-frac 0.1 --pop-frac 0.4 --nov-frac 0.2 \
  --pair-feat --pair-feat-dim 11 --select combo

# MRR arm (deployment)
$UVP uv run --group ml python -m tgn.train --parquet $P --seeds 0 1 2 3 4 \
  --out tgn_global/figures/g_f14_mrr --loss ce --n-neg 5 \
  --hard-frac 0.1 --pop-frac 0.5 --pair-feat --pair-feat-dim 14 --select mrr

# frozen eval + ranking (per arm), then the ensemble diagnostic
$UVP uv run --group ml python -m tgn.evaluate --parquet $P --models <dir> --out <json>
$UVP uv run --group ml python -m tgn.ranking  --parquet $P --models <dir> --out <json>
$UVP uv run --group ml python -m tgn.ensemble --parquet $P \
  --models-a tgn_global/figures/g_f14_mrr --models-b tgn_global/figures/g_headline_ind \
  --out tgn_global/figures/g_ensemble_mrr_ind.json
```

## Open next steps (superseded by Campaign 2 for items 1 and 3 — see below)

1. **hits@1**: the remaining 0.013 to recency lives in partner-ordering
   precision. → SOLVED in Campaign 2 by the rec_rank / pair-depth features.
2. **Two-calibration heads** (07's proposal) with the FIXED two-term
   sampler: one trunk, one head per reference distribution — now actually
   testable, since the old two-term evidence is void (see bug note).
3. The 0.62 → 0.69 inductive gap → mostly closed in Campaign 2 (0.6601).

---

# Campaign 2 (2026-08-14 PM): widening the receptive field

**Directive:** widen the receptive field in time and in graph space.
**Outcome: both open barriers fell.** The recency heuristic is beaten
outright on every ranking metric by a single 5-seed model, and inductive
AU-ROC reached 0.6601 — 96% of the measured feature-family oracle.

## Headline (5 seeds unless noted; recency: MRR .3540, h@1 .2777, h@10 .4734, h@100 .5266)

| model | rand | hist | inductive | MRR | h@1 | h@10 | h@100 |
|---|---|---|---|---|---|---|---|
| **G2 wide + two-hop** (deploy champion) | 0.9789 | 0.8423 | 0.6215 ± .003 | **0.3689 ± .004** | **0.3059 ± .004** | **0.4787 ± .004** | **0.6011 ± .004** |
| W1 wide feats, 1-hop | 0.9782 | 0.8402 | 0.6200 ± .007 | 0.3658 ± .004 | 0.3045 | 0.4731 | 0.5914 |
| **W2 wide + nov .2** (inductive champion) | 0.9465 | 0.8271 | **0.6601 ± .013** | 0.3381 | 0.2817 | 0.4368 | 0.5676 |
| W3 wide + nov .1 (balanced) | 0.9641 | 0.8263 | 0.6534 ± .008 | 0.3525 | 0.2930 | 0.4555 | 0.5873 |
| G3 two-hop + nov .1 (2 seeds) | 0.9696 | 0.8404 | 0.6584 | 0.3530 | 0.2935 | 0.4537 | 0.5858 |

G2 recipe: `--loss ce --n-neg 5 --hard-frac 0.1 --pop-frac 0.5 --pair-feat
--pair-feat-dim 23 --select mrr --n-layers 2`. W2: `--pop-frac 0.4
--nov-frac 0.2 --select combo` (1-hop). Figure:
`figures/tgn_global_frontier2.png`.

## What did it (and what didn't)

1. **Time-deep features (the whole story).** Phase-0 audit: pair_freq
   (0.606) and pair_age (0.598) are the strongest single inductive signals
   ever measured here — and the model previously saw only the pair's LAST
   day. `FEAT_DIM_WIDE = 23` adds pair frequency/age, 7/90-day windows,
   directed distinct-partner degrees, and **rec_rank** — the candidate's
   recency rank among the source's partners, i.e. the recency heuristic's
   own ordering statistic, handed to the head to refine. Day-split oracle:
   0.668 → 0.689. Model effect: MRR +0.035, h@1 +0.040 over the f14 arm,
   and the novelty-vs-MRR trade-off nearly vanished (W2 keeps MRR 0.338
   where the narrow-feature arm fell to 0.238).
2. **Two-hop attention reverses to a win — but only on wide features.**
   Null on narrow features (campaign 1), +0.005–0.009 on every ranking
   metric on wide ones (G2 vs W1, consistent across seeds). Reading:
   with node-level statistics saturated by features, the second hop's
   marginal information (who my counterparties deal with) finally binds.
   Graph-space widening contributes only after time-depth is granted.
3. **Widening the attention window is a dead end** (measured twice):
   stratified half-recent/half-history sampling (S1) and k=64 (G1) are
   both null-to-negative. The missing time-depth is countable statistics,
   not attendable events — attention cannot count.
4. **Day-level graph context is a trap**: it single-handedly DROPS the
   day-split oracle 0.689 → 0.605 (memorizes day regimes). Rejected at
   diagnosis, never trained.

## Honest caveats

- G2's h@10 edge over recency is 0.0054 with σ .005 — call it "at parity
  or slightly above"; MRR/h@1/h@100 are clear wins (≥ 3σ).
- The seen/unseen inductive strata still trade (W1: .763/.521 vs
  W2: .758/.592); novelty negatives remain the only lever for the unseen
  stratum and still cost ~0.03 random AU-ROC.
- Old hard-CE's historical 0.899 remains unmatched by any high-MRR model
  (best here 0.842–0.845); that trade-off is unchanged from 07's geometry.
- G3 (two-hop + novelty) was only run at 2 seeds; its numbers suggest
  two-hop's ranking gain and novelty's inductive gain do NOT stack
  linearly (MRR 0.3530 ≈ W3's, not W3 + 0.005).

## Campaign-2 artifacts

| what | where |
|---|---|
| widened signal audit + oracle | `scripts/diag_widen.py`, `diag_widen.json`, `diag_widen.npz` |
| 5-seed arms | `figures/{w1_wide_mrr,w2_wide_ind,w3_wide_nov10,g2_twohop_w1}/` + `_results/_ranking` JSONs |
| 2-seed probes | `figures/{s1_strat_f14,g1_k64_f14,g3_twohop_nov10}*` |
| figure | `figures/tgn_global_frontier2.png` (`scripts/plot_frontier2.py`) |
| code | `recency.py` FEAT_DIM_WIDE=23; `neighbors.py` strat mode; `--nbr-mode` in train/evaluate/ranking |
| tests | `tests/test_tgn_wide.py` (5); suite 90 |

NOTE for reuse: models trained with `--nbr-mode strat` or nonstandard `--k`
must be evaluated with the same flags (the store is rebuilt at eval; there
is no checkpoint record of it).

## Remaining open questions

1. Two-calibration heads (unchanged; the one lever not yet tried for
   holding hist 0.90 and MRR 0.37 in one artifact).
2. The unseen-inductive stratum (0.59 vs seen 0.76): novelty negatives
   restricted to *seen-in-test-like* pairs might lift unseen without the
   random-AU-ROC cost.
3. Whether G2 + novelty at 5 seeds (G3 extended) lands a single model at
   ind ≥ 0.65 AND MRR ≥ 0.355 — the 2-seed probe says close but not free.

---

# DGB protocol correction (2026-08-16)

A review of the DGB paper's code showed our AU-ROC protocol (built for the
TGAT baseline and frozen since) deviates from the paper in five ways:
per-day global AUROC instead of **per-batch (200 edges) mean**; historical
pool frozen at the test split instead of **accumulating through test**;
inductive pool drawn from ALL test-only pairs (including future ones)
instead of **test-only pairs observed so far, random-padded when short**;
and random negatives as never-seen uniform pairs instead of
**source-preserving random destinations with only a batch-collision
check**. `src/tgn/evaluate_dgb.py` implements the paper-faithful protocol
(7 property tests); the legacy protocol is untouched — its numbers remain
valid as internal comparisons on identical frozen pools.

## Corrected AU-ROC (per-batch mean, 5 seeds; pad_frac inductive = 0.06)

| model | random | historical | inductive |
|---|---|---|---|
| TGN baseline (BCE) | 0.9307 ± .004 | 0.6316 ± .016 | 0.4748 ± .010 |
| TGN hard-CE (old headline) | 0.8746 ± .005 | **0.8818 ± .003** | 0.6754 ± .011 |
| pair-recency pfpop_mrr | 0.9294 ± .005 | 0.8045 ± .010 | 0.5496 ± .011 |
| f14 (c1 MRR arm) | 0.9298 ± .005 | 0.7961 ± .013 | 0.5210 ± .012 |
| c1 inductive arm | 0.8941 ± .008 | 0.8239 ± .010 | 0.7104 ± .010 |
| W1 wide | 0.9269 ± .006 | 0.8192 ± .006 | 0.5878 ± .017 |
| G2 wide + two-hop | 0.9285 ± .005 | 0.8217 ± .003 | 0.5957 ± .010 |
| W3 wide + nov .1 | 0.9189 ± .008 | 0.8156 ± .005 | 0.6933 ± .013 |
| **W2 wide + nov .2** | 0.9024 ± .006 | 0.8195 ± .008 | **0.7298 ± .020** |
| EdgeBank inf / tw | 0.763 / 0.721 | 0.265 / 0.614 | 0.295 / 0.252 |

Figure: `figures/tgn_dgb_correction.png`. Raw JSONs (incl. per-seed and
global-AUROC variants): `figures/dgb/*_dgb.json`.

## What the correction changes — honest revision of conclusions

1. **The "0.60 inductive roof" was partly a protocol artifact.** Under the
   paper's definition (negatives = test-only pairs *already observed*,
   history accumulating), the old hard-CE model already scores 0.6754 —
   there never was a 0.60 roof under the paper's metric. The legacy
   inductive pool included future never-yet-active pairs, which made the
   task closer to "new-pair timing" and capped everything near 0.6.
2. **The novelty-negative mechanism is vindicated, more strongly than
   before.** Corrected inductive rewards exactly what novelty negatives
   teach (demoting recently-first-seen pairs that are off today): W2 leads
   at 0.7298, +0.054 over hard-CE, and the novelty dial is worth +0.13-0.14
   over its no-novelty twins (W1/G2 ≈ 0.59). Ordering among the campaign
   arms is preserved; margins widen.
3. **MRR-oriented mixtures pay an inductive price that legacy hid**:
   pfpop/f14 fall to 0.52-0.55 (they promote recurring pairs — the
   corrected negatives). The G2 deployment champion sits mid-pack (0.596)
   on corrected inductive; W3 (nov .1) is the balanced pick at 0.6933
   inductive with near-champion ranking numbers.
4. **Deployment ranking (MRR/hits@k) is untouched** by this correction —
   the recency-heuristic win stands as reported.
5. EdgeBank's corrected inductive (0.25-0.30) matches the paper's
   qualitative finding that memorization baselines collapse under harder
   negatives — a good sampler-integrity check.

Caveat: per-batch AUROC with per-batch negatives is noisier per seed and
sensitive to the padding fraction (0.06 here, concentrated in the earliest
test batches, where the inductive pool is nearly empty by construction).

---

# Post-correction studies (2026-08-16)

Full details in `experiment_log.md`; design rationale in
`design-notes.md`. Summary:

1. **Validation-span EdgeBank**: the inductive inversion is structural and
   visible at validation (val 0.300 vs test 0.295) — no val/test surprise;
   the joint three-column report is what defeats sign-flip gaming.
2. **3:1 negative ratio**: random/historical unchanged (AUROC prevalence
   invariance); inductive inflates +0.025–0.042 purely because padding
   doubles (6.0% → 13.1%, predicted exactly from pool sizes). 1:1 is the
   conservative setting.
3. **W2 at a fixed 50 epochs**: the stopping criterion peaks at epochs
   3–11 (adaptive stop reproduced), but the FINAL-epoch model is a new
   inductive record — **DGB 0.8240 ± .004** with historical 0.8536 — at
   −0.08 MRR. The epoch count is a calibration dial on the
   paired-vs-ranking frontier; both checkpoints kept
   (`figures/w2_fixed50{,_last}/`), curves with the criterion in
   `figures/w2_fixed50_curves.png`.
4. **TGAT under DGB**: 0.9260 / 0.6805 / 0.4645 — below chance on
   corrected inductive, the mild form of EdgeBank's memorization failure.
5. **G2 ranking numbers corrected** (0.3707 → 0.3689 etc.): the first
   readout caught the per-seed-flushed JSON at 3/5 seeds; deterministic
   re-run verified. Conclusion (beats recency on all four) unchanged.

---

# Additional test modes and the EdgeBank investigation (2026-08-16)

Full chronology in `experiment_log.md`; code in `src/tgn/evaluate_dgb.py`
(`dgb_negatives` strategies `inductive` / `historical` / `random` /
`test_recurring`; `per_batch_auroc(pos_mask=...)`; `edgebank_scores_dgb(
freeze_memory=...)`); runners `scripts/run_dgb_modes.py`,
`scripts/run_sym_fdr_npv.py`; raw results `figures/dgb/modes_dgb.json`,
`figures/dgb/tgat_modes.json`, `figures/dgb/sym_fdr_npv.json`.

## Why two more modes were needed

The standard inductive setting conditions only the NEGATIVES on prior
observation ("test-only pairs already observed"). Two consequences were
measured: (1) a memorization scorer's AUROC becomes a pure base-rate
identity — for a binary scorer AUROC = ½ + (p − q)/2, where p/q are the
positive/negative in-memory rates — inverted (0.295) when memory
accumulates through test (the DGB paper's hardcoded `learn_through_time=
True`, verified in their source) and inflated (0.704) when memory is
frozen at train/val; (2) positives remained a MIX of old and new pairs, so
any model could bank partial credit on the old ones without new-pair
skill. The modes below remove each artifact in turn.

## Methodology

Both modes keep EVERYTHING else identical to the paper-faithful protocol:
200-event chronological batches, history accumulating through test,
strictly-past pools, batch-collision exclusion, 1:1 sampling before any
masking, random padding on shortfall, per-batch mean AUROC.

**`inductive_sym` (symmetric inductive).** Negatives: standard inductive
sampling (pairs never seen before the test span, observed earlier in the
span, not positive in the batch). Positives: restricted to test events
whose pair was never seen before the test span (59.3% of events). Metric:
per batch, kept positives vs ALL that batch's negatives (batches with no
kept positives skipped). Pool prevalence 0.372. Question answered: "can
the model tell when a genuinely new relationship is active vs dormant,
with no help from familiar pairs on either side?"

**`test_recurring`.** Negatives: pairs active earlier in the test span,
with the never-in-train/val filter DROPPED (train-seen pairs eligible;
pool is a superset of the inductive pool, padding 3.7% vs 6.0%).
Positives: unchanged (all test events). Question: "among relationships
active during test regardless of origin, which fire today?"

**Controls.** Frozen-memory EdgeBank (memory stopped at train/val) is
included as the built-in null: on `inductive_sym` every comparison is a
tie, so it must score exactly 0.500 if the mode contains no free signal
for memorization in either direction.

**FDR / NPV procedure (on `inductive_sym`).** Per model and seed: build
the same construction on the VALIDATION span (positives = val events whose
pair was never seen in train; negatives = val-span inductive sampling);
choose the single threshold maximizing Youden's J there; freeze it; on the
test-span pools report FDR = FP/(FP+TP) and NPV = TN/(TN+FN). EdgeBank is
evaluated at its one non-trivial rule (flag = seen). AUROC in that table
is pooled over the masked set (within 0.005 of the per-batch mean
everywhere).

## Results (5 seeds)

| model | inductive_sym AUROC | test_recurring AUROC | std inductive |
|---|---|---|---|
| **W2 @50 epochs** | **0.7813 ± .004** | **0.6918 ± .002** | 0.8240 |
| TGN hard-CE | 0.6765 ± .010 | 0.6248 ± .007 | 0.6754 |
| W2 (early stop) | 0.6519 ± .022 | 0.6166 ± .012 | 0.7298 |
| G2 wide + two-hop | 0.5001 ± .012 | 0.5118 ± .010 | 0.5957 |
| pfpop-mrr | 0.4955 ± .013 | 0.5005 ± .010 | 0.5496 |
| TGAT | 0.4022 ± .006 | 0.4460 ± .006 | 0.4645 |
| EdgeBank ∞ (accumulating) | 0.1313 | 0.2829 | 0.2946 |
| EdgeBank ∞ (frozen, control) | **0.5000** | 0.5353 | 0.7036 |

| model | AUROC (pooled) | FDR @ val-Youden | NPV @ val-Youden |
|---|---|---|---|
| W2 @50 epochs | 0.7774 ± .004 | 0.3362 ± .015 | 0.7941 ± .004 |
| TGN hard-CE | 0.6741 ± .011 | 0.2350 ± .024 | 0.7037 ± .002 |
| W2 (early stop) | 0.6481 ± .023 | 0.1932 ± .018 | 0.7036 ± .002 |
| G2 wide + two-hop | 0.4957 ± .012 | 0.3942 ± .056 | 0.6717 ± .007 |
| pfpop-mrr | 0.4890 ± .013 | 0.4119 ± .068 | 0.6669 ± .008 |
| TGAT | 0.4000 ± .006 | 0.6229 | 0.8688 (tiny flag volume, inverted score) |
| EdgeBank ∞ | 0.1330 | 0.8852 | 0.1134 |
| EdgeBank frozen | 0.5000 | 0.000 (VACUOUS: flags nothing) | 0.6279 (= 1 − prevalence) |

## Findings

1. **The control behaves exactly as designed**: frozen EdgeBank = 0.5000
   on `inductive_sym`. The accumulating variant's 0.1313 is the base-rate
   identity again (p ≈ 0.206 test-repeat positives, q ≈ 0.940).
2. **The symmetric mode unmasks a split the standard protocol hid.** The
   deployment-oriented models (G2, pfpop) are at exact chance on
   new-relationship timing; their standard-inductive 0.55–0.60 was carried
   by seen positives. Genuine new-pair skill lives only in novelty
   training (W2@50 0.7813) and, notably, hard-CE (0.6765). TGAT sits below
   chance (0.40): the graded form of memorization inversion.
3. `test_recurring` preserves the ordering at lower levels — the novelty
   models' edge is not an artifact of the never-in-train filter.
4. **No model meets the program bar (FDR < 0.125 AND NPV > 0.875) at a
   single validation-chosen threshold on this pool.** The joint corner at
   prevalence 0.372 is (FPR 0.066, TPR 0.774) ≈ AUROC 0.945 binormal; the
   best available is 0.777. FDR alone is not comparable across models
   without flag volume (W2-early's lower FDR reflects a more conservative
   threshold, not more skill); read the (FDR, NPV) pair. A three-zone
   abstention deployment on calibrated posteriors remains the route to
   the bar; model quality then sets the auto-decided share.
5. **EdgeBank inversion, verified on three fronts** (was previously
   over-asserted as "intended"): frozen-memory ablation isolates the cause
   entirely to memory accumulation; the DGB source hardcodes that
   accumulation; the paper's own appendix (Tables 6/10) reports EdgeBank∞
   inductive AUROC below 0.5 on 12 of 13 datasets. Whether the
   below-chance direction was intended by the authors is unknowable from
   the paper; that it is a reproducible consequence of their design is
   established. Reverse-direction pairs (1.7%) and known endpoints (30.6%)
   among inductive negatives do not register in a directed-pair memory.
