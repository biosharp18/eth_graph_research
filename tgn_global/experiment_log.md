# tgn_global — experiment log

Chronological, append-only. Every run logged, including failures.
Node: scai3, GPU 0 (A100-40GB), parquet at
`/data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet`,
venv `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv`.

---

## 2026-08-14 — session start

- Read the full `tgn_improvement/` record (01–07). Wrote `rationale.md`.
- Environment check: scai3, GPU 0 free (4 MiB used), node-local parquet
  present, so no rsync needed.
- Next: Phase 0 diagnosis script (no training).

## 2026-08-14 — Phase 0a: inductive-pool signal audit (no training)

`scripts/diag_inductive.py` → `diag_inductive.json`. Streaming-safe signals
scored on the frozen inductive pools, 5 seeds, tie-aware AU-ROC:

| signal | overall | newpos-vs-newneg | seenpos-vs-newneg | allpos-vs-activeneg |
|---|---|---|---|---|
| pair_seen / pair_recency | 0.513 / 0.529 | 0.603 | 1.000 | 0.265 / 0.298 |
| src_freq | **0.591** | 0.545 | 0.770 | **0.547** |
| dst_freq | 0.588 | 0.554 | 0.757 | 0.541 |
| pref_attach | 0.579 | **0.587** | **0.816** | 0.480 |
| src/dst_recency | 0.572 / 0.569 | 0.593 / 0.596 | 0.757 / 0.726 | 0.486 / 0.490 |
| cn / aa | 0.565 / 0.564 | 0.526 | 0.667 | 0.546 |
| two_hop | 0.533 | 0.514 | 0.595 | 0.519 |

Findings:
1. **No signal clears 0.60 overall.** Global-structure signals (cn, aa,
   two-hop) are the *weakest* of the lot on the strata that matter.
2. **50.4% of inductive negatives are already-active-in-test pairs** at
   query time (streaming). On that half every signal is ≤ 0.55, and pair
   signals invert (negatives are 100% streaming-seen, positives only ~half).
3. The "pure new link" stratum (unseen pos vs never-active neg) tops out at
   0.60 (pair_seen — test-time repeats), structure ~0.53. Triadic closure
   is nearly silent in this graph's new-link formation.

Interpretation (provisional): the inductive roof looks substantially
*informational* — the negatives are close to exchangeable with positives
given any past-only signal. Next: oracle-combination ceiling (logistic fit
ON the test pools = upper bound of this feature family) before deciding
whether any training experiment is justified on this axis.

## 2026-08-14 — Phase 0b: oracle combiners overturn the "informational roof" reading

`scripts/oracle_ceiling.py` (fit-on-eval, UPPER BOUND ONLY): logistic 0.686,
MLP-64 0.872 — but the MLP number is memorization (near-unique cn/aa floats
fingerprint rows). Replaced by the honest protocol,
`scripts/oracle_daysplit.py`: fit on inductive-pool rows of the FIRST half of
test days, evaluate on the SECOND half (disjoint rows):

- logistic: **0.532** (the linear direction does not transfer across days)
- MLP-64: **0.688 ± 0.013** — real, generalizing, nonlinear signal
- leak control (drop every late-test row whose (s,d) pair occurred anywhere
  in the training rows, killing pair-fingerprint credit): **0.722**

Feature-subset ablation (same day-split MLP protocol, 5 pools):

| feature set | day-split AU-ROC |
|---|---|
| current model features (pair recency + dst activity, 4) | 0.478 ± 0.017 |
| pair signals only | 0.368 |
| + src/dst frequency and recency (7) | 0.633 ± 0.011 |
| + degrees / pref-attach (8) | 0.676 ± 0.014 |
| all 12 (adds cn, aa, 2-hop) | 0.687 ± 0.019 |
| structure only (cn, aa, 2-hop, pa) | 0.538 |

**Phase-0 conclusions (updating rationale.md's hypothesis):**
1. The 0.6 inductive roof is NOT informational — a 12-feature nonlinear
   combiner clears 0.69 honest / 0.72 leak-controlled. The information the
   models lack is **global-in-time node activity** (total frequency, degree,
   long-horizon activity) and its *nonlinear interaction* with pair novelty
   — exactly what 20-recent-edge attention + a saturating GRU cannot count.
2. **Multi-hop structure is nearly silent** here: cn/aa/2-hop add +0.011 on
   top of activity features. "Global awareness" in this graph is global in
   TIME, not in hop-space. Phase B (2-hop attention) is therefore expected
   to underperform Phase A; it stays in the plan as the direct test of the
   original architectural hypothesis.
3. Single-scalar calibration remains a real constraint (the prior-config
   table shows inductive_seen and inductive_unseen trading in mirror image
   across negative mixtures), so Phase A must pair the new features with a
   training-negative slice that asks the inductive question: negatives drawn
   from *recently-first-seen pairs* ("novelty negatives", mimicking the
   inductive pool's streaming composition).

Prior-config audit (all results JSONs): best inductive ever trained = 0.6022
(`tgn_pf_h10`), and no config's seen/unseen strata are jointly high —
consistent with the calibration mechanism from `tgn_improvement/07`.

## 2026-08-14 — implementation + launches

Code (all property-tested, suite 84 passing, was 73):
- `src/tgn/recency.py`: `FEAT_DIM_GLOBAL = 11` — the 4 existing features +
  [src seen/recency, src freq, dst freq, dst 30-day freq, deg(s), deg(c)],
  streaming with a day-monotonic rolling window. `features`/`features_all`/
  `features_cross` all covered.
- `src/tgn/train.py`: `--nov-frac/--nov-window` novelty negatives — pairs
  whose FIRST appearance is within the window (streaming analogue of the
  inductive NS pool), threaded through val/matched-selection sampling.
- `src/tgn/model.py`: `--n-layers 2` two-hop temporal attention (TGAT
  recursion convention: neighbor embedded at its edge day; (node, day)
  dedup at the hop boundary; memory as leaf state). Checkpoint auto-infers.
- Tests: `tests/test_tgn_global_feats.py` (6), `tests/test_tgn_twohop.py`
  (5, incl. same-day/future leakage properties and a 2-hop-information
  reachability check).

**BUG FOUND (pre-existing, fixed):** `train_one`'s `hard_sampler` (two-term
loss path) was never advanced with `observe_day`, so its pools were
permanently empty and every "all-hard" negative silently fell back to
uniform. Consequence: L5 (02), the pf two-term runs, and the hinge runs in
06 never tested the two-term hypothesis — their "dominated" verdicts are
void. Not re-run yet (mixture recipes are the current frontier); noted for
honesty. Fix is one line + comment in train.py.

Scoping launches (GPU 0, 2 seeds each, out under `tgn_global/figures/`):
- A1 `g_feat11`: ce n5 h.1 p.5 pair-feat-dim 11 select mrr
  (comparator: `tgn_pfpop_mrr` 5-seed — isolates the global features)
- A2 `g_feat11_nov20`: as A1 but pop .4 + nov .2 (novelty negatives)
- B1 `g_twohop`: pfpop_mrr recipe exactly (4-dim features) + n-layers 2
  (isolates longer message passing; comparator the same 5-seed reference)

## 2026-08-14 — Phase A scoping results (2 seeds, frozen eval + ranking)

Reference `tgn_pfpop_mrr` (5 seeds): rand 0.9798, hist 0.8266, ind 0.5767
(seen 0.6603 / unseen 0.5177), MRR 0.3314, h@1 0.2582, h@10 0.4622,
h@100 0.5993. Recency: MRR 0.3540, h@1 0.2777, h@10 0.4734, h@100 0.5266.

| config | rand | hist | ind (seen/uns) | MRR | h@1 | h@10 | h@100 |
|---|---|---|---|---|---|---|---|
| A1 `g_feat11` (feats only) | 0.9723 | 0.8438 | 0.5585 (.628/.510) | 0.3064 | 0.2379 | 0.4294 | 0.5818 |
| A2 `g_feat11_nov20` (+nov .2) | 0.9359 | 0.8245 | **0.6156 (.636/.599)** | 0.2151 | 0.1537 | 0.3318 | 0.5254 |

Readings:
1. **Features alone do not move inductive** (0.5585) — they get spent on the
   training objective (hist +0.017 over reference). The information channel
   exists now, but the loss never asks the inductive question. Exactly the
   calibration mechanism predicted.
2. **Novelty negatives break the record: ind 0.6156** (prior project best
   0.6022, old headline 0.5994), and for the FIRST time both strata are high
   simultaneously (0.636/0.599 — every earlier config mirror-traded them,
   e.g. reference 0.660/0.518). The mechanism works.
3. Price at nov=.2: MRR −0.12, random −0.04. The novelty slice deflates
   recently-first-seen pairs, which are disproportionately the true next
   destinations in deployment ranking. A dial, not a wall (hypothesis).

Follow-up launches (2 seeds each): `g_f11_nov10` (h.1 p.5 nov.1, select
mrr), `g_f11_nov20_cmb` (h.1 p.4 nov.2, select combo), `g_f11_nov30`
(h.1 p.4 nov.3, select mrr — inductive-max end of the frontier).

## 2026-08-14 — Phase B result: two-hop attention is a NULL

`g_twohop` (pfpop_mrr recipe + `--n-layers 2`, 2 seeds): rand 0.9836,
hist 0.8195, ind 0.5725 (seen .660/uns .511), MRR 0.3237, h@1 0.2496,
h@10 0.4602, h@100 0.6078. Every delta vs the 1-hop 5-seed reference is
within 2-seed noise (±0.008). Longer message passing adds nothing on any
axis — as Phase 0 predicted (structural signals ≈ silent). The literal
"longer message passing" form of the architecture hypothesis is rejected;
the "global awareness" form survives via global-in-time activity features
+ novelty negatives (Phase A). Honest verdict, cleanly isolated.

## 2026-08-14 — novelty-dial frontier (2 seeds each)

| config | rand | hist | ind (seen/uns) | MRR | h@1 | h@10 | h@100 |
|---|---|---|---|---|---|---|---|
| ref pfpop_mrr (5s) | .9798 | .8266 | .5767 (.660/.518) | .3314 | .2582 | .4622 | .5993 |
| nov .1 (`g_f11_nov10`) | .9556 | .8240 | .6028 (.635/.578) | .2468 | .1781 | .3770 | .5543 |
| nov .2 (`g_feat11_nov20`) | .9359 | .8245 | .6156 (.636/.599) | .2151 | .1537 | .3318 | .5254 |
| nov .2 combo (`g_f11_nov20_cmb`) | .9366 | .8321 | .6147 (.635/.598) | .2233 | .1601 | .3418 | .5310 |
| nov .3 (`g_f11_nov30`) | .9023 | .8494 | .6184 (.603/.626) | .1983 | .1517 | .2806 | .4508 |

- Inductive saturates ≈ 0.615–0.618 at nov ≥ .2 (oracle says 0.69 is in the
  features; the single-scalar calibration wall stops the trained model at
  ~0.62 — pushing novel-active pairs down inevitably pushes new-pair
  POSITIVES down too, hence the monotone random/MRR cost).
- combo selection ≥ mrr selection at same mixture (hist +0.008, MRR +0.008).
- Chosen headline: **nov .2 + combo** (best inductive per unit MRR cost).

Launched: `g_headline_ind` = h.1 p.4 nov.2, f11, select combo, seeds 0–4;
`g_f14_mrr` = 14-dim (global+sharp-dt-buckets) h.1 p.5 select mrr, 2 seeds
(hits@1 probe aimed at the partner-ordering gap vs recency).
Planned after: 5-seed eval+ranking; `tgn.ensemble` diagnostic of
pfpop_mrr × headline (07 showed z-avg ensembles keep both parents' bests).

## 2026-08-14 — f14 probe (global + sharp dt buckets, 2 seeds): new MRR best

`g_f14_mrr` (h.1 p.5, 14-dim, select mrr): rand 0.9749, hist 0.8077,
ind 0.5812, **MRR 0.3349, h@1 0.2675** (project bests; ref 0.3314/0.2582),
h@10 0.4591, h@100 0.5951, unseen MRR 0.1657 ≥ recency 0.1572. The sharp
dt indicators + global features together do what neither did alone for
ranking. Promoted to 5 seeds (`g_f14_mrr` extended with seeds 2–4;
seeds 0–1 train log backed up to `train_log_seeds01_backup.json` because
tgn.train rewrites the log per run). h@1 still 0.010 short of recency.

## 2026-08-14 — HEADLINE (5 seeds): the inductive roof is broken

`g_headline_ind` = ce n5, h.1 p.4 nov.2, pair-feat-dim 11, select combo:

| metric | value (5 seeds) |
|---|---|
| **inductive AU-ROC** | **0.6163 ± 0.0056** (seen 0.6366 ± 0.0071 / unseen 0.6004 ± 0.0053) |
| historical | 0.8331 ± 0.0091 (seen 0.8884 / unseen 0.7954) |
| random | 0.9373 ± 0.0066 |
| deploy MRR / h@1 / h@10 / h@100 | 0.2381 ± 0.013 / 0.1792 / 0.3467 / 0.5255 |
| amount RMSE | 1.6210 ± 0.0066 (unchanged) |

Old headline: inductive 0.5994 ± 0.0072; best-ever config 0.6022. This is
+0.017 over the old headline with non-overlapping seed bands, and the first
model with BOTH inductive strata ≥ 0.60. Roof cleared by a single scalar
scorer under the frozen protocol. Remaining gap to the 0.69 oracle =
single-scalar calibration cost (measured: random −0.04, MRR −0.09 vs the
MRR arm), not missing information.

## 2026-08-14 — MRR arm at 5 seeds, ensemble, figure, wrap-up

`g_f14_mrr` (5 seeds): rand 0.9767 ± .003, hist 0.8200 ± .015,
ind 0.5766 ± .006, **MRR 0.3313 ± .004, h@1 0.2645 ± .003** (h@1 project
best; MRR ties pfpop_mrr with h@1 +0.006), h@10 0.4558, h@100 0.5933,
unseen MRR 0.1648 ± .001 (> recency 0.1572). Seeds 2–4 trained in a second
invocation; train logs merged (seeds 0–1 from the backup).

Ensemble diagnostic (z-avg f14 × headline_ind, 5 seeds): rand 0.9648,
hist 0.8382, **ind 0.5953 (interpolates — does NOT keep the parent's
0.6163)**, MRR 0.3266, h@1 0.2649. Same conclusion as 07: the novelty
calibration is a property of the score scale; averaging dilutes it. The
roof break needs the novelty-trained scorer itself.

Figure: `figures/tgn_global_frontier.png`
(`scripts/plot_frontier.py`; repo palette/conventions). Full campaign
write-up: `results.md`. Suite: 85 passing.

# Campaign 2 (2026-08-14 PM): widen the receptive field (user directive)

## Phase 0 — widened signal audit (`scripts/diag_widen.py`)

Single-signal AU-ROC on the frozen inductive pools (5 pools):
**pair_freq 0.6057 and pair_age 0.5984 — the two strongest single signals
ever measured on these pools**, and both are pair-history DEPTH, which the
model cannot see (its features carry only the pair's LAST day).
dst_freq90 0.5835; rec_rank and day-context ≈ 0.50 alone (as expected —
rank matters for full ranking, day-context is day-matched away).

Day-split MLP ablations (train early test days, eval late):

| feature set | day-split AU-ROC |
|---|---|
| campaign-1 block (9) | 0.6677 ± 0.006 |
| + pair depth (freq, age) | 0.6853 ± 0.011 |
| + multi-scale windows (7/90/src30) | 0.6853 ± 0.022 |
| + T1 all (18) | **0.6890 ± 0.007** |
| + day-context | 0.6045 (HURTS — memorizes day regimes; G3 dropped) |

Decisions: T1 time-deep features graduate (new FEAT_DIM_WIDE=23 = base4 +
global7 + 9 time-deep + 3 dt-buckets); day-context rejected at diagnosis;
T2 (stratified neighbor sampling) and G1/G2 (k=64, two-hop retest) run as
architecture probes regardless, since they widen what attention sees.

## 2026-08-14 — campaign-2 scoping (2 seeds each): features win, attention nulls

Machinery: `FEAT_DIM_WIDE=23` in recency.py; `NeighborStore(mode="strat")`
(half recent, half evenly-spaced over full history); `--nbr-mode` plumbed
through train/evaluate/ranking (MUST be passed at eval for strat models).
Suite 90 passing.

| config | rand | hist | ind (seen/uns) | MRR | h@1 | h@10 | h@100 |
|---|---|---|---|---|---|---|---|
| **W1** wide f23, h.1 p.5, mrr | 0.9774 | 0.8480 | 0.6169 (.755/.522) | **0.3623** | **0.3013** | 0.4679 | 0.5849 |
| **W2** wide f23 + nov.2, combo | 0.9475 | 0.8318 | **0.6618 (.752/.600)** | 0.3297 | 0.2731 | 0.4286 | 0.5615 |
| S1 strat sampling (f14 recipe) | 0.9744 | 0.8373 | 0.5744 | 0.3109 | 0.2471 | 0.4305 | 0.5835 |
| G1 k=64 (f14 recipe) | 0.9738 | 0.8227 | 0.5769 | 0.3203 | 0.2543 | 0.4485 | 0.5891 |
| ref f14 (5s) | 0.9767 | 0.8200 | 0.5766 | 0.3313 | 0.2645 | 0.4558 | 0.5933 |
| recency | — | — | — | 0.3540 | 0.2777 | 0.4734 | 0.5266 |

Readings (2-seed, confirmation pending):
1. **W1 beats the recency heuristic outright on MRR (+0.008) and hits@1
   (+0.024)** — first learned model to do so — while ALSO sitting at the
   inductive roof (0.617) and hist 0.848. Mechanism: `rec_rank` hands the
   head the heuristic's own ordering statistic to refine, `pair_freq`/`age`
   add the depth recency ignores.
2. **W2: inductive 0.6618**, closing most of the gap to the 0.689 oracle,
   at MRR 0.3297 (the nov-vs-MRR trade nearly vanished with wide features).
3. **Attention-window widening is null on both axes** (S1 strat, G1 k=64):
   the missing time-depth is countable statistics, not attendable events.
Launched: W1/W2 extended to 5 seeds; G2 (two-hop on W1 recipe, 2s);
W3 (nov .1 midpoint, 2s).

## 2026-08-14 — campaign-2 confirmations (5 seeds) + the two-hop reversal

| config | rand | hist | inductive | MRR | h@1 | h@10 | h@100 |
|---|---|---|---|---|---|---|---|
| **W1** wide-mrr (5s) | 0.9782±.001 | 0.8402±.009 | 0.6200±.007 | **0.3658±.004** | **0.3045±.005** | 0.4731±.005 | 0.5914±.006 |
| **W2** wide-ind (5s) | 0.9465±.005 | 0.8271±.005 | **0.6601±.013** | 0.3381±.017 | 0.2817±.017 | 0.4368±.018 | 0.5676±.010 |
| G2 two-hop+wide (2s) | 0.9790 | 0.8454 | 0.6210 | 0.3731 | 0.3106 | **0.4821** | 0.6045 |
| W3 nov.1 midpoint (2s) | 0.9612 | 0.8275 | 0.6516 | 0.3529 | 0.2931 | 0.4569 | 0.5864 |
| recency | — | — | — | 0.3540 | 0.2777 | 0.4734 | 0.5266 |

- **Recency heuristic beaten at 5 seeds** (W1): MRR +0.012, h@1 +0.027,
  h@10 tie, h@100 +0.065, seen-MRR 0.6686 > recency's 0.6404. Same model
  clears the inductive roof (0.6200) with hist 0.840.
- **Inductive 0.6601 ± 0.013 (W2)** — +0.044 over campaign-1's headline,
  ~96% of the measured 0.689 feature-family oracle.
- **Two-hop REVERSES to a win on wide features** (G2 2s: +0.007 MRR,
  +0.009 h@10 over W1, σ≈0.0003): graph-space widening contributes once
  time-depth features exist — it was null on narrow features (campaign 1)
  and on strat/k64 attention windows. Extended to 5 seeds; G3 probe
  (two-hop + nov.1) launched for the single-model all-rounder.

## 2026-08-14 — campaign-2 final (5 seeds) — both barriers down

| config | rand | hist | inductive | MRR | h@1 | h@10 | h@100 |
|---|---|---|---|---|---|---|---|
| **G2 wide+two-hop (5s)** | 0.9789 | 0.8423 | 0.6215±.003 | **0.3689±.004** | **0.3059±.004** | **0.4787±.004** | **0.6011±.004** |
| W3 nov.1 (5s) | 0.9641 | 0.8263 | 0.6534±.008 | 0.3525 | 0.2930 | 0.4555 | 0.5873 |
| G3 two-hop+nov.1 (2s) | 0.9696 | 0.8404 | 0.6584 | 0.3530 | 0.2935 | 0.4537 | 0.5858 |

G2 beats recency on ALL FOUR ranking metrics at 5 seeds (MRR +0.015,
h@1 +0.028, h@10 +0.005, h@100 +0.075) and holds ind 0.6215 / hist 0.8423.
[Corrected 2026-08-16: initial G2 ranking readout caught the JSON at 3/5
seeds (tgn.ranking flushes per seed); 5-seed truth re-verified by a
deterministic re-run — same conclusion, slightly smaller margins.]
W2 (0.6601) stays the inductive champion. Figure
`figures/tgn_global_frontier2.png`; write-up updated in `results.md`.
Campaign closed.

# Protocol correction (2026-08-16, user directive)

The AU-ROC protocol used since the TGAT baseline diverges from the DGB
paper (Poursafaei et al. 2022). Differences found (ours → paper):
1. one GLOBAL AUROC over the test set → per-batch (200 edges) AUROC,
   MEAN over batches;
2. negative pools per DAY → per BATCH;
3. historical pool = pairs seen before the TEST SPLIT (frozen) → pairs
   seen before the CURRENT BATCH (history accumulates through test);
4. inductive pool = ALL test-only pairs incl. future ones → test-only
   pairs OBSERVED SO FAR, short pools PADDED with random pairs (early
   test batches are mostly padding);
5. random negatives = uniform never-seen pairs (both endpoints random)
   → source kept from the positive, random destination from the data's
   destination set, only batch-collision exclusion.

New module `src/tgn/evaluate_dgb.py` (paper-faithful; legacy protocol left
untouched for internal comparability). Reports per-batch-mean AUROC (the
paper's number), global AUROC (reference), and pad_frac (the padding share,
which materially shapes historical/inductive numbers). 7 property tests in
`tests/test_tgn_dgb.py` (accumulation-through-test, first-inductive-batch
fully padded, source preservation, batch-collision exclusion, determinism,
per-batch mean hand-check, EdgeBank batch semantics).
Rerunning: campaign-2 arms (W1, W2, W3, G2), campaign-1 arms
(g_headline_ind, g_f14_mrr), legacy references (pfpop_mrr, tgn_hard,
tgn baseline), EdgeBank inf/tw.

Notable implication to verify in the numbers: under (3) and (4) the
paper's "inductive" negatives are *recurring test pairs seen so far* —
precisely the population our novelty negatives were built to model — while
the legacy pools also included never-yet-active future pairs. Expect level
shifts everywhere; relative ordering is the honest question.

## 2026-08-16 — corrected numbers (10 reruns, 5 seeds each)

Full table in `results.md` §"DGB protocol correction"; figure
`figures/tgn_dgb_correction.png`; JSONs `figures/dgb/`. Highlights:
- **W2 inductive 0.7298 ± .020** (novelty arms: c1-ind 0.7104, W3 0.6933);
  hard-CE 0.6754; no-novelty wide arms 0.588-0.596; pfpop/f14 0.52-0.55;
  EdgeBank inductive collapses to 0.25-0.30 (paper-consistent).
- **Honest revision: the "0.60 roof" was partly a legacy-protocol
  artifact** — under the paper's metric hard-CE was already at 0.675.
  The novelty-negative mechanism is *more* valuable under the corrected
  metric (+0.13 over its no-novelty twins); campaign-arm ordering
  preserved. Ranking metrics unaffected; recency win stands.
- pad_frac (inductive) = 0.06, concentrated in the earliest test batches.

## 2026-08-16 — EdgeBank validation-span check + 3:1 ratio experiment

**Validation-span EdgeBank (user question "was val high, test low?"): NO —
validation mirrors test within ~0.006** (inf: val 0.770/0.272/0.300 vs
test 0.763/0.265/0.295 across random/hist/inductive; compositions p≈0.54,
q≈0.94 in both spans). The inversion is structural (pool construction),
visible at validation time; a val-calibrated sign flip would transfer to
test on hist/induc but craters random (0.77→0.23). Protection = the joint
three-column report. Side benefit: val↔test agreement shows the corrected
protocol is temporally stable. `dgb_negatives`/`edgebank_scores_dgb` now
take a `span` argument.

**3:1 negative:positive ratio (75/25 split)** (`--n-neg 3` support added,
`scripts/run_dgb_ratio.py`, `figures/dgb/ratio3_dgb.json`; 8 tests):
- random/historical: unchanged (Δ ≤ 0.0014) — AUROC prevalence-invariance
  confirmed empirically; historical pool (16,139 pairs) never exhausts.
- inductive: ALL models rise +0.025..+0.042 (tgn_hard 0.675→0.704,
  pfpop 0.550→0.592, G2 0.596→0.633, W2 0.730→0.754; EdgeBank inf
  0.295→0.330). Cause is purely mechanical: the inductive pool (2,904
  distinct pairs) can't fill 3x demand, padding doubles 6.0%→13.1%
  (predicted exactly from pool sizes), and random-like pads dilute the
  hard setting toward random NS. Mixture arithmetic reproduces the shifts.
- Ranking of models unchanged; seed std NOT reduced (0.0185 vs 0.0195) —
  extra shared-pool negatives don't buy precision here.
Conclusion: the 1:1 protocol is the more conservative/faithful setting;
higher ratios quietly weaken the inductive test via padding.

## 2026-08-16 — EdgeBank inversion: proper three-front verification
(user pushback: "intended behavior" was asserted beyond evidence — fixed)

1. **Frozen-memory ablation** (`freeze_memory` flag in
   `edgebank_scores_dgb`, tested): stop EdgeBank's memory at train+val and
   inductive FLIPS 0.295 → 0.704 (inf; tw 0.252 → 0.623), because
   test-only negative pairs become unseen (q 0.94 → 0.00) while 40.7% of
   positives remain known. Historical stays inverted either way (q 0.92
   frozen). The inversion is therefore 100% attributable to memory
   accumulation through test.
2. **DGB source verified** (EdgeBank/link_pred/edge_bank_baseline.py,
   fetched from github.com/fpour/DGB): `learn_through_time` is HARDCODED
   True — history = train+val + all prior test batches, per-batch AUROC
   averaged, current batch excluded. Our accumulating implementation is
   faithful to their code, not just their prose.
3. **Paper's own numbers show the same inversion** (appendix Tables 6 &
   10): EdgeBank∞ inductive AU-ROC = 0.43 (Wikipedia), 0.47 (Reddit),
   0.22 (MOOC), 0.31 (UCI), 0.44 (Flights)... — below 0.5 on 12 of 13
   datasets; historical similar (0.27-0.55). Our 0.295/0.252 are in the
   published range. Caveat kept honest: the paper notes the "significant
   drop" but never explicitly discusses the below-chance direction, so
   "intended" remains an inference; what is verified is that it is a
   direct, reproducible consequence of their hardcoded design.

## 2026-08-16 — two new test modes (user directive): symmetric inductive
   and test-recurring

`inductive_sym`: positives restricted to pairs never seen before test
(59.3% of test events); negatives = standard inductive sampling.
`test_recurring`: negative pool = ALL span-active pairs so far (train-seen
allowed); positives unchanged. Runner `scripts/run_dgb_modes.py`,
results `figures/dgb/modes_dgb.json`, suite 100.

| model | inductive_sym | test_recurring | (std inductive) |
|---|---|---|---|
| W2 @50 epochs | **0.7813 ± .004** | **0.6918 ± .002** | 0.8240 |
| TGN hard-CE | 0.6765 ± .010 | 0.6248 ± .007 | 0.6754 |
| W2 (early stop) | 0.6519 ± .022 | 0.6166 ± .012 | 0.7298 |
| G2 wide+two-hop | 0.5001 ± .012 | 0.5118 ± .010 | 0.5957 |
| pfpop_mrr | 0.4955 ± .013 | 0.5005 ± .010 | 0.5496 |
| EdgeBank inf / frozen | 0.1313 / **0.5000** | 0.2829 / 0.5353 | 0.2946 / 0.7036 |

Findings:
1. **Frozen EdgeBank = exactly 0.5000 in the symmetric mode** — its
   frozen-memory "0.704" was pure construction (both sides unseen → all
   ties), as designed. Accumulating EdgeBank inverts harder (0.131 =
   0.5+(p−q)/2 with p≈0.20 test-repeat positives, q≈0.94).
2. **The symmetric mode unmasks a hidden split**: the deployment-oriented
   models (G2, pfpop) are at CHANCE (0.50) on genuinely-new-pair timing —
   their standard-inductive 0.55-0.60 was carried entirely by seen
   positives. Real skill on new relationships lives in novelty training
   (W2@50 0.7813) and, notably, hard-CE (0.6765).
3. test_recurring preserves the same ordering at lower levels (novelty
   models still lead among pairs active in test regardless of origin).
All pre-registered predictions from the launch message held.

## 2026-08-16 — FDR/NPV on inductive_sym (val-chosen Youden threshold)

`scripts/run_sym_fdr_npv.py` → `figures/dgb/sym_fdr_npv.json`. Threshold
per model/seed = Youden J on the VAL-span inductive_sym construction,
frozen, applied to test. Pool prevalence 0.372. TGAT included; EdgeBank at
its single non-trivial threshold (flag = seen).

| model | AUROC (pooled) | FDR | NPV |
|---|---|---|---|
| W2 @50 | 0.7774±.004 | 0.3362±.015 | 0.7941±.004 |
| hard-CE | 0.6741±.011 | 0.2350±.024 | 0.7037±.002 |
| W2 early | 0.6481±.023 | 0.1932±.018 | 0.7036±.002 |
| G2 | 0.4957±.012 | 0.3942±.056 | 0.6717±.007 |
| pfpop | 0.4890±.013 | 0.4119±.068 | 0.6669±.008 |
| TGAT | 0.4000±.006 | 0.6229 | 0.8688 (low-volume flags) |
| EdgeBank inf | 0.1330 | 0.8852 | 0.1134 |
| EdgeBank frozen | 0.5000 | 0.0000 (VACUOUS: zero flags) | 0.6279 (=1−π, clears all) |

Readings: (1) NO model meets FDR<0.125 AND NPV>0.875 jointly at a single
val-chosen threshold on this pool — the corner needs (FPR .066, TPR .774)
≈ AUROC 0.945 at π=.372 (binormal); best available 0.777. (2) FDR alone
is not comparable across models without flag volume (W2-early's lower FDR
than W2@50 comes from a more conservative val threshold, i.e. fewer
flags); the joint (FDR, NPV) row is the honest read. (3) Degenerate rows
flagged: frozen EdgeBank flags nothing (FDR 0/0→0), TGAT's NPV rides on
tiny flag volume from an inverted score. Three-zone abstention remains
the only route to the program bar with current models. (user directive): epochs are a dial

Setup: W2 recipe, `--epochs 50 --patience 50` (early stopping disabled),
new `--save-last` flag stores the epoch-49 weights alongside the
best-criterion checkpoint. 5 seeds in parallel. Curves (incl. the combo
criterion): `figures/w2_fixed50_curves.png`. Validation dynamics: matched
val-AP saturates ~0.85 by epoch ~12 and plateaus; sampled val-MRR peaks
at epoch 4-5 (~0.575) and decays to ~0.52; combo peaks epochs 3-11 and
declines thereafter — the adaptive stop reproduced the same checkpoints
(best-criterion fixed-50 ≈ original W2 within noise, sanity passed).

**Final-epoch (49) models vs best-criterion models (5 seeds):**

| metric | best-criterion (ep 3-11) | final epoch 49 |
|---|---|---|
| DGB inductive | 0.7356 ± .024 | **0.8240 ± .004** |
| DGB historical | 0.8209 ± .010 | **0.8536 ± .003** |
| DGB random | 0.9038 | 0.8942 |
| legacy ind / hist | 0.6633 / 0.8280 | **0.6789 / 0.8558** |
| deploy MRR / h@1 | 0.3323 / 0.2777 | 0.2517 / 0.2082 |

Reading: the early stop was optimal ONLY for the ranking-weighted combo
objective. Paired-classification metrics keep improving far past it —
epoch-49 W2 is a **new inductive record by a wide margin (DGB 0.8240,
+0.09 over any prior model; legacy 0.6789 also a record)** and nearly
matches hard-CE's historical (0.854 vs 0.882), at a steep ranking cost
(MRR −0.08). Seed variance on DGB inductive collapses (±0.024 → ±0.004).
The epoch count is therefore another calibration dial on the same
paired-vs-ranking frontier: converged = calibrated to the training noise
mixture (best for AUROC benchmarks), early = ranking compromise.
Checkpoints: `figures/w2_fixed50/` (best), `figures/w2_fixed50_last/`
(epoch 49); eval JSONs alongside + `figures/dgb/w2_fixed50*_dgb.json`.

# Campaign 3 (2026-08-24): deeper message passing on top of W2@50

**Directive:** improve on W2 @ fixed 50 epochs (the converged, benchmark-
calibrated checkpoint: DGB ind 0.8240, inductive_sym 0.7813, hist 0.8536)
via architectural depth — more message passing between nodes before the
link prediction. Node scai3, **GPU 2**.

## 2026-08-24 — code: N-hop recursion, stacked layers, inner-hop width

`src/tgn/model.py`: `n_layers` generalized from {1,2} to any N (recursive
`_embed`; hop j uses `attn{j+1}`; (node, edge-day) dedup at every hop
boundary, invalid slots no longer embedded); `n_stack` = extra residual
attention layers over the SAME one-hop neighbor set (depth of processing
without widening the receptive field); `k_inner` = neighbours per inner hop
(bounds the k**depth blow-up; like `--k`, NOT recorded in the checkpoint —
must be passed at eval). `build_from_checkpoint` infers hop and stack depth
from keys. `NeighborStore.sample(k=)` per-call override. CLI flags
`--n-layers N --n-stack S --k-inner K` on train; `--k-inner` on
evaluate/ranking/evaluate_dgb; `run_dgb_modes.py --model NAME=DIR`.
Tests: +6 in `tests/test_tgn_twohop.py` (3-hop reachability with a
same-day perturbation invisible to memory and to 2-hop; no same-day/future
leakage at 3 hops; k_inner is a no-op at 1 hop and binds at 2; stack never
widens the field; deep train smoke; depth inference). Suite 107.
Regression: G2 seed-0 checkpoint scores under old vs new `embed` differ by
max 1.4e-6 on 3000 test queries (kernel-order noise, per 05 §quirks).

Launched (W2 recipe `ce n5 h.1 p.4 nov.2 f23 combo`, `--epochs 50
--patience 50 --save-last`, seeds 0 1 as separate processes, GPU 2), out
under `figures/deep/<name>/s<seed>/`:
- `d_hop2`: `--n-layers 2` (G2's two-hop, now at 50 epochs with novelty)
- `d_hop3`: `--n-layers 3` (k=20 at every hop)
- `d_stack2`: `--n-stack 2` (three attention layers deep, one-hop field)
Comparator: W2@50 per-seed (seeds 0,1) from `w2_fixed50_last_*`. Eval via
`scripts/eval_deep.sh`, table via `scripts/compare_deep.py`.

## 2026-08-24 — d_stack2 result (2 seeds, epoch-49 checkpoint vs W2@50 same seeds)

Training: ~43 s/epoch with 6 trainers sharing GPU 2. Combo-best 0.691/0.666.

| metric | W2@50 (s0,s1) | d_stack2 | Δ |
|---|---|---|---|
| DGB random / hist / inductive | .8881 / .8516 / .8216 | .8799 / .8590 / .8161 | −.008 / +.007 / −.006 |
| inductive_sym / test_recurring | .7787 / .6897 | .7725 / .6916 | −.006 / +.002 |
| legacy hist / inductive | .8532 / .6778 | .8614 / .6696 | +.008 / −.008 |
| MRR / h@1 / h@10 / h@100 | .2537 / .2101 / .3270 / .4768 | .2445 / .2028 / .3140 / .4700 | −.009 / −.007 / −.013 / −.007 |

**Null.** Two extra residual attention layers over the same 20 neighbours
move nothing beyond 2-seed noise (historical +0.007 is the only positive
delta; ranking is uniformly slightly worse). Depth of processing without
a wider receptive field is not the missing ingredient — consistent with the
"attention cannot count" reading in design-notes §5.

## 2026-08-24 — d_hop2 result (2 seeds, epoch-49 vs W2@50 same seeds)

Training ~2 min/epoch shared. Combo-best 0.712/0.700 (W2 0.693).

| metric | W2@50 (s0,s1) | d_hop2 | Δ |
|---|---|---|---|
| DGB random / hist / inductive | .8881 / .8516 / .8216 | .8991 / .8587 / .8060 | **+.011 / +.007 / −.016** |
| inductive_sym / test_recurring | .7787 / .6897 | .7573 / .6792 | **−.021** / −.010 |
| legacy hist / inductive | .8532 / .6778 | .8618 / .6703 | +.009 / −.008 |
| MRR / h@1 / h@10 / h@100 | .2537 / .2101 / .3270 / .4768 | .2535 / .2090 / .3272 / .4815 | 0 / 0 / 0 / +.005 |

**A trade, not a win.** Two-hop at 50 epochs with nov .2 lifts random and
historical AUROC but gives back the new-pair skill that is W2@50's whole
point (inductive_sym −0.021, ~5× W2@50's seed std). Ranking flat (G2's
ranking gain from two-hop does not reappear under the novelty mixture —
consistent with the G3 2-seed hint that the two gains don't stack).
Reading: the second hop's extra capacity is spent on the training mixture
(recurring-pair contrast), pulling calibration away from novel pairs.
Follow-up launched: `d_hop2_nov3` = `--n-layers 2 --nov-frac 0.3
--pop-frac 0.3` — does a stronger novelty dial redirect the gain?

## 2026-08-24 — early-stop (combo-best) checkpoints: hop2 shifts the balanced frontier

Same-seed comparison vs `w2_fixed50` best-criterion checkpoints (W2-early):

| metric | W2-early (s0,s1; ep 11/7) | d_stack2_best (ep 5/4) | d_hop2_best (ep 6/8) |
|---|---|---|---|
| DGB rand / hist / ind | .9042 / .8290 / .7561 | .8846 / .8115 / .7420 | .9122 / .8410 / .7536 |
| inductive_sym / test_rec | .6654 / .6257 (w2_wide_ind s0,s1) | .6706 / .6368 | **.6861** / .6363 |
| MRR / h@1 / h@10 / h@100 | .3135 / .2616 / .4013 / .5406 | .3479 / .2905 / .4458 / .5696 | .3294 / .2743 / .4200 / .5586 |
| selected val_ap / val_mrr | .843,.839 / .570,.545 | .808,.803 / .591,.553 | .839,.839 / **.605,.584** |

Reading: stack2's ranking gain is the epoch dial (earlier stop, lower
val_ap — a different point on the same frontier). hop2's is not: at the
SAME matched val_ap it selects a checkpoint with +0.03 sampled val_mrr,
and on test that is +0.016 MRR, +0.01 rand/hist, +0.02 inductive_sym,
inductive flat — a frontier shift for the balanced single-scalar model.
So two-hop helps the *balanced* W2 checkpoint on every axis and hurts the
*converged* one on new-pair timing. Caveat: 2 seeds; W2-early has seed
std ~0.02 on inductive metrics, so the sym gain needs 5 seeds.

## 2026-08-24 — d_hop2_nov3 first numbers (2 seeds, epoch-49) and the control

DGB rand / hist / ind = .8844 / .8686 / **.8558**; inductive_sym **.8263**;
test_recurring .7178 — vs W2@50 same seeds .8881 / .8516 / .8216, .7787,
.6897. Ranking pending. This is +0.034 inductive and +0.048 inductive_sym,
the largest single move since the novelty mechanism itself. Attribution is
NOT yet clean: nov .3 alone was never run at 50 epochs on wide features
(campaign-1's nov .3 verdict was narrow features + early stop). Launched
the control `d_hop1_nov3` = W2 recipe with `--nov-frac 0.3 --pop-frac 0.3`
at 1 hop, same seeds/epochs. The architectural claim stands only if
hop2_nov3 beats hop1_nov3.

## 2026-08-24 — d_hop2_nov3 full table (2 seeds, epoch-49 vs W2@50 same seeds)

| metric | W2@50 | d_hop2_nov3 | Δ |
|---|---|---|---|
| DGB random / hist / inductive | .8881 / .8516 / .8216 | .8844 / .8686 / .8558 | −.004 / **+.017** / **+.034** |
| inductive_sym / test_recurring | .7787 / .6897 | .8263 / .7178 | **+.048** / **+.028** |
| legacy hist / inductive | .8532 / .6778 | .8671 / .6844 | +.014 / +.007 |
| MRR / h@1 / h@10 / h@100 | .2537 / .2101 / .3270 / .4768 | .2410 / .1997 / .3068 / .4486 | −.013 / −.010 / −.020 / −.028 |

Early-stop checkpoint (ep-best): rand .8946 / hist .8466 / ind .7929,
inductive_sym .7370 — vs W2-early .9042 / .8290 / .7561, .6654.
Ranking cost is the usual novelty-dial price (W2@50 already pays −0.08 MRR
vs W2-early; this adds −0.013). Attribution awaits `d_hop1_nov3`.

## 2026-08-24 — d_hop3 result (2 seeds, epoch-49 vs W2@50 same seeds)

~3 min/epoch shared (≈2.5 h); eval ~1 h. Combo-best 0.720/0.689.

| metric | W2@50 | d_hop3 | Δ |
|---|---|---|---|
| DGB random / hist / inductive | .8881 / .8516 / .8216 | .8947 / .8600 / .8217 | +.007 / +.008 / .000 |
| inductive_sym / test_recurring | .7787 / .6897 | .7803 / .6921 | +.002 / +.002 |
| MRR / h@1 / h@10 / h@100 | .2537 / .2101 / .3270 / .4768 | .2431 / .2009 / .3106 / .4702 | −.011 / −.009 / −.016 / −.007 |

Marginal: three hops keep hop2's random/historical gain without hop2's
inductive loss, but new-pair metrics are unchanged and ranking slips
~0.01. Not worth 6× the compute of one hop on its own.

## 2026-08-24 — control `d_hop1_nov3`: the hop2_nov3 gain is the novelty dial, not the hop

| metric | W2@50 | d_hop1_nov3 (1 hop) | d_hop2_nov3 (2 hops) |
|---|---|---|---|
| DGB random / hist / inductive | .8881 / .8516 / .8216 | .8868 / .8707 / **.8556** | .8844 / .8686 / .8558 |
| inductive_sym / test_recurring | .7787 / .6897 | **.8284** / .7190 | .8263 / .7178 |
| legacy hist / inductive | .8532 / .6778 | .8700 / .6833 | .8671 / .6844 |
| MRR / h@1 / h@10 / h@100 | .2537 / .2101 / .3270 / .4768 | .2362 / .1965 / .2999 / .4509 | .2410 / .1997 / .3068 / .4486 |

hop2 − hop1 at nov .3: every metric within ±0.005 (MRR +0.005 the largest).
**Campaign-3 verdict on the architectural hypothesis: null.** Deeper
message passing (2 hops, 3 hops, 3 stacked layers) does not improve
W2@50 on any new-pair metric; 2-hop alone trades inductive for
random/historical, 3-hop is a marginal no-cost +0.007 on random/hist.
**Incidental finding (the actual improvement):** `--nov-frac 0.3
--pop-frac 0.3` at 50 epochs on wide features = ind 0.856 / sym 0.828 /
hist 0.871 (+0.034 / +0.050 / +0.019 over W2@50) at −0.018 MRR. The
campaign-1 "nov .3 saturates" verdict was measured at narrow features +
early stop; the dial keeps paying once features are wide and the model is
trained to convergence. Promoted to 5 seeds: `d_hop1_nov3` seeds 2 3 4
launched; 5-seed eval to follow.

## 2026-08-24 — `d_hop1_nov3` at 5 seeds (epoch-49 checkpoint): confirmed record

| metric | W2@50 (5s) | nov .3 @50 (5s) | Δ |
|---|---|---|---|
| DGB random | 0.8942 ± .004 | 0.8856 ± .007 | −.009 |
| DGB historical | 0.8536 ± .003 | 0.8628 ± .007 | +.009 |
| DGB inductive | 0.8240 ± .004 | **0.8571 ± .002** | **+.033** |
| inductive_sym | 0.7813 ± .004 | **0.8285 ± .002** | **+.047** |
| test_recurring | 0.6918 ± .002 | **0.7196 ± .002** | +.028 |
| legacy hist / inductive | .8558 / .6789 | .8627 / .6847 | +.007 / +.006 |
| MRR / h@1 / h@10 / h@100 | .2517 / .2082 / .3242 / .4786 | .2356 / .1953 / .3016 / .4506 | −.016 / −.013 / −.023 / −.028 |

Early-stop checkpoint of the same runs (vs W2-early 5s): DGB ind 0.7805
vs 0.7356 (+.045), hist 0.8387 vs 0.8209, inductive_sym 0.7239 (W2-early
0.6519), MRR 0.3117 vs 0.3323 (−.021).
Recipe: `--loss ce --n-neg 5 --hard-frac 0.1 --pop-frac 0.3 --nov-frac 0.3
--pair-feat --pair-feat-dim 23 --select combo --epochs 50 --patience 50
--save-last` (1 hop). Checkpoints `figures/deep/d_hop1_nov3_{last,best}/`.
Campaign 3 closed. Session hopped scai3 → scai4 at the end (eval had
already finished; artifacts are on NFS).

# Campaign 3b (2026-08-24, user directive): train the nov .3 recipe for 200 epochs

Node **scai4, GPU 0**. `--save-every 50` added to `tgn.train` (periodic
`tgn_seed{s}_ep{N}.pt` snapshots; test in `tests/test_tgn_train.py`,
suite 108). Launched `figures/deep/nov3_ep200/s{0..4}/`: nov .3 recipe,
`--epochs 200 --patience 200 --save-last --save-every 50`, 5 seeds as
separate processes. Question: is the 50-epoch record a plateau or still
climbing? Snapshots at 50/100/150/200 give the curve from one run.

## 2026-08-25 — 200-epoch result (5 seeds, snapshots every 50; scai4 GPU 0)

~40 min per 50 epochs with 5 seeds sharing one A100. Snapshots evaluated
with `scripts/eval_snapshots.sh` (DGB standard + modes); epoch-200 also
gets legacy + ranking via `eval_deep.sh`.

| checkpoint | random | hist | inductive | inductive_sym | test_rec | MRR |
|---|---|---|---|---|---|---|
| W2@50 (nov .2) | .8942 ± .007 | .8536 ± .003 | .8240 ± .004 | .7813 ± .004 | .6918 ± .002 | .2517 |
| nov .3 ep 50 (run A, `d_hop1_nov3`) | .8856 ± .007 | .8628 ± .007 | .8571 ± .002 | .8285 ± .002 | .7196 ± .002 | .2356 |
| nov .3 ep 50 (this run) | .8855 ± .003 | .8610 ± .007 | .8568 ± .004 | .8269 ± .003 | .7186 ± .004 | — |
| ep 100 | .8775 ± .007 | .8570 ± .007 | .8611 ± .005 | .8314 ± .006 | .7238 ± .003 | — |
| ep 150 | .8794 ± .004 | .8582 ± .004 | .8622 ± .009 | .8344 ± .014 | .7259 ± .008 | — |
| **ep 200** | .8817 ± .007 | .8515 ± .003 | .8619 ± .006 | .8320 ± .005 | .7254 ± .005 | **.2027** |

Readings:
1. **Epoch-50 snapshot replicates run A within 0.002 on every metric** —
   the nov .3 record is reproducible across independent launches/nodes.
2. **Plateau by epoch 100–150.** New-pair metrics gain +0.005 (inductive
   .857 → .862, sym .827 → .832/.834, test_rec .719 → .725) and then stop;
   the ep150 → ep200 deltas are zero within seed std (sym std at ep150 is
   .014, larger than the gain).
3. **The tail is a calibration slide, not learning:** historical −0.010
   (ep50 → ep200), random −0.004, and MRR −0.033 (.2356 → .2027).
   Training loss keeps falling (1.18 → 1.05 → ~1.0) while matched val-AP
   is flat at ~0.85 from epoch 50 on.
4. Verdict: **50 epochs is the operating point** for this recipe. 100
   buys +0.005 on the inductive metrics for −0.01 historical / ~−0.02
   MRR; nothing past that. Checkpoints kept:
   `figures/deep/nov3_ep200/s*/` (ep50/100/150/200 + best + last),
   assembled dirs `nov3_ep200_ep{50,100,150}/`, `nov3_ep200_{last,best}/`.

## 2026-09-04 — FDR/NPV extended to W1 and the final model (report prep)

Re-ran `run_sym_fdr_npv.py` with `w1_wide_mrr` and `d_hop1_nov3_last`
added to the registry → `figures/dgb/sym_fdr_npv_v2.json` (GPU 6, scai4).
Same protocol: val-Youden threshold, frozen, applied to test-span
inductive_sym (prevalence .372).

| model | AUROC (pooled) | FDR | NPV |
|---|---|---|---|
| **nov .3@50 (`d_hop1_nov3_last`)** | **0.8253** | **0.2805** | **0.8189** |
| W2 @50 | 0.7774 | 0.3362 | 0.7941 |
| W1 (`w1_wide_mrr`) | 0.4847 | 0.3620 | 0.6683 |

Readings: (1) W1 confirmed ≈ chance on the balanced test (0.485), closing
the inferred-from-G2 gap in the report table. (2) The final model improves
BOTH threshold metrics over W2@50 (FDR −0.056, NPV +0.025) — first model
above NPV 0.8; program bar (FDR<0.125 ∧ NPV>0.875) still not met.

# Campaign 4 (2026-09-14, user directive): full-softmax link loss

**Question (user):** why sample 5 negatives at all — train the model to pick
the true destination out of all 11,812 candidates directly. Expected: a
ranking gain (it is the ranking objective) and a balanced-test loss (full
softmax = uniform negatives, the setting that scored 0.59 inductive).

**Code:** `--loss full` in `tgn.train` (`full_softmax_mask`,
`full_softmax_ce_loss`; +3 tests in `tests/test_tgn_full_softmax.py`,
suite 111). Per batch: embed all nodes at day T, score B × n_nodes pairs
through the link head (wide pair features via `PairRecency.features_all`,
cached per unique source), softmax over the whole vocabulary with the
source itself and its other same-day destinations masked. The
hard/pop/nov flags no longer enter the loss; they are kept so the matched
validation negatives (selection metric) stay comparable to `d_hop1_nov3`.

**Launched** scai4 GPUs 1, 2: `d_full` seeds 0, 1 as separate processes,
`tgn_global/figures/deep/d_full/s{0,1}/` — nov .3 recipe with `--loss full`
(no `--n-neg`): `--hard-frac 0.1 --pop-frac 0.3 --nov-frac 0.3 --pair-feat
--pair-feat-dim 23 --select combo --epochs 50 --patience 50 --save-last`.
16.5 GB GPU memory each. Epoch timing to follow.

## 2026-09-14 — `d_full` result (2 seeds, 100 s/epoch, 16.5 GB; eval `d_full_eval.out`)

Epoch time 100 s (est. was 5–15 min: the B × n_nodes link-head pass is
cheap; `features_all` per unique source dominates but is vectorized).
Training curve: val_mrr climbs to ~0.72–0.73 by epoch 10 and holds
(W2/nov .3 peak ~0.58–0.60 then decay); matched val_ap ~0.71–0.73.

| metric | d_full @50 (last) | d_full best-combo | nov .3 @50 (5s) | G2 |
|---|---|---|---|---|
| DGB random / hist / inductive | .9545 / .4566 / **.4014** | .9445 / .4661 / .4374 | .8856 / .8628 / .8571 | .98 / .84 / .60 |
| inductive_sym / test_recurring | **.2093** / .3535 | .2417 / .3785 | .8285 / .7196 | — |
| legacy random / hist / inductive | .9792 / .4687 / .5686 | .9730 / .4754 / .5830 | — / .8627 / .6847 | — |
| MRR / h@1 / h@10 / h@100 | **.3935 / .3285 / .5111 / .6567** | .3931 / .3275 / .5113 / .6549 | .2356 / .1953 / .3016 / .4506 | .3689 / .3059 / .4787 / .6011 |
| ranking, unseen stratum: MRR / h@100 | .170 / .426 | — | .046 / .250 | .161 / .360 |

Readings:
1. **New deployment-ranking record on all four metrics**, +0.025 MRR /
   +0.023 h@1 / +0.032 h@10 / +0.056 h@100 over G2 (two-hop), at one hop
   and no early-stop sensitivity (last ≈ best). Beats recency on all four
   (recency .354/.278/.473/.527). Unseen-stratum h@100 .43 vs G2 .36.
2. **Classification collapses; balanced test far below chance.** Full
   softmax = uniform negatives: a previously-seen pair is almost never a
   negative, so the model learns "known pair ⇒ high" and never "known but
   quiet today ⇒ low". inductive_sym .21 is the EdgeBank inversion
   reproduced by a trained model (positives that are unseen score below
   negatives that are recently-emerged pairs). Historical .46, DGB
   inductive .40.
3. The two losses sit at opposite ends of one frontier: sampled mixture
   (nov .3) buys balanced-test skill at −0.16 MRR; full softmax buys
   ranking at −0.62 inductive_sym. A single scalar cannot hold both —
   direct evidence for the two-head plan (full-softmax head for "who
   next", novelty-mixture head for "is this pair live").
4. Note the legacy-protocol numbers (.979/.469/.569) are NOT the
   externally comparable ones; DGB is .9545/.4566/.4014.

Checkpoints `figures/deep/d_full_{last,best}/`. Seeds 2–4 not run (the
verdict is not seed-limited: seed spread ≤ 0.02 on every metric).
