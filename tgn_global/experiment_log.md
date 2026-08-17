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

## 2026-08-16 — W2 at fixed 50 epochs (user directive): epochs are a dial

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
