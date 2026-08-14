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
