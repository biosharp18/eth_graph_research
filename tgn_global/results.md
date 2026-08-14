# tgn_global — results (2026-08-14)

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

## Open next steps

1. **hits@1**: the remaining 0.013 to recency lives in partner-ordering
   precision. The one untried, principled lever: make the sharp dt==1
   contrast trainable by *excluding* most-recent partners from the hard/pop
   negative slices (they are exactly the candidates that are usually
   right), or a small monotonic prior on the pair-dt feature.
2. **Two-calibration heads** (07's proposal) with the FIXED two-term
   sampler: one trunk, one head per reference distribution — now actually
   testable, since the old two-term evidence is void (see bug note).
3. The 0.62 → 0.69 inductive gap: only closable per-benchmark (a scorer
   calibrated for the inductive reference distribution alone); decide
   whether that reporting mode is wanted before building it.
