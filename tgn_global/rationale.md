# tgn_global — rationale (started 2026-08-14)

## Objective

Two barriers survived every prior campaign (see `../tgn_improvement/`):

1. **Inductive AU-ROC roof ≈ 0.6.** Every model — TGAT (0.551), TGN baseline
   (0.553), hard-CE (0.599), pair-feat frontier (0.57–0.58) — lands in a
   narrow band under inductive negative sampling, while random and historical
   moved by tens of points across the same interventions.
2. **hits@k vs the recency heuristic.** The pair-recency campaign closed MRR
   to 94% of recency and beat it at hits@100, but hits@1 (0.258 vs 0.278) and
   hits@10 (0.462 vs 0.473) still lose.

## Hypothesis (user directive)

The architecture is the suspect: the embedding attends over only the **20
most recent 1-hop edges** of each node, and the 100-d GRU memory compresses
everything else away. The model has **no view of global graph structure** —
which communities exist, which nodes are structurally close, how the graph
has historically been wired. Inductive discrimination (a brand-new pair
forming vs. an arbitrary new pair NOT forming today) is exactly the question
classical link prediction answers with *structural* signals: triadic closure,
common counterparties, community co-membership, preferential attachment.
None of those are computable from 20 recent 1-hop edges + own memory.
Imbuing the model with global awareness — either longer/wider message
passing or explicit global-structure inputs — should therefore attack both
barriers, and especially the inductive roof.

## What the prior record already constrains (honesty checklist)

- **TGAT is 2-layer** and its inductive is *worse* (0.551) than 1-layer TGN
  (0.599). So naive "add a hop" without memory did not break the roof; the
  test must be a 2nd hop *on top of the learned memory*, and expectations
  set accordingly.
- **dim 200 hurt** both TGAT and TGN — capacity is not the lever.
- Loss/negative-mixture changes moved inductive by ≤ 0.05 across five
  designs — the roof is not a training-signal artifact.
- The pair-recency campaign's method lesson: **explicit streaming features
  at the link head beat architectural hope** (Δt features ×5'd MRR after
  every architectural/loss attempt failed). So Phase A tests global
  awareness as *features* (cheap, direct), Phase B as *message passing*
  (the literal architectural hypothesis). Comparing them decides whether
  "global awareness" or specifically "deeper propagation" is the active
  ingredient.

## A subtlety of the inductive protocol that shapes the design

Inductive negatives are test-only pairs (never seen before `val_end`)
excluding day-T positives — but evaluation **streams**: at test day T, a
test-only pair that already transacted on an earlier test day is
indistinguishable-by-recency from a positive. The pair-recency features that
fixed historical NS therefore *cannot* fix inductive: a large share of
inductive negatives are recently-active-in-test pairs. Diagnosis must
measure that share, and split the roof into:
  (a) new-pair positives vs never-yet-active negative pairs (structure should
      help: is this pair *plausible* at all?), and
  (b) anything vs already-active-in-test negatives (timing: is this pair
      active *today*? — recurrence discrimination with short history).

## Plan

**Phase 0 — diagnosis, no training (evidence before intervention).**
Compute streaming-safe candidate signals on the frozen inductive pools
(seeds 0–4) and measure single-feature AU-ROC:
- common-neighbor count (undirected, strictly-past), Adamic-Adar,
  recency-weighted common neighbors;
- 2-hop directed reachability s→x→c;
- node activity recency/frequency (src and dst);
- preferential attachment (deg_s · deg_c);
- pair/dst recency (the existing features, as reference);
- the "already active in test" split above.
Also: per-event ranking forensics — for the residual hits@1/hits@10 losses of
the MRR-max model, do structural signals separate true destinations from the
outrankers (73% never-seen candidates, per 06)?
Decision rule: any signal family with inductive AU-ROC ≥ 0.65 standalone, or
clear separation on outrankers, graduates to Phase A. If NO global signal
separates the inductive pools, the honest conclusion is that the roof is
informational (the benchmark's negatives are statistically exchangeable with
positives given any past-only information) and the write-up must say so
rather than manufacture an architecture story.

**Phase A — global awareness as link-head features.**
Extend the `tgn.recency` streaming pattern with the Phase-0 winners
(`GlobalStructure` module, same days-<-T discipline, same train/eval/ranking
integration). 2-seed scoping × the established recipe
(`ce, n5, hard .1, pop .5, pair-feat, select combo/mrr`). Success bar:
inductive ≥ 0.65 with historical ≥ 0.83 and MRR ≥ 0.33, or hits@1/10 ≥
recency.

**Phase B — longer message passing.**
Second temporal-attention hop over memory (2-hop neighborhood, k=20 per hop)
— the literal "longer message passing approximating global awareness".
Optionally a global context vector (attention over a learned summary of the
whole active graph) if hop-2 shows signal but saturates. 2-seed scoping,
same bars. Run after A because it is slower (~k× embed cost) and A's result
sharpens interpretation either way.

**Phase C — headline.** 5 seeds of the best configuration, frozen
evaluation + deployment ranking, figures, write-up. Negative results get
reported with the same prominence as wins.

## Campaign 2 (2026-08-14 PM, user directive): widen the receptive field

Hypothesis: widening the model's receptive field **in time** and **in graph
space** yields further gains beyond the roof break. The current receptive
field, stated precisely:
- attention: 20 most recent 1-hop edges (a ~few-day sliver for active nodes);
- memory: infinite-horizon in principle, saturating GRU in practice;
- features: pair *last* day only (no pair depth), 30-day dst window only,
  no relationship age, no recency-rank among partners, no day-level context.

Widenings under test, ordered by expected value from Campaign-1 evidence
(time >> hop-space in this graph):
- **T1 time-deep features**: pair frequency + relationship age (pair history
  depth is currently ONE day!), multi-scale windows (7/90), directed distinct
  partner counts, recency-rank of the candidate among the source's partners
  (the recency heuristic's own statistic, currently invisible to the head).
- **T2 time-stratified neighbor sampling**: same k, but half the slots
  reserved for evenly-spaced OLDER edges instead of all-most-recent — widens
  the attention window to full history at fixed compute.
- **G1 wider one-hop (k=64)** on the new recipe (prior k-nulls predate the
  feature/negative machinery).
- **G2 two-hop on the new recipe** (Campaign-1 null was on the pfpop recipe;
  one configured retest for the record).
- **G3 day-context features** (yesterday/EMA global activity) — whole-graph
  receptive field; can only act through nonlinear interactions since NS pools
  are day-matched. Oracle decides if it's worth model runs.
Decision rule as before: day-split oracle ablations first; only feature
families that add generalizing AU-ROC (or clearly target the hits@1
partner-ordering gap) graduate to GPU runs.

## Ground rules (unchanged from all prior campaigns)

- Evaluation protocol frozen; EdgeBank recompute as integrity check.
- Streaming discipline: any new feature/state for day T reflects days < T
  only, in train, eval, and ranking alike; property tests before GPU time.
- 2-seed scoping → 5-seed headline; no cherry-picking seeds.
- All runs logged in `experiment_log.md` as they happen, including failures.
- GPU 0 (`CUDA_VISIBLE_DEVICES=0`), node-local parquet, tmpfs venv.
