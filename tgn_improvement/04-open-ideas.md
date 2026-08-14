# 04 — Open ideas and the MRR frontier

## The one unsolved problem

Deployment full-candidate ranking: recency heuristic MRR 0.354 (seen-pair
hits@100 = 0.99) vs 0.046–0.080 for every learned config. Five loss designs
spanning negative distribution and loss shape moved MRR by ±30%, never ×5.
Working conclusion: the bottleneck is *representational* — "which exact
partner did s pay most recently" doesn't survive compression into a 100-d
node memory + attention over 20 recent edges at the precision full-ranking
needs. Everything below is ordered by expected value.

## Ranked next steps

1. **Pair-recency feature at scoring time** (was deferred by TASK.md §7, now
   clearly the motivated step): give the link head `Δt(s,c) = days since
   (s,c) last transacted` (bucketed or time-encoded), computed streaming-safe
   the same way `edgebank_scores` streams. This hands the model the recency
   heuristic as an input; the learned part then only has to *re-rank around
   it*. Pair with the L2 loss (multi-negative CE, uniform) since that's the
   config that demonstrably improves calibration for ranking. Success bar:
   MRR ≥ 0.354 (i.e., don't lose to the heuristic you were given), while
   keeping historical AU-ROC ≥ 0.88 (check it doesn't collapse).
   Cheap sanity variant to run first: score = recency heuristic with
   model-score tiebreaks (no training), to see the ceiling of "recency +
   model refinement".
2. **Pair-level memory** (EdgeBank-shaped state, learned): a small GRU/EMA per
   *pair* (sparse dict, only pairs ever seen) feeding the link head alongside
   the node embeddings. EdgeBank's degenerate pair memory beats the node
   memory on recurrence by construction; making it learnable and combining
   with node embeddings is the architectural version of idea 1.
3. **In-batch destination negatives for the CE loss**: use the other
   positives' destinations in the same day-batch as extra negatives
   (near-free compute — embeddings already exist). These are "active today
   but not yours" — plausibly the sharpest contrast for full-ranking that we
   have NOT tried (all tried negatives were stale pairs, past partners, or
   uniform). Small change inside `train_one`'s loss block.
- 4. **Val-MRR model selection**: a cheap sampled MRR (rank the positive among
  ~500 sampled candidates per val event) as the early-stopping metric for
  ranking-oriented runs. Removes the metric mismatch suspected in L5's early
  stops.
5. **β/K sweep of the two-term loss with fixed selection** — only after 4;
  L5's failure may be partly a model-selection artifact. Low expected value.
6. **TGB-style multi-negative evaluation** (deferred in TASK.md): would make
  the benchmark itself ranking-shaped, aligning the headline metric with the
  deployment question. Evaluation-side change, no training impact.

## What NOT to redo

- Any pure negative-distribution tweak aimed at MRR (L4 settled it).
- Longer training for the baseline loss (TGAT capacity experiments were null;
  hard-CE converges in <20 epochs with patience 5).
- k=100 neighbors (TGAT k-experiment landed within seed noise of k=20).
- Re-running baselines: EdgeBank recompute is deterministic and committed;
  TGAT/TGN-baseline numbers are stable references in
  `figures/tgat_results.json` / `figures/tgn_results.json`.

## Framing note for any write-up

The loss campaign's positive result reads: "with an evaluation-matched
ranking loss, a learned node memory is worth +0.19 AU-ROC over TGAT under
historical negatives" — but keep reporting the deployment gap honestly; it is
the difference between the benchmark story and the "who gets paid next"
product story, and closing it is the actual open problem.

---

## Outcome update (2026-08-14)

Ideas 1 (pair-recency feature), 3 (in-batch negatives), and 4 (val-MRR
selection) were executed; idea 1 + val-MRR selection + a new ingredient
(popularity-weighted negatives) closed the gap: MRR 0.3314 ± 0.009 vs
recency 0.354, with historical 0.83–0.88. In-batch negatives failed
(swamp the softmax). Idea 2 (pair-level memory) is now low value. Full
record and the new open list: `06-pair-recency-campaign.md`.
