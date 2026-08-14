# tgn_improvement/ — catch-up guide

**What this is:** the complete record of the TGN work on the OFAC k=4 daily
stablecoin flow graph — baseline build, the loss-redesign campaign that
followed, every configuration tried, results, dead ends, and open ideas.
Written 2026-08-14 so a fresh session (human or agent) can resume without
re-deriving anything.

**State of the world right now (updated 2026-08-14, pair-recency campaign):**
- Branch `feat/bigquery-exploration` (unmerged), all work committed.
- The deployment gap is closed to within noise: pair-recency link-head
  features + popularity-weighted negatives + val-MRR selection give
  MRR **0.3314 ± 0.009** (recency heuristic 0.354; hits@100 beats it) while
  historical AU-ROC stays 0.83–0.88 depending on the mixture dial —
  see `06-pair-recency-campaign.md` and `figures/tgn_frontier.png`.
- Best-balanced recipe: `tgn.train --loss ce --n-neg 5 --hard-frac 0.1
  --pop-frac 0.5 --pair-feat --select mrr`.
- The pre-feature best (hard-CE, historical 0.8990 ± 0.0027) still leads
  historical by +0.02–0.07 via the unseen-pair stratum only.

**Reading order:**
1. `01-context.md` — task, data, evaluation protocol, who the players are
   (TGAT, EdgeBank, recency heuristic) and their numbers.
2. `02-loss-iterations.md` — the L0–L5 loss campaign: every config, every
   number, what each iteration taught, including failures.
3. `03-results.md` — headline 5-seed results for the winner, strata, amounts,
   ranking, and where every artifact lives.
4. `04-open-ideas.md` — what to try next and why; what NOT to re-try.
   (Ideas 1, 3, 4 were executed 2026-08-14 — outcomes in 06.)
5. `05-practical-notes.md` — cluster/env gotchas (read before running
   anything), command crib sheet, code and artifact map.
6. `06-pair-recency-campaign.md` — the 2026-08-14 campaign that closed the
   deployment gap: diagnosis, features, negative mixtures, 5-seed frontier.

**Primary sources** (this folder summarizes, they bind):
- `../TASK.md` — the original TGN brief and comparability contract.
- `../TASK_RESULTS.md` — baseline TGN write-up + loss-redesign addendum.
- `../docs/superpowers/specs/2026-08-07-tgat-daily-flow-design.md` — data and
  protocol definitions.
- `../docs/superpowers/plans/2026-08-13-tgn-daily-flow.md` — TGN
  implementation plan (architecture decisions with rationale).

**Revisit (2026-08-14 PM):** `07-beyond-loss-revisit.md` — ensemble/soup/
head/capacity/selection probes show the residual hist-vs-MRR split is a
score-calibration property, not a knowledge or capacity limit;
`--select combo` is the new recommended default.
