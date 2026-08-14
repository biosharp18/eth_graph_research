# tgn_improvement/ — catch-up guide

**What this is:** the complete record of the TGN work on the OFAC k=4 daily
stablecoin flow graph — baseline build, the loss-redesign campaign that
followed, every configuration tried, results, dead ends, and open ideas.
Written 2026-08-14 so a fresh session (human or agent) can resume without
re-deriving anything.

**State of the world right now:**
- Branch `feat/bigquery-exploration` (unmerged), all work committed.
- Best model: TGN trained with hard-negative softmax CE
  (`tgn.train --loss ce --n-neg 5 --hard-frac 0.5`) — historical-NS AU-ROC
  **0.8990 ± 0.0027**, best in the project on all three DGB negative-sampling
  strategies.
- Unsolved: deployment-style full-candidate ranking. Recency heuristic MRR
  0.354 vs ≤0.08 for every learned config. Argued to be a feature/architecture
  limit, not a loss limit (see `04-open-ideas.md`).

**Reading order:**
1. `01-context.md` — task, data, evaluation protocol, who the players are
   (TGAT, EdgeBank, recency heuristic) and their numbers.
2. `02-loss-iterations.md` — the L0–L5 loss campaign: every config, every
   number, what each iteration taught, including failures.
3. `03-results.md` — headline 5-seed results for the winner, strata, amounts,
   ranking, and where every artifact lives.
4. `04-open-ideas.md` — what to try next and why; what NOT to re-try.
5. `05-practical-notes.md` — cluster/env gotchas (read before running
   anything), command crib sheet, code and artifact map.

**Primary sources** (this folder summarizes, they bind):
- `../TASK.md` — the original TGN brief and comparability contract.
- `../TASK_RESULTS.md` — baseline TGN write-up + loss-redesign addendum.
- `../docs/superpowers/specs/2026-08-07-tgat-daily-flow-design.md` — data and
  protocol definitions.
- `../docs/superpowers/plans/2026-08-13-tgn-daily-flow.md` — TGN
  implementation plan (architecture decisions with rationale).
