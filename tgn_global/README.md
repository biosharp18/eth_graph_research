# tgn_global/ — catch-up guide

**What this is:** the record of the global-awareness / receptive-field
campaigns (2026-08-14 → 2026-08-16) on the OFAC k=4 daily stablecoin flow
graph, plus the DGB evaluation-protocol correction. Sequel to
`../tgn_improvement/` (which documents the baseline, loss-redesign, and
pair-recency campaigns and remains the reference for those).

**State of the world (2026-09-14):**
- **Campaign 4: `--loss full` (softmax over all 11,812 candidates) = best
  ranker ever (MRR 0.394, h@1 0.329, h@10 0.511, h@100 0.657; beats G2
  and recency on all four) and worst classifier ever (inductive_sym 0.21,
  below chance). Single scalar cannot hold both → two-head objective is
  the next lever. See `results.md` §Campaign 4.
- **Campaign 3 (deeper message passing on W2@50): architectural null.**
  `--n-layers N` / `--n-stack S` / `--k-inner K` generalized in
  `src/tgn/model.py`; 2-hop, 3-hop and stacked layers do not improve the
  converged W2 on new-pair metrics. The same-mixture control found the
  real lever: **nov .3 at 50 epochs on wide features** — DGB inductive
  **0.857 ± .002, inductive_sym 0.829 ± .002, hist 0.863** (5 seeds,
  `figures/deep/d_hop1_nov3_last`), at −0.016 MRR vs W2@50. See
  `results.md` §Campaign 3.
- **Recency heuristic beaten on all four deployment metrics** (5 seeds):
  G2 (wide features + two-hop) MRR 0.3689, h@1 0.3059, h@10 0.4787,
  h@100 0.6011 vs recency 0.3540/0.2777/0.4734/0.5266.
- **Inductive AU-ROC**: W2 (wide + novelty negatives) 0.7298 DGB at the
  ranking-compromise checkpoint; **0.8240 DGB at full convergence**
  (fixed-50 study). The legacy "0.60 roof" was partly a protocol artifact.
- Evaluation now has two protocols: LEGACY (frozen, internal comparisons)
  and DGB (paper-faithful `tgn.evaluate_dgb`, per-batch mean, accumulating
  pools) — use DGB for externally comparable AU-ROC claims — plus two
  additional DGB test modes: `inductive_sym` (both sides new-to-train;
  the clean "new relationship" measure, with a frozen-EdgeBank = 0.500
  control) and `test_recurring` (train-seen negatives allowed). FDR/NPV
  are reported on `inductive_sym` at validation-chosen thresholds.
  **Key finding:** the deployment champions (G2, pfpop) are at chance on
  genuinely-new-pair timing; only novelty-trained W2@50 (0.78) and
  hard-CE (0.68) carry that skill. See `results.md` §Additional test modes.

**Reading order:**
1. `rationale.md` — hypotheses and pre-registered plans for both campaigns.
2. `design-notes.md` — **the design record**: every deliberate choice in
   W1/W2/G2 with evidence, plus the rejected/null paths (attention
   widening, day-context features, structural features) and eval gotchas.
3. `experiment_log.md` — chronological, append-only: every run, every
   number, the diagnosis scripts, the protocol correction, the two
   partial-read corrections.
4. `results.md` — campaign write-ups, headline tables (legacy + DGB),
   honest-caveat sections, reproduction commands.
5. `figures/` — checkpoints (gitignored), results/ranking JSONs, DGB
   reruns (`figures/dgb/`), summary figures
   (`tgn_global_frontier2.png`, `tgn_dgb_correction.png`,
   `w2_fixed50_curves.png`).
6. `scripts/` — Phase-0 diagnostics (signal audits, day-split oracles),
   frontier/curve plotting, DGB ratio & TGAT adapters.

**Code added by these campaigns** (all property-tested; suite 98):
`src/tgn/recency.py` FEAT_DIM_GLOBAL=11/GLOBAL_BUCKETS=14/WIDE=23;
`src/tgn/train.py` `--nov-frac/--nov-window`, `--n-layers 2`,
`--nbr-mode`, `--save-last`, two-term sampler bugfix;
`src/tgat/neighbors.py` stratified sampling mode;
`src/tgn/evaluate_dgb.py` paper-faithful protocol (span + ratio support).

**Primary sources that bind:** `../TASK.md` (comparability contract),
`../tgn_improvement/05-practical-notes.md` (cluster/env gotchas — read
before running anything), `../2207.10128v2-2.pdf` (the DGB paper).
