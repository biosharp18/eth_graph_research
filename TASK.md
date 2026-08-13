# TASK: TGN baseline for daily flow prediction on the OFAC k=4 graph

**For:** a fresh Claude agent session in this repo.
**Branch:** work on `feat/bigquery-exploration` (do not create worktrees — the
untracked `flow_graphs/` data does not follow them; do not switch branches).
**Date written:** 2026-08-13.

## 1. Problem

We benchmark temporal graph models on daily-aggregated stablecoin flows around
an OFAC-sanctioned seed address, following the evaluation framework of
Poursafaei et al., *Towards Better Evaluation for Dynamic Link Prediction*
(NeurIPS 2022; PDF at repo root: `2207.10128v2-2.pdf`). A two-headed TGAT and
EdgeBank baselines are complete. The next architecture is **TGN (Temporal
Graph Network, Rossi et al. 2020)** — chosen because the graph's dominant
signal is short-horizon edge recurrence (novelty analysis in the repo history),
EdgeBank's memory is a degenerate TGN, and a simple recency heuristic beats
TGAT 6× on deployment-style MRR. The hypothesis TGN tests: **does a *learned*
per-node memory close the gap between TGAT (no recency mechanism) and the
recency heuristic, especially under historical negative sampling?**

Read these before writing any code — they define the task, data semantics, and
evaluation protocol, and they bind unless explicitly renegotiated:

- `docs/superpowers/specs/2026-08-07-tgat-daily-flow-design.md` (task spec:
  data filters, two-part hurdle targets, split, NS evaluation, metrics)
- `docs/superpowers/plans/2026-08-07-tgat-daily-flow.md` (how the TGAT
  implementation was decomposed; mirror its conventions, not its architecture)
- `src/tgat/` — the reference implementation. Reuse its modules wherever they
  are architecture-agnostic: `data.py` (dataset), `neighbors.py` (temporal
  neighbor store), `evaluate.py` (NS pools, EdgeBank recompute, metrics),
  `train.py:average_precision/auroc`, `plot_training.py` (curve figure).

## 2. What stays fixed (comparability contract)

Results must be directly comparable to `figures/tgat_results.json`:

- Same data: `load_daily_graph(...)` output, unchanged (41,161 daily edges,
  stablecoins-as-USD, day index 0–238, day-snapped 70/15/15 split).
- Same targets and loss: link BCE (1 random-destination negative per positive,
  same-day-collision-excluded) + λ·Huber(δ=1) on z-scored log10 USD, λ=1,
  amount loss on positives only.
- Same evaluation: the three DGB NS strategies via the existing
  `build_negative_pools`, one negative per positive, seeds 0–4, AU-ROC + AP
  (tie-aware), seen/unseen-pair stratification, amount RMSE/MAE vs persistence
  and median baselines, per-day AU-ROC curves. EdgeBank numbers recompute
  identically — do not fork their logic.
- Same training protocol defaults: Adam lr 1e-4, batch 200, max 50 epochs,
  patience 5 on val AP, 5 seeds. **Known context:** at 50 epochs TGAT was
  undertrained (val AP still rising; +0.007 val AP at 150 epochs, no test gain
  on hard metrics). Keep 50-epoch defaults for the headline comparison, and
  optionally add one longer run if early stopping never triggers.

## 3. TGN implementation — agent's judgment within these bounds

Follow Rossi et al.'s architecture: per-node memory vector updated by a
message function + GRU on each event, embeddings from memory plus temporal
graph attention over recent neighbors. Sensible simplifications are fine if
documented (e.g., "identity" message function, single attention layer — the
paper's own ablations support both). Non-negotiable semantics:

- **No leakage:** a batch's events must not influence the memory used to
  predict that same batch (predict first with pre-batch memory, then update).
  Same-day events must not see each other (the day is the timestep). The
  existing `NeighborStore` enforces strictly-past sampling if you reuse it.
- **Memory streaming at eval:** during val/test scoring, memory advances with
  observed positives after each day is scored — the same streaming discipline
  as `edgebank_scores` and the TGAT evaluation.
- **Determinism:** seeded like `tgat.train` (torch.manual_seed + one
  numpy default_rng threaded through all sampling).
- Edge features: the same 6-dim vector (z-log-USD, log1p count, asset shares)
  as messages/attention inputs; no node features exist (memory is the node
  state — that is the point of the experiment).

Layout: a `src/tgn/` package mirroring `src/tgat/` (`model.py`, `train.py`
with the same CLI flags plus any TGN-specific ones, reusing `tgat.data` and
whatever else imports cleanly). Tests in `tests/test_tgn_*.py` at the same
rigor as the existing 35 (leakage property tests matter most: memory
no-peek within batch/day, strictly-past neighbors). The full suite must stay
green: `uv run --group ml pytest tests/ -q`.

## 4. Required outcomes

1. **Working, tested TGN** as above, committed in reviewable increments
   (conventional-commit messages, `feat(tgn): ...` / `fix(tgn): ...`, each
   ending with the `Co-Authored-By: Claude ...` trailer used throughout this
   repo's history — copy the format from `git log`).
2. **Trained models:** 5 seeds, default protocol →
   `figures/tgn_models/tgat-style artifacts`: `tgn_seed{0..4}.pt` +
   `train_log.json` (same history schema as TGAT — see §5 — so
   `plot_training.py` works unmodified).
3. **Evaluation:** `figures/tgn_results.json` in exactly the schema of
   `figures/tgat_results.json` (top-level keys `edgebank`, `tgn` (in place of
   `tgat`), `amount`, `split`).
4. **Figures:** training curves PNG (via `tgat.plot_training`) and any
   comparison figure you find honest and useful, in `figures/`, committed.
5. **Deployment ranking numbers** (MRR, hits@1/10/100, seen/unseen strata) for
   TGN vs the recency heuristic, using the same protocol as the TGAT glimpse:
   rank each test event's true destination against all nodes, filtered
   (same-source same-day positives excluded), candidate embeddings computed
   once per day. Reference implementation exists in the session scratchpad but
   is trivial to re-derive from this description; putting a reusable version
   in `src/tgn/` or a shared module is welcome.
6. **A short results write-up** appended to this file (or `TASK_RESULTS.md`):
   the comparison table (TGN / TGAT / EdgeBank × random / historical /
   inductive, AU-ROC and AP), strata, amount metrics, ranking metrics, and an
   honest interpretation. **Negative results are reportable results** — the
   TGAT capacity experiments all came back null and that was the finding.
   State explicitly whether the hypothesis in §1 held.

## 5. Artifact format conventions (match these exactly)

- `train_log.json`: `{seed: {"best_val_ap": float, "epochs_run": int,
  "history": [{"epoch", "train_loss", "train_bce", "train_huber", "train_ap",
  "train_auroc", "val_ap", "val_auroc"}]}}`, flushed after every seed.
- Results JSON: mirror `figures/tgat_results.json` — per-strategy lists over
  seeds for `auroc`, `ap`, `auroc_seen`, `auroc_unseen`, `per_day` (list of
  per-seed dicts day→auroc); `amount` block with `tgn_rmse`, `tgn_mae`,
  `tgn_rmse_seen`, `tgn_rmse_unseen` (lists) and scalar baseline entries;
  `split` block including `train_end`, `val_end`, `n_edges`, `seen_frac`, and
  `train_days`/`val_days`/`test_days` ranges.
- Model checkpoints are gitignored (`figures/tgn_models/` dir, but `git add -f`
  the `train_log.json`), matching `.gitignore` precedent for `tgat_models`.
- Plots: matplotlib, palette blue `#2a78d6` / orange `#eb6834` / aqua
  `#1baf7a` / gray `#898781`, no dual axes, legends always, 170 dpi.

## 6. Environment — read this or lose hours

- **Never run Python from the NFS `.venv`.** The repo lives on NFS
  (`/xuanwu-tank`), and mmap-loading torch from it can take *hours* (measured:
  52 MB in 25 min). Always prefix every `uv run` with
  `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv` (a tmpfs venv; if missing on the
  current node, create once: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv
  UV_LINK_MODE=copy uv sync --group ml`, ~3 min). `/tmp` and `/data1` are
  **node-local**: switching machines means re-syncing both.
- **Data fast path:** train/evaluate with
  `--parquet /data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet`
  (copy from the NFS `flow_graphs/` via rsync if absent on the current node).
  The NFS copy is canonical; never modify either.
- **GPUs are shared, no scheduler in use on scai5.** Check
  `nvidia-smi` and pin with `CUDA_VISIBLE_DEVICES=<free index>`. Do not use
  Slurm (`sbatch` targets other nodes; the user has asked for direct runs).
  Expect a ~2-minute silent startup per command (NFS metadata tax) before GPU
  memory appears.
- Long runs: launch detached (`setsid nohup ... &`) with output to a file in
  the run's `--out` dir; a run that dies with an empty log was killed, not
  crashed — check for concurrent GPU tenants and node identity first.
- Tests take ~40 s warm; the first command on a cold node pays venv setup.

## 7. Out of scope (explicitly deferred, do not block on these)

Hard-negative training, pair-recency features / recency ensembles, node
activity features, CAWN, TGB-style multi-negative ranking evaluation, and any
BigQuery re-extraction. If TGN results suggest one of these urgently, say so
in the write-up rather than doing it.
