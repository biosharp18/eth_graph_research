# 05 — Practical notes: environment, commands, code map

## Environment — read before running anything (expands TASK.md §6)

- **Never run Python from the NFS `.venv`.** Prefix every `uv run` with
  `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv`. If missing on the current node:
  `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv UV_LINK_MODE=copy uv sync --group ml`.
- **`/tmp` and `/data1` are node-local, and sessions HOP NODES.** This
  session started on scai5 and silently resumed on scai3 — the first training
  launch crashed because `/data1/.../edges.parquet` didn't exist on the new
  node. Always `hostname` first; rsync the parquet from NFS `flow_graphs/`
  if absent. Data fast path:
  `/data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet`.
- **Git on this NFS is minutes-slow** (a cold `git log` took 4+ minutes;
  cache evicts). Run all git commands detached/background; never block a
  session on them. A commit killed mid-flight can leave files staged but
  uncommitted — check `git status` after interruptions.
- GPUs are shared, no scheduler. `nvidia-smi`, pin with
  `CUDA_VISIBLE_DEVICES=<free>`. GPU 4 (and 2) were the free ones on scai3.
- Long runs: `setsid nohup ... > log 2>&1 &`; ~2 min silent NFS import tax
  before anything prints. An empty log after 15 min = killed or crashed, not
  slow. `tgn.train` prints one line per epoch (flushed) for liveness.
- First command on a cold node can take ~20 min (venv sync + NFS); the warm
  test suite is ~15–30 s.
- Timings on one A100: baseline-loss seed ≈ 40 min; hard-CE K=5 seed ≈ 30–40
  min (earlier stopping offsets the K cost); full eval (3 strategies × 5
  seeds, EdgeBank + streaming TGN) ≈ 15 min; ranking (5 seeds) ≈ 10 min.

## Command crib sheet (from repo root; `UVP` = `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv`, `P` = node-local parquet)

```bash
# tests
UVP uv run --group ml pytest tests/ -q

# train the winner loss, 5 seeds
CUDA_VISIBLE_DEVICES=4 UVP uv run --group ml python -m tgn.train \
  --parquet $P --seeds 0 1 2 3 4 --out figures/tgn_hard_models \
  --loss ce --n-neg 5 --hard-frac 0.5

# frozen evaluation + deployment ranking
UVP uv run --group ml python -m tgn.evaluate --parquet $P \
  --models figures/tgn_hard_models --out figures/tgn_hard_results.json
UVP uv run --group ml python -m tgn.ranking --parquet $P \
  --models figures/tgn_hard_models --out figures/tgn_hard_ranking.json

# figures
UVP uv run --group ml python -m tgat.plot_training \
  figures/tgn_hard_models/train_log.json figures/tgn_hard_training_curves.png "title"
UVP uv run --group ml python -m tgn.plot_loss_compare figures/tgat_results.json \
  figures/tgn_results.json figures/tgn_hard_results.json figures/tgn_loss_bar.png
```

Loss flags on `tgn.train`: `--loss {bce,ce}`, `--n-neg K`, `--hard-frac p`,
`--src-frac q` (share of hard that is same-source partners), and the two-term
extras `--n-neg-hard K2 --beta-hard β`. Defaults reproduce the baseline loss.

## Code map (`src/tgn/`)

| file | contents |
|---|---|
| `model.py` | `TGN`: `init_memory`, `apply_messages` (one GRU step/node/day, mean-aggregated messages), `embed` (attention, memory as node state), `forward` |
| `streaming.py` | `day_ranges`, `score_pairs_streaming` (pre-day memory, positives-only advance, query-order-preserving), `iter_day_embeddings` (all-node embeddings per test day, for ranking) |
| `train.py` | `StreamingNegatives`, `sample_negatives_streaming`, `softmax_ce_loss`, `train_one` (day-batched, one-day-lag differentiable memory, matched-negative early stopping), CLI |
| `evaluate.py` | `tgn_scores` + CLI; imports pools/EdgeBank/amount from `tgat.evaluate` — never forked |
| `ranking.py` | `filtered_rank` (tie-aware), `recency_ranking`, `tgn_ranking`, CLI |
| `plot_compare.py` / `plot_loss_compare.py` | 3-way / 4-way AU-ROC bar charts |

Tests: `test_tgn_model.py` (memory semantics), `test_tgn_streaming.py`
(leakage properties: same-day/future events can't influence scores, queries
don't advance memory), `test_tgn_train.py` (smoke + determinism),
`test_tgn_negatives.py` (sampler leak-freedom, CE hand-check, bursty-graph
recency smoke), `test_tgn_ranking.py` (hand-computed ranks/filters).

## Conventions that bind (from TASK.md §5)

- `train_log.json` schema shared with TGAT so `tgat.plot_training` works
  unmodified; flushed after every seed.
- Results JSONs mirror `figures/tgat_results.json` exactly (top-level
  `edgebank`/`tgn`/`amount`/`split`).
- Checkpoint dirs gitignored; `git add -f` the `train_log.json`.
- Plots: matplotlib, palette `#2a78d6`/`#eb6834`/`#1baf7a`/`#898781`,
  legends always, no dual axes, 170 dpi.
- Commits: conventional (`feat(tgn): ...`), trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Numeric quirks encountered

- Cross-run score comparisons on CPU can differ by 1 ulp (allocation-history
  dependent kernels); leakage tests that compare *identical* call patterns
  are bitwise, cross-run alignment tests use `allclose(rtol=1e-5)`.
- `average_precision` and `auroc` (in `tgat.train`) are tie-aware; EdgeBank's
  binary scores depend on it.
