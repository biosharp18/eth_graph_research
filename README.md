# eth-graph-research

Exploring Ethereum mainnet data on Google BigQuery
(`bigquery-public-data.goog_blockchain_ethereum_mainnet_us`), as groundwork
for transaction-graph research.

## Setup

1. Install [uv](https://docs.astral.sh/uv/), then: `uv sync`
2. Authenticate (first run opens a Google login in your browser;
   credentials are cached afterwards):

   ```bash
   uv run python -m eth_graph_research.main
   ```

   Queries are billed to the `eth-graph-research` GCP project — the first
   1 TB scanned per month is free.

## Explore

```bash
uv run jupyter lab notebooks/01_explore_ethereum.ipynb
```

## The BigQuery cost model in three sentences

BigQuery charges by bytes **scanned** (LIMIT does not reduce cost), and the
first 1 TB per month is free. All tables in this dataset are partitioned by
month on `block_timestamp`, so filtering on it is the main cost control.
Use `bq.dry_run(sql)` to preview a query's scan size for free;
`bq.run_query(sql, max_gb=10)` refuses queries that would bill more than
the cap.

## Dataset pipeline

One-time extraction of the fixed 90-day ETH edge window (~41 GB scanned,
~15 GB on disk), then free local sampling:

```bash
uv run python -m eth_graph_research.extract   # once; refuses to overwrite
uv run pytest                                  # sampler unit tests
```

Snowball-sample a training subgraph (see `notebooks/02_sample_analysis.ipynb`):

```python
from eth_graph_research.sample import connect_edges, pick_seeds, snowball_sample, save_sample
con = connect_edges()
s = snowball_sample(con, pick_seeds(con).address[0], max_nodes=100_000, per_node_cap=50, rng_seed=0)
save_sample(s, "sample01_100k")
```

## Layout

- `src/eth_graph_research/bq.py` — auth + cost-guarded query helpers
- `src/eth_graph_research/extract.py` — one-shot windowed edge extraction
- `src/eth_graph_research/sample.py` — deterministic snowball sampler (DuckDB)
- `notebooks/01_explore_ethereum.ipynb` — worked intro to the dataset
- `notebooks/02_sample_analysis.ipynb` — sanity checks on the first sample
- `data/` — gitignored; extracted parquet + samples with manifest.json provenance
- `docs/superpowers/` — design specs and plans

---

# Temporal-graph modelling on the OFAC flow graph — handoff (2026-09-25)

Everything below this line is the second half of the repo: temporal graph
models (TGAT, TGN, EdgeBank) for daily stablecoin-flow prediction around an
OFAC-sanctioned seed, evaluated per Poursafaei et al. (NeurIPS 2022, PDF at
repo root). Written so a new person or agent can resume without re-deriving.

## Read in this order

1. `TASK.md` — the binding brief: data semantics, split, comparability
   contract, environment rules.
2. `tgn_improvement/README.md` → its numbered files — baseline TGN, the loss
   campaign, pair-recency features. `05-practical-notes.md` is the cluster
   crib sheet.
3. `tgn_global/README.md` → `design-notes.md` (every deliberate design
   choice and every rejected path), `results.md` (campaign write-ups with
   headline tables), `experiment_log.md` (append-only, every run and number).
4. `monthly_report_september_draft.md` — the narrative as told to the
   program; `HRL-FALCON_Template_Aug18.pptx` is the slide deck (not tracked).

## Environment — the four things that cost hours if skipped

- **Never run Python from the NFS `.venv`.** Prefix every command with
  `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv`; create it once per node with
  `UV_LINK_MODE=copy uv sync --group ml`.
- **`/tmp` and `/data1` are node-local.** Check `hostname` first. Data fast
  path: `/data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet`
  (rsync from the NFS `flow_graphs/` copy if absent; never modify either).
- **GPUs are shared, no scheduler.** `nvidia-smi`, pin with
  `CUDA_VISIBLE_DEVICES`. Long runs: `setsid nohup … > out 2>&1 &`. Expect
  ~2 min of silent NFS import before anything prints.
- **Git on this NFS is minutes-slow.** Run it detached or in the background;
  never block a session on it. Checkpoints (`*.pt`) are gitignored; results
  JSONs and logs next to them are committed.

Tests: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/ -q`
(111 tests, ~30 s warm; property tests for leakage, sampler discipline, and
each loss).

## Code map

| package | what |
|---|---|
| `src/tgat/` | data loader (`load_daily_graph`), temporal neighbour store, TGAT, evaluation pools + EdgeBank, plotting. Architecture-agnostic parts are reused by `tgn`. |
| `src/tgn/model.py` | TGN: per-node GRU memory (one step per node per day), temporal attention over the 20 most recent edges, MLP link head on `[z_s ‖ z_d ‖ pair features]`, separate amount head. `--n-layers/--n-stack/--k-inner` generalize depth. |
| `src/tgn/train.py` | day-batched training with one-day-lag differentiable memory; `StreamingNegatives` (uniform / popularity / historical / novelty mixture); losses `bce`, `ce` (sampled softmax), `full` (softmax over all nodes); `--select {ap,mrr,combo}`; `--save-last`, `--save-every`. |
| `src/tgn/recency.py` | streaming, point-in-time-correct pair/node features (4 / 11 / 14 / 23 dims; `--pair-feat-dim 23` is the wide set). |
| `src/tgn/evaluate.py`, `evaluate_dgb.py` | LEGACY (frozen, internal) and DGB (paper-faithful, externally comparable) protocols. **Quote DGB numbers externally.** |
| `src/tgn/ranking.py` | deployment ranking: true destination vs all 11,812 candidates, filtered, seen/unseen strata; recency-heuristic baseline. |
| `tgn_global/scripts/` | `run_dgb_modes.py` (balanced `inductive_sym` + `test_recurring` modes), `run_sym_fdr_npv.py` (FDR/NPV at val-chosen thresholds), `eval_deep.sh <name> "<seeds>"` (full eval of a per-seed run dir), `eval_snapshots.sh`, diagnostics and plots. |

## The two champion recipes (one hop, wide features, 50 fixed epochs)

```bash
P=/data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet
UVP=UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv

# A. classification / balanced-test champion ("TGN improvements" row on the slides)
CUDA_VISIBLE_DEVICES=1 $UVP uv run --group ml python -m tgn.train --parquet $P \
  --seeds 0 --out tgn_global/figures/deep/d_hop1_nov3/s0 \
  --loss ce --n-neg 5 --hard-frac 0.1 --pop-frac 0.3 --nov-frac 0.3 \
  --pair-feat --pair-feat-dim 23 --select combo --epochs 50 --patience 50 --save-last

# B. ranking champion: same, with the loss over ALL candidates
#    (mixture flags then only shape the validation negatives)
CUDA_VISIBLE_DEVICES=2 $UVP uv run --group ml python -m tgn.train --parquet $P \
  --seeds 0 --out tgn_global/figures/deep/d_full/s0 \
  --loss full --hard-frac 0.1 --pop-frac 0.3 --nov-frac 0.3 \
  --pair-feat --pair-feat-dim 23 --select combo --epochs 50 --patience 50 --save-last

# evaluate a per-seed run dir (DGB, balanced modes, legacy, ranking; last + best)
bash tgn_global/scripts/eval_deep.sh d_full "0 1"
```

Seeds run as separate processes on separate GPUs (~100 s/epoch for `full`,
~2 min/epoch for `ce`). Use the **epoch-50 (`_last`) checkpoint**: the
combo-best early stop is the ranking compromise, the converged one is the
calibrated classifier (design-notes §6).

## State of results (2026-09-25)

| model | DGB inductive | balanced (`inductive_sym`) | FDR / NPV | MRR | Hits@1 / 10 / 100 |
|---|---|---|---|---|---|
| EdgeBank (frozen) | ≤ 0.5 | 0.500 by construction | 0.885 / 0.113 | — | — |
| TGAT (no memory) | 0.46 | 0.40 | 0.623 / 0.869 | — | 0.015 / 0.104 / 0.421 |
| W2 @50 (nov 0.2) | 0.824 | 0.781 | 0.336 / 0.794 | 0.252 | 0.208 / 0.324 / 0.479 |
| **nov 0.3 @50** (A) | **0.857** | **0.829** | **0.281 / 0.819** | 0.236 | 0.195 / 0.302 / 0.451 |
| G2 (wide + 2-hop, early stop) | 0.60 | ≈ chance | — | 0.369 | 0.306 / 0.479 / 0.601 |
| **full softmax @50** (B) | 0.40 | 0.21 (inverted) | vacuous | **0.394** | **0.329 / 0.511 / 0.657** |
| recency heuristic | — | — | — | 0.354 | 0.278 / 0.473 / 0.527 |

Program bar (unmet): AUROC > 0.95, Hits@10 > 0.875, FDR < 0.125, NPV > 0.875.

## What we learned (the short version)

1. **The network cannot count.** Attention over 20 recent edges cannot
   compute frequency / age / degree statistics; the 23 streaming features
   supply them and were the first big win (feature oracle diagnosis in
   `design-notes.md` §2).
2. **Training negatives must look like test negatives.** The sampled-softmax
   score is calibrated against whatever fakes it sees. Novelty negatives
   (pairs first seen within 30 days, quiet today) are what moved the balanced
   test from 0.59 to 0.83. The dial: nov 0 / .2 / .3 at 50 epochs.
3. **Epoch count is a calibration dial.** Ranking peaks by epoch ~8, matched
   val-AP keeps rising to ~50; 200 epochs plateaus by 100.
4. **Ranking and balanced-test skill sit at opposite ends of one frontier.**
   Full softmax (uniform negatives) is the best ranker and inverts the
   balanced test; the novelty mixture is the reverse. A single scalar score
   cannot hold both. This is the argument for the two-head objective below.
5. **Architecture depth was null**: 2-hop, 3-hop, stacked attention layers,
   wider attention (k=64, stratified) — none beat the one-hop model once
   features and negatives are right. Every apparent architectural gain
   needed a same-mixture control before it could be attributed.
6. **Protocol gotchas**: legacy vs DGB AUROC are different numbers; binary
   scorers (EdgeBank) have a closed-form AUROC that depends on pool
   composition and on whether memory accumulates through test; FDR/NPV
   depend on threshold + prevalence, always report the pair at the
   val-chosen threshold (`run_sym_fdr_npv.py`).

## Do not re-try (already null, documented)

Common-neighbour / Adamic-Adar features (+0.01 oracle only), day-level
context features (harmful), popularity windows, larger k, stratified
neighbour sampling, 2-/3-hop and stacked layers on top of the converged
model, training past 50 epochs, TGAT without memory.

## Next levers, in priority order

1. **Two-head objective**: one trunk, a `full`-softmax head for "who next"
   and a novelty-mixture head for "is this pair live". Both losses and
   all plumbing exist in `tgn.train`; only the two-head wiring is new.
2. Sweep `--nov-frac` beyond 0.3 and `--nov-window` (never swept; 30 days
   is a guess).
3. Receptive field without deep hops: propagate memory updates to recent
   counterparties, or a community-level memory (see the September report).
4. Scale the corpus with the BigQuery pipeline in the top half of this
   README; the ablations point at data as the nearest lever after (1).

## Artifact conventions

`train_log.json` per run dir (TGAT schema, flushed per seed); results JSONs
mirror `figures/tgat_results.json`; per-campaign checkpoints under
`tgn_global/figures/deep/<name>/s<seed>/` with assembled `<name>_last/` and
`<name>_best/` dirs holding `dgb.json`, `modes.json`, `results.json`,
`ranking.json`. Readers must check seed count — JSONs are flushed per seed
and a partial file has produced two wrong readouts already.
