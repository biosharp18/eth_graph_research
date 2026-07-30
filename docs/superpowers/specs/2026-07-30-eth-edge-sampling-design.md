# ETH Edge Extraction and Snowball Sampling

**Date:** 2026-07-30
**Status:** Approved

## Goal

Produce a ~100k-node training subgraph for link prediction: download a fixed
90-day window of native-ETH transaction edges from BigQuery once, then
snowball-sample a subgraph locally with full, free iteration on sampling
parameters.

## Background and prior decisions

- Scope is **ETH-only** (no `token_transfers`): edges are rows of
  `bigquery-public-data.goog_blockchain_ethereum_mainnet_us.transactions`
  with `value > 0`. Decided in conversation on 2026-07-30; token edges can
  be joined later via `transaction_hash`.
- Model target: **link prediction** on the transaction graph (~100k nodes,
  trainable on a single GPU).
- Measured on 2026-07-30 for the trailing 90 days: ~103.4M ETH-moving
  edges, ~14.6M unique senders, ~11.7M unique receivers. Full extraction
  scan estimated at 40.8 GB (dry run); verification count ~26 GB.
- Query infrastructure: cost-guarded helpers in `src/eth_graph_research/bq.py`
  (`dry_run`, `run_query(max_gb)`), user auth cached.

## Design

### 1. Extraction — `src/eth_graph_research/extract.py`

- **Fixed window:** `2026-05-01 00:00:00 UTC` (inclusive) to
  `2026-07-30 00:00:00 UTC` (exclusive). Explicit constants, not relative
  dates, so re-runs are reproducible.
- Columns: `block_timestamp, transaction_hash, from_address, to_address,
  value_lossless`; filter `value > 0`.
- Behavior: dry-run first and print the estimate; execute with explicit
  `max_gb=50`; stream results in chunks to parquet via the BigQuery
  Storage API (never hold the full result in memory).
- Store amounts as both the lossless wei string and derived `value_eth`
  FLOAT64.
- Idempotent: refuses to overwrite an existing output directory.
- Verification: row count of written parquet must equal a BigQuery
  `COUNT(*)` for the identical window/filter.

### 2. Storage layout

```
data/                                   # gitignored
  raw/eth_edges_2026-05-01_2026-07-30/
    part-*.parquet
    manifest.json    # exact SQL, window, row count, extraction timestamp
  samples/<name>/
    nodes.parquet
    edges.parquet
    manifest.json    # seed, max_nodes, per_node_cap, rng_seed, counts
```

Expected raw size ~5-8 GB.

### 3. Snowball sampler — `src/eth_graph_research/sample.py`

DuckDB over the raw parquet; queries are batched **per hop** (full scans,
~1-3 s each), never per node.

- `snowball_sample(seed_address, max_nodes=100_000, per_node_cap=50,
  rng_seed) -> Sample`
- Algorithm: BFS from the seed. Each hop: fetch all window edges touching
  the frontier (both directions, one query); per frontier node keep at most
  `per_node_cap` edges chosen uniformly at random (supernode defense);
  counterparties form the next frontier; if adding a hop's counterparties
  would exceed `max_nodes`, admit a uniformly random subset that lands the
  node set exactly on `max_nodes`, then stop (never overshoot).
- Final pass: the output edge set is the **induced subgraph** — all window
  edges whose endpoints are both in the sampled node set (one join), not
  just BFS-traversed edges.
- Deterministic for a given `rng_seed`.
- Seed-picker helper: list candidate addresses by activity band (default:
  50-500 window transactions — active but not hubs).

### 4. Validation and testing

- **Unit tests (pytest) for the sampler** on a small synthetic edge list:
  per-node cap respected; node budget stops expansion; induced edge set is
  complete; identical `rng_seed` reproduces identical output.
- Extraction verified by row-count match (Section 1).
- `notebooks/02_sample_analysis.ipynb`: run one real sample, then
  sanity-check degree distribution, connected components, top hubs.

## Error handling

- Extraction: cost guard via dry run + `max_gb`; refuses to overwrite
  existing data; chunked writes so a crash loses only the current chunk
  (partial dirs are detectable: no manifest.json until success).
- Sampler: clear errors for a seed absent from the window or an empty
  frontier before reaching `max_nodes` (returns the smaller sample with a
  warning rather than failing).

## Out of scope (YAGNI)

- Token edges, USD pricing, `traces` internal transfers.
- Train/test temporal splitting and negative sampling (next phase).
- Node feature engineering and `accounts.is_contract` labeling.
- Any model training code.

## Dependencies

Add `duckdb` (runtime) and `pytest` (dev).

## Cost budget

One-time: ~41 GB extraction + ~26 GB verification count ≈ 7% of the 1 TB
monthly free tier. All sampling iteration is local and free.
