# BigQuery Exploration Setup for `goog_blockchain_ethereum_mainnet_us`

**Date:** 2026-07-29
**Status:** Approved

## Goal

Get from "never used BigQuery" to safely running exploratory queries against
Google's public Ethereum mainnet dataset
(`bigquery-public-data.goog_blockchain_ethereum_mainnet_us`) in a Jupyter
notebook, without accidentally exhausting the 1 TB/month BigQuery free tier.
This is a stepping stone: the exploration will later inform which data to
download for graph research.

## Background

- The project is a `uv`-managed Python package (`eth_graph_research`) with
  `google-cloud-bigquery`, `pandas`, and `pydata-google-auth` already installed.
- `src/eth_graph_research/main.py` contains a working-in-principle query
  against the older community dataset (`crypto_ethereum.tokens`), but it has
  never been run — auth against the `eth-graph-research` GCP project is
  unverified.
- BigQuery bills per bytes **scanned**, not rows returned. `LIMIT` does not
  reduce cost. The `transactions` table alone is on the order of 2 TB, so an
  unfiltered `SELECT *` would double the monthly free tier in one query.
  Tables are partitioned by `block_timestamp`; filtering on it is the primary
  cost control.

## Design

### 1. Verify auth first

Run the existing `main.py` once to confirm Google login and the
`eth-graph-research` GCP project work. Fix any auth/project issues before
building anything new.

### 2. Cost-guarded query helper — `src/eth_graph_research/bq.py`

A small module (~40 lines) shared by notebooks and future scripts:

- `get_client()` — authenticates via `pydata_google_auth` and returns a
  `bigquery.Client` for project `eth-graph-research`.
- `dry_run(sql)` — runs the query in dry-run mode (costs nothing) and prints
  the estimated GB that would be scanned.
- `run_query(sql, max_gb=10)` — executes the query with
  `maximum_bytes_billed` set from `max_gb`; BigQuery refuses to run the query
  (clear error, no charge) if it would scan more. Returns a pandas DataFrame.

### 3. Exploration notebook — `notebooks/01_explore_ethereum.ipynb`

Sections, each with a short markdown explanation of the cost reasoning:

1. Connect via the helper.
2. List tables in the dataset and inspect a table's schema using
   `INFORMATION_SCHEMA` (near-free).
3. Dry-run demo: show estimated scan size of a bad query vs. a
   partition-filtered one.
4. 4–5 example queries on `blocks`, `transactions`, and `token_transfers`,
   each filtered by `block_timestamp` to a small window.

### 4. Dependencies

Add `jupyter` as a dev dependency via `uv`.

### 5. README

Short "how to run" section: auth steps, starting Jupyter, and the BigQuery
cost model in three sentences.

## Error handling

- The cost guard raises a clear error including the estimated scan size
  instead of billing.
- Auth failures are surfaced and fixed in step 1, before anything else.

## Testing

Exploratory tooling: verification is running the notebook end-to-end and
confirming each cell returns data. No unit tests.

## Out of scope (YAGNI)

- Downloading/exporting bulk data (later phase, informed by this exploration).
- Graph construction or analysis code.
- Any fixed research question — this phase is for learning the dataset.
