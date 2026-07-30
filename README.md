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

## Layout

- `src/eth_graph_research/bq.py` — auth + cost-guarded query helpers
- `notebooks/01_explore_ethereum.ipynb` — worked intro to the dataset
- `docs/superpowers/` — design specs and plans
