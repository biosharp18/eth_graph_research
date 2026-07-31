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
~5-8 GB on disk), then free local sampling:

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
