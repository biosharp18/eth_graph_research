# ETH Edge Extraction and Snowball Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download a fixed 90-day window of native-ETH transaction edges (~103M rows) to local parquet once, and build a deterministic, unit-tested snowball sampler that produces ~100k-node subgraphs from it for link-prediction training.

**Architecture:** `extract.py` streams one guarded BigQuery query to chunked parquet under `data/raw/` with a provenance manifest. `sample.py` runs hop-batched snowball sampling in DuckDB over that parquet, returning induced subgraphs saved under `data/samples/`. The sampler is pure local logic, unit-tested against synthetic in-memory edge lists.

**Tech Stack:** Python 3.12+, `uv`, `google-cloud-bigquery` + `google-cloud-bigquery-storage` (streaming), pyarrow/parquet, DuckDB, pytest, matplotlib/scipy (analysis notebook only).

## Global Constraints

- Fixed window: `2026-05-01 00:00:00 UTC` inclusive to `2026-07-30 00:00:00 UTC` exclusive. Never use relative dates (`CURRENT_TIMESTAMP`) in extraction SQL.
- Edge definition: rows of `bigquery-public-data.goog_blockchain_ethereum_mainnet_us.transactions` with `value > 0`; columns `block_timestamp, transaction_hash, from_address, to_address, value_lossless` (+ derived `value_eth` FLOAT64).
- Extraction cost cap: `MAX_GB = 50.0` (dry-run estimate was 40.8 GB).
- `data/` is gitignored; nothing under it is ever committed. Provenance lives in `manifest.json` files beside the data.
- Extraction and `save_sample` refuse to overwrite existing output directories.
- Sampler defaults: `max_nodes=100_000`, `per_node_cap=50`; deterministic for a fixed `rng_seed` — never use Python's builtin `hash()` for sampling decisions (it is salted per process); use `hashlib.md5`.
- All commands run from the repo root via `uv run …`.
- Spec: `docs/superpowers/specs/2026-07-30-eth-edge-sampling-design.md`.

---

### Task 1: Extraction script, then launch the real extraction in the background

**Files:**
- Modify: `src/eth_graph_research/bq.py` (add `_get_credentials`, `get_bqstorage_client`)
- Modify: `.gitignore` (add `data/`)
- Create: `src/eth_graph_research/extract.py`

**Interfaces:**
- Consumes: `bq.DATASET`, `bq.dry_run`, `bq.get_client` (existing).
- Produces:
  - `bq.get_bqstorage_client() -> bigquery_storage.BigQueryReadClient`
  - `extract.WINDOW_START = "2026-05-01"`, `extract.WINDOW_END = "2026-07-30"` (strings, used by Task 2's `RAW_GLOB` and Task 3's verification)
  - `extract.COUNT_SQL: str` (used by Task 3)
  - On-disk: `data/raw/eth_edges_2026-05-01_2026-07-30/part-*.parquet` + `manifest.json` with keys `sql, window_start, window_end, row_count, extracted_at, total_bytes_processed`.

- [ ] **Step 1: Add `data/` to `.gitignore`**

Append to `.gitignore`:

```
# Extracted datasets and samples (provenance in manifest.json files)
data/
```

- [ ] **Step 2: Refactor credentials in `bq.py` and add a Storage API client**

In `src/eth_graph_research/bq.py`, replace the body of `get_client()` and add two functions so credentials logic exists once:

```python
def _get_credentials():
    return pydata_google_auth.get_user_credentials(
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )


def get_client() -> bigquery.Client:
    """Return a cached BigQuery client authenticated as the user."""
    global _client
    if _client is None:
        _client = bigquery.Client(project=PROJECT_ID, credentials=_get_credentials())
    return _client


def get_bqstorage_client():
    """Client for the BigQuery Storage read API (fast bulk downloads)."""
    from google.cloud import bigquery_storage

    return bigquery_storage.BigQueryReadClient(credentials=_get_credentials())
```

- [ ] **Step 3: Write `src/eth_graph_research/extract.py`**

```python
"""One-shot extraction of native-ETH transfer edges for a fixed window.

Run: uv run python -m eth_graph_research.extract
Reproducible by construction: fixed date window, exact SQL recorded in
manifest.json next to the data. Refuses to overwrite existing output.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from google.cloud import bigquery

from eth_graph_research.bq import DATASET, dry_run, get_bqstorage_client, get_client

WINDOW_START = "2026-05-01"
WINDOW_END = "2026-07-30"  # exclusive
MAX_GB = 50.0
ROWS_PER_FILE = 5_000_000
RAW_DIR = Path(f"data/raw/eth_edges_{WINDOW_START}_{WINDOW_END}")

_FILTER = f"""
FROM `{DATASET}.transactions`
WHERE block_timestamp >= TIMESTAMP('{WINDOW_START}')
  AND block_timestamp < TIMESTAMP('{WINDOW_END}')
  AND value > 0
"""
SQL = f"""
SELECT block_timestamp, transaction_hash, from_address, to_address, value_lossless
{_FILTER}
"""
COUNT_SQL = f"SELECT COUNT(*) AS n {_FILTER}"


def _add_value_eth(batch: pa.RecordBatch) -> pa.RecordBatch:
    wei = pc.cast(batch.column("value_lossless"), pa.float64())
    return batch.append_column("value_eth", pc.divide(wei, 1e18))


def main() -> None:
    if RAW_DIR.exists():
        sys.exit(f"{RAW_DIR} already exists - delete it first to re-extract.")
    est_gb = dry_run(SQL)
    if est_gb > MAX_GB:
        sys.exit(f"Estimated scan {est_gb:.1f} GB exceeds MAX_GB={MAX_GB}; aborting.")
    RAW_DIR.mkdir(parents=True)

    config = bigquery.QueryJobConfig(maximum_bytes_billed=int(MAX_GB * 1e9))
    job = get_client().query(SQL, job_config=config)
    rows = job.result(page_size=100_000)

    n_rows = 0
    part = 0
    rows_in_file = 0
    writer = None
    try:
        for batch in rows.to_arrow_iterable(bqstorage_client=get_bqstorage_client()):
            batch = _add_value_eth(batch)
            if writer is None:
                writer = pq.ParquetWriter(
                    RAW_DIR / f"part-{part:04d}.parquet", batch.schema
                )
                rows_in_file = 0
            writer.write_batch(batch)
            n_rows += batch.num_rows
            rows_in_file += batch.num_rows
            if rows_in_file >= ROWS_PER_FILE:
                writer.close()
                writer = None
                part += 1
            if n_rows % 10_000_000 < batch.num_rows:
                print(f"{n_rows:,} rows...", flush=True)
    finally:
        if writer is not None:
            writer.close()

    manifest = {
        "sql": SQL,
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "row_count": n_rows,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "total_bytes_processed": job.total_bytes_processed,
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {n_rows:,} rows to {RAW_DIR}")


if __name__ == "__main__":
    main()
```

Note: `manifest.json` is written only after the stream completes, so a crashed run is detectable (directory without manifest) and should be deleted and rerun.

- [ ] **Step 4: Smoke-check imports and the dry-run path**

Run:

```bash
uv run python -c "
from eth_graph_research.extract import SQL, COUNT_SQL, RAW_DIR
from eth_graph_research.bq import dry_run, get_bqstorage_client
gb = dry_run(SQL)
assert 30 < gb < 50, gb
get_bqstorage_client()
print('OK, estimate', round(gb, 1), 'GB')
"
```

Expected: `OK, estimate ~40.8 GB`. (The bqstorage client constructing without error proves auth works for the Storage API.)

- [ ] **Step 5: Commit**

```bash
git add .gitignore src/eth_graph_research/bq.py src/eth_graph_research/extract.py
git commit -m "feat: add guarded one-shot ETH edge extraction to parquet"
```

- [ ] **Step 6: Launch the real extraction in the background**

Run `uv run python -m eth_graph_research.extract` as a background task (it takes tens of minutes: ~103M rows, ~5-8 GB). Do not wait for it — proceed to Task 2, which uses only synthetic data. Task 3 verifies the result.

---

### Task 2: Snowball sampler (TDD)

**Files:**
- Create: `src/eth_graph_research/sample.py`
- Test: `tests/test_sample.py`
- Modify: `pyproject.toml` via `uv add`

**Interfaces:**
- Consumes: parquet layout from Task 1 (only via the `RAW_GLOB` default; tests inject DataFrames instead).
- Produces (used by Task 4):
  - `sample.connect_edges(parquet_glob: str = RAW_GLOB) -> duckdb.DuckDBPyConnection` — connection with an `edges` view (`from_address, to_address, block_timestamp, transaction_hash, value_eth`)
  - `sample.connect_edges_from_dataframe(df: pandas.DataFrame) -> duckdb.DuckDBPyConnection` — same view over an in-memory DataFrame (testing hook)
  - `sample.snowball_sample(con, seed_address: str, max_nodes: int = 100_000, per_node_cap: int = 50, rng_seed: int = 0) -> Sample`
  - `sample.Sample` dataclass: fields `seed: str`, `params: dict`, `nodes: list[str]` (sorted), `edges: pandas.DataFrame` (induced subgraph, deterministic order)
  - `sample.pick_seeds(con, min_tx: int = 50, max_tx: int = 500, limit: int = 20) -> pandas.DataFrame` with columns `address, n_tx`
  - `sample.save_sample(s: Sample, name: str) -> pathlib.Path` — writes `data/samples/<name>/{nodes.parquet, edges.parquet, manifest.json}`

- [ ] **Step 1: Add dependencies**

```bash
uv add duckdb
uv add --dev pytest
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_sample.py`:

```python
import json

import pandas as pd
import pytest

from eth_graph_research.sample import (
    Sample,
    connect_edges_from_dataframe,
    pick_seeds,
    save_sample,
    snowball_sample,
)


def make_edges(pairs):
    """Build a synthetic edge DataFrame from (from, to) tuples."""
    return pd.DataFrame(
        {
            "from_address": [p[0] for p in pairs],
            "to_address": [p[1] for p in pairs],
            "block_timestamp": pd.to_datetime(
                [f"2026-05-{(i % 28) + 1:02d}" for i in range(len(pairs))], utc=True
            ),
            "transaction_hash": [f"0xhash{i}" for i in range(len(pairs))],
            "value_eth": [1.0 + i for i in range(len(pairs))],
        }
    )


def test_missing_seed_raises():
    con = connect_edges_from_dataframe(make_edges([("a", "b")]))
    with pytest.raises(ValueError, match="no edges"):
        snowball_sample(con, "zzz", max_nodes=10, per_node_cap=5, rng_seed=0)


def test_per_node_cap_limits_expansion():
    pairs = [("hub", f"leaf{i}") for i in range(100)]
    con = connect_edges_from_dataframe(make_edges(pairs))
    s = snowball_sample(con, "hub", max_nodes=1000, per_node_cap=5, rng_seed=0)
    # hub itself plus at most 5 sampled counterparties
    assert len(s.nodes) <= 6


def test_max_nodes_never_overshot():
    pairs = [("seed", f"n{i}") for i in range(50)]
    con = connect_edges_from_dataframe(make_edges(pairs))
    s = snowball_sample(con, "seed", max_nodes=10, per_node_cap=50, rng_seed=0)
    assert len(s.nodes) == 10


def test_induced_edges_include_untraversed():
    # Triangle: BFS from s traverses s-a and s-b; induced output must also
    # contain the a-b edge.
    pairs = [("s", "a"), ("s", "b"), ("a", "b")]
    con = connect_edges_from_dataframe(make_edges(pairs))
    s = snowball_sample(con, "s", max_nodes=10, per_node_cap=10, rng_seed=0)
    assert sorted(s.nodes) == ["a", "b", "s"]
    edge_pairs = set(zip(s.edges["from_address"], s.edges["to_address"]))
    assert ("a", "b") in edge_pairs
    assert len(s.edges) == 3


def test_deterministic_given_rng_seed():
    # Complete graph on 10 nodes: plenty of arbitrary choices to pin down.
    pairs = [(f"x{i}", f"x{j}") for i in range(10) for j in range(i + 1, 10)]
    con = connect_edges_from_dataframe(make_edges(pairs))
    s1 = snowball_sample(con, "x0", max_nodes=5, per_node_cap=3, rng_seed=42)
    s2 = snowball_sample(con, "x0", max_nodes=5, per_node_cap=3, rng_seed=42)
    assert s1.nodes == s2.nodes
    pd.testing.assert_frame_equal(s1.edges, s2.edges)


def test_frontier_exhaustion_returns_component_with_warning():
    pairs = [("a", "b"), ("b", "c")]  # disconnected from the rest of nothing
    con = connect_edges_from_dataframe(make_edges(pairs))
    with pytest.warns(UserWarning, match="exhausted"):
        s = snowball_sample(con, "a", max_nodes=100, per_node_cap=10, rng_seed=0)
    assert sorted(s.nodes) == ["a", "b", "c"]


def test_pick_seeds_activity_band():
    # "mid" has 3 edges, "hub" has 30, leaves have 1 each.
    pairs = [("mid", f"m{i}") for i in range(3)] + [
        ("hub", f"h{i}") for i in range(30)
    ]
    con = connect_edges_from_dataframe(make_edges(pairs))
    seeds = pick_seeds(con, min_tx=2, max_tx=10, limit=5)
    assert list(seeds.columns) == ["address", "n_tx"]
    assert seeds["address"].tolist() == ["mid"]
    assert seeds["n_tx"].tolist() == [3]


def test_save_sample_writes_files_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # save_sample writes under ./data/samples/
    pairs = [("s", "a"), ("s", "b")]
    con = connect_edges_from_dataframe(make_edges(pairs))
    s = snowball_sample(con, "s", max_nodes=10, per_node_cap=10, rng_seed=7)
    out = save_sample(s, "unit")
    assert (out / "nodes.parquet").exists()
    assert (out / "edges.parquet").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == "s"
    assert manifest["rng_seed"] == 7
    assert manifest["n_nodes"] == len(s.nodes)
    assert manifest["n_edges"] == len(s.edges)
    with pytest.raises(FileExistsError):
        save_sample(s, "unit")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_sample.py -v`
Expected: FAIL at import time — `ModuleNotFoundError: No module named 'eth_graph_research.sample'`.

- [ ] **Step 4: Write `src/eth_graph_research/sample.py`**

```python
"""Snowball sampling over the locally extracted ETH edge list.

All queries run in DuckDB against local parquet, batched per BFS hop.
Deterministic for a fixed rng_seed: tie-breaking uses md5/duckdb hash of
row content, never Python's salted builtin hash().
"""

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

RAW_GLOB = "data/raw/eth_edges_2026-05-01_2026-07-30/part-*.parquet"

_EDGE_COLS = "from_address, to_address, block_timestamp, transaction_hash, value_eth"


def connect_edges(parquet_glob: str = RAW_GLOB) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with an `edges` view over the raw parquet."""
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW edges AS SELECT {_EDGE_COLS} FROM read_parquet('{parquet_glob}')"
    )
    return con


def connect_edges_from_dataframe(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """Testing hook: an `edges` view over an in-memory DataFrame."""
    con = duckdb.connect()
    con.register("edges_df", df)
    con.execute(f"CREATE VIEW edges AS SELECT {_EDGE_COLS} FROM edges_df")
    return con


@dataclass
class Sample:
    seed: str
    params: dict
    nodes: list[str]
    edges: pd.DataFrame


def _det_key(value: str, rng_seed: int) -> str:
    """Deterministic pseudo-random sort key (process-independent)."""
    return hashlib.md5(f"{value}|{rng_seed}".encode()).hexdigest()


def snowball_sample(
    con: duckdb.DuckDBPyConnection,
    seed_address: str,
    max_nodes: int = 100_000,
    per_node_cap: int = 50,
    rng_seed: int = 0,
) -> Sample:
    """BFS from seed_address, keeping <= per_node_cap random edges per node.

    Expansion stops exactly at max_nodes (never overshoots); the returned
    edge set is the full induced subgraph on the sampled nodes.
    """
    n_seed_edges = con.execute(
        "SELECT count(*) FROM edges WHERE from_address = ? OR to_address = ?",
        [seed_address, seed_address],
    ).fetchone()[0]
    if n_seed_edges == 0:
        raise ValueError(f"Seed {seed_address} has no edges in the window.")

    sampled: set[str] = {seed_address}
    frontier = [seed_address]
    while frontier and len(sampled) < max_nodes:
        con.register("frontier", pd.DataFrame({"node": frontier}))
        neighbors = (
            con.execute(
                f"""
                WITH touching AS (
                    SELECT f.node AS anchor, e.to_address AS counterparty,
                           e.block_timestamp, e.value_eth
                    FROM edges e JOIN frontier f ON e.from_address = f.node
                    UNION ALL
                    SELECT f.node AS anchor, e.from_address AS counterparty,
                           e.block_timestamp, e.value_eth
                    FROM edges e JOIN frontier f ON e.to_address = f.node
                ),
                capped AS (
                    SELECT anchor, counterparty,
                           row_number() OVER (
                               PARTITION BY anchor
                               ORDER BY hash(
                                   anchor || '|' || counterparty || '|'
                                   || CAST(epoch(block_timestamp) AS VARCHAR)
                                   || '|' || CAST(value_eth AS VARCHAR)
                                   || '|' || CAST({rng_seed} AS VARCHAR)
                               )
                           ) AS rn
                    FROM touching
                )
                SELECT DISTINCT counterparty FROM capped
                WHERE rn <= {per_node_cap}
                ORDER BY counterparty
                """
            )
            .df()["counterparty"]
            .tolist()
        )
        new = [n for n in neighbors if n not in sampled]
        room = max_nodes - len(sampled)
        if len(new) > room:
            new = sorted(new, key=lambda a: _det_key(a, rng_seed))[:room]
        sampled.update(new)
        frontier = new

    if len(sampled) < max_nodes:
        warnings.warn(
            f"Frontier exhausted at {len(sampled)} nodes (< {max_nodes})."
        )

    nodes = sorted(sampled)
    con.register("sampled_nodes", pd.DataFrame({"node": nodes}))
    edges = con.execute(
        """
        SELECT e.from_address, e.to_address, e.block_timestamp,
               e.transaction_hash, e.value_eth
        FROM edges e
        JOIN sampled_nodes a ON e.from_address = a.node
        JOIN sampled_nodes b ON e.to_address = b.node
        ORDER BY e.block_timestamp, e.transaction_hash,
                 e.from_address, e.to_address
        """
    ).df()
    params = {
        "max_nodes": max_nodes,
        "per_node_cap": per_node_cap,
        "rng_seed": rng_seed,
    }
    return Sample(seed=seed_address, params=params, nodes=nodes, edges=edges)


def pick_seeds(
    con: duckdb.DuckDBPyConnection,
    min_tx: int = 50,
    max_tx: int = 500,
    limit: int = 20,
) -> pd.DataFrame:
    """Candidate seeds in an activity band: active, but not hubs."""
    return con.execute(
        f"""
        WITH activity AS (
            SELECT address, COUNT(*) AS n_tx FROM (
                SELECT from_address AS address FROM edges
                UNION ALL
                SELECT to_address AS address FROM edges
            ) GROUP BY address
        )
        SELECT address, n_tx FROM activity
        WHERE n_tx BETWEEN {min_tx} AND {max_tx}
        ORDER BY hash(address), address
        LIMIT {limit}
        """
    ).df()


def save_sample(s: Sample, name: str) -> Path:
    """Write nodes/edges parquet + provenance manifest; refuse overwrite."""
    out = Path("data/samples") / name
    if out.exists():
        raise FileExistsError(f"{out} already exists.")
    out.mkdir(parents=True)
    pd.DataFrame({"address": s.nodes}).to_parquet(out / "nodes.parquet")
    s.edges.to_parquet(out / "edges.parquet")
    manifest = {
        "seed": s.seed,
        **s.params,
        "n_nodes": len(s.nodes),
        "n_edges": len(s.edges),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_sample.py -v`
Expected: all 8 tests PASS. If `test_deterministic_given_rng_seed` flakes, the nondeterminism entered through an unordered query — every SELECT feeding a decision or output must carry an ORDER BY (they do in the code above).

- [ ] **Step 6: Commit**

```bash
git add tests/test_sample.py src/eth_graph_research/sample.py pyproject.toml uv.lock
git commit -m "feat: add deterministic snowball sampler over local edge parquet"
```

---

### Task 3: Verify the completed extraction

**Files:** none created — verification only. Requires Task 1's background extraction to have finished.

**Interfaces:**
- Consumes: `extract.COUNT_SQL`, `extract.RAW_DIR` (Task 1); `bq.run_query` (existing); the raw parquet + manifest.

- [ ] **Step 1: Confirm the extraction completed**

Check the background process output ends with `Wrote … rows to data/raw/eth_edges_2026-05-01_2026-07-30` and that `manifest.json` exists. If the directory exists without `manifest.json`, the run crashed: delete the directory and rerun `uv run python -m eth_graph_research.extract` (Task 1 Step 6) before proceeding.

- [ ] **Step 2: Three-way row-count verification (parquet vs manifest vs BigQuery)**

Run:

```bash
uv run python -c "
import duckdb, json
from eth_graph_research.extract import COUNT_SQL, RAW_DIR
from eth_graph_research.bq import run_query

manifest = json.loads((RAW_DIR / 'manifest.json').read_text())
local = duckdb.sql(f\"SELECT COUNT(*) FROM read_parquet('{RAW_DIR}/part-*.parquet')\").fetchone()[0]
remote = int(run_query(COUNT_SQL, max_gb=10).n[0])
print(f'manifest={manifest[\"row_count\"]:,} parquet={local:,} bigquery={remote:,}')
assert manifest['row_count'] == local == remote
print('row counts match')
"
```

Expected: three equal counts (~100M; the window is fixed, so the exact number is stable) and `row counts match`. The COUNT(*) query scans only the `value` column — a few GB, cheap.

- [ ] **Step 3: Sanity-check the data content**

Run:

```bash
uv run python -c "
import duckdb
from eth_graph_research.extract import RAW_DIR
df = duckdb.sql(f'''
    SELECT MIN(block_timestamp) AS t0, MAX(block_timestamp) AS t1,
           MIN(value_eth) AS min_eth, MAX(value_eth) AS max_eth,
           COUNT(*) AS n
    FROM read_parquet('{RAW_DIR}/part-*.parquet')
''').df()
print(df.to_string())
assert str(df.t0[0]) >= '2026-05-01'
assert str(df.t1[0]) < '2026-07-30'
assert df.min_eth[0] > 0
print('window and values sane')
"
```

Expected: timestamps inside the window, `min_eth > 0`, and `window and values sane`. Nothing to commit (data is gitignored).

---

### Task 4: First real sample + analysis notebook + README

**Files:**
- Create: `notebooks/02_sample_analysis.ipynb` (generated via nbformat, same pattern as notebook 01)
- Modify: `README.md`
- Modify: `pyproject.toml` via `uv add`

**Interfaces:**
- Consumes: `sample.connect_edges`, `sample.pick_seeds`, `sample.snowball_sample`, `sample.save_sample` (Task 2); raw parquet (Task 3-verified).
- Produces: `data/samples/sample01_100k/` on disk; executed analysis notebook committed.

- [ ] **Step 1: Add analysis dependencies**

```bash
uv add --dev matplotlib scipy
```

- [ ] **Step 2: Pick a seed and produce the first real sample**

Run (this is minutes of DuckDB scans, not BigQuery — free):

```bash
uv run python -c "
from eth_graph_research.sample import connect_edges, pick_seeds, snowball_sample, save_sample
con = connect_edges()
seeds = pick_seeds(con)
print(seeds.to_string())
seed = seeds.address[0]
s = snowball_sample(con, seed, max_nodes=100_000, per_node_cap=50, rng_seed=0)
out = save_sample(s, 'sample01_100k')
print(f'{len(s.nodes):,} nodes, {len(s.edges):,} edges -> {out}')
"
```

Expected: a 20-row seed table, then `100,000 nodes, <N> edges -> data/samples/sample01_100k`. If it warns `Frontier exhausted` (seed landed in a small component), rerun with `seeds.address[1]` — delete `data/samples/sample01_100k` first.

- [ ] **Step 3: Generate the analysis notebook**

Before writing any plotting code, invoke the `dataviz` skill and let its guidance refine the chart cells' styling; the cell contents below are the functional baseline. Write this generator to the session scratchpad and run it from the repo root:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Sample analysis: `sample01_100k`

Sanity checks on the first snowball sample before any model training:
size, degree distribution, connectivity, hubs, and temporal coverage."""),

    code("""import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAMPLE = Path("../data/samples/sample01_100k")
manifest = json.loads((SAMPLE / "manifest.json").read_text())
nodes = pd.read_parquet(SAMPLE / "nodes.parquet")
edges = pd.read_parquet(SAMPLE / "edges.parquet")
manifest"""),

    code("""print(f"{len(nodes):,} nodes, {len(edges):,} edges")
print(f"edges per node: {len(edges) / len(nodes):.1f}")
edges.head()"""),

    md("""## Degree distribution

Transaction graphs are heavy-tailed: expect a rough power law, plus a
visible effect of the sampler's `per_node_cap` on BFS-traversed nodes."""),

    code("""deg = pd.concat([
    edges.groupby("from_address").size(),
    edges.groupby("to_address").size(),
], axis=1).fillna(0).sum(axis=1).astype(int)

counts = deg.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(counts.index, counts.values, s=8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("degree"); ax.set_ylabel("number of nodes")
ax.set_title("Degree distribution (log-log)")
plt.show()
deg.describe()"""),

    md("""## Connectivity

The sample grew from one seed, so BFS-reached nodes are connected by
construction — but the *induced* graph can still contain nodes whose only
sampled edges were dropped by the cap. Count components on the undirected
graph."""),

    code("""from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

idx = {a: i for i, a in enumerate(nodes.address)}
r = edges.from_address.map(idx).to_numpy()
c = edges.to_address.map(idx).to_numpy()
n = len(nodes)
adj = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
n_comp, labels = connected_components(adj, directed=False)
sizes = np.bincount(labels)
print(f"{n_comp} components; largest = {sizes.max():,} nodes "
      f"({sizes.max() / n:.1%} of sample)")"""),

    md("""## Top hubs

The highest-degree nodes are usually exchanges/contracts. Worth eyeballing:
if one node dominates, consider a lower `per_node_cap` or hub exclusion in
the next sample."""),

    code("""deg.sort_values(ascending=False).head(10).rename("degree").to_frame()"""),

    md("""## Temporal coverage

Link prediction needs edges spread across the window (we will split by
time later). Check edges per day."""),

    code("""per_day = edges.set_index("block_timestamp").resample("1D").size()
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(per_day.index, per_day.values)
ax.set_ylabel("edges/day"); ax.set_title("Sampled edges over the window")
plt.show()"""),

    md("""## Verdict

If the largest component holds most nodes, the degree distribution is
heavy-tailed but not one-node-dominated, and edges cover the whole window,
this sample is fit for building the link-prediction dataset (temporal
split + negative sampling) in the next phase."""),
]

nb.cells = cells
with open("notebooks/02_sample_analysis.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/02_sample_analysis.ipynb")
```

Note: notebook paths use `../data/...` because nbconvert executes with the notebook's directory as cwd.

- [ ] **Step 4: Execute the notebook and check for errors**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_sample_analysis.ipynb
uv run python -c "
import nbformat
nb = nbformat.read('notebooks/02_sample_analysis.ipynb', as_version=4)
errs = [o for c in nb.cells for o in c.get('outputs', []) if o.output_type == 'error']
assert not errs, errs
print('all cells executed cleanly')
"
```

Expected: exits 0, then `all cells executed cleanly`. Review the printed component/degree numbers for the Verdict cell's criteria.

- [ ] **Step 5: Update README**

In `README.md`, replace the `## Layout` section with:

```markdown
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
```

- [ ] **Step 6: Run the full test suite one last time**

Run: `uv run pytest`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add notebooks/02_sample_analysis.ipynb README.md pyproject.toml uv.lock
git commit -m "feat: add first 100k-node sample and analysis notebook"
```
