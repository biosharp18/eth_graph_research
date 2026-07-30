# BigQuery Ethereum Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safe, cost-guarded exploratory querying of `bigquery-public-data.goog_blockchain_ethereum_mainnet_us` from a Jupyter notebook.

**Architecture:** A single helper module `eth_graph_research.bq` owns auth and cost control (dry-run estimates + `maximum_bytes_billed` cap). A notebook consumes it with worked example queries. No other code paths talk to BigQuery directly.

**Tech Stack:** Python 3.12+, `uv`, `google-cloud-bigquery`, `pydata-google-auth`, pandas, Jupyter.

## Global Constraints

- GCP project ID: `eth-graph-research` (queries are billed here; 1 TB/month free tier).
- Dataset: `bigquery-public-data.goog_blockchain_ethereum_mainnet_us`. All its tables use MONTHLY time partitioning on `block_timestamp`.
- Default query cap: `max_gb=10` per query. Never run an example query without a `block_timestamp` filter.
- BigQuery bills by bytes **scanned**; `LIMIT` does not reduce cost. Dry runs are free.
- Per the spec, no unit tests: verification is running queries/notebook cells and confirming output.
- All Python runs through `uv run …` from the repo root.
- The spec: `docs/superpowers/specs/2026-07-29-bigquery-exploration-design.md`.

**Column-name caveat:** Google does not publish column-level schemas for this dataset on the web, so the example SQL below uses the community-known names. Every query step dry-runs first (free); a dry run fails fast with "Unrecognized name: <col>" if a column differs. If that happens, consult the schema DataFrame produced by the notebook's `INFORMATION_SCHEMA.COLUMNS` cell (Task 3, cell 6) and substitute the correct name — likely variants: `token_transfers.address` may be `token_address`, `transactions.value` may need `SAFE_CAST` from `BIGNUMERIC` or `STRING`.

---

### Task 1: Commit baseline scaffolding and verify Google auth

**Files:**
- Modify: none (commits existing untracked files; possibly small fixes to `src/eth_graph_research/main.py` if the run reveals errors)

**Interfaces:**
- Consumes: nothing.
- Produces: cached Google user credentials on disk (written by `pydata_google_auth`, typically under `~/.config/pydata/`), so every later task can query BigQuery non-interactively. Also the baseline git commit.

- [ ] **Step 1: Commit the existing project scaffolding**

```bash
git add .gitignore .python-version README.md pyproject.toml uv.lock src/
git commit -m "chore: commit initial uv project scaffolding"
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: completes without error; `.venv` populated.

- [ ] **Step 3: Have the user run the auth flow (interactive — cannot be done by an agent)**

The first run of `main.py` opens a Google login in a browser via a local webserver. The user must do this themselves. Ask the user to type this in the Claude Code prompt (the `!` prefix runs it in-session):

```
! uv run python -m eth_graph_research.main
```

Expected on success: a one-row DataFrame printing the USDC token (`symbol=USDC, name=USD Coin, decimals=6`).

Troubleshooting (report to user, don't guess):
- `403 … project eth-graph-research` / "project not found": the GCP project ID in `main.py:4` doesn't match an existing project the user owns — ask the user for the correct project ID from https://console.cloud.google.com.
- Browser doesn't open (headless/SSH session): `pydata_google_auth` needs a browser reachable from this machine. Options: run once from the user's laptop and copy the cached credentials dir, or SSH port-forward the local auth port the tool prints. Stop and surface this to the user.
- `BigQuery API has not been enabled`: user must enable it at https://console.cloud.google.com/apis/library/bigquery.googleapis.com.

- [ ] **Step 4: Verify credentials are cached (non-interactive re-run)**

Run: `uv run python -m eth_graph_research.main`
Expected: same USDC row, with no browser prompt. This proves later tasks can run unattended.

- [ ] **Step 5: Commit (only if main.py needed fixes)**

```bash
git add src/eth_graph_research/main.py
git commit -m "fix: correct auth/project setup in main.py"
```

---

### Task 2: Cost-guarded query helper `bq.py`

**Files:**
- Create: `src/eth_graph_research/bq.py`

**Interfaces:**
- Consumes: cached credentials from Task 1.
- Produces (used verbatim by Tasks 3–4):
  - `eth_graph_research.bq.DATASET: str` — `"bigquery-public-data.goog_blockchain_ethereum_mainnet_us"`
  - `eth_graph_research.bq.get_client() -> bigquery.Client`
  - `eth_graph_research.bq.dry_run(sql: str) -> float` — prints and returns estimated GB scanned; free.
  - `eth_graph_research.bq.run_query(sql: str, max_gb: float = 10.0) -> pandas.DataFrame` — refuses (BigQuery 400 error, nothing billed) if the query would bill more than `max_gb`.

- [ ] **Step 1: Write the module**

Create `src/eth_graph_research/bq.py`:

```python
"""Cost-guarded BigQuery helpers for exploring public blockchain datasets.

BigQuery bills by bytes SCANNED, not rows returned (LIMIT doesn't help).
Always dry_run() a new query first, and keep run_query()'s max_gb cap on.
"""

import pydata_google_auth
from google.cloud import bigquery

PROJECT_ID = "eth-graph-research"
DATASET = "bigquery-public-data.goog_blockchain_ethereum_mainnet_us"

_client: bigquery.Client | None = None


def get_client() -> bigquery.Client:
    """Return a cached BigQuery client authenticated as the user."""
    global _client
    if _client is None:
        credentials = pydata_google_auth.get_user_credentials(
            scopes=["https://www.googleapis.com/auth/bigquery"],
        )
        _client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
    return _client


def dry_run(sql: str) -> float:
    """Estimate how many GB a query would scan, without running it. Free."""
    config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = get_client().query(sql, job_config=config)
    gb = job.total_bytes_processed / 1e9
    print(f"Dry run: would scan {gb:,.3f} GB")
    return gb


def run_query(sql: str, max_gb: float = 10.0):
    """Run a query and return a DataFrame.

    BigQuery refuses the query (error, nothing billed) if it would bill
    more than max_gb. Raise the cap deliberately, per query, when needed.
    """
    config = bigquery.QueryJobConfig(maximum_bytes_billed=int(max_gb * 1e9))
    job = get_client().query(sql, job_config=config)
    df = job.result().to_dataframe()
    scanned = (job.total_bytes_processed or 0) / 1e9
    billed = (job.total_bytes_billed or 0) / 1e9
    print(f"Scanned {scanned:,.3f} GB (billed {billed:,.3f} GB)")
    return df
```

- [ ] **Step 2: Verify dry_run and run_query against the real dataset**

Run:

```bash
uv run python -c "
from eth_graph_research.bq import dry_run, run_query, DATASET
dry_run(f'SELECT block_number FROM \`{DATASET}.blocks\` LIMIT 5')
df = run_query('SELECT 1 AS x')
assert df.x[0] == 1
print('OK')
"
```

Expected: a dry-run GB estimate line, `Scanned 0.000 GB (billed 0.000 GB)` for `SELECT 1`, then `OK`.

- [ ] **Step 3: Verify the cost guard actually refuses expensive queries**

Run:

```bash
uv run python -c "
from eth_graph_research.bq import run_query, DATASET
try:
    run_query(f'SELECT * FROM \`{DATASET}.transactions\`', max_gb=0.01)
    print('FAIL: guard did not trigger')
except Exception as e:
    assert 'bytes billed' in str(e).lower(), e
    print('OK: guard refused, nothing billed')
"
```

Expected: `OK: guard refused, nothing billed`.

- [ ] **Step 4: Commit**

```bash
git add src/eth_graph_research/bq.py
git commit -m "feat: add cost-guarded BigQuery query helpers"
```

---

### Task 3: Exploration notebook

**Files:**
- Modify: `pyproject.toml` (dev dependency, via `uv add`)
- Create: `notebooks/01_explore_ethereum.ipynb` (generated by a throwaway script; write the script to the session scratchpad directory, not the repo)

**Interfaces:**
- Consumes: `dry_run`, `run_query`, `DATASET` from `eth_graph_research.bq` (Task 2).
- Produces: an executed notebook with saved outputs, committed.

- [ ] **Step 1: Add Jupyter as a dev dependency**

```bash
uv add --dev jupyter
```

Expected: `pyproject.toml` gains a `[dependency-groups] dev = ["jupyter>=…"]` entry; `uv.lock` updated.

- [ ] **Step 2: Generate the notebook with nbformat**

Write the following to `<scratchpad>/make_notebook.py` and run `uv run python <scratchpad>/make_notebook.py` from the repo root. Cell sources below are final content — copy verbatim.

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Exploring Ethereum mainnet on BigQuery

Dataset: `bigquery-public-data.goog_blockchain_ethereum_mainnet_us`

**The cost model in three sentences.** BigQuery charges by bytes *scanned*
(first 1 TB per month is free), and `LIMIT` does **not** reduce scanning.
Cost is controlled by selecting few columns and filtering on the partition
column `block_timestamp` (all tables here are partitioned by month on it).
`dry_run()` shows what a query *would* scan for free, and `run_query()`
refuses anything that would bill more than `max_gb` (default 10 GB)."""),

    code("""import pandas as pd

from eth_graph_research.bq import DATASET, dry_run, run_query

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)
DATASET"""),

    md("""## What tables exist?

`INFORMATION_SCHEMA` queries are metadata-only and cost next to nothing."""),

    code("""run_query(f'''
SELECT table_name
FROM `{DATASET}.INFORMATION_SCHEMA.TABLES`
ORDER BY table_name
''')"""),

    md("""## What columns do the key tables have?

Keep this DataFrame handy — it is the ground truth for column names and
types when you write your own queries."""),

    code("""schema = run_query(f'''
SELECT table_name, column_name, data_type
FROM `{DATASET}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name IN ('blocks', 'transactions', 'token_transfers')
ORDER BY table_name, ordinal_position
''')
schema"""),

    md("""## Dry runs: check cost before you spend it

Three estimates for the `transactions` table:
1. `SELECT *`, no filter — scans the entire table (terabytes!).
2. Same with `LIMIT 10` — **identical cost**. LIMIT is applied after scanning.
3. Two columns + one month of `block_timestamp` — a tiny fraction.

Rule of thumb: never run a query against a big table without a
`block_timestamp` filter, and always `dry_run` anything new."""),

    code("""dry_run(f'SELECT * FROM `{DATASET}.transactions`')
dry_run(f'SELECT * FROM `{DATASET}.transactions` LIMIT 10')
dry_run(f'''
SELECT transaction_hash, from_address
FROM `{DATASET}.transactions`
WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
''')"""),

    md("""## Example 1 — blocks from the last day

`blocks` is a small table; a day of it is cheap. Partitions are monthly, so
a 1-day filter still scans the whole current month's partition for the
selected columns — that's fine here."""),

    code("""run_query(f'''
SELECT
  COUNT(*)              AS n_blocks,
  MIN(block_number)     AS first_block,
  MAX(block_number)     AS last_block,
  ROUND(AVG(gas_used))  AS avg_gas_used
FROM `{DATASET}.blocks`
WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
''')"""),

    md("""## Example 2 — transactions per day, last 7 days

The same shape as Google's own example query for this dataset."""),

    code("""run_query(f'''
SELECT
  TIMESTAMP_TRUNC(block_timestamp, DAY) AS day,
  COUNT(*) AS txn_count
FROM `{DATASET}.transactions`
WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY day
ORDER BY day
''')"""),

    md("""## Example 3 — a peek at recent individual transactions

Selecting few columns keeps the scan small; `value` is in wei
(1 ETH = 1e18 wei). The LIMIT here is for display only — the
`block_timestamp` filter is what keeps it cheap."""),

    code("""run_query(f'''
SELECT
  block_timestamp,
  transaction_hash,
  from_address,
  to_address,
  SAFE_CAST(value AS FLOAT64) / 1e18 AS value_eth
FROM `{DATASET}.transactions`
WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
ORDER BY block_timestamp DESC
LIMIT 20
''')"""),

    md("""## Example 4 — busiest token contracts in the last day

`token_transfers` is decoded ERC-20 `Transfer` events. `address` is the
token contract emitting the event. (If the schema cell above shows a
different column name, use that.)"""),

    code("""run_query(f'''
SELECT
  address AS token_contract,
  COUNT(*) AS transfers
FROM `{DATASET}.token_transfers`
WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
GROUP BY token_contract
ORDER BY transfers DESC
LIMIT 10
''')"""),

    md("""## Where next

- Cross-reference token contracts against `crypto_ethereum.tokens`
  (the community dataset) or `decoded_events` to get symbols/decimals.
- For graph research: `transactions` (`from_address` → `to_address`) and
  `token_transfers` are the edge lists. Decide on a time window, dry-run
  the extraction query, then export.
- Raise `max_gb` per call only when a dry run justifies it:
  `run_query(sql, max_gb=50)`."""),
]

nb.cells = cells
with open("notebooks/01_explore_ethereum.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/01_explore_ethereum.ipynb")
```

- [ ] **Step 3: Dry-run-validate the example queries before executing the notebook**

Run each example's SQL through `dry_run` (free) to catch wrong column names:

```bash
uv run python -c "
from eth_graph_research.bq import dry_run, DATASET
queries = {
  'blocks':   f'SELECT COUNT(*) AS n_blocks, MIN(block_number) AS first_block, MAX(block_number) AS last_block, ROUND(AVG(gas_used)) AS avg_gas_used FROM \`{DATASET}.blocks\` WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)',
  'txn/day':  f'SELECT TIMESTAMP_TRUNC(block_timestamp, DAY) AS day, COUNT(*) AS txn_count FROM \`{DATASET}.transactions\` WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY) GROUP BY day',
  'recent':   f'SELECT block_timestamp, transaction_hash, from_address, to_address, SAFE_CAST(value AS FLOAT64)/1e18 AS value_eth FROM \`{DATASET}.transactions\` WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR) LIMIT 20',
  'tokens':   f'SELECT address AS token_contract, COUNT(*) AS transfers FROM \`{DATASET}.token_transfers\` WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) GROUP BY token_contract ORDER BY transfers DESC LIMIT 10',
}
for name, q in queries.items():
    print(name); dry_run(q)
"
```

Expected: four GB estimates, each well under 10 GB.
If any fails with `Unrecognized name: X`: query `INFORMATION_SCHEMA.COLUMNS` for that table (SQL in notebook cell 6), pick the real column, and update **both** this validation snippet and the corresponding cell in `make_notebook.py`, then regenerate the notebook (Step 2) before proceeding.
If `SAFE_CAST(value AS FLOAT64)` fails because `value` is a STRUCT/RECORD: check the schema output for the value column's actual type and adjust the expression (e.g. `value.string_value` or drop the cast) in the same two places.

- [ ] **Step 4: Execute the notebook end-to-end with outputs saved**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_explore_ethereum.ipynb
```

Expected: exits 0. Then confirm every code cell has output and no cell contains an error:

```bash
uv run python -c "
import nbformat
nb = nbformat.read('notebooks/01_explore_ethereum.ipynb', as_version=4)
errs = [o for c in nb.cells for o in c.get('outputs', []) if o.output_type == 'error']
assert not errs, errs
print('all cells executed cleanly')
"
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock notebooks/01_explore_ethereum.ipynb
git commit -m "feat: add cost-guarded Ethereum exploration notebook"
```

---

### Task 4: README

**Files:**
- Modify: `README.md` (currently empty)

**Interfaces:**
- Consumes: everything above (documents it).
- Produces: onboarding doc.

- [ ] **Step 1: Write README.md**

Replace the empty `README.md` with:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add setup and cost-model README"
```
