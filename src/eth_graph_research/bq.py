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
