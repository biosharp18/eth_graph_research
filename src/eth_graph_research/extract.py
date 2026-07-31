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
