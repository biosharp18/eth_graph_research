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
