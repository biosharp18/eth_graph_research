"""Daily-aggregated stablecoin flow dataset for TGAT.

See docs/superpowers/specs/2026-08-07-tgat-daily-flow-design.md.
"""
from dataclasses import dataclass, field

import duckdb
import numpy as np

STABLE_ASSETS = ["USDT", "USDC", "BUSD", "DAI"]


@dataclass
class DailyGraph:
    src: np.ndarray
    dst: np.ndarray
    day: np.ndarray
    edge_feat: np.ndarray
    y_amt: np.ndarray
    usd_sum: np.ndarray
    n_nodes: int
    n_days: int
    train_end: int
    val_end: int
    amt_mean: float
    amt_std: float
    addr_of: list = field(default_factory=list)


def _snap_to_day_boundary(day: np.ndarray, target_idx: int) -> int:
    """Smallest index i >= start-of-day(day[target_idx]) such that no day straddles."""
    boundary_day = day[target_idx]
    return int(np.searchsorted(day, boundary_day, side="left"))


def load_daily_graph(parquet_path: str, train_frac: float = 0.70,
                     val_frac: float = 0.15) -> DailyGraph:
    con = duckdb.connect()
    assets = ", ".join(f"'{a}'" for a in STABLE_ASSETS)
    df = con.sql(f"""
        SELECT source_eoa AS s, target_eoa AS d,
               CAST(CAST(block_timestamp AS TIMESTAMP) AS DATE) AS day,
               asset, TRY_CAST(amount_human AS DOUBLE) AS amt
        FROM '{parquet_path}'
        WHERE block_timestamp IS NOT NULL AND asset IN ({assets})
    """).df()

    piv = df.pivot_table(index=["s", "d", "day"], columns="asset", values="amt",
                         aggfunc="sum", fill_value=0.0)
    piv = piv.reindex(columns=STABLE_ASSETS, fill_value=0.0)
    cnt = df.groupby(["s", "d", "day"]).size().rename("txn")
    agg = piv.join(cnt).reset_index()
    agg["usd_sum"] = agg[STABLE_ASSETS].sum(axis=1)
    agg = agg.sort_values(["day", "s", "d"], kind="stable").reset_index(drop=True)

    day0 = agg["day"].min()
    day_idx = (agg["day"] - day0).dt.days.to_numpy(np.int64)

    addrs = sorted(set(agg["s"]) | set(agg["d"]))
    aid = {a: i for i, a in enumerate(addrs)}
    src = agg["s"].map(aid).to_numpy(np.int64)
    dst = agg["d"].map(aid).to_numpy(np.int64)

    E = len(agg)
    train_end = _snap_to_day_boundary(day_idx, int(E * train_frac))
    val_end = _snap_to_day_boundary(day_idx, int(E * (train_frac + val_frac)))

    usd = agg["usd_sum"].to_numpy(np.float64)
    log_usd = np.log10(usd)
    amt_mean = float(log_usd[:train_end].mean())
    amt_std = float(log_usd[:train_end].std())
    y_amt = ((log_usd - amt_mean) / amt_std).astype(np.float32)

    shares = (agg[STABLE_ASSETS].to_numpy(np.float64) / usd[:, None])
    edge_feat = np.column_stack([
        y_amt,
        np.log1p(agg["txn"].to_numpy(np.float64)),
        shares,
    ]).astype(np.float32)

    return DailyGraph(
        src=src, dst=dst, day=day_idx, edge_feat=edge_feat, y_amt=y_amt,
        usd_sum=usd, n_nodes=len(addrs), n_days=int(day_idx.max()) + 1,
        train_end=train_end, val_end=val_end,
        amt_mean=amt_mean, amt_std=amt_std, addr_of=addrs,
    )
