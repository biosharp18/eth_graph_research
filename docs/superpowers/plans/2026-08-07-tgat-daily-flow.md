# TGAT Daily Flow Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate a two-headed TGAT (link BCE + amount Huber) on daily-aggregated stablecoin flows of the OFAC k=4 graph, compared against EdgeBank recomputed on the identical daily stream.

**Architecture:** A `src/tgat/` package: `data.py` builds the daily-edge tensor dataset with day-snapped chronological splits; `neighbors.py` provides a strictly-past recent-k temporal neighbor store; `model.py` implements functional time encoding, temporal attention, and the two heads; `train.py` is the training CLI; `evaluate.py` recomputes EdgeBank, builds the three NS pools, and writes all metrics to JSON.

**Tech Stack:** Python 3.11+/uv, torch (new `ml` dependency group), duckdb + pandas + numpy (already present), pytest.

**Spec:** `docs/superpowers/specs/2026-08-07-tgat-daily-flow-design.md` — read it before starting.

## Global Constraints

- Assets kept: `USDT, USDC, BUSD, DAI` only, valued 1:1 USD. Rows with null `block_timestamp` dropped.
- Prediction unit: directed `(source_eoa, target_eoa, UTC day)`; day index 0–238 is the TGAT timestamp.
- Split: 70/15/15 chronological by daily-edge index, snapped so no day straddles splits.
- Amount target: `zscore(log10(usd_sum))` with mean/std from the **train slice only**.
- Loss: `BCE + 1.0 * Huber(δ=1)`, Huber only on positives.
- Neighbor sampling: 20 most recent, strictly past (`day' < day`), both directions with a direction flag.
- Model dims: 2 layers, 2 heads, d=100, dropout 0.1, Adam lr 1e-4, batch 200, max 50 epochs, patience 5.
- Seeds: 0–4 everywhere; all randomness through `numpy.random.default_rng(seed)` / `torch.manual_seed(seed)`.
- Every `uv run` command from repo root `/xuanwu-tank/west/gaorory/projects/eth-graph-research`.

---

### Task 1: Package skeleton, dependencies, data module

**Files:**
- Modify: `pyproject.toml` (add `ml` dependency group)
- Create: `src/tgat/__init__.py` (empty), `src/tgat/data.py`
- Test: `tests/test_tgat_data.py`

**Interfaces:**
- Produces: `tgat.data.load_daily_graph(parquet_path: str, train_frac=0.70, val_frac=0.15) -> DailyGraph` where `DailyGraph` is a dataclass with fields `src, dst, day (np.int64 [E], sorted by day asc, ties by (src,dst))`, `edge_feat (np.float32 [E,6])`, `y_amt (np.float32 [E])`, `usd_sum (np.float64 [E])`, `n_nodes (int)`, `n_days (int)`, `train_end (int)`, `val_end (int)`, `amt_mean (float)`, `amt_std (float)`, `addr_of (list[str])`.
- Edge feature columns, in order: `[z_log_usd, log1p_txcount, share_USDT, share_USDC, share_BUSD, share_DAI]`.

- [ ] **Step 1: Add dependencies**

```bash
uv add --group ml torch
uv add --group dev pytest   # skip if already in pyproject
mkdir -p src/tgat && touch src/tgat/__init__.py
```

Then verify: `uv run --group ml python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_tgat_data.py
import numpy as np
import pytest
from tgat.data import load_daily_graph

PARQUET = "flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet"

@pytest.fixture(scope="module")
def g():
    return load_daily_graph(PARQUET)

def test_counts_match_profiling(g):
    assert len(g.src) == 41161          # daily stablecoin edges
    assert g.n_days == 239
    assert g.n_nodes > 10000

def test_sorted_by_day(g):
    assert (np.diff(g.day) >= 0).all()

def test_split_snapped_to_day_boundaries(g):
    # the last train day must differ from the first val day, etc.
    assert g.day[g.train_end - 1] != g.day[g.train_end]
    assert g.day[g.val_end - 1] != g.day[g.val_end]
    # splits are roughly 70/15/15
    assert abs(g.train_end / len(g.src) - 0.70) < 0.03
    assert abs(g.val_end / len(g.src) - 0.85) < 0.03

def test_zscore_stats_from_train_only(g):
    logs = np.log10(g.usd_sum[: g.train_end])
    assert g.amt_mean == pytest.approx(logs.mean(), rel=1e-6)
    assert g.amt_std == pytest.approx(logs.std(), rel=1e-6)
    # y_amt of train slice is standard-normal-ish
    assert g.y_amt[: g.train_end].mean() == pytest.approx(0.0, abs=1e-5)

def test_asset_shares_sum_to_one(g):
    shares = g.edge_feat[:, 2:6].sum(axis=1)
    np.testing.assert_allclose(shares, 1.0, atol=1e-5)

def test_feature_dtypes(g):
    assert g.edge_feat.dtype == np.float32
    assert g.src.dtype == np.int64
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --group ml pytest tests/test_tgat_data.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tgat'` (the `src/` layout may need `pythonpath` — if so add to pyproject: `[tool.pytest.ini_options] pythonpath = ["src"]`).

- [ ] **Step 4: Implement `src/tgat/data.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --group ml pytest tests/test_tgat_data.py -x -q`
Expected: 6 passed. If `test_counts_match_profiling` fails on the exact 41161, investigate (filters must match the spec) — do not loosen the assertion.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/tgat tests/test_tgat_data.py
git commit -m "feat(tgat): daily stablecoin flow dataset with day-snapped splits"
```

---

### Task 2: Temporal neighbor store

**Files:**
- Create: `src/tgat/neighbors.py`
- Test: `tests/test_tgat_neighbors.py`

**Interfaces:**
- Consumes: `DailyGraph` fields (`src, dst, day, edge_feat`).
- Produces: `tgat.neighbors.NeighborStore(src, dst, day, edge_feat, n_nodes, k=20)` with method `sample(nodes: np.ndarray[B], days: np.ndarray[B]) -> tuple[nbr_node [B,k] int64, nbr_dt [B,k] float32, nbr_feat [B,k,F+1] float32, mask [B,k] bool]`. `nbr_feat`'s last column is the direction flag (1.0 = the queried node was the source/sender of that past edge, 0.0 = receiver). `nbr_dt = query_day - edge_day >= 1` where mask is True. Masked slots are zero-filled with `mask=False`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tgat_neighbors.py
import numpy as np
from tgat.neighbors import NeighborStore

def tiny_store(k=3):
    # node 0 sends to 1 on days 1,2,5 ; node 2 sends to 0 on day 4
    src = np.array([0, 0, 2, 0], dtype=np.int64)
    dst = np.array([1, 1, 0, 1], dtype=np.int64)
    day = np.array([1, 2, 4, 5], dtype=np.int64)
    feat = np.arange(8, dtype=np.float32).reshape(4, 2)
    return NeighborStore(src, dst, day, feat, n_nodes=3, k=k)

def test_strictly_past_only():
    st = tiny_store()
    nbr, dt, feat, mask = st.sample(np.array([0]), np.array([4]))
    # day-4 edge (2->0) must NOT be visible at query day 4
    assert mask[0].sum() == 2            # only day-1 and day-2 edges
    assert (dt[0][mask[0]] >= 1).all()

def test_recency_takes_last_k():
    st = tiny_store(k=2)
    nbr, dt, feat, mask = st.sample(np.array([0]), np.array([6]))
    # node 0 has 4 past edges at day 6; k=2 keeps the two most recent (days 4,5)
    assert sorted(dt[0][mask[0]].tolist()) == [1.0, 2.0]

def test_direction_flag():
    st = tiny_store()
    nbr, dt, feat, mask = st.sample(np.array([0]), np.array([6]))
    flags = feat[0, mask[0], -1]
    # 0 sent 3 edges (flag 1) and received 1 (flag 0)
    assert flags.sum() == 3.0

def test_isolated_node_all_masked():
    st = tiny_store()
    nbr, dt, feat, mask = st.sample(np.array([2]), np.array([1]))
    assert not mask[0].any()
    assert (feat[0] == 0).all()

def test_property_never_future():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 30, 500).astype(np.int64)
    dst = rng.integers(0, 30, 500).astype(np.int64)
    day = np.sort(rng.integers(0, 60, 500)).astype(np.int64)
    feat = rng.random((500, 2), dtype=np.float32)
    st = NeighborStore(src, dst, day, feat, n_nodes=30, k=10)
    q_nodes = rng.integers(0, 30, 100).astype(np.int64)
    q_days = rng.integers(0, 60, 100).astype(np.int64)
    _, dt, _, mask = st.sample(q_nodes, q_days)
    assert (dt[mask] >= 1).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group ml pytest tests/test_tgat_neighbors.py -x -q`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: NeighborStore`.

- [ ] **Step 3: Implement `src/tgat/neighbors.py`**

```python
"""Strictly-past, recent-k temporal neighbor lookup over daily edges."""
import numpy as np


class NeighborStore:
    def __init__(self, src, dst, day, edge_feat, n_nodes: int, k: int = 20):
        self.k = k
        F = edge_feat.shape[1]
        self.F = F + 1  # + direction flag
        # per-node arrays sorted by day (input is already day-sorted; concat keeps order)
        nbr, days, feats = [[] for _ in range(n_nodes)], [[] for _ in range(n_nodes)], [[] for _ in range(n_nodes)]
        for i in range(len(src)):
            s, d = int(src[i]), int(dst[i])
            f_out = np.append(edge_feat[i], 1.0).astype(np.float32)
            f_in = np.append(edge_feat[i], 0.0).astype(np.float32)
            nbr[s].append(d); days[s].append(day[i]); feats[s].append(f_out)
            nbr[d].append(s); days[d].append(day[i]); feats[d].append(f_in)
        self.nbr = [np.array(x, dtype=np.int64) for x in nbr]
        self.days = [np.array(x, dtype=np.int64) for x in days]
        self.feats = [np.stack(x) if x else np.zeros((0, self.F), np.float32) for x in feats]

    def sample(self, nodes: np.ndarray, qdays: np.ndarray):
        B, k = len(nodes), self.k
        out_nbr = np.zeros((B, k), np.int64)
        out_dt = np.zeros((B, k), np.float32)
        out_feat = np.zeros((B, k, self.F), np.float32)
        out_mask = np.zeros((B, k), bool)
        for b in range(B):
            n, qd = int(nodes[b]), int(qdays[b])
            cut = int(np.searchsorted(self.days[n], qd, side="left"))
            lo = max(0, cut - k)
            m = cut - lo
            if m == 0:
                continue
            out_nbr[b, :m] = self.nbr[n][lo:cut]
            out_dt[b, :m] = qd - self.days[n][lo:cut]
            out_feat[b, :m] = self.feats[n][lo:cut]
            out_mask[b, :m] = True
        return out_nbr, out_dt, out_feat, out_mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --group ml pytest tests/test_tgat_neighbors.py -x -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tgat/neighbors.py tests/test_tgat_neighbors.py
git commit -m "feat(tgat): strictly-past recent-k temporal neighbor store"
```

---

### Task 3: Model — time encoding, temporal attention, two heads

**Files:**
- Create: `src/tgat/model.py`
- Test: `tests/test_tgat_model.py`

**Interfaces:**
- Consumes: `NeighborStore.sample` output shapes from Task 2.
- Produces:
  - `tgat.model.TimeEncoder(dim)` — `forward(dt: FloatTensor [B,k]) -> [B,k,dim]`.
  - `tgat.model.TGAT(edge_feat_dim: int, dim=100, n_layers=2, n_heads=2, dropout=0.1)` with:
    - `embed(nodes: np.ndarray[B], days: np.ndarray[B], store: NeighborStore, device) -> FloatTensor [B, dim]` (recursive over `n_layers`),
    - `forward(src_nodes, dst_nodes, days, store, device) -> (link_logit [B], amt_pred [B])`.
- `edge_feat_dim` passed to TGAT is the store's `F` (data F=6 plus direction flag = 7).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tgat_model.py
import numpy as np
import torch
from tgat.model import TimeEncoder, TGAT
from tgat.neighbors import NeighborStore

def rand_store(n_nodes=20, E=200, F=6, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_nodes, E).astype(np.int64)
    dst = rng.integers(0, n_nodes, E).astype(np.int64)
    day = np.sort(rng.integers(0, 50, E)).astype(np.int64)
    feat = rng.random((E, F), dtype=np.float32)
    return NeighborStore(src, dst, day, feat, n_nodes, k=5)

def test_time_encoder_shape_and_grad():
    te = TimeEncoder(16)
    dt = torch.rand(4, 5)
    out = te(dt)
    assert out.shape == (4, 5, 16)
    out.sum().backward()
    assert te.w.grad is not None

def test_embed_shape():
    st = rand_store()
    m = TGAT(edge_feat_dim=7, dim=32, n_layers=2, n_heads=2)
    z = m.embed(np.arange(6), np.full(6, 30), st, torch.device("cpu"))
    assert z.shape == (6, 32)

def test_forward_returns_logit_and_amount():
    st = rand_store()
    m = TGAT(edge_feat_dim=7, dim=32)
    logit, amt = m(np.arange(4), np.arange(4, 8), np.full(4, 30), st, torch.device("cpu"))
    assert logit.shape == (4,) and amt.shape == (4,)

def test_histories_differentiate_nodes():
    # zero node features: embeddings must still differ when histories differ
    st = rand_store()
    m = TGAT(edge_feat_dim=7, dim=32)
    torch.manual_seed(0)
    z = m.embed(np.array([0, 1]), np.array([40, 40]), st, torch.device("cpu"))
    assert not torch.allclose(z[0], z[1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group ml pytest tests/test_tgat_model.py -x -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `src/tgat/model.py`**

```python
"""Two-headed TGAT (Xu et al., ICLR 2020) over daily aggregated edges."""
import numpy as np
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """phi(dt) = cos(dt * w + b), learnable."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(
            1.0 / 10 ** np.linspace(0, 4, dim, dtype=np.float32)))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:  # [B,k] -> [B,k,dim]
        return torch.cos(dt.unsqueeze(-1) * self.w + self.b)


class TemporalAttention(nn.Module):
    """One multi-head attention layer over a node's past edges."""

    def __init__(self, dim, edge_feat_dim, time_dim, n_heads, dropout):
        super().__init__()
        kv_dim = dim + edge_feat_dim + time_dim
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          kdim=kv_dim, vdim=kv_dim,
                                          batch_first=True)
        self.merge = nn.Sequential(
            nn.Linear(dim + dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim))

    def forward(self, node_h, nbr_h, nbr_feat, nbr_tenc, mask):
        # node_h [B,d]; nbr_h [B,k,d]; nbr_feat [B,k,Fe]; nbr_tenc [B,k,dt]; mask [B,k]
        kv = torch.cat([nbr_h, nbr_feat, nbr_tenc], dim=-1)
        q = node_h.unsqueeze(1)
        no_nbr = ~mask.any(dim=1)
        pad = ~mask
        pad[no_nbr, 0] = False  # avoid all-masked NaN; output overwritten below
        out, _ = self.attn(q, kv, kv, key_padding_mask=pad)
        out = out.squeeze(1)
        out = torch.where(no_nbr.unsqueeze(1), torch.zeros_like(out), out)
        return self.merge(torch.cat([out, node_h], dim=-1))


class TGAT(nn.Module):
    def __init__(self, edge_feat_dim: int, dim: int = 100, n_layers: int = 2,
                 n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim, self.n_layers = dim, n_layers
        self.time_enc = TimeEncoder(dim)
        self.h0 = nn.Parameter(torch.zeros(dim))  # shared "zero feature" embedding
        self.layers = nn.ModuleList([
            TemporalAttention(dim, edge_feat_dim, dim, n_heads, dropout)
            for _ in range(n_layers)])
        self.link_head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, 1))
        self.amt_head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, 1))

    def _embed(self, nodes, days, store, device, depth):
        B = len(nodes)
        if depth == 0:
            return self.h0.expand(B, self.dim)
        nbr, dt, feat, mask = store.sample(nodes, days)
        k = nbr.shape[1]
        # recurse: neighbor embeddings at one layer shallower, at the edge's day
        nbr_days = (days[:, None] - dt.astype(np.int64)).reshape(-1)
        nbr_h = self._embed(nbr.reshape(-1), nbr_days, store, device,
                            depth - 1).reshape(B, k, self.dim)
        node_h = self._embed(nodes, days, store, device, 0)
        t_dt = torch.from_numpy(dt).to(device)
        return self.layers[depth - 1](
            node_h,
            nbr_h,
            torch.from_numpy(feat).to(device),
            self.time_enc(t_dt),
            torch.from_numpy(mask).to(device),
        )

    def embed(self, nodes, days, store, device):
        return self._embed(np.asarray(nodes), np.asarray(days), store, device,
                           self.n_layers)

    def forward(self, src_nodes, dst_nodes, days, store, device):
        zs = self.embed(src_nodes, days, store, device)
        zd = self.embed(dst_nodes, days, store, device)
        pair = torch.cat([zs, zd], dim=-1)
        return self.link_head(pair).squeeze(-1), self.amt_head(pair).squeeze(-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --group ml pytest tests/test_tgat_model.py -x -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tgat/model.py tests/test_tgat_model.py
git commit -m "feat(tgat): time encoding, temporal attention, two-headed TGAT"
```

---

### Task 4: Training loop and CLI

**Files:**
- Create: `src/tgat/train.py`
- Test: `tests/test_tgat_train.py`

**Interfaces:**
- Consumes: `load_daily_graph`, `NeighborStore`, `TGAT` as defined above.
- Produces: `tgat.train.train_one(g: DailyGraph, seed: int, device, epochs=50, patience=5, batch=200, lam=1.0, lr=1e-4, dim=100, k=20) -> tuple[TGAT, dict]` where the dict has keys `best_val_ap (float)`, `epochs_run (int)`, `history (list[dict])`. Also a CLI: `uv run --group ml python -m tgat.train --parquet <path> --seeds 0 1 2 3 4 --out figures/tgat_models/` that saves `tgat_seed{n}.pt` state dicts plus a `train_log.json`.
- Also produces `tgat.train.sample_training_negatives(rng, src, dst, day, n_nodes, pos_pairs_by_day) -> np.ndarray` (negative dst array, same length as src).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tgat_train.py
import numpy as np
import torch
from tgat.data import DailyGraph
from tgat.train import sample_training_negatives, train_one

def toy_graph(E=2000, n_nodes=100, n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    day = np.sort(rng.integers(0, n_days, E)).astype(np.int64)
    # planted structure: even nodes send to node+1 repeatedly
    src = (2 * rng.integers(0, n_nodes // 2, E)).astype(np.int64)
    dst = src + 1
    usd = 10 ** rng.normal(4, 1, E)
    y = ((np.log10(usd) - 4) / 1).astype(np.float32)
    feat = np.column_stack([y, np.ones(E), np.ones(E), np.zeros(E),
                            np.zeros(E), np.zeros(E)]).astype(np.float32)
    return DailyGraph(src=src, dst=dst, day=day, edge_feat=feat, y_amt=y,
                      usd_sum=usd, n_nodes=n_nodes, n_days=n_days,
                      train_end=int(E * .7), val_end=int(E * .85),
                      amt_mean=4.0, amt_std=1.0)

def test_training_negatives_avoid_same_day_collisions():
    g = toy_graph()
    pos_by_day = {}
    for s, d, t in zip(g.src, g.dst, g.day):
        pos_by_day.setdefault(int(t), set()).add((int(s), int(d)))
    rng = np.random.default_rng(0)
    neg = sample_training_negatives(rng, g.src[:500], g.dst[:500], g.day[:500],
                                    g.n_nodes, pos_by_day)
    assert len(neg) == 500
    for s, nd, t in zip(g.src[:500], neg, g.day[:500]):
        assert nd != s
        assert (int(s), int(nd)) not in pos_by_day[int(t)]

def test_smoke_train_learns_planted_structure():
    g = toy_graph()
    model, info = train_one(g, seed=0, device=torch.device("cpu"),
                            epochs=2, batch=200, dim=32, k=5)
    assert info["epochs_run"] >= 1
    assert info["history"][-1]["train_loss"] < info["history"][0]["train_loss"]
    assert info["best_val_ap"] > 0.6   # planted even->odd structure is learnable
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group ml pytest tests/test_tgat_train.py -x -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `src/tgat/train.py`**

```python
"""Training loop + CLI for the two-headed TGAT."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import DailyGraph, load_daily_graph
from .model import TGAT
from .neighbors import NeighborStore


def sample_training_negatives(rng, src, dst, day, n_nodes, pos_pairs_by_day):
    neg = np.empty(len(src), np.int64)
    for i, (s, t) in enumerate(zip(src, day)):
        pos = pos_pairs_by_day[int(t)]
        while True:
            c = int(rng.integers(0, n_nodes))
            if c != int(s) and (int(s), c) not in pos:
                neg[i] = c
                break
    return neg


def average_precision(y, score):
    order = np.argsort(-score, kind="stable")
    y = np.asarray(y)[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def _pos_pairs_by_day(g: DailyGraph):
    out = {}
    for s, d, t in zip(g.src, g.dst, g.day):
        out.setdefault(int(t), set()).add((int(s), int(d)))
    return out


def train_one(g: DailyGraph, seed: int, device, epochs=50, patience=5,
              batch=200, lam=1.0, lr=1e-4, dim=100, k=20):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=k)
    model = TGAT(edge_feat_dim=store.F, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_by_day = _pos_pairs_by_day(g)

    val_sl = slice(g.train_end, g.val_end)
    val_neg = sample_training_negatives(
        rng, g.src[val_sl], g.dst[val_sl], g.day[val_sl], g.n_nodes, pos_by_day)

    best_ap, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for lo in range(0, g.train_end, batch):
            hi = min(lo + batch, g.train_end)
            s, d, t = g.src[lo:hi], g.dst[lo:hi], g.day[lo:hi]
            nd = sample_training_negatives(rng, s, d, t, g.n_nodes, pos_by_day)
            logit_p, amt_p = model(s, d, t, store, device)
            logit_n, _ = model(s, nd, t, store, device)
            y_amt = torch.from_numpy(g.y_amt[lo:hi]).to(device)
            loss = (F.binary_cross_entropy_with_logits(
                        logit_p, torch.ones_like(logit_p))
                    + F.binary_cross_entropy_with_logits(
                        logit_n, torch.zeros_like(logit_n))) / 2
            loss = loss + lam * F.huber_loss(amt_p, y_amt, delta=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))

        model.eval()
        with torch.no_grad():
            lp, _ = model(g.src[val_sl], g.dst[val_sl], g.day[val_sl], store, device)
            ln, _ = model(g.src[val_sl], val_neg, g.day[val_sl], store, device)
        score = torch.cat([lp, ln]).cpu().numpy()
        y = np.r_[np.ones(len(lp)), np.zeros(len(ln))]
        ap = average_precision(y, score)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_ap": ap})
        if ap > best_ap:
            best_ap, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"best_val_ap": best_ap, "epochs_run": len(history),
                   "history": history}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet",
                     default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap_.add_argument("--out", default="figures/tgat_models")
    ap_.add_argument("--epochs", type=int, default=50)
    ap_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap_.parse_args()

    g = load_daily_graph(args.parquet)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = {}
    for seed in args.seeds:
        model, info = train_one(g, seed, torch.device(args.device),
                                epochs=args.epochs)
        torch.save(model.state_dict(), out / f"tgat_seed{seed}.pt")
        log[seed] = info
        print(f"seed {seed}: best val AP {info['best_val_ap']:.4f} "
              f"({info['epochs_run']} epochs)")
    (out / "train_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --group ml pytest tests/test_tgat_train.py -x -q`
Expected: 2 passed (smoke test takes ~1–2 min on CPU).

- [ ] **Step 5: Commit**

```bash
git add src/tgat/train.py tests/test_tgat_train.py
git commit -m "feat(tgat): training loop with hurdle loss and early stopping"
```

---

### Task 5: Evaluation — NS pools, EdgeBank recompute, metrics, stratification

**Files:**
- Create: `src/tgat/evaluate.py`
- Test: `tests/test_tgat_evaluate.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `tgat.evaluate.auroc(y, score) -> float` (rank-based, tie-safe) and re-export of `average_precision`.
  - `tgat.evaluate.build_negative_pools(g: DailyGraph, strategy: str, seed: int) -> np.ndarray` — for each test daily-edge, one negative `(s, d)` pair, `strategy in {"random","historical","inductive"}` per the DGB definitions (day-level pools, random top-up when short).
  - `tgat.evaluate.edgebank_scores(g, variant: str, pos, negs) -> ...` — streaming daily EdgeBank (`variant in {"inf","tw"}`).
  - CLI `uv run --group ml python -m tgat.evaluate --models figures/tgat_models --out figures/tgat_results.json` producing JSON with structure:
    `{"edgebank": {variant: {strategy: {"auroc": [5 floats], "ap": [...]}}}, "tgat": {strategy: {"auroc": [...], "ap": [...], "auroc_seen": [...], "auroc_unseen": [...], "per_day": {...}}}, "amount": {"tgat_rmse": [...], "tgat_mae": [...], "persistence_rmse": float, "median_rmse": float, "tgat_rmse_seen": [...], "persistence_rmse_seen": float, ...}, "split": {...}}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tgat_evaluate.py
import numpy as np
from tgat.evaluate import auroc, build_negative_pools, edgebank_scores
from tests.test_tgat_train import toy_graph

def test_auroc_matches_hand_computation():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.5, 0.1])
    # pairs: (0.9>0.5),(0.9>0.1),(0.4<0.5),(0.4>0.1) -> 3/4
    assert auroc(y, s) == 0.75

def test_auroc_ties_give_half_credit():
    y = np.array([1, 0])
    s = np.array([0.5, 0.5])
    assert auroc(y, s) == 0.5

def test_historical_negatives_seen_before_test():
    g = toy_graph()
    seen = set(zip(g.src[:g.val_end].tolist(), g.dst[:g.val_end].tolist()))
    pos_by_day = {}
    for s, d, t in zip(g.src, g.dst, g.day):
        pos_by_day.setdefault(int(t), set()).add((int(s), int(d)))
    negs = build_negative_pools(g, "historical", seed=0)
    test = slice(g.val_end, len(g.src))
    n_hist = sum((int(s), int(d)) in seen for (s, d) in negs)
    assert n_hist / len(negs) > 0.5      # mostly drawn from the seen pool
    for (s, d), t in zip(negs, g.day[test]):
        assert (int(s), int(d)) not in pos_by_day[int(t)]   # never a same-day positive

def test_inductive_negatives_unseen_before_test():
    g = toy_graph()
    seen = set(zip(g.src[:g.val_end].tolist(), g.dst[:g.val_end].tolist()))
    negs = build_negative_pools(g, "inductive", seed=0)
    in_seen = sum((int(s), int(d)) in seen for (s, d) in negs)
    assert in_seen == 0 or in_seen / len(negs) < 0.05  # only random top-up may collide... with E_all excluded, 0

def test_edgebank_inf_scores_memory_membership():
    g = toy_graph()
    negs = build_negative_pools(g, "random", seed=0)
    pos_sc, neg_sc = edgebank_scores(g, "inf", negs)
    # planted graph: every test pair was seen in train -> all positives remembered
    assert pos_sc.mean() > 0.95
    assert neg_sc.mean() < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group ml pytest tests/test_tgat_evaluate.py -x -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `src/tgat/evaluate.py`**

```python
"""Evaluation: NS pools, streaming EdgeBank on daily edges, metrics, JSON."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import DailyGraph, load_daily_graph
from .model import TGAT
from .neighbors import NeighborStore
from .train import average_precision


def auroc(y, score):
    y = np.asarray(y); score = np.asarray(score, dtype=np.float64)
    ranks = pd.Series(score).rank(method="average").to_numpy()
    npos = int(y.sum()); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _pair_sets(g: DailyGraph):
    all_pairs = set(zip(g.src.tolist(), g.dst.tolist()))
    before = set(zip(g.src[:g.val_end].tolist(), g.dst[:g.val_end].tolist()))
    test_pairs = set(zip(g.src[g.val_end:].tolist(), g.dst[g.val_end:].tolist()))
    return all_pairs, before, test_pairs - before


def _pos_by_day(g):
    out = {}
    for s, d, t in zip(g.src, g.dst, g.day):
        out.setdefault(int(t), set()).add((int(s), int(d)))
    return out


def build_negative_pools(g: DailyGraph, strategy: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_pairs, before, induct = _pair_sets(g)
    pos_by_day = _pos_by_day(g)
    negs = []
    test_days = g.day[g.val_end:]
    for day in np.unique(test_days):
        idx = np.where(test_days == day)[0]
        npos = len(idx)
        e_t = pos_by_day[int(day)]
        chosen = []
        if strategy in ("historical", "inductive"):
            pool = list((before if strategy == "historical" else induct) - e_t)
            take = min(npos, len(pool))
            for i in rng.choice(len(pool), size=take, replace=False):
                chosen.append(pool[i])
        while len(chosen) < npos:  # random NS and top-up
            s = int(rng.integers(0, g.n_nodes)); d = int(rng.integers(0, g.n_nodes))
            if s != d and (s, d) not in all_pairs:
                chosen.append((s, d))
        negs.extend(chosen)
    return np.array(negs, dtype=np.int64)  # [n_test, 2]


def edgebank_scores(g: DailyGraph, variant: str, negs: np.ndarray):
    test_days = g.day[g.val_end:]
    w = int(test_days.max() - test_days.min()) + 1
    last = {}
    for s, d, t in zip(g.src[:g.val_end], g.dst[:g.val_end], g.day[:g.val_end]):
        last[(int(s), int(d))] = int(t)
    pos_sc = np.zeros(len(test_days)); neg_sc = np.zeros(len(test_days))

    def hit(pair, day):
        if pair not in last:
            return 0.0
        return 1.0 if variant == "inf" or last[pair] > day - w else 0.0

    i = 0
    for day in np.unique(test_days):
        idx = np.where(test_days == day)[0] + g.val_end
        for j in idx:
            pair = (int(g.src[j]), int(g.dst[j]))
            pos_sc[i] = hit(pair, int(day))
            neg_sc[i] = hit((int(negs[i][0]), int(negs[i][1])), int(day))
            i += 1
        for j in idx:  # update after scoring the day
            last[(int(g.src[j]), int(g.dst[j]))] = int(day)
    return pos_sc, neg_sc


def tgat_scores(model, g, store, negs, device, batch=500):
    test = slice(g.val_end, len(g.src))
    s, d, t = g.src[test], g.dst[test], g.day[test]
    ns, nd = negs[:, 0], negs[:, 1]
    outs = []
    model.eval()
    with torch.no_grad():
        for arr_s, arr_d in ((s, d), (ns, nd)):
            sc, am = [], []
            for lo in range(0, len(arr_s), batch):
                logit, amt = model(arr_s[lo:lo + batch], arr_d[lo:lo + batch],
                                   t[lo:lo + batch], store, device)
                sc.append(logit.cpu().numpy()); am.append(amt.cpu().numpy())
            outs.append((np.concatenate(sc), np.concatenate(am)))
    (pos_sc, pos_amt), (neg_sc, _) = outs
    return pos_sc, neg_sc, pos_amt


def amount_baselines(g: DailyGraph):
    """persistence + global-median RMSE/MAE in log10-USD on test positives."""
    log_usd = np.log10(g.usd_sum)
    med = float(np.median(log_usd[:g.train_end]))
    last_amt = {}
    for s, d, la in zip(g.src[:g.val_end], g.dst[:g.val_end], log_usd[:g.val_end]):
        last_amt[(int(s), int(d))] = la
    preds_p, preds_m, actual, seen_mask = [], [], [], []
    for j in range(g.val_end, len(g.src)):
        pair = (int(g.src[j]), int(g.dst[j]))
        seen_mask.append(pair in last_amt)
        preds_p.append(last_amt.get(pair, med))
        preds_m.append(med)
        actual.append(log_usd[j])
        last_amt[pair] = log_usd[j]  # persistence streams forward
    return (np.array(preds_p), np.array(preds_m), np.array(actual),
            np.array(seen_mask))


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b): return float(np.mean(np.abs(a - b)))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet",
                     default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap_.add_argument("--models", default="figures/tgat_models")
    ap_.add_argument("--out", default="figures/tgat_results.json")
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_.add_argument("--dim", type=int, default=100)
    ap_.add_argument("--k", type=int, default=20)
    args = ap_.parse_args()
    device = torch.device(args.device)

    g = load_daily_graph(args.parquet)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=args.k)
    _, before, _ = _pair_sets(g)
    test = slice(g.val_end, len(g.src))
    seen = np.array([(int(s), int(d)) in before
                     for s, d in zip(g.src[test], g.dst[test])])
    test_days = g.day[test]

    res = {"edgebank": {}, "tgat": {}, "amount": {},
           "split": {"train_end": g.train_end, "val_end": g.val_end,
                     "n_edges": len(g.src), "seen_frac": float(seen.mean())}}

    for strategy in ("random", "historical", "inductive"):
        res["edgebank"].setdefault("inf", {})[strategy] = {"auroc": [], "ap": []}
        res["edgebank"].setdefault("tw", {})[strategy] = {"auroc": [], "ap": []}
        res["tgat"][strategy] = {"auroc": [], "ap": [], "auroc_seen": [],
                                 "auroc_unseen": [], "per_day": []}
        for seed in args.seeds:
            negs = build_negative_pools(g, strategy, seed)
            y = np.r_[np.ones(negs.shape[0]), np.zeros(negs.shape[0])]
            for variant in ("inf", "tw"):
                ps, nsc = edgebank_scores(g, variant, negs)
                res["edgebank"][variant][strategy]["auroc"].append(
                    auroc(y, np.r_[ps, nsc]))
                res["edgebank"][variant][strategy]["ap"].append(
                    average_precision(y, np.r_[ps, nsc]))
            model = TGAT(edge_feat_dim=store.F, dim=args.dim).to(device)
            model.load_state_dict(torch.load(
                Path(args.models) / f"tgat_seed{seed}.pt", map_location=device))
            ps, nsc, pos_amt = tgat_scores(model, g, store, negs, device)
            sc = np.r_[ps, nsc]
            res["tgat"][strategy]["auroc"].append(auroc(y, sc))
            res["tgat"][strategy]["ap"].append(average_precision(y, sc))
            for name, m in (("auroc_seen", seen), ("auroc_unseen", ~seen)):
                mm = np.r_[m, m]
                res["tgat"][strategy][name].append(auroc(y[mm], sc[mm]))
            per_day = {int(dy): auroc(np.r_[np.ones((test_days == dy).sum()),
                                            np.zeros((test_days == dy).sum())],
                                      np.r_[ps[test_days == dy],
                                            nsc[test_days == dy]])
                       for dy in np.unique(test_days)}
            res["tgat"][strategy]["per_day"].append(per_day)
            if strategy == "random":
                actual_z = g.y_amt[test]
                pred_log = pos_amt * g.amt_std + g.amt_mean
                actual_log = actual_z * g.amt_std + g.amt_mean
                pp, pm, act, seen_amt = amount_baselines(g)
                a = res["amount"]
                a.setdefault("tgat_rmse", []).append(rmse(pred_log, actual_log))
                a.setdefault("tgat_mae", []).append(mae(pred_log, actual_log))
                a.setdefault("tgat_rmse_seen", []).append(
                    rmse(pred_log[seen_amt], actual_log[seen_amt]))
                a.setdefault("tgat_rmse_unseen", []).append(
                    rmse(pred_log[~seen_amt], actual_log[~seen_amt]))
                a["persistence_rmse"] = rmse(pp, act)
                a["persistence_rmse_seen"] = rmse(pp[seen_amt], act[seen_amt])
                a["median_rmse"] = rmse(pm, act)
                a["persistence_mae"] = mae(pp, act)
                a["median_mae"] = mae(pm, act)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "tgat"} |
                     {"tgat_auroc_means": {s: float(np.mean(v["auroc"]))
                                           for s, v in res["tgat"].items()}},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --group ml pytest tests/test_tgat_evaluate.py -x -q`
Expected: 5 passed.

- [ ] **Step 5: Run the full test suite**

Run: `uv run --group ml pytest tests/ -q`
Expected: all pass, including pre-existing sampler tests.

- [ ] **Step 6: Commit**

```bash
git add src/tgat/evaluate.py tests/test_tgat_evaluate.py
git commit -m "feat(tgat): NS pools, daily EdgeBank recompute, metrics, eval CLI"
```

---

### Task 6: Run the experiment and report

**Files:**
- Create: `figures/tgat_results.json` (generated), `figures/tgat_models/` (generated, gitignored)
- Modify: `.gitignore` (add `figures/tgat_models/`)

**Interfaces:**
- Consumes: the two CLIs from Tasks 4–5.

- [ ] **Step 1: Train 5 seeds on the GPU**

```bash
uv run --group ml python -m tgat.train --seeds 0 1 2 3 4 --out figures/tgat_models
```

Expected: 5 lines of `seed N: best val AP ...`, each run minutes-scale. Sanity: best val AP should be well above 0.5; if any run sits at ~0.5, stop and debug before evaluating.

- [ ] **Step 2: Evaluate**

```bash
uv run --group ml python -m tgat.evaluate --models figures/tgat_models --out figures/tgat_results.json
```

Expected: JSON printed with EdgeBank and TGAT AU-ROC means for all three strategies plus amount metrics.

- [ ] **Step 3: Check results against the spec's success criteria**

- TGAT AU-ROC (historical) vs recomputed EdgeBank_tw AU-ROC (historical).
- TGAT amount RMSE (seen) vs persistence RMSE (seen).
- Record both outcomes in the final report *whichever way they land*.

- [ ] **Step 4: Commit results**

```bash
echo "figures/tgat_models/" >> .gitignore
git add .gitignore figures/tgat_results.json figures/tgat_models/../train_log.json 2>/dev/null || true
git add figures/tgat_models/train_log.json || true
git commit -m "feat(tgat): experiment results, 5 seeds, all NS strategies"
```

(If `train_log.json` sits inside the gitignored dir, `git add -f figures/tgat_models/train_log.json`.)

---

## Self-Review Notes

- Spec coverage: data/filters (T1), splits (T1), neighbor rules (T2), model+loss (T3–T4), training config (T4), NS+EdgeBank recompute+stratification+amount baselines (T5), run+success criteria (T6). Per-day curves are captured in `per_day` (T5) for later plotting; figure regeneration is intentionally deferred to a follow-up once numbers exist.
- Types: `DailyGraph` field names used consistently in Tasks 2–5; `NeighborStore.F` includes the direction flag and is what `TGAT(edge_feat_dim=...)` receives.
- Known simplification: `train.py` samples validation negatives with the training sampler (random, same-day-collision-free) — standard practice; the three DGB strategies are evaluation-only.
