# TGN Daily Flow Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate a two-headed TGN (Rossi et al. 2020: learned per-node memory + temporal graph attention) on the same daily stablecoin stream as TGAT, testing whether a learned memory closes the gap to the recency heuristic, especially under historical negative sampling.

**Architecture:** A `src/tgn/` package mirroring `src/tgat/`. `model.py` holds the TGN module (GRU memory updated once per node per day from mean-aggregated messages; one temporal-attention layer over strictly-past neighbors with memory as node state; the same two heads). `streaming.py` provides the no-grad day-by-day replay used for all evaluation (memory advances with observed positives after each day is scored). `train.py` trains with a one-day-lag differentiable memory update (the TGN paper's one-batch lag, at day granularity). `evaluate.py` mirrors `tgat.evaluate`, importing its NS pools / EdgeBank / amount baselines unmodified. `ranking.py` computes deployment-style MRR/hits for TGN and the pair-recency heuristic.

**Tech Stack:** torch (existing `ml` group), numpy, reuse of `tgat.data`, `tgat.neighbors`, `tgat.model.{TimeEncoder,TemporalAttention}`, `tgat.train.{sample_training_negatives,average_precision,auroc}`, `tgat.evaluate.{build_negative_pools,edgebank_scores,amount_baselines,rmse,mae,_pair_sets}`, `tgat.plot_training`.

**Spec:** `TASK.md` (binding) + `docs/superpowers/specs/2026-08-07-tgat-daily-flow-design.md` (data semantics, protocol).

## Global Constraints

- Comparability contract (TASK.md §2): same `load_daily_graph` data (41,161 daily edges, day 0–238, day-snapped 70/15/15); same loss (link BCE with 1 same-day-collision-excluded random negative + 1.0·Huber(δ=1) on z-log10-USD, positives only); same eval (existing `build_negative_pools`, seeds 0–4, tie-aware AU-ROC/AP, seen/unseen strata, per-day curves, EdgeBank via existing code); same protocol (Adam lr 1e-4, batch 200, ≤50 epochs, patience 5 on val AP, 5 seeds).
- **No leakage:** predictions for day T use memory reflecting days < T only; same-day events never see each other (the day is the timestep); neighbors via existing strictly-past `NeighborStore`.
- **Memory streaming at eval:** memory advances with observed positives after each day is scored.
- Determinism: `torch.manual_seed(seed)` + one `np.random.default_rng(seed)` threaded through all sampling.
- Edge features: the same 6-dim vector; no node features (memory is the node state).
- Every real run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml ...` from repo root, `--parquet /data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet`. Tests: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/ -q`.
- Commits: conventional (`feat(tgn): ...`), each ending with the repo's `Co-Authored-By: Claude ...` trailer. Git is extremely slow on this NFS — run commits detached/背景 and never block on them.
- Artifact formats: TASK.md §5 exactly (train_log.json schema; results JSON with top-level `edgebank`, `tgn`, `amount`, `split`; checkpoints gitignored but `git add -f` the train_log.json; matplotlib palette `#2a78d6`/`#eb6834`/`#1baf7a`/`#898781`, no dual axes, legends always, 170 dpi).

## Design decisions (agent's judgment, documented per TASK.md §3)

1. **Day is the memory timestep.** Each node's memory updates at most once per day: all of a node's events on day T produce messages that are mean-aggregated, then one GRUCell step. (Paper's mean aggregator; avoids intra-day ordering, which is undefined in daily aggregation.)
2. **"Identity" message function** (paper ablation-supported): raw message = `[own_mem ‖ other_mem ‖ φ(Δt) ‖ edge_feat(6) ‖ dir_flag(1)]`, where `Δt = day − last_update[node]`, `φ` = the shared learnable `TimeEncoder`, dir_flag 1.0 = node was sender. The flag breaks sender/receiver symmetry at init when all memories are zero.
3. **One attention layer** (paper's TGN-attn default L=1 when memory is present), reusing `tgat.model.TemporalAttention` with node/neighbor inputs = current memory vectors, plus the same time encoding and `NeighborStore` edge features (F=7 incl. direction flag). `mem_dim = dim = 100`.
4. **One-day-lag differentiable memory (training):** persistent memory M is detached and reflects days ≤ T−2 when day T is being trained; the previous day's (T−1) messages are re-applied differentiably inside each batch's forward pass, so GRU/time-encoder parameters receive gradients through the current loss (this is the official TGN "store raw messages, apply lazily" scheme, at day granularity). After day T's batches, M advances permanently (no_grad) by applying day T−1, and day T becomes pending.
5. **Batches never cross day boundaries** (chunks of ≤200 within each day) — required so all of a day's predictions use the same pre-day memory. Mean events/day ≈ 172, so cost is close to TGAT's batching.
6. **Per-epoch metrics via a fresh no-grad streaming replay** (train-subset + val queries in one pass), rather than reusing in-batch training predictions — keeps the train_log semantics identical to TGAT (fixed subset, fixed negatives, eval mode).
7. **Ranking tie handling:** rank = 1 + #strictly-better + 0.5·#tied-others (tie-aware, consistent with the tie-aware AU-ROC convention). Recency heuristic: score(s,c,T) = −(T − last_day(s,c)) if pair seen before T else −∞; candidate set = all nodes; filtered = same-source same-day positives other than the target excluded.

## File Structure

- `src/tgn/__init__.py` — empty.
- `src/tgn/model.py` — `TGN` module: `init_memory`, `apply_messages`, `embed`, `forward`.
- `src/tgn/streaming.py` — `day_ranges`, `score_pairs_streaming`, `iter_day_embeddings`.
- `src/tgn/train.py` — `train_one` + CLI (same flags as `tgat.train`).
- `src/tgn/evaluate.py` — `tgn_scores` + CLI mirroring `tgat.evaluate` (imports its pools/EdgeBank/amount logic).
- `src/tgn/ranking.py` — filtered tie-aware ranking for TGN + recency heuristic + CLI.
- `src/tgn/plot_compare.py` — comparison bar figure from the two results JSONs.
- Tests: `tests/test_tgn_model.py`, `tests/test_tgn_streaming.py`, `tests/test_tgn_train.py`, `tests/test_tgn_ranking.py`.

---

### Task 1: TGN model — memory, messages, embedding, heads

**Files:**
- Create: `src/tgn/__init__.py` (empty), `src/tgn/model.py`
- Test: `tests/test_tgn_model.py`

**Interfaces:**
- Consumes: `tgat.model.TimeEncoder`, `tgat.model.TemporalAttention`, `tgat.neighbors.NeighborStore`.
- Produces: `tgn.model.TGN(edge_feat_dim: int, raw_feat_dim: int = 6, dim=100, n_heads=2, dropout=0.1)` with:
  - `init_memory(n_nodes, device) -> (mem FloatTensor [N,dim], last_update FloatTensor [N])` (zeros).
  - `apply_messages(mem, last_update, src, dst, day: int, edge_feat) -> (mem, last_update)` — one day's events (numpy arrays), one GRU step per active node on mean-aggregated messages; differentiable; returns new tensors (inputs unmodified).
  - `embed(nodes, days, mem, store, device) -> FloatTensor [B, dim]`.
  - `forward(src_nodes, dst_nodes, days, mem, store, device) -> (link_logit [B], amt_pred [B])`.
- `edge_feat_dim` is the store's `F` (7); `raw_feat_dim` is the data's edge-feature width (6).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tgn_model.py
import numpy as np
import torch
from tgat.neighbors import NeighborStore
from tgn.model import TGN

def make_model(dim=16):
    torch.manual_seed(0)
    return TGN(edge_feat_dim=7, raw_feat_dim=6, dim=dim)

def day_events(pairs, day):
    src = np.array([p[0] for p in pairs], dtype=np.int64)
    dst = np.array([p[1] for p in pairs], dtype=np.int64)
    feat = np.tile(np.arange(6, dtype=np.float32), (len(pairs), 1))
    return src, dst, day, feat

def test_apply_messages_updates_only_active_nodes():
    m = make_model()
    mem, last = m.init_memory(5, torch.device("cpu"))
    src, dst, day, feat = day_events([(0, 1)], 3)
    mem2, last2 = m.apply_messages(mem, last, src, dst, day, feat)
    assert not torch.allclose(mem2[0], mem[0])
    assert not torch.allclose(mem2[1], mem[1])
    for n in (2, 3, 4):
        assert torch.allclose(mem2[n], mem[n])
    assert last2[0] == 3 and last2[1] == 3 and last2[2] == 0

def test_apply_messages_one_gru_step_per_day():
    """A node with two same-day events gets ONE GRU step on the mean message."""
    m = make_model()
    mem, last = m.init_memory(4, torch.device("cpu"))
    mem = mem + torch.randn_like(mem) * 0.1
    src, dst, day, feat = day_events([(0, 1), (0, 2)], 5)
    mem2, _ = m.apply_messages(mem, last, src, dst, day, feat)
    # manual expectation for node 0: mean of its two sender messages
    dt = m.time_enc(torch.tensor([5.0 - 0.0]))[0]
    msg1 = torch.cat([mem[0], mem[1], dt, torch.from_numpy(feat[0]), torch.ones(1)])
    msg2 = torch.cat([mem[0], mem[2], dt, torch.from_numpy(feat[1]), torch.ones(1)])
    expect = m.gru(((msg1 + msg2) / 2).unsqueeze(0), mem[0].unsqueeze(0))[0]
    assert torch.allclose(mem2[0], expect, atol=1e-6)

def test_apply_messages_differentiable():
    m = make_model()
    mem, last = m.init_memory(3, torch.device("cpu"))
    src, dst, day, feat = day_events([(0, 1)], 2)
    mem2, _ = m.apply_messages(mem, last, src, dst, day, feat)
    mem2.sum().backward()
    assert m.gru.weight_ih.grad is not None
    assert m.time_enc.w.grad is not None

def test_embed_uses_memory():
    m = make_model()
    store = NeighborStore(np.array([], np.int64), np.array([], np.int64),
                          np.array([], np.int64), np.zeros((0, 6), np.float32),
                          n_nodes=3, k=5)
    mem, _ = m.init_memory(3, torch.device("cpu"))
    mem = mem.clone(); mem[1] += 1.0
    z = m.embed(np.array([0, 1]), np.array([4, 4]), mem, store, torch.device("cpu"))
    assert z.shape == (2, 16)
    assert not torch.allclose(z[0], z[1])

def test_forward_shapes():
    m = make_model()
    rng = np.random.default_rng(0)
    src = rng.integers(0, 10, 50).astype(np.int64)
    dst = rng.integers(0, 10, 50).astype(np.int64)
    day = np.sort(rng.integers(0, 20, 50)).astype(np.int64)
    feat = rng.random((50, 6), dtype=np.float32)
    store = NeighborStore(src, dst, day, feat, n_nodes=10, k=5)
    mem, _ = m.init_memory(10, torch.device("cpu"))
    logit, amt = m(np.arange(4), np.arange(4, 8), np.full(4, 15), mem, store,
                   torch.device("cpu"))
    assert logit.shape == (4,) and amt.shape == (4,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/test_tgn_model.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tgn'`.

- [ ] **Step 3: Implement `src/tgn/model.py`**

```python
"""Two-headed TGN (Rossi et al., 2020) over daily aggregated edges.

The day is the memory timestep: a node's memory updates at most once per day,
with one GRU step on the mean of that day's messages. Predictions for day T
must use memory reflecting days < T only — enforced by the training loop
(tgn.train) and the streaming scorer (tgn.streaming), not here.
"""
import numpy as np
import torch
import torch.nn as nn

from tgat.model import TemporalAttention, TimeEncoder


class TGN(nn.Module):
    def __init__(self, edge_feat_dim: int, raw_feat_dim: int = 6,
                 dim: int = 100, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.time_enc = TimeEncoder(dim)
        # raw message: [own_mem | other_mem | timeenc(dt) | edge_feat | dir_flag]
        self.gru = nn.GRUCell(2 * dim + dim + raw_feat_dim + 1, dim)
        self.attn = TemporalAttention(dim, edge_feat_dim, dim, n_heads, dropout)
        self.link_head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, 1))
        self.amt_head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, 1))

    def init_memory(self, n_nodes: int, device):
        return (torch.zeros(n_nodes, self.dim, device=device),
                torch.zeros(n_nodes, device=device))

    def apply_messages(self, mem, last_update, src, dst, day: int, edge_feat):
        """One memory step for the events of a single day (numpy inputs)."""
        device = mem.device
        E = len(src)
        node = torch.from_numpy(np.concatenate([src, dst])).to(device)
        other = torch.from_numpy(np.concatenate([dst, src])).to(device)
        feat = torch.from_numpy(np.concatenate([edge_feat, edge_feat])).to(device)
        flag = torch.cat([torch.ones(E, device=device),
                          torch.zeros(E, device=device)]).unsqueeze(1)
        dt = float(day) - last_update[node]
        msgs = torch.cat([mem[node], mem[other], self.time_enc(dt), feat, flag],
                         dim=1)
        active, inv = torch.unique(node, return_inverse=True)
        agg = torch.zeros(len(active), msgs.shape[1], device=device)
        agg.index_add_(0, inv, msgs)
        counts = torch.zeros(len(active), device=device)
        counts.index_add_(0, inv, torch.ones(len(node), device=device))
        agg = agg / counts.unsqueeze(1)
        mem = mem.clone()
        mem[active] = self.gru(agg, mem[active])
        last_update = last_update.clone()
        last_update[active] = float(day)
        return mem, last_update

    def embed(self, nodes, days, mem, store, device):
        nodes = np.asarray(nodes)
        days = np.asarray(days)
        nbr, dt, feat, mask = store.sample(nodes, days)
        B, k = nbr.shape
        node_h = mem[torch.from_numpy(nodes).to(device)]
        nbr_h = mem[torch.from_numpy(nbr.reshape(-1)).to(device)].reshape(
            B, k, self.dim)
        return self.attn(node_h, nbr_h,
                         torch.from_numpy(feat).to(device),
                         self.time_enc(torch.from_numpy(dt).to(device)),
                         torch.from_numpy(mask).to(device))

    def forward(self, src_nodes, dst_nodes, days, mem, store, device):
        zs = self.embed(src_nodes, days, mem, store, device)
        zd = self.embed(dst_nodes, days, mem, store, device)
        pair = torch.cat([zs, zd], dim=-1)
        return self.link_head(pair).squeeze(-1), self.amt_head(pair).squeeze(-1)
```

Note: `TimeEncoder.forward` broadcasts over any input shape (`dt.unsqueeze(-1) * w + b`), so 1-D `dt` gives `[E, dim]` — used directly in `apply_messages`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/test_tgn_model.py -x -q`
Expected: 5 passed. (Watch `test_apply_messages_one_gru_step_per_day`: if it fails on ordering, the concat order `[src-side, dst-side]` vs the manual message construction is the first suspect.)

- [ ] **Step 5: Commit (background — git is minutes-slow on this NFS)**

```bash
git add src/tgn tests/test_tgn_model.py
git commit -m "feat(tgn): memory module with daily GRU updates and attention embedding" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Streaming scorer (evaluation-side memory discipline)

**Files:**
- Create: `src/tgn/streaming.py`
- Test: `tests/test_tgn_streaming.py`

**Interfaces:**
- Consumes: `TGN` from Task 1, `DailyGraph`, `NeighborStore`.
- Produces:
  - `tgn.streaming.day_ranges(day: np.ndarray) -> np.ndarray [n_days+1]` — CSR-style offsets: events of day T are `slice(off[T], off[T+1])` (day array must be sorted; empty days allowed).
  - `tgn.streaming.score_pairs_streaming(model, g, store, device, s, d, t, batch=500) -> (logit np[Q], amt np[Q])` — replays the positive stream from day 0; each query is scored with memory reflecting days < its day; queries never touch memory; output aligned with input order; `t` need not be sorted.
  - `tgn.streaming.iter_day_embeddings(model, g, store, device, days: iterable[int], chunk=2048)` — generator yielding `(day, Z FloatTensor [n_nodes, dim])` for each requested day (ascending), memory streaming with positives in between.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tgn_streaming.py
import numpy as np
import torch
from tgat.data import DailyGraph
from tgat.neighbors import NeighborStore
from tgn.model import TGN
from tgn.streaming import day_ranges, score_pairs_streaming

def graph_from(src, dst, day, n_nodes=12, n_days=10):
    E = len(src)
    feat = np.tile(np.arange(6, dtype=np.float32), (E, 1))
    return DailyGraph(src=np.array(src, np.int64), dst=np.array(dst, np.int64),
                      day=np.array(day, np.int64), edge_feat=feat,
                      y_amt=np.zeros(E, np.float32), usd_sum=np.ones(E),
                      n_nodes=n_nodes, n_days=n_days, train_end=E, val_end=E,
                      amt_mean=0.0, amt_std=1.0)

def scores_for(g, s, d, t, seed=0):
    torch.manual_seed(seed)
    model = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=5)
    return score_pairs_streaming(model, g, store, torch.device("cpu"),
                                 np.array(s, np.int64), np.array(d, np.int64),
                                 np.array(t, np.int64))

def test_day_ranges_offsets():
    off = day_ranges(np.array([0, 0, 2, 2, 2, 5], np.int64), n_days=7)
    assert off.tolist() == [0, 2, 2, 5, 5, 5, 6, 6]

def test_same_day_events_do_not_influence_scores():
    base = graph_from([0, 1], [1, 2], [1, 3])
    plus = graph_from([0, 1, 4], [1, 2, 5], [1, 3, 3])  # extra day-3 event
    a = scores_for(base, [1], [2], [3])[0]
    b = scores_for(plus, [1], [2], [3])[0]
    np.testing.assert_array_equal(a, b)

def test_future_events_do_not_influence_scores():
    base = graph_from([0, 1], [1, 2], [1, 3])
    plus = graph_from([0, 1, 1], [1, 2, 3], [1, 3, 7])  # extra day-7 event
    a = scores_for(base, [1], [2], [3])[0]
    b = scores_for(plus, [1], [2], [3])[0]
    np.testing.assert_array_equal(a, b)

def test_past_events_do_influence_scores():
    base = graph_from([0, 1], [1, 2], [1, 3])
    plus = graph_from([0, 1, 1], [1, 2, 3], [1, 2, 3])  # extra day-2 event on node 1
    a = scores_for(base, [1], [2], [3])[0]
    b = scores_for(plus, [1], [2], [3])[0]
    assert not np.array_equal(a, b)

def test_output_aligned_with_unsorted_input():
    g = graph_from([0, 1, 2], [1, 2, 3], [1, 3, 5])
    s, d, t = [2, 0, 1], [3, 1, 2], [6, 2, 4]
    lo, ao = scores_for(g, s, d, t)
    for i in range(3):
        li, ai = scores_for(g, [s[i]], [d[i]], [t[i]])
        assert lo[i] == li[0] and ao[i] == ai[0]

def test_queries_do_not_advance_memory():
    g = graph_from([0, 1], [1, 2], [1, 3])
    alone = scores_for(g, [1], [2], [5])[0]
    crowd = scores_for(g, [3, 1, 4], [4, 2, 5], [2, 5, 5])[0]
    assert alone[0] == crowd[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/test_tgn_streaming.py -x -q`
Expected: FAIL with `ModuleNotFoundError` on `tgn.streaming`.

- [ ] **Step 3: Implement `src/tgn/streaming.py`**

```python
"""No-grad day-by-day replay for TGN evaluation.

Discipline (TASK.md §3): each day is scored with memory reflecting strictly
earlier days; memory then advances with that day's observed positives — the
same streaming as tgat.evaluate.edgebank_scores. Queries never enter memory.
"""
import numpy as np
import torch


def day_ranges(day: np.ndarray, n_days: int | None = None) -> np.ndarray:
    if n_days is None:
        n_days = int(day.max()) + 1 if len(day) else 0
    return np.searchsorted(day, np.arange(n_days + 1), side="left")


@torch.no_grad()
def score_pairs_streaming(model, g, store, device, s, d, t, batch=500):
    model.eval()
    mem, last = model.init_memory(g.n_nodes, device)
    off = day_ranges(g.day, g.n_days)
    logit = np.empty(len(s), np.float32)
    amt = np.empty(len(s), np.float32)
    order = np.argsort(t, kind="stable")
    qi, Q = 0, len(order)
    top_day = int(np.max(t)) + 1 if len(t) else 0
    for T in range(min(g.n_days, top_day)):
        qj = qi
        while qj < Q and t[order[qj]] == T:
            qj += 1
        for lo in range(qi, qj, batch):
            idx = order[lo:lo + batch]
            lg, am = model(s[idx], d[idx], t[idx], mem, store, device)
            logit[idx] = lg.cpu().numpy()
            amt[idx] = am.cpu().numpy()
        qi = qj
        elo, ehi = off[T], off[T + 1]
        if ehi > elo:
            mem, last = model.apply_messages(
                mem, last, g.src[elo:ehi], g.dst[elo:ehi], T,
                g.edge_feat[elo:ehi])
    return logit, amt


@torch.no_grad()
def iter_day_embeddings(model, g, store, device, days, chunk=2048):
    model.eval()
    mem, last = model.init_memory(g.n_nodes, device)
    off = day_ranges(g.day, g.n_days)
    want = sorted(int(x) for x in set(days))
    wi = 0
    for T in range(g.n_days):
        if wi >= len(want):
            return
        if T == want[wi]:
            zs = [model.embed(np.arange(lo, min(lo + chunk, g.n_nodes)),
                              np.full(min(chunk, g.n_nodes - lo), T),
                              mem, store, device)
                  for lo in range(0, g.n_nodes, chunk)]
            yield T, torch.cat(zs)
            wi += 1
        elo, ehi = off[T], off[T + 1]
        if ehi > elo:
            mem, last = model.apply_messages(
                mem, last, g.src[elo:ehi], g.dst[elo:ehi], T,
                g.edge_feat[elo:ehi])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/test_tgn_streaming.py -x -q`
Expected: 6 passed.

- [ ] **Step 5: Commit (background)**

```bash
git add src/tgn/streaming.py tests/test_tgn_streaming.py
git commit -m "feat(tgn): streaming day-replay scorer with positive-only memory advance" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Training loop and CLI

**Files:**
- Create: `src/tgn/train.py`
- Test: `tests/test_tgn_train.py`

**Interfaces:**
- Consumes: `TGN`, `score_pairs_streaming`, `day_ranges`, `tgat.train.{sample_training_negatives, average_precision, auroc, _pos_pairs_by_day}`, `tgat.data.load_daily_graph`, `NeighborStore`.
- Produces: `tgn.train.train_one(g, seed, device, epochs=50, patience=5, batch=200, lam=1.0, lr=1e-4, dim=100, k=20, train_metric_sample=3000) -> (TGN, dict)` with dict keys `best_val_ap, epochs_run, history` (history entries: `epoch, train_loss, train_bce, train_huber, train_ap, train_auroc, val_ap, val_auroc` — the TGAT schema, so `tgat.plot_training` works unmodified). CLI `python -m tgn.train` with the same flags as `tgat.train` (`--parquet --seeds --out --epochs --k --batch --dim --lr --lam --patience --device`), saving `tgn_seed{n}.pt` + `train_log.json` (flushed per seed) to `--out` (default `figures/tgn_models`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tgn_train.py
import numpy as np
import torch
from tests.test_tgat_train import toy_graph
from tgn.train import train_one

def test_smoke_train_learns_planted_structure():
    g = toy_graph()
    model, info = train_one(g, seed=0, device=torch.device("cpu"),
                            epochs=2, batch=200, dim=32, k=5)
    assert info["epochs_run"] >= 1
    assert info["history"][-1]["train_loss"] < info["history"][0]["train_loss"]
    assert info["best_val_ap"] > 0.6
    h = info["history"][0]
    assert set(h) == {"epoch", "train_loss", "train_bce", "train_huber",
                      "train_ap", "train_auroc", "val_ap", "val_auroc"}

def test_train_determinism():
    g = toy_graph(E=400, n_nodes=40, n_days=20)
    _, a = train_one(g, seed=1, device=torch.device("cpu"), epochs=1,
                     batch=100, dim=16, k=3)
    _, b = train_one(g, seed=1, device=torch.device("cpu"), epochs=1,
                     batch=100, dim=16, k=3)
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/test_tgn_train.py -x -q`
Expected: FAIL with `ModuleNotFoundError` on `tgn.train`.

- [ ] **Step 3: Implement `src/tgn/train.py`**

```python
"""Training loop + CLI for the two-headed TGN.

Memory during training uses the official TGN one-step-lag scheme at day
granularity: the persistent memory is detached and reflects days <= T-2 while
day T trains; day T-1's messages are re-applied differentiably inside each
batch's forward pass (so the GRU and time encoder get gradients), and the
persistent memory advances permanently (no_grad) once day T's batches finish.
Batches never cross day boundaries: every prediction for day T uses memory
reflecting days < T only, and same-day events never see each other.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tgat.data import DailyGraph, load_daily_graph
from tgat.neighbors import NeighborStore
from tgat.train import (_pos_pairs_by_day, auroc, average_precision,
                        sample_training_negatives)

from .model import TGN
from .streaming import day_ranges, score_pairs_streaming


def train_one(g: DailyGraph, seed: int, device, epochs=50, patience=5,
              batch=200, lam=1.0, lr=1e-4, dim=100, k=20,
              train_metric_sample=3000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=k)
    model = TGN(edge_feat_dim=store.F, raw_feat_dim=g.edge_feat.shape[1],
                dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_by_day = _pos_pairs_by_day(g)
    off = day_ranges(g.day, g.n_days)

    val_sl = slice(g.train_end, g.val_end)
    val_neg = sample_training_negatives(
        rng, g.src[val_sl], g.dst[val_sl], g.day[val_sl], g.n_nodes, pos_by_day)

    # fixed train-metric subset with fixed negatives (same convention as TGAT)
    rng_eval = np.random.default_rng(seed + 10_000)
    n_tm = min(train_metric_sample, g.train_end)
    tm_idx = np.sort(rng_eval.choice(g.train_end, size=n_tm, replace=False))
    tm_s, tm_d, tm_t = g.src[tm_idx], g.dst[tm_idx], g.day[tm_idx]
    tm_neg = sample_training_negatives(rng_eval, tm_s, tm_d, tm_t,
                                       g.n_nodes, pos_by_day)

    train_days = np.unique(g.day[:g.train_end])

    def epoch_metrics():
        """One streaming replay scoring the fixed train subset + validation."""
        qs = np.concatenate([tm_s, tm_s, g.src[val_sl], g.src[val_sl]])
        qd = np.concatenate([tm_d, tm_neg, g.dst[val_sl], val_neg])
        qt = np.concatenate([tm_t, tm_t, g.day[val_sl], g.day[val_sl]])
        lg, _ = score_pairs_streaming(model, g, store, device, qs, qd, qt)
        n, v = n_tm, g.val_end - g.train_end
        tr_y = np.r_[np.ones(n), np.zeros(n)]
        va_y = np.r_[np.ones(v), np.zeros(v)]
        tr_sc, va_sc = lg[:2 * n], lg[2 * n:]
        return (average_precision(tr_y, tr_sc), auroc(tr_y, tr_sc),
                average_precision(va_y, va_sc), auroc(va_y, va_sc))

    best_ap, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        mem, last = model.init_memory(g.n_nodes, device)
        pending = None  # (lo, hi, day) of the most recently completed day
        bces, hubers, totals = [], [], []
        for T in train_days:
            dlo, dhi = int(off[T]), int(off[T + 1])
            for blo in range(dlo, dhi, batch):
                bhi = min(blo + batch, dhi)
                s, d, t = g.src[blo:bhi], g.dst[blo:bhi], g.day[blo:bhi]
                nd = sample_training_negatives(rng, s, d, t, g.n_nodes,
                                               pos_by_day)
                if pending is not None:
                    plo, phi, pday = pending
                    mem_eff, _ = model.apply_messages(
                        mem, last, g.src[plo:phi], g.dst[plo:phi], pday,
                        g.edge_feat[plo:phi])
                else:
                    mem_eff = mem
                opt.zero_grad()
                logit_p, amt_p = model(s, d, t, mem_eff, store, device)
                logit_n, _ = model(s, nd, t, mem_eff, store, device)
                y_amt = torch.from_numpy(g.y_amt[blo:bhi]).to(device)
                bce = (F.binary_cross_entropy_with_logits(
                           logit_p, torch.ones_like(logit_p))
                       + F.binary_cross_entropy_with_logits(
                           logit_n, torch.zeros_like(logit_n))) / 2
                huber = lam * F.huber_loss(amt_p, y_amt, delta=1.0)
                (bce + huber).backward()
                opt.step()
                b, h = float(bce.detach()), float(huber.detach())
                bces.append(b); hubers.append(h); totals.append(b + h)
            with torch.no_grad():
                if pending is not None:
                    plo, phi, pday = pending
                    mem, last = model.apply_messages(
                        mem, last, g.src[plo:phi], g.dst[plo:phi], pday,
                        g.edge_feat[plo:phi])
            pending = (dlo, dhi, int(T))

        train_ap, train_auroc, val_ap, val_auroc = epoch_metrics()
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "train_bce": float(np.mean(bces)),
            "train_huber": float(np.mean(hubers)),
            "train_ap": train_ap, "train_auroc": train_auroc,
            "val_ap": val_ap, "val_auroc": val_auroc,
        })
        if val_ap > best_ap:
            best_ap, best_state, bad = val_ap, copy.deepcopy(model.state_dict()), 0
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
    ap_.add_argument("--out", default="figures/tgn_models")
    ap_.add_argument("--epochs", type=int, default=50)
    ap_.add_argument("--k", type=int, default=20)
    ap_.add_argument("--batch", type=int, default=200)
    ap_.add_argument("--dim", type=int, default=100)
    ap_.add_argument("--lr", type=float, default=1e-4)
    ap_.add_argument("--lam", type=float, default=1.0)
    ap_.add_argument("--patience", type=int, default=5)
    ap_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap_.parse_args()

    g = load_daily_graph(args.parquet)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = {}
    for seed in args.seeds:
        model, info = train_one(g, seed, torch.device(args.device),
                                epochs=args.epochs, k=args.k, batch=args.batch,
                                dim=args.dim, lr=args.lr, lam=args.lam,
                                patience=args.patience)
        torch.save(model.state_dict(), out / f"tgn_seed{seed}.pt")
        log[seed] = info
        print(f"seed {seed}: best val AP {info['best_val_ap']:.4f} "
              f"({info['epochs_run']} epochs)", flush=True)
        # flush after every seed so an interrupted run keeps finished histories
        (out / "train_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/test_tgn_train.py -x -q`
Expected: 2 passed (smoke ~1–3 min on CPU). If `best_val_ap` misses 0.6, debug learning (lr, memory flow) before weakening the assertion.

- [ ] **Step 5: Run the full suite**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/ -q`
Expected: all green (35 existing + 13 new).

- [ ] **Step 6: Commit (background)**

```bash
git add src/tgn/train.py tests/test_tgn_train.py
git commit -m "feat(tgn): day-granular training loop with one-day-lag differentiable memory" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Evaluation CLI (mirrors tgat.evaluate; shared pools/EdgeBank)

**Files:**
- Create: `src/tgn/evaluate.py`
- Test: extend `tests/test_tgn_streaming.py` with one integration test (below)

**Interfaces:**
- Consumes: `tgat.evaluate.{_pair_sets, amount_baselines, build_negative_pools, edgebank_scores, mae, rmse}` (imported, never reimplemented), `tgat.train.{auroc, average_precision}`, `score_pairs_streaming`.
- Produces: `tgn.evaluate.tgn_scores(model, g, store, negs, device, batch=500) -> (pos_sc, neg_sc, pos_amt)` and CLI `python -m tgn.evaluate --models figures/tgn_models --out figures/tgn_results.json` writing the exact `tgat_results.json` schema with `tgn` in place of `tgat` (`edgebank`, `tgn`, `amount` with `tgn_rmse/tgn_mae/tgn_rmse_seen/tgn_rmse_unseen` + scalar baselines, `split` incl. day ranges).

- [ ] **Step 1: Write the failing test** (append to `tests/test_tgn_streaming.py`)

```python
def test_tgn_scores_streams_and_aligns():
    from tgat.evaluate import build_negative_pools
    from tgn.evaluate import tgn_scores
    from tests.test_tgat_train import toy_graph
    g = toy_graph(E=600, n_nodes=60, n_days=20)
    torch.manual_seed(0)
    model = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=5)
    negs = build_negative_pools(g, "random", seed=0)
    pos_sc, neg_sc, pos_amt = tgn_scores(model, g, store, negs,
                                         torch.device("cpu"))
    n_test = len(g.src) - g.val_end
    assert pos_sc.shape == (n_test,) and neg_sc.shape == (n_test,)
    assert pos_amt.shape == (n_test,)
    assert np.isfinite(pos_sc).all() and np.isfinite(neg_sc).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/test_tgn_streaming.py::test_tgn_scores_streams_and_aligns -x -q`
Expected: FAIL with `ModuleNotFoundError` on `tgn.evaluate`.

- [ ] **Step 3: Implement `src/tgn/evaluate.py`**

```python
"""TGN evaluation CLI — identical protocol and JSON schema to tgat.evaluate.

NS pools, EdgeBank recompute, amount baselines and all metric conventions are
imported from tgat (never forked). Only the model scoring differs: TGN scores
stream day by day, memory advancing with observed positives (tgn.streaming).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tgat.data import load_daily_graph
from tgat.evaluate import (_pair_sets, amount_baselines, build_negative_pools,
                           edgebank_scores, mae, rmse)
from tgat.neighbors import NeighborStore
from tgat.train import auroc, average_precision

from .model import TGN
from .streaming import score_pairs_streaming


def tgn_scores(model, g, store, negs, device, batch=500):
    test = slice(g.val_end, len(g.src))
    s, d, t = g.src[test], g.dst[test], g.day[test]
    # one replay for positives and negatives: queries never advance memory,
    # so scoring them together is identical to scoring them separately
    qs = np.concatenate([s, negs[:, 0]])
    qd = np.concatenate([d, negs[:, 1]])
    qt = np.concatenate([t, t])
    lg, am = score_pairs_streaming(model, g, store, device, qs, qd, qt, batch)
    n = len(s)
    return lg[:n], lg[n:], am[:n]


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet",
                     default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap_.add_argument("--models", default="figures/tgn_models")
    ap_.add_argument("--out", default="figures/tgn_results.json")
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

    res = {"edgebank": {}, "tgn": {}, "amount": {},
           "split": {"train_end": g.train_end, "val_end": g.val_end,
                     "n_edges": len(g.src), "seen_frac": float(seen.mean()),
                     "train_days": [int(g.day[0]), int(g.day[g.train_end - 1])],
                     "val_days": [int(g.day[g.train_end]), int(g.day[g.val_end - 1])],
                     "test_days": [int(g.day[g.val_end]), int(g.day[-1])]}}

    for strategy in ("random", "historical", "inductive"):
        res["edgebank"].setdefault("inf", {})[strategy] = {"auroc": [], "ap": []}
        res["edgebank"].setdefault("tw", {})[strategy] = {"auroc": [], "ap": []}
        res["tgn"][strategy] = {"auroc": [], "ap": [], "auroc_seen": [],
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
            model = TGN(edge_feat_dim=store.F, raw_feat_dim=g.edge_feat.shape[1],
                        dim=args.dim).to(device)
            model.load_state_dict(torch.load(
                Path(args.models) / f"tgn_seed{seed}.pt", map_location=device))
            ps, nsc, pos_amt = tgn_scores(model, g, store, negs, device)
            sc = np.r_[ps, nsc]
            res["tgn"][strategy]["auroc"].append(auroc(y, sc))
            res["tgn"][strategy]["ap"].append(average_precision(y, sc))
            for name, m in (("auroc_seen", seen), ("auroc_unseen", ~seen)):
                mm = np.r_[m, m]
                res["tgn"][strategy][name].append(auroc(y[mm], sc[mm]))
            per_day = {int(dy): auroc(np.r_[np.ones((test_days == dy).sum()),
                                            np.zeros((test_days == dy).sum())],
                                      np.r_[ps[test_days == dy],
                                            nsc[test_days == dy]])
                       for dy in np.unique(test_days)}
            res["tgn"][strategy]["per_day"].append(per_day)
            if strategy == "random":
                pred_log = pos_amt * g.amt_std + g.amt_mean
                actual_log = g.y_amt[test] * g.amt_std + g.amt_mean
                pp, pm, act, seen_amt = amount_baselines(g)
                a = res["amount"]
                a.setdefault("tgn_rmse", []).append(rmse(pred_log, actual_log))
                a.setdefault("tgn_mae", []).append(mae(pred_log, actual_log))
                a.setdefault("tgn_rmse_seen", []).append(
                    rmse(pred_log[seen_amt], actual_log[seen_amt]))
                a.setdefault("tgn_rmse_unseen", []).append(
                    rmse(pred_log[~seen_amt], actual_log[~seen_amt]))
                a["persistence_rmse"] = rmse(pp, act)
                a["persistence_rmse_seen"] = rmse(pp[seen_amt], act[seen_amt])
                a["median_rmse"] = rmse(pm, act)
                a["persistence_mae"] = mae(pp, act)
                a["median_mae"] = mae(pm, act)
        print(f"{strategy}: tgn auroc mean "
              f"{np.mean(res['tgn'][strategy]['auroc']):.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "tgn"} |
                     {"tgn_auroc_means": {s: float(np.mean(v["auroc"]))
                                          for s, v in res["tgn"].items()}},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass, then full suite**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/test_tgn_streaming.py -x -q && UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/ -q`
Expected: all green.

- [ ] **Step 5: Commit (background)**

```bash
git add src/tgn/evaluate.py tests/test_tgn_streaming.py
git commit -m "feat(tgn): evaluation CLI sharing tgat NS pools and EdgeBank recompute" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Deployment ranking (TGN vs recency heuristic)

**Files:**
- Create: `src/tgn/ranking.py`
- Test: `tests/test_tgn_ranking.py`

**Interfaces:**
- Consumes: `iter_day_embeddings`, `TGN`, `DailyGraph`.
- Produces:
  - `tgn.ranking.filtered_rank(scores: np.ndarray [N], target: int, exclude: np.ndarray[int]) -> float` — tie-aware rank (1 + #strictly-better + 0.5·#tied-others) with excluded candidates removed.
  - `tgn.ranking.recency_ranking(g) -> dict` — streams `last_day[(s,c)]` over the full stream; for each test event, rank of true destination by score `-(T - last_day)` (unseen = −∞, tied); returns `{"mrr", "hits1", "hits10", "hits100"}` each with `"all"/"seen"/"unseen"` floats plus `n_test`, `n_candidates`.
  - `tgn.ranking.tgn_ranking(model, g, store, device) -> dict` — same shape, candidate embeddings once per test day via `iter_day_embeddings`, link-head scores chunked.
  - CLI `python -m tgn.ranking --models figures/tgn_models --out figures/tgn_ranking.json` → `{"tgn": {metric: {"all": [5 per-seed], "seen": [...], "unseen": [...]}}, "recency": {metric: {"all": float, ...}}, "n_test": int, "n_candidates": int}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tgn_ranking.py
import numpy as np
from tests.test_tgn_streaming import graph_from
from tgn.ranking import filtered_rank, recency_ranking

def test_filtered_rank_tie_aware():
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    # target idx 1: one better, two tied-others -> 1 + 1 + 1 = 3
    assert filtered_rank(scores, 1, np.array([], np.int64)) == 3.0
    # excluding the better candidate: 1 + 0 + 1 = 2
    assert filtered_rank(scores, 1, np.array([0], np.int64)) == 2.0

def test_recency_ranking_hand_example():
    # nodes 0..3; train: (0,1)@d0, (0,2)@d2 ; test day 4: (0,1)
    g = graph_from([0, 0, 0], [1, 2, 1], [0, 2, 4], n_nodes=4, n_days=5)
    g.train_end = 2
    g.val_end = 2
    r = recency_ranking(g)
    # candidates for s=0: (0,1) last d0 -> score -4 ; (0,2) last d2 -> -2 ;
    # (0,0),(0,3) unseen -> -inf. Rank of 1: worse than 2, ties none -> 2
    assert r["mrr"]["all"] == 0.5
    assert r["hits1"]["all"] == 0.0
    assert r["hits10"]["all"] == 1.0

def test_recency_ranking_filters_same_day_positives():
    # test day 4 has positives (0,1) and (0,2); (0,2) is more recent (d2 vs d0)
    # but must be excluded when ranking (0,1)'s destination
    g = graph_from([0, 0, 0, 0], [1, 2, 1, 2], [0, 2, 4, 4],
                   n_nodes=4, n_days=5)
    g.train_end = 2
    g.val_end = 2
    r = recency_ranking(g)
    # (0,1): candidates {0,1,3} after excluding 2 -> 1 is best seen -> rank 1
    # (0,2): candidates {0,2,3} after excluding 1 -> rank 1
    assert r["mrr"]["all"] == 1.0

def test_recency_unseen_target_gets_tied_rank():
    # test event (0,3) never seen; candidates (0,*): 1 seen better,
    # {0,3} tied at -inf (2 tied, one is target) -> rank 1+1+0.5 = 2.5
    g = graph_from([0, 0], [1, 3], [0, 4], n_nodes=4, n_days=5)
    g.train_end = 1
    g.val_end = 1
    r = recency_ranking(g)
    assert r["mrr"]["all"] == 1.0 / 2.5
    assert r["mrr"]["unseen"] == 1.0 / 2.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/test_tgn_ranking.py -x -q`
Expected: FAIL with `ModuleNotFoundError` on `tgn.ranking`.

- [ ] **Step 3: Implement `src/tgn/ranking.py`**

```python
"""Deployment-style ranking: each test event's true destination ranked against
all nodes, filtered (same-source same-day positives other than the target are
excluded). Tie-aware rank = 1 + #strictly-better + 0.5 * #tied-others.

Recency heuristic: score(s, c, T) = -(T - last_day(s, c)) if (s, c) appeared
before day T (anywhere in the stream, streaming forward), else -inf.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tgat.data import DailyGraph, load_daily_graph
from tgat.evaluate import _pair_sets
from tgat.neighbors import NeighborStore

from .model import TGN
from .streaming import iter_day_embeddings

HITS = (1, 10, 100)


def filtered_rank(scores, target, exclude):
    s = np.asarray(scores, dtype=np.float64)
    keep = np.ones(len(s), bool)
    keep[exclude] = False
    keep[target] = True
    st = s[target]
    better = int(((s > st) & keep).sum())
    tied = int(((s == st) & keep).sum()) - 1
    return 1.0 + better + 0.5 * tied


def _accumulate(ranks, seen_flags):
    ranks = np.asarray(ranks, dtype=np.float64)
    seen = np.asarray(seen_flags, dtype=bool)
    out = {}
    strata = {"all": np.ones(len(ranks), bool), "seen": seen, "unseen": ~seen}
    out["mrr"] = {k: float((1.0 / ranks[m]).mean()) for k, m in strata.items()}
    for h in HITS:
        out[f"hits{h}"] = {k: float((ranks[m] <= h).mean())
                           for k, m in strata.items()}
    return out


def _test_day_positives(g: DailyGraph):
    """day -> source -> list of destination node ids (test slice only)."""
    by_day = defaultdict(lambda: defaultdict(list))
    for j in range(g.val_end, len(g.src)):
        by_day[int(g.day[j])][int(g.src[j])].append(int(g.dst[j]))
    return by_day


def _seen_flags(g: DailyGraph):
    _, before, _ = _pair_sets(g)
    return [(int(s), int(d)) in before
            for s, d in zip(g.src[g.val_end:], g.dst[g.val_end:])]


def recency_ranking(g: DailyGraph) -> dict:
    last = {}
    ranks = []
    by_day = _test_day_positives(g)
    src_last = defaultdict(dict)  # s -> {c: last_day}
    for s, d, t in zip(g.src, g.dst, g.day):
        pass  # populated in the streaming loop below
    j = g.val_end
    # stream all days; record ranks on test days before updating with them
    ev_i = 0
    E = len(g.src)
    for T in range(g.n_days):
        if T in by_day:
            for s, dsts in by_day[T].items():
                cand = src_last[s]
                for d in dsts:
                    exclude = [c for c in dsts if c != d]
                    sd = cand.get(d)
                    n_excl_seen = sum(1 for c in exclude if c in cand)
                    n_excl = len(exclude)
                    if sd is None:
                        n_unseen = g.n_nodes - len(cand) - (n_excl - n_excl_seen)
                        better = len(cand) - n_excl_seen
                        ranks.append(1.0 + better + 0.5 * (n_unseen - 1))
                    else:
                        better = sum(1 for c, lt in cand.items()
                                     if lt > sd and c not in exclude and c != d)
                        tied = sum(1 for c, lt in cand.items()
                                   if lt == sd and c not in exclude and c != d)
                        ranks.append(1.0 + better + 0.5 * tied)
        while ev_i < E and g.day[ev_i] == T:
            src_last[int(g.src[ev_i])][int(g.dst[ev_i])] = T
            ev_i += 1
    out = _accumulate(ranks, _seen_flags(g))
    out["n_test"] = len(ranks)
    out["n_candidates"] = g.n_nodes
    return out


@torch.no_grad()
def tgn_ranking(model, g: DailyGraph, store, device, chunk=4096) -> dict:
    model.eval()
    by_day = _test_day_positives(g)
    ranks = []
    for T, Z in iter_day_embeddings(model, g, store, device, by_day.keys()):
        for s, dsts in by_day[T].items():
            zs = Z[s].unsqueeze(0)
            scores = []
            for lo in range(0, g.n_nodes, chunk):
                zc = Z[lo:lo + chunk]
                pair = torch.cat([zs.expand(len(zc), -1), zc], dim=1)
                scores.append(model.link_head(pair).squeeze(-1))
            scores = torch.cat(scores).cpu().numpy()
            for d in dsts:
                exclude = np.array([c for c in dsts if c != d], np.int64)
                ranks.append(filtered_rank(scores, d, exclude))
    out = _accumulate(ranks, _seen_flags(g))
    out["n_test"] = len(ranks)
    out["n_candidates"] = g.n_nodes
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet",
                     default="flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet")
    ap_.add_argument("--models", default="figures/tgn_models")
    ap_.add_argument("--out", default="figures/tgn_ranking.json")
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_.add_argument("--dim", type=int, default=100)
    ap_.add_argument("--k", type=int, default=20)
    args = ap_.parse_args()
    device = torch.device(args.device)

    g = load_daily_graph(args.parquet)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=args.k)

    rec = recency_ranking(g)
    res = {"recency": {k: rec[k] for k in rec if k.startswith(("mrr", "hits"))},
           "tgn": {}, "n_test": rec["n_test"], "n_candidates": rec["n_candidates"]}
    for seed in args.seeds:
        model = TGN(edge_feat_dim=store.F, raw_feat_dim=g.edge_feat.shape[1],
                    dim=args.dim).to(device)
        model.load_state_dict(torch.load(
            Path(args.models) / f"tgn_seed{seed}.pt", map_location=device))
        r = tgn_ranking(model, g, store, device)
        for metric, strata in r.items():
            if not metric.startswith(("mrr", "hits")):
                continue
            for stratum, v in strata.items():
                res["tgn"].setdefault(metric, {}).setdefault(stratum, []).append(v)
        print(f"seed {seed}: MRR {r['mrr']['all']:.4f} "
              f"hits@10 {r['hits10']['all']:.4f}", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

(Clean up the leftover `for ... pass` loop and unused vars `last`, `j` before committing — they are plan artifacts, drop them in the implementation.)

- [ ] **Step 4: Run tests to verify they pass, then full suite**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/test_tgn_ranking.py -x -q && UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/ -q`
Expected: all green (35 + 18).

- [ ] **Step 5: Commit (background)**

```bash
git add src/tgn/ranking.py tests/test_tgn_ranking.py
git commit -m "feat(tgn): deployment ranking with tie-aware filtered MRR and recency baseline" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Train 5 seeds, evaluate, rank, figures

**Files:**
- Create (generated): `figures/tgn_models/tgn_seed{0..4}.pt`, `figures/tgn_models/train_log.json`, `figures/tgn_results.json`, `figures/tgn_ranking.json`, `figures/tgn_training_curves.png`, `figures/tgn_vs_tgat_bar.png`
- Create: `src/tgn/plot_compare.py`
- Modify: `.gitignore` (add `figures/tgn_models/`)

- [ ] **Step 1: Pick a free GPU** (`nvidia-smi`; GPUs 2–4 were free at plan time) and launch training detached:

```bash
mkdir -p figures/tgn_models
CUDA_VISIBLE_DEVICES=2 UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv setsid nohup \
  uv run --group ml python -m tgn.train \
  --parquet /data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet \
  --seeds 0 1 2 3 4 --out figures/tgn_models \
  > figures/tgn_models/train.out 2>&1 &
```

Expect ~2 min silent startup (NFS import tax), then per-epoch progress. Sanity: seed 0's `best_val_ap` well above 0.5; if it hovers at 0.5, stop and debug. If early stopping never triggers for most seeds (50 epochs run out), note it and optionally launch one longer run (`--epochs 150 --out figures/tgn_models_long`) afterwards — headline stays 50-epoch.

- [ ] **Step 2: Training curves figure**

```bash
UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml python -m tgat.plot_training \
  figures/tgn_models/train_log.json figures/tgn_training_curves.png \
  "TGN daily flow — training (5 seeds)"
```

- [ ] **Step 3: Evaluate + ranking (GPU, detached, sequential)**

```bash
CUDA_VISIBLE_DEVICES=2 UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv setsid nohup bash -c '
  uv run --group ml python -m tgn.evaluate \
    --parquet /data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet \
    --models figures/tgn_models --out figures/tgn_results.json &&
  uv run --group ml python -m tgn.ranking \
    --parquet /data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet \
    --models figures/tgn_models --out figures/tgn_ranking.json
' > figures/tgn_models/eval.out 2>&1 &
```

Sanity: recomputed EdgeBank numbers must match `figures/tgat_results.json`'s `edgebank` block (same code, same seeds → identical values).

- [ ] **Step 4: Comparison figure — implement `src/tgn/plot_compare.py`**

Grouped bar chart, AU-ROC by NS strategy (random/historical/inductive) for EdgeBank_tw (gray `#898781`), TGAT (blue `#2a78d6`), TGN (orange `#eb6834`), mean over seeds with std error bars, legend, 170 dpi, no dual axes; reads both results JSONs.

```python
"""Bar comparison of TGN vs TGAT vs EdgeBank_tw AU-ROC by NS strategy.

Usage: python -m tgn.plot_compare figures/tgat_results.json \
    figures/tgn_results.json figures/tgn_vs_tgat_bar.png
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
STRATEGIES = ["random", "historical", "inductive"]


def main(tgat_path, tgn_path, out_path):
    tgat = json.load(open(tgat_path))
    tgn = json.load(open(tgn_path))
    series = [
        ("EdgeBank_tw", GRAY, [tgat["edgebank"]["tw"][s]["auroc"] for s in STRATEGIES]),
        ("TGAT", BLUE, [tgat["tgat"][s]["auroc"] for s in STRATEGIES]),
        ("TGN", ORANGE, [tgn["tgn"][s]["auroc"] for s in STRATEGIES]),
    ]
    x = np.arange(len(STRATEGIES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (label, color, vals) in enumerate(series):
        means = [np.mean(v) for v in vals]
        stds = [np.std(v) for v in vals]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3,
               color=color, label=label)
    ax.axhline(0.5, color="#e1e0d9", linewidth=1, zorder=0)
    ax.set_xticks(x, [s.capitalize() for s in STRATEGIES])
    ax.set_ylabel("AU-ROC (test, 1 neg/pos, 5 seeds)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(framealpha=0.9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
```

Run it, inspect the PNG.

- [ ] **Step 5: Commit artifacts (background)**

```bash
printf 'figures/tgn_models/\n' >> .gitignore
git add .gitignore figures/tgn_results.json figures/tgn_ranking.json \
  figures/tgn_training_curves.png figures/tgn_vs_tgat_bar.png src/tgn/plot_compare.py
git add -f figures/tgn_models/train_log.json
git commit -m "feat(tgn): 5-seed experiment results, ranking, and comparison figures" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Results write-up

**Files:**
- Create: `TASK_RESULTS.md`

- [ ] **Step 1: Write `TASK_RESULTS.md`** containing:
  - Comparison table: TGN / TGAT / EdgeBank_inf / EdgeBank_tw × random / historical / inductive, AU-ROC and AP (mean ± std over 5 seeds), from the two results JSONs.
  - Seen/unseen AU-ROC strata for TGN and TGAT.
  - Amount metrics: TGN vs TGAT vs persistence vs median (RMSE/MAE, seen strata).
  - Ranking table: TGN (mean ± std) vs recency heuristic — MRR, hits@1/10/100, seen/unseen.
  - Honest interpretation, explicitly answering §1's hypothesis: did the learned memory close the gap to the recency heuristic (especially under historical NS)? Negative results are reportable.
  - Training behavior notes (epochs run, early stopping, any longer run).
  - Out-of-scope suggestions only if the results urgently demand one (per TASK.md §7).

- [ ] **Step 2: Final full-suite verification**

Run: `UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv uv run --group ml pytest tests/ -q`
Expected: all green.

- [ ] **Step 3: Commit (background)**

```bash
git add TASK_RESULTS.md
git commit -m "docs(tgn): results write-up — TGN vs TGAT vs EdgeBank vs recency" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** TASK.md §3 architecture (T1), no-leakage semantics (T1 memory step, T2 streaming tests, T3 day-batching + one-day lag), memory streaming at eval (T2), determinism (T3 test), edge features/no node features (T1), layout+tests (T1–T5), §4.1 tested increments (per-task commits), §4.2 trained models + train_log schema (T3, T6), §4.3 results JSON schema (T4), §4.4 figures (T6), §4.5 ranking (T5–T6), §4.6 write-up (T7), §5 formats (T3/T4/T6), §6 environment (all run commands carry `UV_PROJECT_ENVIRONMENT` and node-local parquet; long runs detached).
- **Types:** `TGN(edge_feat_dim=store.F, raw_feat_dim=g.edge_feat.shape[1], dim)` used identically in T3/T4/T5; `score_pairs_streaming(model, g, store, device, s, d, t, batch)` consistent across T2–T4; `apply_messages(mem, last_update, src, dst, day, edge_feat)` consistent T1–T3; history dict keys match `tgat.plot_training` exactly.
- **Known simplifications (documented):** identity message function with direction flag; mean message aggregation; single attention layer; one-day-lag gradient path for memory (official-TGN scheme at day granularity); validation negatives from the training sampler (TGAT precedent).
- **Placeholder scan:** ranking.py plan code carries two leftover lines flagged for deletion at implementation time; all other steps are complete code or exact commands.
