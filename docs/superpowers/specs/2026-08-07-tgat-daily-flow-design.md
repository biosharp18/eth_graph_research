# TGAT for daily flow prediction on the OFAC k=4 value-flow graph

**Date:** 2026-08-07
**Status:** approved design, pre-implementation
**Context:** follows the TEA / EdgeBank exploratory analysis of `flow_graphs/OFAC_khop_value_graphs_k4/` (Poursafaei et al., NeurIPS 2022 framework). EdgeBank reference numbers: 0.656 AU-ROC historical NS, 0.477 inductive NS (day-averaged, event-level).

## Goal

Train a TGAT (Xu et al., ICLR 2020) temporal graph attention model on the k=4 OFAC
flow graph to predict **daily aggregated stablecoin flows**: for a directed pair
`(source, destination)` on day `t`, (1) does any flow occur, and (2) if so, how much
USD in total. Beat memorization (EdgeBank) on link prediction under the paper's
harder negative-sampling settings, and beat persistence baselines on amount.

## Data

- Source: `flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet` (49,921 rows).
- Filters: drop rows with null `block_timestamp` (5 contract_passthrough rows);
  keep only stablecoin transfers — asset ∈ {USDT, USDC, BUSD, DAI}, treated as
  USD 1:1. This retains 49,069 events (98.3% of events, ~$2.12B ≈ 99.8% of value).
  ETH/WETH events are dropped entirely (no price series available).
- Aggregation: group by `(source_eoa, target_eoa, UTC day)` → **41,161 daily edges**
  with `usd_sum`, `tx_count`, and per-asset USD breakdown.
- Node ids: contiguous integers over addresses appearing in the filtered stream.
- Time: day index 0–238 (window 2022-08-06 → 2023-04-01), used directly as the
  TGAT timestamp.
- Self-loops (s == d) remain in the history stream; negative samplers never emit them.

## Task and targets

Two-part (hurdle) formulation per candidate `(s, d, t)`:

1. **Link target** `y_link ∈ {0,1}` — 1 for observed daily edges, 0 for sampled
   negatives.
2. **Amount target** `y_amt = zscore(log10(usd_sum))` — defined only where
   `y_link = 1`. The z-score statistics (mean, std of log10 usd_sum) are computed
   on the **training split only** and frozen.

Rationale (from profiling): the outcome is zero-inflated (most pairs never
transact on a given day) with a 7-orders-of-magnitude positive tail; recurring
pairs have median CV 0.71 in daily sums, so magnitude is predictable from history.

## Split

Chronological 70/15/15 by daily-edge index, **snapped to day boundaries** so no
day straddles two splits (~28.8k / 6.2k / 6.2k daily edges). The snapped day
ranges are recorded in the results JSON for reproducibility.

## Model

TGAT with paper defaults, implemented lean in this repo:

- Learnable functional time encoding `cos(ωΔt + φ)`, dim 100.
- 2 attention layers, 2 heads, hidden/node dim 100, dropout 0.1.
- Temporal neighbor sampler: **20 most recent** strictly-past (`t' < t`) daily
  edges per node. Same-day edges are never visible to each other.
- Node features: zeros (no node attributes exist).
- Edge features per past daily edge (6 dims): z-scored `log10(usd_sum)`,
  `log1p(tx_count)`, 4-dim asset USD-share vector.
  The candidate edge being scored contributes **no** features (only `(s, d, t)`).
- Two heads on the concatenated `[z_s ‖ z_d]` embedding:
  - link head: MLP → 1 logit (BCE),
  - amount head: MLP → 1 value (Huber, δ=1, on `y_amt`), masked to positives.
- Total loss: `L = BCE + λ·Huber`, λ = 1.0 (configurable).

## Training

- One random-destination negative per positive (uniform over train nodes,
  excluding s==d and pairs with an observed daily edge on that same day —
  ever-seen pairs are otherwise fair training negatives).
- Batch 200 daily edges in chronological order, Adam lr 1e-4, max 50 epochs,
  early stopping on validation link-AP, patience 5.
- 5 seeds; report mean ± std. Device: CUDA if available, else CPU.

## Evaluation

- **Link**: AU-ROC and AP under the three DGB negative-sampling strategies
  (random / historical / inductive), day-based pools identical in construction
  to the EdgeBank harness, one negative per positive, 5 seeds.
  EdgeBank_inf and EdgeBank_tw are **recomputed on the same daily-edge stream**
  (same filters, same split) so the comparison is exact.
- **Amount**: RMSE and MAE in log10-USD on test positives, vs two baselines:
  persistence (pair's most recent observed daily sum; global median if pair
  unseen) and global train median.
- Both metric families stratified by whether the test pair was seen before the
  test split (~41% seen / 59% unseen).
- Per-test-day link AU-ROC curves saved for the daily-decay plot.

## Success criteria

- Link: beat recomputed EdgeBank_tw AU-ROC under historical NS.
- Amount: beat persistence RMSE on seen pairs (unseen pairs have no persistence
  prediction better than the median, so the interesting comparison is on seen).
- If TGAT fails either bar, that is a reportable finding, not a failed project.

## Code layout

```
src/tgat/
  __init__.py
  data.py        # parquet → daily-edge tensors, splits, feature/label builders
  neighbors.py   # temporal adjacency store + recent-k sampler
  model.py       # time encoding, attention layers, two-headed TGAT
  train.py       # training loop, early stopping, seeding, CLI entry
  evaluate.py    # NS pools, EdgeBank recompute, metrics, stratification, JSON out
tests/test_tgat.py
```

- `torch` added under a `[dependency-groups] ml` group in `pyproject.toml`
  (keeps the default BigQuery/duckdb env light).
- Results JSON → `figures/tgat_results.json`; figures regenerated from it.

## Tests

- Neighbor sampler returns only strictly-past edges (property test on random data).
- Day-boundary split leaves no day in two splits.
- Feature builder: z-score stats computed on train only; asset shares sum to 1.
- Shape checks for time encoding and attention layers.
- Smoke train: 2 epochs on a 2k-edge slice — loss decreases, val AUC > 0.5.

## Risks / notes

- 41.5% of test events involve a cold-start node (no history): expect the
  seen/unseen stratification to show most of any gain on the seen stratum.
- The graph is small for TGAT (41k daily edges); a modest gap over EdgeBank is a
  plausible and publishable-as-negative result.
- Shared A100 (~15 GB free at design time); model is tiny, contention is the
  only concern — CPU fallback works at this scale.
