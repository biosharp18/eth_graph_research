# 01 — Context: task, data, protocol, players

## Task

Daily-aggregated link prediction on stablecoin flows around an OFAC-sanctioned
seed address, following Poursafaei et al., *Towards Better Evaluation for
Dynamic Link Prediction* (NeurIPS 2022, "DGB"; PDF at repo root). Two-part
hurdle target per candidate `(source, destination, day)`:
1. link: does any flow occur (BCE / ranking loss on model logit);
2. amount: `zscore(log10(usd_sum))`, Huber δ=1, positives only, λ=1.

## Data (fixed by the comparability contract, `TASK.md` §2)

- `flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet` → `tgat.data.load_daily_graph`.
- 41,161 daily edges, 11,812 nodes, day index 0–238, stablecoins (USDT/USDC/
  BUSD/DAI) as USD 1:1.
- Day-snapped 70/15/15 chronological split: train_end=28,718, val_end=34,985.
- 6-dim edge features: z-log-USD, log1p(tx count), 4 asset shares. No node
  features (TGN's memory is the node state).
- ~40.7% of test events involve a pair seen before the test split ("seen"
  stratum); mean 172 events/day, max 302.

## Evaluation (frozen — never modified across all experiments)

- Three DGB negative-sampling (NS) strategies via `tgat.evaluate.build_negative_pools`:
  **random** (uniform never-seen pairs), **historical** (previously-seen pairs
  not positive that day), **inductive** (test-only pairs not positive that
  day). 1 negative/positive, seeds 0–4, tie-aware AU-ROC + AP, seen/unseen
  strata, per-day curves.
- EdgeBank (inf and time-window variants) recomputed by the same code on the
  same stream every run — recomputed values are bit-identical across runs,
  which doubles as a pipeline-integrity check.
- Amount: RMSE/MAE in log10-USD vs persistence and train-median baselines.
- **Deployment ranking** (secondary, harder): for each of 6,176 test events,
  rank the true destination against all 11,812 nodes, filtered (same-source
  same-day positives except the target excluded), candidate embeddings
  computed once per day, tie-aware rank = 1 + #better + 0.5·#tied. Metrics:
  MRR, hits@1/10/100, seen/unseen strata. Implemented in `src/tgn/ranking.py`.
- TGN evaluation streams memory day by day: score day T with memory
  reflecting days < T, then advance with day T's observed positives
  (`src/tgn/streaming.py`).

## The players and their numbers (test AU-ROC, 5 seeds where applicable)

| model | random | historical | inductive | deploy MRR |
|---|---|---|---|---|
| EdgeBank_inf | 0.7649 | 0.2649 | 0.5128 | — |
| EdgeBank_tw | 0.7221 | 0.6496 | 0.4700 | — |
| TGAT (2-layer, d=100) | 0.9559 ± 0.0032 | 0.7045 ± 0.0090 | 0.5513 ± 0.0065 | ~0.06 (earlier glimpse) |
| TGN baseline (BCE, 1 random neg) | 0.9640 ± 0.0018 | 0.6505 ± 0.0154 | 0.5534 ± 0.0079 | 0.0657 |
| **TGN hard-CE (winner)** | **0.9679 ± 0.0018** | **0.8990 ± 0.0027** | **0.5994 ± 0.0072** | 0.0456 |
| recency heuristic | — | — | — | **0.3540** |

Recency heuristic: score candidate c for source s at day T by
`-(T - last_day(s,c))`, unseen pairs tied at −∞. On seen pairs it gets
hits@1 = 0.50 and hits@100 = 0.99 — destination identity in this graph is
overwhelmingly "whoever this source paid most recently."

## TGN architecture (unchanged throughout the loss campaign)

`src/tgn/model.py`: per-node GRU memory with the **day as the timestep** (one
GRU step per node per day on the mean of that day's messages; message =
[own_mem ‖ other_mem ‖ time-enc(Δt) ‖ edge_feat ‖ direction flag]); one
temporal-attention layer over the 20 most recent strictly-past edges (reuses
`tgat.model.TemporalAttention` + `TimeEncoder`, memory as node state); the
same two MLP heads as TGAT on `[z_s ‖ z_d]`. dim=100. Training uses the
official-TGN one-step-lag trick at day granularity: persistent memory is
detached at ≤ T−2, day T−1's messages are re-applied differentiably inside
each batch's forward so GRU/time-encoder get gradients; batches never cross
day boundaries. Leakage property tests: `tests/test_tgn_streaming.py`.

## Why the loss campaign happened

The baseline TGN run (2026-08-13) rejected the original hypothesis ("a learned
memory closes the gap to the recency heuristic"): it beat TGAT only under
random NS and dropped to EdgeBank_tw level under historical NS. The write-up's
diagnosis — the 1-random-negative BCE training signal, not model capacity, was
the bottleneck — was then tested directly by redesigning the loss (user
directive 2026-08-13: "Scope out some different loss functions… show
verifiable improvement"). The diagnosis was right; see `02-loss-iterations.md`.
