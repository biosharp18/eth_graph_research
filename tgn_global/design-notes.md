# Design notes — W1/W2/G2 and everything that didn't make it (2026-08-16)

This is the design record for the receptive-field campaign's models: every
deliberate choice, the evidence behind it, and — with equal weight — the
paths that failed or were rejected before GPU time. Numbers cited are
5-seed means unless noted; protocols are LEGACY (frozen internal) or DGB
(paper-faithful, `results.md` §protocol correction).

## 1. The backbone (unchanged, inherited)

Two-headed TGN (Rossi et al. 2020) at day granularity: per-node GRU memory
(one step per node per day on the mean of that day's messages, direction
flag in the message), one temporal-attention hop over the 20 most recent
strictly-past edges (memory as node state), MLP link head on [z_s ‖ z_d ‖
pair features], separate MLP amount head (hurdle design: amount is
conditional-on-occurrence, trained on positives only). Bidirectional
propagation with direction encoded as a role feature; asymmetric decoder.
Nothing in this campaign changed the backbone — that was a finding, not a
constraint (see §4).

## 2. The wide feature set (FEAT_DIM_WIDE = 23) — W1's contribution

**Claim being acted on:** the network cannot COUNT. Attention over the 20
most recent edges is a truncated context window; any statistic of the
unbounded history (total frequency, degree, relationship age) is invisible
to it, and the GRU memory measurably fails to preserve it. Diagnosis: a
day-split MLP oracle over 12 cheap streaming statistics reached 0.688
inductive AU-ROC when every trained model sat at 0.55–0.62, and the
ablation put nearly all of it in global-in-TIME activity statistics.

Layout (all computed strictly from days < T; point-in-time correct in
train, eval, and ranking):

| cols | block | content | why it earned its place |
|---|---|---|---|
| 0–3 | pair recency (campaign-0) | pair seen, log1p(days since pair), dst seen, log1p(days since dst active) | closed the deployment-MRR gap in the pair-recency campaign |
| 4–10 | global activity (campaign-1) | src seen/recency, src/dst total freq, dst 30-day freq, undirected degrees | inductive oracle 0.478 → 0.633–0.676 |
| 11–19 | time-deep (campaign-2) | pair_freq, pair_age, dst_freq7, dst_freq90, src_freq30, out_deg_s, in_deg_c, **rec_rank**, n_partners_s | pair_freq (0.606) and pair_age (0.598) are the two strongest single inductive signals ever measured here; rec_rank targets hits@1 |
| 20–22 | sharp dt buckets | pair-dt == 1, == 2, ≤ 7 indicators | +MRR in the f14 probe; gives the head the recency heuristic's step function |

Notes on individual choices:
- **rec_rank** (candidate's recency rank among the source's partners) is
  the recency heuristic's own ordering statistic. Handing it to the head
  is what finally beat recency at hits@1 (0.2645 → 0.3045/0.3059 for
  W1/G2 vs recency 0.2777). Unseen pairs get a penalty band
  (log1p(n_partners)+1) so the feature is monotone across seen/unseen.
- Multi-scale windows (7/30/90) are rolling, day-monotonic deques — O(1)
  amortized, no bisects in the hot path.
- All pair-level features are DIRECTED (s→d); only deg_s/deg_c are
  undirected.

**Rejected features (documented so nobody re-adds them casually):**
- **Common-neighbors / Adamic-Adar / 2-hop path counts**: +0.011 oracle on
  top of activity features. Not worth the plumbing (sparse-matrix
  maintenance in the training loop); deliberately never implemented in the
  model. Structure is nearly silent in this graph's link formation.
- **Day-level graph context** (yesterday's edge count, activity EMA):
  actively harmful — the day-split oracle DROPS from 0.689 to 0.605 with
  them, because a nonlinear head memorizes day-regime signatures that
  don't transfer across time. Rejected at diagnosis, never trained.
- Windowed popularity, pop_window, larger K: prior-campaign nulls.

## 3. Novelty negatives (`--nov-frac`) — W2's contribution

**Mechanism:** the loss is sampled-softmax CE (InfoNCE-shaped); the score
it learns is calibrated against the training noise distribution q. No
component of the old q (uniform / popularity / seen-pair hard negatives)
resembled the inductive evaluation's negative population — recently
EMERGED pairs queried on dormant days. Features alone therefore did
nothing for inductive (W1: 0.5878 DGB) because the optimization never
asked that contrast.

**Implementation:** a fourth mixture component sampling negatives from
pairs whose FIRST appearance is within the last `nov_window`=30 days
(pair_list is first-seen-ordered, so the eligible set is a contiguous
tail; bisect + uniform draw + today-positive rejection). Streaming-safe
by the same discipline as everything else. Matched validation negatives
include the same component.

**The dial, measured (2-seed scoping, DGB-ind at 5 seeds where run):**
nov .1 → ind 0.6933 DGB / MRR 0.3525; nov .2 → 0.7298 / 0.3381;
nov .3 → (legacy 0.6184) with MRR 0.198 — saturating inductive, steep
ranking cost. W2 fixed nov=.2 (h .1 / pop .4 / nov .2 / uniform .3).
The wide features SHRANK the novelty-vs-MRR trade dramatically: at
narrow features nov .2 cost −0.12 MRR; at wide features −0.03.

**Costs, stated honestly:** random AU-ROC −0.04 vs W1 (pushing down
novel-active pairs inevitably pushes down novel positives — single-scalar
calibration geometry, same as tgn_improvement/07); unseen-stratum
deployment MRR falls (0.126 vs 0.158–0.165 for the non-novelty arms).

## 4. Two-hop attention (`--n-layers 2`) — G2's contribution, and a
   cautionary tale about nulls

Implementation: second TemporalAttention hop, TGAT recursion convention
(neighbor embedded at its edge's day → strictly-past discipline is
preserved by construction), (node, day) dedup at the hop boundary, raw
memory as leaf state. Old checkpoints load unchanged (attn2-key
inference).

**History of the result — this is the important part:**
- On narrow (4-dim) features: NULL. Every metric within 2-seed noise of
  the 1-hop reference (campaign 1). We reported it as a rejection of the
  "longer message passing" hypothesis.
- On wide (23-dim) features: consistent WIN on every ranking metric
  (MRR +0.003–0.007, h@1 +0.001–0.004, h@10 +0.006, h@100 +0.010 vs W1,
  tiny seed variance), no cost anywhere. G2 = W1 recipe + `--n-layers 2`
  is the deployment champion (MRR 0.3689, beats recency on all four).
- Interpretation: the second hop's marginal information (who my
  counterparties deal with) only binds once node-level statistics are
  saturated by features — before that, the optimizer spends the extra
  capacity re-deriving what the features now provide. **Method lesson: an
  architecture null is conditional on the feature set it was tested
  with.** We only caught this because the mandate forced a retest.

## 5. Attention-window widening — the measured dead end

Two direct attempts to widen what attention SEES, both null:
- `NeighborStore(mode="strat")`: half the k slots most-recent, half
  evenly spaced over the full history. MRR 0.311 (below the 0.331
  reference), inductive unchanged.
- `k=64` most-recent: MRR 0.320, inductive unchanged.
Conclusion: the missing long-horizon information is count-shaped, not
event-shaped. Attention with more events still cannot count them; the
features can. (Kept in the codebase — `--nbr-mode strat` works and is
eval-plumbed — for future data where individual old events might matter.)

## 6. Selection criteria and the epoch dial

- `--select combo` = sqrt(matched-negative val AP × sampled val MRR).
  Matched-negative validation is load-bearing: selecting on a
  distribution other than the training mixture silently reverts training
  (measured twice in prior campaigns).
- **Fixed-50-epoch study (2026-08-16): the epoch count is itself a
  calibration dial.** The combo criterion peaks at epochs 3–11 and
  decays; matched val-AP keeps rising to ~epoch 12 then plateaus. Trained
  to 50 epochs, W2's FINAL checkpoint is a new inductive record (DGB
  0.8240 ± .004, +0.09 over the early-stopped best) and near-hard-CE
  historical (0.854), at −0.08 MRR. Early checkpoint = ranking
  compromise; converged checkpoint = classification calibration. Both are
  kept (`w2_fixed50/`, `w2_fixed50_last/`; `--save-last` flag).

## 7. Evaluation gotchas this campaign created (read before reusing)

1. Models trained with `--nbr-mode strat` or non-default `--k` MUST be
   evaluated/ranked with the same flags — the neighbor store is rebuilt
   at eval and nothing in the checkpoint records it.
2. `train_log.json`, ranking/ensemble JSONs are flushed PER SEED: a
   reader that fires on file existence can catch a partial file. This
   produced two wrong readouts (ensemble at 3/5 seeds; G2 ranking at 3/5
   seeds, publicly corrected). Check seed count on read.
3. Legacy vs DGB AU-ROC are different protocols; use `tgn.evaluate_dgb`
   for any externally comparable claim. Deployment ranking is protocol-
   independent of that correction.
4. The two-term loss path (`--n-neg-hard`) was broken (hard_sampler never
   advanced) until 2026-08-14 — all pre-fix two-term/hinge verdicts are
   void, and the fixed path has NOT been re-run.

## 8. Model cards (what to use when)

| model | recipe | use it for | don't use it for |
|---|---|---|---|
| **G2** | h.1 p.5 f23 mrr-select, 2-hop | deployment ranking (MRR 0.3689, beats recency on all four) | inductive AU-ROC claims (0.596 DGB) |
| **W2 (early stop)** | h.1 p.4 nov.2 f23 combo | balanced single scalar (0.90/0.82/0.73 DGB) | max-anything |
| **W2 @50 (final epoch)** | same, no early stop | benchmark AU-ROC (0.894/0.854/0.824 DGB) | ranking (MRR 0.252) |
| W3 | h.1 p.45 nov.1 f23 combo | balanced with ranking parity (MRR 0.3525 ≈ recency) | — |
| W1 | h.1 p.5 f23 mrr-select, 1-hop | ablation reference for G2's 2-hop delta | — |
| hard-CE (legacy) | ce n5 h.5, no features | legacy-historical ceiling (0.899) | anything deployment-shaped (MRR 0.046) |

Open problems inherited by whoever picks this up: two-calibration heads
(one trunk, per-benchmark calibration — the only untried lever for
holding hist 0.88+ AND MRR 0.37 in one artifact, now testable since the
two-term bug fix); the unseen-inductive stratum; G3 (two-hop + novelty)
at 5 seeds if a single all-rounder is wanted.
