# UCLA A3ML Monthly Report — September 1 (draft)

## Summary of Progress Against Planned Objectives

Following last month's plan, we quantified the memorization baseline on our Ethereum data and built the evaluation protocol around it. We then trained and evaluated temporal graph models against this protocol. Our improved Temporal Graph Network (TGN) is the first model in our experiments to clearly beat both memorization baselines and off-the-shelf temporal architectures, reaching a balanced test AUROC of 0.83 (from a 0.5-chance baseline) through feature engineering and a novelty-aware negative sampling scheme. We also completed two studies that inform how to scale: simple forms of deeper message passing do not yet improve these metrics, and training beyond 50 epochs yields no further gains.

## Methods

We constructed a benchmark dataset from Google's BigQuery Ethereum archive: a 90-day window (November 2022 to April 2023) with 11,812 nodes and 41,161 day-aggregated transactions totalling $2.12B in volume. We prefer this over preprocessed Kaggle datasets such as Elliptic2 because it comes directly from the data source with interpretable features, so the same pipeline can construct training data at scale.

Following Poursafaei et al. (2022), we measured that roughly half of all transfers are repeats of previously seen pairs. A pure memorization baseline (EdgeBank) is therefore strong under standard evaluation, and standard random-negative AUROC mostly rewards memorization. We accordingly evaluate on two complementary tests: a standard temporal-split classification test, and a balanced test in which the model must distinguish active from dormant transaction pairs among pairs it has never observed as positives, where memorization provides no signal. We report AUROC, FDR and NPV on the classification tests, and Hits@k for ranking the true destination among all 11,812 candidates.

All models are trained autoregressively on day-discretized data with a combined objective: predict whether a (source, destination, day) transaction occurs, and if so, its amount. The temporal split holds out all activity after day 208 of 239.

## Results

EdgeBank variants score at or below chance on the balanced test (0.5) by construction, confirming that the balanced protocol removes the memorization shortcut. TGAT, which lacks persistent node state, scores below chance on both tests (0.46 standard, 0.40 balanced), indicating that per-node memory is essential at this data scale.

Our TGN with improvements is the clear winner. We've explored four such configurations, labelled W1, W2, W2.1, and W2.2.

W1. *Loss and engineered features.* We replaced the standard binary objective with a sampled-softmax cross-entropy over 5 negatives including hard negatives, and widened the hand-engineered feature set from 4 to 23 dimensions: node activity counts and recencies over multiple time windows, pair frequency and age, and degree statistics. Attention over a node's recent transactions cannot compute these counting statistics on its own. This produced strong ranking (Hits@1 of 0.305).

W2. *Novelty-aware negative sampling.* We draw 20% of training negatives from recently emerged transaction pairs, the training-time analogue of the balanced test. The intuition is that the sampling of training negatives should more match the test time distribution (classify true transactions from recent negatives in the test set). Balanced AUROC to 0.652, and AUROC on never-before-seen pairs from 0.59 to 0.73.

W2.1 *Together with W2, but we also train to convergence.* Training to a fixed 50 epochs raised the balanced AUROC to 0.78 and never-seen-pair AUROC to 0.82.

W2.2 *More novelty negatives.* Raising the novelty fraction from 20% to 30% gave the final model: balanced AUROC 0.83, never-seen-pair AUROC 0.86, Hits@1 of 0.195.

As well, we experimented with simple ways to widen the receptive field of our models, through deeper message passing (2-hop and 3-hop aggregation, and stacked attention layers). This produced no improvement on any metric. This is not to say that expanding the receptive field is useless, rather we are investigating more fine grained ways to expand it: propagating memory updates to a node's recent counterparties on each transaction, so that multi-hop information accumulates in the memories over time while queries remain one-hop, and hierarchical message passing, where nodes exchange messages with a community-level memory so that information travels across the graph without deep hop-by-hop aggregation.

## Planned Activities

Our program targets before scaling up are AUROC > 0.95, Hits@10 > 0.875, and NPV > 0.875. The next steps toward them are: (1) sweep the novelty-sampling fraction and window, (2) a two-head training objective that calibrates one output for detecting new suspicious relationships and one for ranking expected counterparties, to lift both metric families at once; and (3) the receptive-field architectures above, message propagation to counterparties and hierarchical community-level memory; and (4) begin scaling the training corpus using the BigQuery pipeline, since the ablations indicate additional data is the nearest lever.
