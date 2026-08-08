import numpy as np
from tgat.data import DailyGraph
from tgat.evaluate import auroc, average_precision, build_negative_pools, edgebank_scores
from tests.test_tgat_train import toy_graph

def rich_graph(E=3000, n_nodes=400, n_days=40, seed=1):
    """Like toy_graph but with ~600 distinct planted pairs so per-day
    historical pools exceed the day's positive count."""
    rng = np.random.default_rng(seed)
    day = np.sort(rng.integers(0, n_days, E)).astype(np.int64)
    src = (2 * rng.integers(0, n_nodes // 2, E)).astype(np.int64)
    # three deterministic partners per source instead of one
    dst = src + rng.integers(1, 4, E).astype(np.int64)
    usd = 10 ** rng.normal(4, 1, E)
    y = ((np.log10(usd) - 4) / 1).astype(np.float32)
    feat = np.column_stack([y, np.ones(E), np.ones(E), np.zeros(E),
                            np.zeros(E), np.zeros(E)]).astype(np.float32)
    return DailyGraph(src=src, dst=dst, day=day, edge_feat=feat, y_amt=y,
                      usd_sum=usd, n_nodes=n_nodes + 4, n_days=n_days,
                      train_end=int(E * .7), val_end=int(E * .85),
                      amt_mean=4.0, amt_std=1.0)

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
    g = rich_graph()
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
    g = rich_graph()
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

def test_average_precision_ties_give_block_average():
    # one pos, one neg, tied score: single block, precision 0.5, recall 1
    assert average_precision(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5

def test_average_precision_binary_scorer():
    # y=[1,1,0,0], s=[1,0,1,0]: block@1 -> P=.5,R=.5 ; block@0 -> P=.5,R=1
    assert average_precision(np.array([1, 1, 0, 0]), np.array([1.0, 0.0, 1.0, 0.0])) == 0.5
