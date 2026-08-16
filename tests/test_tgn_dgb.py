"""DGB paper-faithful evaluation protocol (tgn.evaluate_dgb)."""
import numpy as np

from tgat.data import DailyGraph
from tgn.evaluate_dgb import (batch_bounds, dgb_negatives, edgebank_scores_dgb,
                              per_batch_auroc)


def toy_graph():
    """12 train/val edges (days 0-3), 12 test edges (days 4-7), splits set
    so test starts at index 12. Repeating pairs so pools are non-trivial."""
    src = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                    0, 3, 1, 0, 3, 4, 0, 3, 4, 5, 3, 4])
    dst = np.array([6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8,
                    6, 9, 7, 9, 9, 8, 6, 9, 6, 7, 9, 8])
    day = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                    4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7])
    E = len(src)
    g = DailyGraph(src=src, dst=dst, day=day,
                   edge_feat=np.zeros((E, 6), np.float32),
                   y_amt=np.zeros(E, np.float32),
                   usd_sum=np.ones(E, np.float64), n_nodes=10,
                   n_days=8, train_end=9, val_end=12,
                   amt_mean=0.0, amt_std=1.0)
    return g


def test_batch_bounds():
    assert batch_bounds(5, 2) == [(0, 2), (2, 4), (4, 5)]


def test_random_negatives_keep_source_and_avoid_batch_positives():
    g = toy_graph()
    ns, nd, pad = dgb_negatives(g, "random", seed=0, batch_size=3)
    ts = g.src[g.val_end:]
    np.testing.assert_array_equal(ns, ts)      # source preserved
    assert not pad.any()
    td = g.dst[g.val_end:]
    for lo, hi in batch_bounds(len(ts), 3):
        batch_pairs = set(zip(ts[lo:hi].tolist(), td[lo:hi].tolist()))
        for i in range(lo, hi):
            assert (int(ns[i]), int(nd[i])) not in batch_pairs


def test_historical_negatives_accumulate_through_test():
    g = toy_graph()
    ns, nd, pad = dgb_negatives(g, "historical", seed=0, batch_size=3)
    ts, td = g.src[g.val_end:], g.dst[g.val_end:]
    seen = set(zip(g.src[:g.val_end].tolist(), g.dst[:g.val_end].tolist()))
    for lo, hi in batch_bounds(len(ts), 3):
        batch_pairs = set(zip(ts[lo:hi].tolist(), td[lo:hi].tolist()))
        for i in range(lo, hi):
            p = (int(ns[i]), int(nd[i]))
            assert p not in batch_pairs
            if not pad[i]:
                assert p in seen  # seen strictly before this batch
        seen |= batch_pairs
    # test-only pairs must be reachable as historical negatives in LATER
    # batches (accumulation through test) for some seed; with pool >= npos
    # from the start this just checks pads are rare here
    assert pad.mean() < 0.5


def test_inductive_pool_is_test_only_pairs_seen_so_far():
    g = toy_graph()
    ns, nd, pad = dgb_negatives(g, "inductive", seed=0, batch_size=3)
    ts, td = g.src[g.val_end:], g.dst[g.val_end:]
    train_val = set(zip(g.src[:g.val_end].tolist(),
                        g.dst[:g.val_end].tolist()))
    test_seen = set()
    for lo, hi in batch_bounds(len(ts), 3):
        batch_pairs = set(zip(ts[lo:hi].tolist(), td[lo:hi].tolist()))
        for i in range(lo, hi):
            p = (int(ns[i]), int(nd[i]))
            if not pad[i]:
                assert p in test_seen and p not in train_val
            assert p not in batch_pairs
        test_seen |= {p for p in batch_pairs if p not in train_val}
    # the first batch has an empty pool -> fully padded
    assert pad[:3].all()


def test_determinism_per_seed():
    g = toy_graph()
    a = dgb_negatives(g, "historical", seed=1, batch_size=3)
    b = dgb_negatives(g, "historical", seed=1, batch_size=3)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_per_batch_auroc_hand_check():
    # batch 1: perfect separation (auroc 1); batch 2: inverted (auroc 0)
    pos = np.array([2.0, 2.0, 0.0, 0.0])
    neg = np.array([1.0, 1.0, 1.0, 1.0])
    mean, vals = per_batch_auroc(pos, neg, batch_size=2)
    assert vals == [1.0, 0.0] and mean == 0.5


def test_edgebank_dgb_counts_earlier_test_batches():
    g = toy_graph()
    ts, td = g.src[g.val_end:], g.dst[g.val_end:]
    # negative = the pair from test batch 0 row 1 -> (3, 9), first seen in
    # test; as a negative for a batch AFTER it appears, EdgeBank must hit
    ns = np.full(len(ts), 3)
    nd = np.full(len(ts), 9)
    ps, nsc = edgebank_scores_dgb(g, "inf", ns, nd, batch_size=3)
    assert nsc[0] == 0.0        # batch 0: (3,9) unseen before test
    assert (nsc[3:] == 1.0).all()  # later batches: seen via batch 0
