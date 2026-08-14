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
        # 1-ulp CPU jitter across separate runs is fine; a misaligned query
        # or leaked memory shifts scores by orders of magnitude more
        np.testing.assert_allclose(lo[i], li[0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ao[i], ai[0], rtol=1e-5, atol=1e-6)

def test_queries_do_not_advance_memory():
    g = graph_from([0, 1], [1, 2], [1, 3])
    alone = scores_for(g, [1], [2], [5])[0]
    crowd = scores_for(g, [3, 1, 4], [4, 2, 5], [2, 5, 5])[0]
    assert alone[0] == crowd[1]

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
