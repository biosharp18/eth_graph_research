import numpy as np
import torch

from tgat.neighbors import NeighborStore
from tgn.model import TGN, build_from_checkpoint
from tgn.recency import FEAT_DIM, PairRecency
from tgn.streaming import score_pairs_streaming

from tests.test_tgn_streaming import graph_from


def scores_for(g, s, d, t, seed=0):
    torch.manual_seed(seed)
    model = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, pair_feat_dim=FEAT_DIM)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=5)
    return score_pairs_streaming(model, g, store, torch.device("cpu"),
                                 np.array(s, np.int64), np.array(d, np.int64),
                                 np.array(t, np.int64))


def test_tracker_features_hand_check():
    tr = PairRecency(n_nodes=6)
    tr.observe_day(np.array([0, 1]), np.array([2, 3]), day=4)
    f = tr.features(np.array([0, 0, 2]), np.array([2, 3, 3]), day=7)
    assert f.shape == (3, FEAT_DIM) and f.dtype == np.float32
    # (0,2) seen on day 4 -> pair dt 3; dst 2 last active day 4 -> dst dt 3
    np.testing.assert_allclose(f[0], [1.0, np.log1p(3), 1.0, np.log1p(3)])
    # (0,3) never seen as a pair, but dst 3 active on day 4
    np.testing.assert_allclose(f[1], [0.0, 0.0, 1.0, np.log1p(3)])
    # (2,3): same dst part, pair unseen
    np.testing.assert_allclose(f[2], f[1])


def test_tracker_uses_most_recent_day():
    tr = PairRecency(n_nodes=4)
    tr.observe_day(np.array([0]), np.array([1]), day=1)
    tr.observe_day(np.array([0]), np.array([1]), day=5)
    f = tr.features(np.array([0]), np.array([1]), day=6)
    np.testing.assert_allclose(f[0], [1.0, np.log1p(1), 1.0, np.log1p(1)])


def test_features_all_matches_features():
    rng = np.random.default_rng(0)
    tr = PairRecency(n_nodes=20)
    for day in range(5):
        tr.observe_day(rng.integers(0, 20, 30), rng.integers(0, 20, 30), day)
    all_f = tr.features_all(3, day=7)
    assert all_f.shape == (20, FEAT_DIM)
    per = tr.features(np.full(20, 3), np.arange(20), day=7)
    np.testing.assert_array_equal(all_f, per)


def test_pairfeat_same_day_events_do_not_influence_scores():
    base = graph_from([0, 1], [1, 2], [1, 3])
    plus = graph_from([0, 1, 1], [1, 2, 2], [1, 3, 3])  # same-day repeat
    a = scores_for(base, [1], [2], [3])[0]
    b = scores_for(plus, [1], [2], [3])[0]
    np.testing.assert_array_equal(a, b)


def test_pairfeat_future_events_do_not_influence_scores():
    base = graph_from([0, 1], [1, 2], [1, 3])
    plus = graph_from([0, 1, 1], [1, 2, 2], [1, 3, 7])
    a = scores_for(base, [1], [2], [3])[0]
    b = scores_for(plus, [1], [2], [3])[0]
    np.testing.assert_array_equal(a, b)


def test_pairfeat_past_pair_recency_influences_scores():
    # identical memory-relevant history, different pair recency for the query
    base = graph_from([0, 1], [1, 2], [1, 3])
    a = scores_for(base, [1], [2], [5])[0]
    b = scores_for(base, [1], [2], [9])[0]
    assert not np.array_equal(a, b)


def test_checkpoint_roundtrip_infers_pair_feat_dim(tmp_path):
    torch.manual_seed(0)
    model = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, pair_feat_dim=FEAT_DIM)
    p = tmp_path / "m.pt"
    torch.save(model.state_dict(), p)
    m2 = build_from_checkpoint(p, edge_feat_dim=7, raw_feat_dim=6, dim=16,
                               device=torch.device("cpu"))
    assert m2.pair_feat_dim == FEAT_DIM
    m0 = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16)
    p0 = tmp_path / "m0.pt"
    torch.save(m0.state_dict(), p0)
    m3 = build_from_checkpoint(p0, edge_feat_dim=7, raw_feat_dim=6, dim=16,
                               device=torch.device("cpu"))
    assert m3.pair_feat_dim == 0


def test_train_and_rank_with_pair_feat_smoke():
    from tests.test_tgat_train import toy_graph
    from tgn.train import train_one
    from tgn.ranking import tgn_ranking

    g = toy_graph(E=400, n_nodes=40, n_days=20)
    model, info = train_one(g, seed=0, device=torch.device("cpu"), epochs=2,
                            batch=100, dim=16, k=3, loss="ce", n_neg=2,
                            pair_feat=True)
    assert model.pair_feat_dim == FEAT_DIM
    assert np.isfinite(info["history"][-1]["val_ap"])
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=3)
    r = tgn_ranking(model, g, store, torch.device("cpu"))
    assert np.isfinite(r["mrr"]["all"]) and 1.0 >= r["mrr"]["all"] > 0.0


def test_hinge_and_mrr_selection_smoke():
    from tests.test_tgat_train import toy_graph
    from tgn.train import train_one

    g = toy_graph(E=400, n_nodes=40, n_days=20)
    model, info = train_one(g, seed=0, device=torch.device("cpu"), epochs=2,
                            batch=100, dim=16, k=3, loss="ce", n_neg=2,
                            n_neg_hard=2, beta_hard=0.5, hard_hinge=1.0,
                            pair_feat=True, select="mrr", val_mrr_events=50,
                            val_mrr_cands=20)
    h = info["history"][-1]
    assert "val_mrr" in h and 0.0 < h["val_mrr"] <= 1.0
    assert info["best_val_ap"] == max(e["val_mrr"] for e in info["history"])


def test_features_cross_matches_features():
    rng = np.random.default_rng(1)
    tr = PairRecency(n_nodes=15)
    for day in range(4):
        tr.observe_day(rng.integers(0, 15, 20), rng.integers(0, 15, 20), day)
    s = np.array([1, 3, 5])
    d = np.array([2, 4, 6])
    cross = tr.features_cross(s, d, day=6)
    for i in range(3):
        row = tr.features(np.full(3, s[i]), d, day=6)
        np.testing.assert_array_equal(cross[i], row)


def test_in_batch_negatives_smoke():
    from tests.test_tgat_train import toy_graph
    from tgn.train import train_one

    g = toy_graph(E=400, n_nodes=40, n_days=20)
    model, info = train_one(g, seed=0, device=torch.device("cpu"), epochs=2,
                            batch=100, dim=16, k=3, loss="ce", n_neg=2,
                            pair_feat=True, in_batch=True)
    assert np.isfinite(info["history"][-1]["train_loss"])
    assert np.isfinite(info["history"][-1]["val_ap"])
