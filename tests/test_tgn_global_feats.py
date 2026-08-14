"""Global-activity features (FEAT_DIM_GLOBAL) and novelty negatives."""
import numpy as np
import torch

from tgn.model import TGN, build_from_checkpoint
from tgn.recency import FEAT_DIM_GLOBAL, WINDOW, PairRecency
from tgn.train import StreamingNegatives, train_one

from tests.test_tgn_negatives import bursty_graph


def test_global_features_hand_check():
    tr = PairRecency(n_nodes=6, feat_dim=FEAT_DIM_GLOBAL)
    tr.observe_day(np.array([0, 2]), np.array([1, 1]), day=0)
    tr.observe_day(np.array([0]), np.array([1]), day=1)
    f = tr.features(np.array([0]), np.array([1]), day=2)
    assert f.shape == (1, FEAT_DIM_GLOBAL) and f.dtype == np.float32
    l1 = np.log1p(1)
    np.testing.assert_allclose(
        f[0],
        [1.0, l1,          # pair (0,1) last day 1
         1.0, l1,          # dst 1 last day 1
         1.0, l1,          # src 0 last day 1
         np.log1p(2),      # src 0 sent 2 events
         np.log1p(3),      # dst 1 received 3 events
         np.log1p(3),      # all 3 within the window
         np.log1p(1),      # deg(0) = {1}
         np.log1p(2)],     # deg(1) = {0, 2}
        rtol=1e-6)


def test_global_window_expiry():
    tr = PairRecency(n_nodes=4, feat_dim=FEAT_DIM_GLOBAL)
    tr.observe_day(np.array([0]), np.array([1]), day=0)
    # day-0 event still inside the window at query day WINDOW...
    f = tr.features(np.array([0]), np.array([1]), day=WINDOW)
    assert f[0, 8] == np.float32(np.log1p(1))
    # ...and expired one day later; totals unaffected
    f = tr.features(np.array([0]), np.array([1]), day=WINDOW + 1)
    assert f[0, 8] == 0.0
    assert f[0, 7] == np.float32(np.log1p(1))


def test_global_features_all_and_cross_match_features():
    rng = np.random.default_rng(0)
    tr = PairRecency(n_nodes=20, feat_dim=FEAT_DIM_GLOBAL)
    for day in range(5):
        tr.observe_day(rng.integers(0, 20, 30), rng.integers(0, 20, 30), day)
    all_f = tr.features_all(3, day=7)
    assert all_f.shape == (20, FEAT_DIM_GLOBAL)
    per = tr.features(np.full(20, 3), np.arange(20), day=7)
    np.testing.assert_array_equal(all_f, per)
    s_arr = np.array([3, 5, 7])
    d_arr = np.array([1, 3, 9])
    cross = tr.features_cross(s_arr, d_arr, day=7)
    for i, s in enumerate(s_arr):
        row = tr.features(np.full(3, s), d_arr, day=7)
        np.testing.assert_array_equal(cross[i], row)


def test_novelty_negatives_are_recent_first_seen_pairs():
    rng = np.random.default_rng(0)
    pos_by_day = {20: {(0, 1)}}
    sam = StreamingNegatives(50, pos_by_day, rng, nov_frac=1.0, nov_window=5)
    # old pairs first seen day 0, novel pairs first seen day 18
    sam.observe_day(np.array([2, 3, 4]), np.array([12, 13, 14]), day=0)
    sam.observe_day(np.array([5, 6, 0]), np.array([15, 16, 1]), day=18)
    first = dict(zip(sam.pair_list, sam.first_seen))
    ns, nd = sam.sample(np.zeros(40, np.int64), np.ones(40, np.int64), 20,
                        n_neg=2)
    for a, b in zip(ns.ravel().tolist(), nd.ravel().tolist()):
        assert first[(a, b)] >= 15          # within nov_window of day 20
        assert (a, b) != (0, 1)             # never today's positive
    # empty novel pool -> uniform fallback still works
    sam2 = StreamingNegatives(50, pos_by_day, rng, nov_frac=1.0, nov_window=5)
    sam2.observe_day(np.array([2]), np.array([12]), day=0)
    ns, nd = sam2.sample(np.zeros(5, np.int64), np.ones(5, np.int64), 20)
    assert ns.shape == (5, 1)


def test_checkpoint_roundtrip_global_dim(tmp_path):
    m = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16,
            pair_feat_dim=FEAT_DIM_GLOBAL)
    p = tmp_path / "m.pt"
    torch.save(m.state_dict(), p)
    m2 = build_from_checkpoint(p, edge_feat_dim=7, raw_feat_dim=6, dim=16,
                               device=torch.device("cpu"))
    assert m2.pair_feat_dim == FEAT_DIM_GLOBAL


def test_train_smoke_global_features_and_novelty():
    g = bursty_graph()
    model, info = train_one(g, seed=0, device=torch.device("cpu"), epochs=2,
                            patience=2, loss="ce", n_neg=2, hard_frac=0.2,
                            pop_frac=0.2, nov_frac=0.2, pair_feat=True,
                            pair_feat_dim=FEAT_DIM_GLOBAL, dim=16, k=5)
    assert np.isfinite(info["history"][-1]["train_loss"])
    assert model.pair_feat_dim == FEAT_DIM_GLOBAL
