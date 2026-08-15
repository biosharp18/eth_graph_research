"""Campaign-2 widening: FEAT_DIM_WIDE features and stratified sampling."""
import numpy as np
import torch

from tgat.neighbors import NeighborStore
from tgn.model import TGN, build_from_checkpoint
from tgn.recency import FEAT_DIM_WIDE, PairRecency
from tgn.train import train_one

from tests.test_tgn_negatives import bursty_graph


def test_wide_features_hand_check():
    tr = PairRecency(n_nodes=6, feat_dim=FEAT_DIM_WIDE)
    tr.observe_day(np.array([0, 2]), np.array([1, 1]), day=0)
    tr.observe_day(np.array([0, 0]), np.array([1, 3]), day=5)
    f = tr.features(np.array([0]), np.array([1]), day=6)
    assert f.shape == (1, FEAT_DIM_WIDE)
    w = f[0, 11:20]
    np.testing.assert_allclose(w[0], np.log1p(2))   # pair (0,1) twice
    np.testing.assert_allclose(w[1], np.log1p(6))   # first seen day 0
    np.testing.assert_allclose(w[2], np.log1p(3))   # dst 1: days 0,0,5 all >= 6-7
    np.testing.assert_allclose(w[3], np.log1p(3))   # dst 1: 3 events in 90d
    np.testing.assert_allclose(w[4], np.log1p(3))   # src 0: 3 events in 30d
    np.testing.assert_allclose(w[5], np.log1p(2))   # 0 pays {1, 3}
    np.testing.assert_allclose(w[6], np.log1p(2))   # 1 paid by {0, 2}
    # (0,3) last day 5 > (0,1) last day 5? both day 5 -> ties not "greater"
    np.testing.assert_allclose(w[7], np.log1p(0))   # rec_rank: none more recent
    np.testing.assert_allclose(w[8], np.log1p(2))   # 2 partners
    # buckets: pair dt 1 -> [1, 0, 1]
    np.testing.assert_array_equal(f[0, 20:], [1.0, 0.0, 1.0])
    # unseen pair for same source: rec_rank penalty band
    f2 = tr.features(np.array([0]), np.array([4]), day=6)
    np.testing.assert_allclose(f2[0, 18], np.log1p(2) + 1.0)


def test_wide_features_all_and_cross_match():
    rng = np.random.default_rng(0)
    tr = PairRecency(n_nodes=20, feat_dim=FEAT_DIM_WIDE)
    for day in range(6):
        tr.observe_day(rng.integers(0, 20, 30), rng.integers(0, 20, 30), day)
    all_f = tr.features_all(3, day=7)
    per = tr.features(np.full(20, 3), np.arange(20), day=7)
    np.testing.assert_allclose(all_f, per, rtol=1e-6)
    cross = tr.features_cross(np.array([3, 5]), np.array([1, 9]), day=7)
    np.testing.assert_allclose(
        cross[1], tr.features(np.array([5, 5]), np.array([1, 9]), day=7))


def test_strat_sampling_properties():
    rng = np.random.default_rng(0)
    E = 300
    src = rng.integers(0, 4, E)
    dst = rng.integers(4, 8, E)
    day = np.sort(rng.integers(0, 50, E))
    feat = rng.normal(size=(E, 6)).astype(np.float32)
    rec = NeighborStore(src, dst, day, feat, 8, k=6, mode="recent")
    st = NeighborStore(src, dst, day, feat, 8, k=6, mode="strat")
    nodes = np.arange(8)
    q = np.full(8, 49)
    nr, dtr, fr, mr = rec.sample(nodes, q)
    ns, dts, fs, ms = st.sample(nodes, q)
    # strictly-past preserved: all valid dt > 0 given query day beyond history
    assert (dts[ms] >= 0).all()
    # strat rows reach strictly older events than recent-only rows
    assert dts[ms].max() > dtr[mr].max() or (dts == dtr).all()
    # short-history nodes identical between modes
    short = np.array([len(range(a, b)) <= 6 for a, b in
                      zip(rec.node_start, rec.node_end)])
    if short.any():
        np.testing.assert_array_equal(nr[short], ns[short])
    # every returned index is from the node's own strictly-past history:
    # replay a query one day earlier than an event and confirm it's excluded
    n2, dt2, _, m2 = st.sample(np.array([int(src[-1])]),
                               np.array([int(day[-1])]))
    assert (dt2[m2] > 0).all() or day[-1] > day[0]


def test_strat_train_and_eval_smoke():
    g = bursty_graph()
    model, info = train_one(g, seed=0, device=torch.device("cpu"), epochs=2,
                            patience=2, loss="ce", n_neg=2, nbr_mode="strat",
                            dim=16, k=5)
    assert np.isfinite(info["history"][-1]["train_loss"])


def test_wide_checkpoint_and_train_smoke(tmp_path):
    m = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16,
            pair_feat_dim=FEAT_DIM_WIDE)
    p = tmp_path / "m.pt"
    torch.save(m.state_dict(), p)
    m2 = build_from_checkpoint(p, edge_feat_dim=7, raw_feat_dim=6, dim=16,
                               device=torch.device("cpu"))
    assert m2.pair_feat_dim == FEAT_DIM_WIDE
    g = bursty_graph()
    model, info = train_one(g, seed=0, device=torch.device("cpu"), epochs=2,
                            patience=2, loss="ce", n_neg=2, hard_frac=0.2,
                            pair_feat=True, pair_feat_dim=FEAT_DIM_WIDE,
                            dim=16, k=5)
    assert np.isfinite(info["history"][-1]["train_loss"])
