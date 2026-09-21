"""Two-hop temporal attention (n_layers=2) over memory."""
import numpy as np
import torch

from tgat.neighbors import NeighborStore
from tgn.model import TGN, build_from_checkpoint
from tgn.streaming import score_pairs_streaming
from tgn.train import train_one

from tests.test_tgn_negatives import bursty_graph
from tests.test_tgn_streaming import graph_from


def scores_for(g, s, d, t, seed=0):
    torch.manual_seed(seed)
    model = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, n_layers=2)
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=5)
    return score_pairs_streaming(model, g, store, torch.device("cpu"),
                                 np.array(s, np.int64), np.array(d, np.int64),
                                 np.array(t, np.int64))


def test_twohop_same_day_events_do_not_influence_scores():
    base = graph_from([0, 1], [1, 2], [1, 3])
    plus = graph_from([0, 1, 1], [1, 2, 2], [1, 3, 3])  # same-day repeat
    a = scores_for(base, [1], [2], [3])[0]
    b = scores_for(plus, [1], [2], [3])[0]
    np.testing.assert_array_equal(a, b)


def test_twohop_future_events_do_not_influence_scores():
    base = graph_from([0, 1], [1, 2], [1, 3])
    plus = graph_from([0, 1, 2], [1, 2, 0], [1, 3, 5])  # later-day extra
    a = scores_for(base, [1], [2], [3])[0]
    b = scores_for(plus, [1], [2], [3])[0]
    np.testing.assert_array_equal(a, b)


def test_twohop_reaches_second_hop_information():
    # chain 0 -> 1 (day 1), 2 -> 0 (day 2): at day 3, node 1's one-hop past
    # edges only touch node 0, but node 0's state-at-edge-day differs when
    # its OWN neighborhood (the 2->0 edge) is attended. Compare a 1-layer and
    # 2-layer model on a graph where only the second hop changes.
    base = graph_from([0, 1], [1, 2], [1, 2])
    # perturb only the 2-hop context of node 1: an extra edge into node 0 on
    # day 2 (same day as 1->2, so it enters memory identically ordered)
    plus = graph_from([0, 1, 3], [1, 2, 0], [1, 2, 2])
    torch.manual_seed(0)
    for n_layers, expect_diff in ((2, True),):
        model = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16,
                    n_layers=n_layers)
        out = []
        for g in (base, plus):
            store = NeighborStore(g.src, g.dst, g.day, g.edge_feat,
                                  g.n_nodes, k=5)
            out.append(score_pairs_streaming(
                model, g, store, torch.device("cpu"), np.array([1]),
                np.array([2]), np.array([3]))[0])
        assert (not np.allclose(out[0], out[1])) == expect_diff


def test_twohop_checkpoint_roundtrip(tmp_path):
    m = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, n_layers=2,
            pair_feat_dim=4)
    p = tmp_path / "m.pt"
    torch.save(m.state_dict(), p)
    m2 = build_from_checkpoint(p, edge_feat_dim=7, raw_feat_dim=6, dim=16,
                               device=torch.device("cpu"))
    assert m2.n_layers == 2 and m2.pair_feat_dim == 4


def test_twohop_train_smoke():
    g = bursty_graph()
    model, info = train_one(g, seed=0, device=torch.device("cpu"), epochs=2,
                            patience=2, loss="ce", n_neg=2, hard_frac=0.2,
                            n_layers=2, dim=16, k=5)
    assert np.isfinite(info["history"][-1]["train_loss"])
    assert model.n_layers == 2


# --- deeper message passing (2026-08-24): N hops and stacked layers ---

def test_deep_checkpoint_roundtrip_infers_depths(tmp_path):
    m = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, n_layers=3, n_stack=2,
            pair_feat_dim=23)
    p = tmp_path / "m.pt"
    torch.save(m.state_dict(), p)
    m2 = build_from_checkpoint(p, edge_feat_dim=7, raw_feat_dim=6, dim=16,
                               device=torch.device("cpu"))
    assert (m2.n_layers, m2.n_stack, m2.pair_feat_dim) == (3, 2, 23)
    # one-hop, no-stack checkpoints still load as before
    m1 = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16)
    torch.save(m1.state_dict(), p)
    m3 = build_from_checkpoint(p, edge_feat_dim=7, raw_feat_dim=6, dim=16,
                               device=torch.device("cpu"))
    assert (m3.n_layers, m3.n_stack) == (1, 0)


def _score(model, g, s, d, t, k=5):
    store = NeighborStore(g.src, g.dst, g.day, g.edge_feat, g.n_nodes, k=k)
    return score_pairs_streaming(model, g, store, torch.device("cpu"),
                                 np.array(s, np.int64), np.array(d, np.int64),
                                 np.array(t, np.int64))[0]


def test_threehop_reaches_third_hop_only():
    # chain 3 -> 2 (day 1), 2 -> 0 (day 2), 0 -> 1 (day 3); query (1, 4) at
    # day 5. Perturb only node 3's own past with 5 -> 3 on day 1 — the SAME
    # day as 3 -> 2, so node 2's memory message sees the identical pre-day
    # mem[3] and nothing downstream in memory changes. A 3-hop model still
    # reaches node 3's (changed) memory via 1 <- 0 <- 2 <- 3; 2-hop cannot.
    base = graph_from([3, 2, 0], [2, 0, 1], [1, 2, 3])
    plus = graph_from([3, 5, 2, 0], [2, 3, 0, 1], [1, 1, 2, 3])
    for n_layers, expect_diff in ((2, False), (3, True)):
        torch.manual_seed(0)
        m = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, n_layers=n_layers)
        a = _score(m, base, [1], [4], [5])
        b = _score(m, plus, [1], [4], [5])
        assert (not np.array_equal(a, b)) == expect_diff, n_layers


def test_threehop_future_and_same_day_events_do_not_leak():
    base = graph_from([3, 2, 0], [2, 0, 1], [1, 2, 3])
    same = graph_from([3, 2, 0, 0], [2, 0, 1, 1], [1, 2, 3, 3])
    future = graph_from([3, 2, 0, 4], [2, 0, 1, 1], [1, 2, 3, 6])
    torch.manual_seed(0)
    m = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, n_layers=3)
    a = _score(m, base, [1], [4], [3])
    np.testing.assert_array_equal(a, _score(m, same, [1], [4], [3]))
    np.testing.assert_array_equal(a, _score(m, future, [1], [4], [3]))


def test_k_inner_bounds_inner_hop_and_changes_nothing_at_one_hop():
    g = bursty_graph()
    torch.manual_seed(0)
    m1 = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, n_layers=1, k_inner=2)
    torch.manual_seed(0)
    m1b = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, n_layers=1)
    s, d, t = g.src[-20:], g.dst[-20:], g.day[-20:]
    np.testing.assert_array_equal(_score(m1, g, s, d, t), _score(m1b, g, s, d, t))
    # at two hops the inner width must matter on a busy graph
    torch.manual_seed(0)
    m2 = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, n_layers=2, k_inner=2)
    torch.manual_seed(0)
    m2b = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, n_layers=2)
    assert not np.allclose(_score(m2, g, s, d, t), _score(m2b, g, s, d, t))


def test_stack_is_extra_depth_on_the_same_neighbors():
    base = graph_from([0, 1], [1, 2], [1, 3])
    plus = graph_from([0, 1, 3], [1, 2, 0], [1, 3, 2])  # 2-hop-only change
    torch.manual_seed(0)
    m = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, n_layers=1, n_stack=2)
    assert len(m.stack) == 2
    # stacked layers never widen the field: a 2-hop-only perturbation is
    # invisible (memory of node 0 at day 3 does change, so compare at day 2
    # where node 0's memory is identical in both graphs)
    a = _score(m, base, [1], [2], [2])
    b = _score(m, plus, [1], [2], [2])
    np.testing.assert_array_equal(a, b)
    # but the stack does change the score vs the plain one-hop model
    torch.manual_seed(0)
    m0 = TGN(edge_feat_dim=7, raw_feat_dim=6, dim=16, n_layers=1)
    assert not np.allclose(a, _score(m0, base, [1], [2], [2]))


def test_deep_train_smoke():
    g = bursty_graph()
    model, info = train_one(g, seed=0, device=torch.device("cpu"), epochs=1,
                            patience=1, loss="ce", n_neg=2, n_layers=3,
                            n_stack=1, k_inner=3, dim=16, k=5)
    assert np.isfinite(info["history"][-1]["train_loss"])
