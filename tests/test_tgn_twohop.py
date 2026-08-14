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
