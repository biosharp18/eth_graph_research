"""Full-softmax link loss: each positive competes against EVERY node as the
candidate destination (the ranking objective itself), instead of K sampled
negatives. Self-pairs and the source's other same-day destinations are
masked out, mirroring the sampler's rejection rule."""
import numpy as np
import torch
from tests.test_tgn_negatives import bursty_graph
from tgn.train import full_softmax_ce_loss, full_softmax_mask, train_one


def test_full_softmax_mask_excludes_self_and_same_day_positives():
    # day-T positives: (0, 1), (0, 2), (3, 1)
    pos = {(0, 1), (0, 2), (3, 1)}
    s = np.array([0, 0, 3]); d = np.array([1, 2, 1])
    m = full_softmax_mask(s, d, pos, n_nodes=5)
    assert m.shape == (3, 5) and m.dtype == bool
    # row 0: query (0 -> 1): mask self (0) and the other positive 2; keep 1
    assert m[0].tolist() == [True, False, True, False, False]
    # row 1: query (0 -> 2): mask self and 1; keep 2
    assert m[1].tolist() == [True, True, False, False, False]
    # row 2: query (3 -> 1): only self
    assert m[2].tolist() == [False, False, False, True, False]


def test_full_softmax_loss_hand_computed():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0],
                           [0.0, 0.0, 0.0, 0.0]])
    d = torch.tensor([0, 2])
    invalid = torch.tensor([[False, False, True, False],   # drop col 2 in row 0
                            [False, False, False, False]])
    loss = full_softmax_ce_loss(logits, d, invalid)
    expect = -np.mean([
        2.0 - np.log(np.exp(2.0) + np.exp(1.0) + np.exp(-1.0)),
        0.0 - np.log(4.0),
    ])
    assert abs(float(loss) - expect) < 1e-6


def test_smoke_train_full_softmax_learns_recency():
    g = bursty_graph()
    model, info = train_one(g, seed=0, device=torch.device("cpu"),
                            epochs=10, batch=200, dim=32, k=5,
                            loss="full", hard_frac=0.5)
    assert info["history"][-1]["train_loss"] < info["history"][0]["train_loss"]
    # val AP is measured against the hard-mixture negatives, like the CE test
    assert info["best_val_ap"] > 0.6
