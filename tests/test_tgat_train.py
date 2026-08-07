import numpy as np
import torch
from tgat.data import DailyGraph
from tgat.train import sample_training_negatives, train_one

def toy_graph(E=2000, n_nodes=100, n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    day = np.sort(rng.integers(0, n_days, E)).astype(np.int64)
    # planted structure: even nodes send to node+1 repeatedly
    src = (2 * rng.integers(0, n_nodes // 2, E)).astype(np.int64)
    dst = src + 1
    usd = 10 ** rng.normal(4, 1, E)
    y = ((np.log10(usd) - 4) / 1).astype(np.float32)
    feat = np.column_stack([y, np.ones(E), np.ones(E), np.zeros(E),
                            np.zeros(E), np.zeros(E)]).astype(np.float32)
    return DailyGraph(src=src, dst=dst, day=day, edge_feat=feat, y_amt=y,
                      usd_sum=usd, n_nodes=n_nodes, n_days=n_days,
                      train_end=int(E * .7), val_end=int(E * .85),
                      amt_mean=4.0, amt_std=1.0)

def test_training_negatives_avoid_same_day_collisions():
    g = toy_graph()
    pos_by_day = {}
    for s, d, t in zip(g.src, g.dst, g.day):
        pos_by_day.setdefault(int(t), set()).add((int(s), int(d)))
    rng = np.random.default_rng(0)
    neg = sample_training_negatives(rng, g.src[:500], g.dst[:500], g.day[:500],
                                    g.n_nodes, pos_by_day)
    assert len(neg) == 500
    for s, nd, t in zip(g.src[:500], neg, g.day[:500]):
        assert nd != s
        assert (int(s), int(nd)) not in pos_by_day[int(t)]

def test_smoke_train_learns_planted_structure():
    g = toy_graph()
    model, info = train_one(g, seed=0, device=torch.device("cpu"),
                            epochs=2, batch=200, dim=32, k=5)
    assert info["epochs_run"] >= 1
    assert info["history"][-1]["train_loss"] < info["history"][0]["train_loss"]
    assert info["best_val_ap"] > 0.6   # planted even->odd structure is learnable
