import numpy as np
import torch
from tgat.data import DailyGraph
from tgat.train import _pos_pairs_by_day
from tgn.train import StreamingNegatives, softmax_ce_loss, train_one


def bursty_graph(n_sources=30, n_days=60, seed=0, p_stay=0.75, p_start=0.04):
    """Pairs transact in bursts: active pairs tend to stay active the next
    day. Positives are therefore predictable from recency — the signal a
    hard-negative / ranking loss should exploit."""
    rng = np.random.default_rng(seed)
    pairs = [(2 * i, 2 * i + 1) for i in range(n_sources)]
    src, dst, day = [], [], []
    active = np.zeros(len(pairs), bool)
    for t in range(n_days):
        p = np.where(active, p_stay, p_start)
        active = rng.random(len(pairs)) < p
        for j in np.nonzero(active)[0]:
            src.append(pairs[j][0]); dst.append(pairs[j][1]); day.append(t)
    E = len(src)
    src = np.array(src, np.int64); dst = np.array(dst, np.int64)
    day = np.array(day, np.int64)
    usd = 10 ** rng.normal(4, 1, E)
    y = ((np.log10(usd) - 4) / 1).astype(np.float32)
    feat = np.column_stack([y, np.ones(E), np.ones(E), np.zeros(E),
                            np.zeros(E), np.zeros(E)]).astype(np.float32)
    cut = np.searchsorted(day, [int(n_days * 0.7), int(n_days * 0.85)])
    return DailyGraph(src=src, dst=dst, day=day, edge_feat=feat, y_amt=y,
                      usd_sum=usd, n_nodes=2 * n_sources, n_days=n_days,
                      train_end=int(cut[0]), val_end=int(cut[1]),
                      amt_mean=4.0, amt_std=1.0)


def _observed_before(g, t):
    m = g.day < t
    return set(zip(g.src[m].tolist(), g.dst[m].tolist()))


def test_hard_negatives_are_strictly_past_pairs():
    g = bursty_graph()
    pos_by_day = _pos_pairs_by_day(g)
    rng = np.random.default_rng(0)
    # src_frac=0 so every draw uses the global historical-pair branch (bursty
    # sources have a single partner, which would force uniform fallback)
    sampler = StreamingNegatives(g.n_nodes, pos_by_day, rng, hard_frac=1.0,
                                 src_frac=0.0)
    t = int(g.day[g.train_end // 2])
    for T in range(t):
        m = g.day == T
        sampler.observe_day(g.src[m], g.dst[m])
    m = g.day == t
    s, d = g.src[m], g.dst[m]
    ns, nd = sampler.sample(s, d, t, n_neg=1)
    past = _observed_before(g, t)
    for a, b in zip(ns[:, 0], nd[:, 0]):
        pair = (int(a), int(b))
        assert pair in past                      # hard = previously seen
        assert pair not in pos_by_day[t]         # never a same-day positive


def test_source_partner_negatives_share_source():
    # source 0 saw partners 1, 2, 3 before day 5; day-5 positive is (0, 1)
    src = np.array([0, 0, 0], np.int64)
    dst = np.array([1, 2, 3], np.int64)
    pos_by_day = {0: {(0, 1)}, 1: {(0, 2)}, 2: {(0, 3)}, 5: {(0, 1)}}
    rng = np.random.default_rng(0)
    sampler = StreamingNegatives(10, pos_by_day, rng, hard_frac=1.0,
                                 src_frac=1.0)
    for T in range(3):
        sampler.observe_day(src[T:T + 1], dst[T:T + 1])
    ns, nd = sampler.sample(np.array([0], np.int64), np.array([1], np.int64),
                            5, n_neg=8)
    # all negatives must be (0, 2) or (0, 3): same source, past partner,
    # never today's positive (0, 1)
    assert (ns == 0).all()
    assert set(nd.ravel().tolist()) <= {2, 3}
    assert {2, 3} <= set(nd.ravel().tolist())


def test_random_negatives_collision_free_and_deterministic():
    g = bursty_graph()
    pos_by_day = _pos_pairs_by_day(g)
    t = int(g.day[100])
    m = g.day == t
    s, d = g.src[m], g.dst[m]
    outs = []
    for _ in range(2):
        rng = np.random.default_rng(7)
        sampler = StreamingNegatives(g.n_nodes, pos_by_day, rng, hard_frac=0.0)
        ns, nd = sampler.sample(s, d, t, n_neg=3)
        outs.append((ns.copy(), nd.copy()))
        assert ns.shape == (len(s), 3) and nd.shape == (len(s), 3)
        for i in range(len(s)):
            for k in range(3):
                assert (int(ns[i, k]), int(nd[i, k])) not in pos_by_day[t]
    np.testing.assert_array_equal(outs[0][0], outs[1][0])
    np.testing.assert_array_equal(outs[0][1], outs[1][1])


def test_softmax_ce_loss_hand_computed():
    pos = torch.tensor([2.0, 0.0])
    neg = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    loss = softmax_ce_loss(pos, neg)
    expect = -np.mean([
        2.0 - np.log(np.exp(2.0) + np.exp(1.0) + np.exp(0.0)),
        0.0 - np.log(3.0),
    ])
    assert abs(float(loss) - expect) < 1e-6


def test_two_term_loss_runs_and_is_deterministic():
    g = bursty_graph()
    outs = []
    for _ in range(2):
        _, info = train_one(g, seed=3, device=torch.device("cpu"),
                            epochs=2, batch=200, dim=16, k=3,
                            loss="ce", n_neg=2, n_neg_hard=2, beta_hard=0.5)
        outs.append(info)
    assert outs[0] == outs[1]
    assert len(outs[0]["history"]) == 2


def test_smoke_hard_ce_learns_recency():
    g = bursty_graph()
    # measured curve: hard-negative val AP 0.53 -> 0.69 over ~10 epochs on
    # the 247-event bursty stream (crosses 0.6 at epoch 6)
    model, info = train_one(g, seed=0, device=torch.device("cpu"),
                            epochs=10, batch=200, dim=32, k=5,
                            loss="ce", n_neg=3, hard_frac=0.5)
    assert info["history"][-1]["train_loss"] < info["history"][0]["train_loss"]
    # val AP is measured against hard-mixture negatives: beating 0.6 requires
    # actually ranking recently-active pairs above stale ones
    assert info["best_val_ap"] > 0.6
