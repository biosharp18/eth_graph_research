import numpy as np
from tgat.neighbors import NeighborStore

def tiny_store(k=3):
    # node 0 sends to 1 on days 1,2,5 ; node 2 sends to 0 on day 4
    src = np.array([0, 0, 2, 0], dtype=np.int64)
    dst = np.array([1, 1, 0, 1], dtype=np.int64)
    day = np.array([1, 2, 4, 5], dtype=np.int64)
    feat = np.arange(8, dtype=np.float32).reshape(4, 2)
    return NeighborStore(src, dst, day, feat, n_nodes=3, k=k)

def test_strictly_past_only():
    st = tiny_store()
    nbr, dt, feat, mask = st.sample(np.array([0]), np.array([4]))
    # day-4 edge (2->0) must NOT be visible at query day 4
    assert mask[0].sum() == 2            # only day-1 and day-2 edges
    assert (dt[0][mask[0]] >= 1).all()

def test_recency_takes_last_k():
    st = tiny_store(k=2)
    nbr, dt, feat, mask = st.sample(np.array([0]), np.array([6]))
    # node 0 has 4 past edges at day 6; k=2 keeps the two most recent (days 4,5)
    assert sorted(dt[0][mask[0]].tolist()) == [1.0, 2.0]

def test_direction_flag():
    st = tiny_store()
    nbr, dt, feat, mask = st.sample(np.array([0]), np.array([6]))
    flags = feat[0, mask[0], -1]
    # last 3 of 4 edges returned: days 2,4,5 with flags 1,0,1 (sent 2, received 1)
    assert flags.sum() == 2.0

def test_isolated_node_all_masked():
    st = tiny_store()
    nbr, dt, feat, mask = st.sample(np.array([2]), np.array([1]))
    assert not mask[0].any()
    assert (feat[0] == 0).all()

def test_property_never_future():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 30, 500).astype(np.int64)
    dst = rng.integers(0, 30, 500).astype(np.int64)
    day = np.sort(rng.integers(0, 60, 500)).astype(np.int64)
    feat = rng.random((500, 2), dtype=np.float32)
    st = NeighborStore(src, dst, day, feat, n_nodes=30, k=10)
    q_nodes = rng.integers(0, 30, 100).astype(np.int64)
    q_days = rng.integers(0, 60, 100).astype(np.int64)
    _, dt, _, mask = st.sample(q_nodes, q_days)
    assert (dt[mask] >= 1).all()

def test_vectorized_matches_reference_and_is_fast():
    # reference: brute-force strictly-past last-k per row
    rng = np.random.default_rng(7)
    E, N, K = 4000, 300, 20
    src = rng.integers(0, N, E).astype(np.int64)
    dst = rng.integers(0, N, E).astype(np.int64)
    day = np.sort(rng.integers(0, 200, E)).astype(np.int64)
    feat = rng.random((E, 3), dtype=np.float32)
    st = NeighborStore(src, dst, day, feat, N, k=K)
    q_nodes = rng.integers(0, N, 500).astype(np.int64)
    q_days = rng.integers(0, 200, 500).astype(np.int64)
    nbr, dt, f, mask = st.sample(q_nodes, q_days)
    # brute-force check on 50 random rows
    for b in rng.integers(0, 500, 50):
        n, qd = int(q_nodes[b]), int(q_days[b])
        # NOTE: a self-loop edge (src[i] == dst[i] == n) contributes TWO incidences
        # to the store (one source-role, one destination-role), same as the old
        # per-node-append implementation. A single list-comprehension branch would
        # under-count self-loops by one, so build the reference with an explicit
        # loop that appends both roles (source-role first, matching insertion order).
        events = []
        for i in range(E):
            if day[i] >= qd:
                continue
            if src[i] == n:
                events.append((int(day[i]), int(dst[i]), i, 1.0))
            if dst[i] == n:
                events.append((int(day[i]), int(src[i]), i, 0.0))
        events = events[-K:]
        assert mask[b].sum() == len(events)
        got = list(zip(nbr[b][mask[b]].tolist(), dt[b][mask[b]].tolist()))
        want = [(o, float(qd - d)) for d, o, _, _ in events]
        assert got == want
    # throughput: 500-row batch must sample in well under 50 ms
    import time
    t0 = time.time()
    for _ in range(20):
        st.sample(q_nodes, q_days)
    assert (time.time() - t0) / 20 < 0.05
