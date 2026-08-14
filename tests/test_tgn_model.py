import numpy as np
import torch
from tgat.neighbors import NeighborStore
from tgn.model import TGN

def make_model(dim=16):
    torch.manual_seed(0)
    return TGN(edge_feat_dim=7, raw_feat_dim=6, dim=dim)

def day_events(pairs, day):
    src = np.array([p[0] for p in pairs], dtype=np.int64)
    dst = np.array([p[1] for p in pairs], dtype=np.int64)
    feat = np.tile(np.arange(6, dtype=np.float32), (len(pairs), 1))
    return src, dst, day, feat

def test_apply_messages_updates_only_active_nodes():
    m = make_model()
    mem, last = m.init_memory(5, torch.device("cpu"))
    src, dst, day, feat = day_events([(0, 1)], 3)
    mem2, last2 = m.apply_messages(mem, last, src, dst, day, feat)
    assert not torch.allclose(mem2[0], mem[0])
    assert not torch.allclose(mem2[1], mem[1])
    for n in (2, 3, 4):
        assert torch.allclose(mem2[n], mem[n])
    assert last2[0] == 3 and last2[1] == 3 and last2[2] == 0

def test_apply_messages_one_gru_step_per_day():
    """A node with two same-day events gets ONE GRU step on the mean message."""
    m = make_model()
    mem, last = m.init_memory(4, torch.device("cpu"))
    mem = mem + torch.randn_like(mem) * 0.1
    src, dst, day, feat = day_events([(0, 1), (0, 2)], 5)
    mem2, _ = m.apply_messages(mem, last, src, dst, day, feat)
    # manual expectation for node 0: mean of its two sender messages
    dt = m.time_enc(torch.tensor([5.0 - 0.0]))[0]
    msg1 = torch.cat([mem[0], mem[1], dt, torch.from_numpy(feat[0]), torch.ones(1)])
    msg2 = torch.cat([mem[0], mem[2], dt, torch.from_numpy(feat[1]), torch.ones(1)])
    expect = m.gru(((msg1 + msg2) / 2).unsqueeze(0), mem[0].unsqueeze(0))[0]
    assert torch.allclose(mem2[0], expect, atol=1e-6)

def test_apply_messages_differentiable():
    m = make_model()
    mem, last = m.init_memory(3, torch.device("cpu"))
    src, dst, day, feat = day_events([(0, 1)], 2)
    mem2, _ = m.apply_messages(mem, last, src, dst, day, feat)
    mem2.sum().backward()
    assert m.gru.weight_ih.grad is not None
    assert m.time_enc.w.grad is not None

def test_embed_uses_memory():
    m = make_model()
    store = NeighborStore(np.array([], np.int64), np.array([], np.int64),
                          np.array([], np.int64), np.zeros((0, 6), np.float32),
                          n_nodes=3, k=5)
    mem, _ = m.init_memory(3, torch.device("cpu"))
    mem = mem.clone(); mem[1] += 1.0
    z = m.embed(np.array([0, 1]), np.array([4, 4]), mem, store, torch.device("cpu"))
    assert z.shape == (2, 16)
    assert not torch.allclose(z[0], z[1])

def test_forward_shapes():
    m = make_model()
    rng = np.random.default_rng(0)
    src = rng.integers(0, 10, 50).astype(np.int64)
    dst = rng.integers(0, 10, 50).astype(np.int64)
    day = np.sort(rng.integers(0, 20, 50)).astype(np.int64)
    feat = rng.random((50, 6), dtype=np.float32)
    store = NeighborStore(src, dst, day, feat, n_nodes=10, k=5)
    mem, _ = m.init_memory(10, torch.device("cpu"))
    logit, amt = m(np.arange(4), np.arange(4, 8), np.full(4, 15), mem, store,
                   torch.device("cpu"))
    assert logit.shape == (4,) and amt.shape == (4,)
