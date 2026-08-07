import numpy as np
import torch
from tgat.model import TimeEncoder, TGAT
from tgat.neighbors import NeighborStore

def rand_store(n_nodes=20, E=200, F=6, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_nodes, E).astype(np.int64)
    dst = rng.integers(0, n_nodes, E).astype(np.int64)
    day = np.sort(rng.integers(0, 50, E)).astype(np.int64)
    feat = rng.random((E, F), dtype=np.float32)
    return NeighborStore(src, dst, day, feat, n_nodes, k=5)

def test_time_encoder_shape_and_grad():
    te = TimeEncoder(16)
    dt = torch.rand(4, 5)
    out = te(dt)
    assert out.shape == (4, 5, 16)
    out.sum().backward()
    assert te.w.grad is not None

def test_embed_shape():
    st = rand_store()
    m = TGAT(edge_feat_dim=7, dim=32, n_layers=2, n_heads=2)
    z = m.embed(np.arange(6), np.full(6, 30), st, torch.device("cpu"))
    assert z.shape == (6, 32)

def test_forward_returns_logit_and_amount():
    st = rand_store()
    m = TGAT(edge_feat_dim=7, dim=32)
    logit, amt = m(np.arange(4), np.arange(4, 8), np.full(4, 30), st, torch.device("cpu"))
    assert logit.shape == (4,) and amt.shape == (4,)

def test_histories_differentiate_nodes():
    # zero node features: embeddings must still differ when histories differ
    st = rand_store()
    m = TGAT(edge_feat_dim=7, dim=32)
    m.eval()
    torch.manual_seed(0)
    z = m.embed(np.array([0, 1]), np.array([40, 40]), st, torch.device("cpu"))
    assert not torch.allclose(z[0], z[1])
