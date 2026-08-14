import numpy as np
import torch
from tests.test_tgat_train import toy_graph
from tgn.train import train_one

def test_smoke_train_learns_planted_structure():
    g = toy_graph()
    # TGN needs a short memory warm-up: val AP crosses 0.6 at epoch ~3
    # (0.50 -> 0.59 -> 0.62 -> 0.65 measured), unlike TGAT's 2-epoch smoke
    model, info = train_one(g, seed=0, device=torch.device("cpu"),
                            epochs=4, batch=200, dim=32, k=5)
    assert info["epochs_run"] >= 1
    assert info["history"][-1]["train_loss"] < info["history"][0]["train_loss"]
    assert info["best_val_ap"] > 0.6
    h = info["history"][0]
    assert set(h) == {"epoch", "train_loss", "train_bce", "train_huber",
                      "train_ap", "train_auroc", "val_ap", "val_auroc"}

def test_train_determinism():
    g = toy_graph(E=400, n_nodes=40, n_days=20)
    _, a = train_one(g, seed=1, device=torch.device("cpu"), epochs=1,
                     batch=100, dim=16, k=3)
    _, b = train_one(g, seed=1, device=torch.device("cpu"), epochs=1,
                     batch=100, dim=16, k=3)
    assert a == b
