import numpy as np
import pytest
from tgat.data import load_daily_graph

PARQUET = "flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet"

@pytest.fixture(scope="module")
def g():
    return load_daily_graph(PARQUET)

def test_counts_match_profiling(g):
    assert len(g.src) == 41161          # daily stablecoin edges
    assert g.n_days == 239
    assert g.n_nodes > 10000

def test_sorted_by_day(g):
    assert (np.diff(g.day) >= 0).all()

def test_split_snapped_to_day_boundaries(g):
    # the last train day must differ from the first val day, etc.
    assert g.day[g.train_end - 1] != g.day[g.train_end]
    assert g.day[g.val_end - 1] != g.day[g.val_end]
    # splits are roughly 70/15/15
    assert abs(g.train_end / len(g.src) - 0.70) < 0.03
    assert abs(g.val_end / len(g.src) - 0.85) < 0.03

def test_zscore_stats_from_train_only(g):
    logs = np.log10(g.usd_sum[: g.train_end])
    assert g.amt_mean == pytest.approx(logs.mean(), rel=1e-6)
    assert g.amt_std == pytest.approx(logs.std(), rel=1e-6)
    # y_amt of train slice is standard-normal-ish
    assert g.y_amt[: g.train_end].mean() == pytest.approx(0.0, abs=1e-5)

def test_asset_shares_sum_to_one(g):
    shares = g.edge_feat[:, 2:6].sum(axis=1)
    np.testing.assert_allclose(shares, 1.0, atol=1e-5)

def test_feature_dtypes(g):
    assert g.edge_feat.dtype == np.float32
    assert g.src.dtype == np.int64
