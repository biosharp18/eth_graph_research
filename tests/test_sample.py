import json

import pandas as pd
import pytest

from eth_graph_research.sample import (
    Sample,
    connect_edges_from_dataframe,
    pick_seeds,
    save_sample,
    snowball_sample,
)


def make_edges(pairs):
    """Build a synthetic edge DataFrame from (from, to) tuples."""
    return pd.DataFrame(
        {
            "from_address": [p[0] for p in pairs],
            "to_address": [p[1] for p in pairs],
            "block_timestamp": pd.to_datetime(
                [f"2026-05-{(i % 28) + 1:02d}" for i in range(len(pairs))], utc=True
            ),
            "transaction_hash": [f"0xhash{i}" for i in range(len(pairs))],
            "value_eth": [1.0 + i for i in range(len(pairs))],
        }
    )


def test_missing_seed_raises():
    con = connect_edges_from_dataframe(make_edges([("a", "b")]))
    with pytest.raises(ValueError, match="no edges"):
        snowball_sample(con, "zzz", max_nodes=10, per_node_cap=5, rng_seed=0)


def test_per_node_cap_limits_expansion():
    pairs = [("hub", f"leaf{i}") for i in range(100)]
    con = connect_edges_from_dataframe(make_edges(pairs))
    with pytest.warns(UserWarning, match="exhausted"):
        s = snowball_sample(con, "hub", max_nodes=1000, per_node_cap=5, rng_seed=0)
    # hub itself plus at most 5 sampled counterparties
    assert len(s.nodes) <= 6


def test_max_nodes_never_overshot():
    pairs = [("seed", f"n{i}") for i in range(50)]
    con = connect_edges_from_dataframe(make_edges(pairs))
    s = snowball_sample(con, "seed", max_nodes=10, per_node_cap=50, rng_seed=0)
    assert len(s.nodes) == 10


def test_induced_edges_include_untraversed():
    # Triangle: BFS from s traverses s-a and s-b; induced output must also
    # contain the a-b edge.
    pairs = [("s", "a"), ("s", "b"), ("a", "b")]
    con = connect_edges_from_dataframe(make_edges(pairs))
    s = snowball_sample(con, "s", max_nodes=3, per_node_cap=10, rng_seed=0)
    assert sorted(s.nodes) == ["a", "b", "s"]
    edge_pairs = set(zip(s.edges["from_address"], s.edges["to_address"]))
    assert ("a", "b") in edge_pairs
    assert len(s.edges) == 3


def test_deterministic_given_rng_seed():
    # Complete graph on 10 nodes: plenty of arbitrary choices to pin down.
    pairs = [(f"x{i}", f"x{j}") for i in range(10) for j in range(i + 1, 10)]
    con = connect_edges_from_dataframe(make_edges(pairs))
    s1 = snowball_sample(con, "x0", max_nodes=5, per_node_cap=3, rng_seed=42)
    s2 = snowball_sample(con, "x0", max_nodes=5, per_node_cap=3, rng_seed=42)
    assert s1.nodes == s2.nodes
    pd.testing.assert_frame_equal(s1.edges, s2.edges)


def test_frontier_exhaustion_returns_component_with_warning():
    pairs = [("a", "b"), ("b", "c")]  # disconnected from the rest of nothing
    con = connect_edges_from_dataframe(make_edges(pairs))
    with pytest.warns(UserWarning, match="exhausted"):
        s = snowball_sample(con, "a", max_nodes=100, per_node_cap=10, rng_seed=0)
    assert sorted(s.nodes) == ["a", "b", "c"]


def test_pick_seeds_activity_band():
    # "mid" has 3 edges, "hub" has 30, leaves have 1 each.
    pairs = [("mid", f"m{i}") for i in range(3)] + [
        ("hub", f"h{i}") for i in range(30)
    ]
    con = connect_edges_from_dataframe(make_edges(pairs))
    seeds = pick_seeds(con, min_tx=2, max_tx=10, limit=5)
    assert list(seeds.columns) == ["address", "n_tx"]
    assert seeds["address"].tolist() == ["mid"]
    assert seeds["n_tx"].tolist() == [3]


def test_save_sample_writes_files_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # save_sample writes under ./data/samples/
    pairs = [("s", "a"), ("s", "b")]
    con = connect_edges_from_dataframe(make_edges(pairs))
    s = snowball_sample(con, "s", max_nodes=3, per_node_cap=10, rng_seed=7)
    out = save_sample(s, "unit")
    assert (out / "nodes.parquet").exists()
    assert (out / "edges.parquet").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == "s"
    assert manifest["rng_seed"] == 7
    assert manifest["n_nodes"] == len(s.nodes)
    assert manifest["n_edges"] == len(s.edges)
    with pytest.raises(FileExistsError):
        save_sample(s, "unit")
