import numpy as np
from tests.test_tgn_streaming import graph_from
from tgn.ranking import filtered_rank, recency_ranking

def test_filtered_rank_tie_aware():
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    # target idx 1: one better, two tied-others -> 1 + 1 + 1 = 3
    assert filtered_rank(scores, 1, np.array([], np.int64)) == 3.0
    # excluding the better candidate: 1 + 0 + 1 = 2
    assert filtered_rank(scores, 1, np.array([0], np.int64)) == 2.0

def test_recency_ranking_hand_example():
    # nodes 0..3; train: (0,1)@d0, (0,2)@d2 ; test day 4: (0,1)
    g = graph_from([0, 0, 0], [1, 2, 1], [0, 2, 4], n_nodes=4, n_days=5)
    g.train_end = 2
    g.val_end = 2
    r = recency_ranking(g)
    # candidates for s=0: (0,1) last d0 -> score -4 ; (0,2) last d2 -> -2 ;
    # (0,0),(0,3) unseen -> -inf. Rank of 1: worse than 2, ties none -> 2
    assert r["mrr"]["all"] == 0.5
    assert r["hits1"]["all"] == 0.0
    assert r["hits10"]["all"] == 1.0

def test_recency_ranking_filters_same_day_positives():
    # test day 4 has positives (0,1) and (0,2); (0,2) is more recent (d2 vs d0)
    # but must be excluded when ranking (0,1)'s destination
    g = graph_from([0, 0, 0, 0], [1, 2, 1, 2], [0, 2, 4, 4],
                   n_nodes=4, n_days=5)
    g.train_end = 2
    g.val_end = 2
    r = recency_ranking(g)
    # (0,1): candidates {0,1,3} after excluding 2 -> 1 is best seen -> rank 1
    # (0,2): candidates {0,2,3} after excluding 1 -> rank 1
    assert r["mrr"]["all"] == 1.0

def test_recency_unseen_target_gets_tied_rank():
    # test event (0,3) never seen; candidates for s=0: 1 seen (better),
    # {0,2,3} tied at -inf (target + 2 tied-others) -> rank 1+1+0.5*2 = 3
    g = graph_from([0, 0], [1, 3], [0, 4], n_nodes=4, n_days=5)
    g.train_end = 1
    g.val_end = 1
    r = recency_ranking(g)
    assert r["mrr"]["all"] == 1.0 / 3.0
    assert r["mrr"]["unseen"] == 1.0 / 3.0
