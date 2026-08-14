"""No-grad day-by-day replay for TGN evaluation.

Discipline (TASK.md §3): each day is scored with memory reflecting strictly
earlier days; memory then advances with that day's observed positives — the
same streaming as tgat.evaluate.edgebank_scores. Queries never enter memory.
"""
import numpy as np
import torch


def day_ranges(day: np.ndarray, n_days: int | None = None) -> np.ndarray:
    if n_days is None:
        n_days = int(day.max()) + 1 if len(day) else 0
    return np.searchsorted(day, np.arange(n_days + 1), side="left")


@torch.no_grad()
def score_pairs_streaming(model, g, store, device, s, d, t, batch=500):
    model.eval()
    mem, last = model.init_memory(g.n_nodes, device)
    off = day_ranges(g.day, g.n_days)
    logit = np.empty(len(s), np.float32)
    amt = np.empty(len(s), np.float32)
    order = np.argsort(t, kind="stable")
    qi, Q = 0, len(order)
    top_day = int(np.max(t)) + 1 if len(t) else 0
    for T in range(top_day):
        qj = qi
        while qj < Q and t[order[qj]] == T:
            qj += 1
        for lo in range(qi, qj, batch):
            idx = order[lo:lo + batch]
            lg, am = model(s[idx], d[idx], t[idx], mem, store, device)
            logit[idx] = lg.cpu().numpy()
            amt[idx] = am.cpu().numpy()
        qi = qj
        if T + 1 < len(off):
            elo, ehi = off[T], off[T + 1]
            if ehi > elo:
                mem, last = model.apply_messages(
                    mem, last, g.src[elo:ehi], g.dst[elo:ehi], T,
                    g.edge_feat[elo:ehi])
    return logit, amt


@torch.no_grad()
def iter_day_embeddings(model, g, store, device, days, chunk=2048):
    model.eval()
    mem, last = model.init_memory(g.n_nodes, device)
    off = day_ranges(g.day, g.n_days)
    want = sorted(int(x) for x in set(days))
    wi = 0
    for T in range(g.n_days):
        if wi >= len(want):
            return
        if T == want[wi]:
            zs = [model.embed(np.arange(lo, min(lo + chunk, g.n_nodes)),
                              np.full(min(chunk, g.n_nodes - lo), T),
                              mem, store, device)
                  for lo in range(0, g.n_nodes, chunk)]
            yield T, torch.cat(zs)
            wi += 1
        elo, ehi = off[T], off[T + 1]
        if ehi > elo:
            mem, last = model.apply_messages(
                mem, last, g.src[elo:ehi], g.dst[elo:ehi], T,
                g.edge_feat[elo:ehi])
