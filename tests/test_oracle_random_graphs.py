"""Unabhängiges Orakel auf zufälligen, nicht geometrischen Bipartitgraphen (die übrigen Tests nutzen Kartengeometrie): Matchinggröße gegen scipy
`maximum_bipartite_matching`, je Phase die kürzeste Wegelänge per eigener Breitensuche im Wechselgraphen, knotendisjunkte Wege, Wegemenge maximal
(danach kein Weg dieser Länge mehr), Weglängen strikt wachsend, Phasenschranke 2 sqrt(nu) + 1 und Knotenüberdeckung (König)."""

import math
import random
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from hk_algorithm import hopcroft_karp


def _shortest_augmenting(n, m, feas, matching):
    mv, mo = dict(matching), {j: i for i, j in matching}
    dist, queue = {}, deque()
    for i in range(n):
        if i not in mv:
            dist[("v", i)] = 0
            queue.append(("v", i))
    while queue:
        kind, x = queue.popleft()
        d = dist[(kind, x)]
        if kind == "v":
            for j in range(m):
                if feas[x][j] and mv.get(x) != j and ("o", j) not in dist:
                    dist[("o", j)] = d + 1
                    queue.append(("o", j))
        else:
            if x not in mo:
                return d
            if ("v", mo[x]) not in dist:
                dist[("v", mo[x])] = d + 1
                queue.append(("v", mo[x]))
    return None


def test_hopcroft_karp_on_random_bipartite_graphs_against_independent_references():
    csgraph = pytest.importorskip("scipy.sparse.csgraph")
    from scipy.sparse import csr_matrix
    rng = random.Random(20261004)
    for t in range(250):
        n, m = rng.randint(1, 14), rng.randint(1, 14)
        p = rng.choice([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0])
        feas = [[rng.random() < p for _ in range(m)] for _ in range(n)]
        start = []
        if t % 2:                                                                  # beliebiges (nicht greedy) Startmatching
            used = set()
            for i in rng.sample(range(n), n):
                free = [j for j in range(m) if feas[i][j] and j not in used]
                if free and rng.random() < 0.7:
                    j = rng.choice(free)
                    used.add(j)
                    start.append((i, j))
        sc = SimpleNamespace(n=n, m=m, feasible=np.array(feas, dtype=bool).reshape(n, m), cost=np.ones((n, m), dtype=np.int64))
        res = hopcroft_karp(sc, tuple(start))
        nu = int((csgraph.maximum_bipartite_matching(csr_matrix(sc.feasible.astype(int)), perm_type="column") >= 0).sum())
        assert res.count == nu
        current, last = set(start), -1
        for ph in res.phases:
            assert ph.length == _shortest_augmenting(n, m, feas, current) and ph.L > last
            last = ph.L
            vs = [v for p_ in ph.paths for v in p_.chain_v]
            os_ = [o for p_ in ph.paths for o in p_.chain_o]
            assert len(set(vs)) == len(vs) and len(set(os_)) == len(os_)
            for p_ in ph.paths:
                adds = {(i, j) for i, j, kind in p_.path if kind == "add"}
                drops = {(i, j) for i, j, kind in p_.path if kind == "drop"}
                assert len(adds) == len(drops) + 1 and drops <= current and not adds & current
                current = (current - drops) | adds
            after = _shortest_augmenting(n, m, feas, current)
            assert after is None or after > ph.length                                  # die Wegemenge war maximal
        assert current == set(res.pairs) and _shortest_augmenting(n, m, feas, current) is None
        assert nu == 0 or len(res.phases) <= 2 * math.sqrt(nu) + 1
        cv, co = set(res.cover_v), set(res.cover_o)
        assert len(cv) + len(co) == res.count and all(i in cv or j in co for i in range(n) for j in range(m) if feas[i][j])
