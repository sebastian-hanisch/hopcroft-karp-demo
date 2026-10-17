"""Eigenschaftstests (metamorph) für Hopcroft-Karp: Aussagen über größte Matchings in bipartiten Graphen, die gelten MÜSSEN, ohne dass man das Ergebnis vorher kennt.

Das vorhandene Orakel (test_oracle_random_graphs.py) vergleicht gegen scipy, Phasenbeweise und Knotenüberdeckung. Hier wird die EINGABE verändert und die Beziehung der
Ausgaben geprüft (feste Seeds, je Eigenschaft 30 bis 60 Zufallsgraphen, plus Randfälle; die Graphen sind beliebig, nicht geometrisch):

 H1 Umnummerieren der Fahrzeuge und Aufträge ändert die Paarzahl nicht; das Ergebnis ist ein gültiges, größtes Matching.
 H2 Spiegelung (Fahrzeuge und Aufträge tauschen die Rollen = transponierte Kantenmatrix): gleiche Paarzahl.
 H3 Eine zusätzliche Kante erhöht die Paarzahl um 0 oder 1, eine entfernte senkt sie um 0 oder 1; ein zusätzlicher Knoten ändert sie um höchstens 1 (nie ein Rückgang).
 H4 Das Startmatching ändert die Paarzahl nicht (leer, zufällig, das eigene Ergebnis); Hopcroft-Karp macht nie ein versorgtes Fahrzeug oder einen versorgten Auftrag
    wieder frei; aus einem schon größten Matching entsteht keine Phase und dieselbe Paarung.
 H5 Kostenblind: beliebige andere Kosten (auch Skalierung) lassen Paare, Phasen und Wege unverändert; nur die Kostensumme folgt.
 H6 Defektform des Hallschen Satzes (Ore): nu = n - max über alle Fahrzeugmengen S von (|S| - |N(S)|), Brute Force über alle Teilmengen (n <= 9).
 H7 Disjunkte Vereinigung: die Paarzahl ist die Summe der Teile, unabhängig von der Reihenfolge der Teile.
 H8 Jede Phase wächst um genau die Zahl ihrer knotendisjunkten Wege; alle Wege einer Phase haben dieselbe Länge 2L + 1; Länge und Zahl der Wege der Summe stimmen mit
    der Paarzahl überein (Berge, Lemma 4 der Cornell-Notizen).
 H9 Randfälle: leerer Graph, keine Kanten, vollständige Graphen, Stern, Permutationsmatrix, die lange Kette (ein Weg durch die ganze Kette).
"""

import itertools
import random
from types import SimpleNamespace

import numpy as np
import pytest

from hk_algorithm import hopcroft_karp


# --- Hilfen ----------------------------------------------------------------------------------------------------------------------------------

def _sc(feas, cost=None):
    f = np.array(feas, dtype=bool)
    if f.ndim != 2:
        f = f.reshape(len(feas), -1)
    n, m = f.shape
    c = np.ones((n, m), dtype=np.int64) if cost is None else np.array(cost, dtype=np.int64).reshape(n, m)
    return SimpleNamespace(n=n, m=m, feasible=f, cost=c)


def _random_graph(rng, nmax=12, mmax=12):
    n, m = rng.randint(1, nmax), rng.randint(1, mmax)
    p = rng.choice([0.05, 0.1, 0.2, 0.35, 0.6, 1.0])
    return [[rng.random() < p for _ in range(m)] for _ in range(n)]


def _random_start(rng, feas):
    n, m = len(feas), len(feas[0]) if feas else 0
    used, out = set(), []
    for i in rng.sample(range(n), n):
        free = [j for j in range(m) if feas[i][j] and j not in used]
        if free and rng.random() < 0.7:
            j = rng.choice(free)
            used.add(j)
            out.append((i, j))
    return tuple(out)


def _is_valid_matching(feas, pairs):
    vs, os_ = [i for i, _ in pairs], [j for _, j in pairs]
    return len(set(vs)) == len(vs) and len(set(os_)) == len(os_) and all(feas[i][j] for i, j in pairs)


def _nu(feas, start=()):
    return hopcroft_karp(_sc(feas), start, record=False).count


def _deficiency_nu(feas):
    """nu = n - max_S (|S| - |N(S)|) (Ore); S läuft über alle Teilmengen der Fahrzeuge."""
    n = len(feas)
    nb = [sum(1 << j for j, x in enumerate(row) if x) for row in feas]
    best = 0
    for mask in range(1 << n):
        union, size = 0, 0
        for i in range(n):
            if mask >> i & 1:
                union |= nb[i]
                size += 1
        best = max(best, size - bin(union).count("1"))
    return n - best


# --- H1 Umnummerieren ------------------------------------------------------------------------------------------------------------------------

def test_h1_relabelling_vehicles_and_orders_keeps_the_matching_size():
    rng = random.Random(101)
    for _ in range(60):
        feas = _random_graph(rng)
        n, m = len(feas), len(feas[0])
        qv, qo = rng.sample(range(n), n), rng.sample(range(m), m)
        feas2 = [[False] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                feas2[qv[i]][qo[j]] = feas[i][j]
        a, b = hopcroft_karp(_sc(feas), record=False), hopcroft_karp(_sc(feas2), record=False)
        assert a.count == b.count and _is_valid_matching(feas2, b.pairs)
        assert len(a.cover_v) + len(a.cover_o) == a.count                        # König: die Überdeckung hat genau nu Ecken, auch nach dem Umnummerieren
        assert len(b.cover_v) + len(b.cover_o) == b.count


# --- H2 Spiegelung ---------------------------------------------------------------------------------------------------------------------------

def test_h2_swapping_the_two_sides_keeps_the_matching_size():
    rng = random.Random(102)
    for _ in range(60):
        feas = _random_graph(rng)
        t = np.array(feas, dtype=bool).T.tolist()
        assert _nu(feas) == _nu(t)


# --- H3 Kanten und Knoten --------------------------------------------------------------------------------------------------------------------

def test_h3_adding_or_removing_one_edge_or_vertex_changes_nu_by_at_most_one_in_the_right_direction():
    rng = random.Random(103)
    for _ in range(60):
        feas = _random_graph(rng)
        n, m = len(feas), len(feas[0])
        base = _nu(feas)
        i, j = rng.randrange(n), rng.randrange(m)
        flipped = [list(r) for r in feas]
        flipped[i][j] = not flipped[i][j]
        new = _nu(flipped)
        assert (new - base) in ((0, 1) if flipped[i][j] else (-1, 0))
        row = [rng.random() < 0.4 for _ in range(m)]                           # zusätzliches Fahrzeug
        assert _nu(feas + [row]) - base in (0, 1)
        col = [rng.random() < 0.4 for _ in range(n)]                           # zusätzlicher Auftrag
        assert _nu([r + [col[k]] for k, r in enumerate(feas)]) - base in (0, 1)
        if n > 1:
            k = rng.randrange(n)                                               # ein Fahrzeug weniger
            assert base - _nu(feas[:k] + feas[k + 1:]) in (0, 1)


# --- H4 Startmatching ------------------------------------------------------------------------------------------------------------------------

def test_h4_the_start_matching_does_not_change_the_size_and_matched_vertices_stay_matched():
    rng = random.Random(104)
    for _ in range(60):
        feas = _random_graph(rng)
        start = _random_start(rng, feas)
        sc = _sc(feas)
        base = hopcroft_karp(sc, record=False)
        res = hopcroft_karp(sc, start, record=False)
        assert res.count == base.count and _is_valid_matching(feas, res.pairs)
        assert {i for i, _ in start} <= {i for i, _ in res.pairs} and {j for _, j in start} <= {j for _, j in res.pairs}
        again = hopcroft_karp(sc, res.pairs, record=False)                     # vom größten Matching aus: nichts mehr zu tun
        assert again.phases == () and again.pairs == res.pairs


# --- H5 kostenblind --------------------------------------------------------------------------------------------------------------------------

def test_h5_costs_never_change_pairs_phases_or_paths():
    rng = random.Random(105)
    for _ in range(40):
        feas = _random_graph(rng)
        n, m = len(feas), len(feas[0])
        start = _random_start(rng, feas)
        c1 = [[rng.randint(0, 50) for _ in range(m)] for _ in range(n)]
        c2 = [[7 * x + 3 for x in row] for row in c1]
        c3 = [[rng.randint(0, 50) for _ in range(m)] for _ in range(n)]       # ganz andere Kosten (nicht nur eine monotone Umrechnung)
        a = hopcroft_karp(_sc(feas, c1), start)
        b = hopcroft_karp(_sc(feas, c2), start)
        c = hopcroft_karp(_sc(feas, c3), start)
        trace = lambda r: [(p.L, [q.path for q in p.paths]) for p in r.phases]
        assert a.pairs == b.pairs == c.pairs and trace(a) == trace(b) == trace(c)
        assert b.cost == 7 * a.cost + 3 * a.count
        assert c.cost == sum(c3[i][j] for i, j in c.pairs)


# --- H6 Defektform des Hallschen Satzes ------------------------------------------------------------------------------------------------------

def test_h6_matching_size_equals_n_minus_maximum_deficiency():
    rng = random.Random(106)
    for _ in range(50):
        feas = _random_graph(rng, nmax=9, mmax=12)
        assert _nu(feas) == _deficiency_nu(feas)


# --- H7 disjunkte Vereinigung ----------------------------------------------------------------------------------------------------------------

def test_h7_nu_is_additive_over_disjoint_components_in_any_order():
    rng = random.Random(107)
    for _ in range(40):
        parts = [_random_graph(rng, 6, 6) for _ in range(rng.randint(2, 4))]
        total = 0
        for order in (parts, parts[::-1]):
            n = sum(len(p) for p in order)
            m = sum(len(p[0]) for p in order)
            big = [[False] * m for _ in range(n)]
            r0 = c0 = 0
            for p in order:
                for i, row in enumerate(p):
                    for j, x in enumerate(row):
                        big[r0 + i][c0 + j] = x
                r0 += len(p)
                c0 += len(p[0])
            val = _nu(big)
            total = total or val
            assert val == total == sum(_nu(p) for p in parts)


# --- H8 Phasenstruktur -----------------------------------------------------------------------------------------------------------------------

def test_h8_each_phase_grows_by_its_number_of_disjoint_paths_of_equal_length():
    rng = random.Random(108)
    for _ in range(60):
        feas = _random_graph(rng, 14, 14)
        start = _random_start(rng, feas)
        res = hopcroft_karp(_sc(feas), start, record=False)
        size = len(start)
        for ph in res.phases:
            assert ph.pairs_before == size and ph.pairs_after == size + len(ph.paths) and len(ph.paths) >= 1
            assert all(p.length == ph.length == 2 * ph.L + 1 for p in ph.paths)
            size = ph.pairs_after
        assert size == res.count == len(start) + res.n_paths


# --- H9 Randfälle ----------------------------------------------------------------------------------------------------------------------------

def test_h9_edge_cases():
    assert hopcroft_karp(_sc(np.zeros((0, 0), dtype=bool)), record=False).count == 0
    assert hopcroft_karp(_sc(np.zeros((3, 0), dtype=bool)), record=False).count == 0
    assert hopcroft_karp(_sc(np.zeros((0, 3), dtype=bool)), record=False).count == 0
    empty = hopcroft_karp(_sc([[False] * 4] * 3), record=False)
    assert empty.count == 0 and empty.phases == ()
    for n, m in itertools.product(range(1, 6), repeat=2):
        full = hopcroft_karp(_sc(np.ones((n, m), dtype=bool)), record=False)
        assert full.count == min(n, m)
    star = hopcroft_karp(_sc([[True]] * 7), record=False)                          # sieben Fahrzeuge, ein Auftrag
    assert star.count == 1 and len(star.phases) == 1
    perm = [3, 0, 4, 1, 2]
    pm = hopcroft_karp(_sc([[j == perm[i] for j in range(5)] for i in range(5)]), record=False)
    assert pm.pairs == tuple(sorted((i, perm[i]) for i in range(5))) and len(pm.phases) == 1
    for k in (1, 3, 8):                                                             # lange Kette: Fahrzeug i grenzt an Auftrag i und i - 1
        feas = [[j == i or j == i - 1 for j in range(k + 1)] for i in range(k + 1)]
        res = hopcroft_karp(_sc(feas), tuple((i + 1, i) for i in range(k)), record=False)      # Start: die k "billigen" Kanten (i + 1, i)
        assert res.count == k + 1 and len(res.phases) == 1 and res.phases[0].length == 2 * k + 1 and len(res.phases[0].paths) == 1
    with pytest.raises(AssertionError):
        hopcroft_karp(_sc([[True, True]]), ((0, 0), (0, 1)), record=False)          # Startpaare müssen ein Matching sein
