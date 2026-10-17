"""Literatur-Referenzwerte für Hopcroft-Karp (alle Quellen am 2026-10-17 eingesehen).

 L1  Dokumentation der Referenzsoftware SciPy, `scipy.sparse.csgraph.maximum_bipartite_matching` (benutzt laut Dokumentation Hopcroft-Karp 1973),
     https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_bipartite_matching.html, Abschnitt "Examples":
     - graph = [[0, 0, 1], [1, 1, 0]]: Ausgabe (perm_type='column') [2 0], also zwei Paare (Zeile 0 - Spalte 2, Zeile 1 - Spalte 0);
     - graph = csr_array((2, 0)): leere Seite, "the matching is empty";
     - a = [[0, 1, 2, 0], [1, 0, 0, 1], [2, 0, 0, 3], [0, 1, 3, 0]]: "square graph with a perfect matching", die Dokumentation gibt die Zeilenpermutation an, bei der alle
       Diagonaleinträge belegt sind (graph[perm] = [[1 0 0 1] [0 1 2 0] [0 1 3 0] [2 0 0 3]]), also vier Paare.
     Die Dokumentation sagt ausdrücklich, dass bei mehreren größten Matchings die Ausgabe je Version abweicht; verglichen wird deshalb die Paarzahl und dass das dokumentierte Matching
     ein gültiges größtes Matching ist, nicht welches gewählt wird.
 L2  Cornell CS 6820 (Fall 2024), Vorlesungsnotizen "Matchings" (26 Aug - 11 Sep 2023), Abschnitt 1.4 "The Hopcroft-Karp algorithm",
     https://www.cs.cornell.edu/courses/cs6820/2024fa/handouts/matchings.pdf:
     - Lemma 4: Sind P1, ..., Pk knotendisjunkte M-Verbesserungswege, ist M xor (P1 ... Pk) ein Matching mit |M| + k Kanten.
     - Lemma 5: Mit k = |M*| - |M| (M* größtes Matching) gibt es einen M-Verbesserungsweg mit weniger als n/k Kanten, n = Zahl der Knoten des Graphen.
     - Lemma 6: Die Mindestlänge eines Verbesserungswegs wächst in jeder Iteration (mit nichtleerer Wegemenge) strikt.
     - Lemma 7: weniger als 2 sqrt(n) Iterationen (die Wurzel ist im Textlayer der PDF-Datei verloren gegangen, die Lesart 2 sqrt(n) folgt aus dem Beweis der Quelle: erst sqrt(n)
       Iterationen bis zur Weglänge > sqrt(n), danach höchstens |M*| - |M| < sqrt(n) Iterationen), n = Zahl der Knoten.
     Das sind Aussagen, keine Rechenwerte: sie werden auf Zufallsgraphen geprüft.

NICHT als Literaturwert verwendet (Quelle nicht einsehbar): die Originalarbeit Hopcroft/Karp 1973 (SIAM J. Comput. 2(4), 225-231; das Hosting der Universität erreichte ich nicht, ein
Zertifikatsfehler), insbesondere ihre genaue Phasenschranke 2*floor(sqrt(s)) + 2 aus einer Zusammenfassung Dritter; kein Beispiel mit Phasenprotokoll gefunden.
"""

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from hk_algorithm import hopcroft_karp


def _sc(feas):
    f = np.array(feas, dtype=bool)
    n, m = f.shape
    return SimpleNamespace(n=n, m=m, feasible=f, cost=np.ones((n, m), dtype=np.int64))


# --- L1 SciPy-Dokumentation --------------------------------------------------------------------------------------------------------------------

def test_l1_scipy_documentation_examples():
    g1 = [[0, 0, 1], [1, 1, 0]]
    res = hopcroft_karp(_sc(g1), record=False)
    assert res.count == 2
    documented = ((0, 2), (1, 0))                                                    # Ausgabe [2 0] der Dokumentation
    assert all(g1[i][j] for i, j in documented) and len(documented) == res.count
    assert hopcroft_karp(_sc(np.zeros((2, 0), dtype=bool)), record=False).count == 0    # csr_array((2, 0)): leeres Matching
    a = [[0, 1, 2, 0], [1, 0, 0, 1], [2, 0, 0, 3], [0, 1, 3, 0]]
    res = hopcroft_karp(_sc(np.array(a) != 0), record=False)
    assert res.count == 4                                                            # perfektes Matching
    perm = (1, 0, 3, 2)                                                              # graph[perm] hat die Diagonale [1, 1, 3, 3]: Zeile perm[k] - Spalte k
    assert all(a[perm[k]][k] != 0 for k in range(4))
    assert [a[perm[k]][k] for k in range(4)] == [1, 1, 3, 3]
    assert sorted(i for i, _ in res.pairs) == [0, 1, 2, 3] and sorted(j for _, j in res.pairs) == [0, 1, 2, 3]


# --- L2 Cornell-Notizen, Lemma 4 bis 7 -----------------------------------------------------------------------------------------------------------

def _graphs(count, seed):
    rng = random.Random(seed)
    for _ in range(count):
        n, m = rng.randint(2, 16), rng.randint(2, 16)
        p = rng.choice([0.08, 0.15, 0.3, 0.6])
        feas = [[rng.random() < p for _ in range(m)] for _ in range(n)]
        used, start = set(), []
        for i in rng.sample(range(n), n):
            free = [j for j in range(m) if feas[i][j] and j not in used]
            if free and rng.random() < 0.6:
                j = rng.choice(free)
                used.add(j)
                start.append((i, j))
        yield feas, tuple(start)


def test_l2_lemma_4_to_7_hold_on_random_graphs_and_starting_matchings():
    csgraph = pytest.importorskip("scipy.sparse.csgraph")
    from scipy.sparse import csr_matrix
    seen_phases = 0
    for feas, start in _graphs(200, 2024):
        sc = _sc(feas)
        nv = sc.n + sc.m
        nu = int((csgraph.maximum_bipartite_matching(csr_matrix(sc.feasible.astype(int)), perm_type="column") >= 0).sum())
        res = hopcroft_karp(sc, start, record=False)
        assert res.count == nu
        size, last = len(start), -1
        for ph in res.phases:
            seen_phases += 1
            k = nu - size
            assert k >= 1 and ph.length < nv / k                                       # Lemma 5: der kürzeste Weg ist kürzer als n / k
            assert ph.pairs_after == ph.pairs_before + len(ph.paths) == size + len(ph.paths)      # Lemma 4: k disjunkte Wege erhöhen das Matching um k
            assert ph.length > last                                                    # Lemma 6: Mindestlänge wächst strikt
            last, size = ph.length, ph.pairs_after
        assert len(res.phases) < 2 * math.sqrt(nv) or not res.phases                   # Lemma 7: weniger als 2 sqrt(n) Iterationen
    assert seen_phases > 120
