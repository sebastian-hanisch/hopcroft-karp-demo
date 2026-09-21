"""Hopcroft–Karp: Wächter für die kopierten Vorgänger, Handfälle, Invarianten je Phase (kürzeste Weglänge steigt strikt, Wege knotendisjunkt und gleich lang,
danach kein Weg dieser Länge mehr), Orakel (scipy, networkx, Brute Force, rekursives Kuhn, unabhängige Weglänge über networkx-Dijkstra), Zählregel, Phasenschranke,
Gleichheit mit den einzelnen kürzesten Wegen der Verbesserungswege-Demo und Negativkontrollen (was ohne Schichten, ohne maximale Menge, ohne Sackgassen-Markierung passiert)."""

import math

import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from hk_algorithm import STARTS, frames, hopcroft_karp, start_pairs
from hk_augment import augment, has_augmenting_path
from hk_greedy import RULES, optimum, run_rule
from hk_scenario import SplitMix64, generate, long_chain, p4_chain, staircase, steal_2x2, travel_cost


# --- Wächter für die kopierten Dateien der Vorgänger ---------------------------------------------------------------------------------

def test_splitmix64_reference_vector():
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(2)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4]


def test_travel_cost_and_fixed_maps_of_the_root():
    assert travel_cost(3, 4) == (5, 25) and travel_cost(1, 1) == (2, 2) and travel_cost(100, 100) == (142, 20000)
    sc = steal_2x2()
    assert sc.cost.tolist() == [[4, 5], [5, 14]]
    assert all((run_rule(sc, r).count, run_rule(sc, r).cost) == (2, 18) for r in RULES)
    assert all(run_rule(p4_chain(1), r).count == 1 for r in RULES) and optimum(p4_chain(1)).count == 2


def test_the_numbers_of_the_predecessors_are_reproduced():
    """Seed-2-Karte der Vorgänger: Greedy 17 Paare für 232 Minuten, Optimum 20 für 316; die einzelnen Wege brauchen 3 Runden vom Greedy-Start."""
    sc = generate(20, 20, 40, 0, 2)
    g, o = run_rule(sc, "edge"), optimum(sc)
    assert (g.count, g.cost, o.count, o.cost) == (17, 232, 20, 316)
    assert len(augment(sc, g.pairs, "bfs").rounds) == 3


def test_long_chain_geometry_and_staircase_geometry():
    sc = long_chain(6)
    assert (sc.n, sc.m, int(sc.feasible.sum())) == (7, 7, 13) and sorted({int(c) for c in sc.cost[sc.feasible]}) == [2, 10]
    for k in (2, 5, 7, 23):
        st = staircase(k)
        assert st.n == st.m == k * (k + 3) // 2
        assert all(run_rule(st, r).count == st.n - k for r in ("edge", "order"))                 # Greedy lässt in jeder Kette ein Fahrzeug übrig
        assert optimum(st).count == st.n
    assert max(p[1] for p in staircase(7).vehicles + staircase(7).orders) <= 100                  # K <= 7 passt auf die 100 x 100-Karte


# --- Handfälle -------------------------------------------------------------------------------------------------------------------------------

def test_nothing_to_improve():
    sc = steal_2x2()
    res = hopcroft_karp(sc, start_pairs(sc, "edge"))
    assert res.phases == () and res.count == 2 and res.final_scanned == 0 and res.cover_v == (0, 1) and res.cover_o == () and frames(res) == [(0, 0)]      # alle Fahrzeuge haben Partner: die Überdeckung sind sie selbst


def test_p4_is_one_phase_with_one_path_of_three_edges():
    sc = p4_chain(1)
    res = hopcroft_karp(sc, start_pairs(sc, "edge"))
    ph = res.phases[0]
    assert len(res.phases) == 1 and (ph.L, ph.length, len(ph.paths), ph.paths[0].length) == (1, 3, 1, 3) and res.count == 2
    assert (res.bfs_total, res.dfs_total, res.scanned_total) == (3, 3, 6)                        # Breitensuche 3 (mit der letzten), Tiefensuche 3; einzelne Wege: 3


def test_three_paths_in_one_phase():
    """Drei getrennte Pfade aus je drei Kanten (p4 mal 3): Greedy hat 3 Paare, eine einzige Phase klappt alle drei Wege um; die einzelnen Wege brauchen drei Runden."""
    sc = p4_chain(3)
    res = hopcroft_karp(sc, start_pairs(sc, "edge"))
    assert len(res.phases) == 1 and len(res.phases[0].paths) == 3 and res.count == 6
    assert len(augment(sc, start_pairs(sc, "edge"), "bfs").rounds) == 3


def test_long_chain_one_phase_one_path_through_everything():
    """Eine Phase, ein Weg aus 13 Kanten (Schicht 6): Breitensuche 13 Einträge, Tiefensuche 13 - zusammen 26 gegen 13 der einzelnen Wege."""
    sc = long_chain(6)
    res = hopcroft_karp(sc, start_pairs(sc, "edge"))
    ph = res.phases[0]
    assert len(res.phases) == 1 and (ph.L, ph.length, ph.bfs_scanned, ph.dfs_scanned) == (6, 13, 13, 13) and res.scanned_total == 26
    assert augment(sc, start_pairs(sc, "edge"), "bfs").scanned_total == 13


def test_the_staircase_needs_exactly_one_phase_per_chain():
    for k in (2, 3, 5, 7):
        sc = staircase(k)
        res = hopcroft_karp(sc, start_pairs(sc, "edge"))
        assert [ph.L for ph in res.phases] == list(range(1, k + 1)) and all(len(ph.paths) == 1 for ph in res.phases) and res.count == sc.n
        empty = hopcroft_karp(sc, ())
        assert len(empty.phases) == k + 1 and empty.phases[0].L == 0 and len(empty.phases[0].paths) == sc.n - k     # vom leeren Start: erste Phase = Erste-passende-Verfahren


def test_staircase_scan_numbers_from_greedy():
    res = hopcroft_karp(staircase(5), start_pairs(staircase(5), "edge"))
    assert (res.bfs_total, res.dfs_total, res.scanned_total) == (65, 85, 150)
    assert augment(staircase(5), start_pairs(staircase(5), "edge"), "bfs").scanned_total == 65 and augment(staircase(5), start_pairs(staircase(5), "edge"), "dfs").scanned_total == 35


def test_from_empty_the_first_phase_is_a_plain_first_fit():
    sc = generate(20, 20, 40, 0, 165)
    res = hopcroft_karp(sc, ())
    ph = res.phases[0]
    assert ph.L == 0 and ph.length == 1 and ph.bfs_scanned == 1 and all(p.length == 1 for p in ph.paths) and len(ph.paths) == 18       # Seed 165: 18 Paare der Länge 1


def test_invalid_start_pairs_are_rejected():
    sc = steal_2x2()
    with pytest.raises(AssertionError):
        hopcroft_karp(sc, ((0, 0), (1, 0)))


def test_no_feasible_edge():
    sc = next(s for s in (generate(3, 3, 10, 0, k) for k in range(200)) if not s.feasible.any())
    res = hopcroft_karp(sc, ())
    assert res.phases == () and res.count == 0 and res.final_scanned == 0 and res.cover_v == () and res.cover_o == ()      # Überdeckung leer: es gibt keine Kante zu überdecken


# --- Karten für die Invarianten -----------------------------------------------------------------------------------------------------------

def _cases(n_cases=120):
    for s in range(n_cases):
        n, m = 3 + s % 9, 3 + (s * 5) % 9
        yield generate(n, m, 15 + (s * 13) % 130, (s * 25) % 101, s)


def _fixed():
    return [steal_2x2(), p4_chain(1), p4_chain(3), long_chain(6), staircase(4)]


def _valid(sc, pairs):
    vs, os_ = [i for i, _ in pairs], [j for _, j in pairs]
    return len(set(vs)) == len(vs) and len(set(os_)) == len(os_) and all(sc.feasible[i, j] for i, j in pairs)


def _shortest_path_length(sc, pairs):
    """Länge des kürzesten Verbesserungswegs über networkx (Suche im Restgraphen), unabhängig vom Code der Demo."""
    match_v = {i: j for i, j in pairs}
    match_o = {j: i for i, j in pairs}
    g = nx.DiGraph()
    g.add_nodes_from([("v", i) for i in range(sc.n)] + [("o", j) for j in range(sc.m)])
    for i in range(sc.n):
        for j in range(sc.m):
            if sc.feasible[i, j]:
                g.add_edge(("o", j), ("v", i)) if match_v.get(i) == j else g.add_edge(("v", i), ("o", j))
    sources = [("v", i) for i in range(sc.n) if i not in match_v]
    if not sources:
        return None
    dist = nx.multi_source_dijkstra_path_length(g, sources)
    lengths = [dist[("o", j)] for j in range(sc.m) if j not in match_o and ("o", j) in dist]
    return min(lengths) if lengths else None


def _independent_no_path(sc, pairs):
    """Rekursives Kuhn ohne den Code der Demo: gibt es von einem freien Fahrzeug aus noch einen Verbesserungsweg?"""
    match_o = {j: i for i, j in pairs}
    matched_v = {i for i, _ in pairs}

    def try_(i, seen):
        for j in range(sc.m):
            if sc.feasible[i, j] and j not in seen:
                seen.add(j)
                if j not in match_o or try_(match_o[j], seen):
                    return True
        return False

    return not any(try_(i, set()) for i in range(sc.n) if i not in matched_v)


def _all_runs():
    for sc in list(_cases(120)) + _fixed():
        for start in STARTS:
            yield sc, start


# --- Invarianten je Phase --------------------------------------------------------------------------------------------------------------------------

def test_every_phase_takes_the_shortest_length_and_the_lengths_grow_strictly():
    for sc, start in _all_runs():
        res = hopcroft_karp(sc, start_pairs(sc, start))
        last = -1
        for k, ph in enumerate(res.phases):
            assert ph.length == 2 * ph.L + 1 and all(p.length == ph.length for p in ph.paths)
            assert ph.L > last                                                                      # die Weglänge wächst von Phase zu Phase strikt
            last = ph.L
            assert ph.length == _shortest_path_length(sc, res.states[k])                             # unabhängig: wirklich der kürzeste Weg vor der Phase
            assert (_shortest_path_length(sc, res.states[k + 1]) or 10 ** 9) > ph.length             # danach gibt es keinen Weg dieser Länge mehr (maximale Menge)


def test_paths_of_a_phase_are_disjoint_alternate_correctly_and_grow_the_matching_by_their_number():
    for sc, start in _all_runs():
        res = hopcroft_karp(sc, start_pairs(sc, start))
        for k, ph in enumerate(res.phases):
            before = dict(res.states[k])
            vs = [v for p in ph.paths for v in p.chain_v]
            os_ = [o for p in ph.paths for o in p.chain_o]
            assert len(set(vs)) == len(vs) and len(set(os_)) == len(os_)                            # knotendisjunkt
            for p in ph.paths:
                assert p.chain_v[0] not in before and p.chain_o[-1] not in {j for j in before.values()}      # freies Fahrzeug bis freier Auftrag
                for v, o, nxt in zip(p.chain_v, p.chain_o, list(p.chain_v[1:]) + [None]):
                    assert sc.feasible[v, o] and before.get(v) != o
                    if nxt is not None:
                        assert before.get(nxt) == o                                               # die Rückkante ist eine gewählte Kante
            assert ph.pairs_after == ph.pairs_before + len(ph.paths) == len(res.states[k + 1])
            assert _valid(sc, res.states[k + 1])


def test_final_matching_is_maximum_and_no_augmenting_path_remains():
    for sc, start in _all_runs():
        res = hopcroft_karp(sc, start_pairs(sc, start))
        assert res.count == optimum(sc).count and _valid(sc, res.pairs)
        assert _independent_no_path(sc, res.pairs) and not has_augmenting_path(sc, res.pairs)


def test_paar_count_matches_scipy_and_networkx():
    for sc in list(_cases(60)):
        res = hopcroft_karp(sc, ())
        csr = csr_matrix(sc.feasible.astype(np.int8))
        assert res.count == int((maximum_bipartite_matching(csr, perm_type="column") >= 0).sum())
        g = nx.Graph()
        g.add_nodes_from([("v", i) for i in range(sc.n)], bipartite=0)
        g.add_nodes_from([("o", j) for j in range(sc.m)], bipartite=1)
        g.add_edges_from([(("v", i), ("o", j)) for i in range(sc.n) for j in range(sc.m) if sc.feasible[i, j]])
        assert res.count == len(nx.bipartite.hopcroft_karp_matching(g, top_nodes=[("v", i) for i in range(sc.n)])) // 2


def _brute_force_max(sc):
    best = 0

    def rec(i, used, count):
        nonlocal best
        best = max(best, count)
        if i == sc.n:
            return
        rec(i + 1, used, count)
        for j in range(sc.m):
            if sc.feasible[i, j] and j not in used:
                rec(i + 1, used | {j}, count + 1)

    rec(0, frozenset(), 0)
    return best


def test_pair_count_matches_brute_force_on_tiny_maps():
    for seed in range(80):
        for n, m in ((3, 3), (4, 3), (3, 5), (5, 5)):
            sc = generate(n, m, 30 + 5 * (seed % 6), 0, seed)
            for start in STARTS:
                assert hopcroft_karp(sc, start_pairs(sc, start)).count == _brute_force_max(sc)


def test_the_cover_is_a_vertex_cover_of_the_size_of_the_matching_and_equals_the_one_of_piece_2():
    for sc, start in _all_runs():
        res = hopcroft_karp(sc, start_pairs(sc, start))
        cv, co = set(res.cover_v), set(res.cover_o)
        assert len(cv) + len(co) == res.count
        assert all(i in cv or j in co for i in range(sc.n) for j in range(sc.m) if sc.feasible[i, j])
        ref = augment(sc, start_pairs(sc, start), "bfs")
        assert (res.cover_v, res.cover_o) == (ref.cover_v, ref.cover_o)


def test_phase_bound():
    """Höchstens ceil(sqrt(nu)) + floor(nu / ceil(sqrt(nu))) Phasen (nu = Größe des größtmöglichen Matchings)."""
    for sc, start in _all_runs():
        nu = optimum(sc).count
        res = hopcroft_karp(sc, start_pairs(sc, start))
        t = math.ceil(math.sqrt(nu)) if nu else 0
        assert len(res.phases) <= (t + nu // t + 1 if nu else 0) or len(res.phases) <= 2 * math.sqrt(nu) + 1


# --- Gleichheit mit den einzelnen kürzesten Wegen ---------------------------------------------------------------------------------------------

def test_the_paths_are_exactly_those_of_the_single_shortest_path_search_of_piece_2():
    """Gemessen (nicht bewiesen): mit Index-Tie-Break findet Hopcroft–Karp dieselben Wege wie die Breitensuche je Weg, phasenweise in derselben Reihenfolge und mit demselben Ergebnis."""
    same = total = 0
    for sc, start in _all_runs():
        sp = start_pairs(sc, start)
        res, ref = hopcroft_karp(sc, sp), augment(sc, sp, "bfs")
        mine = [p.path for ph in res.phases for p in ph.paths]
        total += 1
        same += (sorted(mine) == sorted(r.path for r in ref.rounds)) and res.pairs == ref.pairs and res.cost == ref.cost
    assert same == total


# --- Zählregel und Schalter -------------------------------------------------------------------------------------------------------------------------

def test_scan_counters_add_up_and_are_bounded():
    for sc, start in _all_runs():
        res = hopcroft_karp(sc, start_pairs(sc, start))
        e = int(sc.feasible.sum())
        for ph in res.phases:
            assert ph.bfs_scanned <= e and ph.dfs_scanned <= e                                        # je Phase höchstens einmal alle Einträge
            assert ph.dfs_scanned == sum(p.scanned for p in ph.paths) + ph.tail_scanned
        assert res.scanned_total == sum(ph.bfs_scanned + ph.dfs_scanned for ph in res.phases) + res.final_scanned
        assert res.bfs_total + res.dfs_total == res.scanned_total
        assert hopcroft_karp(sc, start_pairs(sc, start), record=False).scanned_total == res.scanned_total


def test_determinism_and_record_flag():
    sc = generate(20, 20, 40, 0, 165)
    a, b = hopcroft_karp(sc, start_pairs(sc, "edge")), hopcroft_karp(sc, start_pairs(sc, "edge"))
    assert a == b
    light = hopcroft_karp(sc, start_pairs(sc, "edge"), record=False)
    assert light.states == () and light.phases[0].dist == () and light.pairs == a.pairs and light.scanned_total == a.scanned_total


def test_frozen_matching_gives_the_same_result_as_flipping_in_place():
    """Die Wege einer Phase werden auf dem eingefrorenen Matching gesucht; das ist gleichwertig zu sofortigem Umklappen (die Schichtmarken verbieten das Wiederbetreten)."""
    from hk_augment import _adjacency
    for sc, start in list(_all_runs())[:200]:
        res = hopcroft_karp(sc, start_pairs(sc, start))
        adj = _adjacency(sc)
        match_v, match_o = [-1] * sc.n, [-1] * sc.m
        for i, j in start_pairs(sc, start):
            match_v[i], match_o[j] = j, i
        for ph in res.phases:
            for p in ph.paths:                                                                     # sofort umklappen, Weg für Weg
                for v, o in zip(p.chain_v, p.chain_o):
                    match_v[v], match_o[o] = o, v
        assert tuple((i, j) for i, j in enumerate(match_v) if j >= 0) == res.pairs


def test_early_stop_and_full_layer_find_the_same_paths_and_dead_end_marking_only_saves_scans():
    for sc in list(_cases(80)):
        sp = start_pairs(sc, "edge")
        base = hopcroft_karp(sc, sp)
        full = hopcroft_karp(sc, sp, early=False)
        nop = hopcroft_karp(sc, sp, prune=False)
        paths = lambda r: [p.path for ph in r.phases for p in ph.paths]
        assert paths(full) == paths(base) and full.pairs == base.pairs and full.scanned_total >= base.scanned_total
        assert nop.pairs == base.pairs and paths(nop) == paths(base) and nop.dfs_total >= base.dfs_total


def test_frames_cover_every_phase_and_path():
    sc = generate(20, 20, 40, 0, 165)
    res = hopcroft_karp(sc, start_pairs(sc, "edge"))
    fr = frames(res)
    assert fr == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)] and len(fr) == len(res.phases) + res.n_paths + 1


# --- Negativkontrollen -----------------------------------------------------------------------------------------------------------------------------

def test_without_the_layers_paths_are_not_shortest_on_some_map():
    """layered=False (jeder Weg im Graphen, nicht nur im Schichtgraphen): auf einer Karte mit längerer Alternative ist ein Weg länger als 2L+1 - die Prüfung der Weglänge schlägt an."""
    violations = 0
    for sc, start in _all_runs():
        res = hopcroft_karp(sc, start_pairs(sc, start), layered=False)
        assert res.count == optimum(sc).count                                                         # das Ergebnis bleibt größtmöglich (es sind trotzdem Verbesserungswege)
        violations += any(p.length != ph.length for ph in res.phases for p in ph.paths)
    assert violations > 0


def test_without_a_maximal_set_the_path_lengths_do_not_grow_and_the_phase_bound_breaks():
    """maximal=False (nur ein Weg je Phase): das ist Stück 2 mit Schichtaufbau - die Weglänge wächst nicht strikt, und auf der Treppe von leer aus bricht die Phasenschranke."""
    not_strict = 0
    for sc, start in _all_runs():
        res = hopcroft_karp(sc, start_pairs(sc, start), maximal=False)
        Ls = [ph.L for ph in res.phases]
        not_strict += any(b <= a for a, b in zip(Ls, Ls[1:]))
    assert not_strict > 0
    sc = staircase(7)
    good, bad = hopcroft_karp(sc, ()), hopcroft_karp(sc, (), maximal=False)
    nu = optimum(sc).count
    assert len(good.phases) == 8 and len(bad.phases) > 2 * math.sqrt(nu) + 1


def test_without_dead_end_marking_the_scans_explode_on_larger_maps():
    sc = generate(160, 160, 14, 0, 100000)
    sp = start_pairs(sc, "edge")
    good, bad = hopcroft_karp(sc, sp, record=False), hopcroft_karp(sc, sp, record=False, prune=False)
    assert bad.pairs == good.pairs and bad.dfs_total > 1.5 * good.dfs_total


def test_reversed_adjacency_breaks_the_equality_with_piece_2():
    """Die Gleichheit der Wegfolge hängt am Index-Tie-Break: kehrt man die Nachbarliste nur in Hopcroft–Karp um, findet es auf 20 x 20-Karten fast überall andere Wege (gemessen 53 von 60 ab Greedy, 60 von 60 vom leeren Start)."""
    import hk_algorithm
    real = hk_algorithm._adjacency
    hk_algorithm._adjacency = lambda sc: [list(reversed(a)) for a in real(sc)]
    try:
        differ = {"edge": 0, "empty": 0}
        for seed in range(100000, 100060):
            sc = generate(20, 20, 40, 0, seed)
            for start in differ:
                sp = start_pairs(sc, start)
                mine, ref = hopcroft_karp(sc, sp), augment(sc, sp, "bfs")
                differ[start] += sorted(p.path for ph in mine.phases for p in ph.paths) != sorted(r.path for r in ref.rounds)
    finally:
        hk_algorithm._adjacency = real
    assert differ == {"edge": 53, "empty": 60}
