"""Hopcroft-Karp: größtmögliches Matching in Phasen. Je Phase EINE Breitensuche von allen freien Fahrzeugen (Schichten, Abbruch an der ersten
Schicht, in der ein freier Auftrag anliegt) und danach EINE Tiefensuche, die eine maximale Menge knotendisjunkter kürzester
Verbesserungswege im Schichtgraphen findet; alle Wege der Phase werden erst am Ende umgeklappt. Ab dem Wegfall des letzten Wegs liefert die
letzte, erfolglose Breitensuche die Knotenüberdeckung (Beweis, wie in der Verbesserungswege-Demo).

Kostenblind wie die Verbesserungswege: die Nachbarn stehen in Indexreihenfolge (Fahrzeuge aufsteigend, Breitensuche FIFO, die Tiefensuche nimmt die
erste passende Kante). Damit findet Hopcroft-Karp dieselben Wege in derselben Reihenfolge wie die einzelnen kürzesten Wege der Verbesserungswege-Demo
(gemessen, siehe Tests) - nur gebündelt.

Aufwand = geprüfte Adjazenzeinträge (wie dort), Breitensuche und Tiefensuche getrennt gezählt; nie Sekunden. Die Tiefensuche prüft jedes Fahrzeug
höchstens einmal je Phase (danach gesperrt oder Sackgasse); die Position im Stapel ist der "current arc".

Schalter für Negativkontrollen (Standard = richtiges Verfahren): `early` (Breitensuche stoppt beim ersten freien Auftrag), `prune` (Sackgassen
markieren), `layered` (nur Wege durch den Schichtgraphen), `maximal` (die Phase endet erst, wenn keine weiteren Wege der Länge 2L+1 mehr gefunden werden).
"""

from dataclasses import dataclass

from hk_augment import _adjacency, _apply, _path_from_orders, _snapshot
from hk_greedy import run_rule

START_EMPTY = "empty"
STARTS = (START_EMPTY, "edge", "order")


@dataclass(frozen=True)
class Path:
    chain_v: tuple         # Fahrzeuge v0, v1, ...
    chain_o: tuple         # Aufträge o0, o1, ...: (v0,o0) wird gewählt, (v1,o0) freigegeben, (v1,o1) gewählt, ...
    path: tuple            # ((i, j, "add" | "drop"), ...)
    length: int            # 2k+1 Kanten
    scanned: int           # Tiefensuche-Einträge seit dem vorigen akzeptierten Weg (einschließlich gescheiterter Startfahrzeuge)
    dead: tuple            # in diesem Abschnitt als Sackgasse markierte Fahrzeuge


@dataclass(frozen=True)
class Phase:
    L: int                 # Länge der kürzesten Wege in Aufträgen: der Weg hat 2L+1 Kanten (L = Schicht, in der ein freier Auftrag anliegt)
    length: int            # 2L+1
    bfs_scanned: int
    dfs_scanned: int       # gesamt: Summe der Wege plus tail_scanned
    tail_scanned: int      # Tiefensuche-Einträge nach dem letzten akzeptierten Weg
    tail_dead: tuple
    paths: tuple           # akzeptierte Wege (Path), in der Reihenfolge des Findens
    dist: tuple            # Schicht je Fahrzeug (-1: nicht erreicht) - nur bei record=True
    layer_edges: tuple     # ((i, j), ...) Kanten des Schichtgraphen - nur bei record=True
    pairs_before: int
    pairs_after: int

    @property
    def scanned(self):
        return self.bfs_scanned + self.dfs_scanned


@dataclass(frozen=True)
class Result:
    start: tuple
    pairs: tuple
    cost: int
    phases: tuple
    final_scanned: int     # letzte, erfolglose Breitensuche (der Beweis); 0, wenn kein freies Fahrzeug übrig ist
    states: tuple          # Zustand vor Phase 1, nach Phase 1, ...; leer bei record=False
    cover_v: tuple
    cover_o: tuple

    @property
    def count(self):
        return len(self.pairs)

    @property
    def n_paths(self):
        return sum(len(p.paths) for p in self.phases)

    @property
    def bfs_total(self):
        return sum(p.bfs_scanned for p in self.phases) + self.final_scanned

    @property
    def dfs_total(self):
        return sum(p.dfs_scanned for p in self.phases)

    @property
    def scanned_total(self):
        return self.bfs_total + self.dfs_total


def start_pairs(sc, start):
    """Startpaare: leer oder das Ergebnis einer Greedy-Regel."""
    return () if start == START_EMPTY else run_rule(sc, start).pairs


def _bfs(adj, match_v, match_o, n, early):
    """Schichten der Fahrzeuge: alle freien Fahrzeuge in Schicht 0. Rückgabe (dist, L oder None, durchsuchte Einträge)."""
    dist = [-1] * n
    queue = [i for i in range(n) if match_v[i] < 0]
    for i in queue:
        dist[i] = 0
    L, scanned, qi = None, 0, 0
    while qi < len(queue):
        u = queue[qi]
        qi += 1
        if L is not None and dist[u] > L:
            break
        stop = False
        for j in adj[u]:
            scanned += 1
            w = match_o[j]
            if w < 0:
                if L is None:
                    L = dist[u]
                if early:
                    stop = True
                    break
            elif dist[w] < 0 and L is None:
                dist[w] = dist[u] + 1
                queue.append(w)
        if stop:
            break
    return dist, L, scanned


def _layer_edges(adj, match_o, dist, L):
    edges = []
    for i, d in enumerate(dist):
        if d < 0 or d > L:
            continue
        for j in adj[i]:
            w = match_o[j]
            if (w < 0 and d == L) or (w >= 0 and d < L and dist[w] == d + 1):
                edges.append((i, j))
    return tuple(edges)


def _dfs_phase(adj, match_v, match_o, n, dist, L, prune, layered, maximal):
    """Maximale Menge knotendisjunkter Wege der Länge 2L+1 im Schichtgraphen (auf dem eingefrorenen Matching). Rückgabe (Wege, Tail-Einträge, Tail-Sackgassen)."""
    blocked, dead, used_o = [False] * n, [False] * n, set()
    paths, pending, pending_dead = [], 0, []
    for s in [i for i in range(n) if match_v[i] < 0]:
        if blocked[s] or dead[s]:
            continue
        sv, sp, so = [s], [0], []
        seen_o = set()                                           # nur bei layered=False: jeder Auftrag höchstens einmal je Start
        found = False
        while sv:
            u, pos = sv[-1], sp[-1]
            if pos >= len(adj[u]):
                if prune:
                    dead[u] = True
                    pending_dead.append(u)
                sv.pop()
                sp.pop()
                if so:
                    so.pop()
                continue
            sp[-1] += 1
            j = adj[u][pos]
            pending += 1
            w = match_o[j]
            if w < 0:
                if j in used_o or (layered and dist[u] != L):
                    continue
                chain_v, chain_o = tuple(sv), tuple(so + [j])
                for v in chain_v:
                    blocked[v] = True
                used_o.add(j)
                paths.append(Path(chain_v, chain_o, _path_from_orders(chain_v, chain_o), 2 * len(chain_v) - 1, pending, tuple(pending_dead)))
                pending, pending_dead, found = 0, [], True
                break
            if blocked[w] or dead[w]:
                continue
            if layered:
                if dist[u] >= L or dist[w] != dist[u] + 1:
                    continue
            else:
                if j in seen_o:
                    continue
                seen_o.add(j)
            so.append(j)
            sv.append(w)
            sp.append(0)
        if found and not maximal:
            break
    return paths, pending, tuple(pending_dead)


def hopcroft_karp(sc, pairs=(), *, record=True, early=True, prune=True, layered=True, maximal=True, max_phases=None):
    """Größtmögliches Matching aus `pairs` (leer oder Greedy) in Phasen; siehe Modulkopf. `max_phases` bricht früh ab (nur Negativkontrollen)."""
    n, m = sc.n, sc.m
    adj = _adjacency(sc)
    match_v, match_o = [-1] * n, [-1] * m
    for i, j in pairs:
        assert sc.feasible[i, j] and match_v[i] < 0 and match_o[j] < 0, "Startpaare sind kein Matching"
        match_v[i], match_o[j] = j, i
    start = _snapshot(match_v)
    states = [start] if record else []
    phases = []
    final_scanned, dist_last = 0, None
    while True:
        if max_phases is not None and len(phases) >= max_phases:
            break
        before = sum(1 for j in match_v if j >= 0)
        dist, L, bscan = _bfs(adj, match_v, match_o, n, early)
        if L is None:
            final_scanned, dist_last = bscan, dist
            break
        paths, tail, tail_dead = _dfs_phase(adj, match_v, match_o, n, dist, L, prune, layered, maximal)
        if not paths:
            raise RuntimeError("Phase ohne Weg trotz erreichbarem freien Auftrag")
        edges = _layer_edges(adj, match_o, dist, L) if record else ()
        for p in paths:
            _apply(p.path, match_v, match_o)
        after = sum(1 for j in match_v if j >= 0)
        phases.append(Phase(L, 2 * L + 1, bscan, sum(p.scanned for p in paths) + tail, tail, tail_dead, tuple(paths), tuple(dist) if record else (), edges, before, after))
        if record:
            states.append(_snapshot(match_v))
    final = _snapshot(match_v)
    cost = int(sum(sc.cost[i, j] for i, j in final))
    if dist_last is None:
        cover_v, cover_o = (), ()
    else:
        reach_v = {i for i in range(n) if dist_last[i] >= 0}
        reach_o = {j for i in reach_v for j in adj[i]}
        cover_v, cover_o = tuple(i for i in range(n) if i not in reach_v), tuple(sorted(reach_o))
    return Result(start, final, cost, tuple(phases), final_scanned, tuple(states), cover_v, cover_o)


def frames(res):
    """Bildfolge der Wiedergabe: je Phase Stufe 0 (Schichtgraph), dann ein Bild je akzeptiertem Weg; am Ende das Ergebnis mit dem Beweis.
    Ein Bild ist (Phase, Stufe); die Endstufe hat Phase = Zahl der Phasen und Stufe 0."""
    out = []
    for k, ph in enumerate(res.phases):
        out.append((k, 0))
        out.extend((k, t) for t in range(1, len(ph.paths) + 1))
    out.append((len(res.phases), 0))
    return out
