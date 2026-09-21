"""Auswertung: eine Karte (`analyse`, `verdict`, `compare_table`), viele Karten (`distribution`, `effort_table`), Sweeps, Aufwand-Experiment, Treppe.

Alle Größen kommen aus ganzen Zahlen und deterministischen Verfahren; nur die Anzeige-Statistiken (Anteile, Mittel, Mediane) sind Gleitkomma.
Aufwand wird in geprüften Adjazenzeinträgen gezählt (wie in der Verbesserungswege-Demo), nie in Sekunden. Hopcroft-Karp zählt Breitensuche und
Tiefensuche getrennt; die Vergleichsverfahren sind die einzelnen kürzesten Wege (Breitensuche je Weg) und Kuhn (Tiefensuche) aus Stück 2.
Verteilungen werden mit Mittel UND Median genannt.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import hk_constants as C
from hk_algorithm import hopcroft_karp, start_pairs
from hk_augment import augment
from hk_greedy import optimum, run_rule
from hk_scenario import build, generate, staircase

OPTIMAL, MISMATCH, NONE = "maximal", "mismatch", "none"


@dataclass
class Analysis:
    scenario: object
    start: str
    result: object       # Hopcroft-Karp mit Protokoll
    single: object       # einzelne kürzeste Wege (Breitensuche je Weg), Stück 2
    kuhn: object         # Kuhn (Tiefensuche), Stück 2
    opt: object          # Referenz für die Paarzahl (Messlatte aus Stück 1)
    greedy: object


def analyse(sc, start=C.DEFAULT_START):
    sp = start_pairs(sc, start)
    return Analysis(sc, start, hopcroft_karp(sc, sp), augment(sc, sp, "bfs", record=False), augment(sc, sp, "dfs", record=False), optimum(sc), run_rule(sc, "edge"))


def cost_gap_pct(cost, opt_cost):
    return None if opt_cost <= 0 else 100.0 * (cost - opt_cost) / opt_cost


def classify(res, opt):
    if opt.count == 0:
        return NONE
    return OPTIMAL if res.count == opt.count else MISMATCH


def _paths(res):
    return [p.path for ph in res.phases for p in ph.paths]


def verdict(a):
    """(Stufe, Code, Zahlen) für die Anzeige; `Zahlen` enthält jede Zahl, die der Text nennt."""
    r, s, k, o = a.result, a.single, a.kuhn, a.opt
    code = classify(r, o)
    data = {"count": r.count, "opt_count": o.count, "start_count": len(r.start), "phases": len(r.phases), "paths": r.n_paths, "Ls": [ph.L for ph in r.phases],
            "paths_per_phase": [len(ph.paths) for ph in r.phases], "bfs": r.bfs_total, "dfs": r.dfs_total, "scanned": r.scanned_total, "final_scanned": r.final_scanned,
            "single_scanned": s.scanned_total, "single_rounds": len(s.rounds), "kuhn_scanned": k.scanned_total, "kuhn_rounds": len(k.rounds), "cost": r.cost, "opt_cost": o.cost,
            "cost_gap_pct": cost_gap_pct(r.cost, o.cost) if r.count == o.count else None, "same_as_single": _paths(r) == [x.path for x in s.rounds] and r.pairs == s.pairs,
            "edges": int(a.scenario.feasible.sum()), "cover": len(r.cover_v) + len(r.cover_o), "greedy_count": a.greedy.count}
    level = {NONE: "info", OPTIMAL: "success", MISMATCH: "error"}[code]
    return level, code, data


def compare_table(a):
    """Zeilen für Hopcroft-Karp und die beiden Suchen aus Stück 2 auf der aktuellen Karte."""
    r, s, k = a.result, a.single, a.kuhn
    return [{"label": "Hopcroft–Karp (Phasen)", "searches": len(r.phases), "paths": r.n_paths, "scanned": r.scanned_total, "bfs": r.bfs_total, "dfs": r.dfs_total, "cost": r.cost},
            {"label": "Einzelne kürzeste Wege (Breitensuche je Weg)", "searches": len(s.rounds), "paths": len(s.rounds), "scanned": s.scanned_total, "bfs": s.scanned_total, "dfs": 0, "cost": s.cost},
            {"label": "Kuhn (Tiefensuche je Fahrzeug)", "searches": len(k.rounds), "paths": len(k.rounds), "scanned": k.scanned_total, "bfs": 0, "dfs": k.scanned_total, "cost": k.cost}]


# --- viele Karten --------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=256)
def cell_rows(n, m, reach, ballung, start, seeds):
    """Je Seed als Ganzzahlen: Optimum, Hopcroft-Karp, einzelne kürzeste Wege, Kuhn, Startgröße, Wegfolge gleich? Rückgabe: Tupel von Dicts."""
    rows = []
    for sd in seeds:
        sc = generate(n, m, reach, ballung, sd)
        sp = start_pairs(sc, start)
        h = hopcroft_karp(sc, sp, record=False)
        s = augment(sc, sp, "bfs", record=True)
        k = augment(sc, sp, "dfs", record=False)
        o = optimum(sc)
        rows.append({"opt": (o.count, o.cost), "edges": int(sc.feasible.sum()), "start": len(sp),
                     "hk": (h.count, h.cost, len(h.phases), h.n_paths, h.bfs_total, h.dfs_total, h.scanned_total),
                     "single": (s.count, s.cost, len(s.rounds), s.scanned_total), "kuhn": (k.count, k.cost, len(k.rounds), k.scanned_total),
                     "same": _paths(h) == [x.path for x in s.rounds] and h.pairs == s.pairs, "Ls": tuple(ph.L for ph in h.phases)})
    return tuple(rows)


def _med(values):
    return float(np.median(values)) if len(values) else None


def distribution(n, m, reach, ballung, start=C.DEFAULT_START, seeds=C.DIST_SEEDS):
    """Verteilung über viele Karten: Paarzahl, Phasen, Wege, Aufwand (Mittel und Median), Anteil der Karten, auf denen Hopcroft-Karp weniger sucht, Kostenlücke."""
    rows = [r for r in cell_rows(n, m, reach, ballung, start, tuple(seeds)) if r["opt"][0] > 0]
    nv = len(rows)
    if nv == 0:
        return {"n_seeds": len(seeds), "n_valid": 0}
    hk = np.array([r["hk"] for r in rows], dtype=float)
    single = np.array([r["single"] for r in rows], dtype=float)
    kuhn = np.array([r["kuhn"] for r in rows], dtype=float)
    phases = [int(r["hk"][2]) for r in rows]
    gaps = [100.0 * (r["hk"][1] - r["opt"][1]) / r["opt"][1] for r in rows if r["opt"][1] > 0 and r["hk"][0] == r["opt"][0]]
    return {"n_seeds": len(seeds), "n_valid": nv, "share_maximal": float(np.mean([r["hk"][0] == r["opt"][0] for r in rows])), "same_share": float(np.mean([r["same"] for r in rows])),
            "edges_mean": float(np.mean([r["edges"] for r in rows])), "start_mean": float(np.mean([r["start"] for r in rows])), "opt_pairs_mean": float(np.mean([r["opt"][0] for r in rows])),
            "phases_mean": float(hk[:, 2].mean()), "phases_median": _med(phases), "phases_max": max(phases), "phase_hist": {p: phases.count(p) for p in sorted(set(phases))},
            "paths_mean": float(hk[:, 3].mean()), "bfs_mean": float(hk[:, 4].mean()), "dfs_mean": float(hk[:, 5].mean()),
            "hk_mean": float(hk[:, 6].mean()), "hk_median": _med(hk[:, 6]), "single_mean": float(single[:, 3].mean()), "single_median": _med(single[:, 3]),
            "kuhn_mean": float(kuhn[:, 3].mean()), "kuhn_median": _med(kuhn[:, 3]), "single_rounds_mean": float(single[:, 2].mean()),
            "hk_less_single": float(np.mean(hk[:, 6] < single[:, 3])), "hk_less_kuhn": float(np.mean(hk[:, 6] < kuhn[:, 3])),
            "zero_phase_share": float(np.mean([p == 0 for p in phases])), "hk_scans": [int(x) for x in hk[:, 6]], "single_scans": [int(x) for x in single[:, 3]], "kuhn_scans": [int(x) for x in kuhn[:, 3]],
            "gap_median": _med(gaps), "gap_mean": float(np.mean(gaps)) if gaps else None, "cost_same_as_single": float(np.mean([r["hk"][1] == r["single"][1] for r in rows]))}


def effort_table(n, m, reach, ballung, start=C.DEFAULT_START, seeds=C.DIST_SEEDS):
    """Zeilen für Hopcroft-Karp, einzelne kürzeste Wege und Kuhn: Suchen, Wege, durchsuchte Kanten (Mittel und Median), Anteil größtmöglich."""
    d = distribution(n, m, reach, ballung, start, seeds)
    if d["n_valid"] == 0:
        return []
    return [{"label": "Hopcroft–Karp (Phasen)", "searches": d["phases_mean"], "paths": d["paths_mean"], "mean": d["hk_mean"], "median": d["hk_median"]},
            {"label": "Einzelne kürzeste Wege", "searches": d["single_rounds_mean"], "paths": d["single_rounds_mean"], "mean": d["single_mean"], "median": d["single_median"]},
            {"label": "Kuhn (Tiefensuche)", "searches": None, "paths": d["single_rounds_mean"], "mean": d["kuhn_mean"], "median": d["kuhn_median"]}]


def reach_sweep(n, m, ballung, start=C.DEFAULT_START, values=C.REACH_SWEEP, seeds=C.SWEEP_SEEDS):
    """Je Reichweite: Kantensuchen der drei Verfahren und mittlere Phasenzahl."""
    out = []
    for reach in values:
        d = distribution(n, m, reach, ballung, start, seeds)
        if d["n_valid"] == 0:
            out.append({"x": reach, "hk": None, "single": None, "kuhn": None, "phases": None})
            continue
        out.append({"x": reach, "hk": d["hk_mean"], "single": d["single_mean"], "kuhn": d["kuhn_mean"], "phases": d["phases_mean"]})
    return out


def _reach_for(n):
    return max(5, round(C.SCALE_REACH_AT_20 * math.sqrt(20 / n)))


@lru_cache(maxsize=16)
def scaling(start=C.DEFAULT_START, ns=C.SCALE_NS, seeds=C.SCALE_SEEDS, variants=False):
    """Aufwand wächst mit der Karte: n = m bei konstantem mittleren Grad; Kantensuchen von Hopcroft-Karp, einzelnen kürzesten Wegen und Kuhn, mittlere Phasenzahl.
    variants=True zählt zusätzlich Hopcroft-Karp ohne Sackgassen-Markierung und mit voller Schicht (gleiche Wege, mehr Kanten)."""
    out = []
    for n in ns:
        reach = _reach_for(n)
        acc = []
        for sd in seeds:
            sc = generate(n, n, reach, 0, sd)
            sp = start_pairs(sc, start)
            h = hopcroft_karp(sc, sp, record=False)
            row = [int(sc.feasible.sum()), len(h.phases), h.n_paths, h.scanned_total, augment(sc, sp, "bfs", record=False).scanned_total, augment(sc, sp, "dfs", record=False).scanned_total]
            if variants:
                row += [hopcroft_karp(sc, sp, record=False, prune=False).scanned_total, hopcroft_karp(sc, sp, record=False, early=False).scanned_total]
            acc.append(row)
        arr = np.array(acc, dtype=float)
        mean = arr.mean(axis=0)
        r = {"n": n, "reach": reach, "edges": float(mean[0]), "phases": float(mean[1]), "paths": float(mean[2]), "hk": float(mean[3]), "single": float(mean[4]), "kuhn": float(mean[5]),
             "hk_less_single": float(np.mean(arr[:, 3] < arr[:, 4])), "hk_less_kuhn": float(np.mean(arr[:, 3] < arr[:, 5]))}
        if variants:
            r.update({"no_prune": float(mean[6]), "full_layer": float(mean[7])})
        out.append(r)
    return tuple(out)


def slopes(rows):
    """Steigungen im doppelt logarithmischen Bild (Kantensuchen gegen n) und der Phasenzahl gegen n."""
    ns = np.log([r["n"] for r in rows])
    f = lambda key: float(np.polyfit(ns, np.log([r[key] for r in rows]), 1)[0])
    return {"hk": f("hk"), "single": f("single"), "kuhn": f("kuhn"), "phases": f("phases"), "edges": f("edges")}


def crossover(rows, key="single"):
    """Kleinstes n, ab dem Hopcroft-Karp im Mittel weniger sucht als `key` und dort bleibt (None: nie in diesem Gitter)."""
    out = None
    for r in reversed(rows):
        if r["hk"] < r[key]:
            out = r["n"]
        else:
            break
    return out


@lru_cache(maxsize=8)
def staircase_scaling(start=C.DEFAULT_START, ks=C.STAIR_KS):
    """Worst-Case-Treppe: Phasen = K, Kantensuchen der drei Verfahren gegen die Kartengröße n = K (K + 3) / 2."""
    out = []
    for k in ks:
        sc = staircase(k)
        sp = start_pairs(sc, start)
        h = hopcroft_karp(sc, sp, record=False)
        out.append({"k": k, "n": sc.n, "phases": len(h.phases), "hk": h.scanned_total, "single": augment(sc, sp, "bfs", record=False).scanned_total,
                    "kuhn": augment(sc, sp, "dfs", record=False).scanned_total})
    return tuple(out)


def scenario_from_settings(net, n, m, reach, ballung, seed):
    return build(net, n, m, reach, ballung, seed)
