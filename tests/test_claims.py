"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App und der README ist hier über die 100 festen Karten (DIST_SEEDS) belegt.
Alles rechnet mit ganzen Zahlen und einem eigenen Zufallsgenerator - die Werte sind auf jeder Plattform dieselben; die Toleranzen decken nur die Rundung auf die im Text genannten Stellen.
Positive UND negative Aussagen: wo das Bündeln gewinnt (große Karten, gegen Kuhn vom leeren Start) und wo es verliert (20 x 20, die Treppe). Mittel und Median stehen zusammen."""

import numpy as np
import pytest

import hk_constants as C
import hk_evaluation as ev
from hk_algorithm import hopcroft_karp, start_pairs
from hk_scenario import generate, staircase


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


@pytest.fixture(scope="module")
def mid():
    return ev.distribution(20, 20, 40, 0, "edge")


@pytest.fixture(scope="module")
def mid_empty():
    return ev.distribution(20, 20, 40, 0, "empty")


# --- Standardkarte (20 x 20, Reichweite 40) ------------------------------------------------------------------------------------------------

def test_hopcroft_karp_reaches_the_maximum_and_finds_the_paths_of_piece_2_on_every_map_and_start():
    for cfg in ((20, 20, 40, 0), (20, 20, 10, 0), (20, 20, 150, 0), (20, 10, 40, 0), (10, 20, 40, 0), (40, 40, 60, 0), (20, 20, 40, 50), (20, 20, 40, 100)):
        for start in ("edge", "empty"):
            d = ev.distribution(*cfg, start)
            assert d["share_maximal"] == 1.0 and d["same_share"] == 1.0 and d["cost_same_as_single"] == 1.0, (cfg, start)


def test_mid_reach_from_greedy_numbers(mid):
    near(mid["phases_mean"], 1.9, 0.05)                                                             # "im Mittel 1,9 Phasen"
    assert mid["phases_median"] == 2 and mid["phases_max"] == 3 and mid["phase_hist"] == {0: 1, 1: 23, 2: 63, 3: 13}
    near(mid["paths_mean"], 2.7, 0.05)                                                              # "für 2,7 Wege"
    near(mid["opt_pairs_mean"], 19.5, 0.05)
    assert round(mid["hk_mean"]) == 197 and mid["hk_median"] == 180                                 # "197 Kanten (Median 180)"
    assert round(mid["single_mean"]) == 140 and round(mid["kuhn_mean"]) == 121                      # "gegen 140 und 121"
    near(mid["bfs_mean"], 110, 0.6)
    near(mid["dfs_mean"], 86, 0.6)
    assert mid["hk_mean"] > mid["single_mean"] > mid["kuhn_mean"]                                   # negativ: bei 20 x 20 verliert Hopcroft–Karp gegen beide
    near(mid["hk_less_single"], 0.08, 0.001)                                                        # "auf nur 8 von 100 Karten weniger als die einzelnen Wege"
    near(mid["hk_less_kuhn"], 0.19, 0.001)


def test_cost_gap_is_the_one_of_piece_2(mid, mid_empty):
    near(mid["gap_median"], 7.9, 0.05)                                                              # "im Median 8 % (ab Greedy)"
    near(mid_empty["gap_median"], 36.2, 0.05)                                                       # "bzw. 36 % (vom leeren Start)"


def test_mid_reach_from_empty_numbers(mid_empty):
    near(mid_empty["paths_mean"], 19.5, 0.05)                                                       # "19,5 Wege"
    near(mid_empty["phases_mean"], 2.4, 0.05)                                                       # "in 2,4 Phasen"
    assert mid_empty["phases_max"] == 4
    assert round(mid_empty["hk_mean"]) == 209 and round(mid_empty["single_mean"]) == 174 and round(mid_empty["kuhn_mean"]) == 384      # "209 / 174 / 384"
    near(mid_empty["hk_less_kuhn"], 0.96, 0.001)                                                    # "gegen Kuhn auf 96 von 100 Karten"
    near(mid_empty["hk_less_single"], 0.19, 0.001)                                                  # gegen die einzelnen Wege nur auf 19 von 100 (negativ)
    assert mid_empty["hk_mean"] > mid_empty["single_mean"]


def test_the_default_map_and_the_empty_start_preset():
    sc = generate(20, 20, 40, 0, 165)
    a = hopcroft_karp(sc, start_pairs(sc, "edge"))
    assert [ph.L for ph in a.phases] == [1, 2] and [len(ph.paths) for ph in a.phases] == [2, 1] and a.scanned_total == 171
    b = hopcroft_karp(sc, ())
    assert [len(ph.paths) for ph in b.phases] == [18, 1, 1] and b.scanned_total == 208


# --- Reichweite und Größe --------------------------------------------------------------------------------------------------------------------

def test_short_and_long_reach():
    d = ev.distribution(20, 20, 10, 0, "edge")
    near(d["phases_mean"], 0.2, 0.001)                                                              # "0,2 Phasen"
    near(d["zero_phase_share"], 0.80, 0.001)                                                        # "auf 80 von 100 Karten keine Phase"
    assert d["phase_hist"] == {0: 80, 1: 20}                                                        # "20 mit einer"
    assert round(d["hk_mean"]) == 4 and round(d["single_mean"]) == 3 and d["hk_mean"] > d["single_mean"]
    far = ev.distribution(20, 20, 150, 0, "edge")
    assert far["phases_mean"] == 0 and far["zero_phase_share"] == 1.0                               # "ab 150 keine: Greedy hat schon alles"
    far_empty = ev.distribution(20, 20, 150, 0, "empty")
    assert far_empty["phase_hist"] == {1: 100} and far_empty["paths_mean"] == 20 and round(far_empty["hk_mean"]) == 211 and round(far_empty["single_mean"]) == 210 and round(far_empty["kuhn_mean"]) == 1540


def test_big_map_hopcroft_karp_wins_on_average():
    d = ev.distribution(40, 40, 60, 0, "edge")
    assert round(d["hk_mean"]) == 326 and d["hk_median"] == 246.5 and f"{d['hk_median']:.0f}" == "246"     # "326 (Median 246)" (die App rundet 246,5 auf 246)
    assert round(d["single_mean"]) == 393 and f"{d['single_median']:.0f}" == "322"                # "393 (322)" (321,5 wird als 322 angezeigt)
    assert round(d["kuhn_mean"]) == 583 and f"{d['kuhn_median']:.0f}" == "552"                   # "583 (552)"
    near(d["hk_less_single"], 0.61, 0.001)                                                          # "auf 61 von 100 Karten weniger als die einzelnen Wege"
    near(d["phases_mean"], 1.1, 0.05)
    assert d["hk_mean"] < d["single_mean"] < d["kuhn_mean"]


def test_reach_sweep_20x20_hopcroft_karp_never_below_the_single_paths_on_average():
    rows = ev.reach_sweep(20, 20, 0, "edge")
    assert all(r["hk"] >= r["single"] for r in rows)                                                # "bei 20 x 20 bei keiner Reichweite im Mittel unter den einzelnen Wegen"
    assert rows[-1]["hk"] == 0 == rows[-1]["phases"]


# --- Aufwand-Experiment ----------------------------------------------------------------------------------------------------------------------

def test_scaling_from_greedy():
    rows = ev.scaling("edge")
    by = {r["n"]: r for r in rows}
    sl = ev.slopes(rows)
    near(sl["hk"], 1.74, 0.02)                                                                       # Steigung Hopcroft–Karp
    near(sl["single"], 2.01, 0.02)
    near(sl["kuhn"], 1.94, 0.02)
    near(sl["phases"], 0.52, 0.02)
    near(sl["edges"], 1.12, 0.02)
    assert sl["hk"] < sl["kuhn"] < sl["single"] and sl["hk"] < sl["single"] - 0.2                   # flacher als die einzelnen Wege und Kuhn
    assert ev.crossover(rows, "single") == 113 and ev.crossover(rows, "kuhn") == 320               # ab 113 Fahrzeugen weniger als die einzelnen Wege; gegen Kuhn erst bei den größten Karten
    assert round(by[320]["hk"]) == 22389 and round(by[320]["single"]) == 39794 and round(by[320]["kuhn"]) == 25880
    assert by[20]["hk"] > by[20]["single"] and by[80]["hk"] > by[80]["single"]                       # unter der Kreuzung liegt Hopcroft–Karp darüber
    assert [round(by[n]["phases"], 1) for n in (10, 40, 160, 320)] == [1.3, 3.1, 5.3, 7.6]           # Phasen wachsen auf Zufallskarten etwa mit der Wurzel
    near(by[320]["hk_less_single"], 1.0, 1e-9)


def test_scaling_from_empty():
    rows = ev.scaling("empty")
    sl = ev.slopes(rows)
    near(sl["hk"], 1.66, 0.02)
    near(sl["single"], 1.90, 0.02)
    near(sl["kuhn"], 1.79, 0.02)
    near(sl["phases"], 0.41, 0.02)
    assert ev.crossover(rows, "kuhn") == C.SCALE_NS[0]                                               # vom leeren Start schlägt Hopcroft–Karp Kuhn bei jeder Größe
    assert ev.crossover(rows, "single") == 113


def test_the_staircase_is_the_worst_case():
    rows = ev.staircase_scaling("edge")
    assert [r["phases"] for r in rows] == list(C.STAIR_KS) and [r["n"] for r in rows] == [k * (k + 3) // 2 for k in C.STAIR_KS]      # genau K Phasen bei n = K (K + 3) / 2
    assert [r["hk"] for r in rows] == [20, 46, 88, 236, 496, 1166, 3296, 9246]
    assert [r["single"] for r in rows] == [9, 20, 38, 103, 220, 528, 1528, 4370] and [r["kuhn"] for r in rows] == [8, 15, 24, 48, 80, 143, 288, 575]
    assert all(2.0 < r["hk"] / r["single"] < 2.4 for r in rows if r["k"] >= 4)                       # "etwa doppelt so viele wie die einzelnen Wege"
    assert all(r["hk"] > r["single"] > r["kuhn"] for r in rows)                                      # hier verliert das Bündeln immer (negativ)
    assert staircase(5).n == 20 and hopcroft_karp(staircase(5), start_pairs(staircase(5), "edge")).scanned_total == 150


def test_the_random_maps_stay_below_the_root_bound_and_the_staircase_approaches_it():
    """Auf Zufallskarten bleiben die Phasen unter sqrt(2n); die Treppe liegt knapp darunter und nähert sich der Grenze (K / sqrt(2n) = 1 / sqrt(1 + 3 / K))."""
    rows = ev.scaling("edge")
    assert all(r["phases"] < (2 * r["n"]) ** 0.5 for r in rows)
    st = ev.staircase_scaling("edge")
    assert all(r["phases"] < (2 * r["n"]) ** 0.5 for r in st) and all(r["phases"] / (2 * r["n"]) ** 0.5 > 0.88 for r in st if r["k"] >= 16)


def test_dead_end_marking_and_early_stop_save_scans_on_large_maps():
    rows = {r["n"]: r for r in ev.scaling("edge", variants=True)}
    assert rows[226]["no_prune"] / rows[226]["hk"] > 2 and rows[320]["no_prune"] / rows[320]["hk"] > 2       # "bei großen Karten mehr als das Doppelte"
    assert all(r["no_prune"] >= r["hk"] and r["full_layer"] >= r["hk"] for r in rows.values())
