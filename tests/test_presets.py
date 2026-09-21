"""Presets: vollständig, in den Grenzen, und jede Beispielkarte zeigt, was ihr Hilfetext behauptet."""

import numpy as np
import pytest

import hk_constants as C
import hk_evaluation as ev
import hk_presets as P
from hk_algorithm import hopcroft_karp, start_pairs
from hk_scenario import build

KEYS = set(P.PRESET_KEYS)


def _sc(p):
    return build(p["net"], p["n"], p["m"], p["reach"], p["ballung"], p["seed"])


def _run(p):
    sc = _sc(p)
    return sc, hopcroft_karp(sc, start_pairs(sc, p["start"]))


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_values_are_inside_the_bounds_and_on_the_step_grid(name):
    p = C.PRESETS[name]
    assert p["net"] in C.NETS and p["start"] in C.START_LABELS
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        if spec.lo is not None:
            assert spec.lo <= p[key] <= spec.hi, (name, key)
    assert (p["reach"] - C.REACH_MIN) % 5 == 0 and p["ballung"] % 25 == 0


def test_setting_specs_have_room_to_move():
    """Ein Regler mit lo == hi würde Streamlit abstürzen lassen."""
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS, name


def test_the_mid_reach_map_is_the_map_of_the_previous_pieces_and_the_empty_start_shares_it():
    mid, empty = C.PRESETS["🗺️ Mittlere Reichweite"], C.PRESETS["🕳️ Von leer starten"]
    assert (mid["n"], mid["m"], mid["reach"], mid["ballung"], mid["seed"], mid["start"]) == (20, 20, 40, 0, 165, "edge") == (C.DEFAULT_N, C.DEFAULT_M, C.DEFAULT_REACH, C.DEFAULT_BALLUNG, C.DEFAULT_SEED, C.DEFAULT_START)
    assert all(empty[k] == mid[k] for k in ("net", "n", "m", "reach", "ballung", "seed")) and empty["start"] == "empty"


def test_every_preset_reaches_the_maximum():
    for name, p in C.PRESETS.items():
        level, code, d = ev.verdict(ev.analyse(_sc(p), p["start"]))
        assert (level, code) == ("success", ev.OPTIMAL) and d["same_as_single"], name


def test_the_preset_maps_show_what_their_help_says():
    by = {name: _run(p)[1] for name, p in C.PRESETS.items()}
    assert len(by["⚖️ Nichts zu verbessern"].phases) == 0
    paths = by["🔗 Drei Pfade in einer Phase"]
    assert len(paths.phases) == 1 and len(paths.phases[0].paths) == 3
    chain = by["⛓️ Lange Kette"]
    assert [(ph.L, len(ph.paths)) for ph in chain.phases] == [(6, 1)]
    stair = by["🪜 Treppe: Worst Case"]
    assert [ph.length for ph in stair.phases] == [3, 5, 7, 9, 11] and stair.scanned_total == 150
    mid = by["🗺️ Mittlere Reichweite"]
    assert [ph.L for ph in mid.phases] == [1, 2] and [len(ph.paths) for ph in mid.phases] == [2, 1]
    empty = by["🕳️ Von leer starten"]
    assert [ph.L for ph in empty.phases] == [0, 1, 2] and len(empty.phases[0].paths) == 18
    big = by["🧮 Große Karte"]
    assert len(big.phases) == 1 and len(big.phases[0].paths) == 2
    short = by["📡 Knappe Reichweite"]
    assert len(short.phases) == 1 and len(short.phases[0].paths) == 1


def test_the_big_map_preset_is_one_where_hopcroft_karp_wins_and_a_typical_draw():
    """Karte mit Hopcroft–Karp vor beiden Vergleichen (und in der Nähe der Mediane der 100 festen Karten: 246 / 322 / 552)."""
    p = C.PRESETS["🧮 Große Karte"]
    c = ev.compare_table(ev.analyse(_sc(p), p["start"]))
    hk, single, kuhn = c[0]["scanned"], c[1]["scanned"], c[2]["scanned"]
    assert hk < single < kuhn
    dist = ev.distribution(40, 40, 60, 0, "edge")
    assert abs(hk - dist["hk_median"]) <= 0.3 * dist["hk_median"] and abs(single - dist["single_median"]) <= 0.15 * dist["single_median"] and abs(kuhn - dist["kuhn_median"]) <= 0.15 * dist["kuhn_median"]


def test_fixed_presets_hide_the_random_controls():
    assert {n for n, p in C.PRESETS.items() if p["net"] in C.FIXED_NETS} == {"⚖️ Nichts zu verbessern", "🔗 Drei Pfade in einer Phase", "⛓️ Lange Kette", "🪜 Treppe: Worst Case"}
