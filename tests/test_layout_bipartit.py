"""Darstellung "Zwei Seiten": Fahrzeuge links, Aufträge rechts. Nur Koordinaten ändern sich; Kosten, Paare, Schichten und Zahlen bleiben die der Karte.
Das Schichtschema (build_layer_schematic) hat eigene Spaltenkoordinaten und bleibt in beiden Darstellungen unverändert."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import hk_constants as C
import hk_evaluation as ev
from hk_scenario import build
from hk_visualization import _two_sided_order, arrange, build_cover_map, build_layer_map

APP = Path(__file__).resolve().parent.parent / "app.py"
CASES = [("random", 20, 20, 40, 0, 165), ("random", 8, 8, 25, 50, 7), ("random", 5, 12, 40, 0, 3), ("steal", 20, 20, 40, 0, 165), ("chain", 20, 20, 40, 0, 165)]


@pytest.mark.parametrize("params", CASES)
def test_arrange_puts_each_side_in_its_own_column_with_distinct_places(params):
    sc = build(*params)
    arr = arrange(sc, "bipartit")
    assert {x for x, _ in arr.vehicles} == {0} and {x for x, _ in arr.orders} == {100}
    assert len(set(arr.vehicles)) == sc.n and len(set(arr.orders)) == sc.m
    assert all(0 <= y <= 100 for _, y in arr.vehicles + arr.orders)


@pytest.mark.parametrize("params", CASES)
def test_arrange_changes_only_the_coordinates(params):
    sc = build(*params)
    arr = arrange(sc, "bipartit")
    assert np.array_equal(arr.cost, sc.cost) and np.array_equal(arr.feasible, sc.feasible) and arr.reach == sc.reach
    assert arr.n == sc.n and arr.m == sc.m


def test_arrange_map_is_the_identity_and_is_repeatable():
    sc = build("random", 20, 20, 40, 0, 165)
    assert arrange(sc, "map") is sc
    first, second = arrange(sc, "bipartit"), arrange(sc, "bipartit")
    assert first.vehicles == second.vehicles and first.orders == second.orders


def test_every_edge_of_the_two_sided_drawing_joins_the_two_columns():
    """Das ist die Aussage "bipartit" im Bild: jede gezeichnete mögliche Kante beginnt links (Fahrzeug) und endet rechts (Auftrag)."""
    sc = build("random", 12, 12, 40, 0, 165)
    arr = arrange(sc, "bipartit")
    for i in range(sc.n):
        for j in range(sc.m):
            if sc.feasible[i, j]:
                assert arr.vehicles[i][0] < arr.orders[j][0]


def test_barycenter_order_does_not_cross_more_than_the_map_order():
    """Die Anordnung nach den Plätzen der Partner kreuzt nicht mehr als die reine Sortierung nach Kartenlage."""
    def crossings(sc, arr):
        es = [(arr.vehicles[i][1], arr.orders[j][1]) for i in range(sc.n) for j in range(sc.m) if sc.feasible[i, j]]
        return sum(1 for a in range(len(es)) for b in range(a + 1, len(es)) if (es[a][0] - es[b][0]) * (es[a][1] - es[b][1]) < 0)

    better = 0
    for seed in range(100000, 100008):
        sc = build("random", 12, 12, 30, 0, seed)
        vo, oo = _two_sided_order(sc, sweeps=0)
        v0 = {i: 100 - 100 * r / (sc.n - 1) for r, i in enumerate(vo)}
        o0 = {j: 100 - 100 * r / (sc.m - 1) for r, j in enumerate(oo)}
        plain = replace(sc, vehicles=tuple((0, v0[i]) for i in range(sc.n)), orders=tuple((100, o0[j]) for j in range(sc.m)))
        better += crossings(sc, arrange(sc, "bipartit")) <= crossings(sc, plain)
    assert better == 8


@pytest.mark.parametrize("layout", list(C.LAYOUT_LABELS))
@pytest.mark.parametrize("params", CASES)
def test_both_map_figures_build_in_both_layouts(params, layout):
    a = ev.analyse(build(*params), "empty")
    sc, res = a.scenario, a.result
    figs = [build_cover_map(sc, res.pairs, res.cover_v, res.cover_o, layout=layout)]
    if res.phases:
        ph = res.phases[0]
        figs += [build_layer_map(sc, res.states[0], ph, 0, layout=layout), build_layer_map(sc, res.states[0], ph, len(ph.paths), layout=layout)]
    for fig in figs:
        assert len(fig.data) >= 3
        headers = {x.text for x in fig.layout.annotations if x.text and "<b>" in x.text}
        assert headers == ({"<b>Fahrzeuge</b>", "<b>Aufträge</b>"} if layout == "bipartit" else set())


def test_layer_map_has_the_same_markers_in_both_layouts():
    a = ev.analyse(build("random", 8, 8, 40, 0, 165), "empty")
    sc, res = a.scenario, a.result
    ph = res.phases[0]
    rows = {}
    for layout in C.LAYOUT_LABELS:
        fig = build_layer_map(sc, res.states[0], ph, 0, layout=layout)
        rows[layout] = sorted(h for t in fig.data if t.hovertext is not None for h in t.hovertext)
    assert rows["map"] == rows["bipartit"] and rows["map"]


@pytest.mark.parametrize("start", list(C.START_LABELS))
def test_app_runs_in_the_two_sided_layout(start):
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["layout"] = "bipartit"
    at.query_params["start"] = start
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="layout_radio").value == "bipartit"
    at.radio(key="layout_radio").set_value("map").run()
    assert not at.exception and at.radio(key="layout_radio").value == "map"


def test_unknown_layout_in_the_permalink_falls_back_to_the_map():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["layout"] = "kreis"
    at.run()
    assert not at.exception and at.radio(key="layout_radio").value == C.DEFAULT_LAYOUT
