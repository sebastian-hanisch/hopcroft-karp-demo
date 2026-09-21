"""Auswertung: Einordnung, Verdict, Vergleichstabelle, Verteilung über viele Karten, Aufwandstabelle, Sweeps, Aufwand-Experiment, Treppe."""

import dataclasses

import pytest

import hk_constants as C
import hk_evaluation as ev
from hk_scenario import build, generate


def test_build_fixed_nets_ignore_random_parameters():
    a, b = build("chain", 5, 5, 99, 100, 123), build("chain", 30, 30, 10, 0, 1)
    assert a.vehicles == b.vehicles and a.n == 7
    assert build("steal", 1, 1, 1, 1, 1).n == 2 and build("p4", 1, 1, 1, 1, 1).n == 2 and build("paths", 1, 1, 1, 1, 1).n == 6 and build("stair", 1, 1, 1, 1, 1).n == 20


@pytest.mark.parametrize("net,count,phases,paths", [("steal", 2, 0, 0), ("p4", 2, 1, 1), ("paths", 6, 1, 3), ("chain", 7, 1, 1), ("stair", 20, 5, 5)])
def test_verdict_of_the_fixed_nets(net, count, phases, paths):
    level, code, d = ev.verdict(ev.analyse(build(net, 20, 20, 40, 0, 2), "edge"))
    assert (level, code) == ("success", ev.OPTIMAL) and (d["count"], d["phases"], d["paths"]) == (count, phases, paths)
    assert d["same_as_single"] and d["scanned"] == d["bfs"] + d["dfs"]


def test_verdict_mismatch_and_none():
    a = ev.analyse(generate(20, 20, 40, 0, 165))
    broken = dataclasses.replace(a, opt=dataclasses.replace(a.opt, pairs=a.opt.pairs + ((0, 0),)))
    assert ev.verdict(broken)[1] == ev.MISMATCH
    empty = next(s for s in (generate(3, 3, 10, 0, k) for k in range(200)) if not s.feasible.any())
    assert ev.verdict(ev.analyse(empty))[:2] == ("info", ev.NONE)


def test_compare_table_rows():
    rows = ev.compare_table(ev.analyse(generate(20, 20, 40, 0, 165)))
    assert [r["label"] for r in rows][0] == "Hopcroft–Karp (Phasen)" and len(rows) == 3
    assert rows[0]["searches"] == 2 and rows[0]["paths"] == 3 and rows[0]["scanned"] == 171 == rows[0]["bfs"] + rows[0]["dfs"]
    assert (rows[1]["scanned"], rows[2]["scanned"]) == (144, 129) and rows[0]["cost"] == rows[1]["cost"]


def test_distribution_fields():
    d = ev.distribution(20, 20, 40, 0)
    assert d["n_seeds"] == 100 == d["n_valid"] and d["share_maximal"] == 1.0 and len(d["hk_scans"]) == len(d["single_scans"]) == len(d["kuhn_scans"]) == 100
    assert sum(d["phase_hist"].values()) == 100 and d["hk_mean"] == pytest.approx(d["bfs_mean"] + d["dfs_mean"])


def test_distribution_is_repeatable():
    assert ev.distribution(15, 15, 40, 25) == ev.distribution(15, 15, 40, 25)


def test_distribution_without_feasible_pairs():
    bad = tuple(s for s in range(60) if not generate(3, 3, 10, 0, s).feasible.any())
    assert len(bad) >= 3
    assert ev.distribution(3, 3, 10, 0, seeds=bad)["n_valid"] == 0 and ev.effort_table(3, 3, 10, 0, seeds=bad) == []


def test_effort_table_rows():
    rows = ev.effort_table(20, 20, 40, 0)
    assert [r["label"] for r in rows] == ["Hopcroft–Karp (Phasen)", "Einzelne kürzeste Wege", "Kuhn (Tiefensuche)"] and rows[2]["searches"] is None


def test_cell_rows_are_integers():
    rows = ev.cell_rows(10, 10, 40, 0, "edge", tuple(C.SWEEP_SEEDS[:5]))
    for row in rows:
        for key, val in row.items():
            if key == "same":
                assert isinstance(val, bool)
                continue
            assert all(isinstance(v, int) for v in (val if isinstance(val, tuple) else (val,))), key


def test_reach_sweep_scaling_and_staircase_rows():
    sweep = ev.reach_sweep(15, 15, 0, values=(10, 40, 150), seeds=C.SWEEP_SEEDS[:10])
    assert [r["x"] for r in sweep] == [10, 40, 150] and sweep[-1]["phases"] == 0
    rows = ev.scaling("edge", ns=(10, 20), seeds=C.SCALE_SEEDS[:3])
    assert [r["n"] for r in rows] == [10, 20] and [r["reach"] for r in rows] == [57, 40] and all("no_prune" not in r for r in rows)
    var = ev.scaling("edge", ns=(10, 20), seeds=C.SCALE_SEEDS[:3], variants=True)
    assert all(r["no_prune"] >= r["hk"] and r["full_layer"] >= r["hk"] for r in var)
    stair = ev.staircase_scaling("edge", ks=(2, 3))
    assert [(r["k"], r["n"], r["phases"]) for r in stair] == [(2, 5, 2), (3, 9, 3)]


def test_slopes_and_crossover_helpers():
    rows = [{"n": n, "hk": n ** 1.5, "single": n ** 2.0, "kuhn": n ** 2.0, "phases": n ** 0.5, "edges": n} for n in (10, 100, 1000)]
    s = ev.slopes(rows)
    assert s["hk"] == pytest.approx(1.5) and s["single"] == pytest.approx(2.0) and s["phases"] == pytest.approx(0.5)
    two = [{"n": 10, "hk": 5, "single": 4}, {"n": 20, "hk": 6, "single": 9}, {"n": 40, "hk": 7, "single": 10}]
    assert ev.crossover(two, "single") == 20
    assert ev.crossover([{"n": 10, "hk": 5, "single": 4}], "single") is None
