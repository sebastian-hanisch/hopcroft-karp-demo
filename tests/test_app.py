"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, beide Starts (auch beim Abspielen auf mehrbildrigen Karten), Randgrößen, Schritt-Zustand,
ausgeblendete Regler, Permalink, Experimente auf Abruf, Schlüssel und Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import hk_constants as C
from hk_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

# Anfang der Meldung zur gezeigten Karte und (bei Zufallskarten) der Meldung zur Verteilung (Streamlit legt das führende Emoji in `icon`, nicht in `value`)
EXPECTED = {
    "⚖️ Nichts zu verbessern": ("Größtmögliche Paarzahl (2): der Start hat sie schon", None),
    "🔗 Drei Pfade in einer Phase": ("Größtmögliche Paarzahl (6) in 1 Phase(n) mit 3 Weg(en)", None),
    "⛓️ Lange Kette": ("Größtmögliche Paarzahl (7) in 1 Phase(n) mit 1 Weg(en)", None),
    "🪜 Treppe: Worst Case": ("Größtmögliche Paarzahl (20) in 5 Phase(n) mit 5 Weg(en)", None),
    "🗺️ Mittlere Reichweite": ("Größtmögliche Paarzahl (20) in 2 Phase(n) mit 3 Weg(en)", "Auf 100 % der 100 Karten"),
    "🕳️ Von leer starten": ("Größtmögliche Paarzahl (20) in 3 Phase(n) mit 20 Weg(en)", "Auf 100 % der 100 Karten"),
    "🧮 Große Karte": ("Größtmögliche Paarzahl (40) in 1 Phase(n) mit 2 Weg(en)", "Auf 100 % der 100 Karten"),
    "📡 Knappe Reichweite": ("Größtmögliche Paarzahl (9) in 1 Phase(n) mit 1 Weg(en)", "Auf 100 % der 100 Karten"),
}


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input) + list(at.sidebar.radio)}


def _texts(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info)]


def _has(at, prefix):
    return any(t.startswith(prefix) for t in _texts(at))


def _step(at):
    found = [s for s in at.slider if s.key == "hk_step"]
    return found[0] if found else None


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_default_renders_without_exception():
    at = _run()
    assert any("Phasen in Aktion" in m.value for m in at.markdown)
    assert _has(at, EXPECTED["🗺️ Mittlere Reichweite"][0]) and not at.error


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdicts(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    top, dist = EXPECTED[name]
    assert _has(at, top), _texts(at)
    if dist is not None:
        assert _has(at, dist), _texts(at)
    else:
        assert any(t.startswith("Feste Karte") for t in _texts(at))


@pytest.mark.parametrize("start", list(C.START_LABELS))
def test_both_starts_render_every_frame_of_the_default_map(start):
    at = _run(lambda a: a.session_state.__setitem__("start_radio", start))
    assert not at.error and [m.value for m in at.metric if m.label == "Paare (Ergebnis)"] == ["20 von 20"]
    last = int(_step(at).max)
    assert last >= 5 and _step(at).value == last
    for k in range(last + 1):
        _step(at).set_value(k)
        at.run()
        assert not at.exception, (start, k, [e.value for e in at.exception])


@pytest.mark.parametrize("name", ["🗺️ Mittlere Reichweite", "🕳️ Von leer starten", "🪜 Treppe: Worst Case", "🔗 Drei Pfade in einer Phase", "🧮 Große Karte"])
def test_play_runs_through_all_frames_without_duplicate_chart_keys(name):
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert _step(at).max >= 3                                                                       # mindestens vier Bilder, davon mehrere mit Diagrammen gleichen Namens
    _play(at)


def test_play_is_disabled_and_no_slider_when_there_is_nothing_to_improve():
    at = _run(lambda a: a.session_state.__setitem__("net_select", "steal"))
    assert _step(at) is None and any(b.label == "▶️ Abspielen" and b.disabled for b in at.button)
    assert any("Hier gibt es keine Phase" in c.value for c in at.caption)


def test_extreme_sizes_render():
    for n, m, reach in ((C.N_MIN, C.M_MIN, C.REACH_MIN), (C.N_MAX, C.M_MAX, C.REACH_MAX), (C.N_MIN, C.M_MAX, C.REACH_MIN), (C.N_MAX, C.M_MIN, C.REACH_MAX)):
        def setup(at, n=n, m=m, reach=reach):
            at.session_state["n_slider"], at.session_state["m_slider"], at.session_state["reach_slider"] = n, m, reach
        at = _run(setup)
        step = _step(at)
        assert step is None or step.value == step.max


def test_hidden_controls_follow_the_net():
    def labels_for(net):
        return _labels(_run(lambda a: a.session_state.__setitem__("net_select", net)))
    random_labels, fixed = labels_for("random"), labels_for("stair")
    assert {"Karte", "Fahrzeuge", "Aufträge", "Reichweite [min]", "Ballung [%]", "Zufalls-Seed"} <= random_labels
    assert fixed == {"Karte"}                                             # keine toten Regler bei festen Karten


def test_hidden_slider_values_come_back_when_the_random_map_is_shown_again():
    at = _run(lambda a: a.session_state.__setitem__("reach_slider", 90))
    at.session_state["net_select"] = "steal"
    at.run()
    at.session_state["net_select"] = "random"
    at.run()
    assert not at.exception and at.slider(key="reach_slider").value == 90


def test_step_slider_returns_to_the_last_frame_when_the_map_or_the_start_changes():
    at = _run()
    last = int(_step(at).max)
    assert _step(at).value == last
    _step(at).set_value(2)
    at.run()
    assert _step(at).value == 2
    at.session_state["net_select"] = "stair"
    at.run()
    assert not at.exception and _step(at).value == _step(at).max and _step(at).max > last
    _step(at).set_value(1)
    at.run()
    at.session_state["start_radio"] = "empty"
    at.run()
    assert not at.exception and _step(at).value == _step(at).max


def test_first_frame_shows_the_layers_the_next_the_path_and_the_last_the_certificate():
    at = _run()
    _step(at).set_value(0)
    at.run()
    assert not at.exception and any("Breitensuche" in m.value and "Phase 1 von 2" in m.value for m in at.markdown)
    assert not any("Jede mögliche Kante hat eine Ecke in der Überdeckung" in t.value.to_dict("list").get("Bedingung", []) for t in at.table)      # der Beweis erst im letzten Bild
    _step(at).set_value(1)
    at.run()
    assert not at.exception and any("Weg 1 von 2 akzeptiert" in m.value for m in at.markdown)
    _step(at).set_value(int(_step(at).max))
    at.run()
    cert = next(t.value.to_dict("list") for t in at.table if "Jede mögliche Kante hat eine Ecke in der Überdeckung" in t.value.to_dict("list").get("Bedingung", []))
    assert cert["Prüfung"][0].startswith("✅") and cert["Prüfung"][1].startswith("✅")


def test_permalink_parameters_are_clamped_and_snapped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["reach"] = "9999"
    at.query_params["ballung"] = "abc"
    at.query_params["n"] = "-5"
    at.run()
    assert not at.exception
    assert at.slider(key="reach_slider").value == C.REACH_MAX and at.slider(key="ballung_slider").value == C.DEFAULT_BALLUNG and at.slider(key="n_slider").value == C.N_MIN
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["reach"] = "42"
    at.query_params["ballung"] = "60"
    at.run()
    assert at.slider(key="reach_slider").value == 40 and at.slider(key="ballung_slider").value == 50      # auf die Regler-Schritte gerundet


def test_permalink_keeps_a_valid_start_and_falls_back_for_unknown_values():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["start"] = "empty"
    at.run()
    assert not at.exception and at.radio(key="start_radio").value == "empty"
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.query_params["start"] = "order"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET and at.radio(key="start_radio").value == C.DEFAULT_START


def test_randomize_moves_the_seed_but_not_the_distribution():
    at = _run()
    before = {m.label: m.value for m in at.metric}
    seed_before = at.number_input(key="seed_input").value
    [b for b in at.sidebar.button if "Neue Karte" in b.label][0].click()
    at.run()
    assert not at.exception and at.number_input(key="seed_input").value != seed_before
    after = {m.label: m.value for m in at.metric}
    for label in ("Größtmögliche Paarzahl", "Phasen (Mittel | Median)", "Kanten Hopcroft–Karp (Mittel | Median)", "Weniger Kanten als die einzelnen Wege"):
        assert before[label] == after[label], label


def test_experiments_run_on_demand():
    at = _run()
    markers = ("Mittel über 40 feste Karten je Reichweite", "Steigung im doppelt logarithmischen Bild", "Beide Varianten finden dieselben Wege")
    assert not any(any(m in c.value for m in markers) for c in at.caption)
    for key in ("sweep_start", "scaling_start", "variants_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert all(m in text for m in markers)


def test_experiments_from_the_empty_start_and_on_a_fixed_map():
    at = _run(lambda a: (a.session_state.__setitem__("net_select", "stair"), a.session_state.__setitem__("start_radio", "empty")))
    for key in ("sweep_start", "scaling_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_unique_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    assert len(calls) == 10 and all(re.search(r'key=f?"[a-z_]+(_\{\w+\})?"', c) for c in calls), calls
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    assert len(set(keys)) == 10, keys                                                             # jeder Schlüssel nur einmal
    assert sum(1 for c in calls if 'key=f"' in c) == 3                                            # nur die Diagramme der Abspiel-Schleife tragen den Schritt
    viz = (ROOT / "hk_visualization.py").read_text(encoding="utf-8")
    bodies = [b for b in viz.split(chr(10) + "def ") if b.startswith("build_")]
    assert "fixedrange=True" in viz and len(bodies) == 10 and all("_base(" in b or "_map_layout(" in b for b in bodies)


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))


def test_footer_is_verbatim():
    src = APP.read_text(encoding="utf-8")
    assert "https://sebastianhanisch.net/kontakt.html" in src and "Interesse an einer maßgeschneiderten Lösung für" in src and "Operations Research und Machine Learning" in src


def test_runtime_needs_only_numpy_pandas_plotly_streamlit():
    """Konvention der Konzepte-Wurzeln und -Stücke: Referenzbibliotheken (scipy, networkx) nur als Testorakel."""
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "scipy" not in req and "networkx" not in req
    for path in ROOT.glob("*.py"):
        assert not re.search(r"^\s*(import|from)\s+(scipy|networkx)\b", path.read_text(encoding="utf-8"), re.M), path.name
