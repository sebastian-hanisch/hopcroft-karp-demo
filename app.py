"""Hopcroft–Karp - viele kürzeste Wege in einer Phase - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Hopcroft–Karp - und lässt stattdessen das Beispiel wachsen.
Fünftes Stück der Matching-Linie der "Konzepte"-Reihe, Nachfolger der Verbesserungswege: statt eines Wegs je Suche eine ganze Phase disjunkter kürzester Wege. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import hk_constants as C
import hk_evaluation as ev
from hk_algorithm import frames
from hk_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from hk_scenario import build
from hk_visualization import (
    build_cover_map,
    build_layer_map,
    build_layer_schematic,
    build_phase_growth,
    build_phases_bar,
    build_reach_sweep,
    build_scaling,
    build_scans_hist,
    build_stair,
    build_variants,
)

st.set_page_config(page_title="Hopcroft–Karp – Sebastian Hanisch", layout="wide")


def _pct(x, digits=0):
    return "–" if x is None else f"{x:.{digits}f} %".replace(".", ",")


def _share(x):
    """Anteil (0..1) als 'nn %'."""
    return f"{100 * x:.0f} %"


def _f(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _int(x):
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{x:,.0f}".replace(",", " ")


def _path_text(path):
    """'+F1–A1, −F2–A1, ...': + wird gewählt, − wird freigegeben."""
    return ", ".join(f"{'+' if kind == 'add' else '−'}F{i + 1}–A{j + 1}" for i, j, kind in path)


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(params):
    net, n, m, reach, ballung, seed, start = params
    return ev.analyse(build(net, n, m, reach, ballung, seed), start)


@st.cache_data(show_spinner=False)
def _distribution(n, m, reach, ballung, start):
    return ev.distribution(n, m, reach, ballung, start)


@st.cache_data(show_spinner=False)
def _effort(n, m, reach, ballung, start):
    return ev.effort_table(n, m, reach, ballung, start)


@st.cache_data(show_spinner=False)
def _reach_sweep(n, m, ballung, start):
    return ev.reach_sweep(n, m, ballung, start)


@st.cache_data(show_spinner=False)
def _scaling(start, variants):
    return ev.scaling(start, variants=variants)


@st.cache_data(show_spinner=False)
def _stair(start):
    return ev.staircase_scaling(start)


st.title("🌉 Hopcroft–Karp – viele kürzeste Wege in einer Phase")
st.markdown(
    """
Die **Verbesserungswege** finden je Suche **einen** Weg und klappen ihn um; bei großen Karten wächst der Aufwand etwa mit dem Quadrat der Kartengröße. **Hopcroft–Karp** bündelt: je **Phase** eine Breitensuche von allen freien Fahrzeugen, die den Graphen in **Schichten** ordnet
(sie endet in der ersten Schicht, in der ein freier Auftrag anliegt), und danach eine Tiefensuche, die im Schichtgraphen eine **maximale Menge knotendisjunkter kürzester Wege** findet - alle auf einmal umgeklappt. Die Weglängen wachsen von Phase zu Phase, und es sind höchstens etwa **√V Phasen**.
Wichtig für die Einordnung: das Bündeln lohnt sich erst bei **großen** Karten. Bei 20 × 20 Fahrzeugen ist Hopcroft–Karp nicht schneller als die einzelnen Wege (nur auf einzelnen 40 × 40-Karten schon) - der Aufwand-Vergleich unten zeigt, ab wann es im Mittel kippt.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - fünftes Stück der Matching-Linie der \"Konzepte\"-Reihe, Nachfolger der Verbesserungswege - **ein** Verfahren an einem wachsenden Beispiel. "
    "Hopcroft–Karp zählt **Paare, nicht Kosten**: es findet dieselben Wege wie die einzelnen kürzesten Wege der Verbesserungswege-Demo und damit auch dieselben Kosten; die Kosten optimiert die Ungarische Methode (und, dezentral, der Auktionsalgorithmus) - beide sind schon gebaut. "
    "Die Schwächen dieses Stücks sind die Ansatzpunkte der nächsten: **Blossom** und **Gewichteter Blossom** (allgemeine Graphen), **Gale–Shapley** (Vorlieben statt Kosten) und **Online-Matching** - noch nicht gebaut."
)

with st.expander("So funktioniert Hopcroft–Karp", expanded=True):
    st.markdown(
        """
1. **Phase:** eine **Breitensuche** startet gleichzeitig an allen freien Fahrzeugen (Schicht 0). Ein Fahrzeug der Schicht d erreicht über nicht gewählte Kanten Aufträge und über deren Partner die Fahrzeuge der Schicht d+1. Die Suche endet in der ersten Schicht **L**, in der ein freier Auftrag anliegt: die kürzesten Verbesserungswege haben 2L+1 Kanten.
2. **Tiefensuche:** von den freien Fahrzeugen aus wird nur durch den **Schichtgraphen** gesucht (immer eine Schicht tiefer). Ein akzeptierter Weg **sperrt** seine Fahrzeuge; ein Fahrzeug, von dem aus es nicht weitergeht, wird als **Sackgasse** markiert und nie wieder betreten. So prüft die Tiefensuche jedes Fahrzeug höchstens einmal je Phase.
3. **Umklappen:** am Ende der Phase werden **alle** gefundenen Wege umgeklappt - sie sind knotendisjunkt und gleich lang. Danach gibt es keinen Weg der Länge 2L+1 mehr, der nächste ist länger.
4. **Ende:** findet die Breitensuche keinen freien Auftrag mehr, ist die Paarzahl größtmöglich; die erreichten Ecken liefern die **Knotenüberdeckung** als Beweis (wie bei den Verbesserungswegen).
        """
    )

st.caption("🎯 Schnellstart – eine Beispielkarte laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Karte", list(C.NETS), key="net_select", format_func=lambda k: C.NETS[k],
        help="Eine zufällige Karte mit Fahrzeugen und Aufträgen, oder eine der festen Lehrbuchkarten. Die Treppe ist der Worst Case: jede Kette braucht ihre eigene Phase.",
    )
    if net_key == "random":
        n = st.slider("Fahrzeuge", *bounds("n_slider"), key="n_slider", help="Anzahl der Fahrzeuge. Die Regler enden bei 40: der Vorteil von Hopcroft–Karp zeigt sich im Mittel erst bei über hundert Fahrzeugen (Aufwand-Experiment unten).")
        st.session_state[KEPT["n_slider"]] = n
        m = st.slider("Aufträge", *bounds("m_slider"), key="m_slider", help="Anzahl der Aufträge.")
        st.session_state[KEPT["m_slider"]] = m
        reach = st.slider(
            "Reichweite [min]", *bounds("reach_slider"), key="reach_slider", step=5,
            help="Wie weit ein Fahrzeug höchstens fahren darf. Ab Greedy braucht Hopcroft–Karp bei Reichweite 10 im Mittel 0,2 Phasen (auf 80 von 100 Karten keine), bei 40 sind es 1,9, ab 150 keine: Greedy hat dann schon alles.",
        )
        st.session_state[KEPT["reach_slider"]] = reach
        ballung = st.slider("Ballung [%]", *bounds("ballung_slider"), key="ballung_slider", step=25, help="0 = Fahrzeuge und Aufträge gleichmäßig verteilt, 100 = alle um drei Stadtteile gruppiert.")
        st.session_state[KEPT["ballung_slider"]] = ballung
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neue Karte generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed. Die Verteilung über 100 feste Karten weiter unten ändert sich dabei nicht.")
    else:
        n = int(st.session_state.get(KEPT["n_slider"], C.DEFAULT_N))
        m = int(st.session_state.get(KEPT["m_slider"], C.DEFAULT_M))
        reach = int(st.session_state.get(KEPT["reach_slider"], C.DEFAULT_REACH))
        ballung = int(st.session_state.get(KEPT["ballung_slider"], C.DEFAULT_BALLUNG))
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Diese Karte ist fest - es gibt nichts zu erzeugen. Zahl der Fahrzeuge und Aufträge, Reichweite, Ballung und Seed gehören zur zufälligen Karte.")

# --- Phasen in Aktion -----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Phasen in Aktion")
step_col, start_col, play_col = st.columns([4, 4, 2])
with start_col:
    start = st.radio("Start", list(C.START_LABELS), key="start_radio", format_func=lambda k: C.START_LABELS[k],
                     help="Von Greedy aus bleiben wenige Phasen. Vom leeren Start klappt die erste Phase viele Wege der Länge 1 auf einmal um (ein Erste-passende-Verfahren, das die Kosten nicht kennt) - dort zeigt sich der Vorteil gegen Kuhn am deutlichsten.")

params = (net_key, int(n), int(m), int(reach), int(ballung), int(seed))
if net_key in C.FIXED_NETS:
    params = (net_key, C.DEFAULT_N, C.DEFAULT_M, C.DEFAULT_REACH, C.DEFAULT_BALLUNG, C.DEFAULT_SEED)
params = params + (start,)
with st.spinner("Rechne..."):
    a = _analysis(params)
sc, res, opt = a.scenario, a.result, a.opt
level, code, d = ev.verdict(a)
fr = frames(res)
n_frames = len(fr)
if st.session_state.get("hk_step_owner") != params:
    st.session_state["hk_step"] = n_frames - 1
    st.session_state["hk_step_owner"] = params
with step_col:
    if n_frames > 1:
        step = st.slider("Bild", 0, n_frames - 1, key="hk_step", help="Je Phase ein Bild mit dem Schichtgraphen und je akzeptiertem Weg ein weiteres; ganz rechts das Ergebnis mit dem Beweis.")
    else:
        step = 0
        st.caption("Hier gibt es keine Phase: das Start-Matching hat schon die größtmögliche Paarzahl. Rechts der Beweis dafür.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch", disabled=n_frames <= 1)
sync_query_params({"net_select": net_key, "start_radio": start, "n_slider": int(n), "m_slider": int(m), "reach_slider": int(reach), "ballung_slider": int(ballung), "seed_input": int(seed)})
view_slot = st.empty()


def _cert_table():
    feasible = [(i, j) for i in range(sc.n) for j in range(sc.m) if sc.feasible[i, j]]
    touched = all(i in set(res.cover_v) or j in set(res.cover_o) for i, j in feasible)
    ok = lambda b: "✅" if b else "❌"
    rows = [("Jede mögliche Kante hat eine Ecke in der Überdeckung", f"{ok(touched)} {len(feasible)} mögliche Kanten geprüft"),
            ("Die Überdeckung hat genau so viele Ecken wie das Matching Paare", f"{ok(len(res.cover_v) + len(res.cover_o) == res.count)} {len(res.cover_v) + len(res.cover_o)} Ecken, {res.count} Paare"),
            ("Letzte Breitensuche", f"{_int(res.final_scanned)} durchsuchte Kanten, kein freier Auftrag erreichbar")]
    return {"Bedingung": [r[0] for r in rows], "Prüfung": [r[1] for r in rows]}


def _render(k):
    """Bild k: links die Karte, rechts das Schichtschema; am Ende der Beweis."""
    with view_slot.container():
        ph_idx, stage = fr[k]
        if ph_idx >= len(res.phases):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Ergebnis** - {res.count} Paare, {res.cost} Minuten")
            c1.plotly_chart(build_cover_map(sc, res.pairs, res.cover_v, res.cover_o), width="stretch", key=f"cover_map_{k}")
            c2.markdown("**Beweis:** die Knotenüberdeckung")
            c2.table(_cert_table())
            st.caption(f"Die letzte Breitensuche ({res.final_scanned} durchsuchte Kanten) erreicht keinen freien Auftrag. Die violetten Ecken - {len(res.cover_v)} Fahrzeuge und {len(res.cover_o)} Aufträge - berühren jede mögliche Kante; "
                       f"da es genau so viele Ecken wie Paare ({res.count}) sind, kann kein Matching mehr Paare haben.")
            return
        ph = res.phases[ph_idx]
        before = res.states[ph_idx]
        free = sc.n - len(before)
        c1, c2 = st.columns(2)
        if stage == 0:
            c1.markdown(f"**Phase {ph_idx + 1} von {len(res.phases)}: Breitensuche** - von {free} freien Fahrzeugen bis Schicht {ph.L}")
            c2.markdown(f"**Schichtgraph** - Wege haben {ph.length} Kanten")
            cap = (f"Die Breitensuche hat {ph.bfs_scanned} Kanten durchsucht und endet in Schicht {ph.L}, weil dort ein freier Auftrag anliegt: die kürzesten Verbesserungswege haben 2·{ph.L}+1 = {ph.length} Kanten. "
                   f"Die Zahl an einem Fahrzeug ist seine Schicht.")
        else:
            p = ph.paths[stage - 1]
            so_far = sum(x.scanned for x in ph.paths[:stage])
            c1.markdown(f"**Phase {ph_idx + 1}: Weg {stage} von {len(ph.paths)} akzeptiert**")
            c2.markdown(f"**Schichtgraph** - Weg {stage} von {len(ph.paths)}")
            cap = (f"Weg {stage}: {_path_text(p.path)} (+ wird gewählt, − wird freigegeben), {p.length} Kanten. Die Tiefensuche hat bis hier {so_far} Kanten durchsucht, davon {p.scanned} für diesen Weg (Sackgassen: {len(p.dead)}). "
                   f"Die Fahrzeuge des Wegs sind gesperrt.")
            if stage == len(ph.paths):
                cap += (f" Damit ist die Phase fertig: {len(ph.paths)} Weg(e), {ph.tail_scanned} Kanten danach noch vergeblich durchsucht. Alle Wege werden jetzt umgeklappt: {ph.pairs_before} → {ph.pairs_after} Paare.")
        c1.plotly_chart(build_layer_map(sc, before, ph, stage), width="stretch", key=f"layer_map_{k}")
        c2.plotly_chart(build_layer_schematic(sc, before, ph, stage), width="stretch", key=f"schematic_{k}")
        st.caption(cap)


if auto_play:
    for k in range(n_frames):
        _render(k)
        time.sleep(min(0.8, 8.0 / max(n_frames, 1)))
    step = n_frames - 1
else:
    _render(step)

st.plotly_chart(build_phases_bar(res.phases, res.final_scanned), width="stretch", key="phase_chart")
st.caption("Quadrate sind Fahrzeuge (Farbe: Schicht; hohl grau: von keinem freien Fahrzeug erreicht), Kreise Aufträge (ausgefüllt: haben ein Fahrzeug, hohl: frei). Dunkle Linien: Kanten des Schichtgraphen; blau: gewählte Paare; grün: ein akzeptierter Weg wird gewählt, rot gestrichelt: seine Paare werden freigegeben. "
           "Grün umringt: Fahrzeuge auf akzeptierten Wegen, violett gekreuzt: Sackgassen der Tiefensuche. Rechts das Schema: Spalte = Schicht. Unten die durchsuchten Kanten je Phase (Breitensuche und Tiefensuche) und der Beweis.")

st.markdown("---")

# --- Größtmöglich - und was kostet die Suche? -----------------------------------------------------------------------------------------------

st.markdown("## 🎯 Größtmöglich – und was kostet die Suche?")
st.caption("Verglichen werden durchsuchte Kanten: Hopcroft–Karp (Breitensuche und Tiefensuche je Phase, getrennt gezählt) gegen die einzelnen kürzesten Wege und Kuhn aus der Verbesserungswege-Demo, vom selben Start aus. Mittel **und** Median, weil der Aufwand von Karte zu Karte schwankt. Gezählt werden Kanten, nie Sekunden.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Paare (Ergebnis)", f"{res.count} von {opt.count}", help="Nach dem Ende hat das Matching die größtmögliche Paarzahl (Messlatte: die exakte Referenz aus der Greedy-Matching-Demo).")
m2.metric("Phasen", f"{d['phases']}", delta=f"{d['paths']} Wege umgeklappt", delta_color="off", help=f"Schichten der Phasen (halbe Weglänge): {d['Ls']}; Wege je Phase: {d['paths_per_phase']}. Einzelne kürzeste Wege bräuchten {d['single_rounds']} Suchen.")
m3.metric("Durchsuchte Kanten", _int(d["scanned"]), delta=f"Breitensuche {_int(d['bfs'])}, Tiefensuche {_int(d['dfs'])}", delta_color="off",
          help=f"Einzelne kürzeste Wege: {_int(d['single_scanned'])}, Kuhn: {_int(d['kuhn_scanned'])}. Das Netz hat {d['edges']} Kanten. Die letzte, vergebliche Breitensuche ({_int(d['final_scanned'])}) ist enthalten.")
m4.metric("Kosten (nur Info)", f"{res.cost} min", delta=(f"{_pct(d['cost_gap_pct'], 1)} über dem Optimum" if d["cost_gap_pct"] is not None else None), delta_color="off",
          help=f"Hopcroft–Karp kennt die Kosten nicht: dieselben Wege wie die einzelnen kürzesten Wege ergeben dieselben Kosten (hier {'ja' if d['same_as_single'] else 'nein'}). Optimum: {d['opt_cost']} min.")

if code == "none":
    st.info("ℹ️ Keine einzige Kante ist möglich – die Reichweite ist zu klein. Es gibt nichts zuzuordnen.")
elif code == "mismatch":
    st.error(f"Abweichung von der Referenz: {res.count} Paare gegen {opt.count}. Das dürfte nicht vorkommen.")
elif d["phases"] == 0:
    st.success(f"✅ Größtmögliche Paarzahl ({res.count}): der Start hat sie schon, es ist keine Phase nötig - nur die letzte Breitensuche ({_int(d['final_scanned'])} Kanten) als Beweis.")
else:
    better = "weniger" if d["scanned"] < d["single_scanned"] else "mehr"
    st.success(f"✅ Größtmögliche Paarzahl ({res.count}) in {d['phases']} Phase(n) mit {d['paths']} Weg(en). Hopcroft–Karp durchsuchte {_int(d['scanned'])} Kanten ({better} als die einzelnen kürzesten Wege mit {_int(d['single_scanned'])}, Kuhn: {_int(d['kuhn_scanned'])}). "
               f"Die Wege sind dieselben wie bei den einzelnen kürzesten Wegen, deshalb auch die Kosten: {res.cost} statt {d['opt_cost']} Minuten.")

st.markdown("**Die Verfahren im Vergleich auf dieser Karte**")
cmp_rows = ev.compare_table(a)
st.table({"Verfahren": [r["label"] for r in cmp_rows], "Suchen": [r["searches"] for r in cmp_rows], "Wege": [r["paths"] for r in cmp_rows], "durchsuchte Kanten": [_int(r["scanned"]) for r in cmp_rows],
          "davon Breitensuche": [_int(r["bfs"]) for r in cmp_rows], "davon Tiefensuche": [_int(r["dfs"]) for r in cmp_rows], "Kosten [min]": [r["cost"] for r in cmp_rows]})

if net_key in C.FIXED_NETS:
    st.info("Feste Karte: es gibt nur diese eine Ziehung. Für die Verteilung über viele Karten eine zufällige Karte wählen.")
else:
    st.markdown(f"**Nicht nur diese eine Karte:** {len(C.DIST_SEEDS)} feste Karten mit denselben Einstellungen (Fahrzeuge {n}, Aufträge {m}, Reichweite {reach}, Ballung {ballung} %, Start: {C.START_LABELS[start]}), getrennt vom Seed oben.")
    dist = _distribution(int(n), int(m), int(reach), int(ballung), start)
    if dist["n_valid"] == 0:
        st.info("ℹ️ Bei dieser Reichweite gibt es auf keiner der Karten ein mögliches Paar.")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Größtmögliche Paarzahl", _share(dist["share_maximal"]), help="Anteil der Karten, auf denen Hopcroft–Karp die größtmögliche Paarzahl erreicht - hier immer alle.")
        p2.metric("Phasen (Mittel | Median)", f"{_f(dist['phases_mean'])} | {_f(dist['phases_median'], 0)}", delta=f"höchstens {dist['phases_max']}; Wege im Mittel {_f(dist['paths_mean'])}", delta_color="off",
                  help=f"Verteilung der Phasenzahl über die Karten: {dist['phase_hist']}.")
        p3.metric("Kanten Hopcroft–Karp (Mittel | Median)", f"{_int(dist['hk_mean'])} | {_int(dist['hk_median'])}", delta=f"einzelne Wege {_int(dist['single_mean'])} | {_int(dist['single_median'])}", delta_color="off",
                  help=f"Kuhn: {_int(dist['kuhn_mean'])} | {_int(dist['kuhn_median'])}. Die Verteilung ist annähernd symmetrisch (Mittel nahe Median).")
        p4.metric("Weniger Kanten als die einzelnen Wege", _share(dist["hk_less_single"]), delta=f"als Kuhn: {_share(dist['hk_less_kuhn'])}", delta_color="off",
                  help="Anteil der Karten, auf denen Hopcroft–Karp weniger Kanten durchsucht als die einzelnen kürzesten Wege bzw. Kuhn.")
        st.success(f"✅ Auf {_share(dist['share_maximal'])} der {dist['n_seeds']} Karten dieser Einstellung erreicht Hopcroft–Karp die größtmögliche Paarzahl - und findet dieselben Wege in derselben Reihenfolge wie die einzelnen kürzesten Wege (auf {_share(dist['same_share'])}). "
                   f"Kanten: {_int(dist['hk_mean'])} im Mittel (Median {_int(dist['hk_median'])}) gegen {_int(dist['single_mean'])} ({_int(dist['single_median'])}) und {_int(dist['kuhn_mean'])} ({_int(dist['kuhn_median'])}).")
        h1, h2 = st.columns([3, 2])
        h1.plotly_chart(build_scans_hist({"Hopcroft–Karp": dist["hk_scans"], "einzelne kürzeste Wege": dist["single_scans"], "Kuhn": dist["kuhn_scans"]}, current=d["scanned"]), width="stretch", key="scans_hist")
        h1.caption("Durchsuchte Kanten je Karte, die drei Verfahren übereinandergelegt; die Marke ist Hopcroft–Karp auf Ihrer Ziehung.")
        eff = _effort(int(n), int(m), int(reach), int(ballung), start)
        h2.markdown("**Aufwand** (Mittel und Median über die Karten)")
        h2.table({"Verfahren": [r["label"] for r in eff], "Suchen": [_f(r["searches"]) if r["searches"] is not None else "–" for r in eff], "Wege": [_f(r["paths"]) for r in eff],
                  "Kanten (Mittel)": [_int(r["mean"]) for r in eff], "Kanten (Median)": [_int(r["median"]) for r in eff]})

st.markdown("**Wie hängt der Aufwand von der Reichweite ab?**")
if st.button("Reichweite von 10 bis 150 durchfahren (40 Karten je Wert, dauert wenige Sekunden)", key="sweep_start"):
    st.session_state["sweep_on"] = (int(n), int(m), int(ballung), start)
if st.session_state.get("sweep_on") == (int(n), int(m), int(ballung), start):
    with st.spinner(f"Rechne {len(C.REACH_SWEEP)} Reichweiten × {len(C.SWEEP_SEEDS)} Karten × 3 Verfahren..."):
        sweep_rows = _reach_sweep(int(n), int(m), int(ballung), start)
    st.plotly_chart(build_reach_sweep(sweep_rows, current=int(reach) if net_key == "random" else None), width="stretch", key="sweep_chart")
    st.caption("Mittel über 40 feste Karten je Reichweite; Fahrzeuge, Aufträge, Ballung und Start wie oben. Oben in logarithmischer Skala die durchsuchten Kanten, unten die mittlere Phasenzahl. Bei knapper und bei sehr großer Reichweite gibt es kaum etwas zu tun (Greedy hat schon fast alles); "
               "bei 20 × 20 liegt Hopcroft–Karp bei keiner Reichweite im Mittel unter den einzelnen Wegen, auf 40 × 40 mit Reichweite 60 schon.")

st.markdown("---")

# --- Aufwand -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Aufwand: wann lohnt sich das Bündeln?")
if st.button("Karten von 10 bis 320 Fahrzeugen und die Treppe durchrechnen (dauert einige Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = start
if st.session_state.get("scaling_on") == start:
    with st.spinner("Rechne 11 Kartengrößen × 10 Karten × 3 Verfahren und die Treppe..."):
        sc_rows = _scaling(start, False)
        stair_rows = _stair(start)
    sl = ev.slopes(sc_rows)
    cross = ev.crossover(sc_rows, "single")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_scaling(sc_rows), width="stretch", key="scaling_chart")
    c2.table({"Fahrzeuge = Aufträge": [r["n"] for r in sc_rows], "Reichweite": [r["reach"] for r in sc_rows], "Phasen": [_f(r["phases"]) for r in sc_rows], "Hopcroft–Karp": [_int(r["hk"]) for r in sc_rows],
              "einzelne Wege": [_int(r["single"]) for r in sc_rows], "Kuhn": [_int(r["kuhn"]) for r in sc_rows], "HK weniger als einzelne Wege": [_share(r["hk_less_single"]) for r in sc_rows]})
    st.caption(f"Fahrzeuge = Aufträge bei konstantem mittleren Grad (die Reichweite sinkt mit der Wurzel der Kartengröße), Mittel über 10 feste Karten je Größe. Steigung im doppelt logarithmischen Bild: Hopcroft–Karp {_f(sl['hk'], 2)}, einzelne kürzeste Wege {_f(sl['single'], 2)}, Kuhn {_f(sl['kuhn'], 2)} (Kanten des Graphen: {_f(sl['edges'], 2)}). "
               f"Hopcroft–Karp sucht im Mittel ab {cross if cross is not None else 'keiner Größe dieses Gitters'} Fahrzeugen weniger als die einzelnen Wege und bleibt dort darunter; gegen Kuhn ab Greedy erst bei den größten Karten, vom leeren Start aus überall. Der Vorteil liegt in der Steigung, nicht in der Demogröße.")
    g1, g2 = st.columns(2)
    g1.plotly_chart(build_phase_growth(sc_rows, stair_rows), width="stretch", key="growth_chart")
    g2.plotly_chart(build_stair(stair_rows), width="stretch", key="stair_chart")
    st.caption(f"Links die Phasen: auf Zufallskarten wachsen sie mit der Kartengröße (Steigung {_f(sl['phases'], 2)}, weil die Wege bei sinkender Reichweite länger werden); die Treppe braucht genau K Phasen bei n = K (K + 3) / 2 Fahrzeugen, das liegt knapp unter der Schranke √(2n) und nähert sich ihr. "
               "Rechts der Aufwand auf der Treppe: dort lohnt sich das Bündeln nie, denn jede Phase enthält nur einen Weg, aber eine ganze Breitensuche und Tiefensuche.")

if st.button("Was sparen Frühabbruch und Sackgassen-Markierung? (dauert einige Sekunden)", key="variants_start"):
    st.session_state["variants_on"] = start
if st.session_state.get("variants_on") == start:
    with st.spinner("Rechne 11 Kartengrößen × 10 Karten × 3 Varianten..."):
        var_rows = _scaling(start, True)
    v1, v2 = st.columns([3, 2])
    v1.plotly_chart(build_variants(var_rows), width="stretch", key="variants_chart")
    v2.table({"Fahrzeuge": [r["n"] for r in var_rows], "richtig": [_int(r["hk"]) for r in var_rows], "ohne Sackgassen": [_int(r["no_prune"]) for r in var_rows], "volle Schicht": [_int(r["full_layer"]) for r in var_rows]})
    st.caption("Beide Varianten finden dieselben Wege, durchsuchen aber mehr Kanten: ohne Sackgassen-Markierung besucht die Tiefensuche dieselben erfolglosen Fahrzeuge mehrmals (bei großen Karten mehr als das Doppelte), und die volle Schicht durchsucht auch die Fahrzeuge der Schicht L, die der Frühabbruch nicht braucht.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur die Paarzahl zählt** | Hopcroft–Karp kennt die Kosten nicht und findet dieselben Wege wie die einzelnen kürzesten Wege: die Kosten liegen im Median 8 % (ab Greedy) bzw. 36 % (vom leeren Start) über dem Optimum. | **Ungarische Methode** und **Auktionsalgorithmus** (beide gebaut) |
| **Es gibt zwei getrennte Seiten** | Fahrzeuge und Aufträge bilden zwei Gruppen. Sollen sich Fahrer untereinander paaren, gibt es Zyklen ungerader Länge, die der Schichtgraph nicht behandelt. | **Blossom**, dann **Gewichteter Blossom** |
| **Beide Seiten sind gleichgültig gegenüber der Zuordnung** | Hopcroft–Karp fragt nicht, ob ein Fahrzeug lieber einen anderen Auftrag hätte. Haben beide Seiten Vorlieben, ist ein stabiles Ergebnis nicht dasselbe wie ein größtmögliches. | **Gale–Shapley** |
| **Alles ist vorab bekannt** | Aufträge kommen hier alle vor der Rechnung an. Kommen sie nacheinander und sind schon zugesagt, darf nicht mehr umgeklappt werden. | **Online-Matching** |
| **Die Karte ist groß** | Bei 20 × 20 Fahrzeugen sucht das Bündeln im Mittel nicht weniger (auf 40 × 40 mit Reichweite 60 schon): der Vorteil zeigt sich zuverlässig erst bei über hundert Fahrzeugen. Die Phasenschranke √V ist eine Schranke: auf Zufallskarten wachsen die Phasen tatsächlich etwa mit √n, auf der Treppe genau. | größere Karten (Aufwand-Experiment oben) |
"""
)
st.caption("Die Nachbarn der Matching-Linie (noch nicht gebaut): Blossom, Gewichteter Blossom, Gale–Shapley, Stabile Mitbewohner und Online-Matching. Bereits gebaut: die Wurzel (Greedy-Matching), die Verbesserungswege, die Ungarische Methode, der Auktionsalgorithmus und diese Demo.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Bipartiter Graph mit Fahrzeugen $V$, Aufträgen $O$ und möglichen Paaren $E$; gesucht ist ein Matching $M$ mit größtmöglicher Paarzahl $\nu$. Ein **Verbesserungsweg** beginnt an einem freien Fahrzeug, wechselt nicht gewählte und gewählte Kanten und endet an einem freien Auftrag; Umklappen ergibt $|M|+1$ (Berge: $M$ ist größtmöglich $\Leftrightarrow$ es gibt keinen Verbesserungsweg).

**Phase.** Sei $\ell$ die Länge eines kürzesten Verbesserungswegs. Eine Breitensuche von allen freien Fahrzeugen ordnet die Fahrzeuge in Schichten $d(v)$ (Schicht 0: freie Fahrzeuge) und endet in der ersten Schicht $L$, in der ein freier Auftrag anliegt; dann ist $\ell=2L+1$. Der **Schichtgraph** enthält die Kanten $v\to o$ mit $d(\mathrm{Partner}(o))=d(v)+1$ (und zu freien Aufträgen nur aus Schicht $L$). Eine Tiefensuche findet darin eine **maximale** Menge knotendisjunkter Wege; alle werden umgeklappt.

**Kernsatz.** Nach einer Phase ist jeder Verbesserungsweg mindestens $2L+3$ Kanten lang: die Weglänge wächst von Phase zu Phase strikt.

**Phasenschranke.** Nach $t=\lceil\sqrt{\nu}\,\rceil$ Phasen hat jeder Verbesserungsweg mindestens $t$ gewählte Kanten. Die symmetrische Differenz mit einem größtmöglichen Matching besteht aus höchstens $\nu/t$ knotendisjunkten Wegen; also sind höchstens $\nu/t\le\sqrt{\nu}$ weitere Phasen nötig, jede mindestens einen Weg. Insgesamt höchstens $2\sqrt{\nu}+1$ Phasen und $O(|E|\sqrt{|V|})$ Aufwand.

**Treppe.** Kette $i$ ($i=1,\dots,K$) hat $i+1$ Fahrzeuge und genau einen Verbesserungsweg der Länge $2i+1$ (ab Greedy); die Ketten sind unabhängig, also braucht Hopcroft–Karp $K$ Phasen bei $n=K(K+3)/2$ Fahrzeugen, das sind $\approx\sqrt{2n}$.

**Beweis.** Findet die Breitensuche keinen freien Auftrag, seien $Z_V$ die erreichten Fahrzeuge und $Z_O$ die Aufträge, die an ihnen hängen. Dann ist $(V\setminus Z_V)\cup Z_O$ eine Knotenüberdeckung mit genau $|M|$ Ecken (König); kein Matching kann größer sein.

**Aufwand hier.** Gezählt werden geprüfte Adjazenzeinträge, Breitensuche und Tiefensuche getrennt; die Tiefensuche prüft dank Sperren und Sackgassen jedes Fahrzeug höchstens einmal je Phase (die Stapelposition ist der current arc).

**Grenzen.** (1) Nur die Paarzahl. (2) Zwei Seiten. (3) Vorlieben. (4) Alles vorab bekannt. (5) Kleine Karten.

Implementiert in `hk_scenario.py` (Karten, eigener Zufallsgenerator, Treppe), `hk_greedy.py` und `hk_augment.py` (Greedy, einzelne Wege und Kuhn aus den Vorgängerdemos), `hk_algorithm.py` (Hopcroft–Karp, Beweis), `hk_evaluation.py` (Kennzahlen, Verteilung, Aufwand).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
