"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Hopcroft–Karp"."""

# --- Regler (wie in den Vorgängerdemos) ----------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 3, 40, 20          # Fahrzeuge
M_MIN, M_MAX, DEFAULT_M = 3, 40, 20          # Aufträge
REACH_MIN, REACH_MAX, DEFAULT_REACH = 10, 150, 40   # Reichweite in Minuten; ab 142 ist auf der 100x100-Karte alles erreichbar
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG = 0, 100, 0   # ganze Prozent, Schritt 25
DEFAULT_SEED = 165
SEED_MAX = 2_000_000_000

NETS = {"random": "Zufällige Karte", "steal": "Billigste Kante klaut (2×2)", "p4": "Pfad aus vier Punkten", "paths": "Drei Pfade hintereinander", "chain": "Lange Kette (13 Kanten)", "stair": "Treppe: Worst Case (5 Ketten)"}
DEFAULT_NET = "random"
FIXED_NETS = ("steal", "p4", "paths", "chain", "stair")

START_LABELS = {"empty": "Leer", "edge": "Von Greedy (billigste Kante zuerst)"}
DEFAULT_START = "edge"

# --- feste Seed-Mengen (dieselben wie in den Vorgängerdemos; unabhängig vom Nutzer-Seed) --------------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
SWEEP_SEEDS = DIST_SEEDS[:40]
REACH_SWEEP = (10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150)
SCALE_NS = (10, 14, 20, 28, 40, 57, 80, 113, 160, 226, 320)     # Aufwand-Experiment: Fahrzeuge = Aufträge, mittlerer Grad konstant (Reichweite ~ 1/sqrt(n)), Halboktav-Gitter
SCALE_SEEDS = DIST_SEEDS[:10]
SCALE_REACH_AT_20 = 40
STAIR_KS = (2, 3, 4, 6, 8, 11, 16, 23)                           # Treppen-Experiment: n = K (K + 3) / 2

COLORS = {"matched": "#1f77b4", "add": "#2ca02c", "drop": "#d62728", "common": "#8c8c8c", "vehicle": "#111111", "order": "#ff7f0e", "dead": "#9467bd",
          "hk": "#d62728", "single": "#1f77b4", "kuhn": "#2ca02c", "layers": "Viridis"}

# --- Presets ------------------------------------------------------------------------------------------------------------------
_BASE = dict(net="random", start=DEFAULT_START, n=DEFAULT_N, m=DEFAULT_M, reach=DEFAULT_REACH, ballung=DEFAULT_BALLUNG, seed=DEFAULT_SEED)
PRESETS = {
    "⚖️ Nichts zu verbessern": {**_BASE, "net": "steal"},
    "🔗 Drei Pfade in einer Phase": {**_BASE, "net": "paths"},
    "⛓️ Lange Kette": {**_BASE, "net": "chain"},
    "🪜 Treppe: Worst Case": {**_BASE, "net": "stair"},
    "🗺️ Mittlere Reichweite": {**_BASE},
    "🕳️ Von leer starten": {**_BASE, "start": "empty"},
    "🧮 Große Karte": {**_BASE, "n": 40, "m": 40, "reach": 60, "seed": 53},
    "📡 Knappe Reichweite": {**_BASE, "reach": 10, "seed": 57},
}
# Jede Zahl in diesen Texten ist in tests/test_claims.py über die 100 festen Karten (DIST_SEEDS) belegt
PRESET_HELP = {
    "⚖️ Nichts zu verbessern": "Greedy hat hier schon die größtmögliche Paarzahl (2): keine Phase, nur die letzte Breitensuche als Beweis, dass es keinen Verbesserungsweg mehr gibt.",
    "🔗 Drei Pfade in einer Phase": "Drei getrennte Pfade aus je drei Kanten: Greedy findet 3 Paare, Hopcroft–Karp klappt alle drei Wege in EINER Phase um (6 Paare). Die einzelnen kürzesten Wege brauchen dafür drei Suchen.",
    "⛓️ Lange Kette": "Ein einziger Weg aus 13 Kanten durch die ganze Kette: eine Phase mit einem Weg, Schicht 6. Hopcroft–Karp durchsucht 26 Kanten (13 in der Breitensuche, 13 in der Tiefensuche), die einzelnen Wege 13.",
    "🪜 Treppe: Worst Case": "Fünf Ketten mit 2, 3, 4, 5 und 6 Fahrzeugen, die einander nicht erreichen: jede hat genau einen Verbesserungsweg, und die Wege sind 3, 5, 7, 9 und 11 Kanten lang. Weil Hopcroft–Karp die kürzesten Wege zuerst nimmt, braucht es fünf Phasen - eine je Kette; 150 durchsuchte Kanten gegen 65 (einzelne Wege) und 35 (Kuhn). Bei K Ketten sind es K Phasen bei n = K (K + 3) / 2 Fahrzeugen.",
    "🗺️ Mittlere Reichweite": "Auf 100 Karten (20 Fahrzeuge, 20 Aufträge, Reichweite 40, Start von Greedy) braucht Hopcroft–Karp im Mittel 1,9 Phasen für 2,7 Wege - durchsucht aber 197 Kanten (Median 180) gegen 140 (einzelne kürzeste Wege) und 121 (Kuhn): bei dieser Größe lohnt sich das Bündeln nicht (weniger als die einzelnen Wege nur auf 8 von 100 Karten).",
    "🕳️ Von leer starten": "Ohne Greedy sind es im Mittel 19,5 Wege in 2,4 Phasen, die erste Phase klappt viele Wege der Länge 1 auf einmal um. Kuhn braucht 384 Kanten, Hopcroft–Karp 209, die einzelnen Wege 174: gegen Kuhn gewinnt Hopcroft–Karp auf 96 von 100 Karten, gegen die einzelnen Wege nicht.",
    "🧮 Große Karte": "Bei 40 Fahrzeugen und 40 Aufträgen (Reichweite 60, Start von Greedy) durchsucht Hopcroft–Karp im Mittel 326 Kanten (Median 246), die einzelnen Wege 393 (322) und Kuhn 583 (552): erst hier gewinnt das Bündeln - auf 61 von 100 Karten weniger als die einzelnen Wege.",
    "📡 Knappe Reichweite": "Bei Reichweite 10 gibt es auf 80 von 100 Karten gar keine Phase (Greedy hat schon die größtmögliche Paarzahl), im Mittel 0,2. Diese Karte gehört zu den 20 mit einer Phase - und auch dort ist es ein einziger Weg.",
}
