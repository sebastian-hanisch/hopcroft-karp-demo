# Hopcroft–Karp – viele kürzeste Wege in einer Phase – Streamlit-Demo

Fünftes Stück der **Matching-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Nachfolger der [Verbesserungswege-Demo](https://github.com/sebastian-hanisch/augmenting-path-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Hopcroft–Karp** – an einem wachsenden Beispiel.
Die Verbesserungswege finden je Suche **einen** Weg; hier findet je **Phase** eine Breitensuche von allen freien Fahrzeugen den Graphen in **Schichten** (sie endet in der ersten Schicht, in der ein freier Auftrag anliegt), und eine Tiefensuche im Schichtgraphen liefert eine **maximale Menge knotendisjunkter kürzester Wege**, die alle auf einmal umgeklappt werden. Die Weglängen wachsen von Phase zu Phase; es sind höchstens etwa √V Phasen.

Die Demo ist **ehrlich über den Aufwand**: bei den Demogrößen (20 × 20) ist das Bündeln nicht schneller als die einzelnen Wege der Vorgänger-Demo, der Vorteil zeigt sich erst bei über hundert Fahrzeugen (Aufwand-Experiment). Hopcroft–Karp zählt **Paare, nicht Kosten**: es findet dieselben Wege wie die einzelnen kürzesten Wege und damit auch dieselben Kosten.

**Einordnung in die Reihe (die Kanten des Graphen):** dieses Stück behebt die Schwäche der Verbesserungswege, dass jede Suche nur einen Weg findet. Seine eigenen Schwächen sind die Ansatzpunkte der nächsten: es gibt nur zwei getrennte Seiten (**Blossom → Gewichteter Blossom**), Vorlieben statt Zählung (**Gale–Shapley**), alles ist vorab bekannt (**Online-Matching**). Die Kosten optimieren die Ungarische Methode und der Auktionsalgorithmus.
```
greedy-matching-demo (Wurzel: eine gewählte Zuordnung bleibt)                     [gebaut]
  ├─ augmenting-path-demo (Verbesserungswege: Paare optimal, Kosten blind)        [gebaut]
  │    ├─ hopcroft-karp-demo (viele kürzeste Wege je Phase)                       [dieses Stück]
  │    ├─ hungarian-demo (Ungarische Methode: Paare zuerst, dann Kosten)           [gebaut]
  │    │    └─ auction-algorithm-demo (Auktionsalgorithmus: dezentral)             [gebaut]
  │    └─ Blossom                                                                  [nicht gebaut]
  │   Ungarisch + Blossom → Gewichteter Blossom (Konvergenz)                       [nicht gebaut]
  ├─ Gale–Shapley → Stabile Mitbewohner                                            [nicht gebaut]
  └─ Online-Matching                                                               [nicht gebaut]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` über die 100 festen Karten (Seeds 100000–100099) belegt (20 Fahrzeuge, 20 Aufträge, Reichweite 40, Start von Greedy, wo nichts anderes steht). Aufwand = **geprüfte Adjazenzeinträge** (wie in der Verbesserungswege-Demo), Breitensuche und Tiefensuche getrennt gezählt, nie Sekunden. Mittel und Median stehen zusammen.

| Frage | Ergebnis |
|---|---|
| Wird die Paarzahl größtmöglich? | ✅ Ja, auf allen 100 Karten in allen getesteten Einstellungen (beide Starts, Reichweite 10 bis 150, n ≠ m, Ballung 0/50/100, 40 × 40), unabhängig gegen scipy, networkx (`hopcroft_karp_matching`), Brute Force und ein rekursives Kuhn geprüft, dazu je Phase die Invarianten und der Beweis (Knotenüberdeckung). |
| Was hat eine Phase? | ✅ Alle Wege einer Phase sind knotendisjunkt und gleich lang (2L+1 Kanten); die kürzeste Weglänge steigt von Phase zu Phase strikt, unabhängig über eine networkx-Suche nachgeprüft; nach der Phase gibt es keinen Weg dieser Länge mehr. Die Phasenschranke ≈ 2√ν + 1 gilt auf allen Testkarten. |
| Ist Hopcroft–Karp schneller? | ❌ **Bei 20 × 20 nicht:** 197 durchsuchte Kanten im Mittel (Median 180) gegen 140 (einzelne kürzeste Wege) und 121 (Kuhn); weniger als die einzelnen Wege nur auf 8 von 100 Karten, weniger als Kuhn auf 19. Phasen 1,9 (höchstens 3) für 2,7 Wege. |
| Von leer starten | ⚠️ 19,5 Wege in 2,4 Phasen (die erste klappt viele Wege der Länge 1 um). 209 Kanten gegen 174 (einzelne Wege) und 384 (Kuhn): Kuhn wird auf 96 von 100 Karten geschlagen, die einzelnen Wege nicht. |
| Wo es sich lohnt | ✅ 40 × 40, Reichweite 60: 326 Kanten (Median 246) gegen 393 (322) und 583 (552), weniger als die einzelnen Wege auf 61 von 100 Karten. Im Aufwand-Experiment (konstanter mittlerer Grad, 10 bis 320 Fahrzeuge, Halboktav-Gitter) sind die Steigungen im doppelt logarithmischen Bild 1,74 (Hopcroft–Karp) gegen 2,01 (einzelne Wege) und 1,94 (Kuhn); ab 113 Fahrzeugen sucht Hopcroft–Karp im Mittel weniger als die einzelnen Wege (bei 320: 22 389 gegen 39 794), gegen Kuhn ab Greedy erst bei 320 (25 880), vom leeren Start aus bei jeder Größe. |
| Wachsen die Phasen? | ⚠️ Ja, auch auf Zufallskarten: im Mittel 1,3 / 3,1 / 5,3 / 7,6 Phasen bei 10 / 40 / 160 / 320 Fahrzeugen (Steigung 0,52), weil die Wege bei sinkender Reichweite länger werden – nicht nur auf der Worst-Case-Karte. |
| Worst Case: die Treppe | ❌ Ketten 1 bis K mit je einem Verbesserungsweg der Länge 2i+1, die einander nicht erreichen: genau K Phasen bei n = K (K + 3) / 2 Fahrzeugen (knapp unter √(2n)). Aufwand ab Greedy etwa das Doppelte der einzelnen Wege (K = 23, 299 Fahrzeuge: 9 246 gegen 4 370 und 575 für Kuhn). Bei K = 5 (20 Fahrzeuge): 150 gegen 65 und 35. |
| Knappe / große Reichweite | ⚠️ Bei Reichweite 10 gibt es auf 80 von 100 Karten keine Phase (im Mittel 0,2), ab 150 keine (Greedy hat alles). |
| Und die Kosten? | ❌ Dieselben wie bei den einzelnen kürzesten Wegen (dieselben Wege, gemessen auf 100 % der Karten und beiden Starts): Median 8 % über dem Optimum ab Greedy, 36 % vom leeren Start. |
| Was sparen Frühabbruch und Sackgassen? | ⚠️ Ohne Sackgassen-Markierung durchsucht die Tiefensuche bei großen Karten (ab 226 Fahrzeugen) mehr als das Doppelte; mit voller Schicht statt Frühabbruch nie weniger. Beide finden dieselben Wege. |

## Was nicht funktioniert hat / widerlegte Vorab-Hypothesen

- **„Hopcroft–Karp spart schon bei Demogröße.“** Nein: bei 20 × 20 verliert es gegen die einzelnen Wege und gegen Kuhn. Die Verbesserungswege-Demo hatte den Ansatzpunkt „Kartengröße“ richtig benannt – die Demo zeigt ihn im Experiment, nicht im Standardfall.
- **„Die Phasenschranke ≈ √V ist eine Worst-Case-Kuriosität.“** Auch Zufallskarten brauchen mit der Größe mehr Phasen (Steigung 0,52), weil die Reichweite mit der Größe sinkt und die Wege dadurch länger werden.
- **„Hopcroft–Karp findet andere Wege als die einzelnen kürzesten Wege.“** Mit Index-Tie-Break dieselben, in derselben Reihenfolge (gemessen, nicht bewiesen; mit umgekehrter Nachbarliste nur in Hopcroft–Karp weichen die Wege auf 53 von 60 Karten ab). Die Kosten sind deshalb dieselben.
- **„Der Vorteil gegen Kuhn ist überall gleich.“** Vom leeren Start ja (Kuhn braucht viele Anläufe), ab Greedy erst bei den größten Karten: Kuhn ist dort billig, weil fast alles schon zugeordnet ist.
- **Zählregel:** die Tiefensuche prüft jedes Fahrzeug höchstens einmal je Phase (danach gesperrt oder Sackgasse); ohne diese Markierung wären die Zahlen deutlich schlechter, ohne dass die Zuordnung sich ändert.

## Was die Demo zeigt

- **Phasen in Aktion:** Schritt-Slider und ▶️ über alle Bilder: je Phase der **Schichtgraph** (links die Karte mit farbigen Schichten, rechts das Schichtschema), dann je akzeptiertem Weg ein Bild (grün: wird gewählt, rot gestrichelt: wird freigegeben, gesperrte Fahrzeuge umringt, Sackgassen der Tiefensuche gekreuzt); das letzte Bild ist das Ergebnis mit dem **Beweis** (Knotenüberdeckung). Umschalter Start: leer oder von Greedy.
- **Größtmöglich – und was kostet die Suche?** Ergebnis, Vergleichstabelle (Hopcroft–Karp, einzelne kürzeste Wege, Kuhn), Verteilung über 100 feste Karten (Histogramm, Mittel und Median, Anteil der Karten mit weniger Kanten), Aufwandstabelle; auf Abruf Reichweite-Sweep.
- **Aufwand:** 10 bis 320 Fahrzeuge bei konstantem mittleren Grad mit Steigungen und Kreuzung, das Phasenwachstum gegen die Treppe, die Worst-Case-Treppe und der Vergleich der Varianten (ohne Sackgassen, volle Schicht).
- **Feste Lehrbuchkarten** (Nichts zu verbessern, drei Pfade in einer Phase, lange Kette, Treppe) und zufällige Karten; **Wo die Annahmen enden:** welches spätere Stück an welcher Schwäche ansetzt.

## Modell und Verfahren

- **Schichten:** Breitensuche von allen freien Fahrzeugen (Schicht 0); ein Fahrzeug der Schicht d erreicht über nicht gewählte Kanten Aufträge und über deren Partner Fahrzeuge der Schicht d+1; die Suche endet in der ersten Schicht L, in der ein freier Auftrag anliegt, und stoppt (Frühabbruch) beim ersten freien Auftrag: die kürzesten Wege haben 2L+1 Kanten.
- **Tiefensuche im Schichtgraphen** auf dem eingefrorenen Matching: nur zu Fahrzeugen der nächsten Schicht, freie Aufträge nur aus Schicht L; ein akzeptierter Weg sperrt seine Fahrzeuge, ein erfolgloses Fahrzeug wird Sackgasse. Am Ende der Phase werden alle Wege umgeklappt.
- **Beweis:** die letzte, erfolglose Breitensuche liefert die erreichten Fahrzeuge $Z_V$ und Aufträge $Z_O$; $(V\setminus Z_V)\cup Z_O$ ist eine Knotenüberdeckung mit genau $|M|$ Ecken (König).
- **Phasenschranke:** nach $\lceil\sqrt\nu\rceil$ Phasen hat jeder Verbesserungsweg mindestens so viele gewählte Kanten; es bleiben höchstens $\nu/\lceil\sqrt\nu\rceil$ weitere Phasen. Auf der Treppe ist K = Phasen genau.
- **Determinismus:** Nachbarn in Indexreihenfolge, freie Fahrzeuge aufsteigend, Breitensuche FIFO, die Tiefensuche nimmt die erste passende Kante.
- **Treppe:** Kette i (i = 1…K) ist die lange Kette mit i + 1 Fahrzeugen in einer eigenen Zeile (Abstand 14, außer Reichweite der anderen); Fahrzeuge je Kette absteigend, Aufträge aufsteigend indiziert, damit das Erste-passende-Verfahren des leeren Starts dieselben billigen Kanten wählt wie Greedy.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `hk_constants.py` | Regler-Grenzen, Presets und Hilfetexte, feste Seed-Mengen |
| `hk_presets.py` | Permalink, Preset- und Zufalls-Seed-Logik (Standardmuster des Portfolios) |
| `hk_scenario.py` | Karten, eigener Zufallsgenerator, feste Lehrbuchkarten (aus den Vorgängerdemos kopiert, dazu die Treppe) |
| `hk_greedy.py`, `hk_augment.py` | Greedy-Regeln und die einzelnen kürzesten Wege sowie Kuhn aus den Vorgängerdemos (kopiert, ohne Import; Vergleichsverfahren) |
| `hk_algorithm.py` | **Neu:** Hopcroft–Karp (Schichten, Tiefensuche, Phasen), Zählregel, Beweis |
| `hk_evaluation.py` | Einordnung, Verdict, Vergleichstabelle, Verteilung über viele Karten, Sweeps, Aufwand, Treppe |
| `hk_visualization.py` | Plotly-Abbildungen (Achsen gesperrt für Touch-Geräte; Bögen bei Punkten auf einer Geraden) |
| `tests/` | Algorithmus (Handfälle, Invarianten je Phase, scipy, networkx, Brute Force und rekursives Kuhn als Gegenprobe, Gleichheit mit Stück 2, Negativkontrollen), Auswertung, Presets, belegte Zahlen, AppTest-Rauchtests (auch Abspielen auf mehrbildrigen Karten) |

Die Kopien der Vorgänger werden durch Tests bewacht (Zufallsgenerator-Vektor, Seed-2-Karte: 17 / 232 / 20 / 316). Alle Daten sind synthetisch; die Laufzeit braucht nur numpy, pandas, plotly und streamlit (scipy und networkx sind reine Testorakel).

## Lokal starten

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -v
```

Die Logik rechnet ausschließlich mit ganzen Zahlen; die im Text genannten Anteile und Mittelwerte sind deshalb auf jeder Plattform identisch.
Die CI (`.github/workflows/tests.yml`) läuft auf Ubuntu mit Python 3.12, bei jedem Push und wöchentlich mit den jeweils neuesten Bibliotheksversionen.
