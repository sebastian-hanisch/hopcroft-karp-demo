"""Plotly-Abbildungen von Hopcroft–Karp: Karte mit Schichten und Wegen, Schichtschema, Beweis (Knotenüberdeckung), Phasen, Verteilung, Sweeps, Aufwand, Treppe.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Punkte auf einer Geraden werden mit Bögen gezeichnet."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import hk_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.08), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _collinear(sc):
    """Liegen alle Punkte auf einer Geraden? Dann würden sich die Paar-Linien überdecken - sie werden gebogen gezeichnet."""
    pts = list(sc.vehicles + sc.orders)
    (x0, y0), (x1, y1) = pts[0], next((p for p in pts if p != pts[0]), pts[0])
    return all((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) == 0 for x, y in pts)


def _edge_path(sc, i, j, curved, steps=14):
    """Punkte einer Paar-Linie: gerade, oder (bei Punkten auf einer Geraden) als Bogen, dessen Seite je Paar wechselt."""
    (vx, vy), (ox, oy) = sc.vehicles[i], sc.orders[j]
    if not curved:
        return [vx, ox], [vy, oy]
    dx, dy = ox - vx, oy - vy
    side = 1 if (i + j) % 2 == 0 else -1
    cx, cy = (vx + ox) / 2 - side * 0.35 * dy, (vy + oy) / 2 + side * 0.35 * dx      # Kontrollpunkt senkrecht zur Verbindung
    ts = [k / steps for k in range(steps + 1)]
    return ([(1 - t) ** 2 * vx + 2 * (1 - t) * t * cx + t * t * ox for t in ts], [(1 - t) ** 2 * vy + 2 * (1 - t) * t * cy + t * t * oy for t in ts])


def _segments(sc, pairs, curved=False):
    """Linienspur für eine Menge von Paaren (None trennt die Segmente)."""
    x, y = [], []
    for i, j in pairs:
        px, py = _edge_path(sc, i, j, curved)
        x += px + [None]
        y += py + [None]
    return x, y


def _feasible_pairs(sc):
    return [(i, j) for i in range(sc.n) for j in range(sc.m) if sc.feasible[i, j]]


def _map_layout(fig, sc, height):
    xs = [p[0] for p in sc.vehicles + sc.orders]
    ys = [p[1] for p in sc.vehicles + sc.orders]
    pad = 8
    if _collinear(sc):
        # Punkte auf einer Geraden: nur die Bögen brauchen Höhe. Das Seitenverhältnis wird freigegeben, sonst wird eine lange Kette zu einem dünnen Streifen.
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        if max(xs) - min(xs) >= max(ys) - min(ys):
            fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
            fig.update_yaxes(visible=False, range=[cy - span * 0.22, cy + span * 0.22])
        else:
            fig.update_xaxes(visible=False, range=[cx - span * 0.22, cx + span * 0.22])
            fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
        return _base(fig, min(height, 320))
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    return _base(fig, height)


def _vertices(fig, sc, matched_v, matched_o, cover_v=(), cover_o=()):
    """Fahrzeuge (Quadrate) und Aufträge (Kreise); ohne Partner nur als Umriss; Ecken der Überdeckung in Violett."""
    small = sc.n + sc.m <= 16
    cover_v, cover_o = set(cover_v), set(cover_o)
    for kind, pts, matched, cover, symbol, color, prefix, tpos in (("Fahrzeug", sc.vehicles, matched_v, cover_v, "square", C.COLORS["vehicle"], "F", "top center"),
                                                                    ("Auftrag", sc.orders, matched_o, cover_o, "circle", C.COLORS["order"], "A", "bottom center")):
        for on, sym, name in ((True, symbol, f"{kind} mit Partner"), (False, symbol + "-open", f"{kind} ohne Partner")):
            idx = [k for k in range(len(pts)) if (k in matched) == on and k not in cover]
            if idx:
                fig.add_trace(go.Scatter(
                    x=[pts[k][0] for k in idx], y=[pts[k][1] for k in idx], mode="markers+text" if small else "markers", name=name,
                    text=[f"{prefix}{k + 1}" for k in idx] if small else None, textposition=tpos,
                    hovertext=[f"{kind} {k + 1} ({pts[k][0]}, {pts[k][1]})" for k in idx], hoverinfo="text",
                    marker=dict(symbol=sym, size=10 if on else 11, color=color, line=dict(width=2, color=color))))
        idx = sorted(cover)
        if idx:
            fig.add_trace(go.Scatter(
                x=[pts[k][0] for k in idx], y=[pts[k][1] for k in idx], mode="markers+text" if small else "markers", name=f"{kind} in der Überdeckung",
                text=[f"{prefix}{k + 1}" for k in idx] if small else None, textposition=tpos,
                hovertext=[f"{kind} {k + 1} ({pts[k][0]}, {pts[k][1]}) - Ecke der Überdeckung" for k in idx], hoverinfo="text",
                marker=dict(symbol=symbol, size=12, color=C.COLORS["dead"], line=dict(width=2, color="#4b2d73"))))


def build_cover_map(sc, pairs, cover_v, cover_o, height=430):
    """Beweis der Erschöpfung: die violetten Ecken (Knotenüberdeckung) berühren jede mögliche Kante, und es sind genau so viele wie Paare."""
    curved = _collinear(sc)
    fig = go.Figure()
    ex, ey = _segments(sc, _feasible_pairs(sc), curved)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.4)", width=1), hoverinfo="skip", name="mögliche Paare"))
    px, py = _segments(sc, pairs, curved)
    fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color=C.COLORS["matched"], width=3.5), hoverinfo="skip", name="gewählt"))
    _vertices(fig, sc, {i for i, _ in pairs}, {j for _, j in pairs}, cover_v, cover_o)
    fig = _map_layout(fig, sc, height)
    fig.update_layout(showlegend=False)          # die Legende würde in schmalen Spalten die Zeichenfläche auf fast null drücken; die Farben stehen in der Erklärung unter der Karte
    return fig


def _dead_upto(phase, stage):
    """Als Sackgasse markierte Fahrzeuge, soweit die Tiefensuche nach `stage` akzeptierten Wegen schon ist (nach dem letzten Weg auch der Rest)."""
    dead = [v for p in phase.paths[:stage] for v in p.dead]
    if stage >= len(phase.paths):
        dead += list(phase.tail_dead)
    return dead


def build_layer_map(sc, pairs, phase, stage, height=430):
    """Karte vor der Phase: Fahrzeuge nach Schicht eingefärbt (Zahl = Schicht), Kanten des Schichtgraphen dunkel, gewählte Paare blau; nach `stage` akzeptierten Wegen
    diese grün (wird gewählt) und rot gestrichelt (wird freigegeben), Fahrzeuge auf akzeptierten Wegen grün umringt, Sackgassen der Tiefensuche violett gekreuzt."""
    curved = _collinear(sc)
    fig = go.Figure()
    ex, ey = _segments(sc, _feasible_pairs(sc), curved)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.3)", width=1), hoverinfo="skip", name="mögliche Paare"))
    lx, ly = _segments(sc, list(phase.layer_edges), curved)
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="rgba(50,50,50,0.7)", width=1.6), hoverinfo="skip", name="Schichtgraph"))
    kx, ky = _segments(sc, list(pairs), curved)
    fig.add_trace(go.Scatter(x=kx, y=ky, mode="lines", line=dict(color=C.COLORS["matched"], width=3.5), hoverinfo="skip", name="gewählt"))
    on_path = set()
    for p in phase.paths[:stage]:
        drops = [(i, j) for i, j, kind in p.path if kind == "drop"]
        adds = [(i, j) for i, j, kind in p.path if kind == "add"]
        dx, dy = _segments(sc, drops, curved)
        fig.add_trace(go.Scatter(x=dx, y=dy, mode="lines", line=dict(color=C.COLORS["drop"], width=4.5, dash="dash"), hoverinfo="skip", name="wird freigegeben", showlegend=False))
        ax, ay = _segments(sc, adds, curved)
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines", line=dict(color=C.COLORS["add"], width=4.5), hoverinfo="skip", name="wird gewählt", showlegend=False))
        on_path.update(p.chain_v)
    small = sc.n + sc.m <= 24
    dist, top = phase.dist, max(phase.L, 1)
    matched_o = {j for _, j in pairs}
    reached = [i for i in range(sc.n) if dist[i] >= 0]
    other = [i for i in range(sc.n) if dist[i] < 0]
    if other:
        fig.add_trace(go.Scatter(x=[sc.vehicles[k][0] for k in other], y=[sc.vehicles[k][1] for k in other], mode="markers+text" if small else "markers", name="Fahrzeug nicht erreicht",
                                 text=[f"F{k + 1}" for k in other] if small else None, textposition="top center", hovertext=[f"Fahrzeug {k + 1}: von keinem freien Fahrzeug erreicht" for k in other], hoverinfo="text",
                                 marker=dict(symbol="square-open", size=10, color="#999", line=dict(width=2, color="#999"))))
    if reached:
        fig.add_trace(go.Scatter(x=[sc.vehicles[k][0] for k in reached], y=[sc.vehicles[k][1] for k in reached], mode="markers+text" if small else "markers", name="Fahrzeug nach Schicht",
                                 text=[f"F{k + 1}·{dist[k]}" for k in reached] if small else None, textposition="top center",
                                 hovertext=[f"Fahrzeug {k + 1}: Schicht {dist[k]}" for k in reached], hoverinfo="text",
                                 marker=dict(symbol="square", size=12, color=[dist[k] for k in reached], colorscale=C.COLORS["layers"], cmin=0, cmax=top, line=dict(width=1.5, color="#222"))))
    for on, sym, name in ((True, "circle", "Auftrag mit Partner"), (False, "circle-open", "Auftrag frei")):
        idx = [k for k in range(sc.m) if (k in matched_o) == on]
        if idx:
            fig.add_trace(go.Scatter(x=[sc.orders[k][0] for k in idx], y=[sc.orders[k][1] for k in idx], mode="markers+text" if small else "markers", name=name,
                                     text=[f"A{k + 1}" for k in idx] if small else None, textposition="bottom center", hovertext=[f"Auftrag {k + 1}" for k in idx], hoverinfo="text",
                                     marker=dict(symbol=sym, size=10 if on else 11, color=C.COLORS["order"], line=dict(width=2, color=C.COLORS["order"]))))
    if on_path:
        idx = sorted(on_path)
        fig.add_trace(go.Scatter(x=[sc.vehicles[k][0] for k in idx], y=[sc.vehicles[k][1] for k in idx], mode="markers", name="auf einem akzeptierten Weg", hoverinfo="skip",
                                 marker=dict(symbol="circle-open", size=24, color=C.COLORS["add"], line=dict(width=3, color=C.COLORS["add"]))))
    dead = sorted(set(_dead_upto(phase, stage)) - on_path)
    if dead:
        fig.add_trace(go.Scatter(x=[sc.vehicles[k][0] for k in dead], y=[sc.vehicles[k][1] for k in dead], mode="markers", name="Sackgasse", hovertext=[f"Fahrzeug {k + 1}: Sackgasse" for k in dead], hoverinfo="text",
                                 marker=dict(symbol="x", size=13, color=C.COLORS["dead"], line=dict(width=2, color=C.COLORS["dead"]))))
    fig = _map_layout(fig, sc, height)
    fig.update_layout(showlegend=False)          # die Legende würde in schmalen Spalten die Zeichenfläche auf fast null drücken; die Farben stehen in der Erklärung unter der Karte
    return fig


def build_layer_schematic(sc, pairs, phase, stage, height=430):
    """Der Schichtgraph als Schema: Spalte 2d = Fahrzeuge der Schicht d, Spalte 2d+1 = Aufträge, an denen ein Fahrzeug der Schicht d+1 hängt (ganz rechts die freien Aufträge).
    Kanten des Schichtgraphen und Partnerkanten; akzeptierte Wege grün (neu) und rot gestrichelt (frei), Sackgassen violett."""
    dist, L = phase.dist, phase.L
    match_o = {j: i for i, j in pairs}
    cols_v = {d: [i for i in range(sc.n) if dist[i] == d] for d in range(L + 1)}
    cols_o = {}
    for i, j in phase.layer_edges:
        w = match_o.get(j)
        x = 2 * dist[w] - 1 if w is not None else 2 * L + 1
        cols_o.setdefault(x, set()).add(j)
    pos = {}
    for d, vs in cols_v.items():
        for r, i in enumerate(vs):
            pos[("v", i)] = (2 * d, 1 - (r + 0.5) / len(vs))
    for x, js in cols_o.items():
        for r, j in enumerate(sorted(js)):
            pos[("o", j)] = (x, 1 - (r + 0.5) / len(js))
    fig = go.Figure()
    ex, ey = [], []
    for i, j in phase.layer_edges:
        (x0, y0), (x1, y1) = pos[("v", i)], pos[("o", j)]
        ex += [x0, x1, None]
        ey += [y0, y1, None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(80,80,80,0.55)", width=1.3), hoverinfo="skip", name="Schichtkante"))
    mx, my = [], []
    for j, w in match_o.items():
        if ("o", j) in pos and ("v", w) in pos:
            (x0, y0), (x1, y1) = pos[("o", j)], pos[("v", w)]
            mx += [x0, x1, None]
            my += [y0, y1, None]
    fig.add_trace(go.Scatter(x=mx, y=my, mode="lines", line=dict(color=C.COLORS["matched"], width=3), hoverinfo="skip", name="gewählt"))
    for p in phase.paths[:stage]:
        ax, ay, dx, dy = [], [], [], []
        for i, j, kind in p.path:
            if ("v", i) in pos and ("o", j) in pos:
                (x0, y0), (x1, y1) = pos[("v", i)], pos[("o", j)]
                (ax if kind == "add" else dx).extend([x0, x1, None])
                (ay if kind == "add" else dy).extend([y0, y1, None])
        fig.add_trace(go.Scatter(x=dx, y=dy, mode="lines", line=dict(color=C.COLORS["drop"], width=4, dash="dash"), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines", line=dict(color=C.COLORS["add"], width=4), hoverinfo="skip", showlegend=False))
    dead = set(_dead_upto(phase, stage))
    on_path = {v for p in phase.paths[:stage] for v in p.chain_v}
    vx, vy, vt, vc, vs = [], [], [], [], []
    for (kind, k), (x, y) in pos.items():
        if kind == "v":
            vx.append(x)
            vy.append(y)
            vt.append(f"F{k + 1}" if sc.n <= 40 else "")
            vc.append(dist[k])
            vs.append("Fahrzeug %d: Schicht %d%s" % (k + 1, dist[k], " (Sackgasse)" if k in dead else ""))
    fig.add_trace(go.Scatter(x=vx, y=vy, mode="markers+text", text=vt, textposition="middle left", hovertext=vs, hoverinfo="text", name="Fahrzeuge",
                             marker=dict(symbol="square", size=12, color=vc, colorscale=C.COLORS["layers"], cmin=0, cmax=max(L, 1), line=dict(width=1.5, color="#222"))))
    ox, oy, ot, oo = [], [], [], []
    for (kind, k), (x, y) in pos.items():
        if kind == "o":
            ox.append(x)
            oy.append(y)
            ot.append(f"A{k + 1}")
            oo.append(k not in match_o)
    fig.add_trace(go.Scatter(x=ox, y=oy, mode="markers+text", text=ot, textposition="middle right", hoverinfo="text", hovertext=[f"Auftrag {t[1:]}" + (" (frei)" if free else "") for t, free in zip(ot, oo)], name="Aufträge",
                             marker=dict(symbol=["circle-open" if free else "circle" for free in oo], size=11, color=C.COLORS["order"], line=dict(width=2, color=C.COLORS["order"]))))
    if dead:
        idx = [pos[("v", k)] for k in sorted(dead) if ("v", k) in pos]
        fig.add_trace(go.Scatter(x=[p[0] for p in idx], y=[p[1] for p in idx], mode="markers", hoverinfo="skip", name="Sackgasse",
                                 marker=dict(symbol="x", size=13, color=C.COLORS["dead"], line=dict(width=2, color=C.COLORS["dead"]))))
    if on_path:
        idx = [pos[("v", k)] for k in sorted(on_path) if ("v", k) in pos]
        fig.add_trace(go.Scatter(x=[p[0] for p in idx], y=[p[1] for p in idx], mode="markers", hoverinfo="skip", name="Weg",
                                 marker=dict(symbol="circle-open", size=24, color=C.COLORS["add"], line=dict(width=3, color=C.COLORS["add"]))))
    fig.update_xaxes(tickvals=[2 * d for d in range(L + 1)], ticktext=[f"Schicht {d}" for d in range(L + 1)], range=[-0.6, 2 * L + 1.8])
    fig.update_yaxes(visible=False, range=[-0.05, 1.05])
    fig = _base(fig, height)
    fig.update_layout(showlegend=False)
    return fig


def build_phases_bar(phases, final_scanned, k_now=None, height=260):
    """Durchsuchte Kanten je Phase (Breitensuche und Tiefensuche gestapelt), darüber die Zahl der Wege; ganz rechts die letzte Breitensuche (der Beweis)."""
    labels = [f"Phase {k + 1} (Schicht {ph.L})" for k, ph in enumerate(phases)] + ["Beweis"]
    bfs = [ph.bfs_scanned for ph in phases] + [final_scanned]
    dfs = [ph.dfs_scanned for ph in phases] + [0]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=bfs, name="Breitensuche", marker_color=C.COLORS["single"], opacity=0.8))
    fig.add_trace(go.Bar(x=labels, y=dfs, name="Tiefensuche", marker_color=C.COLORS["kuhn"], opacity=0.8, text=[f"{len(ph.paths)} Weg(e)" for ph in phases] + [""], textposition="outside"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category")
    fig.update_yaxes(title="durchsuchte Kanten", rangemode="tozero")
    return _base(fig, height)


def build_scans_hist(scans_by_label, current=None, height=300):
    """Durchsuchte Kanten je Karte, die drei Verfahren übereinandergelegt (die Marke ist Hopcroft–Karp auf Ihrer Ziehung)."""
    colors = [C.COLORS["hk"], C.COLORS["single"], C.COLORS["kuhn"]]
    fig = go.Figure()
    for k, (label, vals) in enumerate(scans_by_label.items()):
        fig.add_trace(go.Histogram(x=vals, name=label, marker_color=colors[k % 3], opacity=0.55))
    fig.update_layout(barmode="overlay")
    if current is not None:
        fig.add_vline(x=current, line=dict(color="#555", dash="dash"), annotation_text="Ihre Ziehung (Hopcroft–Karp)", annotation_position="top")
    fig.update_xaxes(title="durchsuchte Kanten je Karte")
    fig.update_yaxes(title="Karten")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.35), height=height + 50)
    return fig


def build_reach_sweep(rows, current=None, height=420):
    """Oben: durchsuchte Kanten der drei Verfahren (logarithmisch); unten: mittlere Phasenzahl von Hopcroft–Karp."""
    x = [r["x"] for r in rows]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Durchsuchte Kanten (Mittel)", "Phasen von Hopcroft–Karp (Mittel)"))
    for key, name, color in (("hk", "Hopcroft–Karp", C.COLORS["hk"]), ("single", "Einzelne kürzeste Wege", C.COLORS["single"]), ("kuhn", "Kuhn", C.COLORS["kuhn"])):
        fig.add_trace(go.Scatter(x=x, y=[max(r[key], 0.5) if r[key] is not None else None for r in rows], mode="lines+markers", name=name, line=dict(color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["phases"] for r in rows], mode="lines+markers", name="Phasen", line=dict(color=C.COLORS["hk"]), showlegend=False), row=2, col=1)
    if current is not None and min(x) <= current <= max(x):
        fig.add_vline(x=current, line=dict(color="#555", dash="dot"))
    fig.update_xaxes(title="Reichweite [min]", row=2, col=1)
    fig.update_yaxes(title="Kanten", type="log", row=1, col=1)
    fig.update_yaxes(title="Phasen", rangemode="tozero", row=2, col=1)
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def build_scaling(rows, height=340):
    """Durchsuchte Kanten gegen die Kartengröße (doppelt logarithmisch): Hopcroft–Karp, einzelne kürzeste Wege, Kuhn und die Kanten des Graphen."""
    fig = go.Figure()
    ns = [r["n"] for r in rows]
    for key, name, color, dash in (("hk", "Hopcroft–Karp", C.COLORS["hk"], None), ("single", "Einzelne kürzeste Wege", C.COLORS["single"], None), ("kuhn", "Kuhn", C.COLORS["kuhn"], None), ("edges", "Kanten des Graphen", "#555", "dashdot")):
        fig.add_trace(go.Scatter(x=ns, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, dash=dash)))
    fig.update_xaxes(title="Fahrzeuge = Aufträge", type="log")
    fig.update_yaxes(title="durchsuchte Kanten", type="log")
    return _base(fig, height)


def build_phase_growth(rows, stair_rows, height=300):
    """Phasen gegen die Kartengröße (doppelt logarithmisch): Zufallskarten (Mittel), die Worst-Case-Treppe (genau K Phasen bei n = K (K + 3) / 2) und √(2n) als Bezug."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["n"] for r in rows], y=[r["phases"] for r in rows], mode="lines+markers", name="Zufallskarten (Mittel)", line=dict(color=C.COLORS["hk"])))
    fig.add_trace(go.Scatter(x=[r["n"] for r in stair_rows], y=[r["phases"] for r in stair_rows], mode="lines+markers", name="Treppe", line=dict(color=C.COLORS["dead"])))
    ns = sorted({r["n"] for r in rows} | {r["n"] for r in stair_rows})
    fig.add_trace(go.Scatter(x=ns, y=[(2 * n) ** 0.5 for n in ns], mode="lines", name="√(2n)", line=dict(color="#555", dash="dot")))
    fig.update_xaxes(title="Fahrzeuge = Aufträge", type="log")
    fig.update_yaxes(title="Phasen", type="log")
    return _base(fig, height)


def build_stair(rows, height=300):
    """Worst-Case-Treppe: durchsuchte Kanten gegen die Kartengröße (doppelt logarithmisch)."""
    fig = go.Figure()
    for key, name, color in (("hk", "Hopcroft–Karp", C.COLORS["hk"]), ("single", "Einzelne kürzeste Wege", C.COLORS["single"]), ("kuhn", "Kuhn", C.COLORS["kuhn"])):
        fig.add_trace(go.Scatter(x=[r["n"] for r in rows], y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color)))
    fig.update_xaxes(title="Fahrzeuge = Aufträge (n = K (K + 3) / 2)", type="log")
    fig.update_yaxes(title="durchsuchte Kanten", type="log")
    return _base(fig, height)


def build_variants(rows, height=300):
    """Was Frühabbruch und Sackgassen-Markierung sparen: durchsuchte Kanten ohne die Markierung bzw. mit voller Schicht, geteilt durch die des richtigen Verfahrens."""
    labels = [str(r["n"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["no_prune"] / r["hk"] for r in rows], name="ohne Sackgassen-Markierung", marker_color=C.COLORS["dead"]))
    fig.add_trace(go.Bar(x=labels, y=[r["full_layer"] / r["hk"] for r in rows], name="volle Schicht statt Frühabbruch", marker_color=C.COLORS["order"]))
    fig.update_layout(barmode="group")
    fig.add_hline(y=1, line=dict(color="#555", dash="dot"))
    fig.update_xaxes(title="Fahrzeuge = Aufträge", type="category")
    fig.update_yaxes(title="Kanten gegenüber dem Verfahren", rangemode="tozero")
    return _base(fig, height)
