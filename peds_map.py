"""Segment geometry and an interactive sidewalk-condition map for the PEDS audit.

Segment lines are built from OpenStreetMap street centerlines: for each segment, find the two
intersections named in streetview.html, follow the street between them, trim the corners and
shift the line toward the rated side of the street. The OSM pull is cached next to this file.
"""
import heapq, json, math, os, re, urllib.parse, urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BBOX = {"Chinatown": (37.7940, -122.2790, 37.8070, -122.2600),   # south, west, north, east
        "Rockridge": (37.8380, -122.2650, 37.8580, -122.2420)}
AREA_ID = {"Chinatown": "chinatown", "Rockridge": "rockridge"}
SIDE_OFFSET_M, TRIM_M = 9.0, 9.0

# ---------------------------------------------------------------- segment list (from the rating tool)
def tool_segments(area, html=os.path.join(HERE, "streetview.html")):
    s = open(html, encoding="utf-8").read()
    d = json.loads(re.search(r"const DATA = (\{.*?\});\n", s, re.S).group(1))
    routes = {r["id"] for r in d["routes"] if r.get("area") == AREA_ID[area]}
    rows = []
    for x in d["segments"]:
        if x["route"] not in routes: continue
        xa, xb = (x.get("xa"), x.get("xb")) if x.get("xa") else [p.strip() for p in x["between"].split("–")]
        rows.append(dict(code=x["code"], street=x["street"], side=x["side"], xa=xa, xb=xb, route=x["route"]))
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- OSM streets
SUFFIX = {"st", "street", "ave", "avenue", "rd", "road", "way", "blvd", "boulevard", "dr", "drive", "pl", "place"}
def base(name):
    toks = name.lower().replace(".", "").split()
    while len(toks) > 1 and toks[-1] in SUFFIX: toks = toks[:-1]
    return " ".join(toks)

def osm_streets(area, refresh=False):
    path = os.path.join(HERE, f"osm_streets_{area.lower()}.json")
    if refresh or not os.path.exists(path):
        s, w, n, e = BBOX[area]
        q = (f'[out:json][timeout:60];way["highway"~"primary|secondary|tertiary|residential|unclassified|living_street"]'
             f'["name"]({s},{w},{n},{e});out geom;')
        req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                     data=urllib.parse.urlencode({"data": q}).encode(),
                                     headers={"User-Agent": "peds-audit-map/0.1", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            open(path, "wb").write(r.read())
    return json.load(open(path))["elements"]

class StreetNet:
    def __init__(self, ways):
        self.xy, self.adj, self.by_name = {}, defaultdict(lambda: defaultdict(dict)), defaultdict(set)
        lat0 = np.mean([g["lat"] for w in ways for g in w["geometry"]])
        self.kx, self.ky = 111320 * math.cos(math.radians(lat0)), 110540
        for w in ways:
            nm = w["tags"]["name"]
            if "tube" in nm.lower() or nm.lower().endswith((" place", " pl")): continue   # Webster St Tube, Webster Place
            b = base(nm)
            for nid, g in zip(w["nodes"], w["geometry"]):
                self.xy[nid] = (g["lon"], g["lat"]); self.by_name[b].add(nid)
            for u, v in zip(w["nodes"], w["nodes"][1:]):
                d = self.dist(u, v); self.adj[b][u][v] = d; self.adj[b][v][u] = d

    def m(self, nid):
        lon, lat = self.xy[nid]; return lon * self.kx, lat * self.ky
    def dist(self, u, v):
        (x1, y1), (x2, y2) = self.m(u), self.m(v); return math.hypot(x2 - x1, y2 - y1)

    def crossing(self, a, b):
        shared = self.by_name[a] & self.by_name[b]
        if shared: return list(shared)
        best = min(((self.dist(u, v), u) for u in self.by_name[a] for v in self.by_name[b]), default=(1e9, None))
        return [best[1]] if best[0] < 25 else []

    def path(self, street, xa, xb):
        """Shortest path along `street` between its crossings with xa and xb (node ids)."""
        A, B = self.crossing(street, xa), self.crossing(street, xb)
        if not A or not B: return None
        g, best = self.adj[street], None
        for s in A:
            dist, prev, pq = {s: 0}, {}, [(0, s)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, 1e18): continue
                for v, w in g[u].items():
                    if d + w < dist.get(v, 1e18): dist[v] = d + w; prev[v] = u; heapq.heappush(pq, (d + w, v))
            for t in B:
                if t in dist and (best is None or dist[t] < best[0]):
                    p = [t]
                    while p[-1] != s: p.append(prev[p[-1]])
                    best = (dist[t], p[::-1])
        return best[1] if best else None

    def side_line(self, nodes, side):
        pts = np.array([self.m(n) for n in nodes], float)
        # trim the corners
        seg = np.hypot(*np.diff(pts, axis=0).T); cum = np.r_[0, np.cumsum(seg)]; L = cum[-1]
        if L < 3 * TRIM_M: return None
        keep = np.linspace(TRIM_M, L - TRIM_M, max(2, int(L // 10)))
        pts = np.c_[np.interp(keep, cum, pts[:, 0]), np.interp(keep, cum, pts[:, 1])]
        chord = pts[-1] - pts[0]; chord /= np.linalg.norm(chord)
        normal = np.array([-chord[1], chord[0]])
        want = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}[side]
        if normal @ want < 0: normal = -normal
        pts = pts + normal * SIDE_OFFSET_M
        return [(y / self.ky, x / self.kx) for x, y in pts]      # (lat, lon)

def segment_lines(area, final_code=lambda c: c):
    """{final segment code: [(lat, lon), ...]}. Combined half-blocks are joined end to end."""
    net = StreetNet(osm_streets(area))
    segs = tool_segments(area)
    segs["final"] = segs.code.map(final_code)
    out, missing = {}, []
    for code, g in segs.groupby("final"):
        r = g.iloc[0]; st = base(r.street)
        crossings = sorted({base(x) for x in pd.concat([g.xa, g.xb])})
        # a combined block lists three crossings (9th, 10th, 11th): use the pair farthest apart
        paths = [net.path(st, a, b) for i, a in enumerate(crossings) for b in crossings[i + 1:]]
        paths = [p for p in paths if p]
        best = max(paths, key=lambda p: net.dist(p[0], p[-1]), default=None)
        line = net.side_line(best, r.side) if best else None
        if line: out[code] = line
        else: missing.append(code)
    return out, missing


# ---------------------------------------------------------------- measures shown on the map
# Each measure turns one rating into a value, merges several raters' values worst-case, and maps
# the merged value to a legend category. "No sidewalk" outranks every value.
NS = "No sidewalk"
NONE_RATED = "No usable rating"
OUTSIDE = {NS: "#898781", NONE_RATED: "#c3c2b7"}   # gray, outside every color scale

Q6_LABEL = {1: "Poor", 2: "Fair", 3: "Good", 4: "Under repair"}
Q6_WORST_FIRST = ["Poor", "Under repair", "Fair", "Good"]
Q7_BOXES = [("q7_poles_signs", "poles/signs"), ("q7_parked_cars", "parked cars"), ("q7_greenery", "greenery"),
            ("q7_garbage_cans", "garbage cans"), ("q7_other", "other")]
Q11_LABEL = {1: "None", 2: "1 to 4", 3: "More than 4"}

def _q6(r):
    return Q6_LABEL.get(r["q6_condition"]) if pd.notna(r["q6_condition"]) else None

def _q7(r):
    ticked = [lab for col, lab in Q7_BOXES if r.get(col) == 1]
    if ticked: return frozenset(ticked)
    return frozenset() if r.get("q7_none") == 1 else None          # nothing ticked at all: can't tell

def _q7_cat(v):
    return "None" if len(v) == 0 else "1 type" if len(v) == 1 else "2 or more types"

def _q7_text(v):
    return ", ".join(lab for _, lab in Q7_BOXES if lab in v) or "none"

def _q11(r):
    return int(r["q11_curb_cuts"]) if pd.notna(r["q11_curb_cuts"]) else None

MEASURES = {
    "condition": dict(
        name="Sidewalk condition", title="Q6 sidewalk condition",
        value=_q6, merge=lambda vs: min(vs, key=Q6_WORST_FIRST.index),
        category=lambda v: v, text=lambda v: v,
        colors={"Good": "#104281", "Fair": "#2a78d6", "Poor": "#86b6ef", "Under repair": "#eb6834"},  # darker = better
        note="A disagreement is drawn at the poorer answer."),
    "obstructions": dict(
        name="Path obstructions", title="Q7 path obstructions",
        value=_q7, merge=lambda vs: frozenset().union(*vs),
        category=_q7_cat, text=_q7_text,
        colors={"None": "#c9b6ec", "1 type": "#8a63d2", "2 or more types": "#4b2391"},           # darker = more
        note="Raters disagree when they ticked different boxes. The block is drawn with every type any rater ticked."),
    "curb_cuts": dict(
        name="Curb cuts", title="Q11 curb cuts (driveways)",
        value=_q11, merge=max,
        category=lambda v: Q11_LABEL[v], text=lambda v: Q11_LABEL[v],
        colors={"None": "#8fd3c6", "1 to 4": "#2a9d8f", "More than 4": "#0c5a50"},                # darker = more
        note="A disagreement is drawn at the higher count."),
}

def summarize(ratings, measure="condition"):
    """One row per segment and method: the merged value, the category shown, whether raters disagree, who said what."""
    M = MEASURES[measure]
    rows = []
    for (seg, method), g in ratings.groupby(["segment", "method"]):
        who = []
        for _, r in g.iterrows():
            v = NS if r["no_sidewalk"] == 1 else M["value"](r)
            who.append((r.rater_id + (f" (v{r.visit})" if method == "field" else ""), v))
        vals = [v for _, v in who if v is not None]
        if not vals: val = None
        elif NS in vals: val = NS
        else: val = M["merge"](vals)
        shown = NONE_RATED if val is None else NS if val == NS else M["category"](val)
        rows.append(dict(segment=seg, method=method, value=val, shown=shown, disagree=len(set(vals)) > 1, n=len(who),
                         detail="; ".join(f"{w}: {NS if v == NS else M['text'](v) if v is not None else 'can’t tell'}"
                                          for w, v in who)))
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- map
METHOD_NAME = {"field": "Field ratings", "gsv": "Street View ratings", "both": "Field vs Street View"}

def _add_lines(fg, lines, pos, M, method):
    import folium
    style = {**M["colors"], **OUTSIDE}
    txt = lambda r: NS if r.value == NS else M["text"](r.value) if r.value is not None else "–"
    for seg, line in lines.items():
        if method == "both":
            f = pos.loc[(seg, "field")] if (seg, "field") in pos.index else None
            s = pos.loc[(seg, "gsv")] if (seg, "gsv") in pos.index else None
            if f is None and s is None: continue
            shown = f.shown if f is not None else NONE_RATED
            dis = f is not None and s is not None and f.value is not None and s.value is not None and f.value != s.value
            tip = f"Field {txt(f) if f is not None else '–'} · Street View {txt(s) if s is not None else '–'}"
            pop = (f"<b>Field:</b> {f.detail if f is not None else 'none'}<br>"
                   f"<b>Street View:</b> {s.detail if s is not None else 'none'}")
        else:
            if (seg, method) not in pos.index: continue
            r = pos.loc[(seg, method)]
            shown, dis, tip, pop = r.shown, r.disagree, txt(r), r.detail.replace("; ", "<br>")
        label = f"<b>{seg}</b> · {M['name']}: {tip}" + (" · <i>raters disagree</i>" if dis else "")
        folium.PolyLine(line, color=style[shown], weight=5, opacity=1,
                        dash_array="2 8" if dis else None, line_cap="round").add_to(fg)
        if dis:   # a ring at the middle so disagreements are findable at any zoom
            folium.CircleMarker(line[len(line) // 2], radius=5, color="#0b0b0b", weight=2,
                                fill=True, fill_color="#ffffff", fill_opacity=1).add_to(fg)
        # a wide invisible line on top carries the hover and click, so thin marks are easy to hit
        folium.PolyLine(line, weight=18, opacity=0, tooltip=folium.Tooltip(label, sticky=True),
                        popup=folium.Popup(f"<b>{seg}</b> · {M['title']}<br>{pop}", max_width=320)).add_to(fg)

def build_map(ratings, lines, area, layers=("field", "gsv"), measures=tuple(MEASURES)):
    """Interactive map with a Measure and a Ratings picker. Returns (map, condition summary table)."""
    import folium
    allpts = np.array([p for l in lines.values() for p in l])
    m = folium.Map(location=allpts.mean(0).tolist(), zoom_start=16, tiles=None, control_scale=True)
    # Esri light gray canvas (free, no key; Carto's tiles now need an API key). Native tiles stop at zoom 16.
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    attr = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors"
    folium.TileLayer(esri.format("Base"), attr=attr, name="Base map", control=False,
                     max_native_zoom=16, max_zoom=20).add_to(m)
    folium.TileLayer(esri.format("Reference"), attr=attr, name="Street names", control=False, overlay=True,
                     max_native_zoom=16, max_zoom=20).add_to(m)

    methods = tuple(layers) + ("both",)
    groups, summaries = {}, {}
    for mi, key in enumerate(measures):
        M = MEASURES[key]
        summaries[key] = summarize(ratings, key)
        pos = summaries[key].set_index(["segment", "method"])
        for i, method in enumerate(methods):
            fg = folium.FeatureGroup(name=f"{M['name']} · {METHOD_NAME[method]}", show=(mi == 0 and i == 0))
            _add_lines(fg, lines, pos, M, method)
            fg.add_to(m)
            groups[f"{key}|{method}"] = fg.get_name()

    # ---- legend, one block per measure (only the selected one is visible)
    DASH = ' stroke-dasharray="2 8"'
    sw = lambda c, dash=False: (f'<svg width="28" height="10"><line x1="3" y1="5" x2="25" y2="5" stroke="{c}" '
                                f'stroke-width="5" stroke-linecap="round"{DASH if dash else ""}/></svg>')
    ring = ('<svg width="14" height="12"><circle cx="7" cy="6" r="4.5" fill="#fff" stroke="#0b0b0b" '
            'stroke-width="2"/></svg>')
    blocks = []
    for i, key in enumerate(measures):
        M = MEASURES[key]
        rows = "".join(f"<div>{sw(c)} {k}</div>" for k, c in {**M["colors"], **OUTSIDE}.items())
        mid = list(M["colors"].values())[1]
        blocks.append(f"""<div class="peds-legend" data-measure="{key}" style="display:{'block' if i == 0 else 'none'}">
          <div style="font-weight:600;margin-bottom:2px">{area} · {M['title']}</div>{rows}
          <div style="margin-top:4px;color:#52514e">{sw(mid, True)} raters disagree {ring}</div>
          <div style="color:#52514e;max-width:240px">{M['note']} Hover or click a block for each rating.</div></div>""")
    box = ("background:#fcfcfb;color:#0b0b0b;padding:10px 12px;border-radius:8px;"
           "box-shadow:0 0 0 1px rgba(11,11,11,.10),0 2px 6px rgba(0,0,0,.12);"
           "font:12px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif")
    sel = "display:block;width:100%;margin:2px 0 8px;font:inherit;padding:3px 4px"
    m_opts = "".join(f'<option value="{k}">{MEASURES[k]["name"]}</option>' for k in measures)
    r_opts = "".join(f'<option value="{k}">{METHOD_NAME[k]}</option>' for k in methods)
    html = f"""
    <div style="position:fixed;top:12px;right:12px;z-index:9999;min-width:200px;{box}">
      <label style="font-weight:600">Measure<select id="peds-measure" style="{sel}">{m_opts}</select></label>
      <label style="font-weight:600">Ratings<select id="peds-method" style="{sel};margin-bottom:0">{r_opts}</select></label>
    </div>
    <div style="position:fixed;bottom:24px;left:12px;z-index:9999;{box}">{''.join(blocks)}</div>"""
    m.get_root().html.add_child(folium.Element(html))

    # folium declares the map and groups as globals in an inline script, so they exist by the load event
    groups_js = "{" + ",".join(f'"{k}": {v}' for k, v in groups.items()) + "}"
    js = f"""
    window.addEventListener("load", function () {{
      var map = {m.get_name()}, groups = {groups_js};
      var ms = document.getElementById("peds-measure"), rs = document.getElementById("peds-method");
      function update() {{
        var want = ms.value + "|" + rs.value;
        Object.keys(groups).forEach(function (k) {{
          if (k === want) {{ if (!map.hasLayer(groups[k])) map.addLayer(groups[k]); }}
          else if (map.hasLayer(groups[k])) map.removeLayer(groups[k]);
        }});
        map.closePopup();
        document.querySelectorAll(".peds-legend").forEach(function (el) {{
          el.style.display = el.getAttribute("data-measure") === ms.value ? "block" : "none";
        }});
      }}
      ms.addEventListener("change", update); rs.addEventListener("change", update); update();
    }});"""
    m.get_root().html.add_child(folium.Element(f"<script>{js}</script>"))
    return m, summaries.get("condition", next(iter(summaries.values())))
