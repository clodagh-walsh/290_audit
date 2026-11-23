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

# ---------------------------------------------------------------- map
Q6_LABEL = {1: "Poor", 2: "Fair", 3: "Good", 4: "Under repair"}
STYLE = {   # ordinal blue ramp (steps 250 / 450 / 650): darker = better; orange and gray are outside the scale
    "Good": "#104281", "Fair": "#2a78d6", "Poor": "#86b6ef",
    "Under repair": "#eb6834", "No sidewalk": "#898781", "No usable rating": "#c3c2b7"}
WORST_FIRST = ["No sidewalk", "Poor", "Under repair", "Fair", "Good"]

def state(r):
    if r["no_sidewalk"] == 1: return "No sidewalk"
    return Q6_LABEL.get(r["q6_condition"]) if pd.notna(r["q6_condition"]) else None

def summarize(ratings):
    """One row per segment and method: the shown condition, whether raters disagree, and who said what."""
    rows = []
    for (seg, method), g in ratings.groupby(["segment", "method"]):
        who = [(r.rater_id + (f" (v{r.visit})" if method == "field" else ""), state(r)) for _, r in g.iterrows()]
        known = sorted({s for _, s in who if s}, key=WORST_FIRST.index)
        rows.append(dict(segment=seg, method=method, shown=known[0] if known else "No usable rating",
                         disagree=len(known) > 1, n=len(who),
                         detail="; ".join(f"{w}: {s or 'can’t tell'}" for w, s in who)))
    return pd.DataFrame(rows)

def build_map(ratings, lines, area, layers=("field", "gsv")):
    import folium
    summ = summarize(ratings)
    pos = summ.set_index(["segment", "method"])
    allpts = np.array([p for l in lines.values() for p in l])
    m = folium.Map(location=allpts.mean(0).tolist(), zoom_start=16, tiles=None, control_scale=True)
    # Esri light gray canvas (free, no key; Carto's tiles now need an API key). Native tiles stop at zoom 16.
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    attr = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors"
    folium.TileLayer(esri.format("Base"), attr=attr, name="Base map", control=False,
                     max_native_zoom=16, max_zoom=20).add_to(m)
    folium.TileLayer(esri.format("Reference"), attr=attr, name="Street names", control=False, overlay=True,
                     max_native_zoom=16, max_zoom=20).add_to(m)
    names = {"field": "Field ratings", "gsv": "Street View ratings", "both": "Field vs Street View"}
    for i, method in enumerate(layers + ("both",)):
        fg = folium.FeatureGroup(name=names[method], overlay=False, show=(i == 0))
        for seg, line in lines.items():
            if method == "both":
                f = pos.loc[(seg, "field")] if (seg, "field") in pos.index else None
                s = pos.loc[(seg, "gsv")] if (seg, "gsv") in pos.index else None
                if f is None and s is None: continue
                shown = f.shown if f is not None else "No usable rating"
                dis = f is not None and s is not None and f.shown != s.shown
                tip = f"Field {f.shown if f is not None else '–'} · Street View {s.shown if s is not None else '–'}"
                pop = (f"<b>Field:</b> {f.detail if f is not None else 'none'}<br>"
                       f"<b>Street View:</b> {s.detail if s is not None else 'none'}")
            else:
                if (seg, method) not in pos.index: continue
                r = pos.loc[(seg, method)]
                shown, dis, tip, pop = r.shown, r.disagree, r.shown, r.detail.replace("; ", "<br>")
            label = f"<b>{seg}</b> · {tip}" + (" · <i>raters disagree</i>" if dis else "")
            folium.PolyLine(line, color=STYLE[shown], weight=5, opacity=1,
                            dash_array="2 8" if dis else None, line_cap="round").add_to(fg)
            if dis:   # a ring at the middle so disagreements are findable at any zoom
                folium.CircleMarker(line[len(line) // 2], radius=5, color="#0b0b0b", weight=2,
                                    fill=True, fill_color="#ffffff", fill_opacity=1).add_to(fg)
            # a wide invisible line on top carries the hover and click, so thin marks are easy to hit
            folium.PolyLine(line, weight=18, opacity=0, tooltip=folium.Tooltip(label, sticky=True),
                            popup=folium.Popup(f"<b>{seg}</b><br>{pop}", max_width=320)).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    DASH = ' stroke-dasharray="2 8"'
    sw = lambda c, dash=False: (f'<svg width="28" height="10"><line x1="3" y1="5" x2="25" y2="5" stroke="{c}" '
                                f'stroke-width="5" stroke-linecap="round"{DASH if dash else ""}/></svg>')
    rowsh = "".join(f'<div>{sw(c)} {k}</div>' for k, c in STYLE.items())
    legend = f"""
    <div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:#fcfcfb;color:#0b0b0b;
         padding:10px 12px;border-radius:8px;box-shadow:0 0 0 1px rgba(11,11,11,.10),0 2px 6px rgba(0,0,0,.12);
         font:12px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif">
      <div style="font-weight:600;margin-bottom:2px">{area} · Q6 sidewalk condition</div>{rowsh}
      <div style="margin-top:4px;color:#52514e">{sw('#2a78d6', True)} raters disagree
        <svg width="14" height="12"><circle cx="7" cy="6" r="4.5" fill="#fff" stroke="#0b0b0b" stroke-width="2"/></svg></div>
      <div style="color:#52514e;max-width:230px">A disagreement is drawn at the poorer answer. Hover or click a block for each rating.</div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m, summ
