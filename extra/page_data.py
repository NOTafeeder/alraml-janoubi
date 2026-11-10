"""Turns the files in data_extra/ into one JSON object (numbers + small PNG/SVG images) for the page."""
import base64, io, json, math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data_extra"
BBOX = [35.776, 35.495, 35.825, 35.520]


def png(rgb):
    buf = io.BytesIO(); Image.fromarray(rgb.astype(np.uint8)).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ramp(v, stops, vmin, vmax):
    stops = np.array(stops, float)
    t = np.clip((np.nan_to_num(v, nan=vmin) - vmin) / (vmax - vmin), 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


SEA = [40, 70, 140]
HEAT = [[49, 54, 149], [116, 173, 209], [255, 255, 191], [244, 109, 67], [165, 0, 38]]
YLOR = [[255, 247, 214], [253, 196, 112], [236, 112, 52], [180, 40, 40], [100, 10, 40]]


def turbidity(arr):
    """arr: Sentinel-2 bands B02,B03,B04,B08 (x10000). Water coloured by red reflectance."""
    red, nir = arr[2] / 1e4, arr[3] / 1e4
    ndvi = (nir - red) / (nir + red + 1e-9)
    water = (arr[3] < 600) | (ndvi < -0.05)
    grey = np.clip(arr[[2, 1, 0]].mean(0) / 3000, 0, 1) * 150 + 60
    rgb = np.repeat(grey[..., None], 3, -1)
    rgb[water] = ramp(red[water], [[20, 40, 110], [40, 120, 170], [120, 190, 160], [230, 200, 90], [170, 100, 40]], 0.0, 0.05)
    coastal = water.copy()
    return rgb, float(np.mean(red[water])), float(np.percentile(red[water], 95))


def elevation():
    with rasterio.open(D / "dem.tif") as s:
        z = s.read(1).astype(float)
        lat_mid = (s.bounds.top + s.bounds.bottom) / 2
        px = abs(s.transform.a) * 111320 * math.cos(math.radians(lat_mid)) * abs(s.transform.e) * 110570
    land = z > 0.3
    rgb = np.zeros(z.shape + (3,))
    classes = [(0.3, 2, [215, 48, 39]), (2, 5, [252, 141, 89]), (5, 10, [254, 224, 139]),
               (10, 20, [217, 239, 139]), (20, 40, [145, 207, 96]), (40, 999, [26, 152, 80])]
    for lo, hi, c in classes:
        rgb[(z > lo) & (z <= hi)] = c
    rgb[~land] = SEA
    a = land.sum() * px / 1e6
    share = lambda h: round(100 * ((z > 0.3) & (z <= h)).sum() / land.sum(), 1)
    return {"img": png(rgb), "w": z.shape[1], "h": z.shape[0], "land_km2": round(a, 2),
            "below2": share(2), "below5": share(5), "below10": share(10),
            "median": round(float(np.median(z[land])), 1), "max": round(float(z[land].max()), 1)}


def rain():
    d = json.loads((D / "rain_daily.json").read_text())
    t = np.array(d["time"], dtype="datetime64[D]")
    mm = np.array([np.nan if v is None else v for v in d["mm"]], float)
    yr = t.astype("datetime64[Y]").astype(int) + 1970
    mo = t.astype("datetime64[M]").astype(int) % 12 + 1
    hyd = np.where(mo >= 9, yr + 1, yr)            # rain year Sep-Aug, labelled by the year it ends
    years = []
    for y in range(1941, int(hyd.max()) + 1):
        sel = hyd == y
        complete = sel.sum() >= 365
        years.append({"y": y, "mm": round(float(np.nansum(mm[sel]))), "complete": bool(complete)})
    norm = (hyd >= 1991) & (hyd <= 2020)
    normal = float(np.mean([v["mm"] for v in years if 1991 <= v["y"] <= 2020]))
    prob, mmonth = [], []
    for m in range(1, 13):
        s = norm & (mo == m)
        prob.append(round(100 * float(np.mean(mm[s] >= 1)), 0))
        mmonth.append(round(float(np.nansum(mm[s])) / 30))
    full = [v for v in years if v["complete"]]
    dry = sorted(full, key=lambda v: v["mm"])[:3]
    last10 = float(np.mean([v["mm"] for v in full[-10:]]))
    return {"years": years, "normal": round(normal), "prob": prob, "mm_month": mmonth,
            "driest": dry, "last10": round(last10), "cell": [d["lat"], d["lon"]],
            "first": d["time"][0], "last": d["time"][-1]}


def lst():
    scenes = json.loads((D / "lst_scenes.json").read_text())
    out = []
    for s in scenes:
        z = np.load(D / f"lst_{s['year']}.npz")
        T, ndvi, clear = z["lst"], z["ndvi"], z["clear"]
        ok = clear & np.isfinite(T)
        water = ok & (ndvi < 0)
        built = ok & (ndvi >= 0) & (ndvi < 0.2)
        veg = ok & (ndvi > 0.4)
        rgb = ramp(T, HEAT, 25, 55)
        rgb[~ok] = [200, 200, 200]
        out.append({**s, "built": round(float(T[built].mean()), 1), "veg": round(float(T[veg].mean()), 1),
                    "sea": round(float(T[water].mean()), 1), "img": png(rgb)})
    return {"years": out, "w": T.shape[1], "h": T.shape[0], "vmin": 25, "vmax": 55}


def built_pop():
    s = json.loads((D / "built_pop.json").read_text())
    epochs = [b["year"] for b in s["built"]]
    stack = []
    for e in epochs:
        with rasterio.open(D / f"ghsl_{e}.tif") as src:
            stack.append(src.read(1))
            cell = abs(src.transform.a) * 111320 * math.cos(math.radians(35.5)) * abs(src.transform.e) * 110570
    stack = np.array(stack)
    frac = stack / cell
    built = frac >= 0.2
    first = np.full(built.shape[1:], 0)
    for i in range(len(epochs) - 1, -1, -1):
        first[built[i]] = epochs[i]
    cols = {1975: [90, 60, 40], 1985: [140, 90, 50], 1995: [200, 130, 60], 2005: [235, 175, 70],
            2015: [250, 215, 90], 2020: [255, 240, 140], 2025: [255, 255, 200]}
    rgb = np.full(first.shape + (3,), 235.0)
    for e, c in cols.items():
        rgb[first == e] = c
    pops = []
    for p in s["pop"]:
        with rasterio.open(D / f"pop_{p['year']}.tif") as src:
            a = src.read(1)
            pc = abs(src.transform.a) * 111320 * math.cos(math.radians(35.5)) * abs(src.transform.e) * 110570
        pops.append({**p, "max_density": round(float(np.nanmax(a)) / pc * 1e6)})
        last = a
    dens = last / pc * 1e6
    prgb = ramp(dens, YLOR, 0, float(np.nanpercentile(dens, 99)))
    prgb[~np.isfinite(dens) | (dens <= 0)] = [235, 235, 235]
    return {"built": s["built"], "built_img": png(rgb), "first_colors": cols,
            "pop": pops, "pop_img": png(prgb), "pop_vmax": round(float(np.nanpercentile(dens, 99)), -2),
            "w": first.shape[1], "h": first.shape[0]}


def clean_name(t):
    """Arabic name if there is one; drops the Latin half of bilingual names like 'X School مدرسة X'."""
    import re
    n = (t.get("name:ar") or t.get("name") or "").strip()
    if re.search(r"[؀-ۿ]", n) and re.search(r"[A-Za-z]", n):
        ar = re.sub(r"[A-Za-z_'’\-]+", " ", n)
        ar = re.sub(r"\s+", " ", ar).strip(" /-")
        if ar:
            n = ar
    return re.sub(r"\s+", " ", n)


def osm():
    d = json.loads((D / "osm.json").read_text())
    W = 800
    kx = W / (BBOX[2] - BBOX[0])
    ky = kx / math.cos(math.radians(35.5))
    H = round((BBOX[3] - BBOX[1]) * ky)
    P = lambda lo, la: f"{(lo - BBOX[0]) * kx:.1f},{(BBOX[3] - la) * ky:.1f}"
    bld, rd, pts, counts = [], [], [], {}
    cat = {"school": "edu", "kindergarten": "edu", "university": "edu", "college": "edu",
           "hospital": "health", "clinic": "health", "doctors": "health", "pharmacy": "health",
           "place_of_worship": "worship", "fire_station": "safety", "police": "safety",
           "park": "green", "playground": "green", "pitch": "green"}
    for e in d["elements"]:
        t = e.get("tags", {})
        g = e.get("geometry")
        if "building" in t and g:
            bld.append("M" + "L".join(P(p["lon"], p["lat"]) for p in g) + "Z")
            counts["building"] = counts.get("building", 0) + 1
        elif "highway" in t and g:
            w = 2.2 if t["highway"] in ("motorway", "trunk", "primary", "secondary") else 1
            rd.append((w, "M" + "L".join(P(p["lon"], p["lat"]) for p in g)))
            counts["road"] = counts.get("road", 0) + 1
        k = t.get("amenity") or t.get("leisure")
        if k in cat:
            if "lat" in e:
                la, lo = e["lat"], e["lon"]
            elif "center" in e:
                la, lo = e["center"]["lat"], e["center"]["lon"]
            elif "bounds" in e:
                b = e["bounds"]; la, lo = (b["minlat"] + b["maxlat"]) / 2, (b["minlon"] + b["maxlon"]) / 2
            elif g:
                la, lo = np.mean([p["lat"] for p in g]), np.mean([p["lon"] for p in g])
            else:
                continue
            if not (BBOX[1] <= la <= BBOX[3] and BBOX[0] <= lo <= BBOX[2]):
                continue
            x, y = P(lo, la).split(",")
            name = clean_name(t)
            # the same place is often mapped twice (a point and a building outline): keep one
            dup = any(q["k"] == k and q["n"] == name and name and
                      math.hypot(q["lat"] - la, (q["lon"] - lo) * math.cos(math.radians(la))) * 111320 < 150
                      for q in pts)
            if dup:
                continue
            pts.append({"x": float(x), "y": float(y), "c": cat[k], "k": k, "n": name,
                        "lat": round(la, 6), "lon": round(lo, 6), "osm": f'{e["type"]}/{e["id"]}'})
            counts[k] = counts.get(k, 0) + 1
    # names given by residents (data_extra/local_names.json), keyed by OpenStreetMap id
    ln = D / "local_names.json"
    if ln.exists():
        local = {k: v for k, v in json.loads(ln.read_text(encoding="utf-8")).items() if not k.startswith("_")}
        for p in pts:
            if p["osm"] in local and not p["n"]:
                p["n"] = local[p["osm"]]["name"]
                p["local"] = local[p["osm"]].get("source", "معلومة محلية")

    # for unnamed places: the nearest named streets (OSM has no neighbourhood boundaries here)
    segs = []
    for e in d["elements"]:
        t = e.get("tags", {})
        nm = clean_name(t) if "highway" in t else ""
        if nm and e.get("geometry"):
            g = e["geometry"]
            segs += [(nm, a["lat"], a["lon"], b["lat"], b["lon"]) for a, b in zip(g, g[1:])]

    def seg_dist(la, lo, s):
        kx = 111320 * math.cos(math.radians(la)); ky = 110570
        ax, ay, bx, by = (s[2] - lo) * kx, (s[1] - la) * ky, (s[4] - lo) * kx, (s[3] - la) * ky
        L = ax * ax + ay * ay if False else (bx - ax) ** 2 + (by - ay) ** 2
        u = 0 if L == 0 else max(0, min(1, -(ax * (bx - ax) + ay * (by - ay)) / L))
        return math.hypot(ax + u * (bx - ax), ay + u * (by - ay))

    for p in pts:
        if p["n"] or not segs:
            continue
        best = {}
        for s in segs:
            dd = seg_dist(p["lat"], p["lon"], s)
            if dd < best.get(s[0], 1e9):
                best[s[0]] = dd
        near = sorted(best.items(), key=lambda kv: kv[1])[:2]
        p["near"] = [{"street": n, "m": int(round(m, -1))} for n, m in near if m < 400]
        # nearest named facility as a landmark
        kx = 111320 * math.cos(math.radians(p["lat"]))
        lm = sorted(((math.hypot((q["lon"] - p["lon"]) * kx, (q["lat"] - p["lat"]) * 110570), q["n"], q["k"])
                     for q in pts if q["n"]), key=lambda v: v[0])
        if lm and lm[0][0] < 400:
            p["landmark"] = {"name": lm[0][1], "m": int(round(lm[0][0], -1)), "same_kind": lm[0][2] == p["k"]}

    svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" '
           f'aria-label="مباني وطرق ومرافق من OpenStreetMap">'
           f'<rect width="{W}" height="{H}" fill="var(--osm-bg)"/>'
           f'<path d="{"".join(bld)}" fill="var(--osm-bld)" stroke="none"/>'
           + "".join(f'<path d="{p}" fill="none" stroke="var(--osm-road)" stroke-width="{w}"/>' for w, p in rd)
           + "</svg>")
    return {"svg": svg, "w": W, "h": H, "points": pts, "counts": counts, "timestamp": d.get("timestamp")}


def nightlights():
    z = np.load(D / "nightlights.npz")
    meta = json.loads((D / "nightlights.json").read_text())
    maps, lat, lon = z["maps"], z["lat"], z["lon"]
    vmax = float(np.nanpercentile(maps, 99.5))
    # mean over the study area, recomputed here so it follows BBOX
    rr = (lat >= BBOX[1]) & (lat <= BBOX[3]); cc = (lon >= BBOX[0]) & (lon <= BBOX[2])
    for m, y in zip(maps, meta["years"]):
        y["mean_area"] = round(float(np.nanmean(m[np.ix_(rr, cc)])), 2)
    out = []
    for m, y in zip(maps, meta["years"]):
        v = np.log1p(np.clip(np.nan_to_num(m, nan=0), 0, None)) / np.log1p(vmax)
        rgb = ramp(v, [[5, 5, 20], [40, 30, 90], [180, 80, 60], [250, 200, 80], [255, 255, 220]], 0, 1)
        rgb[~np.isfinite(m)] = [90, 90, 90]
        out.append({**y, "img": png(rgb)})
    north, south = lat[0] + 10 / 2400 / 2, lat[-1] - 10 / 2400 / 2
    west, east = lon[0] - 10 / 2400 / 2, lon[-1] + 10 / 2400 / 2
    pos = lambda la, lo: {"x": (lo - west) / (east - west), "y": (north - la) / (north - south)}
    return {"years": out, "w": len(lon), "h": len(lat), "vmax": round(vmax), "mark": pos(35.507, 35.800),
            "bbox": [round(west, 2), round(south, 2), round(east, 2), round(north, 2)]}


def build(s2_tifs):
    turb = {}
    for tif in s2_tifs:
        with rasterio.open(tif) as s:
            rgb, mean_red, p95 = turbidity(s.read().astype(float))
        turb[int(tif.parent.name)] = {"img": png(rgb), "mean": round(mean_red, 4), "p95": round(p95, 4)}
    parts = {"turbidity": turb}
    for name, fn, need in [("elev", elevation, "dem.tif"), ("rain", rain, "rain_daily.json"),
                           ("lst", lst, "lst_scenes.json"), ("bp", built_pop, "built_pop.json"),
                           ("osm", osm, "osm.json"), ("night", nightlights, "nightlights.npz")]:
        parts[name] = fn() if (D / need).exists() else None
    return parts
