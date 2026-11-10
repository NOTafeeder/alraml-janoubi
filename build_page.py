# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio", "numpy", "pillow", "pyproj", "shapely"]
# ///
"""Builds alraml.html: an offline Arabic page with the Sentinel-2 images and sourced facts."""
import base64, io, json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).parent
OUT = ROOT / "sentinel2"

years = []
for tif in sorted(OUT.glob("*/s2_*.tif")):
    with rasterio.open(tif) as src:
        arr = src.read().astype(float)
        source = src.tags().get("source", "")
    red, nir = arr[2], arr[3]
    ndvi = (nir - red) / (nir + red + 1e-9)
    # water: very low near-infrared reflectance (sea surface), or negative NDVI
    water = (nir < 600) | (ndvi < -0.05)
    land = ~water
    veg = land & (ndvi > 0.3)
    date = tif.stem[3:]
    # redraw the NDVI image with this water mask
    t = np.clip((ndvi + 0.1) / 0.7, 0, 1)[..., None]
    brown, yellow, green = np.array([140, 90, 50]), np.array([235, 225, 160]), np.array([20, 120, 40])
    col = np.where(t < 0.5, brown + (yellow - brown) * (t / 0.5), yellow + (green - yellow) * ((t - 0.5) / 0.5))
    col[water] = [40, 70, 140]
    Image.fromarray(col.astype(np.uint8)).save(tif.parent / f"ndvi_{date}.png")
    def b64(p):
        im = Image.open(p)
        buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    years.append({
        "year": int(tif.parent.name), "date": date, "scene": source,
        "rgb": b64(tif.parent / f"rgb_{date}.png"),
        "ndvi": b64(tif.parent / f"ndvi_{date}.png"),
        "land_km2": round(land.sum() * 100 / 1e6, 2),
        "veg_pct": round(100 * veg.sum() / land.sum(), 1),
        "built_pct": round(100 * (land & ~veg).sum() / land.sum(), 1),
        "ndvi_mean": round(float(ndvi[land].mean()), 3),
    })
    print(years[-1]["year"], years[-1]["date"], years[-1]["veg_pct"], years[-1]["ndvi_mean"])

# --- Sentinel-5P NO2, yearly means ---
no2 = {"years": []}
npz = ROOT / "no2" / "no2_year.npz"
if npz.exists():
    d = np.load(npz)
    pts = json.loads((ROOT / "no2" / "no2_year.json").read_text())
    maps, lat, lon = d["maps"], d["lat"], d["lon"]
    vmin, vmax = float(np.nanpercentile(maps, 1)), float(np.nanpercentile(maps, 99))
    stops = np.array([[255, 247, 214], [253, 196, 112], [236, 112, 52], [180, 40, 40], [100, 10, 40]], float)
    for m, p in zip(maps, pts):
        t = np.clip((np.nan_to_num(m, nan=vmin) - vmin) / (vmax - vmin), 0, 1) * (len(stops) - 1)
        i = np.minimum(t.astype(int), len(stops) - 2)
        f = (t - i)[..., None]
        rgb = (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)
        rgb[~np.isfinite(m)] = [200, 200, 200]
        im = Image.fromarray(rgb[::-1])           # north up
        buf = io.BytesIO(); im.save(buf, "PNG")
        no2["years"].append({"year": int(p["start"][:4]), "alraml": round(p["alraml"], 1),
                             "sea": round(p["sea"], 1), "damascus": round(p["damascus"], 1),
                             "product": p["product"],
                             "img": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    north, south = lat[-1] + dlat / 2, lat[0] - dlat / 2
    west, east = lon[0] - dlon / 2, lon[-1] + dlon / 2
    pos = lambda la, lo: {"x": (lo - west) / (east - west), "y": (north - la) / (north - south)}
    no2.update(vmin=round(vmin), vmax=round(vmax), w=len(lon), h=len(lat),
               mark=pos(35.507, 35.800), baniyas=pos(35.18, 35.95),
               bbox=[round(west, 2), round(south, 2), round(east, 2), round(north, 2)])
    print("no2 years", [y["year"] for y in no2["years"]], "vmax", vmax)

import sys
sys.path.insert(0, str(ROOT / "extra"))
import page_data
extra = page_data.build(sorted(OUT.glob("*/s2_*.tif")))
for y in years:
    t = extra["turbidity"].get(y["year"])
    if t:
        y["turb"], y["turb_mean"], y["turb_p95"] = t["img"], t["mean"], t["p95"]
del extra["turbidity"]

# --- the neighbourhood itself: numbers inside the boundary and outlines for every map ---
import hood
from rasterio.crs import CRS
tifs = sorted(OUT.glob("*/s2_*.tif"))
extra["hood"] = {}
if (ROOT / "data_extra" / "boundary_A.geojson").exists():
    with rasterio.open(tifs[-1]) as s:
        s2grid = (s.crs, s.transform, s.width, s.height)
    grids = {"s2": s2grid}
    with rasterio.open(ROOT / "data_extra" / "dem.tif") as s:
        grids["elev"] = (s.crs, s.transform, s.width, s.height)
    with rasterio.open(ROOT / "data_extra" / f"ghsl_{extra['bp']['built'][0]['year']}.tif") as s:
        grids["bp"] = (s.crs, s.transform, s.width, s.height)
    lst_shape = np.load(ROOT / "data_extra" / f"lst_{extra['lst']['years'][0]['year']}.npz")["lst"].shape
    grids["lst"] = (CRS.from_epsg(32637), hood.lst_transform(lst_shape), lst_shape[1], lst_shape[0])
    from rasterio.transform import from_bounds
    if no2["years"]:
        b = no2["bbox"]; grids["no2"] = (CRS.from_epsg(4326), from_bounds(b[0], b[1], b[2], b[3], no2["w"], no2["h"]), no2["w"], no2["h"])
    if extra.get("night"):
        b = extra["night"]["bbox"]; n = extra["night"]
        grids["night"] = (CRS.from_epsg(4326), from_bounds(b[0], b[1], b[2], b[3], n["w"], n["h"]), n["w"], n["h"])
    bb = page_data.BBOX
    grids["osm"] = (CRS.from_epsg(4326), from_bounds(bb[0], bb[1], bb[2], bb[3], 1, 1), 1, 1)
    for v in ["A", "B"]:
        st = hood.stats(v, tifs, extra)
        st["outline"] = {k: hood.outline_norm(v, *g) for k, g in grids.items()}
        extra["hood"][v] = st
        print("hood", v, "land", st["land_km2"], "km2, people", st["pop"][-1], "facilities", st.get("facilities"))

w, h = Image.open(next(OUT.glob("*/rgb_*.png"))).size
html = (ROOT / "template.html").read_text(encoding="utf-8")
def js(o, **kw):
    """JSON for inside <script>: '<' is escaped so data can never close the tag"""
    out = json.dumps(o, **kw).replace("<", "\\u003c")
    return out.replace(chr(0x2028), "\\u2028").replace(chr(0x2029), "\\u2029")
html = html.replace("__YEARS__", js(years)).replace("__NO2__", js(no2)).replace("__EXTRA__", js(extra, ensure_ascii=False)).replace("__W__", str(w)).replace("__H__", str(h))
(ROOT / "alraml.html").write_text(html, encoding="utf-8")
print("wrote alraml.html", round(len(html) / 1e6, 2), "MB")
