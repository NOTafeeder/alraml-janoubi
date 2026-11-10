"""Numbers inside the neighbourhood boundary (data_extra/boundary_A.geojson and boundary_B.geojson).
Every layer is masked with the boundary on its own grid; water is removed with the Sentinel-2 water mask
(or the layer's own water test), so the sea inside the drawn polygon never counts."""
import json, math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from pyproj import Transformer
from shapely.geometry import shape, Point, mapping
from shapely.ops import transform as shp_transform

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data_extra"


def load(version):
    return shape(json.loads((D / f"boundary_{version}.geojson").read_text(encoding="utf-8"))["geometry"])


def mask_for(poly_ll, crs, transform, shape_):
    """True inside the polygon, on a raster grid"""
    if crs and crs.to_epsg() != 4326:
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        poly = shp_transform(lambda x, y, z=None: tr.transform(x, y), poly_ll)
    else:
        poly = poly_ll
    return geometry_mask([mapping(poly)], out_shape=shape_, transform=transform, invert=True, all_touched=False)


def s2_water(arr):
    red, nir = arr[2].astype(float), arr[3].astype(float)
    ndvi = (nir - red) / (nir + red + 1e-9)
    return (nir < 600) | (ndvi < -0.05), ndvi


def stats(version, s2_tifs, extra):
    poly = load(version)
    out = {"version": version}
    # Sentinel-2: land area and green share per year
    years = []
    for tif in s2_tifs:
        with rasterio.open(tif) as s:
            a = s.read(); m = mask_for(poly, s.crs, s.transform, (s.height, s.width))
        water, ndvi = s2_water(a)
        land = m & ~water
        years.append({"year": int(tif.parent.name), "land_km2": round(land.sum() * 100 / 1e6, 3),
                      "green_pct": round(100 * (land & (ndvi > 0.3)).sum() / max(land.sum(), 1), 1)})
    out["s2"] = years
    land_km2 = years[-1]["land_km2"]
    out["land_km2"] = land_km2
    # elevation
    with rasterio.open(D / "dem.tif") as s:
        z = s.read(1); m = mask_for(poly, s.crs, s.transform, z.shape)
    zl = z[m & (z > 0.3)]
    out["elev"] = {"below2": round(100 * (zl <= 2).mean(), 1), "below5": round(100 * (zl <= 5).mean(), 1),
                   "below10": round(100 * (zl <= 10).mean(), 1), "median": round(float(np.median(zl)), 1)}
    # built-up (GHSL) and population (WorldPop)
    bp = json.loads((D / "built_pop.json").read_text())
    built = []
    for b in bp["built"]:
        with rasterio.open(D / f"ghsl_{b['year']}.tif") as s:
            g = s.read(1); m = mask_for(poly, s.crs, s.transform, g.shape)
        built.append({"year": b["year"], "built_km2": round(float(np.nansum(g[m])) / 1e6, 3)})
    out["built"] = built
    pops = []
    for p in bp["pop"]:
        with rasterio.open(D / f"pop_{p['year']}.tif") as s:
            g = s.read(1); m = mask_for(poly, s.crs, s.transform, g.shape)
        n = float(np.nansum(g[m]))
        pops.append({"year": p["year"], "people": round(n), "density": round(n / land_km2)})
    out["pop"] = pops
    # land surface temperature (Landsat grid 30 m, UTM 37N)
    lst = []
    scenes = json.loads((D / "lst_scenes.json").read_text())
    for sc in scenes:
        z = np.load(D / f"lst_{sc['year']}.npz")
        T, nd, clear = z["lst"], z["ndvi"], z["clear"]
        tr_ = lst_transform(T.shape)
        m = mask_for(poly, rasterio.crs.CRS.from_epsg(32637), tr_, T.shape)
        ok = m & clear & np.isfinite(T) & (nd >= 0)
        lst.append({"year": sc["year"], "date": sc["date"], "hood": round(float(T[ok].mean()), 1) if ok.any() else None})
    out["lst"] = lst
    # night lights (15 arc-second cells; cell centres inside the polygon)
    if (D / "nightlights.npz").exists():
        z = np.load(D / "nightlights.npz"); lat, lon = z["lat"], z["lon"]
        inside = np.array([[poly.contains(Point(lo, la)) for lo in lon] for la in lat])
        meta = json.loads((D / "nightlights.json").read_text())
        out["night"] = [{"year": y["year"], "hood": round(float(np.nanmean(mp[inside])), 2)}
                        for y, mp in zip(meta["years"], z["maps"])]
        out["night_cells"] = int(inside.sum())
    # facilities from OpenStreetMap
    if extra.get("osm"):
        cnt = {}
        for p in extra["osm"]["points"]:
            if poly.contains(Point(p["lon"], p["lat"])):
                cnt[p["c"]] = cnt.get(p["c"], 0) + 1
        out["facilities"] = cnt
    return out


def lst_transform(shape_):
    """same 30 m grid as extra/lst.py"""
    from rasterio.transform import from_origin
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from common import BBOX
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)
    xs, ys = zip(*[tr.transform(x, y) for x in BBOX[::2] for y in BBOX[1::2]])
    x0, y1 = np.floor(min(xs) / 30) * 30, np.ceil(max(ys) / 30) * 30
    return from_origin(x0, y1, 30, 30)


def outline_norm(version, crs, transform, width, height):
    """polygon outline as an SVG path in 0..1 map coordinates of a raster (x right, y down)"""
    poly = load(version)
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True) if crs and crs.to_epsg() != 4326 else None
    inv = ~transform
    pts = []
    for lo, la in poly.exterior.coords:
        x, y = tr.transform(lo, la) if tr else (lo, la)
        c, r = inv * (x, y)
        pts.append("%.4f,%.4f" % (c / width, r / height))
    return "M" + "L".join(pts) + "Z"
