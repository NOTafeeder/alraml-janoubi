"""Risk indicators for the neighbourhood (boundary B), computed only from data already in the project.
Returns a dict with numbers and small PNG maps for the risks tab. These are indicators, not an official
risk assessment."""
import json, math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from pyproj import Transformer
from shapely.geometry import shape, mapping, Point
from shapely.ops import transform as shp_transform

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data_extra"
S2 = ROOT / "sentinel2"

# Damage points from the International Charter satellite assessment of 7 Feb 2023 (Pléiades), as printed in the report
DAMAGE = [
    {"name": "شرق الرمل الجنوبي", "lat": 35 + 30 / 60 + 26 / 3600, "lon": 35 + 49 / 60 + 1 / 3600, "level": "مبانٍ مدمّرة أو متضررة بشدة"},
    {"name": "غرب الرمل الجنوبي", "lat": 35 + 31 / 60 + 20 / 3600, "lon": 35 + 47 / 60 + 19 / 3600, "level": "مبانٍ يُحتمل تضررها"},
    {"name": "قطاع مشفى تشرين الجامعي", "lat": 35 + 31 / 60 + 20 / 3600, "lon": 35 + 47 / 60 + 57 / 3600, "level": "مبانٍ يُحتمل تضررها"},
]


def png(rgb):
    import base64, io
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(rgb.astype(np.uint8)).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def s2_stack():
    out = {}
    for tif in sorted(S2.glob("*/s2_*.tif")):
        with rasterio.open(tif) as s:
            out[int(tif.parent.name)] = (s.read().astype(float), s.profile)
    return out


def water_of(a):
    red, nir = a[2], a[3]
    ndvi = (nir - red) / (nir + red + 1e-9)
    return (nir < 600) | (ndvi < -0.05), ndvi


def sea_of(water):
    """water connected to the bottom edge = the sea (removes river, ponds and dark roofs)"""
    from scipy import ndimage
    lab, _ = ndimage.label(water)
    ids = np.unique(lab[-1][water[-1]])
    return np.isin(lab, ids[ids > 0])


def grey(a):
    g = np.clip(a[[2, 1, 0]].mean(0) / 3000, 0, 1) * 120 + 110
    return np.repeat(g[..., None], 3, -1)


def build(extra):
    poly = shape(json.loads((D / "boundary_B.geojson").read_text(encoding="utf-8"))["geometry"])
    H = extra["hood"]["B"]
    stack = s2_stack()
    years = sorted(stack)
    a26, prof = stack[years[-1]]
    crs, tr, Wd, Ht = prof["crs"], prof["transform"], prof["width"], prof["height"]
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    poly_u = shp_transform(lambda x, y, z=None: fwd.transform(x, y), poly)
    inside = geometry_mask([mapping(poly_u)], out_shape=(Ht, Wd), transform=tr, invert=True)
    water26, ndvi26 = water_of(a26)
    land = inside & ~water26
    built = land & (ndvi26 < 0.25)
    out = {"note": "مؤشرات من البيانات المفتوحة، لا تقييم رسمي للمخاطر"}

    # 1. earthquake: documented damage, old and dense buildings
    pts = []
    for d in DAMAGE:
        x, y = fwd.transform(d["lon"], d["lat"]); c, r = ~tr * (x, y)
        pts.append({**d, "x": c / Wd, "y": r / Ht, "in_map": 0 <= c < Wd and 0 <= r < Ht,
                    "in_hood": poly.contains(Point(d["lon"], d["lat"]))})
    b75 = next(b for b in H["built"] if b["year"] == 1975)["built_km2"]
    b20 = next(b for b in H["built"] if b["year"] == 2020)["built_km2"]
    out["quake"] = {"points": pts, "old_share": round(100 * b75 / b20), "cover_pct": round(100 * b20 / H["land_km2"]),
                    "built_1975": b75, "built_2020": b20}

    # 2. sea and flood: built land below 5 m (DEM warped to the 10 m grid)
    with rasterio.open(D / "dem.tif") as s:
        z = np.full((Ht, Wd), np.nan, dtype="float32")
        reproject(s.read(1).astype("float32"), z, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear)
    low = land & (z <= 5)
    low_built = low & built
    img = grey(a26)
    img[water26] = [150, 180, 195]
    img[low & ~built] = [236, 200, 150]
    img[low_built] = [196, 110, 80]
    out["flood"] = {"low_km2": round(low.sum() * 100 / 1e6, 3), "low_built_km2": round(low_built.sum() * 100 / 1e6, 3),
                    "low_built_pct_of_built": round(100 * low_built.sum() / max(built.sum(), 1), 1),
                    "img": png(img), "w": Wd, "h": Ht}
    # heavy rain days (ERA5)
    rd = json.loads((D / "rain_daily.json").read_text())
    t = np.array(rd["time"], dtype="datetime64[D]")
    mm = np.array([np.nan if v is None else v for v in rd["mm"]], float)
    yr = t.astype("datetime64[Y]").astype(int) + 1970
    mo = t.astype("datetime64[M]").astype(int) % 12 + 1
    hy = np.where(mo >= 9, yr + 1, yr)
    heavy = []
    for y in range(1941, int(hy.max()) + 1):
        sel = hy == y
        if sel.sum() < 365:
            continue
        heavy.append({"y": y, "days50": int((mm[sel] >= 50).sum()), "max": round(float(np.nanmax(mm[sel])))})
    per = lambda a, b: round(float(np.mean([h["days50"] for h in heavy if a <= h["y"] <= b])), 2)
    out["heavy_rain"] = {"years": heavy, "p1961_1990": per(1961, 1990), "p1991_2020": per(1991, 2020),
                         "p2016_now": per(2016, heavy[-1]["y"]), "record": max(heavy, key=lambda h: h["max"]),
                         "last": heavy[-1]}

    # 3. shoreline: sea in early years (2017-2019) vs late years (2024-2026), majority vote to damp waves and glint
    def sea_years(ys):
        m = [sea_of(water_of(stack[y][0])[0]) for y in ys if y in stack]
        return np.sum(m, 0) >= (len(m) // 2 + 1)
    early, late = sea_years([2017, 2018, 2019]), sea_years([2024, 2025, 2026])
    from scipy import ndimage
    near = ndimage.binary_dilation(inside, iterations=15)          # the hood and 150 m around it
    eroded = near & ~early & late                                    # land that became sea
    accreted = near & early & ~late                                  # sea that became land
    coast = near & (early ^ ndimage.binary_erosion(early))           # early shoreline pixels
    coast_km = coast.sum() * 10 / 1000
    img = grey(a26)
    img[late & ~eroded] = [160, 190, 205]
    img[eroded] = [200, 90, 80]
    img[accreted] = [110, 160, 110]
    out["shore"] = {"eroded_m2": int(eroded.sum() * 100), "accreted_m2": int(accreted.sum() * 100),
                    "coast_km": round(coast_km, 2),
                    "net_shift_m": round((eroded.sum() - accreted.sum()) * 100 / max(coast_km * 1000, 1), 1),
                    "img": png(img), "w": Wd, "h": Ht, "early": "2017–2019", "late": "2024–2026"}

    # 4. heat: mean summer surface temperature 2022-2026 inside the hood
    lsts = []
    for sc in json.loads((D / "lst_scenes.json").read_text()):
        if sc["year"] >= 2022:
            zz = np.load(D / f"lst_{sc['year']}.npz")
            T = np.where(zz["clear"], zz["lst"], np.nan)
            lsts.append(T); nd = zz["ndvi"]
    T = np.nanmean(np.array(lsts), 0)
    import hood
    m = hood.mask_for(poly, rasterio.crs.CRS.from_epsg(32637), hood.lst_transform(T.shape), T.shape)
    wat = (T < 34) | ((nd < 0) & (T < 36))          # sea: cool in summer (land is above 35 C at 11:00)
    landT = m & ~wat & np.isfinite(T)
    hot = landT & (T >= 45)
    rng = np.clip((T - 30) / 20, 0, 1)
    stops = np.array([[240, 236, 220], [244, 200, 150], [222, 140, 100], [176, 80, 70]], float)
    k = np.minimum((rng * 3).astype(int), 2); f = (rng * 3 - k)[..., None]
    img = stops[k] * (1 - f) + stops[k + 1] * f
    img[~np.isfinite(T)] = [200, 200, 200]
    img[wat] = [160, 190, 205]
    out["heat"] = {"years": "2022–2026", "mean": round(float(np.nanmean(T[landT])), 1), "hot45_pct": round(100 * hot.sum() / landT.sum(), 1),
                   "max": round(float(np.nanpercentile(T[landT], 98)), 1), "img": png(img), "w": T.shape[1], "h": T.shape[0]}

    # 5. drought: rain years below 75 % of the 1991-2020 normal
    R = extra["rain"]; norm = R["normal"]
    full = [y for y in R["years"] if y["complete"]]
    dry = [y for y in full if y["mm"] < 0.75 * norm]
    dec = {}
    for y in full:
        d = (y["y"] // 10) * 10
        dec.setdefault(d, [0, 0]); dec[d][1] += 1
        if y["mm"] < 0.75 * norm:
            dec[d][0] += 1
    out["drought"] = {"threshold": round(0.75 * norm), "dry_years": [{"y": y["y"], "mm": y["mm"]} for y in dry],
                      "by_decade": [{"d": d, "dry": v[0], "n": v[1]} for d, v in sorted(dec.items())]}

    # 6. electricity (night lights inside the hood)
    nt = H.get("night", [])
    out["power"] = {"first": nt[0], "last": nt[-1], "low": min(nt, key=lambda v: v["hood"])} if nt else None

    # 7. services per 10,000 residents (OpenStreetMap, may be incomplete)
    p = H["pop"][-1]["people"]; fac = H.get("facilities", {})
    out["services"] = {"people": p, "year": H["pop"][-1]["year"],
                       "per10k": {k: round(v / p * 1e4, 1) for k, v in fac.items()}, "counts": fac}
    return out
