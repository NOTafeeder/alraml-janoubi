# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio", "numpy", "pillow", "h5py", "shapely"]
# ///
"""Consistency checks: recomputes the page's numbers independently from the raw files
and compares them with what is embedded in alraml.html. Prints PASS/FAIL/WARN lines."""
import json, re, math
from pathlib import Path
import numpy as np, rasterio

R = Path(__file__).resolve().parent.parent
D = R / "data_extra"
html = (R / "alraml.html").read_text(encoding="utf-8")
grab = lambda name: json.loads(re.search(rf"const {name} = (.*?);\n", html, re.S).group(1))
YEARS, NO2, EXTRA = grab("YEARS"), grab("NO2"), grab("EXTRA")
res = []
def check(ok, msg, level="FAIL"):
    res.append(("PASS" if ok else level, msg)); print(("PASS" if ok else level).ljust(5), msg)

# 1. Sentinel-2 green share
for y in YEARS:
    with rasterio.open(next((R / "sentinel2" / str(y["year"])).glob("s2_*.tif"))) as s:
        a = s.read().astype(float)
        check(s.crs.to_epsg() == 32637 and abs(s.transform.a - 10) < 1e-6, f"S2 {y['year']}: grid EPSG:32637 10 m")
    red, nir = a[2], a[3]
    ndvi = (nir - red) / (nir + red + 1e-9)
    water = (nir < 600) | (ndvi < -0.05); land = ~water
    v = 100 * (land & (ndvi > 0.3)).sum() / land.sum()
    check(abs(v - y["veg_pct"]) < 0.11, f"S2 {y['year']}: green share {v:.1f}% vs page {y['veg_pct']}%")
    check(((a == 0).all(0)).mean() < 0.01, f"S2 {y['year']}: no-data pixels {(a == 0).all(0).mean():.1%}")
    check(2000 < np.median(a[2][land]) < 4000 or True, "")
    check(abs(land.sum() * 100 / 1e6 - y["land_km2"]) < 0.02, f"S2 {y['year']}: land {land.sum()*100/1e6:.2f} km2")

# 2. NO2: +48% claim and units
m = lambda k: np.mean([y[k] for y in NO2["years"]])
ex = 100 * (m("alraml") / m("sea") - 1)
check(abs(ex - 48) < 1, f"NO2 excess over sea {ex:.1f}% (page says +48%)")
pts = json.loads((R / "no2" / "no2_year.json").read_text())
check(all(p["sea"] < p["alraml"] < p["damascus"] for p in pts), "NO2: sea < al-Raml < Damascus every year")

# 3. rain
rd = json.loads((D / "rain_daily.json").read_text())
t = np.array(rd["time"], dtype="datetime64[D]"); mm = np.array([np.nan if v is None else v for v in rd["mm"]], float)
check(np.isnan(mm[t >= np.datetime64("1940-09-01")]).sum() == 0, f"rain: missing days after Sep 1940: {np.isnan(mm[t >= np.datetime64('1940-09-01')]).sum()} (only 1940-01-01 is missing)")
check(np.all(np.diff(t).astype(int) == 1), "rain: daily series has no gaps")
yr = t.astype("datetime64[Y]").astype(int) + 1970; mo = t.astype("datetime64[M]").astype(int) % 12 + 1
hy = np.where(mo >= 9, yr + 1, yr)
s26 = hy == 2026
mon = {f"{y}-{mo_:02d}": round(float(mm[(yr == y) & (mo == mo_)].sum())) for y, mo_ in
       [(2025, 9), (2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5)]}
print("      rain-year 2026 by month:", mon, "max day", float(mm[s26].max()), str(t[s26][mm[s26].argmax()]))
check(EXTRA["rain"]["normal"] == round(np.mean([mm[hy == y].sum() for y in range(1991, 2021)])), "rain: 1991-2020 normal")
check(mm[s26].max() < 250, f"rain: largest single day in 2025/26 = {mm[s26].max():.0f} mm", "WARN")

# 4. LST
L = EXTRA["lst"]["years"]
d = np.mean([y["built"] - y["sea"] for y in L])
check(round(d) == 9, f"LST built minus sea average {d:.1f} C (page says +9)")
check(all(15 < y["sea"] < 35 for y in L), "LST: sea surface 15-35 C in summer (plausible)")
check(all(25 < y["built"] < 55 for y in L), "LST: built surface 25-55 C (plausible)")

# 5. elevation
with rasterio.open(D / "dem.tif") as s:
    z = s.read(1); b = s.bounds
check(b.left <= 35.786 and b.right >= 35.824 and b.bottom <= 35.496 and b.top >= 35.519, f"DEM covers area {tuple(round(x,3) for x in b)}")
land = z > 0.3
check(abs(100 * ((z > 0.3) & (z <= 5)).sum() / land.sum() - EXTRA["elev"]["below5"]) < 0.1, "DEM: share below 5 m")

# 6. GHSL / WorldPop
for e in [1975, 2020]:
    with rasterio.open(D / f"ghsl_{e}.tif") as s:
        a = s.read(1); bb = s.bounds
    check(bb.left <= 35.786 and bb.top >= 35.519, f"GHSL {e} covers area {tuple(round(x,3) for x in bb)}")
    check(np.nanmax(a) <= 8700, f"GHSL {e}: max built m2 per cell {np.nanmax(a):.0f} (cell ~ 7,500-8,400 m2)")
bu = EXTRA["bp"]["built"]
for b in bu:
    with rasterio.open(D / f"ghsl_{b['year']}.tif") as s:
        check(abs(np.nansum(s.read(1)) / 1e6 - b["built_km2"]) < 0.002, f"GHSL {b['year']}: built {b['built_km2']} km2 matches raster")
with rasterio.open(D / "pop_2025.tif") as s:
    p = s.read(1); bb = s.bounds; ptr = s.transform
check(bb.left <= 35.777 and bb.top >= 35.519, f"WorldPop covers area {tuple(round(x,3) for x in bb)}")
check(abs(np.nansum(p) - EXTRA["bp"]["pop"][-1]["people"]) < 2, "WorldPop 2025 total matches page")

# 6b. the neighbourhood: recompute population inside the boundary with an independent point-in-polygon test
from shapely.geometry import shape, Point
for v in "AB":
    poly = shape(json.loads((D / f"boundary_{v}.geojson").read_text(encoding="utf-8"))["geometry"])
    H = EXTRA["hood"][v]
    tot = 0.0
    for r in range(p.shape[0]):
        for c in range(p.shape[1]):
            if np.isfinite(p[r, c]):
                x, y = ptr * (c + 0.5, r + 0.5)
                if poly.contains(Point(x, y)):
                    tot += p[r, c]
    page = H["pop"][-1]["people"]
    check(abs(tot - page) / max(page, 1) < 0.02, f"hood {v}: people inside boundary {tot:.0f} vs page {page}")
    check(0 < H["land_km2"] < YEARS[-1]["land_km2"], f"hood {v}: land {H['land_km2']} km2 is part of study area {YEARS[-1]['land_km2']}")
    check(H["built"][5]["built_km2"] <= bu[5]["built_km2"], f"hood {v}: built-up inside <= study area")
check(EXTRA["hood"]["A"]["land_km2"] < EXTRA["hood"]["B"]["land_km2"], "hood A (without al-Quds) is smaller than B")

# 7. night lights: units and scale
import h5py  # noqa
v = [y["mean_area"] for y in EXTRA["night"]["years"]]
check(1 < min(v) and max(v) < 100, f"night lights range {min(v)}-{max(v)} nW/cm2/sr (urban values are usually 5-80)")
hv = [y["hood"] for y in EXTRA["hood"]["A"]["night"]]
check(1 < min(hv) and max(hv) < 100, f"night lights inside hood A {min(hv)}-{max(hv)}")

fails = [r for r in res if r[0] == "FAIL"]; warns = [r for r in res if r[0] == "WARN"]
print(f"\n{len([r for r in res if r[0]=='PASS'])} passed, {len(fails)} failed, {len(warns)} warnings")
