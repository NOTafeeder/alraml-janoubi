# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio", "numpy", "requests"]
# ///
"""Built-up surface (GHSL R2023A, JRC) and population (WorldPop R2025A, constrained).

Whole files are downloaded to a temporary folder, cropped to the study area and deleted.
Saves data_extra/ghsl_<epoch>.tif and data_extra/pop_<year>.tif plus a summary json.
"""
import json, os, sys, zipfile, tempfile, shutil
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import numpy as np, rasterio, requests
from rasterio.windows import from_bounds
from common import BBOX, OUT

GHSL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/"
        "GHS_BUILT_S_E{e}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_S_E{e}_GLOBE_R2023A_4326_3ss_V1_0_R6_C22.zip")
WP = ("https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{y}/SYR/v1/100m/constrained/"
      "syr_pop_{y}_CN_100m_R2025A_v1.tif")
tmp = Path(tempfile.mkdtemp(prefix="alraml_"))


def fetch(url, dest):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def crop(src_path, out_path):
    with rasterio.open(src_path) as src:
        w = from_bounds(*BBOX, src.transform).round_offsets().round_lengths()
        a = src.read(1, window=w).astype("float32")
        nod = src.nodata
        if nod is not None:
            a[a == nod] = np.nan
        a[a < 0] = np.nan
        prof = dict(driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="float32",
                    crs=src.crs, transform=src.window_transform(w), compress="deflate", nodata=np.nan)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(a, 1)
    return a


summary = {"built": [], "pop": []}
try:
    for e in [1975, 1985, 1995, 2005, 2015, 2020, 2025]:
        z = fetch(GHSL.format(e=e), tmp / f"g{e}.zip")
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".tif"))
            zf.extract(name, tmp)
        a = crop(tmp / name, OUT / f"ghsl_{e}.tif")
        km2 = float(np.nansum(a)) / 1e6
        summary["built"].append({"year": e, "built_km2": round(km2, 3)})
        print("ghsl", e, a.shape, f"built {km2:.2f} km2", flush=True)
        for p in tmp.iterdir():
            p.unlink() if p.is_file() else shutil.rmtree(p)
    for y in [2015, 2020, 2025]:
        t = fetch(WP.format(y=y), tmp / f"p{y}.tif")
        a = crop(t, OUT / f"pop_{y}.tif")
        summary["pop"].append({"year": y, "people": round(float(np.nansum(a)))})
        print("pop", y, a.shape, f"{np.nansum(a):.0f} people", flush=True)
        t.unlink()
finally:
    shutil.rmtree(tmp, ignore_errors=True)
(OUT / "built_pop.json").write_text(json.dumps(summary, indent=1))
print("done", flush=True)
