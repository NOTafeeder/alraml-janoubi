# /// script
# requires-python = ">=3.11"
# dependencies = ["pystac-client", "planetary-computer", "rasterio", "numpy", "pyproj"]
# ///
"""Landsat 8/9 Collection 2 Level-2 surface temperature (Microsoft Planetary Computer).

One clear summer scene (June-August) per year 2013-2026, cropped to the study area on a
30 m grid in UTM 37N. Saves data_extra/lst_<year>.npz with surface temperature (deg C),
NDVI and a clear-sky mask, plus lst_scenes.json.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, planetary_computer, rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from common import BBOX, OUT

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
CRS, RES = "EPSG:32637", 30
tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
xs, ys = zip(*[tr.transform(x, y) for x in BBOX[::2] for y in BBOX[1::2]])
x0, y1 = np.floor(min(xs) / RES) * RES, np.ceil(max(ys) / RES) * RES
W, H = int(np.ceil((max(xs) - x0) / RES)), int(np.ceil((y1 - min(ys)) / RES))
T = from_origin(x0, y1, RES, RES)


def read(href, resampling=Resampling.nearest):
    with rasterio.open(href) as src, WarpedVRT(src, crs=CRS, transform=T, width=W, height=H,
                                               resampling=resampling, nodata=0) as v:
        return v.read(1)


cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
scenes = []
for year in range(2013, 2027):
    items = list(cat.search(collections=["landsat-c2-l2"], bbox=BBOX,
                            datetime=f"{year}-06-01/{year}-08-31",
                            query={"eo:cloud_cover": {"lt": 20},
                                   "platform": {"in": ["landsat-8", "landsat-9"]}}).items())
    items.sort(key=lambda i: i.properties["eo:cloud_cover"])
    for it in items[:5]:
        try:
            qa = read(it.assets["qa_pixel"].href)
            # bits: 0 fill, 1 dilated cloud, 3 cloud, 4 shadow
            bad = (qa == 0) | ((qa >> 1) & 1).astype(bool) | ((qa >> 3) & 1).astype(bool) | ((qa >> 4) & 1).astype(bool)
            if bad.mean() > 0.02:
                print(f"  skip {it.id}: {bad.mean():.0%} cloud/fill", flush=True)
                continue
            st = read(it.assets["lwir11"].href, Resampling.bilinear).astype(float)
            red = read(it.assets["red"].href).astype(float) * 0.0000275 - 0.2
            nir = read(it.assets["nir08"].href).astype(float) * 0.0000275 - 0.2
        except Exception as e:
            print(f"  failed {it.id}: {e}", flush=True)
            continue
        lst = np.where(st > 0, st * 0.00341802 + 149.0 - 273.15, np.nan)
        ndvi = (nir - red) / (nir + red + 1e-9)
        np.savez_compressed(OUT / f"lst_{year}.npz", lst=lst.astype("float32"), ndvi=ndvi.astype("float32"), clear=~bad)
        scenes.append({"year": year, "id": it.id, "date": it.datetime.date().isoformat(),
                       "time_utc": it.datetime.strftime("%H:%M"), "cloud": it.properties["eo:cloud_cover"]})
        print(year, it.id, it.datetime.date(), f"median LST {np.nanmedian(lst):.1f} C", flush=True)
        break
    else:
        print(year, "no clear scene", flush=True)
(OUT / "lst_scenes.json").write_text(json.dumps(scenes, indent=1))
print("grid", W, H, "done", flush=True)
