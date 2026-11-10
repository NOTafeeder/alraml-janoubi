# /// script
# requires-python = ">=3.11"
# dependencies = ["pystac-client", "rasterio", "numpy", "pillow", "pyproj"]
# ///
"""Laddar ned Sentinel-2 L2A-utsnitt över الرمل الجنوبي, ett per år (maj–sep, minst moln).

Output in sentinel2/<year>/:
  s2_<datum>.tif      – band B02,B03,B04,B08 (uint16, reflektans*10000), EPSG:32637, 10 m
  rgb_<datum>.png     – sann färg, en pixel per 10 m
  ndvi_<datum>.png    – NDVI (brun = bart/byggt, grön = växtlighet)
Plus jamforelse_rgb.png och jamforelse_ndvi.png med alla år sida vid sida (förstorade ×3).
"""
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from pyproj import Transformer
from PIL import Image, ImageDraw
from pystac_client import Client

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
import functools, builtins
print = functools.partial(builtins.print, flush=True)

BBOX = [35.776, 35.495, 35.825, 35.520]
CRS = "EPSG:32637"
RES = 10
BANDS = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08"}
OUT = Path("sentinel2")

# Gemensamt rutnät i UTM 37N
tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
xs, ys = zip(*[tr.transform(x, y) for x in BBOX[::2] for y in BBOX[1::2]])
x0, y1 = np.floor(min(xs) / RES) * RES, np.ceil(max(ys) / RES) * RES
W = int(np.ceil((max(xs) - x0) / RES))
H = int(np.ceil((y1 - min(ys)) / RES))
TRANSFORM = from_origin(x0, y1, RES, RES)


def read_item(item):
    out = []
    for key in BANDS:
        with rasterio.open(item.assets[key].href) as src, WarpedVRT(
            src, crs=CRS, transform=TRANSFORM, width=W, height=H,
            resampling=Resampling.nearest, nodata=0,
        ) as vrt:
            out.append(vrt.read(1))
    return np.stack(out)


def to_rgb(arr):
    rgb = arr[[2, 1, 0]].astype(float)
    lo, hi = np.percentile(rgb[rgb > 0], [2, 98])
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8
    return (np.moveaxis(rgb, 0, -1) * 255).astype(np.uint8)


def to_ndvi(arr):
    red, nir = arr[2].astype(float), arr[3].astype(float)
    ndvi = np.where(nir + red > 0, (nir - red) / (nir + red + 1e-9), 0)
    t = np.clip((ndvi + 0.1) / 0.7, 0, 1)[..., None]  # -0.1 → brun, 0.6 → grön
    brown, yellow, green = np.array([140, 90, 50]), np.array([235, 225, 160]), np.array([20, 120, 40])
    col = np.where(t < 0.5, brown + (yellow - brown) * (t / 0.5), yellow + (green - yellow) * ((t - 0.5) / 0.5))
    col[ndvi < -0.05] = [40, 70, 140]  # vatten
    return col.astype(np.uint8), ndvi


def sheet(images, labels, path, scale=3):
    w, h = W * scale, H * scale
    cols = 4
    rows = -(-len(images) // cols)
    canvas = Image.new("RGB", (cols * w, rows * (h + 24)), "white")
    d = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(images, labels)):
        x, y = (i % cols) * w, (i // cols) * (h + 24)
        canvas.paste(Image.fromarray(im).resize((w, h), Image.NEAREST), (x, y + 24))
        d.text((x + 6, y + 5), lab, fill="black")
    canvas.save(path)


cat = Client.open("https://earth-search.aws.element84.com/v1")
rgbs, ndvis, labels = [], [], []
for year in range(2017, 2027):
    done = sorted((OUT / str(year)).glob("rgb_*.png"))
    if done:
        date = done[0].stem[4:]
        rgbs.append(np.array(Image.open(done[0])))
        ndvis.append(np.array(Image.open(OUT / str(year) / f"ndvi_{date}.png")))
        labels.append(date)
        print(year, "already downloaded", date)
        continue
    items = list(cat.search(
        collections=["sentinel-2-l2a"], bbox=BBOX,
        datetime=f"{year}-05-01/{year}-09-30",
        query={"eo:cloud_cover": {"lt": 5}}, max_items=100,
    ).items())
    items.sort(key=lambda i: i.properties["eo:cloud_cover"])
    for item in items[:4]:
        print(f"  reading {item.id} ...")
        try:
            arr = read_item(item)
        except Exception as e:
            print(f"  failed {item.id}: {e}")
            continue
        nodata = (arr == 0).all(0).mean()
        # clouds/shadows from the scene classification layer (SCL 3, 8, 9, 10)
        with rasterio.open(item.assets["scl"].href) as src, WarpedVRT(
            src, crs=CRS, transform=TRANSFORM, width=W, height=H,
            resampling=Resampling.nearest, nodata=0,
        ) as vrt:
            scl = vrt.read(1)
        cloud = np.isin(scl, [3, 8, 9, 10]).mean()
        if nodata < 0.01 and cloud < 0.02:
            break
        print(f"  hoppar över {item.id}: nodata {nodata:.0%}, ljust {cloud:.0%}")
    else:
        print(year, "ingen bra scen")
        continue
    date = item.datetime.date().isoformat()
    d = OUT / str(year)
    d.mkdir(parents=True, exist_ok=True)
    with rasterio.open(d / f"s2_{date}.tif", "w", driver="GTiff", width=W, height=H, count=4,
                       dtype="uint16", crs=CRS, transform=TRANSFORM, compress="deflate") as dst:
        dst.write(arr.astype("uint16"))
        dst.descriptions = tuple(BANDS.values())
        dst.update_tags(source=item.id, cloud_cover=item.properties["eo:cloud_cover"])
    rgb = to_rgb(arr)
    ndvi_img, ndvi = to_ndvi(arr)
    Image.fromarray(rgb).save(d / f"rgb_{date}.png")
    Image.fromarray(ndvi_img).save(d / f"ndvi_{date}.png")
    rgbs.append(rgb); ndvis.append(ndvi_img); labels.append(date)
    land = ndvi > -0.05
    print(year, item.id, date, f"medel-NDVI land {ndvi[land].mean():.3f}, andel NDVI>0.3 {(ndvi[land] > 0.3).mean():.1%}")

sheet(rgbs, labels, OUT / "jamforelse_rgb.png")
sheet(ndvis, labels, OUT / "jamforelse_ndvi.png")
print("rutnät", W, "x", H, "pixlar à 10 m")
