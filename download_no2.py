# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "aiohttp", "h5py", "numpy", "requests"]
# ///
"""Reads tropospheric NO2 (Sentinel-5P, ESA S5P-PAL L3 grids) for the Latakia area.

each global file is
one gzip chunk, so ~420 MB is transferred per file; only the Latakia windows are stored. Output: no2/no2.npz and no2/items.json.
"""
import json, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec, h5py, numpy as np, requests

API = "https://data-portal.s5p-pal.com/api/s5p-l3/no2-tropospheric"
MAP = dict(lat=(34.9, 36.1), lon=(35.2, 36.4))         # map window around Latakia
POINTS = {                                             # 3x3 cells around each point
    "alraml": (35.507, 35.800),   # al-Raml al-Janoubi
    "sea": (35.50, 35.35),        # open sea west of Latakia
    "damascus": (33.51, 36.29),   # Damascus, for comparison
}
OUT = Path("no2"); OUT.mkdir(exist_ok=True)
VAR = "tropospheric_NO2_column_number_density"


def get(url):
    for i in range(4):
        try:
            r = requests.get(url, timeout=60); r.raise_for_status(); return r.json()
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def list_items(period):
    items = {}
    for y in get(f"{API}/{period}")["links"]:
        if y["rel"] != "child":
            continue
        for it in get(y["href"])["links"]:
            if it["rel"] != "item":
                continue
            m = re.search(rf"{period}-(\d{{8}})-(\d{{8}})", it["href"])
            start, created = m.group(1), m.group(2)
            if start not in items or created > items[start][0]:   # keep newest version
                items[start] = (created, it["href"])
    return {k: v[1] for k, v in sorted(items.items())}


def grid():
    lat = np.linspace(-90 + 180 / 8192 / 2, 90 - 180 / 8192 / 2, 8192)
    lon = np.linspace(-180 + 360 / 16384 / 2, 180 - 360 / 16384 / 2, 16384)
    return lat, lon


def read(item_url, start):
    cache = OUT / f"year_{start}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["r"].item()
    item = get(item_url)
    href = item["assets"]["product"]["href"]
    for i in range(8):
        try:
            # fresh signed URL each attempt (they expire after ~20 min)
            real = requests.head(href, allow_redirects=False, timeout=60).headers["Location"]
            with fsspec.open(real, block_size=16 * 2**20).open() as f, h5py.File(f, "r") as h:
                lat, lon = h["latitude"][:], h["longitude"][:]
                full = h[VAR][0]   # the variable is one gzip chunk: read it once
            full[(full < -1e30) | (full > 1e30)] = np.nan
            def win(la, lo):
                ii = np.where((lat >= la[0]) & (lat <= la[1]))[0]
                jj = np.where((lon >= lo[0]) & (lon <= lo[1]))[0]
                return full[ii.min():ii.max() + 1, jj.min():jj.max() + 1].copy(), lat[ii], lon[jj]
            m, mlat, mlon = win(MAP["lat"], MAP["lon"])
            pts = {}
            for k, (la, lo) in POINTS.items():
                d = 180 / 8192 * 1.5
                a, _, _ = win((la - d, la + d), (lo - d, lo + d))
                pts[k] = float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
            del full
            r = dict(start=start, map=m, lat=mlat, lon=mlon, pts=pts,
                     product=item["properties"]["product_name"])
            np.savez_compressed(cache, r=np.array(r, dtype=object))
            return r
        except Exception as e:
            print("  retry", start, str(e)[:120], flush=True)
            time.sleep(15 * (i + 1))
    raise RuntimeError(item_url)


for period in ["year"]:
    items = list_items(period)
    print(period, len(items), "items", flush=True)
    results = []
    with ThreadPoolExecutor(2) as ex:
        for r in ex.map(read, items.values(), items.keys()):
            print(" ", period, r["start"], {k: round(v, 1) for k, v in r["pts"].items()}, flush=True)
            results.append(r)
    np.savez_compressed(OUT / f"no2_{period}.npz",
                        maps=np.stack([r["map"] for r in results]),
                        lat=results[0]["lat"], lon=results[0]["lon"],
                        starts=np.array([r["start"] for r in results]))
    (OUT / f"no2_{period}.json").write_text(json.dumps(
        [{"start": r["start"], "product": r["product"], **r["pts"]} for r in results], indent=1))
print("done", flush=True)
