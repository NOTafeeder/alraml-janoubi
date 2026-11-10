# /// script
# requires-python = ">=3.11"
# dependencies = ["h5py", "numpy", "requests"]
# ///
"""NASA Black Marble yearly night lights (VNP46A4, collection 5200), tile h21v05, 2012-2025.

Each ~120 MB tile is downloaded to a temp folder with the user's Earthdata token (read from
.earthdata_token.txt, never printed), cropped and deleted. Saves data_extra/nightlights.npz
and nightlights.json.
"""
import json, os, re, sys, tempfile, shutil, time
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import h5py, numpy as np, requests
from common import BBOX, ROOT, OUT

# the token is kept outside the project (home folder); the project folder is only a fallback
CANDIDATES = [Path.home() / ".earthdata_token", Path.home() / ".earthdata_token.txt",
              ROOT / ".earthdata_token", ROOT / ".earthdata_token.txt"]
TOKEN_FILE = next((p for p in CANDIDATES if p.exists()), None)
if TOKEN_FILE is None:
    sys.exit("No Earthdata token found. Save it as ~/.earthdata_token (see README).")
token = TOKEN_FILE.read_text(encoding="utf-8").strip()
H = {"Authorization": f"Bearer {token}"}
BASE = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A4"
MAP = dict(lat=(35.30, 35.70), lon=(35.65, 36.05))       # city and surroundings
STEP = 10 / 2400                                         # 15 arc-seconds
FIELD = "AllAngle_Composite_Snow_Free"
tmp = Path(tempfile.mkdtemp(prefix="alraml_vnp_"))


def rows(la):  # tile h21v05 covers lat 40..30 N, lon 30..40 E
    return int((40 - la[1]) / STEP), int(np.ceil((40 - la[0]) / STEP))


def cols(lo):
    return int((lo[0] - 30) / STEP), int(np.ceil((lo[1] - 30) / STEP))


maps, info = [], []
try:
    for year in range(2012, 2026):
        listing = requests.get(f"{BASE}/{year}/001.json", timeout=60).json()["content"]
        name = next(x["name"] for x in listing if ".h21v05." in x["name"])
        dest = tmp / name
        for attempt in range(5):
            try:
                with requests.get(f"{BASE}/{year}/001/{name}", headers=H, stream=True, timeout=300) as r:
                    if r.status_code in (401, 403):
                        sys.exit(f"Earthdata refused the token (HTTP {r.status_code})")
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for c in r.iter_content(1 << 20):
                            f.write(c)
                break
            except requests.RequestException as e:
                print("  retry", year, str(e)[:100], flush=True)
                time.sleep(20 * (attempt + 1))
        with h5py.File(dest, "r") as h:
            grp = h["HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"]
            ds = grp[FIELD]
            fill = ds.attrs.get("_FillValue", [65535])[0]
            scale = ds.attrs.get("scale_factor", [0.1])[0]
            r0, r1 = rows(MAP["lat"]); c0, c1 = cols(MAP["lon"])
            a = ds[r0:r1, c0:c1].astype("float32")
            a[a == fill] = np.nan
            a *= scale
        br0, br1 = rows((BBOX[1], BBOX[3])); bc0, bc1 = cols((BBOX[0], BBOX[2]))
        area = a[br0 - r0:br1 - r0, bc0 - c0:bc1 - c0]
        maps.append(a)
        info.append({"year": year, "file": name, "mean_area": round(float(np.nanmean(area)), 2),
                     "mean_map": round(float(np.nanmean(a)), 2)})
        print(year, name, info[-1]["mean_area"], info[-1]["mean_map"], flush=True)
        dest.unlink()
finally:
    shutil.rmtree(tmp, ignore_errors=True)
np.savez_compressed(OUT / "nightlights.npz", maps=np.array(maps),
                    lat=np.array([40 - (rows(MAP["lat"])[0] + i + 0.5) * STEP for i in range(maps[0].shape[0])]),
                    lon=np.array([30 + (cols(MAP["lon"])[0] + j + 0.5) * STEP for j in range(maps[0].shape[1])]))
(OUT / "nightlights.json").write_text(json.dumps({"field": FIELD, "units": "nW/cm2/sr", "years": info}, indent=1))
print("done", flush=True)
