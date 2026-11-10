# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Neighbourhood / quarter names around the study area from OpenStreetMap -> data_extra/osm_places.json
Data (c) OpenStreetMap contributors, ODbL."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import requests
from common import BBOX, OUT

pad = 0.01
b = f"{BBOX[1]-pad},{BBOX[0]-pad},{BBOX[3]+pad},{BBOX[2]+pad}"
Q = f"""[out:json][timeout:90];
(
  nwr["place"~"^(neighbourhood|quarter|suburb|city_block|locality|hamlet|village)$"]({b});
);
out center tags;"""
HEAD = {"User-Agent": "alraml-neighbourhood-study/1.0 (personal research)"}
for url in ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"] * 4:
    try:
        r = requests.post(url, data={"data": Q}, headers=HEAD, timeout=120)
        if r.status_code == 200:
            d = r.json(); break
        print("status", r.status_code, url, flush=True)
    except Exception as e:
        print("error", url, str(e)[:80], flush=True)
    time.sleep(20)
else:
    sys.exit("Overpass not reachable")
places = []
for e in d["elements"]:
    t = e.get("tags", {})
    la, lo = (e["lat"], e["lon"]) if "lat" in e else (e["center"]["lat"], e["center"]["lon"])
    places.append({"name": t.get("name:ar") or t.get("name") or "", "name_all": t.get("name", ""),
                   "place": t["place"], "lat": la, "lon": lo, "osm": f'{e["type"]}/{e["id"]}'})
(OUT / "osm_places.json").write_text(json.dumps({"timestamp": d.get("osm3s", {}).get("timestamp_osm_base"),
                                                 "places": places}, ensure_ascii=False, indent=1), encoding="utf-8")
for p in places:
    print(p["place"], "|", p["name"], "|", p["name_all"], round(p["lat"], 4), round(p["lon"], 4))
