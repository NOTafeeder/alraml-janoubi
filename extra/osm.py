# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""OpenStreetMap buildings, roads and facilities in the study area (Overpass API) -> data_extra/osm.json
Data (c) OpenStreetMap contributors, ODbL."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import requests
from common import BBOX, OUT

b = f"{BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]}"
Q = f"""[out:json][timeout:120];
(
  way["building"]({b});
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street|service)$"]({b});
  nwr["amenity"~"^(school|kindergarten|university|college|hospital|clinic|doctors|pharmacy|place_of_worship|fire_station|police)$"]({b});
  nwr["leisure"~"^(park|playground|pitch)$"]({b});
);
out geom tags;"""
HEAD = {"User-Agent": "alraml-neighbourhood-study/1.0 (personal research)"}
for url in ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"] * 3:
    try:
        r = requests.post(url, data={"data": Q}, headers=HEAD, timeout=180)
        if r.status_code == 200:
            d = r.json()
            break
        print("status", r.status_code, url, flush=True)
    except Exception as e:
        print("error", url, e, flush=True)
    time.sleep(20)
else:
    sys.exit("Overpass not reachable")
(OUT / "osm.json").write_text(json.dumps({"timestamp": d.get("osm3s", {}).get("timestamp_osm_base"),
                                          "elements": d["elements"]}))
kinds = {}
for e in d["elements"]:
    t = e.get("tags", {})
    k = "building" if "building" in t else "highway" if "highway" in t else t.get("amenity") or t.get("leisure")
    kinds[k] = kinds.get(k, 0) + 1
print(kinds, d.get("osm3s", {}).get("timestamp_osm_base"))
