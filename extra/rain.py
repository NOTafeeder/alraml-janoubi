# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Daily precipitation 1940-today from ERA5 (via the Open-Meteo archive API) -> data_extra/rain_daily.json"""
import json, os, sys, datetime
sys.path.insert(0, os.path.dirname(__file__))
import requests
from common import POINT, OUT

end = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=dict(
    latitude=POINT[0], longitude=POINT[1], start_date="1940-01-01", end_date=end,
    daily="precipitation_sum", models="era5", timezone="Asia/Damascus"), timeout=120)
r.raise_for_status()
d = r.json()
(OUT / "rain_daily.json").write_text(json.dumps(
    {"lat": d["latitude"], "lon": d["longitude"], "model": "era5",
     "time": d["daily"]["time"], "mm": d["daily"]["precipitation_sum"]}))
print("rain days", len(d["daily"]["time"]), d["daily"]["time"][0], d["daily"]["time"][-1], d["latitude"], d["longitude"])
