# /// script
# requires-python = ">=3.11"
# dependencies = ["pystac-client"]
# ///
"""Söker efter Sentinel-2 L2A-scener över الرمل الجنوبي (Latakia) – en per år, minst moln."""
from pystac_client import Client

BBOX = [35.785, 35.495, 35.825, 35.520]  # lon_min, lat_min, lon_max, lat_max

cat = Client.open("https://earth-search.aws.element84.com/v1")
for year in range(2016, 2027):
    items = list(cat.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=f"{year}-05-01/{year}-09-30",
        query={"eo:cloud_cover": {"lt": 5}},
        max_items=100,
    ).items())
    items.sort(key=lambda i: i.properties["eo:cloud_cover"])
    if items:
        b = items[0]
        print(year, len(items), b.id, b.datetime.date(), b.properties["eo:cloud_cover"])
    else:
        print(year, 0)
