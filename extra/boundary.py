# /// script
# requires-python = ">=3.11"
# dependencies = ["shapely"]
# ///
"""Builds the neighbourhood boundary of al-Raml al-Janoubi from the user's description:
north-west = railway (it reaches the coast where al-Orouba street does), east = the road running south to the mouth of
Nahr al-Kabir al-Shamali, south = the sea (water is cut away later with the satellite water mask).

A: without the al-Quds area (east edge = the southbound road).
B: with the area east of that road up to the next north-south road (lon ~35.8177) and down to the
   river channel (traced from the Sentinel-2 water mask). B is an approximation: how far al-Quds
   reaches east is not known.
Output: data_extra/boundary_A.geojson, boundary_B.geojson (lon/lat, WGS84)."""
import json
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import substring

D = Path(__file__).resolve().parent.parent / "data_extra"
lines = {e["id"]: e for e in json.loads((D / "osm_lines.json").read_text(encoding="utf-8"))["elements"]}
osm = json.loads((D / "osm.json").read_text(encoding="utf-8"))["elements"]
coords = lambda e: [(p["lon"], p["lat"]) for p in e["geometry"]]

# railway, ordered from the far north-east down to where it meets the coast at the end of al-Orouba
ne = sorted(coords(lines[208215830]), key=lambda p: -p[0])
main = coords(lines[201742378])                      # runs NE -> SW
k = min(range(len(main)), key=lambda i: main[i][1])   # southern-most point = at the coast
rail = LineString(ne + main[:k + 1])
corner = Point(main[k])
print("railway reaches the coast at", round(corner.y, 5), round(corner.x, 5))


def north_edge(lon_east):
    """railway from lon_east (east) down to the coast corner (west), returned west -> east"""
    # first point of the railway at or west of lon_east
    d = next(rail.project(Point(p)) for p in rail.coords if p[0] <= lon_east)
    seg = substring(rail, max(0, d - 0.0008), rail.length)
    pts = [p for p in seg.coords if p[0] <= lon_east]
    return pts[::-1]


SEA = 35.4980
mouth = (35.8118, 35.5003)
road = [(35.8112, 35.5120), (35.8114, 35.5103), (35.8115, 35.5092), (35.8116, 35.5081), (35.8117, 35.5066)]
river = [(35.81126, 35.50022), (35.81258, 35.50035), (35.81389, 35.50056), (35.8152, 35.50087),
         (35.81651, 35.50126), (35.81781, 35.50184)]          # Sentinel-2 water mask, Aug 2026
east_road = [(35.8178, 35.5142), (35.8178, 35.5124), (35.8176, 35.5112), (35.8175, 35.5104), (35.8177, 35.5093), (35.8176, 35.5087)]

sw = (corner.x - 0.0006, SEA)
ringA = north_edge(35.8112) + road + [mouth, (mouth[0], SEA), sw]
ringB = north_edge(35.8178) + east_road + [river[-1]] + river[::-1][1:] + [(mouth[0], SEA), sw]
for name, ring, note in [("A", ringA, "دون حي القدس: الحد الشرقي هو الطريق المتجه جنوباً"),
                         ("B", ringB, "مع حي القدس (تقريبي): حتى الطريق الشمالي–الجنوبي التالي شرقاً ومجرى النهر جنوباً")]:
    poly = Polygon(ring).buffer(0)
    feat = {"type": "Feature", "properties": {"name": "الرمل الجنوبي", "version": name, "note": note,
            "source": "وصف المستخدم + خطوط OpenStreetMap (السكة، شارع العروبة، الطرق) + مجرى النهر من Sentinel-2"},
            "geometry": mapping(poly)}
    (D / f"boundary_{name}.geojson").write_text(json.dumps(feat, ensure_ascii=False), encoding="utf-8")
    b = poly.bounds
    print(name, "valid", poly.is_valid, "bounds", [round(v, 4) for v in b], "vertices", len(poly.exterior.coords))
