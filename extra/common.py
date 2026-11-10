"""Shared settings for the extra layers (all cropped to the same area as the Sentinel-2 images)."""
from pathlib import Path

BBOX = [35.776, 35.495, 35.825, 35.520]          # lon_min, lat_min, lon_max, lat_max
POINT = (35.507, 35.800)                          # al-Raml al-Janoubi (lat, lon)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_extra"
OUT.mkdir(exist_ok=True)
