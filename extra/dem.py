# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio", "numpy"]
# ///
"""Copernicus DEM GLO-30 (AWS open data), cropped to the study area -> data_extra/dem.tif"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import rasterio
from rasterio.windows import from_bounds
from common import BBOX, OUT

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
URL = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
       "Copernicus_DSM_COG_10_N35_00_E035_00_DEM/Copernicus_DSM_COG_10_N35_00_E035_00_DEM.tif")
with rasterio.open(URL) as src:
    w = from_bounds(*BBOX, src.transform).round_offsets().round_lengths()
    a = src.read(1, window=w)
    prof = src.profile | dict(height=a.shape[0], width=a.shape[1], transform=src.window_transform(w),
                              driver="GTiff", compress="deflate", tiled=False)
    prof.pop("blockxsize", None); prof.pop("blockysize", None)
with rasterio.open(OUT / "dem.tif", "w", **prof) as dst:
    dst.write(a, 1)
print("dem", a.shape, float(a.min()), float(a.max()))
