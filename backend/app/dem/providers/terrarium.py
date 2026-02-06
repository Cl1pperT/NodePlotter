from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import requests
from affine import Affine
from PIL import Image

from app.dem.providers.base import DemProvider
from app.dem.types import DemResult

TILE_SIZE = 256
EARTH_RADIUS_M = 6378137.0
ORIGIN_SHIFT = 2 * math.pi * EARTH_RADIUS_M / 2.0
INITIAL_RESOLUTION = 2 * math.pi * EARTH_RADIUS_M / TILE_SIZE
MAX_LAT = 85.05112878


class TerrariumProvider(DemProvider):
  def __init__(
    self,
    cache_dir: Path,
    tile_url: str = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
    max_zoom: int = 14,
    min_zoom: int = 0,
  ) -> None:
    self.cache_dir = cache_dir
    self.tile_url = tile_url
    self.max_zoom = max_zoom
    self.min_zoom = min_zoom

  def get_dem(
    self,
    observer_lat: float,
    observer_lon: float,
    radius_km: float,
    resolution_m: float,
  ) -> DemResult:
    zoom = self._choose_zoom(observer_lat, resolution_m)
    bbox = self._radius_bbox_meters(observer_lat, observer_lon, radius_km)
    min_lat, min_lon, max_lat, max_lon = self._bbox_to_latlon(*bbox)

    min_tile_x, max_tile_x, min_tile_y, max_tile_y = self._tile_range(
      min_lat,
      min_lon,
      max_lat,
      max_lon,
      zoom,
    )

    width_tiles = max_tile_x - min_tile_x + 1
    height_tiles = max_tile_y - min_tile_y + 1
    width_px = width_tiles * TILE_SIZE
    height_px = height_tiles * TILE_SIZE

    mosaic = np.full((height_px, width_px), np.nan, dtype=np.float32)

    for ty in range(min_tile_y, max_tile_y + 1):
      for tx in range(min_tile_x, max_tile_x + 1):
        tile = self._load_tile(zoom, tx, ty)
        if tile is None:
          continue
        y_offset = (ty - min_tile_y) * TILE_SIZE
        x_offset = (tx - min_tile_x) * TILE_SIZE
        mosaic[y_offset : y_offset + TILE_SIZE, x_offset : x_offset + TILE_SIZE] = tile

    transform = self._mosaic_transform(min_tile_x, min_tile_y, zoom)

    metadata: dict[str, Any] = {
      "zoom": zoom,
      "tile_range": {
        "min_x": min_tile_x,
        "max_x": max_tile_x,
        "min_y": min_tile_y,
        "max_y": max_tile_y,
      },
      "tile_url": self.tile_url,
      "version": self.version_for_request(observer_lat, observer_lon, radius_km, resolution_m),
    }

    return DemResult(elevation=mosaic, transform=transform, crs="EPSG:3857", metadata=metadata)

  def version_for_request(
    self,
    observer_lat: float,
    observer_lon: float,
    radius_km: float,
    resolution_m: float,
  ) -> str:
    zoom = self._choose_zoom(observer_lat, resolution_m)
    return f"terrarium:z{zoom}:{self.tile_url}"

  def _choose_zoom(self, lat: float, resolution_m: float) -> int:
    lat_clamped = max(min(lat, MAX_LAT), -MAX_LAT)
    lat_rad = math.radians(lat_clamped)
    target = (INITIAL_RESOLUTION * math.cos(lat_rad)) / max(resolution_m, 1e-6)
    zoom = math.ceil(math.log(target, 2)) if target > 0 else self.min_zoom
    return int(max(self.min_zoom, min(self.max_zoom, zoom)))

  def _radius_bbox_meters(
    self,
    lat: float,
    lon: float,
    radius_km: float,
  ) -> tuple[float, float, float, float]:
    x, y = self._latlon_to_meters(lat, lon)
    radius_m = radius_km * 1000.0
    return (x - radius_m, y - radius_m, x + radius_m, y + radius_m)

  def _bbox_to_latlon(
    self,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
  ) -> tuple[float, float, float, float]:
    min_lat, min_lon = self._meters_to_latlon(min_x, min_y)
    max_lat, max_lon = self._meters_to_latlon(max_x, max_y)
    return (min_lat, min_lon, max_lat, max_lon)

  def _tile_range(
    self,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    zoom: int,
  ) -> tuple[int, int, int, int]:
    min_tile_x, min_tile_y = self._latlon_to_tile(max_lat, min_lon, zoom)
    max_tile_x, max_tile_y = self._latlon_to_tile(min_lat, max_lon, zoom)

    max_index = (1 << zoom) - 1
    min_tile_x = max(0, min(max_index, min_tile_x))
    max_tile_x = max(0, min(max_index, max_tile_x))
    min_tile_y = max(0, min(max_index, min_tile_y))
    max_tile_y = max(0, min(max_index, max_tile_y))

    return (min_tile_x, max_tile_x, min_tile_y, max_tile_y)

  def _load_tile(self, zoom: int, x: int, y: int) -> np.ndarray | None:
    tile_path = self._get_tile_path(zoom, x, y)
    if not tile_path.exists():
      tile_path.parent.mkdir(parents=True, exist_ok=True)
      url = self.tile_url.format(z=zoom, x=x, y=y)
      response = requests.get(url, timeout=20)
      if response.status_code != 200:
        return None
      tmp_path = tile_path.with_suffix(".tmp")
      tmp_path.write_bytes(response.content)
      tmp_path.replace(tile_path)

    with Image.open(tile_path) as image:
      rgb = image.convert("RGB")
      data = np.asarray(rgb, dtype=np.float32)

    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]
    elevation = (r * 256.0 + g + b / 256.0) - 32768.0
    return elevation.astype(np.float32)

  def _get_tile_path(self, zoom: int, x: int, y: int) -> Path:
    return self.cache_dir / str(zoom) / str(x) / f"{y}.png"

  def _mosaic_transform(self, min_tile_x: int, min_tile_y: int, zoom: int) -> Affine:
    res = INITIAL_RESOLUTION / (2**zoom)
    min_x = min_tile_x * TILE_SIZE * res - ORIGIN_SHIFT
    max_y = ORIGIN_SHIFT - min_tile_y * TILE_SIZE * res
    return Affine(res, 0.0, min_x, 0.0, -res, max_y)

  def _latlon_to_meters(self, lat: float, lon: float) -> tuple[float, float]:
    lat_clamped = max(min(lat, MAX_LAT), -MAX_LAT)
    x = EARTH_RADIUS_M * math.radians(lon)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + math.radians(lat_clamped) / 2))
    return x, y

  def _meters_to_latlon(self, x: float, y: float) -> tuple[float, float]:
    lon = math.degrees(x / EARTH_RADIUS_M)
    lat = math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2)
    lat = max(min(lat, MAX_LAT), -MAX_LAT)
    return lat, lon

  def _latlon_to_tile(self, lat: float, lon: float, zoom: int) -> tuple[int, int]:
    lat_clamped = max(min(lat, MAX_LAT), -MAX_LAT)
    lat_rad = math.radians(lat_clamped)
    n = 1 << zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x, y
