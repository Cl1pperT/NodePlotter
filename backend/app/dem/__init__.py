from __future__ import annotations

from pathlib import Path

from app.dem.coords import (
  GridDefinition,
  grid_indices_to_latlon,
  grid_indices_to_meters,
  generate_square_grid,
  latlon_to_meters,
  local_crs,
  meters_to_latlon,
)
from app.dem.providers.base import DemProvider
from app.dem.providers.terrarium import TerrariumProvider
from app.dem.types import DemResult

DEFAULT_CACHE_DIR = Path.home() / "Downloads" / "LocalViewshedExplorer" / "data" / "dem"


def get_dem(
  observer_lat: float,
  observer_lon: float,
  radius_km: float,
  resolution_m: float,
  provider: DemProvider | None = None,
  cache_dir: Path | None = None,
  ) -> DemResult:
  if provider is None:
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    provider = TerrariumProvider(cache_root)

  return provider.get_dem(
    observer_lat=observer_lat,
    observer_lon=observer_lon,
    radius_km=radius_km,
    resolution_m=resolution_m,
  )


def get_dem_for_bbox(
  min_lat: float,
  min_lon: float,
  max_lat: float,
  max_lon: float,
  resolution_m: float,
  provider: DemProvider | None = None,
  cache_dir: Path | None = None,
) -> DemResult:
  if provider is None:
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    provider = TerrariumProvider(cache_root)

  get_for_bbox = getattr(provider, "get_dem_for_bbox", None)
  if not callable(get_for_bbox):
    raise ValueError("DEM provider does not support bounding box requests.")

  return get_for_bbox(
    min_lat=min_lat,
    min_lon=min_lon,
    max_lat=max_lat,
    max_lon=max_lon,
    resolution_m=resolution_m,
  )


def get_dem_version(
  observer_lat: float,
  observer_lon: float,
  radius_km: float,
  resolution_m: float,
  provider: DemProvider | None = None,
  cache_dir: Path | None = None,
) -> str:
  if provider is None:
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    provider = TerrariumProvider(cache_root)

  version_for_request = getattr(provider, "version_for_request", None)
  if callable(version_for_request):
    return str(version_for_request(observer_lat, observer_lon, radius_km, resolution_m))

  return provider.__class__.__name__


def get_dem_version_for_bbox(
  min_lat: float,
  min_lon: float,
  max_lat: float,
  max_lon: float,
  resolution_m: float,
  provider: DemProvider | None = None,
  cache_dir: Path | None = None,
) -> str:
  if provider is None:
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    provider = TerrariumProvider(cache_root)

  version_for_bbox = getattr(provider, "version_for_bbox", None)
  if callable(version_for_bbox):
    return str(
      version_for_bbox(
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        resolution_m=resolution_m,
      )
    )

  return provider.__class__.__name__


__all__ = [
  "DemProvider",
  "DemResult",
  "GridDefinition",
  "get_dem",
  "get_dem_for_bbox",
  "generate_square_grid",
  "grid_indices_to_latlon",
  "grid_indices_to_meters",
  "latlon_to_meters",
  "local_crs",
  "meters_to_latlon",
  "get_dem_version",
  "get_dem_version_for_bbox",
  "TerrariumProvider",
]
