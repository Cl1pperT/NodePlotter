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


__all__ = [
  "DemProvider",
  "DemResult",
  "GridDefinition",
  "get_dem",
  "generate_square_grid",
  "grid_indices_to_latlon",
  "grid_indices_to_meters",
  "latlon_to_meters",
  "local_crs",
  "meters_to_latlon",
  "get_dem_version",
  "TerrariumProvider",
]
