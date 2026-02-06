from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class GridDefinition:
  x_coords: np.ndarray
  y_coords: np.ndarray
  resolution_m: float
  radius_m: float

  @property
  def width(self) -> int:
    return int(self.x_coords.size)

  @property
  def height(self) -> int:
    return int(self.y_coords.size)


def local_crs(observer_lat: float, observer_lon: float) -> CRS:
  """
  Create a local azimuthal equidistant CRS centered on the observer.
  Useful for distance-preserving computations in a neighborhood.
  """

  return CRS.from_proj4(
    f"+proj=aeqd +lat_0={observer_lat} +lon_0={observer_lon} +datum=WGS84 +units=m +no_defs"
  )


def latlon_to_meters(
  lat: float,
  lon: float,
  observer_lat: float,
  observer_lon: float,
) -> tuple[float, float]:
  """Convert lat/lon to local meters relative to the observer-centered CRS."""

  crs_local = local_crs(observer_lat, observer_lon)
  transformer = Transformer.from_crs("EPSG:4326", crs_local, always_xy=True)
  x, y = transformer.transform(lon, lat)
  return float(x), float(y)


def meters_to_latlon(
  x: float,
  y: float,
  observer_lat: float,
  observer_lon: float,
) -> tuple[float, float]:
  """Convert local meters in the observer CRS back to lat/lon."""

  crs_local = local_crs(observer_lat, observer_lon)
  transformer = Transformer.from_crs(crs_local, "EPSG:4326", always_xy=True)
  lon, lat = transformer.transform(x, y)
  return float(lat), float(lon)


def generate_square_grid(radius_m: float, resolution_m: float) -> GridDefinition:
  """
  Generate a square grid centered at (0, 0) that covers the given radius.
  x increases east, y increases north. Row 0 corresponds to north edge.
  """

  if radius_m <= 0 or resolution_m <= 0:
    raise ValueError("radius_m and resolution_m must be positive.")

  step = float(resolution_m)
  x_coords = np.arange(-radius_m, radius_m + step * 0.5, step, dtype=np.float64)
  y_coords = np.arange(radius_m, -radius_m - step * 0.5, -step, dtype=np.float64)

  return GridDefinition(
    x_coords=x_coords,
    y_coords=y_coords,
    resolution_m=step,
    radius_m=float(radius_m),
  )


def grid_indices_to_meters(row: int, col: int, grid: GridDefinition) -> tuple[float, float]:
  """Map grid indices to local meters (x, y)."""

  if row < 0 or row >= grid.height or col < 0 or col >= grid.width:
    raise IndexError("Grid index out of range.")

  x = float(grid.x_coords[col])
  y = float(grid.y_coords[row])
  return x, y


def grid_indices_to_latlon(
  row: int,
  col: int,
  grid: GridDefinition,
  observer_lat: float,
  observer_lon: float,
) -> tuple[float, float]:
  """Map grid indices to lat/lon via the local CRS."""

  x, y = grid_indices_to_meters(row, col, grid)
  return meters_to_latlon(x, y, observer_lat, observer_lon)
