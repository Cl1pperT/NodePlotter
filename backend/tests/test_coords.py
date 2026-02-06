import math

import numpy as np

from app.dem.coords import (
  generate_square_grid,
  grid_indices_to_latlon,
  grid_indices_to_meters,
  latlon_to_meters,
  meters_to_latlon,
)


def test_local_crs_center_is_zero() -> None:
  lat, lon = 47.6062, -122.3321
  x, y = latlon_to_meters(lat, lon, lat, lon)
  assert abs(x) < 1e-6
  assert abs(y) < 1e-6


def test_roundtrip_latlon() -> None:
  origin_lat, origin_lon = 47.6062, -122.3321
  target_lat, target_lon = 47.6162, -122.3121
  x, y = latlon_to_meters(target_lat, target_lon, origin_lat, origin_lon)
  out_lat, out_lon = meters_to_latlon(x, y, origin_lat, origin_lon)

  assert math.isclose(out_lat, target_lat, abs_tol=1e-6)
  assert math.isclose(out_lon, target_lon, abs_tol=1e-6)


def test_generate_square_grid() -> None:
  grid = generate_square_grid(radius_m=100, resolution_m=50)

  assert grid.width == 5
  assert grid.height == 5
  assert np.allclose(grid.x_coords, [-100, -50, 0, 50, 100])
  assert np.allclose(grid.y_coords, [100, 50, 0, -50, -100])


def test_grid_indices_mapping() -> None:
  grid = generate_square_grid(radius_m=100, resolution_m=50)

  x0, y0 = grid_indices_to_meters(0, 0, grid)
  assert (x0, y0) == (-100.0, 100.0)

  x4, y4 = grid_indices_to_meters(4, 4, grid)
  assert (x4, y4) == (100.0, -100.0)


def test_grid_indices_to_latlon_center() -> None:
  origin_lat, origin_lon = 47.6062, -122.3321
  grid = generate_square_grid(radius_m=100, resolution_m=50)

  center_row = grid.height // 2
  center_col = grid.width // 2
  out_lat, out_lon = grid_indices_to_latlon(center_row, center_col, grid, origin_lat, origin_lon)

  assert math.isclose(out_lat, origin_lat, abs_tol=1e-6)
  assert math.isclose(out_lon, origin_lon, abs_tol=1e-6)
