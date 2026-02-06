from __future__ import annotations

import math

import numpy as np


def compute_viewshed(
  dem: np.ndarray,
  observer_rc: tuple[int, int],
  observer_height_m: float,
  cell_size_m: float,
) -> np.ndarray:
  """
  Compute a boolean visibility mask for a DEM using line-of-sight sampling.

  dem: 2D array of elevations (meters)
  observer_rc: (row, col) index of the observer in the DEM
  observer_height_m: observer height above ground (meters)
  cell_size_m: square cell size (meters)
  """

  if dem.ndim != 2:
    raise ValueError("DEM must be a 2D array.")
  if cell_size_m <= 0:
    raise ValueError("cell_size_m must be positive.")
  if observer_height_m < 0:
    raise ValueError("observer_height_m must be non-negative.")

  rows, cols = dem.shape
  obs_r, obs_c = observer_rc
  if not (0 <= obs_r < rows and 0 <= obs_c < cols):
    raise IndexError("Observer index out of bounds.")

  observer_ground = float(dem[obs_r, obs_c])
  if math.isnan(observer_ground):
    raise ValueError("Observer elevation is NaN.")

  observer_elevation = observer_ground + observer_height_m

  visibility = np.zeros((rows, cols), dtype=bool)
  visibility[obs_r, obs_c] = True

  for r in range(rows):
    for c in range(cols):
      if r == obs_r and c == obs_c:
        continue
      target = float(dem[r, c])
      if math.isnan(target):
        continue
      if _line_of_sight(
        dem,
        observer_elevation,
        target,
        (obs_r, obs_c),
        (r, c),
        cell_size_m,
      ):
        visibility[r, c] = True

  return visibility


def _line_of_sight(
  dem: np.ndarray,
  observer_elevation: float,
  target_elevation: float,
  observer_rc: tuple[int, int],
  target_rc: tuple[int, int],
  cell_size_m: float,
) -> bool:
  obs_r, obs_c = observer_rc
  tgt_r, tgt_c = target_rc

  dr = tgt_r - obs_r
  dc = tgt_c - obs_c
  steps = int(max(abs(dr), abs(dc)))
  if steps == 0:
    return True

  # Sample along the line at grid-cell intervals.
  for step in range(1, steps):
    t = step / steps
    r = obs_r + dr * t
    c = obs_c + dc * t

    terrain = _bilinear_sample(dem, r, c)
    if math.isnan(terrain):
      return False

    # Height of line at this fraction of the distance.
    expected = observer_elevation + (target_elevation - observer_elevation) * t
    if terrain > expected:
      return False

  return True


def _bilinear_sample(dem: np.ndarray, row: float, col: float) -> float:
  rows, cols = dem.shape

  r0 = int(math.floor(row))
  c0 = int(math.floor(col))
  r1 = min(r0 + 1, rows - 1)
  c1 = min(c0 + 1, cols - 1)

  if r0 < 0 or c0 < 0 or r0 >= rows or c0 >= cols:
    return float("nan")

  dr = row - r0
  dc = col - c0

  e00 = float(dem[r0, c0])
  e10 = float(dem[r1, c0])
  e01 = float(dem[r0, c1])
  e11 = float(dem[r1, c1])

  if any(math.isnan(value) for value in (e00, e10, e01, e11)):
    return float("nan")

  return (
    e00 * (1 - dr) * (1 - dc)
    + e10 * dr * (1 - dc)
    + e01 * (1 - dr) * dc
    + e11 * dr * dc
  )
