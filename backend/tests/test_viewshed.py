import numpy as np

from app.viewshed import compute_viewshed


def test_flat_dem_all_visible() -> None:
  dem = np.zeros((5, 5), dtype=float)
  mask = compute_viewshed(dem, observer_rc=(2, 2), observer_height_m=1.0, cell_size_m=10.0)
  assert mask.shape == dem.shape
  assert mask.all()


def test_ridge_blocks_line_of_sight() -> None:
  dem = np.zeros((5, 5), dtype=float)
  dem[2, 3] = 5.0
  mask = compute_viewshed(dem, observer_rc=(2, 2), observer_height_m=1.0, cell_size_m=10.0)

  assert bool(mask[2, 3]) is True
  assert bool(mask[2, 4]) is False


def test_off_axis_peak_does_not_block() -> None:
  dem = np.zeros((5, 5), dtype=float)
  dem[1, 2] = 10.0
  mask = compute_viewshed(dem, observer_rc=(2, 2), observer_height_m=1.0, cell_size_m=10.0)

  assert bool(mask[2, 4]) is True
