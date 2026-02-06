from io import BytesIO

import numpy as np
from affine import Affine
from PIL import Image

from app.output import visibility_mask_to_png


def test_visibility_mask_to_png_dimensions_and_alpha() -> None:
  mask = np.array(
    [
      [True, False, True],
      [False, False, True],
    ],
    dtype=bool,
  )

  transform = Affine(1, 0, 0, 0, -1, 0)
  output = visibility_mask_to_png(mask, transform, crs="EPSG:4326", color_rgb=(10, 20, 30), alpha=120)

  image = Image.open(BytesIO(output.png_bytes))
  assert image.size == (3, 2)
  pixels = np.array(image)

  assert int(pixels[0, 0, 3]) == 120
  assert int(pixels[0, 1, 3]) == 0
  assert int(pixels[0, 2, 3]) == 120


def test_visibility_mask_metadata_bounds() -> None:
  mask = np.zeros((3, 2), dtype=bool)
  transform = Affine(10, 0, 100, 0, -10, 200)

  output = visibility_mask_to_png(mask, transform, crs="EPSG:4326")

  assert output.metadata.bounds == (100, 170, 120, 200)
  assert output.metadata.bounds_latlon == (170, 100, 200, 120)
