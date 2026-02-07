from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import numpy as np
from affine import Affine
from PIL import Image
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class OutputMetadata:
  crs: str
  bounds: tuple[float, float, float, float]
  bounds_latlon: tuple[float, float, float, float]
  width: int
  height: int


@dataclass(frozen=True)
class RasterOutput:
  png_bytes: bytes
  metadata: OutputMetadata


def visibility_mask_to_png(
  mask: np.ndarray,
  transform: Affine,
  crs: str,
  color_rgb: tuple[int, int, int] = (37, 99, 235),
  alpha: int = 96,
  background_rgb: tuple[int, int, int] = (255, 255, 255),
  background_alpha: int = 0,
) -> RasterOutput:
  """
  Convert a boolean visibility mask to a PNG with alpha and metadata.

  The output PNG is aligned with the provided affine transform and CRS.
  Visible cells are colored, invisible cells are fully transparent.
  """

  if mask.ndim != 2:
    raise ValueError("mask must be a 2D array.")
  if not (0 <= alpha <= 255):
    raise ValueError("alpha must be between 0 and 255.")
  if not (0 <= background_alpha <= 255):
    raise ValueError("background_alpha must be between 0 and 255.")

  height, width = mask.shape
  rgba = np.zeros((height, width, 4), dtype=np.uint8)
  rgba[:, :, :3] = np.array(background_rgb, dtype=np.uint8)
  rgba[:, :, 3] = background_alpha
  rgba[mask, :3] = np.array(color_rgb, dtype=np.uint8)
  rgba[mask, 3] = alpha

  image = Image.fromarray(rgba, mode="RGBA")
  buffer = BytesIO()
  image.save(buffer, format="PNG")

  bounds = _bounds_from_transform(transform, width, height)
  bounds_latlon = _bounds_to_latlon(bounds, crs)

  metadata = OutputMetadata(
    crs=crs,
    bounds=bounds,
    bounds_latlon=bounds_latlon,
    width=width,
    height=height,
  )

  return RasterOutput(png_bytes=buffer.getvalue(), metadata=metadata)


def visibility_counts_to_png(
  counts: np.ndarray,
  max_count: int,
  transform: Affine,
  crs: str,
  color_single: tuple[int, int, int] = (220, 38, 38),
  color_shared: tuple[int, int, int] = (250, 204, 21),
  color_all: tuple[int, int, int] = (34, 197, 94),
  alpha: int = 128,
  background_rgb: tuple[int, int, int] = (255, 255, 255),
  background_alpha: int = 0,
) -> RasterOutput:
  """
  Convert a visibility count raster into a PNG with alpha and metadata.

  - count == 1: single-observer visibility (red)
  - 1 < count < max_count: shared visibility (yellow)
  - count == max_count: visibility shared by all observers (green)
  """

  if counts.ndim != 2:
    raise ValueError("counts must be a 2D array.")
  if max_count < 1:
    raise ValueError("max_count must be at least 1.")
  if not (0 <= alpha <= 255):
    raise ValueError("alpha must be between 0 and 255.")
  if not (0 <= background_alpha <= 255):
    raise ValueError("background_alpha must be between 0 and 255.")

  height, width = counts.shape
  rgba = np.zeros((height, width, 4), dtype=np.uint8)
  rgba[:, :, :3] = np.array(background_rgb, dtype=np.uint8)
  rgba[:, :, 3] = background_alpha

  counts_int = counts.astype(np.int32, copy=False)
  mask_single = counts_int == 1
  mask_all = counts_int == max_count
  mask_shared = (counts_int > 1) & (counts_int < max_count)

  if mask_single.any():
    rgba[mask_single, :3] = np.array(color_single, dtype=np.uint8)
    rgba[mask_single, 3] = alpha
  if mask_shared.any():
    rgba[mask_shared, :3] = np.array(color_shared, dtype=np.uint8)
    rgba[mask_shared, 3] = alpha
  if mask_all.any():
    rgba[mask_all, :3] = np.array(color_all, dtype=np.uint8)
    rgba[mask_all, 3] = alpha

  image = Image.fromarray(rgba, mode="RGBA")
  buffer = BytesIO()
  image.save(buffer, format="PNG")

  bounds = _bounds_from_transform(transform, width, height)
  bounds_latlon = _bounds_to_latlon(bounds, crs)

  metadata = OutputMetadata(
    crs=crs,
    bounds=bounds,
    bounds_latlon=bounds_latlon,
    width=width,
    height=height,
  )

  return RasterOutput(png_bytes=buffer.getvalue(), metadata=metadata)


def _bounds_from_transform(
  transform: Affine,
  width: int,
  height: int,
) -> tuple[float, float, float, float]:
  x0, y0 = transform * (0, 0)
  x1, y1 = transform * (width, height)

  min_x = min(x0, x1)
  max_x = max(x0, x1)
  min_y = min(y0, y1)
  max_y = max(y0, y1)

  return (min_x, min_y, max_x, max_y)


def _bounds_to_latlon(
  bounds: tuple[float, float, float, float],
  crs: str,
) -> tuple[float, float, float, float]:
  min_x, min_y, max_x, max_y = bounds

  if crs.upper() in {"EPSG:4326", "CRS:84"}:
    return (min_y, min_x, max_y, max_x)

  transformer = Transformer.from_crs(CRS.from_user_input(crs), "EPSG:4326", always_xy=True)
  corners = [
    (min_x, min_y),
    (min_x, max_y),
    (max_x, max_y),
    (max_x, min_y),
  ]

  lats: list[float] = []
  lons: list[float] = []
  for x, y in corners:
    lon, lat = transformer.transform(x, y)
    lats.append(float(lat))
    lons.append(float(lon))

  return (min(lats), min(lons), max(lats), max(lons))
