from __future__ import annotations

import base64
import math
import os
import time
from typing import Any, Literal

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from pyproj import CRS, Transformer

from app.cache import (
  load_cached_payload,
  load_cached_viewshed,
  make_cache_key,
  make_cache_key_multi,
  store_cached_viewshed,
  list_cached_viewsheds,
  delete_cached_viewshed,
)
from app.dem import get_dem, get_dem_for_bbox, get_dem_version, get_dem_version_for_bbox
from app.output import RasterOutput, visibility_counts_to_png, visibility_mask_to_png
from app.viewshed import (
  compute_viewshed as compute_viewshed_baseline,
  compute_viewshed_radial,
  smooth_visibility_mask,
)

app = FastAPI(title="Local Viewshed Explorer API")

MAX_GRID_SIDE = 2000
WARN_CELL_COUNT = 1_000_000
MAX_CELL_COUNT = 4_000_000
MAX_OBSERVERS = 12
MAX_PARALLEL_WORKERS = max(1, (os.cpu_count() or 1))

app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
  ],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


class Observer(BaseModel):
  lat: float
  lon: float

  @field_validator("lat")
  @classmethod
  def validate_lat(cls, value: float) -> float:
    if not -90 <= value <= 90:
      raise ValueError("Latitude must be between -90 and 90.")
    return value

  @field_validator("lon")
  @classmethod
  def validate_lon(cls, value: float) -> float:
    if not -180 <= value <= 180:
      raise ValueError("Longitude must be between -180 and 180.")
    return value


class ViewshedRequest(BaseModel):
  observer: Observer
  observerHeightM: float = Field(gt=0)
  maxRadiusKm: float = Field(gt=0)
  resolutionM: float = Field(gt=0)


class MultiViewshedRequest(BaseModel):
  observers: list[Observer] = Field(min_length=2, max_length=MAX_OBSERVERS)
  observerHeightM: float = Field(gt=0)
  maxRadiusKm: float = Field(gt=0)
  resolutionM: float = Field(gt=0)


class ViewshedResponse(BaseModel):
  observer: Observer
  maxRadiusKm: float
  overlay: dict[str, Any]
  metadata: dict[str, Any]
  warnings: list[str]
  estimate: dict[str, Any]
  timings: dict[str, float] | None = None


class MultiViewshedResponse(BaseModel):
  observers: list[Observer]
  maxRadiusKm: float
  overlay: dict[str, Any]
  metadata: dict[str, Any]
  warnings: list[str]
  estimate: dict[str, Any]
  timings: dict[str, float] | None = None


class ViewshedHistoryItem(BaseModel):
  cacheKey: str
  createdAt: str | None = None
  demVersion: str | None = None
  request: dict[str, Any] | None = None
  boundsLatLon: list[float] | None = None


class ViewshedHistoryResponse(BaseModel):
  items: list[ViewshedHistoryItem]


class ViewshedCacheResponse(BaseModel):
  cacheKey: str
  createdAt: str | None = None
  demVersion: str | None = None
  request: dict[str, Any] | None = None
  overlay: dict[str, Any]
  metadata: dict[str, Any]


@app.get("/health")
def health_check() -> dict:
  return {"status": "ok"}


@app.get("/viewshed/history", response_model=ViewshedHistoryResponse)
def viewshed_history(limit: int = Query(50, ge=1, le=500)) -> ViewshedHistoryResponse:
  items = list_cached_viewsheds(limit=limit)
  return ViewshedHistoryResponse(items=items)


@app.get("/viewshed/cache/{cache_key}", response_model=ViewshedCacheResponse)
def viewshed_cache(cache_key: str) -> ViewshedCacheResponse:
  cached = load_cached_payload(cache_key)
  if cached is None:
    raise HTTPException(status_code=404, detail="Cached viewshed not found.")

  payload = cached.payload
  overlay = payload.get("overlay") if isinstance(payload.get("overlay"), dict) else {}
  bounds = overlay.get("boundsLatLon") if isinstance(overlay.get("boundsLatLon"), list) else None
  if not bounds or len(bounds) != 4:
    raise HTTPException(status_code=500, detail="Cached overlay metadata is invalid.")

  overlay_payload = {
    "pngBase64": base64.b64encode(cached.png_bytes).decode("ascii"),
    "boundsLatLon": bounds,
  }

  return ViewshedCacheResponse(
    cacheKey=cache_key,
    createdAt=payload.get("createdAt"),
    demVersion=payload.get("demVersion"),
    request=payload.get("request"),
    overlay=overlay_payload,
    metadata=payload.get("metadata", {}),
  )


@app.delete("/viewshed/cache/{cache_key}")
def delete_viewshed_cache(cache_key: str) -> dict:
  if not delete_cached_viewshed(cache_key):
    raise HTTPException(status_code=404, detail="Cached viewshed not found.")
  return {"status": "deleted", "cacheKey": cache_key}


@app.post("/viewshed", response_model=ViewshedResponse)
def compute_viewshed_endpoint(
  payload: ViewshedRequest,
  debug: int = Query(0, ge=0, le=1),
  mode: Literal["fast", "accurate"] = Query("accurate"),
) -> ViewshedResponse:
  timings: dict[str, float] = {}
  request_start = time.perf_counter()

  grid_side, cell_count = _estimate_grid(payload.maxRadiusKm, payload.resolutionM)
  warnings: list[str] = []

  if grid_side > MAX_GRID_SIDE or cell_count > MAX_CELL_COUNT:
    raise HTTPException(
      status_code=400,
      detail=(
        f"Requested grid {grid_side}x{grid_side} (~{cell_count:,} cells) exceeds "
        f"limit {MAX_GRID_SIDE}x{MAX_GRID_SIDE} (~{MAX_CELL_COUNT:,} cells). "
        "Reduce radius or increase resolution."
      ),
    )

  if cell_count > WARN_CELL_COUNT:
    warnings.append(
      f"Large request: estimated grid {grid_side}x{grid_side} (~{cell_count:,} cells). "
      "Computation may be slow."
    )

  dem_version = get_dem_version(
    observer_lat=payload.observer.lat,
    observer_lon=payload.observer.lon,
    radius_km=payload.maxRadiusKm,
    resolution_m=payload.resolutionM,
  )
  timings["dem_version_s"] = time.perf_counter() - request_start
  cache_key = make_cache_key(
    observer_lat=payload.observer.lat,
    observer_lon=payload.observer.lon,
    observer_height_m=payload.observerHeightM,
    max_radius_km=payload.maxRadiusKm,
    resolution_m=payload.resolutionM,
    dem_version=dem_version,
    algorithm=mode,
  )
  cached = load_cached_viewshed(cache_key)
  if cached is not None:
    overlay_payload = {
      "pngBase64": base64.b64encode(cached.png_bytes).decode("ascii"),
      "boundsLatLon": cached.overlay_bounds_latlon,
    }
    estimate = {
      "gridSide": grid_side,
      "cellCount": cell_count,
      "cacheHit": True,
    }
    timings["cache_hit_s"] = time.perf_counter() - request_start
    return ViewshedResponse(
      observer=payload.observer,
      maxRadiusKm=payload.maxRadiusKm,
      overlay=overlay_payload,
      metadata=cached.metadata,
      warnings=warnings,
      estimate=estimate,
      timings=timings if debug else None,
    )

  dem_start = time.perf_counter()
  try:
    dem_result = get_dem(
      observer_lat=payload.observer.lat,
      observer_lon=payload.observer.lon,
      radius_km=payload.maxRadiusKm,
      resolution_m=payload.resolutionM,
    )
  except Exception as exc:  # pragma: no cover - provider errors vary by environment
    raise HTTPException(status_code=502, detail=f"DEM provider error: {exc}") from exc
  timings["dem_fetch_s"] = time.perf_counter() - dem_start

  dem = dem_result.elevation
  if dem.size == 0 or not (dem.shape[0] and dem.shape[1]):
    raise HTTPException(status_code=500, detail="DEM response is empty.")

  if not (dem == dem).any():
    raise HTTPException(status_code=502, detail="DEM provider returned no valid data for this area.")

  observer_row, observer_col, cell_size_m = _observer_pixel_and_cell_size(
    payload.observer.lat,
    payload.observer.lon,
    dem_result.transform,
    dem_result.crs,
    dem.shape,
  )
  timings["dem_grid_rows"] = float(dem.shape[0])
  timings["dem_grid_cols"] = float(dem.shape[1])
  timings["dem_cell_size_m"] = float(cell_size_m)

  compute_start = time.perf_counter()
  try:
    if mode == "fast":
      visibility = compute_viewshed_radial(
        dem,
        observer_rc=(observer_row, observer_col),
        observer_height_m=payload.observerHeightM,
        cell_size_m=cell_size_m,
      )
    else:
      visibility = compute_viewshed_baseline(
        dem,
        observer_rc=(observer_row, observer_col),
        observer_height_m=payload.observerHeightM,
        cell_size_m=cell_size_m,
      )
  except Exception as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc

  if mode == "accurate":
    visibility = smooth_visibility_mask(visibility, passes=1, threshold=3)
  else:
    visibility = smooth_visibility_mask(visibility, passes=2, threshold=7)
  timings["viewshed_compute_s"] = time.perf_counter() - compute_start

  encode_start = time.perf_counter()
  overlay_output = visibility_mask_to_png(
    mask=visibility,
    transform=dem_result.transform,
    crs=dem_result.crs,
    color_rgb=(220, 38, 38),
    alpha=128,
    background_rgb=(255, 255, 255),
    background_alpha=0,
  )
  timings["encode_png_s"] = time.perf_counter() - encode_start

  overlay_payload, metadata_payload = _encode_overlay(overlay_output)
  cache_start = time.perf_counter()
  store_cached_viewshed(
    cache_key=cache_key,
    png_bytes=overlay_output.png_bytes,
    overlay_bounds_latlon=overlay_output.metadata.bounds_latlon,
    metadata=metadata_payload,
    request_fingerprint={
      "observer": {"lat": payload.observer.lat, "lon": payload.observer.lon},
      "observerHeightM": payload.observerHeightM,
      "maxRadiusKm": payload.maxRadiusKm,
      "resolutionM": payload.resolutionM,
      "mode": mode,
    },
    dem_version=dem_version,
  )
  timings["cache_store_s"] = time.perf_counter() - cache_start
  timings["total_s"] = time.perf_counter() - request_start
  estimate = {
    "gridSide": grid_side,
    "cellCount": cell_count,
    "cacheHit": False,
  }

  return ViewshedResponse(
    observer=payload.observer,
    maxRadiusKm=payload.maxRadiusKm,
    overlay=overlay_payload,
    metadata=metadata_payload,
    warnings=warnings,
    estimate=estimate,
    timings=timings if debug else None,
  )


_SHARED_DEM: np.ndarray | None = None
_SHARED_DEM_SHM: shared_memory.SharedMemory | None = None


def _init_shared_dem(shm_name: str, shape: tuple[int, int], dtype_str: str) -> None:
  global _SHARED_DEM, _SHARED_DEM_SHM
  _SHARED_DEM_SHM = shared_memory.SharedMemory(name=shm_name)
  _SHARED_DEM = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_SHARED_DEM_SHM.buf)


def _compute_visibility_shared(
  observer_rc: tuple[int, int],
  observer_height_m: float,
  cell_size_m: float,
  mode: Literal["fast", "accurate"],
  smooth_passes: int,
  smooth_threshold: int,
) -> np.ndarray:
  dem = _SHARED_DEM
  if dem is None:
    raise RuntimeError("Shared DEM not initialized.")

  if mode == "fast":
    visibility = compute_viewshed_radial(
      dem,
      observer_rc=observer_rc,
      observer_height_m=observer_height_m,
      cell_size_m=cell_size_m,
    )
  else:
    visibility = compute_viewshed_baseline(
      dem,
      observer_rc=observer_rc,
      observer_height_m=observer_height_m,
      cell_size_m=cell_size_m,
    )

  visibility = smooth_visibility_mask(visibility, passes=smooth_passes, threshold=smooth_threshold)
  return visibility.astype(np.uint8)


@app.post("/viewshed/multi", response_model=MultiViewshedResponse)
def compute_multi_viewshed_endpoint(
  payload: MultiViewshedRequest,
  debug: int = Query(0, ge=0, le=1),
  mode: Literal["fast", "accurate"] = Query("accurate"),
) -> MultiViewshedResponse:
  timings: dict[str, float] = {}
  request_start = time.perf_counter()

  observers = _normalize_observers(payload.observers)
  if len(observers) < 2:
    raise HTTPException(status_code=400, detail="At least two unique observer points are required.")
  if len(observers) > MAX_OBSERVERS:
    raise HTTPException(status_code=400, detail=f"Limit observers to {MAX_OBSERVERS} points.")

  radius_m = payload.maxRadiusKm * 1000.0
  bounds_m = _bounds_from_observers_meters(observers, radius_m)
  grid_width, grid_height, grid_side, cell_count = _estimate_grid_for_bounds(bounds_m, payload.resolutionM)
  warnings: list[str] = []

  if grid_side > MAX_GRID_SIDE or cell_count > MAX_CELL_COUNT:
    raise HTTPException(
      status_code=400,
      detail=(
        f"Requested grid {grid_width}x{grid_height} (~{cell_count:,} cells) exceeds "
        f"limit {MAX_GRID_SIDE}x{MAX_GRID_SIDE} (~{MAX_CELL_COUNT:,} cells). "
        "Reduce radius, reduce observer spread, or increase resolution."
      ),
    )

  if cell_count > WARN_CELL_COUNT:
    warnings.append(
      f"Large request: estimated grid {grid_width}x{grid_height} (~{cell_count:,} cells). "
      "Computation may be slow."
    )

  min_lat, min_lon, max_lat, max_lon = _bounds_to_latlon(bounds_m)
  dem_version = get_dem_version_for_bbox(
    min_lat=min_lat,
    min_lon=min_lon,
    max_lat=max_lat,
    max_lon=max_lon,
    resolution_m=payload.resolutionM,
  )
  timings["dem_version_s"] = time.perf_counter() - request_start
  observers_for_cache = [{"lat": obs.lat, "lon": obs.lon} for obs in observers]
  cache_key = make_cache_key_multi(
    observers=observers_for_cache,
    observer_height_m=payload.observerHeightM,
    max_radius_km=payload.maxRadiusKm,
    resolution_m=payload.resolutionM,
    dem_version=dem_version,
    algorithm=mode,
  )
  cached = load_cached_viewshed(cache_key)
  estimate = {
    "gridWidth": grid_width,
    "gridHeight": grid_height,
    "gridSide": grid_side,
    "cellCount": cell_count,
    "cacheHit": False,
  }
  if cached is not None:
    overlay_payload = {
      "pngBase64": base64.b64encode(cached.png_bytes).decode("ascii"),
      "boundsLatLon": cached.overlay_bounds_latlon,
    }
    estimate["cacheHit"] = True
    timings["cache_hit_s"] = time.perf_counter() - request_start
    return MultiViewshedResponse(
      observers=observers,
      maxRadiusKm=payload.maxRadiusKm,
      overlay=overlay_payload,
      metadata=cached.metadata,
      warnings=warnings,
      estimate=estimate,
      timings=timings if debug else None,
    )

  dem_start = time.perf_counter()
  try:
    dem_result = get_dem_for_bbox(
      min_lat=min_lat,
      min_lon=min_lon,
      max_lat=max_lat,
      max_lon=max_lon,
      resolution_m=payload.resolutionM,
    )
  except Exception as exc:  # pragma: no cover - provider errors vary by environment
    raise HTTPException(status_code=502, detail=f"DEM provider error: {exc}") from exc
  timings["dem_fetch_s"] = time.perf_counter() - dem_start

  dem = dem_result.elevation
  if dem.size == 0 or not (dem.shape[0] and dem.shape[1]):
    raise HTTPException(status_code=500, detail="DEM response is empty.")
  if not (dem == dem).any():
    raise HTTPException(status_code=502, detail="DEM provider returned no valid data for this area.")

  observer_pixels: list[tuple[int, int]] = []
  cell_size_m: float | None = None
  for obs in observers:
    row, col, cell_size = _observer_pixel_and_cell_size(
      obs.lat,
      obs.lon,
      dem_result.transform,
      dem_result.crs,
      dem.shape,
    )
    observer_pixels.append((row, col))
    if cell_size_m is None:
      cell_size_m = cell_size
  if cell_size_m is None:
    raise HTTPException(status_code=500, detail="Unable to determine DEM cell size.")

  timings["dem_grid_rows"] = float(dem.shape[0])
  timings["dem_grid_cols"] = float(dem.shape[1])
  timings["dem_cell_size_m"] = float(cell_size_m)

  compute_start = time.perf_counter()
  counts = np.zeros(dem.shape, dtype=np.uint16)
  smooth_passes = 1 if mode == "accurate" else 2
  smooth_threshold = 3 if mode == "accurate" else 7
  use_parallel = len(observer_pixels) > 1 and MAX_PARALLEL_WORKERS > 1

  if use_parallel:
    shm = shared_memory.SharedMemory(create=True, size=dem.nbytes)
    shm_array = np.ndarray(dem.shape, dtype=dem.dtype, buffer=shm.buf)
    shm_array[:] = dem
    try:
      max_workers = min(len(observer_pixels), MAX_PARALLEL_WORKERS)
      with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_shared_dem,
        initargs=(shm.name, dem.shape, str(dem.dtype)),
      ) as executor:
        futures = [
          executor.submit(
            _compute_visibility_shared,
            observer_rc,
            payload.observerHeightM,
            cell_size_m,
            mode,
            smooth_passes,
            smooth_threshold,
          )
          for observer_rc in observer_pixels
        ]
        for future in futures:
          try:
            counts += future.result().astype(np.uint16)
          except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
      shm.close()
      shm.unlink()
  else:
    for observer_rc in observer_pixels:
      try:
        if mode == "fast":
          visibility = compute_viewshed_radial(
            dem,
            observer_rc=observer_rc,
            observer_height_m=payload.observerHeightM,
            cell_size_m=cell_size_m,
          )
        else:
          visibility = compute_viewshed_baseline(
            dem,
            observer_rc=observer_rc,
            observer_height_m=payload.observerHeightM,
            cell_size_m=cell_size_m,
          )
      except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

      visibility = smooth_visibility_mask(visibility, passes=smooth_passes, threshold=smooth_threshold)
      counts += visibility.astype(np.uint16)

  timings["viewshed_compute_s"] = time.perf_counter() - compute_start

  encode_start = time.perf_counter()
  overlay_output = visibility_counts_to_png(
    counts=counts,
    max_count=len(observers),
    transform=dem_result.transform,
    crs=dem_result.crs,
    color_single=(220, 38, 38),
    color_shared=(250, 204, 21),
    color_all=(34, 197, 94),
    alpha=128,
    background_rgb=(255, 255, 255),
    background_alpha=0,
  )
  timings["encode_png_s"] = time.perf_counter() - encode_start

  overlay_payload, metadata_payload = _encode_overlay(overlay_output)
  cache_start = time.perf_counter()
  store_cached_viewshed(
    cache_key=cache_key,
    png_bytes=overlay_output.png_bytes,
    overlay_bounds_latlon=overlay_output.metadata.bounds_latlon,
    metadata=metadata_payload,
    request_fingerprint={
      "observers": observers_for_cache,
      "observerHeightM": payload.observerHeightM,
      "maxRadiusKm": payload.maxRadiusKm,
      "resolutionM": payload.resolutionM,
      "mode": mode,
    },
    dem_version=dem_version,
  )
  timings["cache_store_s"] = time.perf_counter() - cache_start
  timings["total_s"] = time.perf_counter() - request_start

  return MultiViewshedResponse(
    observers=observers,
    maxRadiusKm=payload.maxRadiusKm,
    overlay=overlay_payload,
    metadata=metadata_payload,
    warnings=warnings,
    estimate=estimate,
    timings=timings if debug else None,
  )


def _normalize_observers(observers: list[Observer]) -> list[Observer]:
  seen: set[tuple[float, float]] = set()
  unique: list[Observer] = []
  for obs in observers:
    key = (float(obs.lat), float(obs.lon))
    if key in seen:
      continue
    seen.add(key)
    unique.append(obs)
  return unique


def _bounds_from_observers_meters(
  observers: list[Observer],
  radius_m: float,
) -> tuple[float, float, float, float]:
  transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
  min_x = math.inf
  min_y = math.inf
  max_x = -math.inf
  max_y = -math.inf

  for obs in observers:
    x, y = transformer.transform(obs.lon, obs.lat)
    min_x = min(min_x, x - radius_m)
    min_y = min(min_y, y - radius_m)
    max_x = max(max_x, x + radius_m)
    max_y = max(max_y, y + radius_m)

  return (min_x, min_y, max_x, max_y)


def _bounds_to_latlon(bounds_m: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
  min_x, min_y, max_x, max_y = bounds_m
  transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
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


def _estimate_grid_for_bounds(
  bounds_m: tuple[float, float, float, float],
  resolution_m: float,
) -> tuple[int, int, int, int]:
  if resolution_m <= 0:
    return (0, 0, 0, 0)
  min_x, min_y, max_x, max_y = bounds_m
  width_m = max_x - min_x
  height_m = max_y - min_y
  grid_width = int(math.ceil(width_m / resolution_m) + 1)
  grid_height = int(math.ceil(height_m / resolution_m) + 1)
  grid_side = max(grid_width, grid_height)
  cell_count = grid_width * grid_height
  return grid_width, grid_height, grid_side, cell_count


def _observer_pixel_and_cell_size(
  lat: float,
  lon: float,
  transform,
  crs: str,
  shape: tuple[int, int],
) -> tuple[int, int, float]:
  transformer = Transformer.from_crs("EPSG:4326", CRS.from_user_input(crs), always_xy=True)
  x, y = transformer.transform(lon, lat)
  inv = ~transform
  col_f, row_f = inv * (x, y)
  row = int(round(row_f))
  col = int(round(col_f))
  rows, cols = shape

  if row < 0 or row >= rows or col < 0 or col >= cols:
    raise HTTPException(status_code=400, detail="Observer location is outside DEM coverage.")

  cell_size_x = float(abs(transform.a))
  cell_size_y = float(abs(transform.e))
  cell_size_m = (cell_size_x + cell_size_y) / 2.0
  if cell_size_m <= 0:
    raise HTTPException(status_code=500, detail="DEM cell size is invalid.")

  return row, col, cell_size_m


def _encode_overlay(output: RasterOutput) -> tuple[dict[str, Any], dict[str, Any]]:
  png_base64 = base64.b64encode(output.png_bytes).decode("ascii")
  overlay = {
    "pngBase64": png_base64,
    "boundsLatLon": output.metadata.bounds_latlon,
  }
  metadata = {
    "crs": output.metadata.crs,
    "bounds": output.metadata.bounds,
    "boundsLatLon": output.metadata.bounds_latlon,
    "width": output.metadata.width,
    "height": output.metadata.height,
  }
  return overlay, metadata


def _estimate_grid(radius_km: float, resolution_m: float) -> tuple[int, int]:
  if resolution_m <= 0:
    return (0, 0)
  radius_m = radius_km * 1000.0
  grid_side = int(math.ceil((2 * radius_m) / resolution_m) + 1)
  cell_count = grid_side * grid_side
  return grid_side, cell_count
