from __future__ import annotations

import base64
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from pyproj import CRS, Transformer

from app.dem import get_dem
from app.output import RasterOutput, visibility_mask_to_png
from app.viewshed import compute_viewshed

app = FastAPI(title="Local Viewshed Explorer API")

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


class ViewshedResponse(BaseModel):
  observer: Observer
  maxRadiusKm: float
  overlay: dict[str, Any]
  metadata: dict[str, Any]


@app.get("/health")
def health_check() -> dict:
  return {"status": "ok"}


@app.post("/viewshed", response_model=ViewshedResponse)
def compute_viewshed(payload: ViewshedRequest) -> ViewshedResponse:
  try:
    dem_result = get_dem(
      observer_lat=payload.observer.lat,
      observer_lon=payload.observer.lon,
      radius_km=payload.maxRadiusKm,
      resolution_m=payload.resolutionM,
    )
  except Exception as exc:  # pragma: no cover - provider errors vary by environment
    raise HTTPException(status_code=502, detail=f"DEM provider error: {exc}") from exc

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

  try:
    visibility = compute_viewshed(
      dem,
      observer_rc=(observer_row, observer_col),
      observer_height_m=payload.observerHeightM,
      cell_size_m=cell_size_m,
    )
  except Exception as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc

  overlay_output = visibility_mask_to_png(
    mask=visibility,
    transform=dem_result.transform,
    crs=dem_result.crs,
  )

  overlay_payload, metadata_payload = _encode_overlay(overlay_output)

  return ViewshedResponse(
    observer=payload.observer,
    maxRadiusKm=payload.maxRadiusKm,
    overlay=overlay_payload,
    metadata=metadata_payload,
  )


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
