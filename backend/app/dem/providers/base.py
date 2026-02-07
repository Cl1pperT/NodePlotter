from __future__ import annotations

from abc import ABC, abstractmethod

from app.dem.types import DemResult


class DemProvider(ABC):
  @abstractmethod
  def get_dem(
    self,
    observer_lat: float,
    observer_lon: float,
    radius_km: float,
    resolution_m: float,
  ) -> DemResult:
    raise NotImplementedError

  def get_dem_for_bbox(
    self,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    resolution_m: float,
  ) -> DemResult:
    raise NotImplementedError("This provider does not support bounding-box DEM requests.")
