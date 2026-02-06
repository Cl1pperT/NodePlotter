from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CACHE_VERSION = "v1"
DEFAULT_CACHE_DIR = Path.home() / "Downloads" / "LocalViewshedExplorer" / "data" / "viewshed"


@dataclass(frozen=True)
class CachedOverlay:
  png_bytes: bytes
  overlay_bounds_latlon: tuple[float, float, float, float]
  metadata: dict[str, Any]


@dataclass(frozen=True)
class CachedPayload:
  png_bytes: bytes
  payload: dict[str, Any]


def make_cache_key(
  observer_lat: float,
  observer_lon: float,
  observer_height_m: float,
  max_radius_km: float,
  resolution_m: float,
  dem_version: str,
) -> str:
  payload = {
    "cacheVersion": CACHE_VERSION,
    "demVersion": dem_version,
    "request": {
      "observer": {
        "lat": float(observer_lat),
        "lon": float(observer_lon),
      },
      "observerHeightM": float(observer_height_m),
      "maxRadiusKm": float(max_radius_km),
      "resolutionM": float(resolution_m),
    },
  }

  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def load_cached_viewshed(cache_key: str, cache_dir: Path | None = None) -> CachedOverlay | None:
  root = cache_dir or DEFAULT_CACHE_DIR
  entry_dir = root / cache_key
  metadata_path = entry_dir / "metadata.json"
  png_path = entry_dir / "overlay.png"

  if not metadata_path.exists() or not png_path.exists():
    return None

  try:
    metadata = json.loads(metadata_path.read_text())
    overlay = metadata.get("overlay")
    bounds = overlay.get("boundsLatLon") if isinstance(overlay, dict) else None
    if not bounds or len(bounds) != 4:
      return None
    png_bytes = png_path.read_bytes()
  except Exception:
    return None

  return CachedOverlay(
    png_bytes=png_bytes,
    overlay_bounds_latlon=(
      float(bounds[0]),
      float(bounds[1]),
      float(bounds[2]),
      float(bounds[3]),
    ),
    metadata=metadata.get("metadata", {}),
  )


def load_cached_payload(cache_key: str, cache_dir: Path | None = None) -> CachedPayload | None:
  root = cache_dir or DEFAULT_CACHE_DIR
  entry_dir = root / cache_key
  metadata_path = entry_dir / "metadata.json"
  png_path = entry_dir / "overlay.png"

  if not metadata_path.exists() or not png_path.exists():
    return None

  try:
    payload = json.loads(metadata_path.read_text())
    png_bytes = png_path.read_bytes()
  except Exception:
    return None

  return CachedPayload(png_bytes=png_bytes, payload=payload)


def store_cached_viewshed(
  cache_key: str,
  png_bytes: bytes,
  overlay_bounds_latlon: tuple[float, float, float, float],
  metadata: dict[str, Any],
  request_fingerprint: dict[str, Any],
  dem_version: str,
  cache_dir: Path | None = None,
) -> None:
  root = cache_dir or DEFAULT_CACHE_DIR
  entry_dir = root / cache_key
  entry_dir.mkdir(parents=True, exist_ok=True)

  png_path = entry_dir / "overlay.png"
  tmp_png = png_path.with_suffix(".png.tmp")
  tmp_png.write_bytes(png_bytes)
  tmp_png.replace(png_path)

  metadata_payload = {
    "cacheVersion": CACHE_VERSION,
    "demVersion": dem_version,
    "createdAt": datetime.now(timezone.utc).isoformat(),
    "request": request_fingerprint,
    "overlay": {"boundsLatLon": list(overlay_bounds_latlon)},
    "metadata": metadata,
  }

  metadata_path = entry_dir / "metadata.json"
  tmp_metadata = metadata_path.with_suffix(".json.tmp")
  tmp_metadata.write_text(json.dumps(metadata_payload, sort_keys=True, indent=2))
  tmp_metadata.replace(metadata_path)


def list_cached_viewsheds(limit: int = 50, cache_dir: Path | None = None) -> list[dict[str, Any]]:
  root = cache_dir or DEFAULT_CACHE_DIR
  if not root.exists():
    return []

  entries: list[dict[str, Any]] = []
  for entry_dir in root.iterdir():
    if not entry_dir.is_dir():
      continue
    metadata_path = entry_dir / "metadata.json"
    if not metadata_path.exists():
      continue
    try:
      payload = json.loads(metadata_path.read_text())
    except Exception:
      continue

    created_at = payload.get("createdAt")
    if not created_at:
      try:
        created_at = datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=timezone.utc).isoformat()
      except Exception:
        created_at = None

    overlay = payload.get("overlay") if isinstance(payload.get("overlay"), dict) else {}
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}

    entries.append(
      {
        "cacheKey": entry_dir.name,
        "createdAt": created_at,
        "demVersion": payload.get("demVersion"),
        "request": request,
        "boundsLatLon": overlay.get("boundsLatLon"),
      }
    )

  def sort_key(item: dict[str, Any]) -> float:
    created_at = item.get("createdAt")
    if not created_at:
      return 0.0
    try:
      return datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
    except Exception:
      return 0.0

  entries.sort(key=sort_key, reverse=True)
  return entries[: max(0, limit)]
