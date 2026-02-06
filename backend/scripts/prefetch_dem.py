#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.dem.providers.terrarium import TerrariumProvider

# Utah state bounding box (approximate state extremes in degrees).
UTAH_BBOX = {
  "min_lat": 37.0,
  "min_lon": -114.05,
  "max_lat": 42.0,
  "max_lon": -109.0,
}

PRESETS = {
  "fast": 90.0,
  "medium": 60.0,
  "high": 30.0,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Prefetch Terrarium DEM tiles for a bounding box.")
  parser.add_argument("--min-lat", type=float, help="Minimum latitude")
  parser.add_argument("--min-lon", type=float, help="Minimum longitude")
  parser.add_argument("--max-lat", type=float, help="Maximum latitude")
  parser.add_argument("--max-lon", type=float, help="Maximum longitude")
  parser.add_argument(
    "--preset",
    choices=sorted(PRESETS.keys()),
    default="fast",
    help="Resolution preset (meters)",
  )
  parser.add_argument("--resolution-m", type=float, help="Resolution in meters per cell (overrides preset)")
  parser.add_argument(
    "--resume",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Resume from previous failed tiles list when available.",
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=8,
    help="Number of parallel download workers.",
  )
  parser.add_argument(
    "--state",
    choices=["utah"],
    default="utah",
    help="Prefetch preset bounding box (currently only Utah)",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  if args.min_lat is None or args.min_lon is None or args.max_lat is None or args.max_lon is None:
    if args.state == "utah":
      min_lat = UTAH_BBOX["min_lat"]
      min_lon = UTAH_BBOX["min_lon"]
      max_lat = UTAH_BBOX["max_lat"]
      max_lon = UTAH_BBOX["max_lon"]
    else:
      raise SystemExit("Missing bounding box. Provide --min-lat/--min-lon/--max-lat/--max-lon.")
  else:
    min_lat = args.min_lat
    min_lon = args.min_lon
    max_lat = args.max_lat
    max_lon = args.max_lon

  resolution_m = float(args.resolution_m) if args.resolution_m else PRESETS[args.preset]

  # The provider expects an explicit cache dir; use the default from the DEM module.
  from app.dem import DEFAULT_CACHE_DIR

  provider = TerrariumProvider(cache_dir=DEFAULT_CACHE_DIR)
  print(f"Using cache dir: {DEFAULT_CACHE_DIR}")
  DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

  zoom, min_x, max_x, min_y, max_y = provider.tile_range_for_bbox(
    min_lat=min_lat,
    min_lon=min_lon,
    max_lat=max_lat,
    max_lon=max_lon,
    resolution_m=resolution_m,
  )
  manifest_path = _manifest_path(DEFAULT_CACHE_DIR, args.state, zoom, resolution_m)
  failed_tiles = _load_failed_tiles(manifest_path, min_lat, min_lon, max_lat, max_lon, resolution_m, zoom) if args.resume else []

  if failed_tiles:
    print(f"Resuming {len(failed_tiles)} failed tiles from {manifest_path.name}")

  tiles_downloaded = 0
  tiles_cached = 0
  tiles_failed = 0
  remaining_failed: set[tuple[int, int]] = set()

  tile_queue: list[tuple[int, int]] = []
  seen: set[tuple[int, int]] = set()

  for tile in failed_tiles:
    if tile not in seen:
      tile_queue.append(tile)
      seen.add(tile)

  for ty in range(min_y, max_y + 1):
    for tx in range(min_x, max_x + 1):
      tile = (tx, ty)
      if tile not in seen:
        tile_queue.append(tile)
        seen.add(tile)

  total_tiles = len(tile_queue)
  if total_tiles == 0:
    print("Nothing to fetch. All tiles are cached.")
  else:
    print(f"Fetching {total_tiles} tiles with {args.workers} workers...")

  def fetch_one(tile: tuple[int, int]) -> tuple[str, tuple[int, int]]:
    tx, ty = tile
    status = provider.fetch_tile(zoom, tx, ty)
    return status, tile

  completed = 0
  with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
    futures = [executor.submit(fetch_one, tile) for tile in tile_queue]
    for future in as_completed(futures):
      status, tile = future.result()
      if status == "downloaded":
        tiles_downloaded += 1
      elif status == "cached":
        tiles_cached += 1
      else:
        tiles_failed += 1
        remaining_failed.add(tile)

      completed += 1
      if completed % 200 == 0 or completed == total_tiles:
        print(f"  progress: {completed}/{total_tiles}")

  tile_count = (max_x - min_x + 1) * (max_y - min_y + 1)
  stats = {
    "zoom": zoom,
    "tile_count": tile_count,
    "tiles_downloaded": tiles_downloaded,
    "tiles_cached": tiles_cached,
    "tiles_failed": tiles_failed,
    "tile_range": (min_x, max_x, min_y, max_y),
  }

  _write_manifest(
    manifest_path,
    min_lat,
    min_lon,
    max_lat,
    max_lon,
    resolution_m,
    zoom,
    sorted(remaining_failed),
  )

  print("Prefetch complete:")
  for key, value in stats.items():
    print(f"  {key}: {value}")
  if remaining_failed:
    print(f"  failed_manifest: {manifest_path}")

def _manifest_path(cache_dir: Path, label: str, zoom: int, resolution_m: float) -> Path:
  safe_label = label.replace(" ", "_").lower()
  res_tag = str(resolution_m).replace(".", "p")
  return cache_dir / f"prefetch_{safe_label}_z{zoom}_res{res_tag}.json"


def _load_failed_tiles(
  manifest_path: Path,
  min_lat: float,
  min_lon: float,
  max_lat: float,
  max_lon: float,
  resolution_m: float,
  zoom: int,
) -> list[tuple[int, int]]:
  if not manifest_path.exists():
    return []
  try:
    data = json.loads(manifest_path.read_text())
  except Exception:
    return []

  expected = {
    "min_lat": min_lat,
    "min_lon": min_lon,
    "max_lat": max_lat,
    "max_lon": max_lon,
    "resolution_m": resolution_m,
    "zoom": zoom,
  }
  if any(data.get(key) != value for key, value in expected.items()):
    return []
  failed = data.get("failed_tiles", [])
  if not isinstance(failed, list):
    return []
  result: list[tuple[int, int]] = []
  for item in failed:
    if isinstance(item, list) and len(item) == 2:
      result.append((int(item[0]), int(item[1])))
  return result


def _write_manifest(
  manifest_path: Path,
  min_lat: float,
  min_lon: float,
  max_lat: float,
  max_lon: float,
  resolution_m: float,
  zoom: int,
  failed_tiles: list[tuple[int, int]],
) -> None:
  payload = {
    "min_lat": min_lat,
    "min_lon": min_lon,
    "max_lat": max_lat,
    "max_lon": max_lon,
    "resolution_m": resolution_m,
    "zoom": zoom,
    "failed_tiles": failed_tiles,
  }
  manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
