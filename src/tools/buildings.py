"""
fetch_buildings: building footprints + height for a given point + radius.

Source priority
---------------
1. Google Open Buildings v3 (primary)
   - URL  : gs://open-buildings-data/v3/polygons_s2_level_4_gzip/{s2_token}_buildings.csv.gz
   - Schema: latitude, longitude, area_in_meters, confidence, geometry (WKT), full_plus_code
   - Heights: NOT provided — estimated from footprint area (see _area_to_height)
   - Tile  : S2 level-4 cell (~1000 km² each), streamed and bbox-filtered
   - Cache : filtered results saved as GeoParquet per S2 token

2. Microsoft Global ML Building Footprints (fallback)
   - URL  : https://minedbuildings.z5.web.core.windows.net/global-buildings/
   - Schema: height (meters, ML-predicted), confidence, geometry (WKT)
   - Tile  : Bing quadkey zoom-9, downloaded as GeoJSONL.gz and cached as GeoParquet

Height resolution
-----------------
  Google source → no heights → estimate from area_in_meters:
    < 20 m²  : 3.0 m  (shed / kiosk)
    20–100   : 4.5 m  (small residential, ~1.5 floors)
    100–500  : 6.0 m  (residential, ~2 floors)
    500–2000 : 9.0 m  (commercial, ~3 floors)
    > 2000   : 12.0 m (industrial / large commercial)

  Microsoft source → use ML-predicted height when ≥ 2.5 m; area heuristic otherwise.
"""

import io
import csv
import gzip
import json
import time
import logging
import urllib.request
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import shape, box
from shapely.wkt import loads as wkt_loads

logger = logging.getLogger(__name__)

_CACHE_DIR          = Path(__file__).resolve().parents[2] / "data" / "buildings"
_GOOGLE_BASE        = "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip"
_MS_INDEX_URL       = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
_MS_TILE_ZOOM       = 9
_MIN_HEIGHT         = 2.5   # metres — discard implausible ML predictions below this
_DEFAULT_HEIGHT     = 8.0


def fetch_buildings(lat: float, lon: float, radius_m: float = 500) -> dict:
    """
    Fetch building footprints + heights within radius_m of (lat, lon).

    Returns
    -------
    dict:
        gdf        : GeoDataFrame (EPSG:4326), columns [geometry, height_m, source_height]
        count      : number of buildings
        bbox       : (W, S, E, N)
        source     : data source description
        elapsed_s  : wall time
    """
    t0   = time.time()
    bbox = _bbox_from_point(lat, lon, radius_m)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Attempt 1: Microsoft Building Footprints (primary for CONUS) ─────────
    # Note: Google Open Buildings has sparse coverage in the US — most tiles contain
    # rural data only and miss major metros. Microsoft has full CONUS coverage with
    # ML-predicted heights, making it the better primary source for US locations.
    # Google Open Buildings is retained as fallback for non-CONUS locations.
    try:
        gdf, src_label = _fetch_microsoft(lat, lon, bbox)
        gdf = _resolve_heights_ms(gdf)
        elapsed = time.time() - t0
        logger.info(f"[buildings] Microsoft OK | count={len(gdf)} elapsed={elapsed:.2f}s")
        return _result(gdf, bbox, src_label, elapsed)
    except Exception as e:
        logger.warning(f"[buildings] Microsoft failed ({e}). Trying Google…")

    # ── Attempt 2: Google Open Buildings (fallback, better for non-CONUS) ────
    try:
        gdf, src_label = _fetch_google(lat, lon, bbox)
        gdf = _resolve_heights_google(gdf)
        if len(gdf) == 0:
            raise ValueError("Google tile found but contains 0 buildings in bbox "
                             "(tile may not cover this region yet)")
        elapsed = time.time() - t0
        logger.info(f"[buildings] Google OK | count={len(gdf)} elapsed={elapsed:.2f}s")
        return _result(gdf, bbox, src_label, elapsed)
    except Exception as e:
        logger.warning(f"[buildings] Google failed ({e}). Returning empty GDF.")

    # ── Empty fallback ───────────────────────────────────────────────────────
    empty = gpd.GeoDataFrame({"geometry": [], "height_m": [], "source_height": []},
                             crs="EPSG:4326")
    return _result(empty, bbox, "none (all sources failed)", time.time() - t0)


# ---------------------------------------------------------------------------
# Google Open Buildings
# ---------------------------------------------------------------------------

def _fetch_google(lat, lon, bbox):
    import s2sphere
    cell  = s2sphere.CellId.from_lat_lng(
                s2sphere.LatLng.from_degrees(lat, lon)).parent(4)
    token = cell.to_token()
    cache = _CACHE_DIR / f"google_{token}.parquet"

    if cache.exists():
        logger.info(f"[buildings] Google cache hit: {cache}")
        tile_gdf = gpd.read_parquet(cache)
    else:
        url = f"{_GOOGLE_BASE}/{token}_buildings.csv.gz"
        logger.info(f"[buildings] Downloading Google tile {token} → {url}")
        tile_gdf = _stream_google_tile(url, token, cache)

    # Filter to bbox
    bbox_geom = box(*bbox)
    gdf = tile_gdf[tile_gdf.geometry.intersects(bbox_geom)].copy().reset_index(drop=True)
    return gdf, f"Google Open Buildings v3 (S2 token={token})"


def _stream_google_tile(url: str, token: str, cache: Path) -> gpd.GeoDataFrame:
    rows = []
    with urllib.request.urlopen(url, timeout=60) as resp:
        with gzip.open(resp) as gz:
            reader = csv.DictReader(io.TextIOWrapper(gz))
            for row in reader:
                try:
                    geom = wkt_loads(row["geometry"])
                    rows.append({
                        "geometry":       geom,
                        "area_in_meters": float(row.get("area_in_meters", 0) or 0),
                        "confidence":     float(row.get("confidence", 0) or 0),
                    })
                except Exception:
                    continue

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_parquet(cache)
    logger.info(f"[buildings] Cached {len(gdf)} Google buildings → {cache}")
    return gdf


def _resolve_heights_google(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Estimate building height from footprint area (Google has no heights)."""
    if gdf.empty:
        gdf = gdf.copy()
        gdf["height_m"]      = np.array([], dtype=float)
        gdf["source_height"] = np.array([], dtype=str)
        return gdf

    def area_to_height(a):
        if a < 20:   return 3.0
        if a < 100:  return 4.5
        if a < 500:  return 6.0
        if a < 2000: return 9.0
        return 12.0

    gdf = gdf.copy()
    gdf["height_m"]      = gdf["area_in_meters"].apply(area_to_height).astype(float)
    gdf["source_height"] = "google_area_heuristic"
    return gdf


# ---------------------------------------------------------------------------
# Microsoft Building Footprints
# ---------------------------------------------------------------------------

def _fetch_microsoft(lat, lon, bbox):
    import mercantile
    tile  = mercantile.tile(lon, lat, _MS_TILE_ZOOM)
    qk    = mercantile.quadkey(tile)
    cache = _CACHE_DIR / f"{qk}.parquet"

    if cache.exists():
        logger.info(f"[buildings] MS cache hit: {cache}")
        tile_gdf = gpd.read_parquet(cache)
    else:
        url = _get_ms_tile_url(qk)
        tile_gdf = _download_ms_tile(url, qk, cache)

    bbox_geom = box(*bbox)
    gdf = tile_gdf[tile_gdf.geometry.intersects(bbox_geom)].copy().reset_index(drop=True)
    return gdf, f"Microsoft Global ML Building Footprints (quadkey={qk})"


def _get_ms_tile_url(quadkey: str) -> str:
    with urllib.request.urlopen(_MS_INDEX_URL) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode())))
    match = next((row for row in rows if row["QuadKey"] == quadkey), None)
    if match is None:
        raise ValueError(f"QuadKey {quadkey} not found in MS index")
    logger.info(f"[buildings] MS tile URL: {match['Url']}  size={match['Size']}")
    return match["Url"]


def _download_ms_tile(url: str, quadkey: str, cache: Path) -> gpd.GeoDataFrame:
    logger.info(f"[buildings] Downloading MS tile {quadkey}…")
    features = []
    with urllib.request.urlopen(url) as resp:
        with gzip.open(resp) as gz:
            for line in gz:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                feat  = json.loads(line)
                props = feat.get("properties", {})
                geom  = feat.get("geometry")
                if geom is None:
                    continue
                features.append({
                    "geometry":   shape(geom),
                    "height":     props.get("height"),
                    "confidence": props.get("confidence"),
                })
    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    gdf.to_parquet(cache)
    logger.info(f"[buildings] Cached {len(gdf)} MS buildings → {cache}")
    return gdf


def _resolve_heights_ms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        gdf["height_m"]      = []
        gdf["source_height"] = []
        return gdf

    gdf_m = gdf.to_crs("EPSG:3857")
    areas = gdf_m.geometry.area

    heights = []
    labels  = []
    raw_h   = gdf["height"].values if "height" in gdf.columns else [None] * len(gdf)

    for h, area in zip(raw_h, areas):
        try:
            hf = float(h)
            if hf >= _MIN_HEIGHT:
                heights.append(hf);  labels.append("ms_predicted")
                continue
        except (TypeError, ValueError):
            pass
        floors = max(1, min(int(area / 150), 10))
        heights.append(floors * 3.5)
        labels.append("area_heuristic")

    gdf = gdf.copy()
    gdf["height_m"]      = np.clip(heights, _MIN_HEIGHT, 150.0)
    gdf["source_height"] = labels
    return gdf


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bbox_from_point(lat: float, lon: float, radius_m: float) -> tuple:
    lat_deg = 1 / 111_320
    lon_deg = 1 / (111_320 * np.cos(np.radians(lat)))
    d_lat, d_lon = radius_m * lat_deg, radius_m * lon_deg
    return (lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat)


def _result(gdf, bbox, source, elapsed):
    n = len(gdf)
    if n > 0:
        h_min = gdf["height_m"].min()
        h_max = gdf["height_m"].max()
        logger.info(f"[buildings] OK | count={n} heights=[{h_min:.1f}–{h_max:.1f}]m "
                    f"source={source} elapsed={elapsed:.2f}s")
    else:
        logger.info(f"[buildings] OK | count=0 source={source}")
    return {"gdf": gdf, "count": n, "bbox": bbox, "source": source, "elapsed_s": elapsed}
