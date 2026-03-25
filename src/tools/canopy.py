"""
fetch_canopy: fetch Meta Global Canopy Height for a given point + radius.

Resolution strategy (highest first):
  1. Meta alsgedi_global_v6_float — zoom-9 quadkey tiles, ~1.2m/px, uint8=meters, EPSG:3857
     Stream-read window directly from S3 (no full-tile download needed via COG).
  2. Meta alsgedi_global_v6_float_epsg4326_v3_10deg — 10° WGS84 tiles, ~27m/px, uint16=cm
     Same S3 bucket, anonymous access.
  3. Simulation fallback — reproducible seeded noise (labeled clearly).

Both real sources stream only the required window block from S3 using rasterio COG reads.

S3 dataset info
---------------
Bucket : dataforgood-fb-data  (public, anonymous)
1m path: forests/v1/alsgedi_global_v6_float/chm/{quadkey9}.tif
         CRS: EPSG:3857  |  dtype: uint8  |  value = meters (0–255)
         Tile naming: 9-char Bing quadkey (zoom 9, ~78km×78km per tile)
27m path: forests/v1/alsgedi_global_v6_float_epsg4326_v3_10deg/meta_chm_lat={top}_lon={left}_{type}.tif
          CRS: EPSG:4326  |  dtype: uint16  |  value = centimeters  |  nodata=65535
          Tile naming: top-left corner, 10°×10° tiles
"""

import math
import time
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

_META_S3_BUCKET   = "dataforgood-fb-data"
_PREFIX_1M        = "forests/v1/alsgedi_global_v6_float/chm"          # ~1.2m res, EPSG:3857
_PREFIX_27M       = "forests/v1/alsgedi_global_v6_float_epsg4326_v3_10deg"  # ~27m res, EPSG:4326
_TILE_TYPE_27M    = "median"
_NODATA_27M       = 65535
_SCALE_27M        = 0.01    # cm → m


def fetch_canopy(lat: float, lon: float, radius_m: float = 500) -> dict:
    """
    Fetch canopy height (meters above ground) for a bounding box around (lat, lon).

    Returns
    -------
    dict:
        array      : 2D np.ndarray, canopy height in meters (0 = bare ground)
        resolution : pixel size in meters
        bbox       : (W, S, E, N) in WGS84
        crs        : "EPSG:4326"
        source     : human-readable data source string
        simulated  : bool
        elapsed_s  : wall time in seconds
    """
    t0   = time.time()
    bbox = _bbox_from_point(lat, lon, radius_m)

    logger.info(f"[canopy] Fetching | center=({lat},{lon}) radius={radius_m}m")

    # ── Attempt 1: 1m high-resolution tile (EPSG:3857 stream) ───────────────
    # NOTE: The 1m Meta tiles use strip layout (blockysize=1), NOT Cloud-Optimised
    # GeoTIFF. A windowed read requires streaming ~12 MB of strips even for a small
    # 200m window, making each call ~40s. Enable explicitly when a local tile cache
    # is available or network speed is not a constraint.
    if getattr(fetch_canopy, "_use_1m", False):
        try:
            arr, res = _stream_1m_tile(lat, lon, bbox)
            elapsed  = time.time() - t0
            logger.info(
                f"[canopy] OK (1m) | shape={arr.shape} res≈{res:.1f}m "
                f"max={arr.max():.1f}m elapsed={elapsed:.2f}s"
            )
            return _result(arr, res, bbox, "Meta GCH 1m (S3 EPSG:3857 stream)", False, elapsed)
        except Exception as e:
            logger.warning(f"[canopy] 1m tile failed ({e}). Trying 27m tile…")

    # ── Attempt 2: 27m downsampled tile (EPSG:4326 stream) ──────────────────
    try:
        arr, res = _stream_27m_tile(lat, lon, bbox)
        elapsed  = time.time() - t0
        logger.info(
            f"[canopy] OK (27m) | shape={arr.shape} res≈{res:.1f}m "
            f"max={arr.max():.1f}m elapsed={elapsed:.2f}s"
        )
        return _result(arr, res, bbox, "Meta GCH 27m downsampled (S3 EPSG:4326 stream)", False, elapsed)
    except Exception as e:
        logger.warning(f"[canopy] 27m tile failed ({e}). Using simulation.")

    # ── Attempt 3: simulation fallback ──────────────────────────────────────
    arr, res = _simulate_canopy(lat, lon, radius_m)
    elapsed  = time.time() - t0
    logger.warning(f"[canopy] SIMULATED | shape={arr.shape} max={arr.max():.1f}m")
    return _result(arr, res, bbox, "SIMULATED (replace with real data)", True, elapsed)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bbox_from_point(lat: float, lon: float, radius_m: float) -> tuple:
    lat_deg = 1 / 111_320
    lon_deg = 1 / (111_320 * np.cos(np.radians(lat)))
    d_lat, d_lon = radius_m * lat_deg, radius_m * lon_deg
    return (lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat)


def _result(arr, res, bbox, source, simulated, elapsed):
    return {
        "array":     arr,
        "resolution": res,
        "bbox":      bbox,
        "crs":       "EPSG:4326",
        "source":    source,
        "simulated": simulated,
        "elapsed_s": elapsed,
    }


def _stream_1m_tile(lat: float, lon: float, bbox_4326: tuple) -> tuple[np.ndarray, float]:
    """
    Stream a window from the zoom-9 quadkey tile (~1.2m/px, EPSG:3857, uint8 = meters).
    Reprojects bbox from WGS84 to EPSG:3857 before the window lookup.
    """
    import mercantile
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds as tb
    from pyproj import Transformer

    tile = mercantile.tile(lon, lat, 9)
    qk   = mercantile.quadkey(tile)
    s3   = f"s3://{_META_S3_BUCKET}/{_PREFIX_1M}/{qk}.tif"

    # Reproject bbox WGS84 → EPSG:3857
    tr   = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    w4, s4, e4, n4 = bbox_4326
    x0, y0 = tr.transform(w4, s4)
    x1, y1 = tr.transform(e4, n4)
    bbox_3857 = (x0, y0, x1, y1)

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(s3) as src:
            win  = from_bounds(*bbox_3857, transform=src.transform)
            raw  = src.read(1, window=win)   # uint8, values = meters

    if raw.size == 0:
        raise ValueError(f"Empty window for tile {qk}")

    arr = raw.astype(np.float32)

    # Reproject from EPSG:3857 to EPSG:4326 so output matches terrain/buildings CRS
    w4, s4, e4, n4 = bbox_4326
    dst_h = max(raw.shape[0], 10)
    dst_w = max(raw.shape[1], 10)
    arr_wgs = np.zeros((dst_h, dst_w), dtype=np.float32)
    src_transform_3857 = from_bounds(*bbox_3857, dst_w, dst_h)  # approximate local transform

    # Simple: just return raw array in its native pixel dims with WGS84 metadata
    # The surface.py resampler will handle CRS alignment via rasterio.warp
    res_m = (x1 - x0) / raw.shape[1]  # m/px in EPSG:3857
    return arr, res_m


def _stream_27m_tile(lat: float, lon: float, bbox_4326: tuple) -> tuple[np.ndarray, float]:
    """Stream a window from the 10° WGS84 tile (~27m/px, uint16 = cm)."""
    import rasterio
    from rasterio.windows import from_bounds

    lat_tl = float(math.ceil(lat / 10) * 10)
    lon_tl = float(math.floor(lon / 10) * 10)
    tile_id = f"meta_chm_lat={lat_tl}_lon={lon_tl}_{_TILE_TYPE_27M}"
    s3 = f"s3://{_META_S3_BUCKET}/{_PREFIX_27M}/{tile_id}.tif"

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(s3) as src:
            win = from_bounds(*bbox_4326, transform=src.transform)
            raw = src.read(1, window=win)

    if raw.size == 0:
        raise ValueError(f"Empty window for 27m tile {tile_id}")

    arr = raw.astype(np.float32)
    arr[raw == _NODATA_27M] = 0.0
    arr = np.clip(arr * _SCALE_27M, 0, 100)

    w, s, e, n = bbox_4326
    res = (n - s) * 111_320 / arr.shape[0] if arr.shape[0] > 0 else float("nan")
    return arr, res


def _simulate_canopy(lat: float, lon: float, radius_m: float) -> tuple[np.ndarray, float]:
    """Reproducible seeded simulation — clearly flagged as not real data."""
    rng = np.random.default_rng(seed=abs(hash(f"{int(lat)},{int(lon)}")))
    px  = max(50, int(2 * radius_m))
    base = np.zeros((px, px), dtype=np.float32)
    for _ in range(rng.integers(3, 10)):
        cx, cy = rng.integers(5, px - 5, size=2)
        r_px   = rng.integers(3, 12)
        h      = rng.uniform(5, 25)
        yy, xx = np.ogrid[:px, :px]
        mask   = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_px ** 2
        base[mask] = np.maximum(base[mask], h * rng.uniform(0.7, 1.0, base[mask].shape))
    base += rng.uniform(0, 0.5, base.shape).astype(np.float32)
    return base, (2 * radius_m) / px
