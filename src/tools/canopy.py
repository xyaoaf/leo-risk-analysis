"""
fetch_canopy: fetch Meta 1m Global Canopy Height for a given point + radius.

Resolution strategy (highest first):
  1. GEE Meta Trees 1m — projects/sat-io/open-datasets/facebook/meta-canopy-height
     Tolan et al. 2024 (Remote Sensing of Environment, 300, p.113888).
     Genuinely 1m resolution derived from Maxar satellite imagery via DiNOv2.
     Accessed via Google Earth Engine computePixels API (~0.2s per 200m window).
     Results cached locally as GeoTIFF keyed by bbox to avoid repeated API calls.

  2. Meta ALSGEDI 27m downsampled — dataforgood-fb-data S3 (EPSG:4326, ~27m/px)
     Fallback when GEE is unavailable (no credentials, quota exceeded, etc.).

  3. Simulation fallback — reproducible seeded noise (labelled clearly).

GEE licensing note
------------------
  Current use: personal non-profit GEE account (development / challenge prototype).
  Production deployment: requires a Google Earth Engine commercial license.
  Set GEE_PROJECT env var or pass project= to fetch_canopy() to override project.

GEE setup (one-time):
  pip install earthengine-api
  earthengine authenticate        # opens browser, sets ~/.config/earthengine/credentials
"""

import io
import math
import time
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_GEE_ASSET    = "projects/sat-io/open-datasets/facebook/meta-canopy-height"
_GEE_BAND     = "cover_code"           # uint8, values = canopy height in metres
_GEE_PROJECT  = os.environ.get("GEE_PROJECT", "ardent-fusion-421917")

_META_S3_BUCKET = "dataforgood-fb-data"
_PREFIX_27M     = "forests/v1/alsgedi_global_v6_float_epsg4326_v3_10deg"
_TILE_TYPE_27M  = "median"
_NODATA_27M     = 65535
_SCALE_27M      = 0.01    # cm → m

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "tiles" / "canopy"

# GEE ImageCollection handle — initialised once on first use
_gee_col = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_canopy(
    lat: float,
    lon: float,
    radius_m: float = 500,
    project: str | None = None,
) -> dict:
    """
    Fetch canopy height (metres above ground) for a bounding box around (lat, lon).

    Parameters
    ----------
    lat, lon   : WGS84 decimal degrees
    radius_m   : analysis radius in metres
    project    : GEE Cloud project ID override (default: GEE_PROJECT env var)

    Returns
    -------
    dict:
        array      : 2D np.ndarray, canopy height in metres (0 = bare ground)
        resolution : pixel size in metres
        bbox       : (W, S, E, N) in WGS84
        crs        : "EPSG:4326"
        source     : human-readable data source string
        simulated  : bool
        elapsed_s  : wall time in seconds
    """
    t0   = time.time()
    bbox = _bbox_from_point(lat, lon, radius_m)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"[canopy] Fetching | center=({lat},{lon}) radius={radius_m}m")

    # ── Attempt 1: GEE Meta Trees 1m ─────────────────────────────────────────
    try:
        arr, res = _fetch_gee_1m(lat, lon, bbox, project or _GEE_PROJECT)
        elapsed  = time.time() - t0
        logger.info(
            f"[canopy] OK (GEE 1m) | shape={arr.shape} res≈{res:.2f}m "
            f"max={arr.max():.1f}m elapsed={elapsed:.2f}s"
        )
        return _result(arr, res, bbox,
                       "Meta Trees 1m GEE (Tolan et al. 2024, projects/sat-io/open-datasets/facebook/meta-canopy-height)",
                       False, elapsed)
    except Exception as e:
        logger.warning(f"[canopy] GEE 1m failed ({e}). Trying 27m S3 tile…")

    # ── Attempt 2: ALSGEDI 27m S3 tile ───────────────────────────────────────
    try:
        arr, res = _stream_27m_tile(lat, lon, bbox)
        elapsed  = time.time() - t0
        logger.info(
            f"[canopy] OK (27m S3) | shape={arr.shape} res≈{res:.1f}m "
            f"max={arr.max():.1f}m elapsed={elapsed:.2f}s"
        )
        return _result(arr, res, bbox,
                       "Meta ALSGEDI 27m downsampled (S3 EPSG:4326 stream)",
                       False, elapsed)
    except Exception as e:
        logger.warning(f"[canopy] 27m S3 tile failed ({e}). Using simulation.")

    # ── Attempt 3: simulation fallback ───────────────────────────────────────
    arr, res = _simulate_canopy(lat, lon, radius_m)
    elapsed  = time.time() - t0
    logger.warning(f"[canopy] SIMULATED | shape={arr.shape} max={arr.max():.1f}m")
    return _result(arr, res, bbox, "SIMULATED (replace with real data)", True, elapsed)


# ---------------------------------------------------------------------------
# GEE 1m fetcher
# ---------------------------------------------------------------------------

def _gee_init(project: str):
    """Initialise Earth Engine once; re-use handle on subsequent calls."""
    global _gee_col
    if _gee_col is not None:
        return _gee_col
    import ee
    ee.Initialize(project=project)
    _gee_col = ee.ImageCollection(_GEE_ASSET)
    logger.info(f"[canopy] GEE initialised | project={project} asset={_GEE_ASSET}")
    return _gee_col


def _fetch_gee_1m(
    lat: float,
    lon: float,
    bbox_4326: tuple,
    project: str,
) -> tuple[np.ndarray, float]:
    """
    Fetch a window from the GEE Meta Trees 1m ImageCollection.

    Uses ee.data.computePixels (NPY format) — no intermediate file download.
    Results are cached as GeoTIFF under data/tiles/canopy/ keyed by bbox.

    Returns (array_metres, resolution_m).
    """
    w, s, e, n = bbox_4326

    # Cache key: rounded bbox at ~1m precision
    cache_key = f"gee1m_{lat:.5f}_{lon:.5f}_{int(round((e-w)*111320/2))}.tif"
    cache_path = _CACHE_DIR / cache_key

    if cache_path.exists():
        arr, res = _read_cached_tif(cache_path, bbox_4326)
        logger.info(f"[canopy] GEE cache hit: {cache_path.name} | max={arr.max():.1f}m res={res:.2f}m")
        return arr, res

    # Compute target pixel grid: ~1m resolution in EPSG:4326 degrees
    # 1m ≈ 1/111320 deg latitude; adjust longitude by cos(lat)
    lat_rad  = np.radians(lat)
    deg_per_m_lat = 1.0 / 111_320.0
    deg_per_m_lon = 1.0 / (111_320.0 * np.cos(lat_rad))
    height_m = (n - s) / deg_per_m_lat
    width_m  = (e - w) / deg_per_m_lon
    npx_h    = max(10, int(round(height_m)))   # rows ≈ metres north–south
    npx_w    = max(10, int(round(width_m)))    # cols ≈ metres east–west

    dx = (e - w) / npx_w
    dy = (n - s) / npx_h

    import ee
    col    = _gee_init(project)
    region = ee.Geometry.Rectangle([w, s, e, n])
    mosaic = col.filterBounds(region).mosaic()

    raw = ee.data.computePixels({
        "expression": mosaic,
        "fileFormat": "NPY",
        "bandIds":    [_GEE_BAND],
        "grid": {
            "dimensions":      {"width": npx_w, "height": npx_h},
            "affineTransform": {
                "scaleX":    dx,
                "shearX":    0,
                "translateX": w,
                "shearY":    0,
                "scaleY":    -dy,
                "translateY": n,
            },
            "crsCode": "EPSG:4326",
        },
    })

    structured = np.load(io.BytesIO(raw))
    arr = structured[_GEE_BAND].astype(np.float32)   # height in metres

    res = height_m / npx_h   # metres per pixel (north–south direction)

    # Cache result as GeoTIFF for future calls
    _write_cached_tif(arr, bbox_4326, cache_path)
    logger.info(f"[canopy] GEE cached → {cache_path.name}")

    return arr, res


# ---------------------------------------------------------------------------
# Local GeoTIFF cache helpers
# ---------------------------------------------------------------------------

def _write_cached_tif(arr: np.ndarray, bbox: tuple, path: Path):
    """Save a float32 canopy array as a single-band GeoTIFF."""
    import rasterio
    from rasterio.transform import from_bounds
    w, s, e, n = bbox
    nrows, ncols = arr.shape
    transform = from_bounds(w, s, e, n, ncols, nrows)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=nrows, width=ncols,
        count=1, dtype="float32", crs="EPSG:4326",
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(arr[np.newaxis, :, :])


def _read_cached_tif(path: Path, bbox: tuple) -> tuple[np.ndarray, float]:
    """Read a cached GeoTIFF and return (array, resolution_m)."""
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    w, s, e, n = bbox
    res = (n - s) * 111_320 / arr.shape[0] if arr.shape[0] > 0 else float("nan")
    return arr, res


# ---------------------------------------------------------------------------
# ALSGEDI 27m S3 fallback
# ---------------------------------------------------------------------------

def _stream_27m_tile(lat: float, lon: float, bbox_4326: tuple) -> tuple[np.ndarray, float]:
    """Stream a window from the 10° WGS84 Meta ALSGEDI tile (~27m/px, uint16 = cm)."""
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


# ---------------------------------------------------------------------------
# Simulation fallback
# ---------------------------------------------------------------------------

def _simulate_canopy(lat: float, lon: float, radius_m: float) -> tuple[np.ndarray, float]:
    """Reproducible seeded simulation — clearly flagged as not real data."""
    rng  = np.random.default_rng(seed=abs(hash(f"{int(lat)},{int(lon)}")))
    px   = max(50, int(2 * radius_m))
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bbox_from_point(lat: float, lon: float, radius_m: float) -> tuple:
    lat_deg = 1 / 111_320
    lon_deg = 1 / (111_320 * np.cos(np.radians(lat)))
    d_lat, d_lon = radius_m * lat_deg, radius_m * lon_deg
    return (lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat)


def _result(arr, res, bbox, source, simulated, elapsed):
    return {
        "array":      arr,
        "resolution": res,
        "bbox":       bbox,
        "crs":        "EPSG:4326",
        "source":     source,
        "simulated":  simulated,
        "elapsed_s":  elapsed,
    }
