"""
fetch_terrain: fetch USGS 3DEP DEM for a given point + radius.
Returns a 2D numpy array of elevation (meters) with metadata.
"""

import time
import logging
import numpy as np
from pyproj import Transformer

logger = logging.getLogger(__name__)


def _bbox_from_point(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Convert (lat, lon) + radius in meters to a (west, south, east, north) bbox in WGS84."""
    # Approximate degrees per meter at given latitude
    lat_deg_per_m = 1 / 111_320
    lon_deg_per_m = 1 / (111_320 * np.cos(np.radians(lat)))
    d_lat = radius_m * lat_deg_per_m
    d_lon = radius_m * lon_deg_per_m
    return (lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat)  # (W, S, E, N)


def fetch_terrain(lat: float, lon: float, radius_m: float = 500, resolution_hint: int = 1) -> dict:
    """
    Fetch DEM elevation data from USGS 3DEP for a bounding box around (lat, lon).

    Parameters
    ----------
    lat, lon          : center point in WGS84
    radius_m          : search radius in meters
    resolution_hint   : preferred resolution in meters (1, 3, 10, or 30).
                        The fetcher tries this value first, then falls back to coarser.
                        Use 10 for large radii (>500m) to avoid huge arrays.

    Returns
    -------
    dict with keys:
        array       : 2D np.ndarray of elevation in meters
        resolution  : approximate pixel size in meters
        bbox        : (west, south, east, north) in WGS84
        crs         : coordinate reference system string
        source      : data source description
        elapsed_s   : fetch time in seconds
        nodata      : nodata value (if any)
    """
    import py3dep

    _LADDER = (1, 3, 10, 30)
    t0 = time.time()
    bbox = _bbox_from_point(lat, lon, radius_m)
    logger.info(
        f"[terrain] Fetching 3DEP DEM | center=({lat},{lon}) "
        f"radius={radius_m}m hint={resolution_hint}m"
    )

    # Start from resolution_hint; fall back to coarser if unavailable.
    # Each resolution is attempted up to MAX_ATTEMPTS times with exponential
    # backoff to handle transient 3DEP WMS rate-limiting (HTTP 429/503).
    # In concurrent batch runs the 3DEP endpoint rejects bursts; backoff gives
    # the server time to recover before retrying or trying a coarser resolution.
    MAX_ATTEMPTS = 4
    _BACKOFF_BASE = 5  # seconds; doubles on each retry: 5, 10, 20s
    start_idx = next((i for i, r in enumerate(_LADDER) if r >= resolution_hint), 0)
    for resolution in _LADDER[start_idx:]:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                dem_xr = py3dep.get_map(
                    "DEM",
                    geometry=bbox,
                    resolution=resolution,
                    geo_crs="EPSG:4326",
                    crs="EPSG:4326",
                )
                arr = dem_xr.values.squeeze().astype(np.float32)
                nodata = dem_xr.rio.nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan

                elapsed = time.time() - t0
                actual_res = _estimate_resolution(dem_xr, bbox)
                logger.info(
                    f"[terrain] OK | resolution={resolution}m actual≈{actual_res:.1f}m "
                    f"shape={arr.shape} min={np.nanmin(arr):.1f}m max={np.nanmax(arr):.1f}m "
                    f"elapsed={elapsed:.2f}s"
                )
                return {
                    "array": arr,
                    "resolution": actual_res,
                    "bbox": bbox,
                    "crs": "EPSG:4326",
                    "source": f"USGS 3DEP (py3dep), requested {resolution}m",
                    "elapsed_s": elapsed,
                    "nodata": nodata,
                }
            except Exception as e:
                sleep_s = _BACKOFF_BASE * (2 ** (attempt - 1))  # 5, 10, 20, 40s
                logger.warning(
                    f"[terrain] resolution={resolution}m attempt {attempt}/{MAX_ATTEMPTS} "
                    f"failed: {type(e).__name__}: {e}. "
                    + (f"Retrying in {sleep_s}s…" if attempt < MAX_ATTEMPTS
                       else "Trying lower res…")
                )
                if attempt < MAX_ATTEMPTS:
                    time.sleep(sleep_s)

    raise RuntimeError(f"[terrain] All resolutions failed for bbox={bbox}")


def _estimate_resolution(xr_da, bbox) -> float:
    """Estimate pixel size in meters from xarray DataArray shape and bbox."""
    _, s, _, n = bbox
    height_m = (n - s) * 111_320
    nrows = xr_da.shape[-2] if xr_da.ndim >= 2 else 1
    return height_m / nrows if nrows > 0 else float("nan")
