"""
Horizon profile computation, blockage evaluation, and obstruction classification.

Physics
-------
A Starlink dish needs a clear sky cone:
  - Minimum elevation angle : 25° above horizon
  - FOV arc                 : ~100° centred northward for CONUS
    (azimuths 310°–50°, i.e. ±50° around north = 0°)

For each azimuth we cast a radial ray outward from the dish and record
the maximum elevation angle to any obstruction along that ray:

    elev_angle(d) = arctan2( (surface[d] − h_dish), d_m )

Implementation: vectorised NumPy — all N azimuths × K steps computed in one
broadcast, ~50× faster than the equivalent Python loop.

New in v2
---------
  compute_local_slope  — micro-topography slope at a pixel
  evaluate_blockage    — structured dict from a single horizon profile
  compute_horizon_profile now accepts center_yx and h_dish_override for
    off-centre analysis (used by find_better_nearby tile reuse).
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Starlink operating parameters
DISH_HEIGHT_M     = 3.0    # default mount height above ground (metres)
SKY_THRESHOLD_DEG = 25.0   # elevation angle below which sky is "clear"

# Starlink FOV: ~100° northward arc for CONUS
FOV_CENTER_DEG = 0.0    # azimuth of FOV centre (0 = true north)
FOV_HALF_DEG   = 50.0   # half-width of FOV arc


# ---------------------------------------------------------------------------
# Horizon profile
# ---------------------------------------------------------------------------

def compute_horizon_profile(
    surface: np.ndarray,
    bbox: tuple,
    n_azimuths: int = 72,
    dish_height_m: float = DISH_HEIGHT_M,
    center_yx: tuple[int, int] | None = None,
    h_dish_override: float | None = None,
) -> np.ndarray:
    """
    Compute max obstruction elevation angle for each azimuth (vectorised).

    Parameters
    ----------
    surface          : 2D float32 array, absolute elevation in metres (terrain +
                       canopy + buildings, or any subset)
    bbox             : (W, S, E, N) in WGS84
    n_azimuths       : azimuth samples; 72 = every 5°
    dish_height_m    : dish height above the surface at the centre pixel (metres)
    center_yx        : (row, col) of dish location; defaults to array centre.
                       Use non-default for off-centre local-search candidates.
    h_dish_override  : absolute dish elevation (metres ASL) override.
                       If None, computed as surface[cy,cx] + dish_height_m.

    Returns
    -------
    angles : shape (n_azimuths,), degrees; index 0 = north, clockwise.
    """
    nrows, ncols = surface.shape
    w, s, e, n   = bbox
    lat_c        = (s + n) / 2.0

    dy_m = (n - s) * 111_320.0 / nrows
    dx_m = (e - w) * 111_320.0 * np.cos(np.radians(lat_c)) / ncols

    cy, cx = center_yx if center_yx is not None else (nrows // 2, ncols // 2)
    cy, cx = int(cy), int(cx)
    cy = np.clip(cy, 0, nrows - 1)
    cx = np.clip(cx, 0, ncols - 1)

    if h_dish_override is not None:
        h_dish = float(h_dish_override)
    else:
        h_dish = float(surface[cy, cx]) + dish_height_m

    max_steps = int(min(nrows, ncols) * 0.48)
    if max_steps < 2:
        return np.full(n_azimuths, -90.0)

    # ── Vectorised ray-cast ─────────────────────────────────────────────────
    az_deg = np.linspace(0.0, 360.0, n_azimuths, endpoint=False)  # [N]
    drow   = -np.cos(np.radians(az_deg))    # [N]  north → row decreases
    dcol   =  np.sin(np.radians(az_deg))    # [N]  east  → col increases

    steps  = np.arange(1, max_steps, dtype=np.float32)             # [K]

    # Sample pixel coordinates [N, K]
    rows = (cy + np.outer(drow, steps)).round().astype(np.int32)
    cols = (cx + np.outer(dcol, steps)).round().astype(np.int32)

    # Validity mask (inside grid)
    valid = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)

    # Clamp for safe indexing; out-of-bounds slots will be masked anyway
    rows_c = np.clip(rows, 0, nrows - 1)
    cols_c = np.clip(cols, 0, ncols - 1)

    # Distance from dish to each step [N, K]
    dist_m = steps[np.newaxis, :] * np.sqrt(
        (drow[:, np.newaxis] * dy_m) ** 2 +
        (dcol[:, np.newaxis] * dx_m) ** 2
    )

    # Elevation difference [N, K]
    elev_diff = surface[rows_c, cols_c].astype(np.float32) - h_dish

    # Elevation angles [N, K]
    angles = np.degrees(np.arctan2(elev_diff, dist_m))
    angles[~valid] = -90.0   # mask out-of-bounds

    return angles.max(axis=1)   # [N]


# ---------------------------------------------------------------------------
# Blockage evaluation
# ---------------------------------------------------------------------------

def evaluate_blockage(
    horizon: np.ndarray,
    fov_mask_arr: np.ndarray,
    threshold_deg: float = SKY_THRESHOLD_DEG,
) -> dict:
    """
    Evaluate how many FOV azimuths are blocked above threshold.

    Parameters
    ----------
    horizon       : shape (N,), elevation angles in degrees
    fov_mask_arr  : shape (N,), bool — True for FOV azimuths
    threshold_deg : blocking threshold in degrees

    Returns
    -------
    dict:
        blocked_frac        : fraction of FOV azimuths blocked (0–1)
        max_angle_deg       : max elevation angle across FOV azimuths
        max_angle_all_deg   : max elevation angle across ALL azimuths
        blocked_azimuths    : indices of blocked FOV azimuths
        n_blocked           : count of blocked FOV azimuths
        n_fov               : total FOV azimuths
    """
    fov_angles  = horizon[fov_mask_arr]
    blocked_idx = np.where(fov_mask_arr & (horizon > threshold_deg))[0]

    n_fov     = int(fov_mask_arr.sum())
    n_blocked = len(blocked_idx)

    return {
        "blocked_frac":      n_blocked / n_fov if n_fov > 0 else 0.0,
        "max_angle_deg":     float(fov_angles.max()) if len(fov_angles) else -90.0,
        "max_angle_all_deg": float(horizon.max()),
        "blocked_azimuths":  blocked_idx.tolist(),
        "n_blocked":         n_blocked,
        "n_fov":             n_fov,
    }


# ---------------------------------------------------------------------------
# Local slope (micro-topography)
# ---------------------------------------------------------------------------

def compute_local_slope(
    dem: np.ndarray,
    cy: int,
    cx: int,
    res_m: float,
) -> float:
    """
    Compute terrain slope at pixel (cy, cx) using a 3×3 Sobel-like finite difference.

    Parameters
    ----------
    dem   : 2D elevation array (metres)
    cy/cx : centre pixel row/column
    res_m : pixel size in metres

    Returns
    -------
    slope_deg : slope angle in degrees (0 = flat, 90 = vertical)
    """
    nrows, ncols = dem.shape
    r0 = max(0, cy - 1);  r1 = min(nrows - 1, cy + 1)
    c0 = max(0, cx - 1);  c1 = min(ncols - 1, cx + 1)

    # Fill NaN with local median for gradient stability
    patch = dem[r0:r1+1, c0:c1+1].copy()
    if np.any(np.isnan(patch)):
        patch = np.where(np.isnan(patch), np.nanmedian(patch), patch)

    # Central differences (fall back to one-sided at boundaries)
    dz_dx = (float(dem[cy, c1]) - float(dem[cy, c0])) / ((c1 - c0) * res_m + 1e-9)
    dz_dy = (float(dem[r0, cx]) - float(dem[r1, cx])) / ((r1 - r0) * res_m + 1e-9)

    return float(np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))))


# ---------------------------------------------------------------------------
# FOV mask
# ---------------------------------------------------------------------------

def fov_mask(n_azimuths: int) -> np.ndarray:
    """Boolean mask of which azimuths fall within Starlink's northward FOV."""
    az_arr = np.arange(n_azimuths) * 360.0 / n_azimuths
    diff   = np.abs(((az_arr - FOV_CENTER_DEG + 180) % 360) - 180)
    return diff <= FOV_HALF_DEG


# ---------------------------------------------------------------------------
# Obstruction classification (unchanged API)
# ---------------------------------------------------------------------------

def classify_obstruction(
    horizon_terrain:   np.ndarray,
    horizon_canopy:    np.ndarray,
    horizon_buildings: np.ndarray,
    n_azimuths: int = 72,
) -> dict:
    """
    Classify dominant obstruction type from three horizon profiles.

    Returns
    -------
    dict:
        dominant              : "clear" | "terrain" | "vegetation" | "building"
        blocked_frac_terrain  : fraction of FOV blocked by terrain only
        blocked_frac_canopy   : fraction of FOV blocked by terrain+canopy
        blocked_frac_buildings: fraction of FOV blocked by all layers
        canopy_contribution   : incremental blocking fraction from canopy
        building_contribution : incremental blocking fraction from buildings
        max_angle_terrain     : peak elevation angle (terrain only), degrees
        max_angle_canopy      : peak elevation angle (terrain+canopy), degrees
        max_angle_buildings   : peak elevation angle (all layers), degrees
        horizon_terrain/canopy/buildings : full profile arrays
    """
    mask = fov_mask(n_azimuths)

    def bf(angles):
        return float((angles[mask] > SKY_THRESHOLD_DEG).mean())

    bf_t = bf(horizon_terrain)
    bf_c = bf(horizon_canopy)
    bf_b = bf(horizon_buildings)

    canopy_contrib   = max(0.0, bf_c - bf_t)
    building_contrib = max(0.0, bf_b - bf_c)

    if bf_t > 0.10:
        dominant = "terrain"
    elif canopy_contrib >= building_contrib and (canopy_contrib > 0.02 or bf_c > 0.05):
        dominant = "vegetation"
    elif building_contrib > 0.02 or bf_b > 0.05:
        dominant = "building"
    else:
        dominant = "clear"

    logger.info(
        f"[horizon] dominant={dominant}  "
        f"terrain={bf_t:.2f}  canopy={bf_c:.2f}  buildings={bf_b:.2f}  "
        f"(Δveg={canopy_contrib:.2f} Δbld={building_contrib:.2f})"
    )

    return {
        "dominant":               dominant,
        "blocked_frac_terrain":   bf_t,
        "blocked_frac_canopy":    bf_c,
        "blocked_frac_buildings": bf_b,
        "canopy_contribution":    canopy_contrib,
        "building_contribution":  building_contrib,
        "max_angle_terrain":      float(horizon_terrain.max()),
        "max_angle_canopy":       float(horizon_canopy.max()),
        "max_angle_buildings":    float(horizon_buildings.max()),
        "horizon_terrain":        horizon_terrain,
        "horizon_canopy":         horizon_canopy,
        "horizon_buildings":      horizon_buildings,
    }
