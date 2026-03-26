"""
feasibility.py — Full Starlink installation feasibility analysis.

Entry point
-----------
    result = analyze_location(lat, lon)

Returns a structured dict conforming to the output schema in docs/DESIGN.md:
  feasible          bool
  on_building       bool
  dish_height_asl_m float
  constraints       dict of {pass, …} per constraint
  horizon           dict of profile arrays
  classification    dict (dominant, contributions, …)
  risk              dict (score, tier, components, explanation)
  best_nearby       dict or None
  warnings          list[str]
  data_sources      dict
  elapsed_s         float

Design rules (see docs/DESIGN.md §8)
---------------------------------------
  feasible = C_terrain_large  (≤10% FOV blocked by far terrain)
         AND C_terrain_small  (local slope < 20°)
         AND C_veg_at_point   (canopy height at dish < 1 m)
         AND C_vegetation     (≤15% FOV blocked by canopy)
         AND C_building_nearby(≤5% FOV blocked by buildings)
         AND C_roof_usable    (only if on_building)

All constraints are logical AND — a single failure → infeasible.
Risk score is continuous even for feasible locations.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure src/ is on path when called directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.terrain   import fetch_terrain
from tools.canopy    import fetch_canopy
from tools.buildings import fetch_buildings
from tools.surface   import build_obstruction_surface
from tools.horizon   import (
    compute_horizon_profile,
    compute_local_slope,
    classify_obstruction,
    evaluate_blockage,
    fov_mask,
    DISH_HEIGHT_M,
    SKY_THRESHOLD_DEG,
)
from scoring import score_risk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constraint thresholds
# ---------------------------------------------------------------------------
_C_TERRAIN_LARGE_TOL  = 0.10   # ≤10% of FOV azimuths blocked by terrain
_C_VEGETATION_TOL     = 0.15   # ≤15% (seasonal)
_C_BUILDING_TOL       = 0.05   # ≤5%  (permanent)
_C_SLOPE_MAX_DEG      = 20.0   # standard ground mount limit
_C_CANOPY_AT_DISH_HI  = 1.0    # m — high-res canopy (<5 m/px): individual tree detectable
_C_CANOPY_AT_DISH_LO  = 8.0    # m — coarse canopy (≥5 m/px): pixel covers ~30m×30m area;
                                #     only flag if clearly dense forest (>8 m)

# Default analysis parameters
_R_FAR   = 1500
_R_NEAR  = 100
_N_AZ    = 72
_LOCAL_SEARCH_RADIUS = 50     # metres

# Local search distance penalty: each metre of displacement adds this many
# risk points to the candidate objective, so nearby fixes are preferred.
_SEARCH_LAMBDA = 0.3   # risk pts / metre


# ===========================================================================
# Main entry point
# ===========================================================================

def analyze_location(
    lat: float,
    lon: float,
    radius_near: int   = _R_NEAR,
    radius_far:  int   = _R_FAR,
    dish_height: float = DISH_HEIGHT_M,
    n_az:        int   = _N_AZ,
    run_local_search: bool = True,
    local_search_radius: int = _LOCAL_SEARCH_RADIUS,
) -> dict:
    """
    Full Starlink installation feasibility analysis for a single location.

    Parameters
    ----------
    lat, lon             : WGS84 decimal degrees
    radius_near          : near-field analysis radius in metres (canopy/buildings)
    radius_far           : far-field terrain horizon radius in metres
    dish_height          : dish mount height above ground or rooftop (metres)
    n_az                 : number of azimuth samples (72 = every 5°)
    run_local_search     : if True and location is infeasible or risk > 20,
                           run find_better_nearby automatically
    local_search_radius  : search radius for find_better_nearby (metres)

    Returns
    -------
    dict — see module docstring for schema
    """
    t0       = time.time()
    warnings = []

    # ── 1. Data fetch ───────────────────────────────────────────────────────
    logger.info(f"[feasibility] analyze_location({lat:.5f}, {lon:.5f})")

    terrain_far  = fetch_terrain(lat, lon, radius_m=radius_far,  resolution_hint=10)
    terrain_near = fetch_terrain(lat, lon, radius_m=radius_near, resolution_hint=1)
    canopy       = fetch_canopy( lat, lon, radius_m=radius_near)
    buildings    = fetch_buildings(lat, lon, radius_m=radius_near)

    if canopy["simulated"]:
        warnings.append("Canopy data is SIMULATED — vegetation results are estimates.")

    if buildings["count"] == 0:
        warnings.append(
            "No buildings detected in the analysis area. "
            "If structures are present, blockage may be underestimated."
        )

    # Compute building height confidence summary from source_height column
    _bldg_height_conf = _building_height_confidence(buildings)

    # ── 2. Surface composition ──────────────────────────────────────────────
    surf = build_obstruction_surface(terrain_near, canopy, buildings)

    S_full    = surf["surface"]
    S_canopy  = surf["terrain"] + surf["canopy_resampled"]
    S_terrain = surf["terrain"]
    bbox_near = surf["bbox"]
    res_m     = surf["resolution"]
    nrows, ncols = S_full.shape
    cy, cx = nrows // 2, ncols // 2

    # ── 3. On-building check & dish elevation ───────────────────────────────
    on_building, bldg_row = _point_in_footprint(lat, lon, surf["gdf_classified"])
    roof_usable_flag      = True

    if on_building and bldg_row is not None:
        bldg_h = float(bldg_row["height_m"])
        if bool(bldg_row["blocked"]):
            # Change F: surface as degraded candidate — still mount on roof so
            # the horizon analysis captures the actual sky view from rooftop.
            # The canopy overhang will naturally raise the obstruction surface
            # and appear in the risk score; don't hard-fail here.
            roof_usable_flag = False    # flagged for reporting only
            h_dish = S_terrain[cy, cx] + bldg_h + dish_height
            warnings.append(
                "On-building location has vegetation overhang (roof degraded — "
                "canopy constraints may fail)."
            )
        else:
            h_dish = S_terrain[cy, cx] + bldg_h + dish_height
    else:
        h_dish = S_terrain[cy, cx] + dish_height

    # Check for NaN at dish location
    if np.isnan(h_dish):
        med = float(np.nanmedian(S_terrain))
        h_dish = med + dish_height
        warnings.append("DEM NaN at dish pixel — using median elevation.")

    # ── 4. Slope check ──────────────────────────────────────────────────────
    slope_deg = compute_local_slope(S_terrain, cy, cx, res_m)

    # ── 4b. Resolution-aware canopy-at-dish threshold ──────────────────────
    # At 27 m/px one pixel covers ~800 m²; a >1 m canopy value doesn't mean the
    # specific dish point has a tree. Use the coarse threshold unless resolution
    # is good enough (<5 m/px) to identify individual trees.
    canopy_res_m = canopy.get("resolution", 30.0)
    canopy_at_dish_threshold = (
        _C_CANOPY_AT_DISH_HI if canopy_res_m < 5.0 else _C_CANOPY_AT_DISH_LO
    )

    # ── 5. Horizon profiles ─────────────────────────────────────────────────
    hz_far     = compute_horizon_profile(
        terrain_far["array"], terrain_far["bbox"],
        n_az, h_dish_override=h_dish
    )
    hz_terrain = compute_horizon_profile(
        S_terrain, bbox_near, n_az, h_dish_override=h_dish
    )
    hz_canopy  = compute_horizon_profile(
        S_canopy, bbox_near, n_az, h_dish_override=h_dish
    )
    hz_full    = compute_horizon_profile(
        S_full, bbox_near, n_az, h_dish_override=h_dish
    )

    # ── 6. Blockage evaluation ──────────────────────────────────────────────
    fov = fov_mask(n_az)
    blk_far  = evaluate_blockage(hz_far,    fov)
    blk_can  = evaluate_blockage(hz_canopy, fov)
    blk_full = evaluate_blockage(hz_full,   fov)

    # Canopy contribution relative to terrain baseline
    canopy_contrib_frac = max(0.0, blk_can["blocked_frac"] - evaluate_blockage(hz_terrain, fov)["blocked_frac"])

    # ── 7. Constraint logic ─────────────────────────────────────────────────
    C = _evaluate_constraints(
        blk_far, blk_can, blk_full,
        canopy_contrib_frac,
        slope_deg,
        S_canopy[cy, cx] - S_terrain[cy, cx],   # canopy height at dish
        canopy_at_dish_threshold,
        on_building, roof_usable_flag,
        canopy["simulated"],
    )
    feasible     = all(C[k]["pass"] for k in C)
    failure_mode = _classify_failure_mode(C)

    # ── 8. Classification & risk score ─────────────────────────────────────
    # Use hz_terrain (near-field bare, 100 m) as the terrain baseline so that
    # canopy contribution is compared at the same spatial scale as hz_canopy.
    # Using hz_far (1500 m) here was a bug: a far-field ridge raises bf_t above
    # bf_c, making canopy_contrib negative (clamped to 0), hiding real vegetation.
    classification = classify_obstruction(hz_terrain, hz_canopy, hz_full, n_az)

    # Estimate vegetation seasonality from elevation + latitude.
    # Higher elevation / higher latitude → more deciduous → lower evergreen fraction.
    # Used by scoring to adjust the permanence penalty for vegetation-dominant sites.
    elev_m = float(np.nanmedian(S_terrain))
    evergreen_frac, canopy_type = _estimate_canopy_type(lat, elev_m)

    risk = score_risk(classification, evergreen_fraction=evergreen_frac)

    # ── 9. Optional local search ────────────────────────────────────────────
    # Change B: skip local search when far-field terrain is the dominant cause —
    # moving 50 m within the same terrain bowl cannot escape a regional blocker.
    best_nearby = None
    if (run_local_search
            and failure_mode != "regional_terrain"
            and (not feasible or risk["risk_score"] > 20)):
        try:
            best_nearby = find_better_nearby(
                lat, lon,
                tiles=(S_full, S_canopy, S_terrain, bbox_near,
                       terrain_far["array"], terrain_far["bbox"]),
                dish_height=dish_height,
                n_az=n_az,
                radius_m=local_search_radius,
            )
        except Exception as exc:
            logger.warning(f"[feasibility] Local search failed: {exc}")
            warnings.append(f"Local search failed: {exc}")

    elapsed = time.time() - t0
    logger.info(
        f"[feasibility] done | feasible={feasible}  "
        f"risk={risk['risk_score']:.1f} [{risk['risk_tier']}]  "
        f"dominant={classification['dominant']}  elapsed={elapsed:.1f}s"
    )

    return {
        "lat":             lat,
        "lon":             lon,
        "feasible":        feasible,
        "failure_mode":    failure_mode,
        "on_building":     on_building,
        "mount_type":      "rooftop" if on_building else "ground",
        "canopy_type":     canopy_type,
        "evergreen_frac":  round(evergreen_frac, 2),
        "building_height_confidence": _bldg_height_conf,
        "dish_height_asl_m": round(h_dish, 1),
        "constraints":     C,
        "horizon": {
            "n_azimuths":    n_az,
            "fov_center_deg": 0.0,
            "fov_half_deg":  50.0,
            "terrain_far":   hz_far.tolist(),
            "canopy":        hz_canopy.tolist(),
            "full":          hz_full.tolist(),
        },
        "classification":  {k: v for k, v in classification.items()
                            if not isinstance(v, np.ndarray)},
        "risk":            risk,
        "slope_deg":       round(slope_deg, 1),
        "best_nearby":     best_nearby,
        "warnings":        warnings,
        # ── Raw arrays for visualisation (underscore prefix → skipped by JSON serialiser)
        "_terrain_far_array": terrain_far["array"],
        "_terrain_far_bbox":  terrain_far["bbox"],
        "_surf_terrain":      S_terrain,
        "_surf_canopy":       surf["canopy_resampled"],
        "_surf_bbox":         bbox_near,
        "_gdf_classified":    surf["gdf_classified"],
        "data_sources": {
            "terrain_far":  terrain_far["source"],
            "terrain_near": terrain_near["source"],
            "canopy":       canopy["source"],
            "buildings":    buildings["source"],
        },
        "building_count":  buildings["count"],
        "canopy_max_m":    round(float(canopy["array"].max()), 1),
        "canopy_simulated": canopy["simulated"],
        "elapsed_s":       round(elapsed, 1),
    }


# ===========================================================================
# Local search: find_better_nearby
# ===========================================================================

def find_better_nearby(
    lat: float,
    lon: float,
    tiles: tuple,
    dish_height: float = DISH_HEIGHT_M,
    n_az: int = _N_AZ,
    radius_m: float = _LOCAL_SEARCH_RADIUS,
    strategy: str = "two_scale",
) -> dict:
    """
    Search for a better installation point within radius_m of (lat, lon).

    KEY OPTIMISATION: All candidates within 50 m of origin fall within the
    already-fetched 100 m near-field bbox.  This function reuses the tile
    data — no additional network requests.

    Parameters
    ----------
    lat, lon   : origin point
    tiles      : (S_full, S_canopy, S_terrain, bbox_near,
                  terrain_far_array, terrain_far_bbox)
    dish_height: default mount height
    n_az       : azimuth count
    radius_m   : search radius (should be ≤ near-field radius to reuse tiles)
    strategy   : "two_scale" (dense 10–15 m + coarse 30–50 m, default)
                 "ring"      (26 pts at 0.3×/0.6×/1.0× radius)
                 "grid"      (dense ~78 pts at 10 m step)

    Returns
    -------
    dict:
        best            : lowest-risk candidate result dict
        improvement     : delta risk, feasibility gain flag, distance
        origin_score    : risk score at the origin
        n_evaluated     : candidates successfully evaluated
        candidates      : list of all candidate summaries
    """
    S_full, S_canopy, S_terrain, bbox_near, far_arr, far_bbox = tiles
    nrows, ncols = S_full.shape
    cy0, cx0 = nrows // 2, ncols // 2

    # Origin risk (cheap — tile already in memory)
    origin_result = _evaluate_candidate(
        lat, lon, lat, lon,
        S_full, S_canopy, S_terrain, bbox_near,
        far_arr, far_bbox,
        cy0, cx0, nrows, ncols, dish_height, n_az,
    )

    # Sample candidates (Change D: two_scale is now the default)
    candidates_latlon = _sample_candidates(lat, lon, radius_m, strategy)

    w, s, e, n_b = bbox_near
    dy_m_px = (n_b - s) / nrows
    dx_m_px = (e   - w) / ncols

    results = []
    for c_lat, c_lon in candidates_latlon:
        # Convert lat/lon to pixel within pre-fetched bbox
        cy_c = int(round((c_lat - s) / dy_m_px))
        cx_c = int(round((c_lon - w) / dx_m_px))

        # Skip if outside fetched area
        if not (2 <= cy_c < nrows - 2 and 2 <= cx_c < ncols - 2):
            continue

        r = _evaluate_candidate(
            c_lat, c_lon, lat, lon,
            S_full, S_canopy, S_terrain, bbox_near,
            far_arr, far_bbox,
            cy_c, cx_c, nrows, ncols, dish_height, n_az,
        )
        # Attach distance for penalty ranking
        r["distance_m"] = round(_haversine_m(lat, lon, c_lat, c_lon), 1)
        results.append(r)

    if not results:
        return {
            "best": None,
            "improvement": None,
            "origin_score": origin_result["risk_score"],
            "n_evaluated": 0,
            "candidates": [],
        }

    # Change C: rank by objective = risk_score + λ × distance_m
    # Feasible candidates still come before infeasible ones.
    results.sort(key=lambda r: (
        not r["feasible"],
        r["risk_score"] + _SEARCH_LAMBDA * r["distance_m"],
    ))
    best = results[0]

    dist_m = best.get("distance_m", _haversine_m(lat, lon, best["lat"], best["lon"]))
    improvement = {
        "risk_delta":      round(origin_result["risk_score"] - best["risk_score"], 1),
        "feasible_gained": (not origin_result["feasible"]) and best["feasible"],
        "distance_m":      round(dist_m, 1),
        "dominant_change": origin_result["dominant"] != best["dominant"],
        "explanation":     _explain_improvement(origin_result, best),
    }

    logger.info(
        f"[feasibility] local_search | "
        f"n_candidates={len(results)}  best_risk={best['risk_score']:.1f}  "
        f"origin_risk={origin_result['risk_score']:.1f}  "
        f"Δ={improvement['risk_delta']:.1f}  dist={dist_m:.0f}m"
    )

    return {
        "best":         best,
        "improvement":  improvement,
        "origin_score": origin_result["risk_score"],
        "n_evaluated":  len(results),
        "candidates":   results,
    }


# ===========================================================================
# Internal helpers
# ===========================================================================

def _evaluate_candidate(
    c_lat, c_lon, origin_lat, origin_lon,
    S_full, S_canopy, S_terrain, bbox_near,
    far_arr, far_bbox,
    cy_c, cx_c, nrows, ncols,
    dish_height, n_az,
) -> dict:
    """Evaluate a single candidate point using pre-fetched tiles."""
    h_dish = float(S_terrain[cy_c, cx_c]) + dish_height
    if np.isnan(h_dish):
        h_dish = float(np.nanmedian(S_terrain)) + dish_height

    fov = fov_mask(n_az)

    # Far-field terrain — use same array but shifted dish elevation
    # (origin and candidate are close enough that far_arr centre is valid)
    hz_far  = compute_horizon_profile(
        far_arr, far_bbox, n_az,
        h_dish_override=h_dish
    )
    hz_can  = compute_horizon_profile(
        S_canopy, bbox_near, n_az,
        center_yx=(cy_c, cx_c), h_dish_override=h_dish
    )
    hz_full = compute_horizon_profile(
        S_full, bbox_near, n_az,
        center_yx=(cy_c, cx_c), h_dish_override=h_dish
    )

    blk_far  = evaluate_blockage(hz_far,  fov)
    blk_full = evaluate_blockage(hz_full, fov)
    blk_can  = evaluate_blockage(hz_can,  fov)

    slope_deg = compute_local_slope(
        S_terrain, cy_c, cx_c,
        (bbox_near[3] - bbox_near[1]) * 111_320 / nrows
    )

    canopy_at = float(S_canopy[cy_c, cx_c]) - float(S_terrain[cy_c, cx_c])

    # Hard constraints (simplified — no on-building check for candidates;
    # canopy at-point uses the coarse threshold since local search always uses
    # the same 27m canopy tile as the origin analysis)
    feasible = (
        blk_far["blocked_frac"]  <= _C_TERRAIN_LARGE_TOL and
        slope_deg                 <  _C_SLOPE_MAX_DEG and
        canopy_at                 <  _C_CANOPY_AT_DISH_LO and
        blk_can["blocked_frac"]  <= _C_VEGETATION_TOL and
        blk_full["blocked_frac"] <= _C_BUILDING_TOL
    )

    cl = classify_obstruction(hz_far, hz_can, hz_full, n_az)
    elev_m = float(np.nanmedian(S_terrain))
    ev_frac, _ = _estimate_canopy_type(c_lat, elev_m)
    rk = score_risk(cl, evergreen_fraction=ev_frac)

    return {
        "lat":        c_lat,
        "lon":        c_lon,
        "feasible":   feasible,
        "risk_score": rk["risk_score"],
        "risk_tier":  rk["risk_tier"],
        "dominant":   cl["dominant"],
        "max_angle_deg": blk_full["max_angle_deg"],
        "blocked_frac":  blk_full["blocked_frac"],
        "slope_deg":  round(slope_deg, 1),
        "canopy_at_m": round(canopy_at, 1),
    }


def _evaluate_constraints(
    blk_far: dict,
    blk_canopy: dict,
    blk_full: dict,
    canopy_contrib_frac: float,
    slope_deg: float,
    canopy_at_dish_m: float,
    canopy_at_dish_threshold: float,
    on_building: bool,
    roof_usable: bool,
    canopy_simulated: bool,
) -> dict:
    """
    Evaluate all hard constraints.  Returns dict[name → {pass, value, threshold, …}].
    """
    confidence = "low" if canopy_simulated else "high"

    C_terrain_large = {
        "pass":       blk_far["blocked_frac"] <= _C_TERRAIN_LARGE_TOL,
        "blocked_frac": round(blk_far["blocked_frac"], 3),
        "max_angle_deg": round(blk_far["max_angle_deg"], 1),
        "threshold":  _C_TERRAIN_LARGE_TOL,
        "description": "Far-field terrain horizon ≤ 10% of FOV blocked",
    }
    C_terrain_small = {
        "pass":       slope_deg < _C_SLOPE_MAX_DEG,
        "slope_deg":  round(slope_deg, 1),
        "threshold":  _C_SLOPE_MAX_DEG,
        "description": "Local slope < 20° (standard ground mount feasible)",
    }
    C_veg_at_point = {
        "pass":       canopy_at_dish_m < canopy_at_dish_threshold,
        "canopy_height_m": round(canopy_at_dish_m, 1),
        "threshold":  canopy_at_dish_threshold,
        "confidence": confidence,
        "description": (
            f"Canopy height at dish < {canopy_at_dish_threshold:.0f} m "
            f"({'1m hi-res rule' if canopy_at_dish_threshold < 5 else '8m coarse-data rule'})"
        ),
    }
    C_vegetation = {
        "pass":       blk_canopy["blocked_frac"] <= _C_VEGETATION_TOL,
        "blocked_frac": round(blk_canopy["blocked_frac"], 3),
        "max_angle_deg": round(blk_canopy["max_angle_deg"], 1),
        "threshold":  _C_VEGETATION_TOL,
        "confidence": confidence,
        "description": "Vegetation blocks ≤ 15% of FOV",
    }
    C_building_nearby = {
        "pass":       blk_full["blocked_frac"] <= _C_BUILDING_TOL,
        "blocked_frac": round(blk_full["blocked_frac"], 3),
        "max_angle_deg": round(blk_full["max_angle_deg"], 1),
        "threshold":  _C_BUILDING_TOL,
        "description": "Nearby buildings block ≤ 5% of FOV",
    }
    C_roof_usable = {
        "pass":       (not on_building) or roof_usable,
        "applicable": on_building,
        "roof_usable": roof_usable,
        "description": "Rooftop free of vegetation overhang (if on building)",
    }

    return {
        "C_terrain_large":   C_terrain_large,
        "C_terrain_small":   C_terrain_small,
        "C_veg_at_point":    C_veg_at_point,
        "C_vegetation":      C_vegetation,
        "C_building_nearby": C_building_nearby,
        "C_roof_usable":     C_roof_usable,
    }


def _classify_failure_mode(C: dict) -> str:
    """
    Classify the primary reason a location is infeasible (or 'feasible').

    Returns one of:
      feasible         — all constraints pass
      regional_terrain — C_terrain_large fails (far-field horizon blocks FOV)
      local_terrain    — C_terrain_small fails (slope too steep)
      local_canopy     — C_veg_at_point or C_vegetation fails
      local_building   — C_building_nearby fails
      roof_unusable    — C_roof_usable fails (on-building vegetation overhang)
      mixed            — two or more constraints fail
    """
    failed = [k for k, v in C.items() if not v["pass"]]
    if not failed:
        return "feasible"
    if len(failed) > 1:
        return "mixed"
    f = failed[0]
    if f == "C_terrain_large":
        return "regional_terrain"
    if f == "C_terrain_small":
        return "local_terrain"
    if f in ("C_veg_at_point", "C_vegetation"):
        return "local_canopy"
    if f == "C_building_nearby":
        return "local_building"
    if f == "C_roof_usable":
        return "roof_unusable"
    return "unknown"


def _point_in_footprint(
    lat: float,
    lon: float,
    gdf,
) -> tuple[bool, Any]:
    """
    Check if (lat, lon) lies within any building footprint.

    Returns
    -------
    (on_building: bool, row: pandas.Series or None)
    """
    if gdf is None or len(gdf) == 0:
        return False, None

    from shapely.geometry import Point
    pt = Point(lon, lat)
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is not None and not geom.is_empty and geom.contains(pt):
            return True, row
    return False, None


def _sample_candidates(
    lat: float, lon: float, radius_m: float, strategy: str
) -> list[tuple[float, float]]:
    """Sample candidate lat/lon pairs within radius_m."""
    lat_deg = 1.0 / 111_320.0
    lon_deg = 1.0 / (111_320.0 * np.cos(np.radians(lat)))

    candidates = []

    if strategy == "two_scale":
        # Change D: dense inner ring (tree-crown escape, ~10–15 m) +
        #           coarser outer ring (different patch, ~30–50 m).
        # Inner: 3 rings × 8 pts = 24 candidates
        for r_m, n in [(10, 8), (12, 8), (15, 8)]:
            for i in range(n):
                theta = 2 * np.pi * i / n
                candidates.append((
                    lat + r_m * np.cos(theta) * lat_deg,
                    lon + r_m * np.sin(theta) * lon_deg,
                ))
        # Outer: 3 rings × 12 pts = 36 candidates (capped to radius_m)
        outer_radii = [30, 40, min(50, radius_m)]
        for r_m in outer_radii:
            if r_m > radius_m:
                continue
            for i in range(12):
                theta = 2 * np.pi * i / 12
                candidates.append((
                    lat + r_m * np.cos(theta) * lat_deg,
                    lon + r_m * np.sin(theta) * lon_deg,
                ))

    elif strategy == "ring":
        rings   = [0.3, 0.6, 1.0]
        n_per   = [6,   8,   12]
        for ring_frac, n in zip(rings, n_per):
            r = radius_m * ring_frac
            for i in range(n):
                theta = 2 * np.pi * i / n
                candidates.append((
                    lat + r * np.cos(theta) * lat_deg,
                    lon + r * np.sin(theta) * lon_deg,
                ))

    elif strategy == "grid":
        step_m = 10.0
        steps  = np.arange(-radius_m, radius_m + step_m, step_m)
        for dy in steps:
            for dx in steps:
                if dx**2 + dy**2 <= radius_m**2:
                    candidates.append((
                        lat + dy * lat_deg,
                        lon + dx * lon_deg,
                    ))

    return candidates


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two WGS84 points."""
    R   = 6_371_000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a   = (np.sin(dlat / 2) ** 2
           + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
           * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def _building_height_confidence(buildings: dict) -> str:
    """
    Summarise the height data quality for the buildings in the analysis area.

    Returns
    -------
    str — one of:
        "ml_predicted"  : all (or ≥80%) heights from Microsoft ML predictions
        "mixed"         : mix of ML-predicted and area-heuristic estimates
        "area_estimate" : all (or ≥80%) heights estimated from footprint area
        "none"          : no buildings detected
    """
    gdf = buildings.get("gdf")
    if gdf is None or len(gdf) == 0:
        return "none"
    if "source_height" not in gdf.columns:
        return "unknown"

    counts = gdf["source_height"].value_counts()
    total = len(gdf)
    n_ml = int(counts.get("ms_predicted", 0))
    pct_ml = n_ml / total

    if pct_ml >= 0.80:
        return "ml_predicted"
    if pct_ml <= 0.20:
        return "area_estimate"
    return "mixed"


def _estimate_canopy_type(lat: float, elev_m: float) -> tuple[float, str]:
    """
    Estimate vegetation evergreen fraction and canopy type from latitude + elevation.

    This is a simple heuristic based on broad US forest composition patterns:
      - Low elevation coastal plain (Southeast US): longleaf/loblolly pine → high evergreen
      - Mid-elevation piedmont: mixed pine/hardwood → moderate evergreen
      - High elevation / high latitude: deciduous hardwood (oak, maple) → low evergreen

    Returns
    -------
    (evergreen_fraction, canopy_type)
      evergreen_fraction : float 0–1 (0 = fully deciduous, 1 = fully coniferous)
      canopy_type        : "conifer" | "mixed" | "deciduous" | "unknown"
    """
    if elev_m < 0 or np.isnan(elev_m):
        return 0.5, "unknown"

    # Base fraction from elevation (higher → more deciduous)
    if elev_m < 150:
        base = 0.75   # coastal plain — pine flatwoods (SE US)
    elif elev_m < 500:
        base = 0.55   # piedmont — mixed pine-hardwood
    elif elev_m < 1000:
        base = 0.40   # montane — mostly deciduous hardwood
    else:
        base = 0.45   # subalpine — spruce/fir mix returns

    # Latitude adjustment: higher latitude → more deciduous
    # CONUS range ~25°–50°N; at 50°N fully deciduous
    lat_factor = max(0.0, min(1.0, (lat - 25.0) / 25.0))   # 0 at 25°N, 1 at 50°N
    ev_frac = max(0.1, base - 0.20 * lat_factor)

    if ev_frac >= 0.65:
        canopy_type = "conifer"
    elif ev_frac >= 0.38:
        canopy_type = "mixed"
    else:
        canopy_type = "deciduous"

    return round(ev_frac, 2), canopy_type


def _explain_improvement(origin: dict, best: dict) -> str:
    dr = origin["risk_score"] - best["risk_score"]
    if dr <= 0:
        return "No improvement found nearby."
    parts = [f"Moving {_haversine_m(origin['lat'], origin['lon'], best['lat'], best['lon']):.0f} m "
             f"reduces risk by {dr:.0f} points ({origin['risk_score']:.0f} → {best['risk_score']:.0f})."]
    if best["dominant"] != origin["dominant"]:
        parts.append(
            f"Dominant obstruction changes from {origin['dominant']} to {best['dominant']}."
        )
    if (not origin["feasible"]) and best["feasible"]:
        parts.append("Location becomes feasible.")
    return " ".join(parts)
