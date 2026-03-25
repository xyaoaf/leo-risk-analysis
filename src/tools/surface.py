"""
build_obstruction_surface: combine terrain + canopy + buildings into a
single obstruction surface on the near-field analysis grid.

Multi-scale design
------------------
  Far-field  (terrain_far)  : 1500m radius, ~10m/px  — large-scale horizon
  Near-field (terrain_near) : 100m radius,   ~1m/px  — local baseline
  Canopy                    : 100m radius,  ~27m/px  — resampled to near-field grid
  Buildings                 : 100m radius,  vector   — rasterized to near-field grid

Masking rule
------------
  If canopy_resampled > CANOPY_MIN_HEIGHT at any pixel inside a building footprint:
    → building is marked "blocked" (vegetation covers roof, dish mounting impeded)
  Otherwise → building is "usable".

Output surface (absolute elevation, meters ASL)
-----------------------------------------------
  surface = terrain_near
          + canopy_resampled          (trees add to ground elevation)
          + usable_building_raster    (only unblocked buildings contribute extra height)
"""

import logging
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)

CANOPY_MIN_HEIGHT = 0.5   # meters — canopy below this is treated as bare ground


def build_obstruction_surface(
    terrain_near: dict,
    canopy: dict,
    buildings: dict,
) -> dict:
    """
    Build the combined obstruction surface on the near-field grid.

    Parameters
    ----------
    terrain_near : output of fetch_terrain(..., radius=100, resolution_hint=1)
    canopy       : output of fetch_canopy(..., radius=100)
    buildings    : output of fetch_buildings(..., radius=100)

    Returns
    -------
    dict:
        surface           : 2D ndarray, absolute elevation of top surface (m)
        terrain           : 2D ndarray, bare-ground DEM (m)
        canopy_resampled  : 2D ndarray, canopy height on near-field grid (m)
        building_all      : 2D ndarray, all building heights rasterized (m, 0=no building)
        building_usable   : 2D ndarray, only unblocked buildings (m)
        building_blocked  : 2D ndarray, only canopy-blocked buildings (m)
        gdf_classified    : GeoDataFrame with 'blocked' bool column per building
        bbox              : (W, S, E, N)
        transform         : rasterio Affine transform for near-field grid
        resolution        : pixel size in meters (near-field)
    """
    dem   = terrain_near["array"]
    bbox  = terrain_near["bbox"]
    w, s, e, n = bbox
    nrows, ncols = dem.shape

    transform = from_bounds(w, s, e, n, ncols, nrows)
    res_m = terrain_near["resolution"]

    logger.info(
        f"[surface] Building obstruction surface | "
        f"grid={nrows}×{ncols} res≈{res_m:.1f}m bbox={bbox}"
    )

    # ── 1. Resample canopy to near-field grid ───────────────────────────────
    canopy_r = _resample_to_grid(
        canopy["array"], canopy["bbox"], dem.shape, bbox
    )
    canopy_r = np.clip(canopy_r, 0, None)
    logger.info(f"[surface] Canopy resampled | max={canopy_r.max():.1f}m")

    # ── 2. Rasterize all buildings ──────────────────────────────────────────
    gdf = buildings["gdf"]
    building_all = _rasterize_buildings(gdf, "height_m", dem.shape, transform)
    logger.info(
        f"[surface] Buildings rasterized | "
        f"coverage={100*(building_all>0).mean():.1f}%"
    )

    # ── 3. Classify buildings: usable vs. blocked ───────────────────────────
    gdf_c = _classify_buildings(gdf, canopy_r, dem.shape, transform)
    usable  = gdf_c[~gdf_c["blocked"]]
    blocked = gdf_c[ gdf_c["blocked"]]

    n_blocked = blocked["blocked"].sum() if len(gdf_c) > 0 else 0
    logger.info(
        f"[surface] Building classification | "
        f"total={len(gdf_c)} usable={len(usable)} blocked={int(n_blocked)}"
    )

    building_usable  = _rasterize_buildings(usable,  "height_m", dem.shape, transform)
    building_blocked = _rasterize_buildings(blocked, "height_m", dem.shape, transform)

    # ── 4. Compose obstruction surface ─────────────────────────────────────
    # Rule: surface = terrain + canopy everywhere
    #        + usable building height where it exceeds terrain+canopy
    terrain_safe = np.where(np.isnan(dem), np.nanmedian(dem), dem)
    surface = terrain_safe + canopy_r
    # Buildings that aren't blocked push the surface higher if taller than canopy
    surface = np.maximum(surface, terrain_safe + building_usable)

    above_terrain = surface - terrain_safe
    logger.info(
        f"[surface] Done | obstruction above ground: "
        f"mean={above_terrain.mean():.1f}m max={above_terrain.max():.1f}m"
    )

    return {
        "surface":          surface,
        "terrain":          terrain_safe,
        "canopy_resampled": canopy_r,
        "building_all":     building_all,
        "building_usable":  building_usable,
        "building_blocked": building_blocked,
        "gdf_classified":   gdf_c,
        "bbox":             bbox,
        "transform":        transform,
        "resolution":       res_m,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample_to_grid(
    arr: np.ndarray,
    src_bbox: tuple,
    target_shape: tuple,
    dst_bbox: tuple,
) -> np.ndarray:
    """
    Resample `arr` (on src_bbox) to target_shape covering dst_bbox.
    Uses rasterio for proper geographic resampling.
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    import tempfile, os

    src_w, src_s, src_e, src_n = src_bbox
    dst_w, dst_s, dst_e, dst_n = dst_bbox
    src_rows, src_cols = arr.shape
    dst_rows, dst_cols = target_shape

    src_transform = from_bounds(src_w, src_s, src_e, src_n, src_cols, src_rows)
    dst_transform = from_bounds(dst_w, dst_s, dst_e, dst_n, dst_cols, dst_rows)

    src_data = arr.astype(np.float32)
    dst_data = np.zeros(target_shape, dtype=np.float32)

    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.bilinear,
    )
    return dst_data


def _rasterize_buildings(
    gdf: gpd.GeoDataFrame,
    height_col: str,
    shape: tuple,
    transform,
) -> np.ndarray:
    """Burn building heights into a raster of given shape."""
    if gdf is None or len(gdf) == 0:
        return np.zeros(shape, dtype=np.float32)

    valid = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if len(valid) == 0:
        return np.zeros(shape, dtype=np.float32)

    # Sort ascending by height so tallest polygon is written last (wins on overlap)
    valid_sorted = valid.copy()
    valid_sorted["_h"] = valid_sorted[height_col].astype(float)
    valid_sorted = valid_sorted.sort_values("_h", ascending=True)

    shapes = [
        (geom, float(h))
        for geom, h in zip(valid_sorted.geometry, valid_sorted[height_col])
        if h is not None and not np.isnan(float(h))
    ]
    if not shapes:
        return np.zeros(shape, dtype=np.float32)

    result = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0.0,
        dtype=np.float32,
    )
    return result


def _classify_buildings(
    gdf: gpd.GeoDataFrame,
    canopy_resampled: np.ndarray,
    shape: tuple,
    transform,
) -> gpd.GeoDataFrame:
    """
    Add a 'blocked' boolean column to gdf.
    A building is blocked if ANY canopy pixel within its footprint
    exceeds CANOPY_MIN_HEIGHT (vegetation covers the roof).
    """
    if gdf is None or len(gdf) == 0:
        result = gdf.copy() if gdf is not None else gpd.GeoDataFrame()
        result["blocked"] = []
        return result

    blocked_flags = []
    canopy_binary = (canopy_resampled > CANOPY_MIN_HEIGHT).astype(np.uint8)

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            blocked_flags.append(False)
            continue
        # Rasterize this single footprint as a mask
        footprint_mask = rasterize(
            [(geom, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        # Blocked if any canopy pixel falls within footprint
        has_canopy = bool(np.any((footprint_mask == 1) & (canopy_binary == 1)))
        blocked_flags.append(has_canopy)

    result = gdf.copy()
    result["blocked"] = blocked_flags
    return result
