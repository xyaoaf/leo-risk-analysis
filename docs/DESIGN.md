# Technical Design: Starlink Installation Feasibility Analysis System

**Version 2.0 | Deterministic Geospatial Model**

---

## 1. System Overview

### 1.1 Problem Statement

Given a target point P = (lat, lon), determine whether a Starlink dish can achieve
reliable connectivity, quantify the degree of risk, and find the best alternative
within 50 m if the primary point is infeasible.

### 1.2 Core Physics

Starlink operates in LEO (~550 km altitude). From any ground point, active satellites
appear within a sky cone defined by:

- **Minimum elevation angle**: 25° above horizontal. Below this, the atmospheric path
  length and terrain horizon block the link.
- **FOV arc**: for CONUS, satellites concentrate in a northward arc of ±50° around
  azimuth 0° (true north), because Starlink's 53° orbital inclination means satellites
  spend more time at high latitudes passing over from north to south.
- **Obstruction model**: any object that subtends an elevation angle ≥ 25° within the
  FOV arc constitutes a blocking obstruction.

The elevation angle from the dish to the top of an obstruction at horizontal distance d:

    α = arctan( (h_obs_top − h_dish) / d )

Where `h_obs_top` is the absolute elevation of the obstruction top (terrain + canopy +
building), `h_dish` is the absolute elevation of the dish, and `d` is horizontal
distance in meters.

---

## 2. Coordinate and Projection Conventions

All internal computation uses metric approximations centred on the target point:

    dx_m(Δlon) = Δlon × 111_320 × cos(lat_center_rad)
    dy_m(Δlat) = Δlat × 111_320

Accurate to < 0.1% within a 5 km radius at mid-latitudes. WGS84 is used only for
data ingestion and output reporting.

---

## 3. Data Layer Specifications

| Layer            | Source                   | Resolution | Radius  | CRS        |
|------------------|--------------------------|------------|---------|------------|
| DEM far-field    | USGS 3DEP (py3dep)       | 10–30 m/px | 1500 m  | EPSG:4326  |
| DEM near-field   | USGS 3DEP (py3dep)       | 1–3 m/px   | 100 m   | EPSG:4326  |
| Canopy height    | Meta GCH (S3 COG)        | 1–27 m/px  | 100 m   | EPSG:4326  |
| Building footprt | Microsoft ML Footprints  | Vector     | 100 m   | EPSG:4326  |

All raster layers are composed onto the near-field DEM grid for surface analysis.
The far-field DEM is kept separate for the large-scale horizon profile.

---

## 4. Obstruction Surface Composition

A single obstruction surface S(row, col) in absolute meters ASL:

    S(r,c) = DEM_near(r,c)                    # bare-ground baseline
           + max(canopy(r,c), 0)               # trees add height above ground
           + building_usable(r,c)              # usable buildings override where taller

`building_usable` = rasterised height of buildings NOT blocked by vegetation.
The far-field horizon uses DEM-only (buildings/canopy negligible beyond ~300 m).

---

## 5. Component C₁: Large-Scale Terrain Horizon

### 5.1 Purpose

Detect ridgelines and hills within 100–1500 m that raise the apparent horizon
above 25° in the satellite FOV arc.

### 5.2 Horizon Profile Algorithm

For each azimuth θᵢ = i × (360/N) degrees:

    drow = −cos(θᵢ_rad)          # north → row decreases
    dcol = +sin(θᵢ_rad)          # east  → col increases

    for step k = 1, 2, ..., max_steps:
        r = round(cy + drow × k)
        c = round(cx + dcol × k)
        if (r, c) out of bounds: break

        dist_m    = k × √( (drow×dy_m)² + (dcol×dx_m)² )
        elev_diff = S[r,c] − h_dish
        α_k       = arctan2(elev_diff, dist_m)
        H(θᵢ)     = max(H(θᵢ), α_k)

**Vectorised implementation** (NumPy broadcasting, ~50× faster than Python loop):

    az_deg = linspace(0, 360, N, endpoint=False)          # [N]
    drow   = −cos(radians(az_deg))                        # [N]
    dcol   =  sin(radians(az_deg))                        # [N]
    steps  = arange(1, max_steps)                         # [K]

    rows   = round(cy + outer(drow, steps))               # [N, K]
    cols   = round(cx + outer(dcol, steps))               # [N, K]
    valid  = in_bounds(rows, cols)                        # [N, K]

    dist_m     = steps × √( (drow[:,None]×dy_m)² + (dcol[:,None]×dx_m)² )
    elev_diff  = S[rows_clamped, cols_clamped] − h_dish
    angles     = degrees(arctan2(elev_diff, dist_m))
    angles[~valid] = −90

    H = angles.max(axis=1)                                # [N]

### 5.3 Dish Elevation

    h_dish = DEM_near[cy, cx] + dish_mount_height_m      # ground mount
    h_dish = DEM_near[cy, cx] + building.height_m        # on-roof mount
           + dish_mount_height_m

### 5.4 Constraint Check

    FOV_mask[i]       = |θᵢ − FOV_center|_wrapped ≤ FOV_half
    blocked_terrain[i] = (H_terrain(θᵢ) > 25°) AND FOV_mask[i]

    C_terrain_large = (mean(blocked_terrain) ≤ 0.10)    # ≤ 10% of FOV

---

## 6. Component C₂: Near-Field Obstructions

### 6.1 Three Horizon Profiles

    H_terrain_near   ← ray-cast on DEM_near only
    H_canopy         ← ray-cast on DEM_near + canopy_resampled
    H_full           ← ray-cast on S_full (terrain + canopy + usable buildings)

### 6.2 Vegetation Constraint

**At-point** (dish cannot be placed inside a tree):

    canopy_at_dish = canopy_r[cy, cx]
    C_veg_at_point = (canopy_at_dish < 1.0 m)

**Near-field blockage**:

    canopy_contrib[i] = max(0, H_canopy[i] − H_terrain_near[i])
    blocked_canopy[i] = (H_canopy[i] > 25°) AND FOV_mask[i] AND (canopy_contrib[i] > 0)

    C_vegetation = C_veg_at_point AND (mean(blocked_canopy) ≤ 0.15)

### 6.3 Building Constraint

**On-building installation**:

    on_building = point_within_any_footprint(lat, lon, gdf_buildings)

    if on_building AND NOT building.blocked_by_veg:
        h_dish = DEM_near[cy,cx] + building.height_m + dish_height
        C_roof_usable = True
    elif on_building AND building.blocked_by_veg:
        C_roof_usable = False           # vegetation overhang on roof
    else:
        C_roof_usable = True            # ground mount — N/A

**Vegetation-on-roof masking**: A building is "blocked" if ANY canopy pixel within
its footprint exceeds 0.5 m. Blocked buildings are excluded from the usable layer.

**Nearby blockage**:

    building_contrib[i] = max(0, H_full[i] − H_canopy[i])
    blocked_bldg[i]     = (H_full[i] > 25°) AND FOV_mask[i] AND (building_contrib > 0)

    C_building_nearby = (mean(blocked_bldg) ≤ 0.05)    # ≤ 5% (permanent)

---

## 7. Component C₃: Small-Scale Terrain (Micro-topography)

### 7.1 Local Slope Calculation

3×3 Sobel kernel on near-field DEM:

    ∂z/∂x ≈ (T[r, c+1] − T[r, c−1]) / (2 × dx_m)
    ∂z/∂y ≈ (T[r−1, c] − T[r+1, c]) / (2 × dy_m)

    slope_deg = degrees( arctan( √( (∂z/∂x)² + (∂z/∂y)² ) ) )

### 7.2 Constraint Check

    C_terrain_small = (slope_at_dish < 20°)

Starlink hardware tolerates ±5°; shimmed mount extends this to ~15–20°.
Beyond 20°, a pole mount is required (increases h_dish).

---

## 8. Composite Feasibility Logic (Hard Constraints)

    feasible = C_terrain_large
           AND C_terrain_small
           AND C_veg_at_point
           AND C_vegetation
           AND C_building_nearby
           AND C_roof_usable         # only evaluated if on_building

This is a **logical AND** — any single failure makes the point infeasible.
The system does NOT average or soften failures.

### 8.1 Constraint Hierarchy (failure explanation priority)

1. `C_veg_at_point`     → "Dish location is inside tree canopy"
2. `C_terrain_large`    → "Ridge or hill blocks satellite arc"
3. `C_building_nearby`  → "Adjacent building blocks FOV"
4. `C_vegetation`       → "Tree canopy blocks FOV"
5. `C_terrain_small`    → "Ground slope too steep for standard mount"
6. `C_roof_usable`      → "Rooftop has vegetation overhang"

---

## 9. Risk Scoring System

The risk score (0–100) is continuous and provides graded severity even when
`feasible = True`. A score of 45 on a feasible location is marginal.

### 9.1 Component A: FOV Blockage (0–50 pts)

    bf = fraction of FOV azimuths where H_full[i] > 25°
    A  = bf × 50

### 9.2 Component B: Angle Severity (0–30 pts)

    max_ang_fov = max(H_full[i]) for i in FOV
    if max_ang_fov > 25°:
        severity = clamp((max_ang_fov − 25) / 65, 0, 1)
    else:
        severity = 0
    B = severity × 30

### 9.3 Component C: Permanence Penalty (0–20 pts)

    perm_factor = {building: 1.0, terrain: 1.0, vegetation: 0.6, clear: 0.0}
    C = perm_factor[dominant] × clamp(bf × 40, 0, 20)

### 9.4 Final Score and Tiers

    risk_score = clamp(A + B + C, 0, 100)

    [0,  20) → low        (good)
    [20, 45) → moderate   (marginal)
    [45, 70) → high       (impaired)
    [70,100] → critical   (not viable)

---

## 10. Local Search: find_better_nearby()

### 10.1 Key Optimization: Tile Reuse

All candidates within a 50 m search radius fall within the already-fetched
100 m near-field window. No re-downloading — only the center pixel shifts.

### 10.2 Candidate Sampling (ring strategy)

    rings   = [0.3, 0.6, 1.0] × radius_m
    n_pts   = [6,   8,   12]

    for each ring (r, n):
        for i in range(n):
            θ = 2π × i / n
            cand_lat = lat + r×cos(θ) / 111_320
            cand_lon = lon + r×sin(θ) / (111_320 × cos(lat_rad))

26 candidates per call. Grid strategy (10 m step) gives ~78 candidates
for thorough search.

### 10.3 Per-Candidate Analysis (cheap)

For each candidate (cy_c, cx_c) in the existing surface array:

    h_dish_c = S_terrain[cy_c, cx_c] + dish_height
    H_full_c = compute_horizon_profile(S_full, bbox, center_yx=(cy_c, cx_c),
                                        h_dish_override=h_dish_c)
    risk_c   = score_risk(classify_obstruction(...))

Return the candidate with lowest risk_score (feasible candidates ranked first).

---

## 11. Function Architecture

    analyze_location(lat, lon)
    │
    ├── fetch_terrain(lat, lon, R_FAR=1500, res=10)     → {array, bbox, ...}
    ├── fetch_terrain(lat, lon, R_NEAR=100, res=1)      → {array, bbox, ...}
    ├── fetch_canopy(lat, lon, R_NEAR=100)              → {array, bbox, ...}
    ├── fetch_buildings(lat, lon, R_NEAR=100)           → {gdf, count, ...}
    │
    ├── build_obstruction_surface(terrain_near, canopy, buildings)
    │   ├── resample_to_grid(canopy, ...)
    │   ├── classify_building_usability(gdf, canopy_r, ...)
    │   └── → {surface, terrain, canopy_r, building_usable, gdf_classified}
    │
    ├── compute_horizon_profile(terrain_far, bbox, h_dish)    → H_far[N]
    ├── compute_horizon_profile(terrain_near, bbox, h_dish)   → H_near[N]
    ├── compute_horizon_profile(S_canopy, bbox, h_dish)       → H_canopy[N]
    ├── compute_horizon_profile(S_full, bbox, h_dish)         → H_full[N]
    │
    ├── evaluate_blockage(H, fov_mask, threshold)    → {bf, max_angle, blocked[]}
    ├── compute_local_slope(dem, cy, cx, res_m)      → slope_deg
    ├── point_in_footprint(lat, lon, gdf)            → (bool, building_row)
    │
    ├── _evaluate_constraints(...)                   → {C_terrain_large, ...}
    ├── classify_obstruction(H_far, H_canopy, H_full)→ {dominant, contribs, ...}
    ├── score_risk(classification)                   → {score, tier, components}
    │
    └── find_better_nearby(lat, lon, tiles, radius_m=50)
        ├── sample_candidates(lat, lon, radius_m)
        ├── for each c: compute_horizon_profile(center_yx=(cy_c,cx_c))
        │              evaluate_blockage() → score_risk()
        └── → {best, improvement, all_candidates}

---

## 12. Assumptions

| Parameter                   | Value    | Rationale                                    |
|-----------------------------|----------|----------------------------------------------|
| Dish mount height           | 3.0 m    | Pole or eave mount; Starlink spec             |
| Sky threshold angle         | 25°      | Starlink minimum operational elevation        |
| FOV center (CONUS)          | 0° N     | 53° orbit inclination                         |
| FOV half-width              | 50°      | ±50° ≈ 100° arc matching Starlink spec        |
| Vegetation permanence       | 0.6      | Deciduous winter coverage loss ~40%           |
| Max install slope           | 20°      | Beyond this, standard ground mount fails      |
| Building height minimum     | 2.5 m    | Below this, ML height predictions unreliable  |
| Canopy-on-roof threshold    | 0.5 m    | Any detectable canopy = impractical install   |
| Terrain large tolerance     | 10%      | Orbital redundancy absorbs marginal clipping  |
| Building blockage tolerance | 5%       | Buildings are permanent — tight tolerance     |
| Vegetation tolerance        | 15%      | Seasonal — looser tolerance                   |
| Local search radius         | 50 m     | Practical relocation distance                 |

---

## 13. Edge Cases

| Case                             | Handling                                                      |
|----------------------------------|---------------------------------------------------------------|
| NaN in DEM                       | Fill with nanmedian (not 0 — avoids artificial valleys)       |
| Point on cliff / DEM void        | Detect via local std > 30 m; inflate slope constraint         |
| Missing building height          | Area heuristic: floors = max(1, min(area/150, 10)) × 3.5 m   |
| All canopy data simulated        | Tag result; lower confidence on C_vegetation                  |
| No buildings in tile             | Flag as potential undercount; note in warnings                |
| Water body                       | Detect via elevation < 0 or mask; return infeasible           |
| Horizon ray hits grid boundary   | Break at boundary; note effective analysis radius             |
| On-building, roof has vegetation | C_roof_usable = False                                         |
| Search candidate at tile edge    | Truncated horizon; valid for relative ranking, not absolute   |
