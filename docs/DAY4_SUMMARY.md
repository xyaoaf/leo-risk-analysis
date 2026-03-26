# Day 4 Engineering Audit — LEO Risk Analysis Pipeline

**Date**: 2026-03-26
**Author**: Engineering Review (Claude Code)
**Scope**: Full repository audit against intended system design specification

---

## Executive Summary

The LEO Risk Analysis pipeline is architecturally sound and covers all five intended workflow layers. The core physics (ray-casting, multi-scale radius, FOV masking) are correctly implemented. However, the audit identified **three critical code bugs** in the scoring and classification path that produce incorrect or contradictory outputs for a meaningful subset of locations — specifically anywhere terrain or buildings dominate. These bugs must be fixed before results are published or reported. Seven additional gaps were identified at the medium-priority level.

**Critical (P1–P3)**: Fix before publishing any batch results
**High (P4)**: Fix before any long batch run to avoid data loss
**Medium (P5–P7)**: Improve before final submission
**Low (P8–P11)**: Nice-to-have polish

---

## 1. System Objectives — Mapping

| Question | Status | Notes |
|---|---|---|
| Is this location feasible for Starlink installation? | ✅ Fully answered | 6 hard constraints, logical AND |
| What is the dominant obstruction type? | ⚠️ Partially correct | Bug in terrain baseline (Gap 3) |
| How severe is the obstruction? | ⚠️ Partially correct | Bug in angle severity scope (Gap 2) |
| What risk tier is this location? | ⚠️ Contradicted | Terrain-dominant infeasible → risk=0 (Gap 1) |
| Where nearby is better? | ✅ Implemented | Local search with distance penalty |
| What does the area look like spatially? | ✅ Implemented | All 5 output modes (maps, web app, zonal) |

---

## 2. Analysis Architecture — Audit

### 2.1 Multi-Scale Design (Correct)

The pipeline correctly separates two spatial scales:

| Scale | Radius | Surface | Purpose |
|---|---|---|---|
| Far-field | 1500 m | `terrain_far` (DEM only) | Regional terrain horizon |
| Near-field | 100 m | `S_terrain`, `S_canopy`, `S_full` | Vegetation + building detail |

This is physically correct: a ridge 1 km away blocks Starlink at ≈1° elevation angle, while a tree 30 m away blocks it at ≈18°. Using a single radius for both would miss rolling terrain (if too small) or waste building tile downloads (if too large).

Four horizon profiles are computed:
- `hz_far` — far-field terrain (1500 m)
- `hz_terrain` — near-field bare terrain (100 m)
- `hz_canopy` — near-field terrain + canopy (100 m)
- `hz_full` — near-field terrain + canopy + buildings (100 m)

### 2.2 Feasibility Constraints (Correct)

All six constraints are implemented correctly and use the right spatial scale:

```
C_terrain_large  → hz_far  blockage ≤ 10%    (far-field, correct)
C_terrain_small  → slope < 20°               (local DEM, correct)
C_veg_at_point   → canopy height at dish < 1m (point sample, correct)
C_vegetation     → hz_canopy blockage ≤ 15%  (near-field, correct)
C_building_nearby→ hz_full  blockage ≤ 5%    (near-field, correct)
C_roof_usable    → vegetation overhang check  (conditional, correct)
```

`feasibility.py:211` also correctly computes canopy contribution using `hz_terrain` as baseline:
```python
canopy_contrib_frac = max(0.0, blk_can["blocked_frac"] - evaluate_blockage(hz_terrain, fov)["blocked_frac"])
```

### 2.3 Scoring Path — THREE CRITICAL BUGS

---

## 3. Critical Gap Analysis

---

### Gap 1 (P1 — CRITICAL): Risk Score Is Always Zero for Terrain-Dominant Infeasible Locations

**File**: `src/scoring.py`, line 68
**Severity**: Silent wrong output. Infeasible location shows `risk_score=0`, `risk_tier="low"` — directly contradicts `feasible=False`.

**Root cause**: The score formula unconditionally uses `blocked_frac_buildings` (near-field 100 m) as the blockage fraction input — regardless of which layer is dominant.

```python
# scoring.py:68-69 — BUG
bf      = float(classification.get("blocked_frac_buildings", 0.0))  # near-field ALWAYS
max_ang = float(classification.get("max_angle_buildings",    0.0))  # near-field ALWAYS
```

When a far-field terrain ridge makes the location infeasible (`C_terrain_large` fails, `dominant="terrain"`), the near-field 100 m window is clear. So `blocked_frac_buildings=0` → `fov_score=0`, `angle_score=0`, `perm_score=0` → `risk_score=0`, `risk_tier="low"`.

**Impact**: Every terrain-dominant infeasible location shows contradictory output. The Charlotte and Harnett test points are vegetation-dominant (correct), but any point near a ridge would trigger this bug.

**Fix**: Use the correct blockage fraction for each dominant type:

```python
# scoring.py — PROPOSED FIX
dominant = str(classification.get("dominant", "clear"))
if dominant == "terrain":
    bf      = float(classification.get("blocked_frac_terrain",   0.0))
    max_ang = float(classification.get("max_angle_terrain",      0.0))
elif dominant == "vegetation":
    bf      = float(classification.get("canopy_contribution",    0.0))
    max_ang = float(classification.get("max_angle_canopy",       0.0))
else:  # building or clear
    bf      = float(classification.get("blocked_frac_buildings", 0.0))
    max_ang = float(classification.get("max_angle_buildings",    0.0))
```

---

### Gap 2 (P2 — CRITICAL): Angle Severity Uses Global Max, Not FOV-Filtered Max

**File**: `src/tools/horizon.py`, lines 284–286
**Severity**: Inflates `angle_score` for any obstruction outside the Starlink FOV (e.g., buildings or trees to the south). A 30 m building due south at 60° elevation angle would add 25 angle points to a location that is actually clear to the north.

**Root cause**: `classify_obstruction()` computes `max_angle_*` as `.max()` over all 72 azimuths:

```python
# horizon.py:284-286 — BUG
"max_angle_terrain":   float(horizon_terrain.max()),    # all 72 azimuths
"max_angle_canopy":    float(horizon_canopy.max()),     # all 72 azimuths
"max_angle_buildings": float(horizon_buildings.max()),  # all 72 azimuths
```

The FOV mask is already computed inside `classify_obstruction()` (`mask = fov_mask(n_azimuths)`) and used for blockage fractions — but not for the max angle outputs.

**Fix**: Filter by `mask` before taking max:

```python
# horizon.py — PROPOSED FIX
"max_angle_terrain":   float(horizon_terrain[mask].max()),
"max_angle_canopy":    float(horizon_canopy[mask].max()),
"max_angle_buildings": float(horizon_buildings[mask].max()),
```

Note: `evaluate_blockage()` already returns both `max_angle_deg` (FOV-filtered) and `max_angle_all_deg` (global) — this distinction should be the same in `classify_obstruction()`.

---

### Gap 3 (P3 — CRITICAL): Canopy Baseline Scale Mismatch in `classify_obstruction()`

**File**: `src/feasibility.py`, line 227
**Severity**: Canopy contribution is systematically underestimated in terrain-heavy areas, leading to misclassification of dominant obstruction type.

**Root cause**: `classify_obstruction()` receives `hz_far` (1500 m terrain) as its terrain baseline, then computes canopy contribution as `bf_c - bf_t`. But `hz_canopy` is at 100 m scale. The two scales are incompatible:

```python
# feasibility.py:227 — BUG
classification = classify_obstruction(hz_far, hz_canopy, hz_full, n_az)
#                                     ^^^^^
#                                     1500m far-field terrain used as
#                                     baseline for 100m canopy comparison
```

Inside `classify_obstruction()`:
```python
bf_t = bf(horizon_terrain)   # blocked_frac from hz_far (1500m)
bf_c = bf(horizon_canopy)    # blocked_frac from hz_canopy (100m)
canopy_contrib = max(0.0, bf_c - bf_t)   # ← comparing apples and oranges
```

If a terrain ridge is present at 1500 m, `bf_t` can be 0.15 (15% blocked). At 100 m, bare terrain may be flat: `bf_c=0.08`. Result: `canopy_contrib = max(0, 0.08 - 0.15) = 0`. The algorithm concludes there is no vegetation, when in fact there is moderate canopy.

Contrast with `feasibility.py:211`, which correctly uses the near-field terrain baseline:
```python
canopy_contrib_frac = max(0.0, blk_can["blocked_frac"] - evaluate_blockage(hz_terrain, fov)["blocked_frac"])
```

**Fix**: Pass `hz_terrain` (near-field bare) as the terrain baseline:

```python
# feasibility.py:227 — PROPOSED FIX
classification = classify_obstruction(hz_terrain, hz_canopy, hz_full, n_az)
#                                     ^^^^^^^^^
#                                     near-field bare terrain (100m) — correct scale
```

Note: This changes the semantics of `blocked_frac_terrain` in the output (it will reflect near-field terrain, not far-field). The feasibility constraint still uses `hz_far` directly via `blk_far` — that is correct and unchanged.

---

### Gap 4 (P4 — HIGH): Batch Results Lost on Crash — No Incremental Writes

**File**: `batch_nc_analysis.py`
**Severity**: If the batch process is killed (OOM, network timeout, power loss) after processing 500 of 594 points, all 500 results are lost. The CSV is only written after `as_completed()` exhausts all futures.

**Fix**: Open the output CSV in append mode at the start of the batch, write each result immediately after `future.result()` returns, and track progress via a completion set. This is a standard streaming-results pattern for long-running batch jobs.

---

### Gap 5 (P5 — MEDIUM): No Explicit Rooftop Opportunity Field

**File**: `src/feasibility.py`
**Issue**: When `on_building=True` and the analysis passes, there is no output field explicitly labeling this as a "rooftop mount opportunity." The downstream consumer must infer this from `on_building + feasible + failure_mode`. A dedicated `mount_type: "ground" | "rooftop"` field would make the API clearer.

---

### Gap 6 (P6 — MEDIUM): Polar Plot Missing Scale Annotation

**File**: `main.py`, `_plot_polar_panel()`
**Issue**: The polar plot shows terrain at 1500 m radius but canopy/buildings at 100 m radius. The three horizon profiles are drawn on the same polar axes with no annotation that the terrain profile comes from a different spatial extent. A reader could interpret the terrain ring as representing the same 100 m window as canopy/buildings.

**Fix**: Add a text annotation near the terrain fill (e.g., "1500 m radius") and near the canopy ring ("100 m radius"), or use a dual-scale axis.

---

### Gap 7 (P7 — MEDIUM): Zonal Summary Missing Census Tract Level

**File**: `zonal_summary.py`
**Issue**: Aggregation is at block-group and county. Census tract (11-digit GEOID) is the standard unit used by the FCC for broadband coverage analysis and by most policy researchers. Block-group is finer but not widely used in policy context; county is too coarse for infrastructure planning.

**Fix**: Add `geoid_cb[:11]` (tract) as an aggregation level between block-group and county.

---

## 4. Output Modes — Status Summary

| Mode | Script | Status | Known Issues |
|---|---|---|---|
| Single-point diagnostic | `run_analysis.py` | ✅ Complete | Score wrong for terrain-dominant (Gap 1) |
| Batch processing | `batch_nc_analysis.py` | ✅ Complete | No incremental writes (Gap 4) |
| Static Folium map | `make_folium_map.py` | ✅ Complete | None |
| Web app (FastAPI) | `app.py` + `templates/map.html` | ✅ Complete | None |
| AOI screening | `aoi_screen.py` | 🔶 Prototype | None |
| Zonal summary | `zonal_summary.py` | 🔶 Prototype | Missing tract level (Gap 7) |

---

## 5. Scoring Formula — Verification

The intended formula from `docs/DESIGN.md`:

```
risk_score = fov_blockage_score (0–50)
           + angle_severity_score (0–30)
           + permanence_score (0–20)
```

**Component 1 — FOV blockage**: `bf * 50` where `bf` = fraction of FOV azimuths blocked above 25°
→ Currently: `bf = blocked_frac_buildings` (near-field ALWAYS) — **Bug: Gap 1**

**Component 2 — Angle severity**: `(max_ang - 25) / 65 * 30` for max_ang > 25°
→ Currently: `max_ang = max_angle_buildings.max()` over all 72 azimuths — **Bug: Gap 2**

**Component 3 — Permanence**: `permanence_factor * min(bf * 40, 20)` capped at 20
→ Correct in formula; inherits `bf` bug from Component 1

**Permanence factors**: building/terrain = 1.0, vegetation = 0.6, clear = 0.0 — Correct.

**Tiers**: 0–19 low, 20–44 moderate, 45–69 high, 70–100 critical — Correct.

---

## 6. Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Published results have risk=0 for infeasible terrain locations | High | High | Fix Gap 1 before batch results used |
| South-facing tall buildings inflate angle score | Medium | Medium | Fix Gap 2 |
| Vegetation misclassified in hilly terrain | Medium | Medium | Fix Gap 3 |
| 594-pt batch lost on crash | Low | High | Fix Gap 4 (incremental writes) |
| Canopy GEE quota exceeded at scale | Medium | High | Document; pre-cache by county |
| Results differ between single-point and batch (different scoring path) | Low | Medium | Verify after Gaps 1–3 are fixed |

---

## 7. Recommended Improvements — Prioritized

### P1 — Fix risk score for terrain-dominant locations (scoring.py:68)
**Effort**: 15 min. **Impact**: Correct output for all terrain-infeasible locations.

### P2 — Fix angle severity to use FOV-filtered max (horizon.py:284)
**Effort**: 5 min. **Impact**: Prevent south-facing obstructions from inflating score.

### P3 — Fix canopy classification baseline (feasibility.py:227)
**Effort**: 5 min (change `hz_far` → `hz_terrain` in one function call). **Impact**: Correct dominant classification in hilly terrain areas.

### P4 — Incremental batch result writes
**Effort**: 30 min. **Impact**: Batch run is restartable; no work lost on crash.

### P5 — Add mount_type field to result schema
**Effort**: 10 min. **Impact**: Cleaner API; easier downstream filtering.

### P6 — Annotate polar plot radius scales
**Effort**: 15 min. **Impact**: Prevents misinterpretation of the 5-panel figure.

### P7 — Add census tract aggregation to zonal_summary.py
**Effort**: 20 min. **Impact**: Results usable in FCC/policy context.

### P8 — Async web app analysis (non-blocking)
**Effort**: 2 hrs. **Impact**: Prevents browser timeout on slow locations (>30s).

### P9 — Soft vegetation metric (seasonal adjustment)
**Effort**: 1 hr. **Impact**: Distinguish conifer (year-round) vs deciduous (seasonal).

### P10 — Latitude-adaptive FOV center
**Effort**: 1 hr. **Impact**: Correct FOV geometry for northern latitudes.

### P11 — Building height confidence tags in output
**Effort**: 30 min. **Impact**: Flag locations where ML height estimates are uncertain.

---

## 8. Implementation Status (as of Day 4)

| Component | Design Spec | Implemented | Correct |
|---|---|---|---|
| Terrain fetch (3DEP, 4-attempt backoff) | ✅ | ✅ | ✅ |
| Canopy fetch (GEE 1m + 27m fallback) | ✅ | ✅ | ✅ |
| Building footprints (Microsoft primary) | ✅ | ✅ | ✅ |
| Multi-scale radius (1500m / 100m) | ✅ | ✅ | ✅ |
| Horizon ray-casting (vectorised) | ✅ | ✅ | ✅ |
| FOV mask (±50° around North) | ✅ | ✅ | ✅ |
| Feasibility constraints (6 checks) | ✅ | ✅ | ✅ |
| Dominant obstruction classification | ✅ | ✅ | ⚠️ Gap 3 |
| Risk score formula | ✅ | ✅ | ⚠️ Gaps 1, 2 |
| Local search (find_better_nearby) | ✅ | ✅ | ✅ |
| 5-panel visualization | ✅ | ✅ | ✅ |
| Batch processing | ✅ | ✅ | ⚠️ Gap 4 |
| Interactive Folium map | ✅ | ✅ | ✅ |
| FastAPI web app | ✅ | ✅ | ✅ |
| AOI screening | ✅ | 🔶 Prototype | ✅ |
| Zonal summary | ✅ | 🔶 Prototype | ⚠️ Gap 7 |

---

## 9. Next Steps — Recommended Sprint Plan

### Immediate (before using batch results):
1. Apply P1: Fix `scoring.py:68` — select blockage fraction by dominant type
2. Apply P2: Fix `horizon.py:284-286` — filter max angle by FOV mask
3. Apply P3: Fix `feasibility.py:227` — pass `hz_terrain` not `hz_far` to `classify_obstruction()`
4. Apply P4: Add incremental CSV writes to `batch_nc_analysis.py`
5. Regenerate the 8-point NC validation set and Charlotte reports after fixes

### After batch completes:
6. Apply P6: Add radius annotations to polar plot
7. Apply P7: Add census tract level to `zonal_summary.py`
8. Run `zonal_summary.py` on full batch results

### Polish before submission:
9. Apply P5: Add `mount_type` field
10. Review and update `docs/DATA_SOURCES.md` with any new findings from batch run
11. Verify the web app loads batch results after server restart

---

*This report was generated by an engineering audit of the full codebase against the system design specification in `docs/DESIGN.md` and `docs/ARCHITECTURE.md`.*
