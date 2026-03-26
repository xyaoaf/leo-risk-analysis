# System Workflows

This document describes the five user-facing workflows supported by the LEO risk analysis
pipeline, with entry commands, input/output specs, and implementation status for each.

---

## Status legend

| Symbol | Meaning |
|--------|---------|
| ✅ Implemented | Fully working; outputs exist or can be reproduced by running the command |
| 🔶 Prototype | Working code exists; not yet integrated into a UI or fully productized |
| 🔲 Future | Design documented; not yet implemented |

---

## Workflow overview

```
┌──────────────────────────────────────────────────────────────┐
│                    User entry points                         │
│                                                              │
│  [1] Single point   [2] Batch CSV   [4] AOI polygon          │
│       ↓                  ↓               ↓                   │
│  run_analysis.py    batch_nc_analysis.py ← aoi_screen.py     │
│       ↓                  ↓                                   │
│  outputs/reports/   outputs/batch/                           │
│       ↓                  ↓                                   │
│  [3] Interactive map (make_folium_map.py)                    │
│       ↓                                                      │
│  [5] Zonal summary  (zonal_summary.py)                       │
│       ↓                                                      │
│  outputs/zonal/  →  planning / grant reporting               │
└──────────────────────────────────────────────────────────────┘
```

All five workflows call the same geospatial pipeline (`src/feasibility.py →
analyze_location()`), which is deterministic and contains no LLM calls.
Claude is used only in the interactive agent mode (`src/agent.py`) for intent
parsing and natural language reporting.

---

## Layer 1 — Single-point diagnostic report ✅

**Purpose**: Detailed suitability assessment for one address or GPS coordinate.
Suitable for field technicians, grant reviewers, or developers testing the pipeline.

### Entry point

```bash
conda run -n cs378 python run_analysis.py LAT LON [--id ID] [--no-search] [--out-dir DIR]
```

### Examples

```bash
# Charlotte NC — dense canopy
conda run -n cs378 python run_analysis.py 35.061 -80.666 --id charlotte

# Jones County NC — coastal flat farmland
conda run -n cs378 python run_analysis.py 34.752 -77.296 --id jones_county

# Skip neighbourhood search (faster, ~30s instead of ~60s)
conda run -n cs378 python run_analysis.py 35.061 -80.666 --id charlotte --no-search
```

### Inputs

| Parameter | Description |
|-----------|-------------|
| `LAT LON` | WGS84 decimal degrees (required) |
| `--id` | Label used in filenames and the report header |
| `--no-search` | Skip the 50 m neighbourhood search |
| `--out-dir` | Output directory root (default: `outputs/reports`) |

### Outputs

```
outputs/reports/{id}/
  {id}.png   — 4-panel visual: terrain hillshade · canopy+buildings · constraints · polar horizon
  {id}.json  — Full structured result (all fields, all constraint values)
```

The terminal also prints a formatted diagnostic report:

```
══════════════════════════════════════════════════════════════
  LEO SATELLITE RISK DIAGNOSTIC REPORT
══════════════════════════════════════════════════════════════
  Location : charlotte  (35.06100, -80.66600)
  Elapsed  : 45.2 s
──────────────────────────────────────────────────────────────
  RESULT   :  ✗  CRITICAL  (risk = 90/100)
  Feasible :  NO  [failure: local_canopy]
  Dominant :  VEGETATION
──────────────────────────────────────────────────────────────
  CONSTRAINTS
  ✓  Terrain horizon  ≤10% FOV    blocked=0%   max=-0.0°
  ✓  Local slope      <20°        slope=1.6°
  ✗  Canopy at dish   <1 m        canopy=27.0 m
  ✗  Vegetation FOV   ≤15%        blocked=82%  max=89.9°
  ✓  Buildings FOV    ≤5%         blocked=0%   max=0.0°
...
```

### What is implemented vs simplified

| Feature | Status |
|---------|--------|
| All 6 hard constraints | ✅ Fully implemented |
| 4-panel PNG visual | ✅ Identical to main.py demo |
| Neighbourhood search (50 m) | ✅ Implemented (can be skipped with --no-search) |
| JSON output (all fields) | ✅ Implemented |
| Interactive agent mode | ✅ Available via `python src/agent.py "Analyze..."` |
| Web API / on-demand endpoint | 🔲 Future (FastAPI or Lambda wrapper around run_analysis.py) |

---

## Layer 2 — Batch processing workflow ✅

**Purpose**: Evaluate thousands of addresses from a CSV, producing aggregate statistics
and choropleth maps. Designed for state broadband offices running county-level screening.

### Entry point

```bash
conda run -n cs378 python batch_nc_analysis.py [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--workers N` | `2` | Parallel workers. **Max recommended: 2** to avoid 3DEP rate-limiting. |
| `--pts-per-county N` | `5` | Points per county in stratified sample (ignored with `--input-csv`) |
| `--max-points N` | none | **Dry-run safety valve.** Truncate to first N points before running. |
| `--input-csv FILE` | none | Pre-filtered CSV from `aoi_screen.py`. Skips stratified sampling. |
| `--skip-pipeline` | false | Skip pipeline; reload existing results CSV and regenerate maps only. |

### Examples

```bash
# Full NC batch (~594 points, ~2–3 hours)
conda run -n cs378 python batch_nc_analysis.py

# Dry-run: 10 points to validate pipeline end-to-end
conda run -n cs378 python batch_nc_analysis.py --max-points 10

# Charlotte AOI subset (from aoi_screen.py output)
conda run -n cs378 python batch_nc_analysis.py \
    --input-csv outputs/aoi/charlotte_metro_points.csv \
    --workers 2

# Regenerate maps from existing results without re-running pipeline
conda run -n cs378 python batch_nc_analysis.py --skip-pipeline
```

### Input schema

Any CSV with these columns (produced by `DATA_CHALLENGE_50.csv` or `aoi_screen.py`):

| Column | Type | Description |
|--------|------|-------------|
| `location_id` | string | Unique address identifier |
| `latitude` | float | WGS84 decimal degrees |
| `longitude` | float | WGS84 decimal degrees |
| `geoid_cb` | string | 15-digit Census block GEOID |
| `county_fips` | string | 5-digit FIPS code (derived from geoid_cb if absent) |

### Outputs

```
outputs/batch/
  batch_results.csv          — Per-point results: risk score, tier, dominant, constraints
  batch_county_summary.csv   — County-level aggregates (mean risk, % feasible, % critical)
  batch_sample_points.csv    — The stratified sample that was analyzed
  map_county_risk.png        — Choropleth: mean risk + % infeasible by county
  map_risk_scatter.png       — Scatter: all tested points colored by tier
  chart_tier_dist.png        — Bar chart: tier distribution + failure mode breakdown
```

### Reliability notes (terrain source)

The pipeline fetches terrain from **USGS 3DEP WMS** via `py3dep` for every point
(two requests per point: 1500 m far-field at 10 m/px + 100 m near-field at 1 m/px).
The endpoint rate-limits concurrent bursts:

- Default workers changed from 6 → **2** after observing 100% failure at 6 workers
- Each worker is staggered by 8 s at startup to spread the initial burst
- Retry logic uses exponential backoff (5 s → 10 s → 20 s, up to 4 attempts per resolution)
- Full resolution fallback ladder: 1 m → 3 m → 10 m → 30 m

Expected throughput: **~120 points/hour** at 2 workers (~30 s average per point including
occasional backoff retries). Full 594-point run takes ~2–3 hours.

---

## Layer 3 — Interactive map exploration ✅

**Purpose**: Zoomable, clickable map showing all analyzed points color-coded by risk tier.
Suitable for visual exploration by planners or grant reviewers who want geographic context.

### Entry point

```bash
# Build map from currently available data
conda run -n cs378 python make_folium_map.py

# Include full batch results (after batch run completes)
conda run -n cs378 python make_folium_map.py --batch-csv outputs/batch/batch_results.csv
```

### Current data in the map

| Layer | Points | Status |
|-------|--------|--------|
| Challenge-50 sample | 37 NC points | ✅ Available now |
| NC validation set | 8 geographically spread NC points | ✅ Available now |
| Full NC batch | ~594 NC points | 🔶 Populated after batch run |

### Map features

- **Tier-colored markers**: green (low) · orange (moderate) · dark red (high) · red (critical)
- **Clickable popups**: location ID, score, feasibility, dominant obstruction, slope, canopy, buildings
- **Layer toggle**: show/hide each risk tier independently
- **Stats box**: tier distribution per dataset in a floating legend
- **"Infeasible" indicator**: × icon on marker for infeasible locations

### Output

```
outputs/nc_risk_map.html   — self-contained HTML (opens in any browser)
```

### Current capabilities vs future extension

| Capability | Status |
|-----------|--------|
| View pre-computed points | ✅ Implemented |
| Filter by tier (layer toggle) | ✅ Implemented |
| Popup with all attributes | ✅ Implemented |
| Load full batch results | ✅ After batch: `--batch-csv outputs/batch/batch_results.csv` |
| Click-to-compute (on-demand) | 🔲 Future: requires a web backend (FastAPI / AWS Lambda) calling `run_analysis.py` |
| County choropleth overlay | 🔲 Future: merge batch county summary into map as a tile layer |

---

## Layer 4 — AOI-based screening 🔶

**Purpose**: Given a geographic area of interest (county, neighborhood, bounding box),
extract all matching address points from the master dataset and feed them into the batch pipeline.

**Status**: Prototype — script-based, no UI.

### Entry point

```bash
conda run -n cs378 python aoi_screen.py [--bbox | --geojson] --out OUTPUT_CSV
```

### Examples

```bash
# Bounding-box filter: Charlotte metro area
conda run -n cs378 python aoi_screen.py \
    --bbox 35.0 35.5 -81.1 -80.4 \
    --out  outputs/aoi/charlotte_metro_points.csv

# GeoJSON polygon filter
conda run -n cs378 python aoi_screen.py \
    --geojson data/aoi/mecklenburg_county.geojson \
    --out     outputs/aoi/mecklenburg_points.csv

# Then run batch on the selected points:
conda run -n cs378 python batch_nc_analysis.py \
    --input-csv outputs/aoi/charlotte_metro_points.csv \
    --max-points 50 --workers 2
```

### Supported AOI input formats

| Format | Flag | Notes |
|--------|------|-------|
| Bounding box | `--bbox lat_min lat_max lon_min lon_max` | Vectorised, fast (< 2 s for 4.67M rows) |
| GeoJSON Polygon / MultiPolygon | `--geojson FILE` | Requires geopandas; uses spatial join |

### Output CSV schema

Same as `DATA_CHALLENGE_50.csv`:
`location_id, latitude, longitude, geoid_cb, county_fips`

This CSV can be passed directly to `batch_nc_analysis.py --input-csv`.

### How AOI screening connects to the batch pipeline

```
DATA_CHALLENGE_50.csv (4.67M rows)
         ↓
   aoi_screen.py --bbox / --geojson
         ↓
  outputs/aoi/{name}_points.csv   (subset, same schema)
         ↓
  batch_nc_analysis.py --input-csv {file}
         ↓
  outputs/batch/batch_results.csv
```

### Next-step productization path

1. Add a county name → FIPS lookup (so users can type `--county "Mecklenburg"` instead of `--bbox`)
2. Integrate county boundary GeoJSONs from Census TIGER as built-in AOI presets
3. Add a web form or CLI menu to select counties interactively

---

## Layer 5 — Zonal summary / block-group-level aggregation 🔶

**Purpose**: Aggregate point-level risk results to census reporting units (block groups,
counties) to support broadband grant planning and reporting.

**Status**: Prototype — produces real output on the 37-point Challenge-50 sample now;
scales automatically to full batch results when available.

### Entry point

```bash
# Using Challenge-50 sample (available now)
conda run -n cs378 python zonal_summary.py

# Using full NC batch results (after batch run)
conda run -n cs378 python zonal_summary.py \
    --results-csv outputs/batch/batch_results.csv
```

### How geographic IDs are derived

The 15-digit `geoid_cb` column in `DATA_CHALLENGE_50.csv` encodes the Census block GEOID:

```
37  179  020316  2002
│   │    │       │
│   │    │       └── Block number (4 digits)
│   │    └────────── Census tract (6 digits)
│   └─────────────── County FIPS (3 digits)
└─────────────────── State FIPS (2 digits = NC)

Block group = geoid_cb[:12]  (first 12 characters)
County FIPS = geoid_cb[:5]
```

The pipeline joins results back to `DATA_CHALLENGE_50.csv` on `location_id` to recover `geoid_cb`,
then derives block group and county IDs.

### Outputs

```
outputs/zonal/
  block_group_summary.csv   — Per block group: n_points, mean_risk, pct_feasible, pct_critical, ...
  county_summary.csv        — Per county: same fields aggregated to county level
  county_choropleth.png     — 3-panel static map: mean risk / % infeasible / % critical
```

### Aggregated fields per reporting unit

| Field | Description |
|-------|-------------|
| `n_points` | Number of sampled addresses in this unit |
| `n_feasible` / `n_infeasible` | Count of feasible / infeasible locations |
| `pct_feasible` | % feasible locations (0–100) |
| `mean_risk` / `median_risk` | Risk score statistics |
| `pct_critical` / `pct_high` / `pct_moderate` / `pct_low` | Tier breakdown (%) |
| `dominant_mode` | Most common dominant obstruction type (clear / vegetation / building / terrain) |

### Why this matters for grant planning

State broadband offices can use the block-group summary to answer:
- "Which census block groups have the highest proportion of infeasible or critical locations?"
- "Is this a vegetation problem (deciduous = seasonal) or terrain/building (permanent)?"
- "Which counties should get on-site survey budgets?"

The full 594-point batch results will cover all 100 NC counties with ~5 points each —
enough for a county-level screening map but not for block-group-level granularity.
For block-group resolution, 20–50 points per block group would be needed.

### Next-step productization path

1. Run full batch on all 4.67M addresses (pre-filtered to unserved/underserved by BEAD eligibility)
2. Aggregate to block group level (typical: 400–2000 addresses per block group)
3. Join to Census demographic data (income, rural/urban classification, existing broadband tiers)
4. Export to GeoJSON for ESRI/QGIS import or Tableau/PowerBI dashboard

---

## Cross-reference: scripts and their workflows

| Script | Layer | Status |
|--------|-------|--------|
| `run_analysis.py` | 1 — Single point | ✅ |
| `main.py` | 1 — Single point (demo, Austin TX) | ✅ |
| `test_nc_points.py` | 1 — Single point (NC validation, 8 pts) | ✅ |
| `src/agent.py` | 1 — Single point (Claude-orchestrated) | ✅ |
| `batch_nc_analysis.py` | 2 — Batch | ✅ |
| `make_folium_map.py` | 3 — Interactive map | ✅ |
| `aoi_screen.py` | 4 — AOI screening | 🔶 Prototype |
| `zonal_summary.py` | 5 — Zonal summary | 🔶 Prototype |
| `explore_challenge50.py` | 2 + 3 (EDA + maps) | ✅ |
| `remake_maps.py` | 2 (regenerate EDA maps) | ✅ |
