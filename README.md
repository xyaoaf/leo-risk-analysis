# LEO Satellite Coverage Risk Analysis

An agent-driven geospatial pipeline that identifies at-risk locations where environmental
obstructions (terrain, trees, buildings) are likely to degrade Starlink service quality —
even at addresses providers have listed as "served."

---

## Quick Start

```bash
conda activate cs378

# 1. Analyze any single address (Layer 1) — outputs 5-panel PNG + JSON
conda run -n cs378 python run_analysis.py 35.061 -80.666 --id charlotte

# 2. Run batch analysis on NC dataset (~594 stratified points, ~2–3 hrs)
conda run -n cs378 python batch_nc_analysis.py              # full run
conda run -n cs378 python batch_nc_analysis.py --max-points 10  # dry-run (10 pts)

# 3a. Interactive Folium map (static HTML, no server needed)
open outputs/nc_risk_map.html
conda run -n cs378 python make_folium_map.py  # regenerate from latest results

# 3b. Interactive web app with satellite imagery + click-to-compute + draw-to-batch
conda run -n cs378 uvicorn app:app --reload --port 8000
# → open http://localhost:8000 in a browser

# 4. Filter an AOI and batch-analyze it (Layer 4)
conda run -n cs378 python aoi_screen.py --bbox 35.0 35.5 -81.1 -80.4 \
    --out outputs/aoi/charlotte_points.csv
conda run -n cs378 python batch_nc_analysis.py \
    --input-csv outputs/aoi/charlotte_points.csv --workers 2

# 5. Aggregate results to census block groups / counties (Layer 5)
conda run -n cs378 python zonal_summary.py

# Run the Claude agent interactively
export ANTHROPIC_API_KEY=sk-ant-...
conda run -n cs378 python src/agent.py "Analyze Starlink risk at 35.06, -80.67"
```

See [docs/WORKFLOWS.md](docs/WORKFLOWS.md) for full documentation of all five workflows.

---

## System Workflows

Five workflows are supported; see [docs/WORKFLOWS.md](docs/WORKFLOWS.md) for full documentation.

| # | Workflow | Script | Status |
|---|----------|--------|--------|
| 1 | **Single-point diagnostic** | `run_analysis.py LAT LON` | ✅ Implemented |
| 2 | **Batch processing** | `batch_nc_analysis.py` | ✅ Implemented |
| 3a | **Static interactive map** | `make_folium_map.py` → `outputs/nc_risk_map.html` | ✅ Implemented |
| 3b | **Web app** (satellite + click-to-compute + draw-to-batch) | `uvicorn app:app --port 8000` | ✅ Implemented |
| 4 | **AOI screening** | `aoi_screen.py --bbox / --geojson` | 🔶 Prototype |
| 5 | **Zonal summary** | `zonal_summary.py` | 🔶 Prototype |

---

## Repository Structure

```
leo_risk_analysis/
├── run_analysis.py            # [Layer 1] Single-point CLI diagnostic (5-panel PNG + JSON)
├── main.py                    # [Layer 1] Demo runner (5 Austin TX test points)
├── test_nc_points.py          # [Layer 1] 8-point NC validation set
├── batch_nc_analysis.py       # [Layer 2] Stratified batch runner + county charts
├── make_folium_map.py         # [Layer 3] Interactive Folium map builder (static HTML)
├── app.py                     # [Layer 3] FastAPI web app (satellite map, click-to-compute)
├── templates/map.html         # [Layer 3] Leaflet.js map template (Esri satellite + Draw)
├── aoi_screen.py              # [Layer 4] AOI bbox/polygon point filter
├── zonal_summary.py           # [Layer 5] Block-group / county aggregation
├── explore_challenge50.py     # EDA + pipeline test on Challenge-50 dataset
├── remake_maps.py             # Regenerate publication-ready EDA maps
├── src/
│   ├── feasibility.py         # analyze_location() — orchestrator + constraints
│   ├── agent.py               # Claude-orchestrated agent (tool_use loop)
│   ├── scoring.py             # Deterministic 0–100 risk score
│   └── tools/
│       ├── terrain.py         # USGS 3DEP DEM (1m/3m/10m/30m, 4-attempt backoff)
│       ├── canopy.py          # Meta Trees 1m via GEE · 27m S3 fallback
│       ├── buildings.py       # Microsoft ML Footprints (Google fallback)
│       ├── surface.py         # Obstruction surface compositor
│       └── horizon.py         # Vectorised ray-casting + classifier
├── docs/
│   ├── WORKFLOWS.md           # All five workflow layers (entry commands + I/O specs)
│   ├── ARCHITECTURE.md        # Mermaid diagram + component table
│   ├── DESIGN.md              # Full algorithm specification
│   ├── ANALYSIS_RATIONALE.md  # From install guide to methodology
│   ├── DATA_SOURCES.md        # Dataset choices + quality issues
│   └── CHALLENGE.md           # Original challenge specification
├── data/
│   ├── buildings/             # Cached building tile GeoParquets (by quadkey)
│   └── tiles/canopy/          # Cached GEE 1m canopy GeoTIFFs
└── outputs/
    ├── reports/               # [Layer 1] Per-point reports (run_analysis.py)
    ├── batch/                 # [Layer 2] Batch results + county maps
    ├── challenge50/           # Challenge-50 sample results (37 pts)
    ├── nc_test/               # NC validation set (8 pts)
    ├── zonal/                 # [Layer 5] Block-group + county summaries
    ├── aoi/                   # [Layer 4] AOI-filtered point CSVs
    └── nc_risk_map.html       # [Layer 3] Interactive map (open in browser)
```

---

## Decision Log

### 1. No LLM in the scoring loop

**Decision**: All risk computation is deterministic Python (horizon ray-casting + scoring formula).
Claude is used only for workflow orchestration and natural language explanation.

**Alternatives considered**:
- LLM as scorer: "Given this canopy description, rate the risk 0–10"
- ML model trained on known signal quality outcomes

**Reasoning**: At 1M locations, per-location LLM calls would cost ~$500–2000 and take hours.
A deterministic formula runs in seconds, is reproducible (same inputs → same output), and
explainable (every score component has a physical meaning).

**What I'd revisit**: Train a lightweight ML model on real Starlink signal quality data
(if available) to calibrate the score weights against ground truth.

---

### 2. Microsoft Building Footprints as primary (not Google)

**Decision**: Microsoft ML Building Footprints is the primary source for CONUS; Google Open
Buildings is fallback.

**Alternatives considered**: Google first, Microsoft fallback (original implementation).

**Reasoning**: Google Open Buildings has near-zero coverage for US metros. Austin's S2 tile
covers a 6°×5° area but physically only contains buildings from a rural border region
(Del Rio TX), returning 0 buildings for Austin. Microsoft has full CONUS coverage with ML
heights. Swapped priority after empirical testing.

**What I'd revisit**: Source OpenStreetMap building data as a third fallback for regions
where both are sparse.

---

### 3. Canopy data source: GEE Meta Trees 1m (Tolan et al. 2024)

**Decision**: Use the Meta Trees 1m product accessed via Google Earth Engine
(`projects/sat-io/open-datasets/facebook/meta-canopy-height`) as primary canopy source.
The 27m ALSGEDI S3 product is retained as a fallback.

**Alternatives considered**:
- 27m S3 as primary (original implementation)
- 1m S3 tiles as primary

**Reasoning**: The 27m product is derived from GEDI satellite LiDAR (~25m footprints averaged
to 27m pixels) — it has no genuine sub-30m spatial information. At a 100m analysis radius it
resolves to only ~4×7 pixels, barely a texture. The 1m S3 tiles are not Cloud-Optimised
(strip layout, blockysize=1) — streaming a 200m window takes ~40s.

The GEE product (Tolan et al. 2024, *Remote Sensing of Environment*) is genuinely 1m, derived
from Maxar satellite imagery using DiNOv2 vision features. Access via `ee.data.computePixels()`
returns a 200×200 array in ~0.2s. Results are cached locally as GeoTIFFs so repeated calls
are instant. **GEE licensing note**: prototype uses a personal non-profit GEE account;
production deployment requires a commercial GEE license.

**What I'd revisit**: The GEE dependency adds an authentication step (`earthengine authenticate`)
and a cloud project requirement. For fully self-contained deployment, host the GEE tiles
locally on object storage (GeoTIFF COGs served via a tile proxy).

---

### 4. Multi-scale radius (1500m far-field, 100m near-field)

**Decision**: Terrain computed at 1500m; canopy + buildings at 100m.

**Alternatives considered**: Single radius for all layers (~500m).

**Reasoning**: Terrain horizon is significant at 500–2000m (a ridge 1km away at 20m above
the dish creates a ~1° horizon angle; a tree 30m away at 10m creates a ~18° angle).
Using a 100m radius for terrain misses rolling topography that would appear as a gradual
horizon rise. Using 1500m for buildings would download large tiles unnecessarily and
building effects beyond 200m are below the 25° threshold for any realistic height.

---

### 5. Fixed FOV as ±50° around North

**Decision**: Model Starlink FOV as 100° arc centered on North (azimuths 310°–50°).

**Alternatives considered**: Full 360° blockage analysis.

**Reasoning**: Starlink Gen2 and Gen3 constellations are primarily in 53° and 43° inclination
shells. For a US location at 30°N, active satellites are disproportionately to the north and
northwest. The install guide and field experience both confirm that southern obstructions matter
far less. Using full 360° would under-penalize northern obstructions by averaging them with
clear southern sky.

**What I'd revisit**: Vary FOV center by latitude. At 60°N the geometry tilts more northward;
at 20°N it becomes more equatorial.

---

## Results (NC Dataset — DATA_CHALLENGE_50)

The provided dataset contains **4,674,917 NC addresses** across all 100 NC counties
(FIPS 37xxx). A stratified sample of ~5 points per county (~500 total) was analyzed
to produce state-wide risk statistics. Full results in `outputs/batch/`.

**8-point validation sample** (geographically spread across NC):

| Location | Risk Score | Tier | Dominant | Canopy Max |
|---|---|---|---|---|
| Charlotte-Suburban | 90.2 | critical | vegetation | 27 m |
| Harnett-County | 90.0 | critical | vegetation | 25 m |
| Jones-County-Coast | 7.2 | low | clear | 19 m |
| Rowan-County | 0.0 | low | clear | 19 m |
| Lincoln-County | 0.0 | low | clear | 3 m |
| Gaston-County | 0.0 | low | clear | 20 m |
| Iredell-County | 0.0 | low | clear | 22 m |
| Davidson-County | 0.0 | low | clear | 23 m |

Critical-tier sites (Charlotte, Harnett) have dense tree canopy (25–27 m) at the dish
location. The 1m GEE canopy source is essential for these results — the prior 27m product
returned 0 m canopy at these same locations due to insufficient spatial resolution.

---

## Scalability Notes

To run on the full 1M-location dataset:

1. **Batch processing**: `batch_analyze()` in the agent runs sequentially; parallelize with
   `concurrent.futures.ProcessPoolExecutor` for ~10× throughput.
2. **Terrain caching**: py3dep tiles are WMS-based; add a GeoTIFF tile cache keyed on
   `(z, x, y)` to avoid re-fetching overlapping areas.
3. **Building tiles**: Already cached as GeoParquet per quadkey. 1M US locations cover ~120K
   unique zoom-9 quadkeys; pre-fetch tiles by state/county.
4. **Canopy tiles (GEE 1m)**: Results are cached as GeoTIFFs under `data/tiles/canopy/`.
   Cold calls ~0.2s; cached calls instant. For 1M locations most unique 100m windows will
   not overlap — expect full GEE quota consumption; use batch quotas or pre-cache by county.
5. **Parallelism bound**: 3DEP WMS rate-limits concurrent bursts (6 workers caused 100% failure
   in testing). Default is **2 workers** with exponential backoff (5→10→20 s). For higher
   throughput, implement a semaphore that limits concurrent 3DEP requests to ≤4.

Estimated throughput at full scale: ~500 locations/hour single-threaded → ~5,000/hour with
10-process pool and 3DEP semaphore.
