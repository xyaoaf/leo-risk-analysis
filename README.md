# LEO Satellite Site Suitability Engine

A geospatial analysis engine that assesses whether ~4.67M candidate Starlink installation sites across North Carolina can receive usable service, by analyzing terrain, vegetation, and building obstructions.

Developed for the **ReadyNet Data Challenge**.

**Full report:** [LEO_Satellite_Site_Suitability_Final_Report.docx](LEO_Satellite_Site_Suitability_Final_Report.docx)

**Video demo:** [Google Drive](https://drive.google.com/file/d/1RIcfK-dPumTxaiKv1zxje2zkW-W9n8xF/view?usp=drive_link)

**Large data files (not in repo):** [Google Drive](https://drive.google.com/drive/folders/1anQTlt1YvmdY1P4gP45NXJPhSc728cNk?usp=sharing)

---

## Quick Start

### Prerequisites

```bash
conda activate cs378
# For agent mode only:
export ANTHROPIC_API_KEY=sk-ant-...
```

Place `DATA_CHALLENGE_50.csv` (from Google Drive) in the `leo_risk_analysis/` root directory.

### Layers 1, 2, 5 — Run from command line

```bash
# Layer 1 — Single-point diagnostic (5-panel PNG + JSON report)
python run_analysis.py 35.061 -80.666 --id charlotte

# Layer 2 — Batch processing (~594 stratified NC points across all 100 counties)
python batch_nc_analysis.py
python batch_nc_analysis.py --max-points 10  # quick test (10 points)

# Layer 5 — Zonal aggregation (county + census block group summaries)
python zonal_summary.py
```

### Layers 3, 4 — Open the interactive web app

```bash
uvicorn app:app --port 8000
# → open http://localhost:8000
```

- **Layer 3 — Interactive Map:** Multi-scale visualization with county choropleth (zoom < 9), block group choropleth (zoom 9–12), and individual address dots (zoom ≥ 12). Click any point for on-demand analysis (~30s) with a 5-panel diagnostic drawer.
- **Layer 4 — AOI Analysis:** Draw a rectangle on the map to select an area of interest. The system filters the 4.67M dataset, applies spatially uniform sampling (up to 100 points), and runs batch analysis with live progress tracking.

### Claude Agent (natural language interface)

```bash
python src/agent.py "Analyze Starlink risk at 35.06, -80.67"
python src/agent.py "Find a better location near 35.78, -82.56"
```

---

## How It Works

The system answers: **"Given a candidate location, is it suitable for Starlink dish installation, and if not, where nearby is better?"**

1. **Fetch** terrain (USGS 3DEP, 1–30m), canopy (Meta 1m via GEE), and buildings (Microsoft ML Footprints)
2. **Composite** into a unified obstruction surface
3. **Ray-cast** 72 azimuth directions across Starlink's northward FOV (±50° around North)
4. **Evaluate** 6 hard feasibility constraints (slope, canopy, vegetation/building/terrain blockage)
5. **Score** a deterministic 0–100 risk value (FOV blockage + angle severity + permanence penalty)
6. **Search** nearby (~50m) for better alternatives if the location fails

All risk computation is deterministic Python — no LLM in the scoring loop. Claude is used only for agent orchestration and natural-language reporting.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline diagram.

---

## Five Output Layers

| Layer | Workflow | Entry Point | Status |
|---|----------|-------------|--------|
| 1 | **Single-point diagnostic** — full 5-panel PNG + JSON per location | `python run_analysis.py LAT LON` | ✅ |
| 2 | **Batch processing** — stratified sampling, concurrent workers, auto-skip analyzed | `python batch_nc_analysis.py` | ✅ |
| 3 | **Interactive web map** — multi-scale choropleth, click-to-analyze, diagnostic drawer | `uvicorn app:app --port 8000` | ✅ |
| 4 | **AOI / region analysis** — draw rectangle, spatially uniform sampling, batch execution | Draw on map or `python aoi_screen.py --bbox ...` | ✅ |
| 5 | **Zonal aggregation** — county + block group pass rates, choropleth maps | `python zonal_summary.py` or button in web app | ✅ |

---

## Repository Structure

```
leo_risk_analysis/
├── app.py                     # [Layer 3/4] FastAPI web server (Leaflet.js map + API)
├── run_analysis.py            # [Layer 1] Single-point CLI diagnostic
├── main.py                    # [Layer 1] Batch demo runner (5 test points)
├── batch_nc_analysis.py       # [Layer 2] Stratified batch runner + county charts
├── aoi_screen.py              # [Layer 4] AOI bbox/polygon point filter
├── zonal_summary.py           # [Layer 5] Block-group / county aggregation
├── templates/map.html         # [Layer 3/4] Leaflet.js frontend (multi-scale, on-demand)
├── src/
│   ├── agent.py               # Claude-orchestrated agent (tool_use loop)
│   ├── feasibility.py         # analyze_location() — orchestrator + 6 constraints
│   ├── scoring.py             # Deterministic 0–100 risk score (3 components)
│   └── tools/
│       ├── terrain.py         # USGS 3DEP DEM (1m→3m→10m→30m fallback)
│       ├── canopy.py          # Meta Trees 1m (GEE) + 27m S3 fallback
│       ├── buildings.py       # Microsoft ML Footprints (Google fallback)
│       ├── surface.py         # Obstruction surface compositor
│       └── horizon.py         # Vectorized NumPy ray-casting (72 azimuths)
├── docs/
│   ├── ARCHITECTURE.md        # Mermaid diagram + component table
│   ├── ANALYSIS_RATIONALE.md  # Install guide → methodology reasoning
│   ├── DATA_SOURCES.md        # Dataset choices, quality issues, limitations
│   ├── DESIGN.md              # Full algorithm specification
│   ├── WORKFLOWS.md           # All five layers (commands + I/O specs)
│   └── CHALLENGE.md           # Original challenge specification
├── data/                      # Cached tiles (gitignored, auto-downloaded at runtime)
├── outputs/
│   ├── reports/               # [Layer 1] Per-point JSON + 5-panel PNG diagnostics
│   ├── batch/                 # [Layer 2] batch_results.csv + county charts
│   ├── zonal/                 # [Layer 5] County + block group summaries + choropleth
│   └── challenge50/           # Challenge-50 sample results
├── LEO_Satellite_Site_Suitability_Final_Report.docx
├── AI_TOOLS.md
└── requirements.txt
```

---

## Data Files (Google Drive)

Large files not included in the repository are available at:
https://drive.google.com/drive/folders/1anQTlt1YvmdY1P4gP45NXJPhSc728cNk?usp=sharing

| File | Size | Description |
|---|---|---|
| `DATA_CHALLENGE_50.csv` | ~600 MB | 4.67M NC candidate locations (required for Layers 2–5) |
| `outputs/` | ~200 MB | Pre-computed analysis results (JSON + PNG per point) |
| `data/boundaries/` | ~12 MB | County + block group GeoJSON boundaries (NC) |
| Demo video | ~64 MB | [Screen recording of full workflow](https://drive.google.com/file/d/1RIcfK-dPumTxaiKv1zxje2zkW-W9n8xF/view?usp=drive_link) |

---

## Key Design Decisions

### 1. No LLM in the scoring loop

All risk computation is deterministic Python (horizon ray-casting + scoring formula). Claude is used only for workflow orchestration and natural language explanation. At 1M locations, per-location LLM calls would cost ~$500–2,000 and take hours. A deterministic formula runs in seconds, is reproducible, and explainable.

### 2. Microsoft Building Footprints over Google

Google Open Buildings has near-zero coverage for US urban areas. The Austin TX S2 tile returned 0 buildings. Microsoft ML Footprints provides full CONUS coverage with ML-predicted heights. Swapped priority after empirical testing.

### 3. GEE Meta Trees 1m as primary canopy

The 27m S3 product has no genuine sub-30m spatial information (GEDI footprints averaged to 27m). The 1m GEE product (Tolan et al. 2024, *Remote Sensing of Environment*) detects individual tree crowns via `computePixels()` in ~0.2s. Critical for detecting canopy directly at the dish location.

### 4. Multi-scale radius (1,500m far-field + 100m near-field)

Terrain horizon is significant at 500–2,000m (a ridge 1km away creates a ~1° horizon rise). Canopy and building effects fade beyond 100m. Single-radius approaches either miss terrain or download unnecessary building data.

### 5. ±50° FOV centered on North

Starlink Gen2/Gen3 constellations at 53°/43° inclination concentrate satellites northward for US locations. Southern obstructions matter far less. Using full 360° would under-penalize northern blockage.

---

## Results Summary (NC Dataset)

- **850+ locations analyzed** across all 100 NC counties
- **Bimodal distribution**: locations are either heavily wooded (critical-tier) or open (low-tier)
- **Western NC** (Appalachian mountains): 50–70% infeasibility rates, terrain-dominated
- **Central NC** (Piedmont): 10–30% infeasibility, mixed vegetation and building obstructions
- **Eastern NC** (Coastal Plain): <10% infeasibility, flat terrain with minimal obstructions

### Scalability

| Configuration | Throughput |
|---|---|
| Current (single-threaded) | ~500 loc/hr |
| ProcessPoolExecutor(10) | ~5,000 loc/hr |
| Production (tile caching + horizontal) | ~50,000 loc/hr |
