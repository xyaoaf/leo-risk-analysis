# LEO Satellite Site Suitability Engine

A geospatial analysis engine that assesses whether candidate locations are suitable for
Starlink satellite dish installation by analyzing terrain, vegetation, and building obstructions.
Developed for the ReadyNet Data Challenge.

**Full report:** [LEO_Satellite_Site_Suitability_Final_Report.docx](LEO_Satellite_Site_Suitability_Final_Report.docx)

---

## Quick Start

```bash
conda activate cs378

# Interactive web app (multi-scale map + on-demand analysis)
uvicorn app:app --port 8000
# → open http://localhost:8000

# Single-point CLI diagnostic (5-panel PNG + JSON)
python run_analysis.py 35.061 -80.666 --id charlotte

# Batch analysis (~594 stratified NC points)
python batch_nc_analysis.py
python batch_nc_analysis.py --max-points 10  # dry-run

# Claude agent (natural language queries)
export ANTHROPIC_API_KEY=sk-ant-...
python src/agent.py "Analyze Starlink risk at 35.06, -80.67"
```

---

## How It Works

The system answers: **"Given a candidate location, is it suitable for Starlink dish installation, and if not, where nearby is better?"**

1. **Fetch** terrain (USGS 3DEP, 1–30m), canopy (Meta 1m via GEE), and buildings (Microsoft ML Footprints)
2. **Composite** into a unified obstruction surface
3. **Ray-cast** 72 azimuth directions across Starlink's northward FOV
4. **Score** a deterministic 0–100 risk value (FOV blockage + angle severity + permanence)
5. **Evaluate** 6 hard feasibility constraints (slope, canopy, vegetation/building/terrain blockage)
6. **Search** nearby (~50m) for better alternatives if the location fails

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline diagram.

---

## Output Workflows

| # | Workflow | Entry Point |
|---|----------|-------------|
| 1 | Single-point diagnostic | `python run_analysis.py LAT LON` |
| 2 | Batch processing | `python batch_nc_analysis.py` |
| 3 | Interactive web map | `uvicorn app:app --port 8000` |
| 4 | AOI / region analysis | `python aoi_screen.py --bbox ...` or draw rectangle on map |
| 5 | Zonal aggregation | `python zonal_summary.py` or "Update Zonal Statistics" button |

---

## Repository Structure

```
leo_risk_analysis/
├── app.py                     # FastAPI web server (Leaflet.js map + API)
├── main.py                    # Batch demo runner
├── run_analysis.py            # Single-point CLI
├── batch_nc_analysis.py       # Stratified batch runner + county charts
├── aoi_screen.py              # AOI bbox/polygon point filter
├── zonal_summary.py           # Block-group / county aggregation
├── templates/map.html         # Leaflet.js frontend (multi-scale, on-demand)
├── src/
│   ├── agent.py               # Claude-orchestrated agent (tool_use loop)
│   ├── feasibility.py         # analyze_location() — orchestrator + constraints
│   ├── scoring.py             # Deterministic 0–100 risk score
│   └── tools/
│       ├── terrain.py         # USGS 3DEP DEM (1m/3m/10m/30m fallback)
│       ├── canopy.py          # Meta Trees 1m (GEE) + 27m S3 fallback
│       ├── buildings.py       # Microsoft ML Footprints (Google fallback)
│       ├── surface.py         # Obstruction surface compositor
│       └── horizon.py         # Vectorized ray-casting + classifier
├── docs/
│   ├── ARCHITECTURE.md        # Pipeline diagram + component table
│   ├── DATA_SOURCES.md        # Dataset choices + quality issues
│   ├── DESIGN.md              # Full algorithm specification
│   ├── ANALYSIS_RATIONALE.md  # From install guide to methodology
│   ├── WORKFLOWS.md           # All five workflow layers
│   └── CHALLENGE.md           # Original challenge specification
├── data/                      # Cached tiles (gitignored, auto-downloaded)
├── outputs/                   # Analysis results (key files tracked)
│   ├── zonal/                 # County + block group summaries
│   ├── batch/                 # Batch results + charts
│   └── reports/               # Per-point JSON + PNG diagnostics
└── LEO_Satellite_Site_Suitability_Final_Report.docx
```

---

## Data Files (Google Drive)

Large files not included in the repository:

| File | Size | Description |
|---|---|---|
| `DATA_CHALLENGE_50.csv` | ~600 MB | 4.67M NC candidate locations |
| `outputs/` | ~200 MB | Pre-computed analysis results (JSON + PNG) |
| `data/boundaries/` | ~12 MB | County + block group GeoJSON (NC) |
| Demo video | ~64 MB | Screen recording of full workflow |

---

## Key Design Decisions

1. **No LLM in the scoring loop** — All risk computation is deterministic Python. Claude is used only for orchestration and explanation. At 1M locations, per-location LLM calls would cost ~$500–2000.

2. **Microsoft Buildings over Google** — Google Open Buildings has near-zero US urban coverage. Microsoft ML Footprints provides full CONUS coverage with ML-predicted heights.

3. **GEE Meta Trees 1m as primary canopy** — The 27m S3 product has no genuine sub-30m spatial information. The 1m GEE product (Tolan et al. 2024) detects individual tree crowns in ~0.2s per call.

4. **Multi-scale radius** — Terrain at 1,500m (far-field ridges), canopy + buildings at 100m (near-field obstructions). Effects at different scales require different analysis radii.

5. **±50° FOV around North** — Starlink Gen2/Gen3 satellites concentrate at 53°/43° inclination, making northern sky visibility critical for US locations.

---

## Results Summary (NC Dataset)

- **850+ locations analyzed** across all 100 NC counties
- **Bimodal distribution**: locations are either heavily wooded (critical-tier) or open (low-tier)
- **Western NC** (Appalachian mountains): 50–70% infeasibility, terrain-dominated
- **Central NC** (Piedmont): 10–30% infeasibility, mixed vegetation/building
- **Eastern NC** (Coastal Plain): <10% infeasibility, flat terrain with minimal obstructions
- **Throughput**: ~500 loc/hr single-threaded → ~5,000/hr with 10-process pool
