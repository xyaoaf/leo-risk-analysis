# LEO Satellite Coverage Risk Analysis

An agent-driven geospatial pipeline that identifies at-risk locations where environmental
obstructions (terrain, trees, buildings) are likely to degrade Starlink service quality —
even at addresses providers have listed as "served."

---

## Quick Start

```bash
# Clone + environment setup
conda activate cs378

# Run the 5-point Austin TX demo
conda run -n cs378 python main.py

# Run the Claude agent interactively
export ANTHROPIC_API_KEY=sk-ant-...
conda run -n cs378 python src/agent.py "Analyze Starlink risk at 30.2867, -97.7113"
```

Outputs land in `outputs/`:
- `point_{label}.png` — 3-panel visualization per location
- `point_{label}.json` — risk score + obstruction breakdown JSON
- `summary.png` — multi-point comparison chart

---

## Repository Structure

```
leo_risk_analysis/
├── main.py                    # Batch pipeline runner (5 Austin TX test points)
├── src/
│   ├── agent.py               # Claude-orchestrated agent (tool_use loop)
│   ├── scoring.py             # Deterministic 0–100 risk score
│   └── tools/
│       ├── terrain.py         # USGS 3DEP DEM fetcher
│       ├── canopy.py          # Meta Global Canopy Height fetcher
│       ├── buildings.py       # Microsoft / Google building footprints
│       ├── surface.py         # Multi-layer obstruction surface builder
│       └── horizon.py         # Radial ray-casting + obstruction classifier
├── docs/
│   ├── ARCHITECTURE.md        # Mermaid diagram + component table
│   ├── ANALYSIS_RATIONALE.md  # From install guide to methodology
│   ├── DATA_SOURCES.md        # Dataset choices + quality issues
│   └── CHALLENGE.md           # Original challenge specification
├── data/
│   └── buildings/             # Cached building tile GeoParquets
└── outputs/                   # Generated figures and JSON results
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

### 3. 27m canopy resolution (not 1m)

**Decision**: Use the 27m Meta GCH tiles by default; 1m tiles are opt-in.

**Alternatives considered**: 1m tiles as default for maximum accuracy.

**Reasoning**: The 1m Meta tiles use strip layout (blockysize=1), not COG. Even a 200m window
requires downloading ~12MB of interleaved strips, taking ~40s per location. At 1M locations
this would add ~11,000 compute-hours.  The 27m product is a true COG; a 200m window streams
in <1s.

**What I'd revisit**: Locally cache and index the 1m tiles by quadkey for near-field analysis.
At 100m radius, the 27m data resolves to only ~4×4 pixels — barely a texture.

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

## Results (Austin TX Test Points)

| Location | Risk Score | Tier | Dominant | Max Angle |
|---|---|---|---|---|
| Austin-Suburban (30.2867, -97.7113) | 0.0 | low | clear | 5.7° |
| Austin-Urban-Core (30.2943, -97.7000) | 0.0 | low | clear | 19.3° |
| Austin-West (30.2798, -97.7557) | 0.0 | low | clear | 4.1° |
| Austin-Far-West (30.3042, -97.7981) | 0.0 | low | clear | 14.5° |
| North-Austin (30.3222, -97.7747) | 12.9 | low | vegetation | 37.7° |

North-Austin is the only point that approaches a meaningful obstruction: the canopy there
reaches 37.7° max elevation angle, classified as vegetation-dominant. The 12.9/100 risk score
reflects that this affects only a small fraction of the Starlink FOV and is seasonal (deciduous
canopy, permanence factor 0.6).

The "clear" results for the other four points are geographically accurate — central Austin is
relatively flat and the test radii (100m near-field, 1500m far-field) capture the key obstructions.

---

## Scalability Notes

To run on the full 1M-location dataset:

1. **Batch processing**: `batch_analyze()` in the agent runs sequentially; parallelize with
   `concurrent.futures.ProcessPoolExecutor` for ~10× throughput.
2. **Terrain caching**: py3dep tiles are WMS-based; add a GeoTIFF tile cache keyed on
   `(z, x, y)` to avoid re-fetching overlapping areas.
3. **Building tiles**: Already cached as GeoParquet per quadkey. 1M US locations cover ~120K
   unique zoom-9 quadkeys; pre-fetch tiles by state/county.
4. **Canopy tiles**: Stream-read is fast for 27m product; no caching needed unless running
   many points in the same 10° tile (batch the reads).
5. **Parallelism bound**: 3DEP WMS has a rate limit (~10 concurrent requests). Use a semaphore.

Estimated throughput at full scale: ~500 locations/hour single-threaded → ~5,000/hour with
10-process pool and 3DEP semaphore.
