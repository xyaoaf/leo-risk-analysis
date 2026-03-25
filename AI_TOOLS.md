# AI Tool Disclosure

## Tools Used

| Tool | Purpose |
|------|---------|
| **Claude Code (claude-sonnet-4-6)** | Primary development assistant: implemented all Python modules, debugged data source issues, wrote documentation |
| **Claude claude-opus-4-6 (via API)** | Agent runtime: orchestrates the geospatial pipeline in `src/agent.py` |

No other AI coding tools (GitHub Copilot, Cursor, v0, Codex, etc.) were used.

---

## Claude Code Usage Details

Claude Code was used throughout the project as an interactive pair programmer. Specific areas:

- **`src/tools/terrain.py`**: Scaffolded initial implementation; Claude suggested the resolution
  fallback ladder (1→3→10→30m) and the MAX_ATTEMPTS retry pattern for 3DEP timeouts.

- **`src/tools/canopy.py`**: Claude initially structured the `_stream_27m_tile()` function.
  Tile naming bug (ceil vs. floor for latitude) was identified through empirical testing;
  Claude then explained the off-by-one and proposed the fix.

- **`src/tools/buildings.py`**: Claude drafted the source-priority logic. The decision to swap
  priority (Microsoft first, not Google first) came after testing revealed Google returns
  0 buildings for Austin — a real-world discovery Claude could not have predicted.

- **`src/tools/surface.py`**: Claude implemented the `_classify_buildings()` per-footprint
  rasterize loop and the `rasterio.features.rasterize` `merge_alg` pattern. The sort-ascending
  fix (to avoid the removed lambda API in newer rasterio) came from a runtime error, not Claude.

- **`src/agent.py`** and **`src/scoring.py`**: Claude wrote the initial drafts of both;
  the risk score component weights (50/30/20) were tuned manually to produce intuitive tiers
  for the Austin test results.

- **Documentation** (`ARCHITECTURE.md`, `ANALYSIS_RATIONALE.md`, `DATA_SOURCES.md`, `README.md`):
  All drafts generated with Claude Code, then reviewed and tightened for accuracy.

---

## Cases Where I Diverged from AI-Generated Output

### 1. Building data source priority (Google → Microsoft)

Claude's initial implementation followed the challenge's implicit suggestion of using Google
Open Buildings as the primary source with Microsoft as fallback. After running the pipeline on
Austin test points, I discovered Google's S2 tile for central Texas contains only rural border
data and returns 0 buildings for the urban test points.

**Divergence**: Swapped the priority to Microsoft-first for CONUS. Claude did not anticipate
this because the dataset coverage gap is a real-world empirical issue, not something inferrable
from documentation.

### 2. Risk score permanence weights

Claude's initial score formula used equal weights for all obstruction types. I changed the
`vegetation` permanence factor from 1.0 to 0.6 to reflect that deciduous trees are seasonal
obstructions — this is physically meaningful (a house in a leafy suburb is different in summer
vs. winter) and produces more nuanced tier assignments.

### 3. Canopy tile resolution choice

Claude initially proposed using the 1m resolution tiles as the default (following the challenge's
"try the 1m resolution tree map" suggestion). After testing showed ~40s per window due to
strip layout, I made the 1m tiles opt-in (`fetch_canopy._use_1m = True`) and defaulted to
the 27m COG tiles. This was a pure performance trade-off not visible from the dataset spec alone.
