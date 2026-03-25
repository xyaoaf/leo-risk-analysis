# Analysis Rationale: From Install Guide to Methodology

## 1. Physical Requirements from the Starlink Install Guide

Reading the Starlink Business Install Guide surfaced four key physical constraints:

| Guide Requirement | Key Excerpt |
|---|---|
| Clear sky cone | The dish needs an unobstructed view of the sky, roughly 100° wide, directed northward for Northern Hemisphere installations. |
| Minimum elevation angle | Objects below ~25° above the horizon are outside the satellite's operational arc and don't cause problems; objects above 25° within the FOV cone do. |
| Obstruction sources | Trees, terrain, buildings, and utility poles are explicitly listed as causes of "obstruction" and "signal interruption." |
| Installation position | The dish should be mounted as high as practical (typically on roof or pole) to clear nearby obstructions. |

**What the guide cannot tell you remotely:**
- Exact dish mounting position on the property (ground vs roof vs pole)
- Presence of seasonal deciduous vs. evergreen vegetation
- Interior building features (dish inside a garage/barn)
- Temporary obstructions (construction cranes, parked vehicles)

---

## 2. From Physical Requirements to Analytical Approach

### 2a. Sky visibility as a horizon profile

The core question is: *for a dish at this location, what is the maximum angle any obstruction
reaches above the horizon in each direction?*

We model this as a **horizon profile** — a 72-point (every 5°) radial survey where each point
records the maximum elevation angle (degrees above horizontal) to any obstruction.

```
elevation_angle(d) = arctan( (obstruction_top_elevation - dish_elevation) / horizontal_distance )
```

This is physically correct and avoids any approximation: the dish either "sees" the satellite
(elevation > 25° within FOV) or it doesn't.

### 2b. Multi-scale analysis

Different obstruction types have different spatial scales:
- **Terrain**: significant at 500–2000m from the dish (rolling hills, ridgelines)
- **Tree canopy**: significant at 20–200m (trees right next to the dish)
- **Buildings**: significant at 5–100m (adjacent structures)

We use two analysis radii:
- **Far-field** (1500m, ~10m/px DEM): captures terrain horizon
- **Near-field** (100m, ~1m/px DEM + canopy + buildings): captures local obstructions

### 2c. Three-layer horizon decomposition

By computing horizon profiles at three stages (terrain only → terrain+canopy →
terrain+canopy+buildings), we can attribute blockage to its source:

```
canopy_contribution   = blocked_frac(terrain+canopy)  - blocked_frac(terrain)
building_contribution = blocked_frac(all layers)       - blocked_frac(terrain+canopy)
```

This attribution directly answers the question: *is this a vegetation problem or a building problem?*

### 2d. Why this approach over alternatives

| Alternative | Why not used |
|---|---|
| ML model trained on outcomes | Requires labelled outcome data (signal quality per location) which isn't publicly available |
| Viewshed analysis (GIS standard) | Equivalent to our ray-casting but typically slower and overkill for 72 azimuths |
| Satellite imagery classification | Cannot measure height accurately without LiDAR; canopy height data is already available from Meta |
| LLM scoring per location | Inconsistent, slow, costly at 1M locations, not reproducible |

---

## 3. Defining "At-Risk"

A location is **at-risk** if any obstruction within the Starlink FOV exceeds 25° elevation.

**FOV definition for CONUS**: ±50° around North (azimuths 310°–360° and 0°–50°).
This is derived from the geometry of Starlink's polar orbit inclination (~53°) which places
active satellites predominantly to the north for US latitudes.

**Risk score (0–100)**:
```
risk = FOV_blockage_score   (0–50)   # fraction of FOV azimuths blocked × 50
     + angle_severity_score (0–30)   # (max_angle – 25°) / 65° × 30
     + permanence_penalty   (0–20)   # buildings/terrain × 1.0; vegetation × 0.6
```

**Tiers**:
- **Low (0–19)**: Clear sky, no concern.
- **Moderate (20–44)**: Marginal obstruction, likely seasonal. Monitor.
- **High (45–69)**: Meaningful FOV blockage; service will be impaired during satellite passes.
- **Critical (70–100)**: Dish cannot reliably see enough satellites; location is essentially unserviceable.

**Explaining to a non-technical state broadband officer**:
> "Imagine standing at this address and looking toward where Starlink's satellites orbit — mostly
> to the north and northeast. If trees or buildings block more than about 10% of that view above
> a steep enough angle, the dish struggles to stay connected. Our score tells you what fraction
> of that critical sky window is blocked, and what's doing the blocking."

---

## 4. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Canopy resolution ~27m/px | Individual trees not resolved; cluster-level accuracy only | Use 1m Meta tiles locally (opt-in flag) |
| No leaf-on/leaf-off distinction | Deciduous trees may be over- or under-counted | Apply permanence factor × 0.6 for vegetation |
| Buildings assumed mountable | Dish may not actually be installable on all detected rooftops | `blocked` flag excludes buildings with tree cover |
| 3DEP USGS coverage | Only CONUS; international locations fall back to 30m SRTM | Acceptable for the challenge's US-focused dataset |
| Dish height fixed at 3m | Actual mounting varies; rooftop mounts can be 5–10m | User-configurable `dish_height_m` parameter |
| No real-time satellite visibility | Analysis assumes full orbital coverage; outages during passes not modeled | Out of scope for static risk assessment |
| No atmospheric effects | Rain fade, multipath reflections not modeled | Acceptable at this scale; physical obstruction dominates |
