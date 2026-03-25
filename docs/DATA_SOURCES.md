# Data Sources & Quality

## Summary Table

| Dataset | Source | Obstruction Factor | Resolution | Coverage | License |
|---|---|---|---|---|---|
| USGS 3DEP DEM | USGS / py3dep | Terrain horizon | 1–10 m/px | CONUS | Public domain |
| Meta Global Canopy Height | Meta AI / AWS S3 | Tree canopy | ~27 m/px (1 m available) | Global | CC-BY 4.0 |
| Microsoft ML Building Footprints | Microsoft / Bing | Structures | Vector polygons w/ ML height | Global | ODbL |
| Google Open Buildings v3 | Google / GCS | Structures (fallback) | Vector polygons | Global (uneven) | CC-BY 4.0 |

---

## 1. USGS 3DEP Digital Elevation Model

**Why chosen**: The USGS 3D Elevation Program is the authoritative bare-earth DEM for the US.
It directly captures the terrain horizon that blocks low-elevation satellite links. The 1m/3m
resolution products capture street-scale topography.

**Access method**: `py3dep` library → USGS WMS endpoint → returned as xarray DataArray.

**Quality issues encountered**:
- **Service timeouts**: The 3DEP WMS endpoint is flaky with transient 502/503 errors,
  especially for 1m resolution requests. Mitigated with 2-attempt retry per resolution level.
- **Fallback ladder**: If 1m fails → 3m → 10m → 30m. For a 100m near-field radius at 10m/px
  this gives only a ~10×10 pixel grid, which is too coarse. In practice 1m succeeds ~85% of
  the time for Austin TX.
- **Coverage**: CONUS only. Hawaii and Alaska have partial coverage.
- **Bare-earth vs. surface**: 3DEP is a bare-earth DEM (vegetation and buildings removed by
  LiDAR processing), which is exactly what we need as the terrain baseline before adding
  canopy and building heights.

**Obstruction factor linked**: Terrain horizon — rolling hills, ridgelines, bluffs within 1–2km
that raise the effective horizon angle above the satellite arc.

---

## 2. Meta Global Canopy Height (GCH)

**Why chosen**: This is currently the highest-resolution global canopy height dataset available
publicly. It was produced by Meta AI Research using airborne LiDAR (GEDI + ALS fusion).
The challenge explicitly asks for vegetation as an obstruction factor, and this dataset
provides actual height in meters (not just presence/absence).

**Two product versions**:

| Version | Path on S3 | Resolution | CRS | Format |
|---|---|---|---|---|
| 1m tiles | `forests/v1/alsgedi_global_v6_float/chm/{quadkey9}.tif` | ~1.2 m/px | EPSG:3857 | uint8, value=meters |
| 27m tiles | `forests/v1/alsgedi_global_v6_float_epsg4326_v3_10deg/meta_chm_lat={top}_lon={left}_median.tif` | ~27 m/px | EPSG:4326 | uint16, value=cm, nodata=65535 |

**Access method**: AWS S3 anonymous access via `rasterio.Env(AWS_NO_SIGN_REQUEST="YES")`.
Windowed read (COG) used to stream only the required bbox without downloading the full tile.

**Quality issues encountered**:
- **Tile naming (critical bug)**: The 10° tiles are named by their **top-left corner** (not
  bottom-left). For Austin at lat=30.28°: `lat_tl = ceil(30.28/10)*10 = 40°`, not 30°.
  Using `floor` instead of `ceil` for latitude caused empty windows because the bbox fell
  outside the tile bounds.
- **1m tiles NOT Cloud-Optimised**: Despite `.tif` extension, the 1m tiles use `blockysize=1`
  (strip layout). A windowed read for a 200m window requires streaming ~12MB of strips (~40s).
  Disabled by default; use `fetch_canopy._use_1m = True` to enable with local caching.
- **27m tiles in centimetres**: The uint16 values are centimetres, not metres. Must divide by
  100 after masking nodata (65535).
- **nodata handling**: Pixels with value 65535 (nodata) set to 0.0 (bare ground), which is
  slightly conservative — it treats unobserved forest as bare ground.

**Obstruction factor linked**: Tree canopy — tall trees (>5m) within 100m of the dish that
block the northern sky arc.

---

## 3. Microsoft Global ML Building Footprints

**Why chosen**: Microsoft provides ML-predicted building heights alongside footprints — the only
global building dataset with height data. Height is critical: a 3m shed is very different from
a 15m apartment block in terms of obstruction angle.

**Access method**:
1. Download index CSV from `https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv`
2. Look up the Bing quadkey (zoom-9) for the target lat/lon using `mercantile`
3. Download GeoJSONL.gz from the tile URL; parse line by line
4. Cache result as GeoParquet in `data/buildings/{quadkey}.parquet`

**Schema**: `geometry` (WKT), `height` (meters, ML-predicted), `confidence` (0–1)

**Quality issues encountered**:
- **Height below minimum**: Some ML-predicted heights are <2.5m (likely noise). These are
  replaced with an area-based heuristic: `floors = max(1, min(area_m²/150, 10))`, height = floors × 3.5m.
- **Very large tiles**: Zoom-9 quadkeys cover ~78km × 78km. Tiles in dense urban areas can
  have hundreds of thousands of buildings. Mitigated by bbox-filtering after download.
- **Cache invalidation**: The dataset is updated periodically. Cached tiles are not invalidated
  automatically; re-running on fresh data requires deleting `data/buildings/*.parquet`.

**Obstruction factor linked**: Buildings — adjacent structures that create hard geometric
obstructions (walls, rooflines) blocking specific azimuth sectors.

---

## 4. Google Open Buildings v3 (Fallback)

**Why chosen / why fallback**: Google Open Buildings provides high-quality footprints for much
of the developing world (Sub-Saharan Africa, South/Southeast Asia, Latin America) where
Microsoft's coverage is weaker.

**Quality issues encountered**:
- **Sparse US coverage**: The S2 level-4 tile `865` nominally covers all of central Texas
  (lat 28.26–34.40, lon -100.62 to -95.06), but the actual file only contains buildings in
  the Del Rio, TX area (lat 28–29, lon -100.6 to -100.3). Austin metro (~30.3°N, -97.7°W)
  returns 0 buildings.
- **No height data**: Google Open Buildings provides footprint area but no height. We estimate
  height from area using a heuristic (`<20m²→3m, <100m²→4.5m, …`), which is very imprecise for
  multi-story buildings.
- **Decision**: Microsoft first for CONUS; Google fallback for non-CONUS only.

---

## 5. What Cannot Be Modeled with Public Data

| Gap | Why Unmodelable |
|---|---|
| Exact dish mount point on property | Parcel data doesn't specify where a dish would be placed |
| Trees obscuring a specific window/door | Object-level placement requires on-site survey |
| Temporary obstructions (cranes, trucks) | No real-time or historical snapshot in public data |
| Interior obstructions (dish inside barn) | Not visible from any remote sensing |
| Atmospheric signal degradation (rain fade) | Would need real-time weather + link budget model |
| Exact satellite visibility windows | Full TLE orbit propagation needed (out of scope here) |
| Dish orientation/tilt constraints | Install guide constraint; not derivable remotely |
