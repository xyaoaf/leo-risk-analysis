# Data Sources & Quality

## Summary Table

| Dataset | Source | Obstruction Factor | Resolution | Coverage | License |
|---|---|---|---|---|---|
| USGS 3DEP DEM | USGS / py3dep | Terrain horizon | 1–10 m/px | CONUS | Public domain |
| Meta Trees 1m (GEE) | Meta / Google Earth Engine · Tolan et al. 2024 | Tree canopy (primary) | **1 m/px** | Global | CC-BY 4.0 · GEE commercial license for production |
| Meta ALSGEDI 27m | Meta AI / AWS S3 | Tree canopy (fallback) | ~27 m/px | Global | CC-BY 4.0 |
| Microsoft ML Building Footprints | Microsoft / Bing | Structures | Vector polygons w/ ML height | Global | ODbL |
| Google Open Buildings v3 | Google / GCS | Structures (fallback) | Vector polygons | Global (uneven) | CC-BY 4.0 |

---

## 1. USGS 3DEP Digital Elevation Model

**Why chosen**: The USGS 3D Elevation Program is the authoritative bare-earth DEM for the US.
It directly captures the terrain horizon that blocks low-elevation satellite links. The 1m/3m
resolution products capture street-scale topography.

**Access method**: `py3dep` library → USGS WMS endpoint → returned as xarray DataArray.

**Quality issues encountered**:
- **Service timeouts / rate limiting**: The 3DEP WMS endpoint returns transient 502/503 errors
  and rejects concurrent bursts. Mitigated with **4-attempt exponential backoff per resolution**
  (5 s → 10 s → 20 s between retries). Batch runs should use ≤ 2 parallel workers to stay
  below the endpoint's implicit concurrency limit (6 workers caused 100% failure rate).
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

## 2. Meta Trees 1m — Google Earth Engine (Primary Canopy Source)

**Why chosen**: Tolan et al. 2024 (*Remote Sensing of Environment*, 300, p.113888) produced
the first genuinely 1m-resolution global canopy height map by applying DiNOv2 to Maxar
commercial satellite imagery. This is fundamentally different from GEDI/ALS-derived products
(which are upsampled LiDAR footprints at 25–30m effective resolution). At 1m/px we can
detect individual tree crowns and differentiate canopy at the dish placement level.

**GEE asset**: `projects/sat-io/open-datasets/facebook/meta-canopy-height`
**Band**: `cover_code` (uint8, values = canopy height in metres)
**Access method**: `ee.data.computePixels()` with NPY format and explicit affineTransform
grid specifying the exact 200×200 pixel window at ~1m/px in EPSG:4326.

**GEE licensing note**: Current use is a personal non-profit GEE account (prototype only).
Production deployment serving commercial locations requires a Google Earth Engine commercial
license. Set `GEE_PROJECT` env var or pass `project=` to `fetch_canopy()` to override.

**Local tile cache**: Results cached as LZW-compressed GeoTIFF under
`data/tiles/canopy/gee1m_{lat}_{lon}_{radius}.tif`. Typical cold-call latency ~0.2s;
cached calls are instant.

**Quality issues encountered**:
- **GEE asset is ImageCollection not Image**: Must use `.filterBounds(region).mosaic()` to
  get a single image for `computePixels`. Direct `ee.Image()` fails.
- **computePixels returns structured numpy array**: Output dtype is `[('cover_code', 'u1')]`.
  Extract the field: `structured['cover_code'].astype(np.float32)`.
- **Resolution at high latitude**: The 1m grid in EPSG:4326 degrees uses
  `deg_per_m_lon = 1/(111320 × cos(lat))`. At 35°N this is ~0.0000109° — correctly computed.

**Obstruction factor linked**: Tree canopy — individual-tree-level height within 100m.

---

## 3. Meta ALSGEDI 27m — AWS S3 (Canopy Fallback)

**Why fallback**: Despite the name "1m Global", the ALSGEDI S3 product is derived from
GEDI satellite waveform LiDAR (~25m footprint) averaged onto a 27m grid. It has no genuine
1m spatial information. Used only when GEE is unavailable (no credentials, quota exceeded).

**Two product versions on S3**:

| Version | Path on S3 | Resolution | CRS | Format |
|---|---|---|---|---|
| 27m tiles | `dataforgood-fb-data/forests/v1/alsgedi_global_v6_float_epsg4326_v3_10deg/meta_chm_lat={top}_lon={left}_median.tif` | ~27 m/px | EPSG:4326 | uint16, value=cm, nodata=65535 |

**Access method**: AWS S3 anonymous access via `rasterio.Env(AWS_NO_SIGN_REQUEST="YES")`.
Windowed read (COG) to stream only the required bbox.

**Quality issues encountered**:
- **Tile naming**: The 10° tiles are named by their **top-left corner** (not bottom-left).
  For lat=35.06°: `lat_tl = ceil(35.06/10)*10 = 40°`.
- **Values in centimetres**: uint16 values are centimetres → divide by 100.
- **nodata**: 65535 → set to 0.0 (bare ground).

**Obstruction factor linked**: Tree canopy (coarse fallback; ≥5m threshold used instead of ≥1m
since 27m pixel covers ~700 m² — a non-zero value doesn't mean the specific dish point has a tree).

---

## 4. Microsoft Global ML Building Footprints

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

## 5. Google Open Buildings v3 (Fallback)

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

## 6. What Cannot Be Modeled with Public Data

| Gap | Why Unmodelable |
|---|---|
| Exact dish mount point on property | Parcel data doesn't specify where a dish would be placed |
| Trees obscuring a specific window/door | Object-level placement requires on-site survey |
| Temporary obstructions (cranes, trucks) | No real-time or historical snapshot in public data |
| Interior obstructions (dish inside barn) | Not visible from any remote sensing |
| Atmospheric signal degradation (rain fade) | Would need real-time weather + link budget model |
| Exact satellite visibility windows | Full TLE orbit propagation needed (out of scope here) |
| Dish orientation/tilt constraints | Install guide constraint; not derivable remotely |
