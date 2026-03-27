"""
aoi_screen.py — Filter DATA_CHALLENGE_50 address points by an Area of Interest (AOI).

Selects points within a bounding box or GeoJSON polygon from the full 4.67M-row
NC dataset and writes a subset CSV that batch_nc_analysis.py can consume directly
via its --input-csv flag.

Status: Implemented (prototype — no UI; script-based workflow).

Usage
-----
    # Bounding-box filter (lat_min lat_max lon_min lon_max)
    conda run -n cs378 python aoi_screen.py \\
        --bbox 35.0 35.5 -81.1 -80.4 \\
        --out  outputs/aoi/charlotte_metro_points.csv

    # GeoJSON polygon filter
    conda run -n cs378 python aoi_screen.py \\
        --geojson data/aoi/my_county.geojson \\
        --out     outputs/aoi/my_county_points.csv

    # Then feed into the batch pipeline:
    conda run -n cs378 python batch_nc_analysis.py \\
        --input-csv outputs/aoi/charlotte_metro_points.csv \\
        --max-points 30 --workers 2

Input formats
-------------
    --bbox   lat_min lat_max lon_min lon_max   (four floats, WGS84)
    --geojson FILE                             GeoJSON file with a single Polygon
                                               or MultiPolygon feature/FeatureCollection

Output CSV columns (same schema as DATA_CHALLENGE_50.csv + county_fips)
-------------------------------------------------------------------------------
    location_id, latitude, longitude, geoid_cb, county_fips

    This schema is what batch_nc_analysis.py expects via --input-csv.

Notes
-----
    - Loading 4.67M rows takes ~5 s.  The spatial filter itself is vectorised
      (bbox) or uses geopandas spatial join (polygon).
    - For very large AOIs with many thousands of points, consider adding
      --max-points to the subsequent batch call to test before the full run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).parent
CSV_PATH = ROOT / "DATA_CHALLENGE_50.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# Filtering logic
# ═══════════════════════════════════════════════════════════════════════════════

def _load_dataset() -> pd.DataFrame:
    t0 = time.time()
    print(f"Loading {CSV_PATH.name} …", end=" ", flush=True)
    df = pd.read_csv(CSV_PATH, dtype={"geoid_cb": str})
    df["county_fips"] = df["geoid_cb"].str[:5]
    print(f"{len(df):,} rows  ({time.time()-t0:.1f}s)")
    return df


def filter_by_bbox(df: pd.DataFrame,
                   lat_min: float, lat_max: float,
                   lon_min: float, lon_max: float) -> pd.DataFrame:
    """Return rows within (lat_min..lat_max, lon_min..lon_max)."""
    mask = (
        (df["latitude"]  >= lat_min) & (df["latitude"]  <= lat_max) &
        (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
    )
    return df[mask].copy()


def filter_by_geojson(df: pd.DataFrame, geojson_path: str) -> pd.DataFrame:
    """Return rows whose (lon, lat) falls within the GeoJSON polygon."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        print("ERROR: geopandas + shapely required for --geojson filter.", file=sys.stderr)
        sys.exit(1)

    with open(geojson_path) as f:
        gj = json.load(f)

    # Accept Feature, FeatureCollection, or bare Geometry
    from shapely.geometry import shape
    if gj["type"] == "FeatureCollection":
        geometries = [shape(feat["geometry"]) for feat in gj["features"]]
    elif gj["type"] == "Feature":
        geometries = [shape(gj["geometry"])]
    else:
        geometries = [shape(gj)]

    from shapely.ops import unary_union
    aoi_geom = unary_union(geometries)

    # Vectorised point-in-polygon via geopandas spatial join
    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    gdf_aoi = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326")
    joined  = gpd.sjoin(gdf_pts, gdf_aoi, how="inner", predicate="within")
    return df.loc[joined.index].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# County name → FIPS lookup
# ═══════════════════════════════════════════════════════════════════════════════

_NC_COUNTY_FIPS: dict[str, str] | None = None


def _load_nc_county_fips() -> dict[str, str]:
    """Return {lowercase_county_name: 5-digit FIPS} for all NC counties.

    Uses the Census Bureau county FIPS CSV (downloaded once and cached).
    """
    global _NC_COUNTY_FIPS
    if _NC_COUNTY_FIPS is not None:
        return _NC_COUNTY_FIPS

    cache_dir = ROOT / "data" / "boundaries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / "nc_county_fips.json"

    if fp.exists():
        _NC_COUNTY_FIPS = json.loads(fp.read_text())
        return _NC_COUNTY_FIPS

    # Download Census county FIPS listing
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    print("Downloading county boundaries for name lookup …", end=" ", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"FAILED: {e}")
        print("Falling back to FIPS extraction from dataset geoid_cb.")
        _NC_COUNTY_FIPS = {}
        return _NC_COUNTY_FIPS

    # Extract NC counties and their names from GeoJSON properties
    mapping = {}
    for feat in data["features"]:
        fips = str(feat.get("id", ""))
        if fips.startswith("37"):
            # GeoJSON properties may have "NAME" or we derive from FIPS
            name = feat.get("properties", {}).get("NAME", "")
            if name:
                mapping[name.lower()] = fips
    # If GeoJSON didn't have NAME property, use a hardcoded NC list
    if not mapping:
        mapping = _hardcoded_nc_counties()

    fp.write_text(json.dumps(mapping, indent=2))
    print(f"OK ({len(mapping)} counties)")
    _NC_COUNTY_FIPS = mapping
    return _NC_COUNTY_FIPS


def _hardcoded_nc_counties() -> dict[str, str]:
    """Fallback: hardcoded NC county name → FIPS mapping (100 counties)."""
    # Source: US Census Bureau FIPS codes for North Carolina (state=37)
    counties = {
        "alamance": "37001", "alexander": "37003", "alleghany": "37005",
        "anson": "37007", "ashe": "37009", "avery": "37011",
        "beaufort": "37013", "bertie": "37015", "bladen": "37017",
        "brunswick": "37019", "buncombe": "37021", "burke": "37023",
        "cabarrus": "37025", "caldwell": "37027", "camden": "37029",
        "carteret": "37031", "caswell": "37033", "catawba": "37035",
        "chatham": "37037", "cherokee": "37039", "chowan": "37041",
        "clay": "37043", "cleveland": "37045", "columbus": "37047",
        "craven": "37049", "cumberland": "37051", "currituck": "37053",
        "dare": "37055", "davidson": "37057", "davie": "37059",
        "duplin": "37061", "durham": "37063", "edgecombe": "37065",
        "forsyth": "37067", "franklin": "37069", "gaston": "37071",
        "gates": "37073", "graham": "37075", "granville": "37077",
        "greene": "37079", "guilford": "37081", "halifax": "37083",
        "harnett": "37085", "haywood": "37087", "henderson": "37089",
        "hertford": "37091", "hoke": "37093", "hyde": "37095",
        "iredell": "37097", "jackson": "37099", "johnston": "37101",
        "jones": "37103", "lee": "37105", "lenoir": "37107",
        "lincoln": "37109", "macon": "37113", "madison": "37115",
        "martin": "37117", "mcdowell": "37111", "mecklenburg": "37119",
        "mitchell": "37121", "montgomery": "37123", "moore": "37125",
        "nash": "37127", "new hanover": "37129", "northampton": "37131",
        "onslow": "37133", "orange": "37135", "pamlico": "37137",
        "pasquotank": "37139", "pender": "37141", "perquimans": "37143",
        "person": "37145", "pitt": "37147", "polk": "37149",
        "randolph": "37151", "richmond": "37153", "robeson": "37155",
        "rockingham": "37157", "rowan": "37159", "rutherford": "37161",
        "sampson": "37163", "scotland": "37165", "stanly": "37167",
        "stokes": "37169", "surry": "37171", "swain": "37173",
        "transylvania": "37175", "tyrrell": "37177", "union": "37179",
        "vance": "37181", "wake": "37183", "warren": "37185",
        "washington": "37187", "watauga": "37189", "wayne": "37191",
        "wilkes": "37193", "wilson": "37195", "yadkin": "37197",
        "yancey": "37199",
    }
    return counties


def filter_by_county(df: pd.DataFrame, county_name: str) -> pd.DataFrame:
    """Filter dataset rows to those in the named NC county."""
    lookup = _load_nc_county_fips()
    key = county_name.strip().lower()
    fips = lookup.get(key)

    if fips is None:
        # Fuzzy match: find counties containing the search string
        matches = [name for name in lookup if key in name]
        if len(matches) == 1:
            fips = lookup[matches[0]]
            print(f"  Matched '{county_name}' → {matches[0].title()} ({fips})")
        elif matches:
            print(f"  Ambiguous county name '{county_name}'. Did you mean one of:")
            for m in sorted(matches):
                print(f"    --county \"{m.title()}\"  (FIPS {lookup[m]})")
            sys.exit(1)
        else:
            print(f"  County '{county_name}' not found in NC. Available counties:")
            for name in sorted(lookup):
                print(f"    {name.title()} ({lookup[name]})")
            sys.exit(1)

    return df[df["county_fips"] == fips].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

def _print_summary(subset: pd.DataFrame, aoi_desc: str):
    n = len(subset)
    if n == 0:
        print("  No points found within AOI.")
        return
    print(f"\n  AOI: {aoi_desc}")
    print(f"  Points selected : {n:,}")
    print(f"  Counties covered: {subset['county_fips'].nunique()}")
    print(f"  Lat range       : {subset['latitude'].min():.4f} — {subset['latitude'].max():.4f}")
    print(f"  Lon range       : {subset['longitude'].min():.4f} — {subset['longitude'].max():.4f}")
    top_counties = subset["county_fips"].value_counts().head(5)
    print(f"  Top counties    : {', '.join(f'{fips}({n})' for fips,n in top_counties.items())}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Filter DATA_CHALLENGE_50 points by AOI for targeted batch analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bbox", nargs=4, type=float,
                       metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                       help="Bounding box filter")
    group.add_argument("--geojson", metavar="FILE",
                       help="GeoJSON polygon file")
    group.add_argument("--county", metavar="NAME",
                       help="NC county name (e.g. \"Mecklenburg\", \"Wake\")")
    parser.add_argument("--out", required=True,
                        help="Output CSV path (e.g. outputs/aoi/my_points.csv)")
    args = parser.parse_args()

    df = _load_dataset()

    if args.bbox:
        lat_min, lat_max, lon_min, lon_max = args.bbox
        aoi_desc = (f"bbox lat=[{lat_min},{lat_max}] "
                    f"lon=[{lon_min},{lon_max}]")
        subset = filter_by_bbox(df, lat_min, lat_max, lon_min, lon_max)
    elif args.county:
        aoi_desc = f"county={args.county}"
        subset = filter_by_county(df, args.county)
    else:
        aoi_desc = f"geojson={args.geojson}"
        subset = filter_by_geojson(df, args.geojson)

    _print_summary(subset, aoi_desc)

    if len(subset) == 0:
        print("Nothing to save.")
        sys.exit(0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write with the columns batch_nc_analysis.py expects
    out_cols = ["location_id", "latitude", "longitude", "geoid_cb", "county_fips"]
    subset[out_cols].to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}  ({len(subset):,} rows)")
    print(f"\n  Next step — run batch on this AOI:")
    print(f"    conda run -n cs378 python batch_nc_analysis.py \\")
    print(f"        --input-csv {args.out} \\")
    print(f"        --workers 2")


if __name__ == "__main__":
    main()
