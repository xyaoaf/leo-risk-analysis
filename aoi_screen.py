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
from pathlib import Path

import numpy as np
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
    parser.add_argument("--out", required=True,
                        help="Output CSV path (e.g. outputs/aoi/my_points.csv)")
    args = parser.parse_args()

    df = _load_dataset()

    if args.bbox:
        lat_min, lat_max, lon_min, lon_max = args.bbox
        aoi_desc = (f"bbox lat=[{lat_min},{lat_max}] "
                    f"lon=[{lon_min},{lon_max}]")
        subset = filter_by_bbox(df, lat_min, lat_max, lon_min, lon_max)
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
