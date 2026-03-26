"""
zonal_summary.py — Aggregate point-level risk results to census reporting units.

Joins pipeline result CSVs back to DATA_CHALLENGE_50.csv on location_id to
recover the 15-digit Census block GEOID (geoid_cb), then aggregates to:

  Census block group  (geoid_bg = first 12 chars of geoid_cb)
  County              (county_fips = first 5 chars)

This is the "planning layer" of the system: state broadband offices can use
these aggregated statistics to prioritise which census areas need on-site
verification or targeted grant allocation.

Output files
------------
    outputs/zonal/block_group_summary.csv  — per block group
    outputs/zonal/county_summary.csv       — per county
    outputs/zonal/county_choropleth.png    — static choropleth map (mean risk)

Usage
-----
    # Using the Challenge-50 sample (available now)
    conda run -n cs378 python zonal_summary.py

    # Using the full NC batch results (available after batch run)
    conda run -n cs378 python zonal_summary.py \\
        --results-csv outputs/batch/batch_results.csv

Notes on geoid_cb structure (15-digit Census block GEOID)
----------------------------------------------------------
    Position  0- 1: State FIPS     (e.g. "37" = North Carolina)
    Position  2- 4: County FIPS    (e.g. "179" = Union County)
    Position  5-10: Census Tract   (6 digits, e.g. "020316")
    Position 11-14: Block          (4 digits, e.g. "2002")
    Block Group     = geoid_cb[:11] + geoid_cb[11]  →  first 12 chars
    (The block group is the leading digit of the 4-digit block number)

Status: Implemented — produces real outputs on Challenge-50 (37 pts).
        Scales automatically to full batch results when available.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT        = Path(__file__).parent
CSV_PATH    = ROOT / "DATA_CHALLENGE_50.csv"
OUT_DIR     = ROOT / "outputs" / "zonal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_NC_LAT  = 35.5
_ASPECT  = 1.0 / np.cos(np.radians(_NC_LAT))
LON_LIM  = (-84.5, -75.2)
LAT_LIM  = (33.7,  36.7)

TIER_COLORS = {
    "low":      "#2ecc71",
    "moderate": "#f39c12",
    "high":     "#e67e22",
    "critical": "#e74c3c",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_results(results_csv: Path) -> pd.DataFrame:
    """Load pipeline results CSV; keep only successfully scored rows."""
    df = pd.read_csv(results_csv)
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        df = df[df["error"].isna()].copy()
        if n_err:
            print(f"  Dropped {n_err} error rows; {len(df)} successful rows remain")
    # Coerce numeric columns
    for col in ("risk_score", "slope_deg", "canopy_max_m", "building_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "feasible" in df.columns:
        df["feasible"] = (df["feasible"].astype(str).str.lower()
                          .map({"true": True, "false": False, "1": True, "0": False})
                          .fillna(True))
    # Ensure location_id is string for join
    df["location_id"] = df["location_id"].astype(str)
    return df


def join_geoids(results: pd.DataFrame) -> pd.DataFrame:
    """Join results to DATA_CHALLENGE_50.csv to recover geoid_cb."""
    print(f"Loading geoid_cb from {CSV_PATH.name} …", end=" ", flush=True)
    ref = pd.read_csv(CSV_PATH, usecols=["location_id", "geoid_cb"],
                      dtype={"location_id": str, "geoid_cb": str})
    print(f"{len(ref):,} rows")
    merged = results.merge(ref, on="location_id", how="left")
    missing = merged["geoid_cb"].isna().sum()
    if missing:
        print(f"  Warning: {missing} rows could not be matched to a geoid_cb "
              f"(may be from ad-hoc coordinates rather than the challenge dataset)")
    # Derive geographic IDs
    merged["county_fips"] = merged["geoid_cb"].str[:5]
    merged["geoid_bg"]    = merged["geoid_cb"].str[:12]   # block group (12-char GEOID)
    merged["tract_id"]    = merged["geoid_cb"].str[:11]   # census tract (11-char)
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _agg_unit(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Generic aggregation to any geographic unit."""
    grp = df.groupby(group_col).agg(
        n_points        = ("risk_score",  "count"),
        mean_risk       = ("risk_score",  "mean"),
        median_risk     = ("risk_score",  "median"),
        n_feasible      = ("feasible",    "sum"),
        pct_feasible    = ("feasible",    "mean"),
        pct_critical    = ("risk_tier",   lambda s: (s == "critical").mean()),
        pct_high        = ("risk_tier",   lambda s: (s == "high").mean()),
        pct_moderate    = ("risk_tier",   lambda s: (s == "moderate").mean()),
        pct_low         = ("risk_tier",   lambda s: (s == "low").mean()),
        dominant_mode   = ("dominant",    lambda s: s.value_counts().index[0]
                                          if len(s) > 0 else "unknown"),
    ).reset_index()

    grp["n_feasible"]   = grp["n_feasible"].astype(int)
    grp["n_infeasible"] = grp["n_points"] - grp["n_feasible"]
    for col in ("mean_risk", "median_risk"):
        grp[col] = grp[col].round(1)
    for col in ("pct_feasible", "pct_critical", "pct_high", "pct_moderate", "pct_low"):
        grp[col] = (grp[col] * 100).round(1)
    return grp


def block_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "geoid_bg" not in df.columns or df["geoid_bg"].isna().all():
        return pd.DataFrame()
    return _agg_unit(df[df["geoid_bg"].notna()], "geoid_bg")


def tract_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to census tract level (11-digit GEOID, standard FCC reporting unit)."""
    if "tract_id" not in df.columns or df["tract_id"].isna().all():
        return pd.DataFrame()
    return _agg_unit(df[df["tract_id"].notna()], "tract_id")


def county_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "county_fips" not in df.columns or df["county_fips"].isna().all():
        return pd.DataFrame()
    return _agg_unit(df[df["county_fips"].notna()], "county_fips")


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def _load_nc_counties() -> gpd.GeoDataFrame:
    bd = ROOT / "data" / "boundaries"
    bd.mkdir(parents=True, exist_ok=True)
    fp = bd / "us_counties_fips.geojson"
    if not fp.exists():
        url = ("https://raw.githubusercontent.com/plotly/datasets/master/"
               "geojson-counties-fips.json")
        print(f"Downloading county boundaries …")
        with urllib.request.urlopen(url, timeout=60) as r:
            fp.write_bytes(r.read())
    gdf = gpd.read_file(fp)
    gdf["county_fips"] = gdf["id"].astype(str).str.zfill(5)
    nc = gdf[gdf["county_fips"].str.startswith("37")].copy()
    if nc.crs is None:
        nc = nc.set_crs("EPSG:4326")
    elif nc.crs.to_epsg() != 4326:
        nc = nc.to_crs("EPSG:4326")
    return nc


def plot_county_choropleth(county_sum: pd.DataFrame, nc_counties: gpd.GeoDataFrame,
                           out_path: Path, n_pts_total: int):
    merged = nc_counties.merge(county_sum, on="county_fips", how="left")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.patch.set_facecolor("white")
    plt.rcParams.update({"font.family": "sans-serif"})

    def _base(ax, title):
        ax.set_facecolor("#f8fafc")
        nc_counties.plot(ax=ax, color="#e8eef4", edgecolor="#9ca3af", linewidth=0.35)
        ax.set_xlim(*LON_LIM); ax.set_ylim(*LAT_LIM)
        ax.set_aspect(_ASPECT)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#d1d5db"); sp.set_linewidth(0.5)
        ax.set_title(title, fontsize=9, fontweight="bold", color="#1f2937", pad=6)

    # Panel 1: mean risk score
    ax = axes[0]
    _base(ax, "Mean Risk Score by County")
    merged.plot(column="mean_risk", ax=ax, cmap="RdYlGn_r",
                vmin=0, vmax=100, edgecolor="#6b7280", linewidth=0.35,
                missing_kwds={"color": "#f3f4f6", "label": "No data"},
                legend=True,
                legend_kwds={"label": "Mean risk (0–100)",
                             "shrink": 0.7, "orientation": "vertical"})

    # Panel 2: % infeasible
    ax = axes[1]
    _base(ax, "% Infeasible Locations by County")
    merged["pct_infeasible"] = (100 - merged["pct_feasible"].fillna(50)).round(1)
    merged.plot(column="pct_infeasible", ax=ax, cmap="Reds",
                vmin=0, vmax=100, edgecolor="#6b7280", linewidth=0.35,
                missing_kwds={"color": "#f3f4f6"},
                legend=True,
                legend_kwds={"label": "% Infeasible",
                             "shrink": 0.7, "orientation": "vertical",
                             "format": FuncFormatter(lambda x, _: f"{x:.0f}%")})

    # Panel 3: % critical
    ax = axes[2]
    _base(ax, "% Critical-Risk Locations by County")
    merged.plot(column="pct_critical", ax=ax, cmap="OrRd",
                vmin=0, vmax=100, edgecolor="#6b7280", linewidth=0.35,
                missing_kwds={"color": "#f3f4f6"},
                legend=True,
                legend_kwds={"label": "% Critical",
                             "shrink": 0.7, "orientation": "vertical",
                             "format": FuncFormatter(lambda x, _: f"{x:.0f}%")})

    # Data note
    counties_with_data = county_sum["county_fips"].nunique()
    note = (f"Based on {n_pts_total} scored points across "
            f"{counties_with_data} counties.  "
            f"Grey counties have no sampled points in this run.")
    fig.text(0.5, -0.01, note, ha="center", fontsize=8, color="#6b7280",
             style="italic")

    fig.suptitle("NC Starlink Risk Analysis — County-Level Zonal Summary",
                 fontsize=12, fontweight="bold", color="#111827", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate risk results to census block groups and counties",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-csv",
        default="outputs/challenge50/challenge50_results.csv",
        help="Path to pipeline results CSV (default: challenge50 sample)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    if not results_path.exists():
        print(f"ERROR: results CSV not found: {results_path}")
        return

    print(f"\nLoading results: {results_path}")
    results = load_results(results_path)
    print(f"  {len(results)} scored rows")

    # Join geoid_cb from the master dataset
    df = join_geoids(results)

    # Block-group aggregation
    print("\nBlock-group aggregation …")
    bg_sum = block_group_summary(df)
    if not bg_sum.empty:
        bg_out = OUT_DIR / "block_group_summary.csv"
        bg_sum.to_csv(bg_out, index=False)
        print(f"  {len(bg_sum)} block groups → {bg_out}")
    else:
        print("  No block-group data (geoid_cb not matched — ad-hoc points?)")

    # Census tract aggregation (standard FCC broadband reporting unit)
    print("\nCensus tract aggregation …")
    tr_sum = tract_summary(df)
    if not tr_sum.empty:
        tr_out = OUT_DIR / "tract_summary.csv"
        tr_sum.to_csv(tr_out, index=False)
        print(f"  {len(tr_sum)} tracts → {tr_out}")
    else:
        print("  No tract data (geoid_cb not matched — ad-hoc points?)")

    # County aggregation
    print("\nCounty aggregation …")
    co_sum = county_summary(df)
    if not co_sum.empty:
        co_out = OUT_DIR / "county_summary.csv"
        co_sum.to_csv(co_out, index=False)
        print(f"  {len(co_sum)} counties → {co_out}")

        # Print summary table
        print("\n  County Summary Table:")
        print(f"  {'County':>8}  {'N':>5}  {'Mean Risk':>9}  "
              f"{'Feasible%':>10}  {'Critical%':>10}  Dominant")
        print("  " + "─" * 65)
        for _, row in co_sum.sort_values("mean_risk", ascending=False).iterrows():
            print(f"  {row['county_fips']:>8}  {row['n_points']:>5}  "
                  f"{row['mean_risk']:>9.1f}  {row['pct_feasible']:>9.1f}%  "
                  f"{row['pct_critical']:>9.1f}%  {row['dominant_mode']}")

    # Choropleth map
    print("\nGenerating county choropleth …")
    try:
        nc_counties = _load_nc_counties()
        if not co_sum.empty:
            plot_county_choropleth(co_sum, nc_counties,
                                   OUT_DIR / "county_choropleth.png",
                                   n_pts_total=len(results))
    except Exception as e:
        print(f"  Map generation failed: {e}")

    print(f"\nAll zonal outputs → {OUT_DIR}/")
    print("  Rerun with --results-csv outputs/batch/batch_results.csv "
          "once the full NC batch is complete for county-level statistics "
          "across all 100 NC counties.")


if __name__ == "__main__":
    main()
