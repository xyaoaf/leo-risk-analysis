"""
explore_challenge50.py — Exploratory analysis + pipeline test on DATA_CHALLENGE_50.csv

Steps:
  1. Load & summarise the NC dataset (~4.67M rows)
  2. Map A: CONUS overview  — NC highlighted inside continental US
  3. Map B: NC state overview — hex-density of all 4.67M points
  4. Maps C1–C3: Three zoomed cluster maps  (W / Central / E North Carolina)
  5. Sample 50 spatially-representative test points (10×5 grid over NC bbox)
  6. Run analyze_location() on each; record elapsed_s per point
  7. Map D: NC risk map of the 50 tested points
  8. Print timing + results table; save outputs/challenge50_results.csv

Usage:
    conda run -n cs378 python explore_challenge50.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import contextily as ctx
from shapely.geometry import Point, box
import urllib.request, json as _json

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
CSV_PATH   = ROOT / "DATA_CHALLENGE_50.csv"
OUTPUT_DIR = ROOT / "outputs" / "challenge50"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from feasibility import analyze_location

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Color scheme ─────────────────────────────────────────────────────────────
DARK_BG   = "#111827"
DARK_AX   = "#1a1a2e"
RISK_CMAP = "RdYlGn_r"

TIER_COLORS = {
    "low":      "#2ecc71",
    "moderate": "#f39c12",
    "high":     "#e67e22",
    "critical": "#e74c3c",
}
DOM_COLORS = {
    "clear":      "#2ecc71",
    "vegetation": "#27ae60",
    "building":   "#e74c3c",
    "terrain":    "#8e6b3e",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load dataset
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset() -> pd.DataFrame:
    log.info("Loading DATA_CHALLENGE_50.csv …")
    t0 = time.time()
    df = pd.read_csv(CSV_PATH, dtype={"geoid_cb": str})
    df["state_fips"]  = df["geoid_cb"].str[:2]
    df["county_fips"] = df["geoid_cb"].str[:5]
    elapsed = time.time() - t0
    log.info(f"  Loaded {len(df):,} rows in {elapsed:.1f}s")
    log.info(f"  Lat: {df['latitude'].min():.4f} – {df['latitude'].max():.4f}")
    log.info(f"  Lon: {df['longitude'].min():.4f} – {df['longitude'].max():.4f}")
    log.info(f"  Unique counties: {df['county_fips'].nunique()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Download boundary data
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_geojson(url: str, cache_path: Path) -> gpd.GeoDataFrame:
    if not cache_path.exists():
        log.info(f"  Downloading {url} …")
        with urllib.request.urlopen(url, timeout=30) as r:
            cache_path.write_bytes(r.read())
    return gpd.read_file(cache_path)


def load_boundaries() -> dict:
    cache = ROOT / "data" / "boundaries"
    cache.mkdir(parents=True, exist_ok=True)

    # US states (Natural Earth 110m via GitHub)
    states = _fetch_geojson(
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_1_states_provinces.geojson",
        cache / "ne_110m_states.geojson",
    )
    # Filter to continental US
    us_states = states[states["iso_a2"] == "US"].copy()
    conus = us_states[~us_states["name"].isin(["Alaska", "Hawaii"])]

    # All US counties — Plotly dataset (all 3221 counties, FIPS as feature id)
    all_counties_path = cache / "us_counties_fips.geojson"
    if not all_counties_path.exists():
        log.info("  Downloading US counties GeoJSON …")
        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        with urllib.request.urlopen(url, timeout=60) as r:
            all_counties_path.write_bytes(r.read())
    all_counties = gpd.read_file(all_counties_path)
    # FIPS is in 'id' column; filter to NC (37xxx)
    all_counties["county_fips"] = all_counties["id"].astype(str).str.zfill(5)
    nc_counties = all_counties[all_counties["county_fips"].str.startswith("37")].copy()

    return {"conus": conus, "nc_counties": nc_counties}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Map A — CONUS overview
# ═══════════════════════════════════════════════════════════════════════════════

def map_conus_overview(df: pd.DataFrame, boundaries: dict, out_path: Path):
    log.info("Generating Map A: CONUS overview …")
    conus     = boundaries["conus"].to_crs("EPSG:4326")
    nc_state  = conus[conus["name"] == "North Carolina"]

    # Subsample for density visualisation (every 50th point ≈ 93k pts)
    sample = df.iloc[::50][["latitude", "longitude"]].copy()

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Continental US outline
    conus.plot(ax=ax, color="#1f2d47", edgecolor="#374151", linewidth=0.6)

    # North Carolina highlighted
    nc_state.plot(ax=ax, color="#1e3a5f", edgecolor="#60a5fa", linewidth=1.5)

    # Density scatter of NC points
    ax.scatter(
        sample["longitude"], sample["latitude"],
        s=0.08, c="#60a5fa", alpha=0.3, linewidths=0, rasterized=True,
        label=f"NC addresses ({len(df):,.0f} total)",
    )

    # Annotation arrow
    nc_cx = float(nc_state.geometry.centroid.x)
    nc_cy = float(nc_state.geometry.centroid.y)
    ax.annotate(
        "North Carolina\n4.67 M addresses",
        xy=(nc_cx, nc_cy), xytext=(nc_cx - 12, nc_cy - 7),
        color="white", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#60a5fa", lw=1.2),
        ha="center",
    )

    ax.set_xlim(-125, -66); ax.set_ylim(24, 50)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(
        "DATA_CHALLENGE_50 — Geographic Coverage\nAll NC Addresses (FIPS 37, 100 counties)",
        color="white", fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(loc="lower right", fontsize=8,
              facecolor="#1a1a2e", edgecolor="#374151", labelcolor="white")

    # Inset: state-level zoom
    ax_ins = fig.add_axes([0.58, 0.08, 0.38, 0.35], facecolor=DARK_AX)
    conus.plot(ax=ax_ins, color="#1f2d47", edgecolor="#374151", linewidth=0.5)
    nc_state.plot(ax=ax_ins, color="#1e3a5f", edgecolor="#60a5fa", linewidth=1.5)
    ax_ins.set_xlim(-88, -74); ax_ins.set_ylim(30, 40)
    ax_ins.set_xticks([]); ax_ins.set_yticks([])
    ax_ins.set_title("Southeast region", color="white", fontsize=7)
    for sp in ax_ins.spines.values(): sp.set_edgecolor("#374151")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Map B — NC state density overview
# ═══════════════════════════════════════════════════════════════════════════════

def map_nc_density(df: pd.DataFrame, boundaries: dict, out_path: Path):
    log.info("Generating Map B: NC density overview …")
    nc_counties = boundaries["nc_counties"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=DARK_BG)

    # ── Left: hex-density (all 4.67M points) ────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(DARK_AX)
    hb = ax.hexbin(
        df["longitude"], df["latitude"],
        gridsize=100, cmap="YlOrRd", bins="log",
        mincnt=1, linewidths=0.1,
    )
    # County outlines
    try:
        nc_counties.plot(ax=ax, color="none",
                         edgecolor="#374151", linewidth=0.5)
    except Exception:
        pass
    cb = plt.colorbar(hb, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("log₁₀(address count)", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    _style_map_ax(ax, "NC Address Density — All 4.67M Points\n(log scale per hex cell)")

    # ── Right: county-level choropleth ───────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(DARK_AX)
    county_counts = df.groupby("county_fips").size().rename("count").reset_index()
    try:
        # Merge with geometry (county_fips already set in load_boundaries)
        nc_geo = nc_counties.copy()
        merged = nc_geo.merge(county_counts, on="county_fips", how="left")
        merged["count"] = merged["count"].fillna(0)
        merged.plot(
            column="count", ax=ax,
            cmap="YlOrRd", edgecolor="#374151", linewidth=0.5,
            legend=True,
            legend_kwds={"label": "Addresses per county", "shrink": 0.7,
                         "orientation": "vertical"},
        )
    except Exception as e:
        log.warning(f"County choropleth fallback (scatter): {e}")
        # Fallback: scatter coloured by county count
        county_map = dict(zip(county_counts["county_fips"], county_counts["count"]))
        sample = df.sample(200_000, random_state=42)
        c_vals = sample["county_fips"].map(county_map).fillna(0)
        sc = ax.scatter(
            sample["longitude"], sample["latitude"],
            c=c_vals, s=0.05, cmap="YlOrRd", alpha=0.4,
            linewidths=0, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, shrink=0.7).set_label(
            "County address count", color="white", fontsize=8)

    _style_map_ax(ax, "Address Count per County\n(100 NC counties)")

    plt.suptitle(
        "North Carolina — DATA_CHALLENGE_50 Dataset Overview",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Maps C1–C3 — Three regional cluster maps
# ═══════════════════════════════════════════════════════════════════════════════

REGIONS = {
    "West_Mountains": {
        "lon": (-84.32, -81.0),
        "lat": (34.9,   36.6),
        "label": "Western NC — Appalachian Mountains",
        "color": "#8e6b3e",
    },
    "Central_Piedmont": {
        "lon": (-81.0,  -78.5),
        "lat": (34.8,   36.5),
        "label": "Central NC — Piedmont & Charlotte Metro",
        "color": "#3b82f6",
    },
    "East_CoastalPlain": {
        "lon": (-78.5,  -75.46),
        "lat": (33.84,  36.1),
        "label": "Eastern NC — Coastal Plain & Outer Banks",
        "color": "#10b981",
    },
}


def map_regional_clusters(df: pd.DataFrame, out_path: Path):
    log.info("Generating Maps C1–C3: regional cluster zoom …")
    fig, axes = plt.subplots(1, 3, figsize=(21, 8), facecolor=DARK_BG)

    for ax, (region_key, region) in zip(axes, REGIONS.items()):
        lon_lo, lon_hi = region["lon"]
        lat_lo, lat_hi = region["lat"]
        col            = region["color"]

        mask   = (
            (df["longitude"] >= lon_lo) & (df["longitude"] <= lon_hi) &
            (df["latitude"]  >= lat_lo) & (df["latitude"]  <= lat_hi)
        )
        sub = df[mask]
        log.info(f"  {region_key}: {len(sub):,} addresses")

        ax.set_facecolor(DARK_AX)
        ax.hexbin(
            sub["longitude"], sub["latitude"],
            gridsize=60, cmap="YlOrRd", bins="log",
            mincnt=1, linewidths=0.0,
        )
        ax.set_xlim(lon_lo, lon_hi)
        ax.set_ylim(lat_lo, lat_hi)

        # County count annotation
        n_counties = sub["county_fips"].nunique()
        n_k = len(sub) / 1000
        ax.text(
            0.02, 0.97,
            f"{n_k:.0f}k addresses\n{n_counties} counties",
            transform=ax.transAxes, color="white", fontsize=9,
            va="top", ha="left",
            bbox=dict(facecolor="#111827", edgecolor=col, boxstyle="round,pad=0.3"),
        )
        _style_map_ax(ax, region["label"])

    plt.suptitle(
        "North Carolina — Three Geographic Regions (address density)",
        color="white", fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Sample 50 representative test points (10 × 5 spatial grid over NC bbox)
# ═══════════════════════════════════════════════════════════════════════════════

def select_test_points(df: pd.DataFrame, n_cols: int = 10, n_rows: int = 5) -> pd.DataFrame:
    """
    Divide the NC bounding box into a n_cols × n_rows grid (50 cells).
    For each cell, pick the data point closest to the cell center.
    Returns a DataFrame of 50 (or fewer) representative points.
    """
    lat_lo, lat_hi = df["latitude"].min(),  df["latitude"].max()
    lon_lo, lon_hi = df["longitude"].min(), df["longitude"].max()

    lat_edges = np.linspace(lat_lo, lat_hi, n_rows + 1)
    lon_edges = np.linspace(lon_lo, lon_hi, n_cols + 1)

    test_rows = []
    for i in range(n_rows):
        for j in range(n_cols):
            lat_c = (lat_edges[i] + lat_edges[i + 1]) / 2
            lon_c = (lon_edges[j] + lon_edges[j + 1]) / 2

            # Points in this cell
            mask = (
                (df["latitude"]  >= lat_edges[i])     &
                (df["latitude"]  <  lat_edges[i + 1]) &
                (df["longitude"] >= lon_edges[j])     &
                (df["longitude"] <  lon_edges[j + 1])
            )
            cell = df[mask]
            if len(cell) == 0:
                continue  # empty cell (e.g. ocean/state border)

            # Point nearest to cell center
            dist2 = (cell["latitude"] - lat_c) ** 2 + (cell["longitude"] - lon_c) ** 2
            best  = cell.loc[dist2.idxmin()].copy()
            best["grid_row"] = i
            best["grid_col"] = j
            test_rows.append(best)

    result = pd.DataFrame(test_rows).reset_index(drop=True)
    log.info(f"Selected {len(result)} test points from {n_rows}×{n_cols} grid")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Run pipeline on test points
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(test_points: pd.DataFrame) -> list[dict]:
    results = []
    n = len(test_points)

    for idx, row in test_points.iterrows():
        lat  = float(row["latitude"])
        lon  = float(row["longitude"])
        lid  = str(row["location_id"])
        label = f"NC_{idx+1:02d}_{lid}"

        log.info(f"\n{'='*60}")
        log.info(f"[{idx+1}/{n}] {label}  ({lat:.5f}, {lon:.5f})")
        log.info(f"{'='*60}")

        t_start = time.time()
        try:
            res = analyze_location(lat, lon, run_local_search=False)
        except Exception as exc:
            import traceback
            log.error(f"FAILED: {exc}")
            traceback.print_exc()
            results.append({
                "idx": idx + 1,
                "location_id": lid,
                "label": label,
                "lat": lat,
                "lon": lon,
                "error": str(exc),
                "elapsed_s": round(time.time() - t_start, 1),
            })
            continue

        res["label"]       = label
        res["location_id"] = lid
        res["idx"]         = idx + 1
        results.append(res)

        log.info(
            f"  feasible={res['feasible']}  "
            f"risk={res['risk']['risk_score']:.1f} [{res['risk']['risk_tier']}]  "
            f"dominant={res['classification']['dominant']}  "
            f"elapsed={res['elapsed_s']}s"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Map D — Risk map of 50 tested points
# ═══════════════════════════════════════════════════════════════════════════════

def map_risk_results(results: list[dict], df_all: pd.DataFrame,
                     boundaries: dict, out_path: Path):
    log.info("Generating Map D: risk results map …")

    ok = [r for r in results if "error" not in r]
    if not ok:
        log.warning("No successful results to plot.")
        return

    lats   = [r["lat"] for r in ok]
    lons   = [r["lon"] for r in ok]
    scores = [r["risk"]["risk_score"]  for r in ok]
    tiers  = [r["risk"]["risk_tier"]   for r in ok]
    doms   = [r["classification"]["dominant"] for r in ok]
    feas   = [r["feasible"]            for r in ok]
    times  = [r["elapsed_s"]           for r in ok]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=DARK_BG)

    # ── Panel 1: Risk score map ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(DARK_AX)
    # Background density of all points (very sparse subsample)
    sample_bg = df_all.sample(min(500_000, len(df_all)), random_state=1)
    ax.scatter(
        sample_bg["longitude"], sample_bg["latitude"],
        s=0.01, c="#2d3748", alpha=0.4, linewidths=0, rasterized=True,
    )
    norm = Normalize(vmin=0, vmax=100)
    sm   = ScalarMappable(cmap=RISK_CMAP, norm=norm)
    sm.set_array([])
    colors = [sm.to_rgba(s) for s in scores]
    sc = ax.scatter(lons, lats, c=scores, s=120,
                    cmap=RISK_CMAP, vmin=0, vmax=100,
                    edgecolors="white", linewidths=0.6, zorder=5)
    cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("Risk score (0–100)", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    _style_map_ax(ax, "Starlink Risk Score — 50 Test Points")

    # ── Panel 2: Dominant obstruction ────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(DARK_AX)
    ax.scatter(
        sample_bg["longitude"], sample_bg["latitude"],
        s=0.01, c="#2d3748", alpha=0.4, linewidths=0, rasterized=True,
    )
    dom_c = [DOM_COLORS[d] for d in doms]
    for dom_type, dom_col in DOM_COLORS.items():
        mask_d = [d == dom_type for d in doms]
        lons_d = [l for l, m in zip(lons, mask_d) if m]
        lats_d = [l for l, m in zip(lats, mask_d) if m]
        if lons_d:
            ax.scatter(lons_d, lats_d, c=dom_col, s=120,
                       edgecolors="white", linewidths=0.6,
                       label=dom_type.capitalize(), zorder=5)
    # Mark infeasible with X
    infeas_lons = [lo for lo, f in zip(lons, feas) if not f]
    infeas_lats = [la for la, f in zip(lats, feas) if not f]
    if infeas_lons:
        ax.scatter(infeas_lons, infeas_lats, marker="x", c="white", s=80,
                   linewidths=1.5, zorder=6, label="Infeasible")
    ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#374151",
              labelcolor="white", loc="lower right", markerscale=0.8)
    _style_map_ax(ax, "Dominant Obstruction Type\n(✗ = infeasible)")

    # ── Panel 3: Processing time ──────────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor(DARK_AX)
    ax.scatter(
        sample_bg["longitude"], sample_bg["latitude"],
        s=0.01, c="#2d3748", alpha=0.4, linewidths=0, rasterized=True,
    )
    t_norm = Normalize(vmin=0, vmax=max(times) * 1.1 if times else 60)
    t_sm   = ScalarMappable(cmap="plasma", norm=t_norm)
    t_sm.set_array([])
    sc3 = ax.scatter(lons, lats, c=times, s=120,
                     cmap="plasma", vmin=0, vmax=max(times) * 1.1 if times else 60,
                     edgecolors="white", linewidths=0.6, zorder=5)
    cb3 = plt.colorbar(sc3, ax=ax, shrink=0.7, pad=0.01)
    cb3.set_label("Processing time (s)", color="white", fontsize=8)
    cb3.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb3.ax.yaxis.get_ticklabels(), color="white")
    _style_map_ax(ax, f"Processing Time per Point\nmedian={np.median(times):.1f}s  max={max(times):.1f}s")

    plt.suptitle(
        "LEO Risk Analysis — Pipeline Results: 50 NC Test Points",
        color="white", fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Timing + summary chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_timing_summary(results: list[dict], out_path: Path):
    log.info("Generating timing + summary chart …")
    ok    = [r for r in results if "error" not in r]
    errs  = [r for r in results if "error" in r]
    if not ok:
        return

    times  = [r["elapsed_s"] for r in ok]
    scores = [r["risk"]["risk_score"] for r in ok]
    tiers  = [r["risk"]["risk_tier"] for r in ok]
    doms   = [r["classification"]["dominant"] for r in ok]
    labels = [r["label"].replace("NC_", "") for r in ok]
    feas   = [r["feasible"] for r in ok]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=DARK_BG)
    axes = axes.flatten()

    # ── A: Processing time per point ─────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(DARK_AX)
    bar_c = [TIER_COLORS.get(t, "#888") for t in tiers]
    ax.bar(range(len(times)), times, color=bar_c, edgecolor="#374151", linewidth=0.3)
    ax.axhline(np.mean(times),  color="white",   lw=1.2, ls="--",
               label=f"Mean {np.mean(times):.1f}s")
    ax.axhline(np.median(times), color="#f59e0b", lw=1.2, ls=":",
               label=f"Median {np.median(times):.1f}s")
    ax.set_xlabel("Point index", color="white", fontsize=8)
    ax.set_ylabel("Elapsed time (s)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    _style_ax_dark(ax, f"Processing Time per Point  (n={len(ok)}, {len(errs)} errors)")
    # Tier legend patches
    for tier, tc in TIER_COLORS.items():
        ax.bar([], [], color=tc, label=tier)
    ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444",
              labelcolor="white", loc="upper right", ncol=2)

    # ── B: Risk score distribution ────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(DARK_AX)
    ax.hist(scores, bins=20, color="#3b82f6", edgecolor="#1e40af", linewidth=0.5, alpha=0.85)
    ax.axvline(np.mean(scores),  color="white",   lw=1.5, ls="--",
               label=f"Mean {np.mean(scores):.1f}")
    ax.axvline(np.median(scores), color="#f59e0b", lw=1.5, ls=":",
               label=f"Median {np.median(scores):.1f}")
    for v, col, lbl in [(20, "#2ecc71", "low/mod"), (45, "#f39c12", "mod/high"),
                         (70, "#e74c3c", "high/crit")]:
        ax.axvline(v, color=col, lw=0.8, ls=":", alpha=0.6, label=lbl)
    ax.set_xlabel("Risk score (0–100)", color="white", fontsize=8)
    ax.set_ylabel("Count", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    _style_ax_dark(ax, "Risk Score Distribution")

    # ── C: Dominant obstruction breakdown ────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor(DARK_AX)
    from collections import Counter
    dom_counts = Counter(doms)
    dom_labels = list(dom_counts.keys())
    dom_vals   = [dom_counts[k] for k in dom_labels]
    dom_cols   = [DOM_COLORS.get(k, "#888") for k in dom_labels]
    wedges, texts, autotexts = ax.pie(
        dom_vals, labels=dom_labels, colors=dom_cols,
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"edgecolor": "#374151", "linewidth": 0.8},
        textprops={"color": "white", "fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_color("white")
    n_feas = sum(feas)
    ax.set_title(
        f"Dominant Obstruction Type\n"
        f"feasible={n_feas}/{len(ok)}  ({100*n_feas/len(ok):.0f}%)",
        color="white", fontsize=9, fontweight="bold",
    )

    # ── D: Score vs time scatter ──────────────────────────────────────────────
    ax = axes[3]
    ax.set_facecolor(DARK_AX)
    sc = ax.scatter(times, scores, c=scores, cmap=RISK_CMAP, vmin=0, vmax=100,
                    s=80, edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_xlabel("Processing time (s)", color="white", fontsize=8)
    ax.set_ylabel("Risk score", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    plt.colorbar(sc, ax=ax, shrink=0.7).set_label(
        "Risk score", color="white", fontsize=7)
    _style_ax_dark(ax, "Processing Time vs Risk Score")

    plt.suptitle(
        "LEO Risk Analysis — Timing & Results Summary  (50 NC Points)",
        color="white", fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Save CSV results + print table
# ═══════════════════════════════════════════════════════════════════════════════

def save_results_csv(results: list[dict], out_path: Path):
    rows = []
    for r in results:
        base = {
            "idx":         r.get("idx"),
            "location_id": r.get("location_id"),
            "lat":         r.get("lat"),
            "lon":         r.get("lon"),
            "elapsed_s":   r.get("elapsed_s"),
        }
        if "error" in r:
            base["error"] = r["error"]
        else:
            rk = r["risk"]
            cl = r["classification"]
            base.update({
                "feasible":    r["feasible"],
                "risk_score":  rk["risk_score"],
                "risk_tier":   rk["risk_tier"],
                "dominant":    cl["dominant"],
                "slope_deg":   r.get("slope_deg"),
                "max_angle_buildings": cl.get("max_angle_buildings"),
                "canopy_max_m": r.get("canopy_max_m"),
                "building_count": r.get("building_count"),
                "data_src_terrain": r.get("data_sources", {}).get("terrain_far"),
                "warnings":    "; ".join(r.get("warnings", [])),
            })
        rows.append(base)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log.info(f"Results CSV saved: {out_path}")
    return df


def print_table(results: list[dict]):
    ok   = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]

    print("\n" + "─" * 105)
    print(
        f"{'#':>3}  {'Location':>12}  {'Lat':>9}  {'Lon':>10}  "
        f"{'Feas':^5}  {'Score':>5}  {'Tier':>8}  {'Dominant':^11}  "
        f"{'Slope°':>6}  {'Time(s)':>7}"
    )
    print("─" * 105)
    for r in results:
        if "error" in r:
            print(f"{r['idx']:>3}  {r['location_id']:>12}  "
                  f"{r['lat']:>9.5f}  {r['lon']:>10.5f}  "
                  f"{'ERR':^5}  {'—':>5}  {'—':>8}  {'—':^11}  "
                  f"{'—':>6}  {r['elapsed_s']:>7.1f}")
            continue
        rk = r["risk"]
        cl = r["classification"]
        print(
            f"{r['idx']:>3}  {r['location_id']:>12}  "
            f"{r['lat']:>9.5f}  {r['lon']:>10.5f}  "
            f"{'YES' if r['feasible'] else 'NO':^5}  "
            f"{rk['risk_score']:>5.1f}  "
            f"{rk['risk_tier']:>8}  "
            f"{cl['dominant']:^11}  "
            f"{r['slope_deg']:>6.1f}  "
            f"{r['elapsed_s']:>7.1f}"
        )
    print("─" * 105)

    # Timing stats
    times = [r["elapsed_s"] for r in results]
    scores_ok = [r["risk"]["risk_score"] for r in ok]
    print(f"\nTotal points: {len(results)}  |  OK: {len(ok)}  |  Errors: {len(errs)}")
    print(f"Time — mean: {np.mean(times):.1f}s  median: {np.median(times):.1f}s  "
          f"min: {np.min(times):.1f}s  max: {np.max(times):.1f}s  "
          f"total: {np.sum(times):.0f}s ({np.sum(times)/60:.1f} min)")
    if scores_ok:
        print(f"Risk score — mean: {np.mean(scores_ok):.1f}  "
              f"median: {np.median(scores_ok):.1f}  "
              f"min: {np.min(scores_ok):.1f}  max: {np.max(scores_ok):.1f}")
        n_feas = sum(r["feasible"] for r in ok)
        print(f"Feasible: {n_feas}/{len(ok)} ({100*n_feas/len(ok):.0f}%)")
        from collections import Counter
        dom_ct = Counter(r["classification"]["dominant"] for r in ok)
        print(f"Dominant: { {k: v for k, v in dom_ct.most_common()} }")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _style_map_ax(ax, title: str):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#374151")
    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)


def _style_ax_dark(ax, title: str):
    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#374151")
    ax.tick_params(colors="white")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    # 1. Load
    df = load_dataset()

    # 2. Boundaries
    log.info("Loading boundary data …")
    boundaries = load_boundaries()

    # 3. Map A — CONUS
    map_conus_overview(df, boundaries, OUTPUT_DIR / "mapA_conus_overview.png")

    # 4. Map B — NC density
    map_nc_density(df, boundaries, OUTPUT_DIR / "mapB_nc_density.png")

    # 5. Maps C — Regional clusters
    map_regional_clusters(df, OUTPUT_DIR / "mapC_regional_clusters.png")

    # 6. Sample 50 test points
    test_points = select_test_points(df, n_cols=10, n_rows=5)
    test_points.to_csv(OUTPUT_DIR / "test_points_50.csv", index=False)
    log.info(f"Test points saved: {OUTPUT_DIR / 'test_points_50.csv'}")

    # 7. Run pipeline
    log.info(f"\n{'#'*60}")
    log.info("Running pipeline on test points …")
    log.info(f"{'#'*60}\n")
    results = run_pipeline(test_points)

    # 8. Map D — Risk results
    map_risk_results(results, df, boundaries, OUTPUT_DIR / "mapD_risk_results.png")

    # 9. Timing summary chart
    plot_timing_summary(results, OUTPUT_DIR / "timing_summary.png")

    # 10. Save + print
    df_res = save_results_csv(results, OUTPUT_DIR / "challenge50_results.csv")
    print_table(results)

    log.info(f"\nAll done. Total wall time: {(time.time()-t_total)/60:.1f} min")
    log.info(f"Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
