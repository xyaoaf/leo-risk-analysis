"""
batch_nc_analysis.py — Stratified batch risk analysis over NC.

Samples ~5 points per county (100 counties × 5 = ~500 pts), runs
analyze_location() in a multi-process pool, then produces:
  outputs/batch/batch_results.csv         — per-point results
  outputs/batch/batch_county_summary.csv  — per-county risk averages
  outputs/batch/map_county_risk.png       — choropleth: mean risk by county
  outputs/batch/map_risk_scatter.png      — scatter: all points colored by tier
  outputs/batch/chart_tier_dist.png       — bar chart: tier distribution

Usage:
    conda run -n cs378 python batch_nc_analysis.py [--workers N] [--pts-per-county N]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

ROOT       = Path(__file__).parent
CSV_PATH   = ROOT / "DATA_CHALLENGE_50.csv"
OUTPUT_DIR = ROOT / "outputs" / "batch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TIER_COLORS = {
    "low":      "#2ecc71",
    "moderate": "#f39c12",
    "high":     "#e67e22",
    "critical": "#e74c3c",
}
TIER_ORDER = ["low", "moderate", "high", "critical"]

_NC_LAT  = 35.5
_ASPECT  = 1.0 / np.cos(np.radians(_NC_LAT))


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling: ~N points per county, spatially spread within each county
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_sample(df: pd.DataFrame, pts_per_county: int = 5,
                      seed: int = 42) -> pd.DataFrame:
    """
    Pick pts_per_county points per NC county, spread across the county bbox
    (one per equal grid cell within the county extent).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for fips, group in df.groupby("county_fips"):
        n = len(group)
        if n == 0:
            continue
        # If county has fewer points than requested, take all of them
        k = min(pts_per_county, n)
        if k == n:
            rows.append(group)
            continue

        # Divide county bbox into a grid, pick closest real point per cell
        lat_lo, lat_hi = group["latitude"].min(),  group["latitude"].max()
        lon_lo, lon_hi = group["longitude"].min(), group["longitude"].max()

        # Figure out grid dimensions closest to pts_per_county
        cols = max(1, int(round(np.sqrt(k * (lon_hi - lon_lo + 1e-9) /
                                         (lat_hi - lat_lo + 1e-9)))))
        rws  = max(1, int(np.ceil(k / cols)))

        lat_edges = np.linspace(lat_lo, lat_hi, rws  + 1)
        lon_edges = np.linspace(lon_lo, lon_hi, cols + 1)

        selected = set()
        for i in range(rws):
            for j in range(cols):
                lat_c = (lat_edges[i] + lat_edges[i + 1]) / 2
                lon_c = (lon_edges[j] + lon_edges[j + 1]) / 2
                mask  = (
                    (group["latitude"]  >= lat_edges[i]) &
                    (group["latitude"]  <  lat_edges[i + 1]) &
                    (group["longitude"] >= lon_edges[j]) &
                    (group["longitude"] <  lon_edges[j + 1])
                )
                cell = group[mask]
                if len(cell) == 0:
                    continue
                dist2 = ((cell["latitude"]  - lat_c) ** 2 +
                         (cell["longitude"] - lon_c) ** 2)
                idx = dist2.idxmin()
                selected.add(idx)

        rows.append(group.loc[list(selected)])

    result = pd.concat(rows, ignore_index=True)
    log.info(
        f"Sampled {len(result)} points across "
        f"{result['county_fips'].nunique()} counties "
        f"(~{pts_per_county} pts/county)"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Worker (runs in separate process)
# ═══════════════════════════════════════════════════════════════════════════════

def _worker(args):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    idx, lat, lon, location_id, county_fips, start_delay_s = args
    # Stagger worker startup to avoid simultaneous bursts to the 3DEP WMS
    # endpoint, which returns errors when hit with many concurrent requests.
    if start_delay_s > 0:
        time.sleep(start_delay_s)
    # Ensure PROJ database is found in spawned worker processes
    import os
    if "PROJ_DATA" not in os.environ:
        os.environ["PROJ_DATA"] = "/opt/miniconda3/envs/cs378/share/proj"
    # Import inside worker — each process gets its own module state
    from feasibility import analyze_location

    t0 = time.time()
    try:
        res = analyze_location(lat, lon, run_local_search=False)
        rk  = res["risk"]
        cl  = res["classification"]
        return {
            "idx":             idx,
            "location_id":     location_id,
            "county_fips":     county_fips,
            "lat":             lat,
            "lon":             lon,
            "feasible":        res["feasible"],
            "failure_mode":    res.get("failure_mode", "unknown"),
            "mount_type":      res.get("mount_type", "ground"),
            "risk_score":      rk["risk_score"],
            "risk_tier":       rk["risk_tier"],
            "dominant":        cl["dominant"],
            "canopy_type":     res.get("canopy_type", "unknown"),
            "evergreen_frac":  res.get("evergreen_frac"),
            "slope_deg":       res.get("slope_deg"),
            "canopy_max_m":    res.get("canopy_max_m"),
            "building_count":  res.get("building_count"),
            "elapsed_s":       round(time.time() - t0, 1),
        }
    except Exception as exc:
        return {
            "idx":         idx,
            "location_id": location_id,
            "county_fips": county_fips,
            "lat":         lat,
            "lon":         lon,
            "error":       str(exc),
            "elapsed_s":   round(time.time() - t0, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_batch(sample: pd.DataFrame, n_workers: int = 2,
              incremental_csv: Path | None = None) -> list[dict]:
    # Default workers reduced from 6 → 2.  The USGS 3DEP WMS endpoint rejects
    # concurrent bursts; 2 workers limits simultaneous terrain requests to 4
    # (2 far + 2 near), which the API handles reliably.  Each worker is also
    # staggered by (slot * 8s) so the initial wave is spread over ~8 seconds.
    #
    # incremental_csv: if provided, each result is appended to this CSV
    # immediately after the worker completes — so a crash loses at most one
    # in-flight result rather than all completed work.
    tasks = [
        (i, float(row["latitude"]), float(row["longitude"]),
         str(row["location_id"]), str(row["county_fips"]),
         (i - 1) % n_workers * 8)   # start_delay_s: 0s or 8s per slot
        for i, (_, row) in enumerate(sample.iterrows(), 1)
    ]
    n = len(tasks)
    log.info(f"Running {n} points with {n_workers} workers …")

    # Open incremental CSV (append mode; write header only if file is new/empty)
    inc_file   = None
    inc_writer = None
    inc_header_written = False
    if incremental_csv is not None:
        incremental_csv.parent.mkdir(parents=True, exist_ok=True)
        inc_file = open(incremental_csv, "a", newline="")

    results  = [None] * n
    done     = 0
    t_start  = time.time()
    errors   = 0

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {pool.submit(_worker, t): t[0] - 1 for t in tasks}
            for fut in as_completed(future_to_idx):
                pos = future_to_idx[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    r = {"idx": pos + 1, "error": str(exc), "elapsed_s": 0}
                    errors += 1
                results[pos] = r
                done += 1

                if "error" in r:
                    errors += 1
                    log.warning(f"[{done}/{n}] #{r['idx']} ERROR: {r['error']}")
                else:
                    elapsed_wall = time.time() - t_start
                    rate = done / elapsed_wall
                    eta  = (n - done) / rate if rate > 0 else 0
                    log.info(
                        f"[{done}/{n}] #{r['idx']} "
                        f"risk={r['risk_score']:.0f} [{r['risk_tier']}] "
                        f"dom={r['dominant']}  {r['elapsed_s']:.1f}s  "
                        f"ETA {eta/60:.1f}min"
                    )

                # Incremental write: flush result to CSV immediately
                if inc_file is not None:
                    import csv as _csv
                    if inc_writer is None:
                        inc_writer = _csv.DictWriter(
                            inc_file, fieldnames=list(r.keys()),
                            extrasaction="ignore",
                        )
                        # Write header only if file was empty before this run
                        if inc_file.tell() == 0 or not inc_header_written:
                            inc_writer.writeheader()
                            inc_header_written = True
                    inc_writer.writerow(r)
                    inc_file.flush()
    finally:
        if inc_file is not None:
            inc_file.close()

    ok = [r for r in results if r and "error" not in r]
    log.info(
        f"\nBatch complete: {len(ok)}/{n} OK  {errors} errors  "
        f"wall={time.time()-t_start:.0f}s"
    )
    return [r for r in results if r]


# ═══════════════════════════════════════════════════════════════════════════════
# County summary
# ═══════════════════════════════════════════════════════════════════════════════

def county_summary(results: list[dict]) -> pd.DataFrame:
    ok = pd.DataFrame([r for r in results if "error" not in r])
    if ok.empty:
        return pd.DataFrame()
    grp = ok.groupby("county_fips").agg(
        n_points=("risk_score", "count"),
        mean_risk=("risk_score", "mean"),
        pct_feasible=("feasible", "mean"),
        pct_critical=("risk_tier", lambda s: (s == "critical").mean()),
        pct_high=("risk_tier",    lambda s: (s == "high").mean()),
        dominant_mode=("dominant", lambda s: s.value_counts().index[0]),
    ).reset_index()
    grp["mean_risk"] = grp["mean_risk"].round(1)
    grp["pct_feasible"]  = (grp["pct_feasible"] * 100).round(1)
    grp["pct_critical"]  = (grp["pct_critical"] * 100).round(1)
    grp["pct_high"]      = (grp["pct_high"]     * 100).round(1)
    return grp


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_nc_counties() -> gpd.GeoDataFrame:
    bd = ROOT / "data" / "boundaries"
    bd.mkdir(parents=True, exist_ok=True)
    fp = bd / "us_counties_fips.geojson"
    if not fp.exists():
        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        with urllib.request.urlopen(url, timeout=60) as r:
            fp.write_bytes(r.read())
    gdf = gpd.read_file(fp)
    gdf["county_fips"] = gdf["id"].astype(str).str.zfill(5)
    nc = gdf[gdf["county_fips"].str.startswith("37")].copy()
    # GeoJSON is already WGS84; skip to_crs to avoid pyproj PROJ-DB issues
    if nc.crs is None:
        nc = nc.set_crs("EPSG:4326")
    elif nc.crs.to_epsg() != 4326:
        nc = nc.to_crs("EPSG:4326")
    return nc


LON_LIM = (-84.5, -75.2)
LAT_LIM = (33.7,  36.7)


def _base_nc(ax, nc_counties, title=""):
    ax.set_facecolor("#f8fafc")
    nc_counties.plot(ax=ax, color="#e8eef4", edgecolor="#9ca3af", linewidth=0.35)
    ax.set_xlim(*LON_LIM); ax.set_ylim(*LAT_LIM)
    ax.set_aspect(_ASPECT)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#d1d5db"); sp.set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", color="#1f2937", pad=6)


def map_county_risk(summary: pd.DataFrame, nc_counties: gpd.GeoDataFrame,
                    out_path: Path):
    """Choropleth: mean risk score per county."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
    fig.patch.set_facecolor("white")
    plt.rcParams.update({"font.family": "sans-serif"})

    merged = nc_counties.merge(summary, on="county_fips", how="left")

    # Left: mean risk score
    ax = axes[0]
    ax.set_facecolor("#f8fafc")
    merged.plot(column="mean_risk", ax=ax,
                cmap="RdYlGn_r", vmin=0, vmax=100,
                edgecolor="#6b7280", linewidth=0.35,
                legend=True,
                legend_kwds={"label": "Mean risk score (0–100)",
                             "shrink": 0.75, "orientation": "vertical"})
    ax.set_xlim(*LON_LIM); ax.set_ylim(*LAT_LIM)
    ax.set_aspect(_ASPECT)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#d1d5db"); sp.set_linewidth(0.5)
    ax.set_title("Mean Starlink Risk Score by County",
                 fontsize=10, fontweight="bold", color="#1f2937", pad=8)

    # Right: % infeasible (100 - pct_feasible)
    ax = axes[1]
    ax.set_facecolor("#f8fafc")
    merged["pct_infeasible"] = 100 - merged["pct_feasible"].fillna(50)
    merged.plot(column="pct_infeasible", ax=ax,
                cmap="Reds", vmin=0, vmax=100,
                edgecolor="#6b7280", linewidth=0.35,
                legend=True,
                legend_kwds={"label": "% Infeasible locations",
                             "shrink": 0.75, "orientation": "vertical",
                             "format": FuncFormatter(lambda x, _: f"{x:.0f}%")})
    ax.set_xlim(*LON_LIM); ax.set_ylim(*LAT_LIM)
    ax.set_aspect(_ASPECT)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#d1d5db"); sp.set_linewidth(0.5)
    ax.set_title("% Infeasible Locations by County",
                 fontsize=10, fontweight="bold", color="#1f2937", pad=8)

    fig.suptitle("NC Starlink Risk Analysis — County-Level Summary",
                 fontsize=12, fontweight="bold", color="#111827", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


def map_risk_scatter(results: list[dict], nc_counties: gpd.GeoDataFrame,
                     out_path: Path):
    """Scatter: all tested points colored by risk tier."""
    ok = [r for r in results if "error" not in r]
    if not ok:
        return

    lats   = [r["lat"]        for r in ok]
    lons   = [r["lon"]        for r in ok]
    scores = [r["risk_score"] for r in ok]
    tiers  = [r["risk_tier"]  for r in ok]
    feas   = [r["feasible"]   for r in ok]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
    fig.patch.set_facecolor("white")

    # Left: risk score gradient
    ax = axes[0]
    _base_nc(ax, nc_counties, "Starlink Risk Score — All Tested Locations")
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    norm = Normalize(vmin=0, vmax=100)
    sc = ax.scatter(lons, lats, c=scores, s=25,
                    cmap="RdYlGn_r", norm=norm,
                    edgecolors="#374151", linewidths=0.3, zorder=5, alpha=0.85)
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Risk score (0–100)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # Infeasible X
    inf_lons = [lo for lo, f in zip(lons, feas) if not f]
    inf_lats = [la for la, f in zip(lats, feas) if not f]
    if inf_lons:
        ax.scatter(inf_lons, inf_lats, marker="x", c="#111827", s=35,
                   linewidths=1.4, zorder=7)

    # Right: risk tier dots
    ax = axes[1]
    _base_nc(ax, nc_counties, "Risk Tier by Location  (✗ = infeasible)")
    for tier in TIER_ORDER:
        t_lons = [lo for lo, t in zip(lons, tiers) if t == tier]
        t_lats = [la for la, t in zip(lats, tiers) if t == tier]
        if t_lons:
            ax.scatter(t_lons, t_lats, c=TIER_COLORS[tier], s=25,
                       edgecolors="#374151", linewidths=0.3,
                       label=tier.capitalize(), zorder=5, alpha=0.85)
    if inf_lons:
        ax.scatter(inf_lons, inf_lats, marker="x", c="#111827", s=35,
                   linewidths=1.4, zorder=7, label="Infeasible")

    # Summary stats box
    n_ok   = len(ok)
    n_feas = sum(feas)
    tier_ct = pd.Series(tiers).value_counts()
    lines = [f"n={n_ok}  feasible={n_feas} ({100*n_feas/n_ok:.0f}%)"] + \
            [f"  {t}: {tier_ct.get(t,0)} ({100*tier_ct.get(t,0)/n_ok:.0f}%)"
             for t in TIER_ORDER]
    ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
            fontsize=7.5, va="bottom", color="#374151",
            bbox=dict(facecolor="white", edgecolor="#d1d5db",
                      boxstyle="round,pad=0.4", alpha=0.92))
    ax.legend(fontsize=7.5, loc="upper right",
              facecolor="white", edgecolor="#d1d5db", framealpha=0.9)

    fig.suptitle(
        f"NC LEO Risk Analysis — {n_ok} Stratified Locations "
        f"(~5 per county, {len(set(r['county_fips'] for r in ok))} counties)",
        fontsize=11, fontweight="bold", color="#111827", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


def chart_tier_distribution(results: list[dict], out_path: Path):
    """Bar chart + pie: tier distribution and failure mode breakdown."""
    ok = [r for r in results if "error" not in r]
    if not ok:
        return

    scores  = [r["risk_score"]   for r in ok]
    tiers   = [r["risk_tier"]    for r in ok]
    fmodes  = [r.get("failure_mode", "unknown") for r in ok]
    doms    = [r["dominant"]     for r in ok]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("white")
    plt.rcParams.update({"font.family": "sans-serif"})

    n = len(ok)

    # ── Panel 1: risk score histogram ─────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("white")
    # Color each bar by its tier
    bins = np.linspace(0, 100, 21)
    counts, edges = np.histogram(scores, bins=bins)
    for i, (cnt, left) in enumerate(zip(counts, edges[:-1])):
        mid = (left + edges[i + 1]) / 2
        if   mid < 20: c = TIER_COLORS["low"]
        elif mid < 45: c = TIER_COLORS["moderate"]
        elif mid < 70: c = TIER_COLORS["high"]
        else:          c = TIER_COLORS["critical"]
        ax.bar(left, cnt, width=(edges[1] - edges[0]) * 0.95,
               color=c, edgecolor="white", linewidth=0.3, align="edge")
    ax.axvline(np.mean(scores),   color="#374151", lw=1.5, ls="--",
               label=f"Mean {np.mean(scores):.1f}")
    ax.axvline(np.median(scores), color="#6b7280", lw=1.2, ls=":",
               label=f"Median {np.median(scores):.1f}")
    for v, lbl in [(20, "low"), (45, "mod"), (70, "high")]:
        ax.axvline(v, color="#9ca3af", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Risk score (0–100)", fontsize=9)
    ax.set_ylabel("Number of locations", fontsize=9)
    ax.legend(fontsize=8, facecolor="white", edgecolor="#d1d5db")
    ax.set_title("Risk Score Distribution", fontsize=10, fontweight="bold",
                 color="#1f2937")
    patches = [mpatches.Patch(color=c, label=t.capitalize())
               for t, c in TIER_COLORS.items()]
    ax.legend(handles=patches, fontsize=8, loc="upper center",
              facecolor="white", edgecolor="#d1d5db")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # ── Panel 2: tier bar chart ────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("white")
    tier_ct = pd.Series(tiers).value_counts().reindex(TIER_ORDER, fill_value=0)
    bars = ax.bar(tier_ct.index, tier_ct.values,
                  color=[TIER_COLORS[t] for t in tier_ct.index],
                  edgecolor="#374151", linewidth=0.5, width=0.6)
    for bar, cnt in zip(bars, tier_ct.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{cnt}\n({100*cnt/n:.0f}%)",
                ha="center", va="bottom", fontsize=8, color="#374151")
    ax.set_xlabel("Risk tier", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(f"Tier Distribution  (n={n}, feasible={sum(r['feasible'] for r in ok)})",
                 fontsize=10, fontweight="bold", color="#1f2937")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # ── Panel 3: failure mode / dominant obstruction ──────────────────────────
    ax = axes[2]
    ax.set_facecolor("white")
    fm_ct = pd.Series(fmodes).value_counts()
    fm_colors = {
        "feasible":         "#2ecc71",
        "local_canopy":     "#16a34a",
        "local_building":   "#3b82f6",
        "local_terrain":    "#92400e",
        "regional_terrain": "#78350f",
        "mixed":            "#f59e0b",
        "roof_unusable":    "#a78bfa",
        "unknown":          "#9ca3af",
    }
    wedge_cols = [fm_colors.get(k, "#9ca3af") for k in fm_ct.index]
    wedges, texts, autotexts = ax.pie(
        fm_ct.values,
        labels=[k.replace("_", "\n") for k in fm_ct.index],
        colors=wedge_cols,
        autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 0.8},
        textprops={"fontsize": 8, "color": "#1f2937"},
    )
    for at in autotexts:
        at.set_fontsize(7.5); at.set_color("white")
    ax.set_title("Failure Mode Breakdown", fontsize=10, fontweight="bold",
                 color="#1f2937")

    fig.suptitle("NC Starlink Risk Analysis — Statistical Summary",
                 fontsize=12, fontweight="bold", color="#111827", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stratified batch LEO risk analysis over NC address points.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full NC batch (~594 pts, ~2-3 hrs)
  conda run -n cs378 python batch_nc_analysis.py

  # Dry-run: first 20 points to validate pipeline
  conda run -n cs378 python batch_nc_analysis.py --max-points 20

  # Custom AOI subset (from aoi_screen.py output)
  conda run -n cs378 python batch_nc_analysis.py --input-csv outputs/aoi/charlotte_points.csv

  # Regenerate maps from existing results without re-running pipeline
  conda run -n cs378 python batch_nc_analysis.py --skip-pipeline
""")
    parser.add_argument("--workers",        type=int, default=2,
                        help="Parallel workers (default 2; max recommended 2 to avoid "
                             "3DEP rate-limiting)")
    parser.add_argument("--pts-per-county", type=int, default=5,
                        help="Points per county in stratified sample (default 5, "
                             "ignored when --input-csv is given)")
    parser.add_argument("--max-points",     type=int, default=None,
                        help="Limit total points processed (dry-run safety valve). "
                             "E.g. --max-points 20 for a quick validation run.")
    parser.add_argument("--input-csv",      default=None,
                        help="Path to a pre-filtered points CSV (e.g. from aoi_screen.py). "
                             "Must have columns: location_id, latitude, longitude, "
                             "geoid_cb, county_fips.  Skips stratified sampling.")
    parser.add_argument("--skip-pipeline",  action="store_true",
                        help="Skip pipeline entirely; load existing results CSV and "
                             "regenerate maps only.")
    args = parser.parse_args()

    results_csv = OUTPUT_DIR / "batch_results.csv"

    if args.skip_pipeline and results_csv.exists():
        log.info(f"Loading existing results from {results_csv}")
        df_res = pd.read_csv(results_csv)
        results = df_res.to_dict("records")
    else:
        if args.input_csv:
            log.info(f"Loading custom input CSV: {args.input_csv}")
            sample = pd.read_csv(args.input_csv, dtype={"geoid_cb": str,
                                                         "county_fips": str})
            # Ensure county_fips exists
            if "county_fips" not in sample.columns and "geoid_cb" in sample.columns:
                sample["county_fips"] = sample["geoid_cb"].str[:5]
            log.info(f"  {len(sample):,} rows | {sample['county_fips'].nunique()} counties")
        else:
            log.info("Loading DATA_CHALLENGE_50.csv …")
            df = pd.read_csv(CSV_PATH, dtype={"geoid_cb": str})
            df["county_fips"] = df["geoid_cb"].str[:5]
            log.info(f"  {len(df):,} rows | {df['county_fips'].nunique()} counties")
            sample = stratified_sample(df, pts_per_county=args.pts_per_county)

        if args.max_points is not None and args.max_points < len(sample):
            log.info(f"--max-points {args.max_points}: truncating sample from "
                     f"{len(sample)} → {args.max_points} points (dry-run mode)")
            sample = sample.head(args.max_points)

        sample.to_csv(OUTPUT_DIR / "batch_sample_points.csv", index=False)

        # Incremental CSV path: results written row-by-row as each worker
        # completes so a crash loses at most one in-flight result.
        incremental_csv = OUTPUT_DIR / "batch_results_incremental.csv"
        if incremental_csv.exists():
            incremental_csv.unlink()  # start fresh for this run

        results = run_batch(sample, n_workers=args.workers,
                            incremental_csv=incremental_csv)

        # Save final consolidated CSV (sorted by original sample order)
        pd.DataFrame(results).to_csv(results_csv, index=False)
        log.info(f"Results saved → {results_csv}")

    # Summary stats
    ok = [r for r in results if "error" not in r]
    if ok:
        scores = [r["risk_score"] for r in ok]
        tiers  = pd.Series([r["risk_tier"] for r in ok]).value_counts()
        log.info(
            f"\n{'─'*50}\n"
            f"Total: {len(results)}  OK: {len(ok)}  "
            f"Errors: {len(results)-len(ok)}\n"
            f"Mean risk: {np.mean(scores):.1f}  Median: {np.median(scores):.1f}\n"
            f"Feasible: {sum(r['feasible'] for r in ok)}/{len(ok)} "
            f"({100*sum(r['feasible'] for r in ok)/len(ok):.0f}%)\n"
            f"Tiers: {tiers.to_dict()}\n{'─'*50}"
        )

    # County summary
    summary = county_summary(results)
    if not summary.empty:
        summary.to_csv(OUTPUT_DIR / "batch_county_summary.csv", index=False)
        log.info(f"County summary → {OUTPUT_DIR}/batch_county_summary.csv")

    # Maps
    log.info("Generating maps …")
    nc_counties = _load_nc_counties()
    if not summary.empty:
        map_county_risk(summary, nc_counties,
                        OUTPUT_DIR / "map_county_risk.png")
    map_risk_scatter(results, nc_counties,
                     OUTPUT_DIR / "map_risk_scatter.png")
    chart_tier_distribution(results,
                            OUTPUT_DIR / "chart_tier_dist.png")
    log.info(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
