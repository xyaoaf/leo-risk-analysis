"""
remake_maps.py — Regenerate Maps A, B, D with publication-ready style.

Fixes applied vs original explore_challenge50.py:
  Map A: NC masked by inset → side-by-side CONUS + NC zoom, no overlap
  Map B: Hexagons distorted → correct aspect ratio via geographic projection
  Map D: Points floating in space → overlaid on NC county boundaries

All maps: dark background → white/light publication style.

Usage:
    conda run -n cs378 python remake_maps.py
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter

ROOT       = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs" / "challenge50"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Publication style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})

RISK_CMAP = "RdYlGn_r"

TIER_COLORS = {
    "low":      "#2ecc71",
    "moderate": "#f39c12",
    "high":     "#e67e22",
    "critical": "#e74c3c",
}
DOM_COLORS = {
    "clear":      "#4ade80",
    "vegetation": "#16a34a",
    "building":   "#3b82f6",
    "terrain":    "#92400e",
}

# Approximate NC centre latitude for aspect correction
_NC_LAT = 35.5
_ASPECT  = 1.0 / np.cos(np.radians(_NC_LAT))   # ~1.22


# ═══════════════════════════════════════════════════════════════════════════════
# Boundary loader (cached)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch(url: str, cache: Path) -> gpd.GeoDataFrame:
    if not cache.exists():
        log.info(f"  Downloading {url.split('/')[-1]} …")
        with urllib.request.urlopen(url, timeout=60) as r:
            cache.write_bytes(r.read())
    return gpd.read_file(cache)


def load_boundaries() -> dict:
    bd = ROOT / "data" / "boundaries"
    bd.mkdir(parents=True, exist_ok=True)

    states = _fetch(
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_1_states_provinces.geojson",
        bd / "ne_110m_states.geojson",
    )
    us_states = states[states["iso_a2"] == "US"].copy()
    conus     = us_states[~us_states["name"].isin(["Alaska", "Hawaii"])]

    counties_path = bd / "us_counties_fips.geojson"
    if not counties_path.exists():
        log.info("  Downloading US counties GeoJSON …")
        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        with urllib.request.urlopen(url, timeout=60) as r:
            counties_path.write_bytes(r.read())
    all_counties = gpd.read_file(counties_path)
    all_counties["county_fips"] = all_counties["id"].astype(str).str.zfill(5)
    nc_counties = all_counties[all_counties["county_fips"].str.startswith("37")].copy()

    return {"conus": conus.to_crs("EPSG:4326"),
            "nc_counties": nc_counties.to_crs("EPSG:4326")}


# ═══════════════════════════════════════════════════════════════════════════════
# Map A — CONUS overview  (side-by-side, no inset masking NC)
# ═══════════════════════════════════════════════════════════════════════════════

def map_conus_overview(df: pd.DataFrame, boundaries: dict, out_path: Path):
    log.info("Generating Map A …")
    conus      = boundaries["conus"]
    nc_state   = conus[conus["name"] == "North Carolina"]
    other      = conus[conus["name"] != "North Carolina"]

    sample = df.iloc[::50][["latitude", "longitude"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("white")

    # ── Left: full CONUS ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#f0f4f8")
    other.plot(ax=ax, color="#dde5ef", edgecolor="#9ca3af", linewidth=0.5)
    nc_state.plot(ax=ax, color="#bfdbfe", edgecolor="#1d4ed8", linewidth=1.5)
    ax.scatter(
        sample["longitude"], sample["latitude"],
        s=0.05, c="#1d4ed8", alpha=0.25, linewidths=0, rasterized=True,
    )
    # Label NC with a clean annotation
    _nc_proj = nc_state.to_crs("EPSG:32617")
    nc_cx = float(_nc_proj.geometry.centroid.iloc[0].x)
    nc_cy = float(_nc_proj.geometry.centroid.iloc[0].y)
    # Convert back to WGS84 for annotation on geographic axes
    import pyproj
    _t = pyproj.Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True)
    nc_cx, nc_cy = _t.transform(nc_cx, nc_cy)
    ax.annotate(
        "North Carolina\n4.67 M addresses",
        xy=(nc_cx, nc_cy + 0.3),
        xytext=(nc_cx - 10, nc_cy - 6),
        fontsize=9, color="#1d4ed8", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.2),
        ha="center",
    )
    ax.set_xlim(-125, -66); ax.set_ylim(23, 50)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#9ca3af"); sp.set_linewidth(0.5)
    ax.set_title("Continental U.S. — All data in North Carolina (FIPS 37)",
                 fontsize=10, fontweight="bold", color="#1f2937", pad=8)

    # ── Right: NC zoom ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#f0f4f8")
    # Neighboring states for context
    neighbors = conus[conus["name"].isin(
        ["South Carolina", "Georgia", "Tennessee", "Virginia"])]
    neighbors.plot(ax=ax2, color="#dde5ef", edgecolor="#9ca3af", linewidth=0.5)
    nc_state.plot(ax=ax2, color="#bfdbfe", edgecolor="#1d4ed8", linewidth=1.8)
    ax2.scatter(
        sample["longitude"], sample["latitude"],
        s=0.2, c="#1d4ed8", alpha=0.35, linewidths=0, rasterized=True,
    )
    # NC bbox with a small buffer
    ax2.set_xlim(-85.0, -74.5); ax2.set_ylim(33.5, 37.0)
    ax2.set_aspect(_ASPECT)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_edgecolor("#9ca3af"); sp.set_linewidth(0.5)
    ax2.set_title("North Carolina — 4,674,917 addresses",
                  fontsize=10, fontweight="bold", color="#1f2937", pad=8)

    # Shared legend
    leg = mpatches.Patch(color="#bfdbfe", ec="#1d4ed8",
                         label=f"NC addresses ({len(df):,.0f} total)")
    axes[0].legend(handles=[leg], fontsize=8, loc="lower left",
                   facecolor="white", edgecolor="#d1d5db")

    fig.suptitle("DATA_CHALLENGE_50 — Geographic Coverage",
                 fontsize=12, fontweight="bold", color="#111827", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Map B — NC address density  (corrected aspect ratio)
# ═══════════════════════════════════════════════════════════════════════════════

def map_nc_density(df: pd.DataFrame, boundaries: dict, out_path: Path):
    log.info("Generating Map B …")
    nc_counties = boundaries["nc_counties"]

    # Project to UTM 17N for undistorted hex grid
    nc_proj = nc_counties.to_crs("EPSG:32617")
    pts = gpd.GeoDataFrame(
        df[["longitude", "latitude"]],
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32617")
    x_proj = pts.geometry.x.values
    y_proj = pts.geometry.y.values

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("white")

    # ── Left: hexbin density (projected, undistorted) ─────────────────────────
    ax = axes[0]
    ax.set_facecolor("#f8fafc")
    hb = ax.hexbin(
        x_proj, y_proj,
        gridsize=80, cmap="YlOrRd", bins="log",
        mincnt=1, linewidths=0.0,
    )
    nc_proj.plot(ax=ax, color="none", edgecolor="#374151", linewidth=0.4)
    cb = plt.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Address count (log₁₀ scale)", fontsize=8, color="#374151")
    cb.ax.tick_params(labelsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#d1d5db"); sp.set_linewidth(0.5)
    ax.set_title("Address Density — 4.67M Points (log scale per hex cell)",
                 fontsize=10, fontweight="bold", color="#1f2937", pad=8)
    ax.set_aspect("equal")

    # ── Right: county choropleth ──────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#f8fafc")
    county_counts = df.groupby("county_fips").size().rename("count").reset_index()
    nc_geo = nc_counties.copy()
    merged = nc_geo.merge(county_counts, on="county_fips", how="left")
    merged["count"] = merged["count"].fillna(0)
    merged.plot(
        column="count", ax=ax,
        cmap="YlOrRd", edgecolor="#6b7280", linewidth=0.4,
        legend=True,
        legend_kwds={
            "label": "Addresses per county",
            "shrink": 0.75, "orientation": "vertical",
            "format": FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"),
        },
    )
    ax.set_aspect(_ASPECT)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#d1d5db"); sp.set_linewidth(0.5)
    ax.set_title("Address Count per County (100 NC counties)",
                 fontsize=10, fontweight="bold", color="#1f2937", pad=8)

    fig.suptitle("North Carolina — DATA_CHALLENGE_50 Dataset Overview",
                 fontsize=12, fontweight="bold", color="#111827", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Map D — Risk results on NC map  (geographic context added)
# ═══════════════════════════════════════════════════════════════════════════════

def map_risk_results(results_csv: Path, boundaries: dict, out_path: Path):
    log.info("Generating Map D …")
    df = pd.read_csv(results_csv)
    ok = df.dropna(subset=["risk_score"]).copy()
    if ok.empty:
        log.warning("No valid results."); return

    nc_counties = boundaries["nc_counties"]

    lats   = ok["lat"].values
    lons   = ok["lon"].values
    scores = ok["risk_score"].values
    tiers  = ok["risk_tier"].values
    doms   = ok["dominant"].values
    feas   = ok["feasible"].values.astype(bool)

    # NC bounds with small buffer
    LON_LIM = (-84.5, -75.2)
    LAT_LIM = (33.7,  36.7)

    def _base(ax, title):
        """Draw NC county grid on ax."""
        ax.set_facecolor("#f8fafc")
        nc_counties.plot(ax=ax, color="#e8eef4", edgecolor="#9ca3af", linewidth=0.35)
        ax.set_xlim(*LON_LIM); ax.set_ylim(*LAT_LIM)
        ax.set_aspect(_ASPECT)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#d1d5db"); sp.set_linewidth(0.5)
        ax.set_title(title, fontsize=9, fontweight="bold", color="#1f2937", pad=6)

    def _infeasible_markers(ax):
        inf_lons = lons[~feas]
        inf_lats = lats[~feas]
        if len(inf_lons):
            ax.scatter(inf_lons, inf_lats, marker="x", c="#111827", s=60,
                       linewidths=1.8, zorder=7, label="Infeasible")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.patch.set_facecolor("white")

    # ── Panel 1: Risk score ───────────────────────────────────────────────────
    ax = axes[0]
    _base(ax, "Starlink Risk Score (0–100)")
    norm = Normalize(vmin=0, vmax=100)
    sc = ax.scatter(lons, lats, c=scores, s=90,
                    cmap=RISK_CMAP, norm=norm,
                    edgecolors="#374151", linewidths=0.5, zorder=5)
    _infeasible_markers(ax)
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, orientation="vertical")
    cb.set_label("Risk score", fontsize=8); cb.ax.tick_params(labelsize=7)
    for v, c, lbl in [(20, "#22c55e", "low"), (45, "#f59e0b", "moderate"),
                       (70, "#ef4444", "high/critical")]:
        cb.ax.axhline(v, color=c, lw=1.2, ls="--", alpha=0.85)

    # ── Panel 2: Dominant obstruction ─────────────────────────────────────────
    ax = axes[1]
    _base(ax, "Dominant Obstruction Type  (✗ = infeasible)")
    for dom_type, dom_col in DOM_COLORS.items():
        mask = doms == dom_type
        if mask.any():
            ax.scatter(lons[mask], lats[mask], c=dom_col, s=90,
                       edgecolors="#374151", linewidths=0.5,
                       label=dom_type.capitalize(), zorder=5)
    _infeasible_markers(ax)
    ax.legend(fontsize=7.5, loc="lower right",
              facecolor="white", edgecolor="#d1d5db",
              framealpha=0.9, markerscale=0.9)

    # ── Panel 3: Risk tier with feasibility summary ────────────────────────────
    ax = axes[2]
    _base(ax, "Risk Tier by Location")
    for tier, tc in TIER_COLORS.items():
        mask = tiers == tier
        if mask.any():
            ax.scatter(lons[mask], lats[mask], c=tc, s=90,
                       edgecolors="#374151", linewidths=0.5,
                       label=tier.capitalize(), zorder=5)
    _infeasible_markers(ax)

    n_feas = int(feas.sum())
    n_tot  = len(ok)
    tier_counts = pd.Series(tiers).value_counts()
    summary = "\n".join([
        f"n={n_tot}   feasible={n_feas} ({100*n_feas/n_tot:.0f}%)",
    ] + [f"  {t}: {tier_counts.get(t, 0)}" for t in TIER_COLORS])
    ax.text(0.02, 0.02, summary, transform=ax.transAxes,
            fontsize=7.5, va="bottom", color="#374151",
            bbox=dict(facecolor="white", edgecolor="#d1d5db",
                      boxstyle="round,pad=0.4", alpha=0.9))
    ax.legend(fontsize=7.5, loc="upper right",
              facecolor="white", edgecolor="#d1d5db",
              framealpha=0.9, markerscale=0.9)

    fig.suptitle(
        "LEO Risk Analysis — Pipeline Results: 37 NC Test Points  "
        "(DATA_CHALLENGE_50, 5×10 spatial grid)",
        fontsize=11, fontweight="bold", color="#111827", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Loading dataset …")
    df = pd.read_csv(ROOT / "DATA_CHALLENGE_50.csv", dtype={"geoid_cb": str})
    df["county_fips"] = df["geoid_cb"].str[:5]
    log.info(f"  {len(df):,} rows loaded")

    log.info("Loading boundaries …")
    bd = load_boundaries()

    map_conus_overview(df, bd, OUTPUT_DIR / "mapA_conus_overview.png")
    map_nc_density(df, bd,     OUTPUT_DIR / "mapB_nc_density.png")
    map_risk_results(
        OUTPUT_DIR / "challenge50_results.csv", bd,
        OUTPUT_DIR / "mapD_risk_results.png",
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
