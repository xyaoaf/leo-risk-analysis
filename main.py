"""
main.py — LEO Risk Analysis: multi-point feasibility + obstruction pipeline.

For each test point:
  1. Fetch terrain (far + near), canopy, buildings
  2. Compose obstruction surface
  3. Evaluate six hard constraints (logical AND → feasible flag)
  4. Compute three horizon profiles
  5. Classify dominant obstruction type
  6. Score risk (0–100)
  7. Find best nearby if infeasible or risk > 20
  8. Save per-point PNG + JSON

Usage:
    conda run -n cs378 python main.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource, Normalize
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.collections import PatchCollection

sys.path.insert(0, str(Path(__file__).parent / "src"))
from feasibility import analyze_location

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test points (Austin, TX area)
# ---------------------------------------------------------------------------
TEST_POINTS = [
    (30.2866988, -97.7112808, "Austin-Suburban"),
    (30.294267,  -97.700033,  "Austin-Urban-Core"),
    (30.279818,  -97.755716,  "Austin-West"),
    (30.304173,  -97.798148,  "Austin-Far-West"),
    (30.322219,  -97.774722,  "North-Austin"),
]

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DOMINANT_COLORS = {
    "clear":      "#2ecc71",
    "vegetation": "#27ae60",
    "building":   "#e74c3c",
    "terrain":    "#8e6b3e",
}
TIER_COLORS = {
    "low":      "#2ecc71",
    "moderate": "#f39c12",
    "high":     "#e67e22",
    "critical": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_results = []

    for lat, lon, label in TEST_POINTS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Point: {label}  ({lat}, {lon})")
        logger.info(f"{'='*60}")

        try:
            result = analyze_location(lat, lon, run_local_search=True)
        except Exception as exc:
            logger.error(f"[main] {label} FAILED: {exc}")
            import traceback; traceback.print_exc()
            continue

        result["label"] = label
        all_results.append(result)

        try:
            _plot_point(result, OUTPUT_DIR / f"point_{label}.png")
        except Exception as exc:
            logger.error(f"[main] Plot failed for {label}: {exc}")
            import traceback; traceback.print_exc()

        _save_json(result, OUTPUT_DIR / f"point_{label}.json")

    _plot_summary(all_results, OUTPUT_DIR / "summary.png")
    _print_table(all_results)


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def _print_table(results: list):
    print("\n" + "─" * 95)
    print(
        f"{'Label':<22} {'Feasible':^9} {'Dominant':^12} "
        f"{'Slope°':>6} {'MaxAng°':>7} {'Score':>6} {'Tier':>10} {'Warnings':>4}"
    )
    print("─" * 95)
    for r in results:
        c  = r["classification"]
        rk = r["risk"]
        warn_n = len(r["warnings"])
        feasible_s = "YES" if r["feasible"] else "NO "
        print(
            f"{r['label']:<22} "
            f"{feasible_s:^9} "
            f"{c['dominant']:^12} "
            f"{r['slope_deg']:>6.1f} "
            f"{c['max_angle_buildings']:>7.1f} "
            f"{rk['risk_score']:>6.1f} "
            f"{rk['risk_tier']:>10} "
            f"{'*'*warn_n:>4}"
        )
    print("─" * 95)

    # Constraint breakdown
    first = results[0] if results else None
    if first:
        print("\nConstraint summary (✓ = pass, ✗ = fail):")
        constraint_keys = list(first["constraints"].keys())
        header = f"{'Label':<22} " + "  ".join(f"{k[2:]:^12}" for k in constraint_keys)
        print(header)
        print("-" * len(header))
        for r in results:
            marks = "  ".join(
                f"{'✓':^12}" if r["constraints"][k]["pass"] else f"{'✗':^12}"
                for k in constraint_keys
            )
            print(f"{r['label']:<22} {marks}")
        print()


# ---------------------------------------------------------------------------
# Visualisation — per-point 5-panel figure
# ---------------------------------------------------------------------------

def _plot_point(result: dict, out_path: Path):
    fig = plt.figure(figsize=(25, 5), facecolor="#111827")
    gs  = fig.add_gridspec(1, 5, wspace=0.35)

    label   = result["label"]
    lat, lon = result["lat"], result["lon"]
    rk      = result["risk"]
    cl      = result["classification"]
    dom     = cl["dominant"]
    dom_col = DOMINANT_COLORS[dom]
    tier_col = TIER_COLORS[rk["risk_tier"]]

    canopy_cmap = mcolors.LinearSegmentedColormap.from_list(
        "canopy", ["#d4edda", "#52b788", "#1b4332"])

    def hillshade_norm(dem):
        filled = np.where(np.isnan(dem), np.nanmedian(dem), dem)
        hs  = LightSource(azdeg=315, altdeg=35).hillshade(filled, vert_exag=4)
        rng = np.ptp(filled) + 1e-9
        nrm = (filled - filled.min()) / rng
        return hs, nrm

    def ext(bbox):
        w, s, e, n = bbox
        return [w, e, s, n]

    # Reconstruct what we need from result for plotting
    # (we only have serialised data — re-fetch surface info from JSON)
    hz_far    = np.array(result["horizon"]["terrain_far"])
    hz_canopy = np.array(result["horizon"]["canopy"])
    hz_full   = np.array(result["horizon"]["full"])
    n_az      = result["horizon"]["n_azimuths"]

    _plot_terrain_panel(fig, gs[0], result, ext, hillshade_norm, lat, lon)
    _plot_near_panel(fig, gs[1], result, ext, hillshade_norm, canopy_cmap, lat, lon)
    _plot_constraints_panel(fig, gs[2], result)
    _plot_polar_panel(fig, gs[3], hz_far, hz_canopy, hz_full, n_az,
                      dom, dom_col, rk)
    _plot_suitability_panel(fig, gs[4], result, lat, lon)

    fig.suptitle(
        f"{label}  ({lat:.5f}, {lon:.5f})  |  "
        f"feasible={'YES' if result['feasible'] else 'NO'}  "
        f"risk={rk['risk_score']:.0f}/100 [{rk['risk_tier']}]  "
        f"dominant={dom.upper()}",
        color=tier_col, fontsize=9, y=1.01, fontweight="bold",
    )
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: {out_path}")

    # Also save individual panels as separate figures
    _save_individual_panels(result, out_path.parent)


PANEL_NAMES = ["terrain", "canopy_buildings", "constraints", "horizon", "neighborhood"]
PANEL_TITLES = [
    "Terrain (far-field, 1500 m)",
    "Canopy + Buildings (100 m)",
    "Feasibility Constraints",
    "Horizon Profile",
    "Neighbourhood Suitability",
]


def _save_individual_panels(result: dict, out_dir: Path):
    """Render and save each of the 5 panels as a standalone figure."""
    lat, lon = result["lat"], result["lon"]
    rk = result["risk"]
    cl = result["classification"]
    dom = cl["dominant"]
    dom_col = DOMINANT_COLORS[dom]

    canopy_cmap = mcolors.LinearSegmentedColormap.from_list(
        "canopy", ["#d4edda", "#52b788", "#1b4332"])

    def hillshade_norm(dem):
        filled = np.where(np.isnan(dem), np.nanmedian(dem), dem)
        hs = LightSource(azdeg=315, altdeg=35).hillshade(filled, vert_exag=4)
        rng = np.ptp(filled) + 1e-9
        nrm = (filled - filled.min()) / rng
        return hs, nrm

    def ext(bbox):
        w, s, e, n = bbox
        return [w, e, s, n]

    hz_far = np.array(result["horizon"]["terrain_far"])
    hz_canopy = np.array(result["horizon"]["canopy"])
    hz_full = np.array(result["horizon"]["full"])
    n_az = result["horizon"]["n_azimuths"]

    # Panel renderers: each creates a single-panel figure and saves it
    panel_fns = [
        lambda fig, gs: _plot_terrain_panel(fig, gs, result, ext, hillshade_norm, lat, lon),
        lambda fig, gs: _plot_near_panel(fig, gs, result, ext, hillshade_norm, canopy_cmap, lat, lon),
        lambda fig, gs: _plot_constraints_panel(fig, gs, result),
        lambda fig, gs: _plot_polar_panel(fig, gs, hz_far, hz_canopy, hz_full, n_az, dom, dom_col, rk),
        lambda fig, gs: _plot_suitability_panel(fig, gs, result, lat, lon),
    ]

    for name, title, fn in zip(PANEL_NAMES, PANEL_TITLES, panel_fns):
        try:
            fig = plt.figure(figsize=(5, 5), facecolor="#111827")
            gs = fig.add_gridspec(1, 1)
            fn(fig, gs[0])
            fig.suptitle(title, color="white", fontsize=9, y=0.98, fontweight="bold")
            panel_path = out_dir / f"panel_{name}.png"
            plt.savefig(panel_path, dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
        except Exception as exc:
            logger.warning(f"Panel {name} failed: {exc}")
            plt.close("all")


def _plot_terrain_panel(fig, gs_pos, result, ext_fn, hillshade_norm, lat, lon):
    """Far-field terrain panel — only if _terrain_far_array attached."""
    ax = fig.add_subplot(gs_pos, facecolor="#1a1a2e")
    arr = result.get("_terrain_far_array")
    bbox = result.get("_terrain_far_bbox")
    if arr is None:
        ax.text(0.5, 0.5, "Terrain\n(re-run to\ncache rasters)",
                ha="center", va="center", color="white", fontsize=7,
                transform=ax.transAxes)
        _style_ax(ax, "Terrain (far-field)")
        return
    hs, nrm = hillshade_norm(arr)
    e = ext_fn(bbox)
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=e,
              interpolation="bilinear", aspect="auto")
    ax.imshow(nrm, cmap="terrain", alpha=0.45, extent=e,
              interpolation="bilinear", aspect="auto")
    nb = result.get("_surf_bbox")
    if nb:
        w2, s2, e2, n2 = nb
        ax.add_patch(Rectangle((w2, s2), e2-w2, n2-s2,
                                lw=1.5, ec="#f59e0b", fc="none", ls="--", zorder=5))
    ax.plot(lon, lat, "w+", ms=10, mew=1.5, zorder=6)
    _style_ax(ax, f"Terrain far-field (1500 m)\nelev {np.nanmin(arr):.0f}–{np.nanmax(arr):.0f} m")


def _plot_near_panel(fig, gs_pos, result, ext_fn, hillshade_norm, canopy_cmap, lat, lon):
    """Near-field canopy + buildings panel."""
    ax = fig.add_subplot(gs_pos, facecolor="#1a1a2e")
    dem  = result.get("_surf_terrain")
    chm  = result.get("_surf_canopy")
    gdf  = result.get("_gdf_classified")
    bbox = result.get("_surf_bbox")
    if dem is None:
        ax.text(0.5, 0.5, "Canopy\n(re-run to\ncache rasters)",
                ha="center", va="center", color="white", fontsize=7,
                transform=ax.transAxes)
        _style_ax(ax, "Canopy + Buildings (100 m)")
        return
    hs, nrm = hillshade_norm(dem)
    e = ext_fn(bbox)
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=e,
              interpolation="bilinear", aspect="auto")
    ax.imshow(nrm, cmap="terrain", alpha=0.25, extent=e,
              interpolation="bilinear", aspect="auto")
    chm_m = np.ma.masked_where(chm < 0.5, chm)
    ax.imshow(chm_m, cmap=canopy_cmap, vmin=0, vmax=max(float(chm.max()), 1),
              alpha=0.7, extent=e, interpolation="bilinear", aspect="auto")
    if gdf is not None:
        _draw_buildings(ax, gdf)
    ax.plot(lon, lat, "w+", ms=10, mew=1.5, zorder=8)
    n_tot = len(gdf) if gdf is not None else 0
    n_blk = int(gdf["blocked"].sum()) if (gdf is not None and n_tot) else 0
    legend_h = [
        mpatches.Patch(fc="#3b82f6", ec="w", alpha=.7, label=f"Usable ({n_tot-n_blk})"),
        mpatches.Patch(fc="#f59e0b", ec="w", alpha=.7, label=f"Blocked ({n_blk})"),
        mpatches.Patch(fc="#52b788", ec="none", alpha=.7, label="Canopy"),
    ]
    ax.legend(handles=legend_h, loc="lower right", fontsize=6,
              facecolor="#222", edgecolor="#555", labelcolor="white")
    sim = "  [SIM]" if result.get("canopy_simulated") else ""
    _style_ax(ax, f"Canopy{sim} + Buildings (100 m)\nmax={result.get('canopy_max_m', '?')} m  blds={n_tot}")


def _plot_constraints_panel(fig, gs_pos, result):
    """Constraint checklist panel."""
    ax = fig.add_subplot(gs_pos, facecolor="#1a1a2e")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    C      = result["constraints"]
    rk     = result["risk"]
    keys   = list(C.keys())
    n      = len(keys)
    y_step = 0.12
    y0     = 0.90

    ax.text(0.5, 0.97, "Constraints", ha="center", va="top",
            color="white", fontsize=8, fontweight="bold",
            transform=ax.transAxes)

    for i, key in enumerate(keys):
        c   = C[key]
        y   = y0 - i * y_step
        ok  = c["pass"]
        sym = "✓" if ok else "✗"
        col = "#2ecc71" if ok else "#e74c3c"
        short = key[2:].replace("_", " ")
        ax.text(0.05, y, sym, color=col, fontsize=10, fontweight="bold",
                transform=ax.transAxes, va="center")
        ax.text(0.18, y, short, color="white", fontsize=6.5,
                transform=ax.transAxes, va="center")
        # Value annotation
        val_str = ""
        if "blocked_frac" in c:
            val_str = f"{c['blocked_frac']*100:.0f}%"
        elif "slope_deg" in c:
            val_str = f"{c['slope_deg']:.1f}°"
        elif "canopy_height_m" in c:
            val_str = f"{c['canopy_height_m']:.1f} m"
        if val_str:
            ax.text(0.78, y, val_str, color="#aaa", fontsize=6,
                    ha="right", transform=ax.transAxes, va="center")

    # Risk summary
    y_score = y0 - (n + 0.5) * y_step
    tier_col = TIER_COLORS[rk["risk_tier"]]
    ax.text(0.5, y_score, f"Risk: {rk['risk_score']:.0f}/100  [{rk['risk_tier'].upper()}]",
            ha="center", va="center", color=tier_col, fontsize=8,
            fontweight="bold", transform=ax.transAxes)

    # Warnings
    for j, w in enumerate(result.get("warnings", [])[:2]):
        ax.text(0.5, y_score - 0.10 * (j+1), f"⚠ {w[:50]}",
                ha="center", va="center", color="#f39c12", fontsize=5,
                wrap=True, transform=ax.transAxes)

    _style_ax(ax, "Feasibility Constraints")


def _plot_polar_panel(fig, gs_pos, hz_far, hz_canopy, hz_full, n_az,
                      dom, dom_col, rk):
    """Polar horizon profile panel with incremental fills per obstruction type."""
    ax = fig.add_subplot(gs_pos, polar=True, facecolor="#1a1a2e")

    angles_rad = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    # Convert compass angle (CW from N) → math angle (CCW from E) for polar plot
    theta = (np.pi / 2 - angles_rad) % (2 * np.pi)

    # Compute FOV-blocked percentages (FOV = ±50° around North, threshold = 25°)
    threshold_deg = 25.0
    step_deg      = 360.0 / n_az
    n_fov_half    = int(50.0 / step_deg)  # azimuths each side of North
    fov_idx       = list(range(n_fov_half + 1)) + list(range(n_az - n_fov_half, n_az))
    hz_far_fov    = hz_far[fov_idx]
    hz_can_fov    = hz_canopy[fov_idx]
    hz_full_fov   = hz_full[fov_idx]

    pct_terrain  = 100.0 * (hz_far_fov  > threshold_deg).mean()
    pct_canopy   = 100.0 * ((hz_can_fov  > threshold_deg) & (hz_far_fov  <= threshold_deg)).mean()
    pct_building = 100.0 * ((hz_full_fov > threshold_deg) & (hz_can_fov  <= threshold_deg)).mean()

    # --- Incremental fill helper ------------------------------------------------
    # Each layer fills only the *delta* above the previous layer, so colours
    # never overlap: terrain (brown) → canopy-delta (green) → building-delta (red).
    def _closed(arr):
        """Close an array by appending its first element."""
        return np.append(arr, arr[0])

    def _ring_polygon(theta_c, r_outer, r_inner):
        """
        Return (theta_poly, r_poly) forming a closed ring between r_inner and
        r_outer, traced forward along the outer and backward along the inner.
        """
        t = np.concatenate([theta_c, theta_c[::-1]])
        r = np.concatenate([r_outer, r_inner[::-1]])
        return t, r

    t_c       = _closed(theta)
    far_c     = np.clip(_closed(hz_far),    0, 90)
    can_top_c = np.clip(_closed(np.maximum(hz_far,    hz_canopy)), 0, 90)
    bld_top_c = np.clip(_closed(np.maximum(hz_canopy, hz_full)),   0, 90)

    # Layer 1: terrain (0 → hz_far, 1500 m radius)
    ax.fill(t_c, far_c, color="#8e6b3e", alpha=0.90,
            label=f"Terrain  {pct_terrain:.0f}%  (1500 m)")
    ax.plot(t_c, far_c, color="#a07840", lw=0.8)

    # Layer 2: canopy delta (hz_far → max(hz_far, hz_canopy), 100 m radius)
    t_poly, r_poly = _ring_polygon(t_c, can_top_c, far_c)
    ax.fill(t_poly, r_poly, color="#27ae60", alpha=0.90,
            label=f"Canopy Δ  {pct_canopy:.0f}%  (100 m)")
    ax.plot(t_c, can_top_c, color="#2ecc71", lw=0.8)

    # Layer 3: building delta (max(hz_far,hz_can) → max(hz_can,hz_full), 100 m radius)
    t_poly2, r_poly2 = _ring_polygon(t_c, bld_top_c, can_top_c)
    ax.fill(t_poly2, r_poly2, color="#e74c3c", alpha=0.90,
            label=f"Buildings Δ  {pct_building:.0f}%  (100 m)")
    ax.plot(t_c, bld_top_c, color="#e74c3c", lw=0.8)

    # 25° reference ring
    t_ring = np.linspace(0, 2 * np.pi, 200)
    ax.plot(t_ring, np.full_like(t_ring, threshold_deg), "w--", lw=0.8, alpha=0.6)
    ax.text(np.pi / 4, threshold_deg + 2, f"{threshold_deg:.0f}°",
            color="white", fontsize=6)

    # FOV arc shading (±50° around N)
    fov_az = np.linspace(-50, 50, 40)
    fov_th = (np.pi / 2 - np.radians(fov_az)) % (2 * np.pi)
    ax.fill_between(fov_th, 0, 4, color="#4a90d9", alpha=0.15)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, max(35, float(hz_full.max()) + 5))
    ax.set_yticks([10, 20, 25, 30])
    ax.tick_params(colors="white", labelsize=6)
    ax.set_facecolor("#1a1a2e")
    ax.spines["polar"].set_color("#555")
    ax.grid(color="#333", lw=0.4)
    ax.legend(loc="lower left", fontsize=5.5,
              facecolor="#222", edgecolor="#555", labelcolor="white",
              bbox_to_anchor=(-0.30, -0.18))
    ax.set_title(
        f"Horizon Profile\n{dom.upper()}  {rk['risk_score']:.0f}/100 [{rk['risk_tier']}]",
        color=dom_col, fontsize=8, pad=10, fontweight="bold",
    )


def _plot_suitability_panel(fig, gs_pos, result, lat, lon):
    """
    Suitability / neighbourhood panel.

    Shows:
      • Satellite tile background (Esri World Imagery via contextily, if available)
      • Neighbourhood search candidates coloured by risk score (green → red)
      • Best nearby point as a gold star
      • Queried origin as a white crosshair
      • Key metrics text box
    """
    ax = fig.add_subplot(gs_pos, facecolor="#1a1a2e")

    bbox = result.get("_surf_bbox")   # (west, south, east, north) in WGS84
    best_nearby = result.get("best_nearby")

    if bbox is None:
        ax.text(0.5, 0.5, "Suitability\n(re-run to\ncache rasters)",
                ha="center", va="center", color="white", fontsize=7,
                transform=ax.transAxes)
        _style_ax(ax, "Neighbourhood Suitability")
        return

    west, south, east, north = bbox
    # Expand slightly for visual breathing room
    pad_lon = (east  - west)  * 0.05
    pad_lat = (north - south) * 0.05
    ax.set_xlim(west  - pad_lon, east  + pad_lon)
    ax.set_ylim(south - pad_lat, north + pad_lat)

    # ── Satellite tile background ────────────────────────────────────────────
    sat_ok = False
    try:
        import contextily as ctx
        ctx.add_basemap(
            ax, crs="EPSG:4326",
            source=ctx.providers.Esri.WorldImagery,
            zoom="auto", attribution=False,
        )
        sat_ok = True
    except Exception:
        ax.set_facecolor("#1e293b")   # dark blue-grey fallback

    # ── Candidate scatter ────────────────────────────────────────────────────
    n_candidates = 0
    if best_nearby and best_nearby.get("candidates"):
        cands = best_nearby["candidates"]
        n_candidates = len(cands)
        c_lons   = np.array([c["lon"]        for c in cands])
        c_lats   = np.array([c["lat"]        for c in cands])
        c_scores = np.array([c["risk_score"] for c in cands], dtype=float)

        sc = ax.scatter(
            c_lons, c_lats,
            c=c_scores, cmap="RdYlGn_r", vmin=0, vmax=100,
            s=22, alpha=0.80, zorder=5,
            edgecolors="white", linewidths=0.4,
        )

        # Best nearby — gold star
        best = best_nearby.get("best")
        if best:
            ax.plot(best["lon"], best["lat"],
                    "*", color="#fbbf24", ms=13, zorder=7,
                    mec="white", mew=0.8,
                    label=f"Best ({best['risk_score']:.0f}/100)")

    # ── Origin crosshair ─────────────────────────────────────────────────────
    ax.plot(lon, lat, "w+", ms=13, mew=2.0, zorder=8)

    # ── Metrics text box ─────────────────────────────────────────────────────
    rk  = result["risk"]
    imp = (best_nearby or {}).get("improvement")
    lines = [f"Score: {rk['risk_score']:.0f}/100 [{rk['risk_tier'].upper()}]"]
    if imp:
        lines.append(
            f"Best Δ: −{imp['risk_delta']:.0f} pts  "
            f"@ {imp['distance_m']:.0f} m"
        )
        if imp.get("feasible_gained"):
            lines.append("→ becomes feasible")
    ax.text(
        0.03, 0.97, "\n".join(lines),
        transform=ax.transAxes,
        color="white", fontsize=6.0, va="top", ha="left",
        bbox=dict(fc="#111827", ec="#555", alpha=0.85, pad=2.5),
        zorder=9,
    )

    # ── Legend dot ───────────────────────────────────────────────────────────
    if n_candidates > 0:
        # Compact colourbar as a single legend entry is cleaner; skip full cb
        ax.plot([], [], "o", color="#27ae60", ms=5, label="Low risk")
        ax.plot([], [], "o", color="#e74c3c", ms=5, label="High risk")
        ax.legend(loc="lower right", fontsize=5.5,
                  facecolor="#222", edgecolor="#555", labelcolor="white",
                  framealpha=0.85)

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    sat_note = "" if sat_ok else " (no satellite tile)"
    ax.set_title(
        f"Neighbourhood ({n_candidates} candidates, 50 m){sat_note}",
        color="white", fontsize=7.0, pad=4,
    )


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def _plot_summary(results: list, out_path: Path):
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#111827")

    labels    = [r["label"] for r in results]
    t_vals    = [r["classification"]["blocked_frac_terrain"]   * 100 for r in results]
    v_vals    = [r["classification"]["canopy_contribution"]    * 100 for r in results]
    b_vals    = [r["classification"]["building_contribution"]  * 100 for r in results]
    max_angs  = [r["classification"]["max_angle_buildings"]         for r in results]
    scores    = [r["risk"]["risk_score"]                            for r in results]
    dominants = [r["classification"]["dominant"]                    for r in results]
    feasible  = [r["feasible"]                                      for r in results]

    x = np.arange(n)
    w = 0.25

    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    ax.bar(x - w, t_vals, w, label="Terrain",    color="#8e6b3e", alpha=0.9)
    ax.bar(x,     v_vals, w, label="Vegetation", color="#27ae60", alpha=0.9)
    ax.bar(x + w, b_vals, w, label="Buildings",  color="#e74c3c", alpha=0.9)
    ax.axhline(0, color="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("-", "\n") for l in labels],
                       color="white", fontsize=7)
    ax.set_ylabel("Blocked FOV fraction (%)", color="white", fontsize=8)
    ax.tick_params(colors="white")
    ax.legend(fontsize=7, facecolor="#222", edgecolor="#555", labelcolor="white")
    ax.set_title("Obstruction contributions per location",
                 color="white", fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    ax = axes[1]
    ax.set_facecolor("#1a1a2e")
    bar_colors = [DOMINANT_COLORS[d] for d in dominants]
    ax.barh(x, scores, color=bar_colors, alpha=0.85, edgecolor="white", lw=0.4)
    ax.axvline(20, color="#aaa", lw=1, ls=":", alpha=0.5)
    ax.axvline(45, color="#f39c12", lw=1, ls="--", alpha=0.7)
    ax.axvline(70, color="#e74c3c", lw=1, ls="--", alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, color="white", fontsize=8)
    ax.set_xlabel("Risk score (0–100)", color="white", fontsize=8)
    ax.tick_params(colors="white")
    for i, (sc, dom, feas) in enumerate(zip(scores, dominants, feasible)):
        feas_s = "✓" if feas else "✗"
        ax.text(sc + 0.5, i, f"{sc:.0f}  {feas_s}  [{dom}]",
                va="center", color="white", fontsize=7)
    ax.set_title("Risk score + feasibility",
                 color="white", fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    fig.suptitle("LEO Risk Analysis — Multi-Point Summary  (Austin TX)",
                 color="white", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Summary saved: {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_buildings(ax, gdf, alpha=0.65):
    if gdf is None or len(gdf) == 0:
        return
    for color, subset in [("#3b82f6", gdf[~gdf["blocked"]]),
                           ("#f59e0b",  gdf[ gdf["blocked"]])]:
        patches = []
        for _, row in subset.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                patches.append(MplPolygon(np.array(poly.exterior.coords), closed=True))
        if patches:
            ax.add_collection(PatchCollection(
                patches, fc=color, ec="white", lw=0.4, alpha=alpha, zorder=4))


def _style_ax(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.set_title(title, color="white", fontsize=7.5, pad=4)


def _save_json(result: dict, path: Path):
    """Serialise result to JSON (strip numpy arrays and non-serialisable objects)."""
    def _clean(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()
                    if not k.startswith("_")}      # skip _cached arrays
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    path.write_text(json.dumps(_clean(result), indent=2))
    logger.info(f"JSON saved: {path}")


if __name__ == "__main__":
    main()
