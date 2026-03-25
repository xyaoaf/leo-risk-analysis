"""
test_nc_points.py — Quick pipeline test on 8 representative NC points
from DATA_CHALLENGE_50.csv (geographically spread across the state).

Usage:
    conda run -n cs378 python test_nc_points.py
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

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from feasibility import analyze_location
from main import _plot_point, _save_json, DOMINANT_COLORS, TIER_COLORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# 8 points sampled evenly across the 4.67M-row NC dataset
# (covers Piedmont, western foothills, coastal plain, eastern plain)
NC_TEST_POINTS = [
    (35.061343, -80.665707, "Charlotte-Suburban",    "40023115"),
    (35.630165, -80.465388, "Rowan-County",          "178869394"),
    (35.413664, -80.912646, "Lincoln-County",        "193007291"),
    (34.752810, -77.296012, "Jones-County-Coast",    "25713839"),
    (35.110129, -81.000655, "Gaston-County",         "40864332"),
    (35.429130, -78.739189, "Harnett-County",        "195510910"),
    (35.558051, -80.862340, "Iredell-County",        "40225238"),
    (35.922026, -80.066134, "Davidson-County",       "38089709"),
]

OUTPUT_DIR = ROOT / "outputs" / "nc_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    all_results = []

    for lat, lon, label, loc_id in NC_TEST_POINTS:
        log.info(f"\n{'='*60}")
        log.info(f"Point: {label}  ({lat}, {lon})  [id={loc_id}]")
        log.info(f"{'='*60}")

        t0 = time.time()
        try:
            result = analyze_location(lat, lon, run_local_search=False)
        except Exception as exc:
            import traceback
            log.error(f"FAILED: {exc}")
            traceback.print_exc()
            all_results.append({
                "label": label, "lat": lat, "lon": lon,
                "error": str(exc),
                "elapsed_s": round(time.time() - t0, 1),
            })
            continue

        result["label"] = label
        all_results.append(result)

        try:
            _plot_point(result, OUTPUT_DIR / f"point_{label}.png")
        except Exception as exc:
            log.warning(f"Plot failed for {label}: {exc}")

        _save_json(result, OUTPUT_DIR / f"point_{label}.json")

    _print_table(all_results)
    _plot_summary(all_results, OUTPUT_DIR / "nc_summary.png")
    log.info(f"\nOutputs in: {OUTPUT_DIR}")


def _print_table(results):
    print("\n" + "─" * 110)
    print(f"{'Label':<24} {'Lat':>9} {'Lon':>10}  {'Feas':^5}  "
          f"{'Score':>5}  {'Tier':>8}  {'Dominant':^11}  "
          f"{'Slope°':>6}  {'Canopy(m)':>9}  {'Bldgs':>5}  {'Time(s)':>7}")
    print("─" * 110)
    for r in results:
        if "error" in r:
            print(f"{'  ' + r['label']:<24} {r['lat']:>9.5f} {r['lon']:>10.5f}  "
                  f"{'ERR':^5}  {'—':>5}  {'—':>8}  {'—':^11}  "
                  f"{'—':>6}  {'—':>9}  {'—':>5}  {r['elapsed_s']:>7.1f}")
            continue
        rk = r["risk"]
        cl = r["classification"]
        print(
            f"  {r['label']:<22} {r['lat']:>9.5f} {r['lon']:>10.5f}  "
            f"{'YES' if r['feasible'] else 'NO ':^5}  "
            f"{rk['risk_score']:>5.1f}  "
            f"{rk['risk_tier']:>8}  "
            f"{cl['dominant']:^11}  "
            f"{r['slope_deg']:>6.1f}  "
            f"{r.get('canopy_max_m', '?'):>9}  "
            f"{r.get('building_count', '?'):>5}  "
            f"{r['elapsed_s']:>7.1f}"
        )
    print("─" * 110)

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    if ok:
        times = [r["elapsed_s"] for r in ok]
        scores = [r["risk"]["risk_score"] for r in ok]
        print(f"\nOK: {len(ok)}  Errors: {len(errs)}")
        print(f"Time  — mean: {np.mean(times):.1f}s  median: {np.median(times):.1f}s  "
              f"min: {np.min(times):.1f}s  max: {np.max(times):.1f}s  "
              f"total: {sum(times):.0f}s")
        print(f"Score — mean: {np.mean(scores):.1f}  min: {np.min(scores):.1f}  "
              f"max: {np.max(scores):.1f}")
        n_feas = sum(r["feasible"] for r in ok)
        print(f"Feasible: {n_feas}/{len(ok)}")
    print()


def _plot_summary(results, out_path):
    ok = [r for r in results if "error" not in r]
    if not ok:
        return

    labels = [r["label"].replace("-", "\n") for r in ok]
    scores = [r["risk"]["risk_score"] for r in ok]
    times  = [r["elapsed_s"] for r in ok]
    doms   = [r["classification"]["dominant"] for r in ok]
    feas   = [r["feasible"] for r in ok]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#111827")

    # Risk scores
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    colors = [DOMINANT_COLORS[d] for d in doms]
    bars = ax.barh(range(len(ok)), scores, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.4)
    for v in [20, 45, 70]:
        ax.axvline(v, color="#666", lw=0.8, ls="--", alpha=0.6)
    ax.set_yticks(range(len(ok)))
    ax.set_yticklabels(labels, color="white", fontsize=7)
    ax.set_xlabel("Risk score (0–100)", color="white", fontsize=8)
    ax.tick_params(colors="white")
    for i, (sc, f) in enumerate(zip(scores, feas)):
        ax.text(sc + 0.5, i, f"{sc:.0f}{'✓' if f else '✗'}",
                va="center", color="white", fontsize=7)
    ax.set_title("Risk Score (✓=feasible)", color="white", fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # Processing time
    ax = axes[1]
    ax.set_facecolor("#1a1a2e")
    ax.bar(range(len(ok)), times, color="#3b82f6", edgecolor="#1e40af", alpha=0.85)
    ax.axhline(np.mean(times), color="white", lw=1.2, ls="--",
               label=f"mean {np.mean(times):.1f}s")
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels(labels, color="white", fontsize=6)
    ax.set_ylabel("Elapsed (s)", color="white", fontsize=8)
    ax.tick_params(colors="white")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#555", labelcolor="white")
    ax.set_title("Processing Time per Point", color="white", fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # Obstruction breakdown
    ax = axes[2]
    ax.set_facecolor("#1a1a2e")
    t_vals = [r["classification"]["blocked_frac_terrain"] * 100 for r in ok]
    v_vals = [r["classification"]["canopy_contribution"]  * 100 for r in ok]
    b_vals = [r["classification"]["building_contribution"]* 100 for r in ok]
    x = np.arange(len(ok))
    w = 0.25
    ax.bar(x - w, t_vals, w, label="Terrain",    color="#8e6b3e", alpha=0.9)
    ax.bar(x,     v_vals, w, label="Vegetation", color="#27ae60", alpha=0.9)
    ax.bar(x + w, b_vals, w, label="Buildings",  color="#e74c3c", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=6)
    ax.set_ylabel("Blocked FOV %", color="white", fontsize=8)
    ax.tick_params(colors="white")
    ax.legend(fontsize=7, facecolor="#222", edgecolor="#555", labelcolor="white")
    ax.set_title("Obstruction Contributions", color="white", fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    fig.suptitle("LEO Risk Analysis — 8 NC Test Points  (DATA_CHALLENGE_50)",
                 color="white", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"Summary saved: {out_path}")


if __name__ == "__main__":
    main()
