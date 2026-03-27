"""
run_analysis.py — Single-point Starlink LEO risk diagnostic.

Runs the full geospatial pipeline for one lat/lon pair and produces:
  - A formatted text report in the terminal
  - A 5-panel PNG (terrain hillshade · canopy+buildings · constraints · polar horizon · suitability map)
  - A full JSON result

Usage
-----
    conda run -n cs378 python run_analysis.py LAT LON [options]

Examples
--------
    # Charlotte, NC — dense urban canopy
    conda run -n cs378 python run_analysis.py 35.061 -80.666 --id charlotte

    # Coastal NC — low risk, flat farmland
    conda run -n cs378 python run_analysis.py 34.752 -77.296 --id jones_county

    # Use a location ID from the challenge dataset
    conda run -n cs378 python run_analysis.py 35.912 -79.326 --id 37728629

    # Skip neighbourhood search (faster)
    conda run -n cs378 python run_analysis.py 35.061 -80.666 --id charlotte --no-search

Options
-------
    --id TEXT        Identifier used in output filenames and the report header.
                     Defaults to "lat_lon" if omitted.
    --no-search      Skip the 50 m neighbourhood best-point search.
    --out-dir DIR    Root output directory (default: outputs/reports).

Outputs
-------
    {out_dir}/{id}/{id}.png    — 5-panel visual diagnostic
    {out_dir}/{id}/{id}.json   — full structured result (all fields)

Implementation notes
--------------------
    - Visualization reuses _plot_point() from main.py (identical 5-panel layout).
    - JSON serialization reuses _save_json() from main.py.
    - Both single-point and batch modes call the same analyze_location() pipeline;
      this script adds only the CLI wrapper and the formatted text report.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from feasibility import analyze_location

# Reuse visualization and JSON helpers from main.py
import main as _main_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Risk-tier display config ─────────────────────────────────────────────────
_TIER_SYMBOLS = {
    "low":      "✓  LOW",
    "moderate": "⚡  MODERATE",
    "high":     "⚠  HIGH",
    "critical": "✗  CRITICAL",
}
_TIER_WIDTH = 30   # for separator lines


# ═══════════════════════════════════════════════════════════════════════════════
# Text report
# ═══════════════════════════════════════════════════════════════════════════════

def _print_report(result: dict, location_id: str, out_dir: Path):
    rk  = result["risk"]
    cl  = result["classification"]
    C   = result["constraints"]
    ds  = result["data_sources"]
    nb  = result.get("best_nearby")
    feas = result["feasible"]

    sep  = "═" * 62
    thin = "─" * 62
    tier_label = _TIER_SYMBOLS.get(rk["risk_tier"], rk["risk_tier"].upper())

    print(f"\n{sep}")
    print(f"  LEO SATELLITE RISK DIAGNOSTIC REPORT")
    print(f"{sep}")
    print(f"  Location : {location_id}  ({result['lat']:.5f}, {result['lon']:.5f})")
    print(f"  Run at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Elapsed  : {result['elapsed_s']:.1f} s")
    print(thin)

    feasible_str = "YES" if feas else f"NO  [failure: {result.get('failure_mode','?')}]"
    print(f"  RESULT   :  {tier_label}  (risk = {rk['risk_score']:.0f}/100)")
    print(f"  Feasible :  {feasible_str}")
    print(f"  Dominant :  {cl['dominant'].upper()}")
    if result.get("on_building"):
        print(f"  On bldg  :  YES  (dish elevation {result['dish_height_asl_m']:.1f} m ASL)")
    print(thin)

    # Constraints
    print("  CONSTRAINTS")
    _cline = lambda label, ok, val: (
        f"  {'✓' if ok else '✗'}  {label:<30} {val}"
    )
    t_l = C["C_terrain_large"]
    t_s = C["C_terrain_small"]
    v_p = C["C_veg_at_point"]
    veg = C["C_vegetation"]
    bld = C["C_building_nearby"]
    rf  = C["C_roof_usable"]

    print(_cline("Terrain horizon  ≤10% FOV",
                 t_l["pass"],
                 f"blocked={t_l['blocked_frac']*100:.0f}%  max={t_l['max_angle_deg']:.1f}°"))
    print(_cline("Local slope      <20°",
                 t_s["pass"],
                 f"slope={t_s['slope_deg']:.1f}°"))
    print(_cline(f"Canopy at dish   <{v_p['threshold']:.0f} m",
                 v_p["pass"],
                 f"canopy={v_p['canopy_height_m']:.1f} m"))
    print(_cline("Vegetation FOV   ≤15%",
                 veg["pass"],
                 f"blocked={veg['blocked_frac']*100:.0f}%  max={veg['max_angle_deg']:.1f}°"))
    print(_cline("Buildings FOV    ≤5%",
                 bld["pass"],
                 f"blocked={bld['blocked_frac']*100:.0f}%  max={bld['max_angle_deg']:.1f}°"))
    if rf["applicable"]:
        print(_cline("Roof usable",
                     rf["pass"],
                     "YES" if rf["roof_usable"] else "NO (vegetation overhang)"))
    print(thin)

    # Obstruction attribution
    print("  OBSTRUCTION BREAKDOWN (fraction of FOV)")
    print(f"     Terrain    : {cl['blocked_frac_terrain']*100:.1f}%")
    print(f"     Vegetation : {cl['canopy_contribution']*100:.1f}%")
    print(f"     Buildings  : {cl['building_contribution']*100:.1f}%")
    print(f"     Max angle  : {cl['max_angle_buildings']:.1f}°  (threshold 25°)")
    print(thin)

    # Data sources
    print("  DATA SOURCES")
    print(f"     Terrain (far)  : {ds.get('terrain_far','?')}")
    print(f"     Terrain (near) : {ds.get('terrain_near','?')}")
    print(f"     Canopy         : {ds.get('canopy','?')}")
    print(f"     Buildings      : {ds.get('buildings','?')}")
    if result.get("canopy_simulated"):
        print("     ⚠  Canopy data is SIMULATED")
    print(thin)

    # Warnings
    if result.get("warnings"):
        print("  WARNINGS")
        for w in result["warnings"]:
            print(f"     ⚠  {w}")
        print(thin)

    # Neighbourhood search
    if nb is None:
        print("  NEIGHBOURHOOD SEARCH  (skipped or not triggered)")
    elif nb.get("best") is None:
        print("  NEIGHBOURHOOD SEARCH  — no improvement found within 50 m")
    else:
        b = nb["best"]
        imp = nb["improvement"]
        print("  NEIGHBOURHOOD SEARCH  — better location found")
        print(f"     Candidate : ({b['lat']:.5f}, {b['lon']:.5f})")
        print(f"     Distance  : {imp['distance_m']:.0f} m")
        print(f"     Risk      : {b['risk_score']:.0f}/100  [{b['risk_tier']}]  "
              f"(Δ = −{imp['risk_delta']:.0f})")
        if imp["feasible_gained"]:
            print("     ✓  Location becomes feasible at the candidate point")
        print(f"     Note      : {imp['explanation']}")
    print(thin)

    # Output files
    print("  OUTPUTS")
    print(f"     {out_dir}/")
    print(f"       {location_id}.png   — 5-panel visual diagnostic")
    print(f"       {location_id}.json  — full structured result")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Single-point Starlink LEO risk diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("lat",       type=float, help="Latitude  (WGS84 decimal degrees)")
    parser.add_argument("lon",       type=float, help="Longitude (WGS84 decimal degrees)")
    parser.add_argument("--id",      default=None,
                        help="Location identifier for output filenames")
    parser.add_argument("--no-search", action="store_true",
                        help="Skip the 50 m neighbourhood best-point search")
    parser.add_argument("--out-dir", default="outputs/reports",
                        help="Root output directory (default: outputs/reports)")
    args = parser.parse_args()

    lat = args.lat
    lon = args.lon
    location_id = args.id or f"{lat:.4f}_{lon:.4f}".replace("-", "m")
    run_search  = not args.no_search
    out_dir     = ROOT / args.out_dir / location_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analyzing ({lat}, {lon})  id={location_id}  search={run_search}")

    try:
        result = analyze_location(lat, lon, run_local_search=run_search)
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    result["label"] = location_id

    # Save PNG (reuse main.py's _plot_point)
    png_path = out_dir / f"{location_id}.png"
    try:
        _main_module._plot_point(result, png_path)
    except Exception as exc:
        logger.warning(f"PNG generation failed: {exc}")

    # Save JSON (reuse main.py's _save_json)
    json_path = out_dir / f"{location_id}.json"
    _main_module._save_json(result, json_path)

    # Print text report
    _print_report(result, location_id, out_dir)


if __name__ == "__main__":
    main()
