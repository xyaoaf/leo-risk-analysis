"""
make_folium_map.py — Interactive Folium risk map.

Reads available result CSVs/JSONs and produces outputs/nc_risk_map.html:
a zoomable NC map with color-coded risk markers and clickable popups.

Two data layers are shown:
  • Challenge-50 sample   — 37 points from the stratified NC dataset
  • NC validation set     — 8 hand-selected geographically-spread points
                            (loaded from outputs/nc_test/*.json)

After the full 594-point NC batch runs, regenerate this map by running:
    conda run -n cs378 python make_folium_map.py --batch-csv outputs/batch/batch_results.csv

Future extension (not yet implemented):
    Click-to-compute: a web backend (e.g. FastAPI) could accept lat/lon from
    a map click, call analyze_location(), and stream back a risk popup without
    a pre-computed CSV.

Usage:
    conda run -n cs378 python make_folium_map.py
    conda run -n cs378 python make_folium_map.py --batch-csv outputs/batch/batch_results.csv
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster

ROOT = Path(__file__).parent
C50_CSV    = ROOT / "outputs" / "challenge50" / "challenge50_results.csv"
NC_TEST_DIR = ROOT / "outputs" / "nc_test"
OUT_HTML   = ROOT / "outputs" / "nc_risk_map.html"

TIER_COLORS = {
    "low":      "#2ecc71",
    "moderate": "#f39c12",
    "high":     "#e67e22",
    "critical": "#e74c3c",
}
TIER_BOOTSTRAP = {
    "low":      "green",
    "moderate": "orange",
    "high":     "darkred",
    "critical": "red",
}
TIER_ICONS = {
    "low":      "check",
    "moderate": "info-sign",
    "high":     "warning-sign",
    "critical": "remove",
}


# ── Popup HTML ────────────────────────────────────────────────────────────────

def _popup_html(row: dict, source_label: str) -> str:
    tier_color = TIER_COLORS.get(row.get("risk_tier", "low"), "#9ca3af")
    score    = row.get("risk_score", "—")
    feasible = "Yes" if row.get("feasible", True) else "No"
    dom      = row.get("dominant", "—")
    slope    = row.get("slope_deg", "—")
    canopy   = row.get("canopy_max_m", "—")
    bldgs    = row.get("building_count", "—")
    warn     = row.get("warnings", "")
    loc_id   = row.get("location_id", "—")
    warn_html = (
        f'<p style="color:#b45309;font-size:11px;margin:4px 0">'
        f'<b>Note:</b> {warn}</p>'
        if warn and str(warn).strip() else ""
    )
    src_badge = (
        f'<span style="background:#e0f2fe;color:#0369a1;font-size:10px;'
        f'padding:1px 6px;border-radius:10px">{source_label}</span>'
    )
    try:
        score_fmt = f"{float(score):.0f}/100"
    except (TypeError, ValueError):
        score_fmt = "—"
    try:
        slope_fmt = f"{float(slope):.1f}°"
    except (TypeError, ValueError):
        slope_fmt = str(slope)
    try:
        canopy_fmt = f"{float(canopy):.0f} m"
    except (TypeError, ValueError):
        canopy_fmt = str(canopy)

    return f"""
<div style="font-family:sans-serif;min-width:220px;max-width:300px">
  <div style="background:{tier_color};color:white;padding:6px 10px;
              border-radius:4px 4px 0 0;font-weight:bold;font-size:13px">
    {row.get('risk_tier','?').upper()} — Score {score_fmt}
  </div>
  <div style="padding:8px 10px;border:1px solid #e5e7eb;border-top:none;
              border-radius:0 0 4px 4px;background:white">
    <div style="margin-bottom:6px">{src_badge}</div>
    <table style="width:100%;font-size:12px;border-collapse:collapse">
      <tr><td style="color:#6b7280">Location ID</td>
          <td style="text-align:right"><b>{loc_id}</b></td></tr>
      <tr><td style="color:#6b7280">Feasible</td>
          <td style="text-align:right"><b>{feasible}</b></td></tr>
      <tr><td style="color:#6b7280">Dominant</td>
          <td style="text-align:right"><b>{dom}</b></td></tr>
      <tr><td style="color:#6b7280">Slope</td>
          <td style="text-align:right">{slope_fmt}</td></tr>
      <tr><td style="color:#6b7280">Canopy max</td>
          <td style="text-align:right">{canopy_fmt}</td></tr>
      <tr><td style="color:#6b7280">Buildings</td>
          <td style="text-align:right">{bldgs}</td></tr>
      <tr><td style="color:#6b7280">Coords</td>
          <td style="text-align:right">{row.get('lat','?'):.4f}, {row.get('lon','?'):.4f}</td></tr>
    </table>
    {warn_html}
  </div>
</div>
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_nc_test_points() -> list[dict]:
    """Load nc_test results from per-point JSON files in outputs/nc_test/."""
    rows = []
    for jf in sorted(NC_TEST_DIR.glob("point_*.json")):
        try:
            r = json.loads(jf.read_text())
            rk = r.get("risk", {})
            cl = r.get("classification", {})
            rows.append({
                "location_id":    jf.stem.replace("point_", ""),
                "lat":            r.get("lat"),
                "lon":            r.get("lon"),
                "feasible":       r.get("feasible", True),
                "risk_score":     rk.get("risk_score", 0),
                "risk_tier":      rk.get("risk_tier", "low"),
                "dominant":       cl.get("dominant", "clear"),
                "slope_deg":      r.get("slope_deg", 0),
                "canopy_max_m":   r.get("canopy_max_m", 0),
                "building_count": r.get("building_count", 0),
                "warnings":       "; ".join(r.get("warnings", [])),
            })
        except Exception as e:
            print(f"  Skipping {jf.name}: {e}")
    return rows


def _add_markers(layer_groups: dict[str, folium.FeatureGroup],
                 rows: list[dict], source_label: str, circle: bool = False):
    """Add markers for all rows into the appropriate tier layer group."""
    for row in rows:
        tier     = row.get("risk_tier", "low")
        feasible = bool(row.get("feasible", True))
        color    = TIER_BOOTSTRAP.get(tier, "gray")
        icon_nm  = "times" if not feasible else TIER_ICONS.get(tier, "info-sign")

        kwargs = dict(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(
                folium.IFrame(_popup_html(row, source_label), width=320, height=235),
                max_width=320,
            ),
            tooltip=(f"{tier.capitalize()} ({row.get('risk_score','?')}/100)"
                     f" — {row.get('dominant','?')}  [{source_label}]"),
            icon=folium.Icon(color=color, icon=icon_nm, prefix="fa"),
        )
        if circle:
            # Use CircleMarker for dense batch datasets to keep the map light
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color=TIER_COLORS.get(tier, "#9ca3af"),
                fill=True,
                fill_color=TIER_COLORS.get(tier, "#9ca3af"),
                fill_opacity=0.75,
                popup=folium.Popup(
                    folium.IFrame(_popup_html(row, source_label), width=320, height=235),
                    max_width=320,
                ),
                tooltip=(f"{tier.capitalize()} ({row.get('risk_score','?')}/100)"
                         f" — {row.get('dominant','?')}  [{source_label}]"),
            ).add_to(layer_groups[tier])
        else:
            folium.Marker(**kwargs).add_to(layer_groups[tier])


def _stats_box_html(datasets: list[tuple[str, list[dict]]]) -> str:
    """Build a floating stats-box HTML showing tier counts per dataset."""
    lines = []
    for label, rows in datasets:
        n = len(rows)
        if n == 0:
            continue
        feasible = sum(1 for r in rows if r.get("feasible", True))
        by_tier  = {}
        for r in rows:
            t = r.get("risk_tier", "low")
            by_tier[t] = by_tier.get(t, 0) + 1
        tier_parts = "  ".join(
            f'<span style="color:{TIER_COLORS[t]}">{by_tier.get(t,0)}&nbsp;{t}</span>'
            for t in ["low", "moderate", "high", "critical"]
            if by_tier.get(t, 0) > 0
        )
        lines.append(
            f'<div style="margin-bottom:6px">'
            f'<b style="font-size:12px">{label}</b> '
            f'<span style="color:#6b7280;font-size:11px">(n={n}, '
            f'feasible={feasible})</span><br>'
            f'<div style="font-size:11px;margin-top:2px">{tier_parts}</div>'
            f'</div>'
        )
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<div style="width:12px;height:12px;border-radius:50%;'
        f'background:{color};border:1px solid #374151"></div>'
        f'<span style="font-size:11px">{tier.capitalize()}</span>'
        f"</div>"
        for tier, color in TIER_COLORS.items()
    )
    return f"""
<div style="position:fixed;bottom:30px;right:10px;z-index:9999;
            background:white;padding:12px 16px;border-radius:8px;
            border:1px solid #d1d5db;box-shadow:0 2px 8px rgba(0,0,0,0.12);
            font-family:sans-serif;max-width:240px">
  <div style="font-weight:bold;font-size:13px;margin-bottom:8px;color:#111827">
    NC LEO Risk Analysis
  </div>
  {''.join(lines)}
  <div style="border-top:1px solid #e5e7eb;padding-top:8px;margin-top:4px">
    <b style="font-size:11px;color:#374151">Risk Tier</b>
    {legend_items}
    <div style="font-size:10px;color:#9ca3af;margin-top:4px">✗ = infeasible</div>
  </div>
</div>
"""


# ── County choropleth helpers ─────────────────────────────────────────────────

def _load_nc_counties_geojson() -> dict | None:
    """Load NC county boundaries as raw GeoJSON dict (for Folium choropleth)."""
    bd = ROOT / "data" / "boundaries"
    bd.mkdir(parents=True, exist_ok=True)
    fp = bd / "us_counties_fips.geojson"
    if not fp.exists():
        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                fp.write_bytes(r.read())
        except Exception as e:
            print(f"  Could not download county boundaries: {e}")
            return None
    full = json.loads(fp.read_text())
    # Filter to NC (FIPS starting with "37")
    nc_features = [f for f in full["features"]
                   if str(f.get("id", "")).startswith("37")]
    return {"type": "FeatureCollection", "features": nc_features}


def _county_summary_from_rows(rows: list[dict]) -> pd.DataFrame:
    """Compute per-county mean risk and % infeasible from batch result rows."""
    df = pd.DataFrame(rows)
    if df.empty or "county_fips" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("county_fips").agg(
        mean_risk=("risk_score", "mean"),
        pct_infeasible=("feasible", lambda s: (1 - s.mean()) * 100),
        n_points=("risk_score", "count"),
    ).reset_index()
    grp["mean_risk"] = grp["mean_risk"].round(1)
    grp["pct_infeasible"] = grp["pct_infeasible"].round(1)
    return grp


def _add_choropleth(m: folium.Map, geojson: dict, summary: pd.DataFrame):
    """Add a county-level choropleth overlay to the Folium map."""
    # Mean risk choropleth
    folium.Choropleth(
        geo_data=geojson,
        data=summary,
        columns=["county_fips", "mean_risk"],
        key_on="feature.id",
        fill_color="RdYlGn_r",
        fill_opacity=0.45,
        line_opacity=0.3,
        nan_fill_opacity=0.05,
        legend_name="Mean Risk Score (0–100)",
        name="County: Mean Risk",
        show=False,
    ).add_to(m)

    # % infeasible choropleth
    folium.Choropleth(
        geo_data=geojson,
        data=summary,
        columns=["county_fips", "pct_infeasible"],
        key_on="feature.id",
        fill_color="Reds",
        fill_opacity=0.45,
        line_opacity=0.3,
        nan_fill_opacity=0.05,
        legend_name="% Infeasible Locations",
        name="County: % Infeasible",
        show=False,
    ).add_to(m)


# ── Map builder ───────────────────────────────────────────────────────────────

def build_map(c50_rows: list[dict], nc_test_rows: list[dict],
              batch_rows: list[dict]) -> folium.Map:
    all_lats = [r["lat"] for r in c50_rows + nc_test_rows + batch_rows if r.get("lat")]
    all_lons = [r["lon"] for r in c50_rows + nc_test_rows + batch_rows if r.get("lon")]
    center   = [sum(all_lats)/len(all_lats), sum(all_lons)/len(all_lons)] if all_lats else [35.5, -79.5]

    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    # Title
    m.get_root().html.add_child(folium.Element("""
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
            z-index:9999;background:white;padding:10px 20px;border-radius:8px;
            border:1px solid #d1d5db;box-shadow:0 2px 8px rgba(0,0,0,0.12);
            font-family:sans-serif;white-space:nowrap">
  <span style="font-weight:bold;font-size:14px;color:#111827">
    NC Starlink LEO Risk Map
  </span>
  <span style="font-size:12px;color:#6b7280;margin-left:8px">
    Click any marker for details
  </span>
</div>
"""))

    # One feature group per tier (shared across all datasets)
    tier_groups = {
        tier: folium.FeatureGroup(name=f"Risk: {tier.capitalize()}", show=True)
        for tier in TIER_COLORS
    }

    datasets = []
    if c50_rows:
        _add_markers(tier_groups, c50_rows, "Challenge-50 sample")
        datasets.append(("Challenge-50 sample", c50_rows))
    if nc_test_rows:
        _add_markers(tier_groups, nc_test_rows, "NC validation set")
        datasets.append(("NC validation set", nc_test_rows))
    if batch_rows:
        _add_markers(tier_groups, batch_rows, "Full NC batch", circle=True)
        datasets.append(("Full NC batch", batch_rows))

    for g in tier_groups.values():
        g.add_to(m)

    # County choropleth overlays (only when batch data with county_fips is available)
    if batch_rows and any("county_fips" in r for r in batch_rows):
        geojson = _load_nc_counties_geojson()
        if geojson:
            summary = _county_summary_from_rows(batch_rows)
            if not summary.empty:
                _add_choropleth(m, geojson, summary)

    folium.LayerControl(collapsed=False).add_to(m)

    m.get_root().html.add_child(folium.Element(_stats_box_html(datasets)))
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build interactive NC risk map")
    parser.add_argument("--batch-csv", default=None,
                        help="Optional path to full batch results CSV "
                             "(e.g. outputs/batch/batch_results.csv)")
    args = parser.parse_args()

    # Challenge-50 sample
    c50_rows = []
    if C50_CSV.exists():
        df = pd.read_csv(C50_CSV)
        for col in ("risk_score", "slope_deg", "canopy_max_m", "building_count"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["feasible"] = (df["feasible"].astype(str).str.lower()
                          .map({"true": True, "false": False, "1": True, "0": False})
                          .fillna(True))
        c50_rows = df.to_dict("records")
        print(f"Challenge-50 sample: {len(c50_rows)} points")

    # NC test set (from per-point JSONs)
    nc_test_rows = _load_nc_test_points()
    print(f"NC validation set:   {len(nc_test_rows)} points")

    # Optional full batch results
    batch_rows = []
    if args.batch_csv:
        bpath = Path(args.batch_csv)
        if bpath.exists():
            bdf = pd.read_csv(bpath)
            bdf = bdf[bdf["error"].isna()] if "error" in bdf.columns else bdf
            for col in ("risk_score", "slope_deg", "canopy_max_m", "building_count"):
                if col in bdf.columns:
                    bdf[col] = pd.to_numeric(bdf[col], errors="coerce").fillna(0)
            if "feasible" in bdf.columns:
                bdf["feasible"] = (bdf["feasible"].astype(str).str.lower()
                                   .map({"true": True, "false": False})
                                   .fillna(True))
            batch_rows = bdf.to_dict("records")
            print(f"Full batch results:  {len(batch_rows)} points")
        else:
            print(f"Batch CSV not found: {bpath}")

    if not c50_rows and not nc_test_rows and not batch_rows:
        print("No data found — nothing to map.")
        return

    m = build_map(c50_rows, nc_test_rows, batch_rows)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    total = len(c50_rows) + len(nc_test_rows) + len(batch_rows)
    print(f"Saved: {OUT_HTML}  ({total} total points)")


if __name__ == "__main__":
    main()
