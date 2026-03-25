"""
agent.py — Claude-orchestrated LEO risk analysis agent.

Architecture
------------
Claude (claude-opus-4-6) acts as the reasoning and workflow orchestration layer.
All geospatial computation is deterministic Python — no LLM involvement in scoring.

Claude's role:
  - Parse user intent (extract coordinates, interpret context)
  - Select which tools to call and in what order
  - Interpret results and compose a human-readable risk report
  - Handle ambiguous inputs gracefully

Tools available to the agent:
  analyze_location      — full pipeline for a single lat/lon
  find_better_nearby    — scan a grid of candidate locations for better sky visibility
  batch_analyze         — analyze a list of (lat, lon) pairs

Usage
-----
    from src.agent import run_agent

    report = run_agent("What is the Starlink risk at 30.2867, -97.7113 in Austin TX?")
    print(report)

    # Or with an environment that has ANTHROPIC_API_KEY set:
    #   conda run -n cs378 python -c "from src.agent import run_agent; print(run_agent(...))"
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool implementations (deterministic Python — no LLM)
# ---------------------------------------------------------------------------

def _analyze_location(lat: float, lon: float, radius_near: int = 100,
                       radius_far: int = 1500) -> dict:
    """
    Run the full geospatial pipeline for a single location.
    Delegates to feasibility.analyze_location (hard constraints + risk score).
    Returns a JSON-serialisable summary dict.
    """
    try:
        from feasibility import analyze_location
        result = analyze_location(
            lat, lon,
            radius_near=radius_near,
            radius_far=radius_far,
            run_local_search=True,
        )

        # Build a compact, JSON-safe summary for the agent
        cl = result["classification"]
        rk = result["risk"]
        failed = [k for k, v in result["constraints"].items() if not v["pass"]]

        summary = {
            "status":            "ok",
            "lat":               lat,
            "lon":               lon,
            "feasible":          result["feasible"],
            "on_building":       result["on_building"],
            "dish_height_asl_m": result["dish_height_asl_m"],
            "dominant_obstruction": cl["dominant"],
            "max_angle_deg":     round(cl["max_angle_buildings"], 1),
            "slope_deg":         result["slope_deg"],
            "blocked_frac_terrain_pct":   round(cl["blocked_frac_terrain"]   * 100, 1),
            "blocked_frac_canopy_pct":    round(cl["blocked_frac_canopy"]    * 100, 1),
            "blocked_frac_buildings_pct": round(cl["blocked_frac_buildings"] * 100, 1),
            "building_count":    result["building_count"],
            "canopy_max_m":      result["canopy_max_m"],
            "canopy_simulated":  result["canopy_simulated"],
            "risk_score":        rk["risk_score"],
            "risk_tier":         rk["risk_tier"],
            "risk_components":   rk["components"],
            "explanation":       rk["explanation"],
            "failed_constraints": failed,
            "warnings":          result["warnings"],
            "data_sources":      result["data_sources"],
            "elapsed_s":         result["elapsed_s"],
        }

        # Attach best_nearby if local search ran
        bn = result.get("best_nearby")
        if bn and bn.get("best"):
            summary["best_nearby"] = {
                "lat":         bn["best"]["lat"],
                "lon":         bn["best"]["lon"],
                "risk_score":  bn["best"]["risk_score"],
                "risk_tier":   bn["best"]["risk_tier"],
                "dominant":    bn["best"]["dominant"],
                "distance_m":  bn["improvement"]["distance_m"],
                "risk_delta":  bn["improvement"]["risk_delta"],
                "explanation": bn["improvement"]["explanation"],
            }

        return summary

    except Exception as exc:
        logger.error(f"[agent._analyze_location] Failed for ({lat},{lon}): {exc}")
        return {"status": "error", "lat": lat, "lon": lon, "error": str(exc)}


def _find_better_nearby(lat: float, lon: float,
                         search_radius_m: int = 50,
                         n_candidates: int = 8) -> dict:
    """
    Find a better nearby dish location using tile-reuse local search.
    Delegates to feasibility.analyze_location with run_local_search=True,
    which internally calls find_better_nearby with pre-fetched tiles.
    """
    try:
        from feasibility import analyze_location
        result = analyze_location(
            lat, lon,
            run_local_search=True,
            local_search_radius=search_radius_m,
        )
        bn = result.get("best_nearby")
        if not bn or not bn.get("best"):
            return {
                "status": "ok",
                "message": "No improvement found nearby.",
                "origin_risk": result["risk"]["risk_score"],
            }
        return {
            "status":       "ok",
            "origin":       {"lat": lat, "lon": lon,
                             "risk_score": result["risk"]["risk_score"]},
            "best":         bn["best"],
            "improvement":  bn["improvement"],
            "n_evaluated":  bn["n_evaluated"],
            "candidates":   bn["candidates"],
        }
    except Exception as exc:
        logger.error(f"[agent._find_better_nearby] Failed: {exc}")
        return {"status": "error", "error": str(exc)}


def _batch_analyze(locations: list[dict]) -> dict:
    """
    Analyze a list of {lat, lon, label?} dicts.
    Returns summary statistics + per-location results.
    """
    results = []
    for loc in locations:
        lat  = float(loc["lat"])
        lon  = float(loc["lon"])
        label = loc.get("label", f"{lat:.4f},{lon:.4f}")
        r = _analyze_location(lat, lon)
        r["label"] = label
        results.append(r)

    ok = [r for r in results if r["status"] == "ok"]
    tier_counts: dict[str, int] = {"low": 0, "moderate": 0, "high": 0, "critical": 0}
    for r in ok:
        tier_counts[r["risk_tier"]] = tier_counts.get(r["risk_tier"], 0) + 1

    at_risk_pct = (
        (tier_counts["high"] + tier_counts["critical"]) / len(ok) * 100
        if ok else 0
    )

    return {
        "status":       "ok",
        "n_requested":  len(locations),
        "n_completed":  len(ok),
        "tier_summary": tier_counts,
        "at_risk_pct":  round(at_risk_pct, 1),
        "results":      results,
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "analyze_location",
        "description": (
            "Analyze Starlink LEO satellite obstruction risk at a specific geographic location. "
            "Fetches terrain (DEM), tree canopy height, and building footprints from public "
            "datasets, computes the sky-visibility horizon profile in all directions, and "
            "returns a 0–100 risk score with dominant obstruction type. "
            "Use this for single-point queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Latitude in decimal degrees (WGS84)",
                },
                "lon": {
                    "type": "number",
                    "description": "Longitude in decimal degrees (WGS84)",
                },
                "radius_near": {
                    "type": "integer",
                    "description": (
                        "Radius in meters for near-field analysis (canopy + buildings). "
                        "Default 100m."
                    ),
                    "default": 100,
                },
                "radius_far": {
                    "type": "integer",
                    "description": (
                        "Radius in meters for far-field terrain horizon analysis. "
                        "Default 1500m."
                    ),
                    "default": 1500,
                },
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "find_better_nearby",
        "description": (
            "Search for alternative dish-mounting locations within a buffer around the "
            "specified point that have better Starlink sky visibility (lower risk score). "
            "Samples candidate points on a ring at ~70% of search_radius_m and returns "
            "the best candidate along with all scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Center latitude (WGS84)"},
                "lon": {"type": "number", "description": "Center longitude (WGS84)"},
                "search_radius_m": {
                    "type": "integer",
                    "description": "How far to search for alternatives (meters). Default 300m.",
                    "default": 300,
                },
                "n_candidates": {
                    "type": "integer",
                    "description": "Number of candidate points to evaluate. Default 8.",
                    "default": 8,
                },
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "batch_analyze",
        "description": (
            "Analyze a list of locations in sequence and return aggregated statistics. "
            "Use this when the user provides multiple coordinates or asks about a set "
            "of locations. Returns per-location results + tier distribution summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "locations": {
                    "type": "array",
                    "description": "List of location objects to analyze.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "lat":   {"type": "number"},
                            "lon":   {"type": "number"},
                            "label": {"type": "string", "description": "Optional name"},
                        },
                        "required": ["lat", "lon"],
                    },
                }
            },
            "required": ["locations"],
        },
    },
]


_TOOL_FNS: dict[str, Any] = {
    "analyze_location":   _analyze_location,
    "find_better_nearby": _find_better_nearby,
    "batch_analyze":      _batch_analyze,
}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a geospatial analysis assistant specialising in LEO satellite (Starlink) connectivity risk.

Your job is to help users understand whether a specific location is likely to have good or poor
Starlink signal quality due to environmental obstructions (terrain, tree canopy, buildings).

You have access to the following tools:
- analyze_location    : run the full geospatial pipeline for a single lat/lon
- find_better_nearby  : search for a better nearby dish location
- batch_analyze       : analyze multiple locations at once

Workflow guidelines:
1. If the user provides coordinates, call analyze_location immediately.
2. If the user describes a place without exact coordinates, ask for them.
3. After getting results, always explain:
   - The risk score and tier (low/moderate/high/critical)
   - The dominant obstruction type and why it matters
   - A practical recommendation (e.g. try a taller mount, relocate dish)
4. If risk is high/critical and the user is interested, offer to call find_better_nearby.
5. All scoring is deterministic — you do NOT guess or estimate scores yourself; call the tools.
6. Be concise. Non-technical users should understand the key takeaway in the first sentence.
"""


def run_agent(user_message: str, api_key: str | None = None,
              model: str = "claude-opus-4-6", max_iterations: int = 10) -> str:
    """
    Run the LEO risk analysis agent for a single user query.

    Parameters
    ----------
    user_message  : natural language query from the user
    api_key       : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
    model         : Claude model to use
    max_iterations: safeguard against runaway tool loops

    Returns
    -------
    str : final natural-language response from the agent
    """
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "No Anthropic API key found. "
            "Set ANTHROPIC_API_KEY environment variable or pass api_key=..."
        )

    client   = anthropic.Anthropic(api_key=key)
    messages = [{"role": "user", "content": user_message}]

    logger.info(f"[agent] Starting | model={model} query={user_message[:80]!r}")

    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        logger.info(
            f"[agent] iter={iteration} stop_reason={response.stop_reason} "
            f"blocks={len(response.content)}"
        )

        # ── Final answer — no more tool calls ──────────────────────────────
        if response.stop_reason == "end_turn":
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            return "\n".join(text_blocks).strip()

        # ── Process tool calls ─────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name   = block.name
                tool_input  = block.input
                tool_use_id = block.id

                logger.info(f"[agent] Tool call: {tool_name}({tool_input})")

                fn = _TOOL_FNS.get(tool_name)
                if fn is None:
                    result_payload = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        result_payload = fn(**tool_input)
                    except Exception as exc:
                        logger.error(f"[agent] Tool {tool_name} raised: {exc}")
                        result_payload = {"status": "error", "error": str(exc)}

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_use_id,
                    "content":     json.dumps(result_payload, default=str),
                })

            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason
        logger.warning(f"[agent] Unexpected stop_reason={response.stop_reason}")
        break

    return "[Agent reached iteration limit without a final answer]"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="LEO Risk Analysis Agent")
    parser.add_argument("query", nargs="?",
                        default="Analyze Starlink risk at 30.2867, -97.7113 (Austin TX)")
    parser.add_argument("--model", default="claude-opus-4-6")
    args = parser.parse_args()

    answer = run_agent(args.query, model=args.model)
    print("\n" + "=" * 60)
    print(answer)
    print("=" * 60)
