"""
scoring.py — Deterministic risk score from obstruction classification.

Score formula
-------------
The risk score (0–100) reflects how likely a Starlink installation at this
location will experience degraded service due to environmental obstructions.

Three independent components are summed:

  1. FOV blockage (0–50 pts)
     Fraction of the Starlink northward FOV (~100°) where any obstruction
     exceeds the 25° elevation threshold.  Linear scale: 100% blocked = 50 pts.

  2. Angle severity (0–30 pts)
     How far above the 25° threshold the worst obstruction reaches.
     At 25° → 0 pts; at 90° → 30 pts.  Captures situations where the dish
     sees a solid wall, not just a marginal clip.

  3. Permanence penalty (0–20 pts)
     Buildings and terrain are year-round, permanent obstacles.
     Vegetation varies seasonally (deciduous trees lose leaves in winter).
     Factor: building/terrain = 1.0, vegetation = 0.6, clear = 0.0

Risk tiers
----------
  0–19  : low       — sky likely clear, service expected good
  20–44 : moderate  — some obstruction, seasonal or off-axis, may degrade
  45–69 : high      — meaningful FOV blockage, service likely impaired
  70–100: critical  — severe obstruction, Starlink unlikely to work reliably
"""

from __future__ import annotations


# Permanence factors per dominant obstruction type
_PERMANENCE = {
    "building":   1.0,   # concrete/steel — permanent
    "terrain":    1.0,   # geology — permanent
    "vegetation": 0.6,   # deciduous trees lose leaves; conifers year-round
    "clear":      0.0,   # no blockage
}

_TIER_THRESHOLDS = [
    (70, "critical"),
    (45, "high"),
    (20, "moderate"),
    ( 0, "low"),
]


def score_risk(classification: dict, evergreen_fraction: float | None = None) -> dict:
    """
    Convert an obstruction classification dict to a 0–100 risk score.

    Parameters
    ----------
    classification      : output of horizon.classify_obstruction()
    evergreen_fraction  : optional float 0–1 for vegetation-dominant locations.
                          0 = fully deciduous (loses leaves in winter, lower permanence),
                          1 = fully coniferous (year-round blockage, higher permanence).
                          If None, defaults to the fixed 0.6 blended value from _PERMANENCE.

    Returns
    -------
    dict:
        risk_score     : float 0–100 (higher = more at risk)
        risk_tier      : "low" | "moderate" | "high" | "critical"
        components     : dict with fov_blockage, angle_severity, permanence scores
        explanation    : one-sentence human-readable summary
    """
    dominant = str(classification.get("dominant", "clear"))

    # Select the blockage fraction and peak angle for the dominant layer.
    # Using near-field "buildings" metrics for all types was a bug: when terrain
    # dominates (far-field ridge), the near-field window is clear → score = 0
    # despite feasible = False.
    if dominant == "terrain":
        bf      = float(classification.get("blocked_frac_terrain",   0.0))
        max_ang = float(classification.get("max_angle_terrain",      0.0))
    elif dominant == "vegetation":
        bf      = float(classification.get("blocked_frac_canopy",    0.0))
        max_ang = float(classification.get("max_angle_canopy",       0.0))
    else:  # "building" or "clear"
        bf      = float(classification.get("blocked_frac_buildings", 0.0))
        max_ang = float(classification.get("max_angle_buildings",    0.0))

    # ── Component 1: FOV blockage ──────────────────────────────────────────
    fov_score = min(50.0, bf * 50.0)   # 100% blocked → 50 pts

    # ── Component 2: Angle severity ────────────────────────────────────────
    if max_ang > 25.0:
        angle_severity = min(1.0, (max_ang - 25.0) / 65.0)   # 25° → 0, 90° → 1
    else:
        angle_severity = 0.0
    angle_score = angle_severity * 30.0

    # ── Component 3: Permanence penalty ────────────────────────────────────
    perm_factor = _PERMANENCE.get(dominant, 0.5)
    # For vegetation, adjust permanence by evergreen fraction when available.
    # Formula: 0.35 (fully deciduous) → 0.90 (fully evergreen).
    # Default _PERMANENCE["vegetation"] = 0.60 ≈ 46% evergreen (blended SE US).
    if dominant == "vegetation" and evergreen_fraction is not None:
        perm_factor = 0.35 + 0.55 * float(evergreen_fraction)
    # Scale by blockage so "vegetation with 0% blockage" still scores 0
    perm_score  = perm_factor * min(bf * 40.0, 20.0)   # cap at 20

    total = min(100.0, fov_score + angle_score + perm_score)

    # ── Tier ───────────────────────────────────────────────────────────────
    tier = "low"
    for threshold, label in _TIER_THRESHOLDS:
        if total >= threshold:
            tier = label
            break

    # ── Human-readable explanation ─────────────────────────────────────────
    explanation = _explain(total, tier, dominant, bf, max_ang)

    return {
        "risk_score": round(total, 1),
        "risk_tier":  tier,
        "components": {
            "fov_blockage":   round(fov_score,  1),
            "angle_severity": round(angle_score, 1),
            "permanence":     round(perm_score,  1),
        },
        "explanation": explanation,
    }


def _explain(score: float, tier: str, dominant: str, bf: float, max_ang: float) -> str:
    tier_phrases = {
        "low":      "Sky visibility is good",
        "moderate": "Moderate obstructions detected",
        "high":     "Significant obstructions likely to degrade service",
        "critical": "Severe obstructions — reliable Starlink service unlikely",
    }
    phrase = tier_phrases[tier]

    if dominant == "clear":
        detail = "no significant obstructions identified"
    elif dominant == "terrain":
        detail = f"terrain blocks {bf*100:.0f}% of FOV (max angle {max_ang:.1f}°)"
    elif dominant == "vegetation":
        detail = f"tree canopy blocks {bf*100:.0f}% of FOV (max angle {max_ang:.1f}°)"
    elif dominant == "building":
        detail = f"nearby buildings block {bf*100:.0f}% of FOV (max angle {max_ang:.1f}°)"
    else:
        detail = f"max obstruction angle {max_ang:.1f}°"

    return f"{phrase} — {detail}. Risk score: {score:.0f}/100."
