"""
Turn raw per-vessel features into normalized [0, 1] scores and combine
them into a transparent, configurable weighted final_score.

IMPORTANT: final_score is a *potential source-vessel correlation* score,
not a probability and not proof of causation.
"""

DEFAULT_WEIGHTS = {
    "proximity": 0.35,
    "temporal": 0.25,
    "trajectory": 0.20,
    "behavior": 0.10,
    "vessel": 0.10,
}

# Vessel types more commonly associated with oil cargo/transfer, used only
# as a mild contextual nudge -- never a strong signal on its own.
VESSEL_TYPE_PRIOR = {
    "tanker": 0.9,
    "cargo": 0.6,
    "container": 0.5,
    "fishing": 0.3,
    "unknown": 0.5,
}


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _decay_score(value, scale):
    """1.0 at value=0, decaying toward 0 as value grows past `scale`."""
    if value is None or value != value:  # NaN check
        return 0.5  # neutral when unknown
    return _clamp01(1.0 - (value / scale))


def score_proximity(feat, search_radius_km):
    """Closer minimum distance -> higher score. Uses nearest_k_mean_km for stability."""
    return _decay_score(feat["nearest_k_mean_km"], scale=search_radius_km)


def score_temporal(feat, time_window_hours):
    """Smaller time difference from origin_time -> higher score."""
    return _decay_score(feat["time_difference_hours"], scale=time_window_hours)


def score_trajectory(feat):
    """Rewards entering the uncertainty zone and having multiple nearby points."""
    zone_bonus = 0.5 if feat["entered_uncertainty_zone"] else 0.0
    density = min(feat["points_within_5km"] / 5.0, 1.0) * 0.3
    density_tight = min(feat["points_within_2km"] / 3.0, 1.0) * 0.2
    return _clamp01(zone_bonus + density + density_tight)


def score_behavior(feat):
    """
    Sudden speed or heading change near the origin is treated as mildly
    suspicious (e.g. slowing/stopping or maneuvering), but this is a weak
    signal by design (only 10% weight) since normal traffic also varies.
    """
    speed_change = feat["speed_change"]
    heading_change = feat["heading_change"]

    speed_component = 0.5 if speed_change != speed_change else _clamp01(speed_change / 8.0)
    heading_component = 0.5 if heading_change != heading_change else _clamp01(heading_change / 90.0)

    return _clamp01(0.5 * speed_component + 0.5 * heading_component)


def score_vessel(feat):
    vtype = str(feat.get("vessel_type", "unknown")).strip().lower()
    return VESSEL_TYPE_PRIOR.get(vtype, VESSEL_TYPE_PRIOR["unknown"])


def score_vessel_features(feat, search_radius_km, time_window_hours, weights=None):
    """Compute all sub-scores + weighted final_score for one vessel's feature dict."""
    weights = weights or DEFAULT_WEIGHTS

    proximity = score_proximity(feat, search_radius_km)
    temporal = score_temporal(feat, time_window_hours)
    trajectory = score_trajectory(feat)
    behavior = score_behavior(feat)
    vessel = score_vessel(feat)

    final_score = (
        weights["proximity"] * proximity
        + weights["temporal"] * temporal
        + weights["trajectory"] * trajectory
        + weights["behavior"] * behavior
        + weights["vessel"] * vessel
    )

    return {
        "proximity_score": round(proximity, 3),
        "temporal_score": round(temporal, 3),
        "trajectory_score": round(trajectory, 3),
        "behavior_score": round(behavior, 3),
        "vessel_score": round(vessel, 3),
        "final_score": round(_clamp01(final_score), 3),
    }