"""
Public entry point for the AIS vessel attribution module.

    rank_vessels(ais_data, origin_lat, origin_lon, origin_time, ...)

Takes raw AIS data (DataFrame or CSV path) plus an estimated spill origin
(from Member 2, or a mock during standalone development) and returns
ranked candidate source vessels.
"""

import pandas as pd

from .preprocessing import clean_ais_data, filter_by_time_window, filter_by_geography, build_trajectories
from .features import compute_all_features
from .scoring import score_vessel_features, DEFAULT_WEIGHTS


def _generate_explanation(feat, scores, uncertainty_km):
    parts = []
    if feat["entered_uncertainty_zone"]:
        parts.append(f"entered the {uncertainty_km:.0f} km origin uncertainty zone")
    parts.append(f"came within {feat['min_distance_km']:.1f} km of the estimated origin")
    parts.append(f"was tracked {feat['time_difference_hours']:.1f} h from the estimated spill time")
    if feat["points_within_2km"] > 0:
        parts.append(f"had {feat['points_within_2km']} AIS point(s) within 2 km")
    if scores["behavior_score"] >= 0.6:
        parts.append("showed a notable speed/heading change near the origin")
    return f"{feat['vessel_name']} ({feat['mmsi']}) " + "; ".join(parts) + "."


def rank_vessels(ais_data, origin_lat, origin_lon, origin_time, uncertainty_km=5,
                  search_radius_km=20, time_window_hours=12, weights=None, top_n=None):
    """
    Rank vessels by potential correlation with an estimated oil-spill origin.

    Parameters
    ----------
    ais_data : pd.DataFrame or str
        Raw AIS records, or a path to a CSV file with AIS columns.
    origin_lat, origin_lon : float
        Estimated spill origin coordinates.
    origin_time : str or datetime
        Estimated spill origin time (ISO8601 string or datetime; treated as UTC).
    uncertainty_km : float
        Radius of the origin uncertainty zone.
    search_radius_km : float
        Geographic filter radius around the origin; also the proximity-score scale.
    time_window_hours : float
        Time filter window around origin_time; also the temporal-score scale.
    weights : dict, optional
        Override DEFAULT_WEIGHTS (proximity/temporal/trajectory/behavior/vessel), must sum to ~1.0.
    top_n : int, optional
        If set, return only the top N ranked candidates.

    Returns
    -------
    dict with:
        "candidates": list of ranked candidate dicts (JSON-serializable, minus trajectory objects)
        "trajectories_geojson": dict of {mmsi: GeoJSON LineString Feature} for the dashboard
        "meta": run parameters + counts, for transparency/debugging
    """
    if isinstance(ais_data, str):
        ais_data = pd.read_csv(ais_data)

    weights = weights or DEFAULT_WEIGHTS

    cleaned = clean_ais_data(ais_data)
    n_raw_vessels = cleaned["mmsi"].nunique()

    time_filtered = filter_by_time_window(cleaned, origin_time, time_window_hours)
    geo_filtered = filter_by_geography(time_filtered, origin_lat, origin_lon, search_radius_km)

    trajectories = build_trajectories(geo_filtered)

    if not trajectories:
        return {
            "candidates": [],
            "trajectories_geojson": {},
            "meta": {
                "origin": {"lat": origin_lat, "lon": origin_lon, "time": str(origin_time)},
                "n_vessels_input": n_raw_vessels,
                "n_vessels_in_window": 0,
                "message": "No vessels found within the given time/geographic window.",
            },
        }

    features = compute_all_features(trajectories, origin_lat, origin_lon, origin_time, uncertainty_km)

    candidates = []
    geojson = {}
    for feat in features:
        scores = score_vessel_features(feat, search_radius_km, time_window_hours, weights)
        explanation = _generate_explanation(feat, scores, uncertainty_km)

        candidates.append({
            "mmsi": feat["mmsi"],
            "vessel_name": feat["vessel_name"],
            "vessel_type": feat["vessel_type"],
            "minimum_distance_km": round(feat["min_distance_km"], 2),
            "time_difference_hours": round(feat["time_difference_hours"], 2),
            "points_within_2km": feat["points_within_2km"],
            "points_within_5km": feat["points_within_5km"],
            "entered_uncertainty_zone": feat["entered_uncertainty_zone"],
            "proximity_score": scores["proximity_score"],
            "temporal_score": scores["temporal_score"],
            "trajectory_score": scores["trajectory_score"],
            "behavior_score": scores["behavior_score"],
            "vessel_score": scores["vessel_score"],
            "final_score": scores["final_score"],
            "explanation": explanation,
        })

        traj = feat["trajectory"]
        geojson[feat["mmsi"]] = {
            "type": "Feature",
            "properties": {"mmsi": feat["mmsi"], "vessel_name": feat["vessel_name"]},
            "geometry": {
                "type": "LineString",
                "coordinates": traj[["longitude", "latitude"]].values.tolist(),
            },
        }

    candidates.sort(key=lambda c: c["final_score"], reverse=True)
    for i, c in enumerate(candidates, start=1):
        c["rank"] = i
    candidates = [{"rank": c.pop("rank"), **c} for c in candidates]

    if top_n:
        candidates = candidates[:top_n]

    return {
        "candidates": candidates,
        "trajectories_geojson": geojson,
        "meta": {
            "origin": {"lat": origin_lat, "lon": origin_lon, "time": str(origin_time),
                       "uncertainty_km": uncertainty_km},
            "search_radius_km": search_radius_km,
            "time_window_hours": time_window_hours,
            "weights": weights,
            "n_vessels_input": n_raw_vessels,
            "n_vessels_in_window": len(trajectories),
            "note": "final_score reflects potential source-vessel correlation, not proof of causation.",
        },
    }