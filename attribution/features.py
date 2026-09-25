"""
Per-vessel feature extraction relative to an estimated spill origin.

All features are computed from a single vessel's trajectory DataFrame
(timestamp-sorted) plus the origin lat/lon/time.
"""

import numpy as np
import pandas as pd

from .preprocessing import haversine_km


def heading_diff(a, b):
    """Smallest angular difference between two headings, handling wraparound."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def compute_vessel_features(traj: pd.DataFrame, origin_lat, origin_lon, origin_time,
                             uncertainty_km=5.0) -> dict:
    """
    Compute spatial/temporal/behavior features for one vessel's trajectory.

    Returns a dict of raw (unnormalized) feature values plus the point
    (row) at minimum distance, used later for explanations.
    """
    origin_time = pd.to_datetime(origin_time, utc=True)

    dists = haversine_km(traj["latitude"].values, traj["longitude"].values, origin_lat, origin_lon)
    traj = traj.copy()
    traj["_dist_km"] = dists

    min_idx = int(np.argmin(dists))
    min_dist_km = float(dists[min_idx])
    closest_point = traj.iloc[min_idx]

    time_diff_hours = abs((closest_point["timestamp"] - origin_time).total_seconds()) / 3600.0

    points_within_2km = int((dists <= 2.0).sum())
    points_within_5km = int((dists <= 5.0).sum())

    entered_uncertainty_zone = bool((dists <= uncertainty_km).any())

    # Trajectory proximity: average of the closest few points (smoother than a single min)
    k = min(3, len(dists))
    nearest_k_mean_km = float(np.sort(dists)[:k].mean())

    # Speed/heading change "near" the origin (within the closest point's neighborhood)
    speed_change = np.nan
    heading_change = np.nan
    if len(traj) >= 2:
        window = max(1, len(traj) // 6)  # small window around the closest point
        lo = max(0, min_idx - window)
        hi = min(len(traj), min_idx + window + 1)
        local = traj.iloc[lo:hi]
        if local["speed"].notna().sum() >= 2:
            speed_change = float(local["speed"].max() - local["speed"].min())
        if local["heading"].notna().sum() >= 2:
            headings = local["heading"].dropna().values
            diffs = [heading_diff(headings[i], headings[i + 1]) for i in range(len(headings) - 1)]
            heading_change = float(max(diffs)) if diffs else 0.0

    vessel_type = traj["vessel_type"].iloc[0] if "vessel_type" in traj.columns else "Unknown"
    vessel_name = traj["vessel_name"].iloc[0] if "vessel_name" in traj.columns else "UNKNOWN"

    return {
        "mmsi": traj["mmsi"].iloc[0],
        "vessel_name": vessel_name,
        "vessel_type": vessel_type,
        "min_distance_km": min_dist_km,
        "time_difference_hours": time_diff_hours,
        "points_within_2km": points_within_2km,
        "points_within_5km": points_within_5km,
        "entered_uncertainty_zone": entered_uncertainty_zone,
        "nearest_k_mean_km": nearest_k_mean_km,
        "speed_change": speed_change,
        "heading_change": heading_change,
        "n_points": len(traj),
        "closest_point_time": closest_point["timestamp"],
        "trajectory": traj,  # kept for visualization/GeoJSON export
    }


def compute_all_features(trajectories: dict, origin_lat, origin_lon, origin_time,
                          uncertainty_km=5.0) -> list:
    """Compute features for every vessel trajectory. Returns a list of feature dicts."""
    features = []
    for mmsi, traj in trajectories.items():
        try:
            features.append(
                compute_vessel_features(traj, origin_lat, origin_lon, origin_time, uncertainty_km)
            )
        except Exception as e:
            # Don't let one malformed vessel trajectory kill the whole run
            print(f"Warning: skipping vessel {mmsi} due to feature error: {e}")
    return features