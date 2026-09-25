"""
Cleaning and filtering of raw AIS data.

Handles: missing/invalid rows, duplicates, unsorted timestamps, timezone
normalization, and geographic/time-window filtering around a spill origin.
"""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["mmsi", "timestamp", "latitude", "longitude"]
OPTIONAL_COLUMNS = ["speed", "heading", "vessel_name", "vessel_type"]

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km. Accepts scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c


def clean_ais_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw AIS DataFrame:
      - validates required columns exist
      - drops rows with missing/invalid mmsi, timestamp, lat, lon
      - coerces timestamp to UTC datetime
      - clips implausible lat/lon and negative speeds
      - removes exact duplicate records
      - sorts by mmsi, timestamp
    Fills missing optional columns with sensible defaults so downstream
    code never has to special-case their absence.
    """
    df = df.copy()

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"AIS data missing required columns: {missing_required}")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # mmsi as string (avoid float/int mismatches from CSV loading)
    df["mmsi"] = df["mmsi"].astype(str).str.strip()
    df = df[df["mmsi"].notna() & (df["mmsi"] != "") & (df["mmsi"].str.lower() != "nan")]

    # timestamps -> UTC datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # numeric coercion
    for col in ["latitude", "longitude", "speed", "heading"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows missing anything essential
    df = df.dropna(subset=["mmsi", "timestamp", "latitude", "longitude"])

    # plausible ranges
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)]
    df = df[(df["speed"].isna()) | (df["speed"].between(0, 60))]
    df["heading"] = df["heading"].mod(360)

    # de-duplicate exact repeats of (mmsi, timestamp, lat, lon)
    df = df.drop_duplicates(subset=["mmsi", "timestamp", "latitude", "longitude"])

    # fill remaining optional gaps
    df["vessel_name"] = df["vessel_name"].fillna("UNKNOWN")
    df["vessel_type"] = df["vessel_type"].fillna("Unknown")

    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    return df


def filter_by_time_window(df: pd.DataFrame, origin_time, time_window_hours: float) -> pd.DataFrame:
    """Keep only AIS points within +/- time_window_hours of origin_time."""
    origin_time = pd.to_datetime(origin_time, utc=True)
    lo = origin_time - pd.Timedelta(hours=time_window_hours)
    hi = origin_time + pd.Timedelta(hours=time_window_hours)
    return df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)].copy()


def filter_by_geography(df: pd.DataFrame, origin_lat, origin_lon, search_radius_km: float) -> pd.DataFrame:
    """Keep only AIS points within search_radius_km of the origin point."""
    dist = haversine_km(df["latitude"].values, df["longitude"].values, origin_lat, origin_lon)
    df = df.copy()
    df["_dist_to_origin_km"] = dist
    return df[df["_dist_to_origin_km"] <= search_radius_km].copy()


def build_trajectories(df: pd.DataFrame) -> dict:
    """
    Group cleaned AIS points into per-vessel trajectories.
    Returns {mmsi: DataFrame sorted by timestamp}.
    Vessels with fewer than 1 point are skipped (nothing to analyze).
    """
    trajectories = {}
    for mmsi, group in df.groupby("mmsi"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        if len(g) >= 1:
            trajectories[mmsi] = g
    return trajectories