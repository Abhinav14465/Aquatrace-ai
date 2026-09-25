"""
Generate synthetic AIS data for the oil-spill attribution demo.

Scenario is centered on a mock spill origin:
    lat=12.840, lon=80.520, time=2026-09-19T18:00:00Z

Produces ~15 vessels with deliberately varied behavior so the ranking
pipeline has something meaningful to differentiate:

  - VESSEL_STRONG   : passes right through origin at origin time (the culprit)
  - VESSEL_CLOSE_WRONGTIME : geographically close, but hours away in time
  - VESSEL_FARTIME  : near-correct time, but far away spatially
  - VESSEL_ERRATIC  : odd speed/heading changes, but nowhere near the origin
  - VESSEL_BG_*     : normal background traffic, unrelated to the spill

Output: data/ais_demo.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
ORIGIN_LAT = 12.840
ORIGIN_LON = 80.520
ORIGIN_TIME = datetime.fromisoformat("2026-09-19T18:00:00+00:00")

KM_PER_DEG_LAT = 111.0


def _deg_offset_km(lat, lon, d_north_km, d_east_km):
    """Offset a lat/lon point by given km north/east (approximation, fine for demo scale)."""
    dlat = d_north_km / KM_PER_DEG_LAT
    dlon = d_east_km / (KM_PER_DEG_LAT * np.cos(np.radians(lat)))
    return lat + dlat, lon + dlon


def _track(mmsi, vessel_name, vessel_type, start_lat, start_lon, start_time,
           heading_deg, speed_knots, n_points, interval_min, rng,
           heading_jitter=5.0, speed_jitter=0.5, dropout_prob=0.0):
    """Generate a simple straight-ish trajectory with small noise."""
    rows = []
    lat, lon = start_lat, start_lon
    t = start_time
    heading = heading_deg
    speed = speed_knots
    for i in range(n_points):
        if rng.random() < dropout_prob:
            t += timedelta(minutes=interval_min)
            continue
        rows.append({
            "mmsi": mmsi,
            "timestamp": t.isoformat(),
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "speed": round(max(0.0, speed + rng.normal(0, speed_jitter)), 2),
            "heading": round(heading % 360, 1),
            "vessel_name": vessel_name,
            "vessel_type": vessel_type,
        })
        # advance position: speed(knots) * interval -> distance in km
        dist_km = speed * 1.852 * (interval_min / 60.0)
        heading_rad = np.radians(heading)
        d_north = dist_km * np.cos(heading_rad)
        d_east = dist_km * np.sin(heading_rad)
        lat, lon = _deg_offset_km(lat, lon, d_north, d_east)
        heading += rng.normal(0, heading_jitter)
        t += timedelta(minutes=interval_min)
    return rows


def generate_demo_ais(seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    all_rows = []

    # 1) VESSEL_STRONG: approaches, passes directly through origin at origin_time
    start_t = ORIGIN_TIME - timedelta(hours=3)
    start_lat, start_lon = _deg_offset_km(ORIGIN_LAT, ORIGIN_LON, d_north_km=-30, d_east_km=-10)
    all_rows += _track(
        mmsi="111111111", vessel_name="VESSEL STRONG", vessel_type="Tanker",
        start_lat=start_lat, start_lon=start_lon, start_time=start_t,
        heading_deg=25, speed_knots=10, n_points=36, interval_min=5, rng=rng,
    )

    # 2) VESSEL_CLOSE_WRONGTIME: passes near origin location, but 8 hours earlier
    start_t = ORIGIN_TIME - timedelta(hours=11)
    start_lat, start_lon = _deg_offset_km(ORIGIN_LAT, ORIGIN_LON, d_north_km=-25, d_east_km=15)
    all_rows += _track(
        mmsi="222222222", vessel_name="VESSEL CLOSE WRONGTIME", vessel_type="Cargo",
        start_lat=start_lat, start_lon=start_lon, start_time=start_t,
        heading_deg=200, speed_knots=12, n_points=30, interval_min=5, rng=rng,
    )

    # 3) VESSEL_FARTIME: near the correct time, but stays ~40km away the whole time
    start_t = ORIGIN_TIME - timedelta(hours=2)
    start_lat, start_lon = _deg_offset_km(ORIGIN_LAT, ORIGIN_LON, d_north_km=40, d_east_km=40)
    all_rows += _track(
        mmsi="333333333", vessel_name="VESSEL FARTIME", vessel_type="Cargo",
        start_lat=start_lat, start_lon=start_lon, start_time=start_t,
        heading_deg=150, speed_knots=9, n_points=24, interval_min=5, rng=rng,
    )

    # 4) VESSEL_ERRATIC: unusual speed/heading swings, but nowhere near origin
    start_t = ORIGIN_TIME - timedelta(hours=4)
    start_lat, start_lon = _deg_offset_km(ORIGIN_LAT, ORIGIN_LON, d_north_km=-80, d_east_km=60)
    rows = _track(
        mmsi="444444444", vessel_name="VESSEL ERRATIC", vessel_type="Fishing",
        start_lat=start_lat, start_lon=start_lon, start_time=start_t,
        heading_deg=90, speed_knots=6, n_points=40, interval_min=5, rng=rng,
        heading_jitter=40.0, speed_jitter=4.0,
    )
    all_rows += rows

    # 5) Background vessels: normal traffic scattered around, unrelated
    bg_specs = [
        (5, -60, -60, 300, 14, "Cargo"),
        (6, 70, -20, 260, 8, "Tanker"),
        (7, -10, 90, 340, 11, "Cargo"),
        (8, 90, 5, 180, 13, "Container"),
        (9, -90, -90, 60, 9, "Fishing"),
        (10, 20, -100, 100, 15, "Cargo"),
        (11, -50, 100, 220, 7, "Fishing"),
        (12, 100, -50, 310, 12, "Tanker"),
        (13, -20, -30, 45, 10, "Cargo"),
        (14, 60, 60, 275, 9, "Container"),
        (15, -100, 20, 130, 11, "Cargo"),
    ]
    for idx, d_north, d_east, heading, speed, vtype in bg_specs:
        start_t = ORIGIN_TIME - timedelta(hours=float(rng.integers(1, 10)))
        s_lat, s_lon = _deg_offset_km(ORIGIN_LAT, ORIGIN_LON, d_north_km=d_north, d_east_km=d_east)
        mmsi = f"9{idx:08d}"  # valid 9-digit MMSI-style id, unique per background vessel
        all_rows += _track(
            mmsi=mmsi, vessel_name=f"VESSEL BG {idx}", vessel_type=vtype,
            start_lat=s_lat, start_lon=s_lon, start_time=start_t,
            heading_deg=heading, speed_knots=speed,
            n_points=int(rng.integers(15, 30)), interval_min=5, rng=rng,
        )

    df = pd.DataFrame(all_rows)

    # Inject some realistic messiness: duplicates, a few missing values, unsorted order
    dup_rows = df.sample(n=5, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)
    df.loc[df.sample(frac=0.02, random_state=seed).index, "vessel_name"] = None
    df.loc[df.sample(frac=0.02, random_state=seed + 1).index, "vessel_type"] = None
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle order

    return df


if __name__ == "__main__":
    df = generate_demo_ais()
    out_path = "data/ais_demo.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} AIS rows for {df['mmsi'].nunique()} vessels to {out_path}")