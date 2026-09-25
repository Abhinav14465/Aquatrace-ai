"""
Run the AIS vessel attribution demo end-to-end.

    python run_demo.py

Generates synthetic AIS data (if not already present), runs rank_vessels()
against a mock spill origin, prints the ranked candidates, and saves an
interactive map to data/attribution_map.html.
"""

import os
import pandas as pd

from attribution.attribution import rank_vessels
from generate_demo_data import generate_demo_ais, ORIGIN_LAT, ORIGIN_LON, ORIGIN_TIME

AIS_CSV_PATH = "data/ais_demo.csv"
MAP_OUTPUT_PATH = "data/attribution_map.html"


def ensure_demo_data():
    if not os.path.exists(AIS_CSV_PATH):
        os.makedirs("data", exist_ok=True)
        df = generate_demo_ais()
        df.to_csv(AIS_CSV_PATH, index=False)
        print(f"Generated {len(df)} synthetic AIS rows -> {AIS_CSV_PATH}")
    return pd.read_csv(AIS_CSV_PATH)


def print_candidates(candidates):
    if not candidates:
        print("No candidate vessels found in the given search window.")
        return

    header = f"{'Rank':<5}{'MMSI':<12}{'Vessel':<26}{'Type':<10}{'Dist(km)':<10}{'ΔT(h)':<8}{'Score':<7}"
    print(header)
    print("-" * len(header))
    for c in candidates:
        print(f"{c['rank']:<5}{c['mmsi']:<12}{c['vessel_name']:<26}{c['vessel_type']:<10}"
              f"{c['minimum_distance_km']:<10}{c['time_difference_hours']:<8}{c['final_score']:<7}")

    print("\nTop candidate explanations:")
    for c in candidates[:3]:
        print(f"  #{c['rank']} {c['explanation']}")


def main():
    ais_df = ensure_demo_data()

    result = rank_vessels(
        ais_df,
        origin_lat=ORIGIN_LAT,
        origin_lon=ORIGIN_LON,
        origin_time=ORIGIN_TIME.isoformat(),
        uncertainty_km=5,
        search_radius_km=60,
        time_window_hours=12,
    )

    print(f"\nOrigin: ({ORIGIN_LAT}, {ORIGIN_LON}) @ {ORIGIN_TIME.isoformat()}")
    print(f"Vessels in input: {result['meta']['n_vessels_input']} | "
          f"Vessels in search window: {result['meta']['n_vessels_in_window']}\n")

    print_candidates(result["candidates"])
    print(f"\nNote: {result['meta']['note']}")

    try:
        from attribution.visualization import build_attribution_map
        m = build_attribution_map(result, ORIGIN_LAT, ORIGIN_LON, uncertainty_km=5)
        m.save(MAP_OUTPUT_PATH)
        print(f"\nMap saved to {MAP_OUTPUT_PATH}")
    except ImportError:
        print("\n(folium not installed — skipping map generation)")


if __name__ == "__main__":
    main()