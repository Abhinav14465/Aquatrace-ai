import streamlit as st
import pandas as pd

from attribution.attribution import rank_vessels


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AquaTrace AI",
    page_icon="🌊",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🌊 AquaTrace AI")
st.subheader("Marine Oil Spill Detection & Vessel Attribution")

st.write(
    "AI-assisted analysis of potential oil-spill incidents using "
    "satellite observations and AIS vessel trajectories."
)

st.divider()


# -----------------------------
# Demo incident parameters
# -----------------------------
st.header("🛰️ Spill Incident")

col1, col2, col3 = st.columns(3)

with col1:
    origin_lat = st.number_input(
        "Estimated Origin Latitude",
        value=12.70
    )

with col2:
    origin_lon = st.number_input(
        "Estimated Origin Longitude",
        value=80.45
    )

with col3:
    origin_time = st.text_input(
        "Estimated Spill Time (UTC)",
        value="2026-09-19T16:00:00+00:00"
    )


st.divider()


# -----------------------------
# Analyze button
# -----------------------------
if st.button("🔍 Analyze Vessel Attribution", type="primary"):

    with st.spinner("Analyzing AIS vessel trajectories..."):

        try:
            ais_file = "data/ais_demo.csv"

            result = rank_vessels(
                ais_file,
                origin_lat,
                origin_lon,
                origin_time,
                uncertainty_km=5,
                search_radius_km=20,
                time_window_hours=12,
                top_n=5
            )

            candidates = result["candidates"]

            st.success("Analysis completed!")

            # -----------------------------
            # Summary
            # -----------------------------
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Vessels in Search Area",
                    result["meta"]["n_vessels_in_window"]
                )

            with col2:
                st.metric(
                    "Candidate Vessels",
                    len(candidates)
                )

            with col3:
                st.metric(
                    "Search Radius",
                    f'{result["meta"]["search_radius_km"]} km'
                )

            st.divider()

            # -----------------------------
            # Candidate vessel table
            # -----------------------------
            st.header("🚢 Candidate Vessels")

            if candidates:

                display_data = []

                for vessel in candidates:
                    display_data.append({
                        "Rank": vessel["rank"],
                        "Vessel": vessel["vessel_name"],
                        "MMSI": vessel["mmsi"],
                        "Type": vessel["vessel_type"],
                        "Distance (km)": vessel["minimum_distance_km"],
                        "Time Difference (h)": vessel["time_difference_hours"],
                        "Correlation Score": round(
                            vessel["final_score"], 3
                        )
                    })

                df = pd.DataFrame(display_data)

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                # -----------------------------
                # Detailed explanation
                # -----------------------------
                st.header("📋 Attribution Explanation")

                for vessel in candidates:
                    with st.expander(
                        f'Rank #{vessel["rank"]} — {vessel["vessel_name"]}'
                    ):
                        st.write(vessel["explanation"])

                        st.write(
                            f'**Final correlation score:** '
                            f'{vessel["final_score"]:.3f}'
                        )

            else:
                st.warning(
                    "No vessels were found within the selected "
                    "time and geographic window."
                )

            # -----------------------------
            # Disclaimer
            # -----------------------------
            st.divider()

            st.info(
                "⚠️ Demo note: The vessel score represents potential "
                "source-vessel correlation. It does not prove that a "
                "vessel caused the spill."
            )

        except Exception as e:

            st.error("Something went wrong while running the analysis.")

            st.code(str(e))