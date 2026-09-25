"""
Folium map visualization for ranked vessel attribution results.
Draws the estimated origin, uncertainty zone, and vessel trajectories
color-coded by final_score, with a dark nautical theme.
"""

import folium
from folium import plugins


def _score_color(score):
    """Red-hot for high suspicion, cooling down to blue for low suspicion."""
    if score >= 0.7:
        return "#ff3b3b"       # red — prime suspect
    if score >= 0.55:
        return "#ff9d3b"       # orange
    if score >= 0.4:
        return "#ffd93b"       # yellow
    return "#4fa8ff"           # blue — low suspicion


def _score_label(score):
    if score >= 0.7:
        return "High suspicion"
    if score >= 0.55:
        return "Moderate suspicion"
    if score >= 0.4:
        return "Low-moderate suspicion"
    return "Low suspicion"


def build_attribution_map(result: dict, origin_lat, origin_lon, uncertainty_km=5):
    """
    Build a folium.Map from a rank_vessels() result dict.
    Returns the Map object; caller can .save(path) it.
    """
    m = folium.Map(
        location=[origin_lat, origin_lon],
        zoom_start=9,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # Optional alternate basemap the viewer can switch to
    folium.TileLayer("CartoDB positron", name="Light map").add_to(m)

    # --- Spill origin: pulsing beacon ---------------------------------
    pulse_html = f"""
    <div style="position: relative; width: 22px; height: 22px;">
        <div style="
            position: absolute; top: 50%; left: 50%;
            width: 14px; height: 14px; margin: -7px 0 0 -7px;
            background: #00e5ff; border-radius: 50%;
            box-shadow: 0 0 8px 2px #00e5ff;
            z-index: 2;">
        </div>
        <div style="
            position: absolute; top: 50%; left: 50%;
            width: 14px; height: 14px; margin: -7px 0 0 -7px;
            background: rgba(0,229,255,0.6); border-radius: 50%;
            animation: pulse 1.6s infinite;
            z-index: 1;">
        </div>
    </div>
    <style>
        @keyframes pulse {{
            0%   {{ transform: scale(1);   opacity: 0.8; }}
            70%  {{ transform: scale(3.2); opacity: 0;   }}
            100% {{ transform: scale(3.2); opacity: 0;   }}
        }}
    </style>
    """
    folium.Marker(
        [origin_lat, origin_lon],
        tooltip="⚠ Estimated spill origin",
        icon=folium.DivIcon(html=pulse_html, icon_size=(22, 22), icon_anchor=(11, 11)),
    ).add_to(m)

    folium.Circle(
        [origin_lat, origin_lon],
        radius=uncertainty_km * 1000,
        color="#00e5ff",
        weight=1.5,
        dash_array="6,6",
        fill=True,
        fill_color="#00e5ff",
        fill_opacity=0.07,
        tooltip=f"Uncertainty zone ({uncertainty_km} km)",
    ).add_to(m)

    # --- Vessel trajectories -------------------------------------------
    geojson_map = result.get("trajectories_geojson", {})
    candidates = result.get("candidates", [])

    for candidate in candidates:
        mmsi = candidate["mmsi"]
        feature = geojson_map.get(mmsi)
        if not feature:
            continue

        coords = feature["geometry"]["coordinates"]  # [lon, lat] pairs
        latlon_coords = [(lat, lon) for lon, lat in coords]
        if not latlon_coords:
            continue

        rank = candidate["rank"]
        score = candidate["final_score"]
        color = _score_color(score)
        is_top = rank <= 3

        line = folium.PolyLine(
            latlon_coords,
            color=color,
            weight=4 if is_top else 2,
            opacity=0.9 if is_top else 0.55,
            tooltip=(f"#{rank} {candidate['vessel_name']} — "
                     f"{_score_label(score)} (score={score})"),
        ).add_to(m)

        # Directional arrows along the path so you can see heading toward origin
        plugins.PolyLineTextPath(
            line,
            "   ►   ",
            repeat=True,
            offset=6,
            attributes={"fill": color, "font-weight": "bold", "font-size": "13"},
        ).add_to(m)

        # Rank-badge marker at the vessel's last known position
        badge_size = 30 if is_top else 22
        badge_html = f"""
        <div style="
            width: {badge_size}px; height: {badge_size}px;
            background: {color};
            border: 2px solid white; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            color: #111; font-weight: bold; font-size: {12 if is_top else 10}px;
            box-shadow: 0 0 6px rgba(0,0,0,0.6);
            font-family: sans-serif;">
            {rank}
        </div>
        """
        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 180px;">
            <b>#{rank} {candidate['vessel_name']}</b><br>
            <span style="color:{color}; font-weight:bold;">{_score_label(score)}</span><br>
            <hr style="margin:4px 0;">
            MMSI: {mmsi}<br>
            Type: {candidate.get('vessel_type', 'N/A')}<br>
            Distance: {candidate['minimum_distance_km']} km<br>
            Time Δ: {candidate['time_difference_hours']} h<br>
            Score: {score}<br>
            <i>{candidate.get('explanation', '')}</i>
        </div>
        """
        folium.Marker(
            latlon_coords[-1],
            tooltip=f"#{rank} {candidate['vessel_name']} ({score})",
            popup=folium.Popup(popup_html, max_width=280),
            icon=folium.DivIcon(html=badge_html, icon_size=(badge_size, badge_size),
                                 icon_anchor=(badge_size // 2, badge_size // 2)),
        ).add_to(m)

    # --- Legend -----------------------------------------------------------
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 20px; z-index: 9999;
        background: rgba(20,20,20,0.85); color: white;
        padding: 12px 16px; border-radius: 10px;
        font-family: sans-serif; font-size: 13px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);">
        <b>Suspicion Level</b><br>
        <span style="color:#ff3b3b;">●</span> High (≥0.70)<br>
        <span style="color:#ff9d3b;">●</span> Moderate (0.55–0.69)<br>
        <span style="color:#ffd93b;">●</span> Low-moderate (0.40–0.54)<br>
        <span style="color:#4fa8ff;">●</span> Low (&lt;0.40)<br>
        <hr style="margin:6px 0; border-color:#444;">
        <span style="color:#00e5ff;">◉</span> Spill origin &amp; uncertainty zone
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # --- Nice-to-haves -------------------------------------------------
    plugins.Fullscreen(position="topright").add_to(m)
    plugins.MiniMap(toggle_display=True, tile_layer="CartoDB positron").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    return m