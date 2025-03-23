# app.py

import streamlit as st
from streamlit_folium import st_folium
import folium
import geopandas as gpd
from shapely.geometry import Point
import pyproj
import json
import joblib
import time
import numpy as np

from scripts.features import compute_features  # <- import your utility function

# ───────────────────────────────
# App Setup
# ───────────────────────────────
st.set_page_config(layout="wide")
st.title("GeoWatt ZH")
st.markdown("Interactive tool for assessing shallow geothermal potential in the canton of Zürich.")

# ───────────────────────────────
# Load Data
# ───────────────────────────────

@st.cache_resource
def load_boundary():
    return gpd.read_file("data/raw/zh_boundary.geojson").to_crs(epsg=4326)

@st.cache_data
def load_restrictions():
    return gpd.read_file("data/transformed/zh_combined_restrictions.geojson").to_crs(epsg=2056)

@st.cache_data
def load_geothermal_probes():
    return gpd.read_file("data/transformed/zh_geothermal_probes_with_density_elevation.geojson").to_crs("EPSG:2056")

@st.cache_resource
def load_borehole_tree():
    return joblib.load("data/borehole_tree.pkl")

# Load all cached data
boundary = load_boundary()
restrictions_gdf = load_restrictions()
zh_geothermal_probes_gdf = load_geothermal_probes()
borehole_tree = load_borehole_tree()

# Coordinate transformer: WGS84 → LV95
to_lv95 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

# ───────────────────────────────
# UI Layout – Default settings
# ───────────────────────────────

# Set up session state for clicked point
if "clicked_coords" not in st.session_state:
    # Default point: Zürich center
    st.session_state.clicked_coords = (47.3769, 8.5417)

# Optional: also initialize trigger_analysis state
if "trigger_analysis" not in st.session_state:
    st.session_state.trigger_analysis = False

# ───────────────────────────────
# UI Layout – Two Columns
# ───────────────────────────────
col1, col2 = st.columns([1, 1])  # Left = map, Right = info

## COLUMN 1 ##

with col1:
    st.subheader("🗺️ Map")
    center = boundary.geometry.centroid.iloc[0].coords[0][::-1]
    m = folium.Map(location=center, zoom_start=10)

    # Add boundary
    folium.GeoJson(
        boundary.geometry,
        style_function=lambda feature: {
            "fillColor": "#000000",
            "color": "#000000",
            "weight": 2,
            "fillOpacity": 0
        }
    ).add_to(m)

    # Always display marker if we have coords
    if st.session_state.clicked_coords:
        lat, lon = st.session_state.clicked_coords
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
            tooltip="Selected location"
        ).add_to(m)

    m.add_child(folium.LatLngPopup())
    result = st_folium(m, height=600, width=700)

    # Immediately update coordinates on click
    if result and result.get("last_clicked"):
        new_lat = result["last_clicked"]["lat"]
        new_lon = result["last_clicked"]["lng"]
        st.session_state.clicked_coords = (new_lat, new_lon)

        st.session_state.trigger_analysis = False


## COLUMN 2 ##

with col2:
    st.subheader("📋 Location Information")

    if st.session_state.clicked_coords:
        lat, lon = st.session_state.clicked_coords
        st.markdown(f"**Coordinates:** `{lat:.5f}, {lon:.5f}`")

        # Analysis button (sets flag in session state)
        if st.button("🔍 Analyse Location"):
            st.session_state.trigger_analysis = True

        # Only run analysis if button was clicked
        if st.session_state.get("trigger_analysis", False):
            with st.spinner("⏳ Processing..."):
                time.sleep(0.2)
                restriction_status, features = compute_features(
                    lat, lon,
                    to_lv95,
                    restrictions_gdf,
                    borehole_tree,
                    zh_geothermal_probes_gdf,
                    get_depth_info
                )

                if features:
                    # Status
                    if restriction_status == "Allowed":
                        st.success("✅ Drilling is allowed.")
                    elif "conditions" in restriction_status:
                        st.warning("⚠️ Drilling allowed with conditions.")
                    else:
                        st.error(f"⛔ {restriction_status}")

                    # Feature list
                    st.markdown("#### 🔍 Computed Site Features")
                    for label, value in features.items():
                        if isinstance(value, float):
                            st.markdown(f"- **{label}**: `{value:.2f}`")
                        else:
                            st.markdown(f"- **{label}**: `{value}`")
                else:
                    st.error(restriction_status)