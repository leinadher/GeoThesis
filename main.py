# app.py

import streamlit as st
import pydeck as pdk
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pyproj
import json
import joblib
import time
import numpy as np
from geopy.geocoders import Nominatim

from scripts.depth_query import get_depth_info
from scripts.features import compute_features  # <- import your utility function
from scripts.predict_energy import predict_energy_yield
from scripts.geocode import reverse_geocode

def load_svg_icon(path: str) -> str:
    with open(path, "r") as file:
        return file.read()

# ───────────────────────────────
# App Setup
# ───────────────────────────────

st.set_page_config(layout="wide")

svg_icon = load_svg_icon("assets/geowatt.svg")

# Create two columns
col1, col2 = st.columns([0.4, 5])  # Adjust ratio as needed

with col1:
    st.image(svg_icon, width=200)

with col2:
    # Display title and subtitle in the right column
    st.title("GeoWatt ZH")
    st.subheader("Shallow Geothermal Potential in the Canton of Zürich.")

st.markdown("---")

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

@st.cache_data
def load_hex_layer():
    hex_gdf = gpd.read_file("data/transformed/hex_inverted_density_potential.geojson")
    return hex_gdf.to_crs(epsg=4326)

# Load all cached data
boundary = load_boundary()
restrictions_gdf = load_restrictions()
zh_geothermal_probes_gdf = load_geothermal_probes()
borehole_tree = load_borehole_tree()
hex_gdf = load_hex_layer()
hex_gdf["potential_score"] = hex_gdf["potential_score"].round(2)

# Optimize for quicker loading
# hex_gdf['geometry'] = hex_gdf['geometry'].simplify(tolerance=10)

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
col1, spacer, col2 = st.columns([1, 0.1, 1])

## COLUMN 1 ##

with col1:

    # Load Zurich boundary in WGS84 for pydeck
    boundary_geo = boundary.__geo_interface__  # GeoJSON format

    # Get current point (if any)
    lat, lon = st.session_state.clicked_coords

    with st.expander("Map Layers", expanded=False):
        show_canton = st.checkbox("Cantonal Border", value=True)
        show_hex = st.checkbox("Potential by Density", value=False)
        show_boreholes = st.checkbox("Approved Installations", value=False)

    # Create pydeck layers
    layers = []
    tooltip = None

    if show_hex:
        hex_layer = pdk.Layer(
            "GeoJsonLayer",
            data=hex_gdf,
            get_fill_color="""
                [
                    255 * (1 - potential_score), 
                    255 * potential_score,
                    100,
                    140
                ]
                """,
            pickable=True,
            stroked=False,
            filled=True,
            auto_highlight=True,
        )

        tooltip = {
                "html": """
                    <b>Approved Installations:</b> {borehole_density}<br/>
                    <b>Potential Score:</b> {potential_score}
                """,
                "style": {
                    "backgroundColor": "rgba(30, 30, 30, 0.9)",
                    "color": "white"
                }
            }

        layers.append(hex_layer)

    if show_boreholes:
        borehole_layer = pdk.Layer(
            "ScatterplotLayer",
            data=zh_geothermal_probes_gdf,
            get_position='[lon, lat]',
            get_fill_color='[91, 144, 247, 255]',
            radius_min_pixels=2,
            radius_max_pixels=6,
            pickable=True,
            radius_scale=1
        )

        tooltip = {
            "html": """
                <b>Yield:</b> {Waermeentnahme} kW<br/>
                <b>Return:</b> {Waermeeintrag} kW<br/>
                <b>Depth:</b> {Sondentiefe} m<br/>
                <b># Probes:</b> {Gesamtsondenzahl}
            """,
            "style": {
                "backgroundColor": "rgba(30, 30, 30, 0.9)",
                "color": "white"
            }
        }

        layers.append(borehole_layer)

    # Boundary outline (black line)
    if show_canton:
        boundary_layer = pdk.Layer(
            "GeoJsonLayer",
            data=boundary_geo,
            get_line_color=[0, 0, 0],
            line_width_min_pixels=2,
            filled=False,
        )
        layers.append(boundary_layer)

    # Marker icon
    icon_data = [{
        "position": [lon, lat],
        "lat": lat,
        "lon": lon,
        "icon": "marker"
    }]

    icon_layer = pdk.Layer(
        "IconLayer",
        data=icon_data,
        get_icon="icon",
        get_size=2,
        size_scale=15,
        get_position="position",
        icon_atlas="https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png",
        icon_mapping={
            "marker": {
                "x": 0,
                "y": 0,
                "width": 128,
                "height": 128,
                "anchorY": 128,
                
            }
        },
        pickable=False
    )
    layers.append(icon_layer)

    # Create deck object
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/outdoors-v11",
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=10,
            pitch=0,
        ),
        layers=layers,
        tooltip=tooltip
    )

    # Display pydeck map
    st.pydeck_chart(deck)
 
    ### Initialize geolocator ###
    geolocator = Nominatim(user_agent="geowatt_zh")

    # Input field for search
    st.markdown("##### 🔍 Select Location")
    query = st.text_input("Type an address or place (e.g. Herrliberg, ETH...)", placeholder="Search")

    # Search result handling
    if query.strip():  # Only run if the query isn't empty or just whitespace
        try:
            location = geolocator.geocode(query, exactly_one=True, addressdetails=True, timeout=5)

            if location:
                if st.button(f"📍 {location.address}"):
                    lat, lon = location.latitude, location.longitude
                    st.session_state.clicked_coords = (lat, lon)
                    st.session_state.trigger_analysis = False
                    st.rerun()  # Optional: refresh map and UI
            else:
                st.warning("❌ No matching location found. Try a more specific name.")

        except Exception as e:
            st.error(f"⚠️ Geocoding error: {str(e)}")

    # Add coordinate selector below the map
    with st.expander("Adjust Coordinates"):
        lat = st.number_input("Latitude", value=lat, step=0.0001, format="%.6f")
        lon = st.number_input("Longitude", value=lon, step=0.0001, format="%.6f")

    # Update state when changed
    if (lat, lon) != st.session_state.clicked_coords:
        st.session_state.clicked_coords = (lat, lon)
        st.session_state.trigger_analysis = False

## COLUMN 2 ##

with col2:
    if st.session_state.clicked_coords:
        with st.spinner("⏳ Processing location..."):
            lat, lon = st.session_state.clicked_coords
            location_name = reverse_geocode(lat, lon)
            with st.container():
                st.markdown("### 📍 Current Location", unsafe_allow_html=True)
            time.sleep(0.2)
            st.markdown(f"##### {location_name}")
            st.markdown(f"##### Coordinates: `{lat:.5f}, {lon:.5f}`")

        # Analysis button
        if st.button("🔍 Analyse Potential"):
            st.session_state.trigger_analysis = True

        # Run analysis if triggered
        if st.session_state.get("trigger_analysis", False):
            with st.spinner("⏳ Processing..."):
                time.sleep(0.5)
                restriction_status, features = compute_features(
                    lat, lon,
                    to_lv95,
                    restrictions_gdf,
                    borehole_tree,
                    zh_geothermal_probes_gdf,
                    get_depth_info
                )

                if features:
                    ### Legal restriction output ###
                    if restriction_status == "Allowed":
                        st.success("✅ Drilling is allowed.")
                    elif "conditions" in restriction_status:
                        st.warning("⚠️ Drilling is allowed with conditions.")
                    else:
                        st.error(f"⛔ Drilling is not allowed.")

                    ### Display computed features ###
                    # Define mapping from internal keys (used in features.py) to display labels
                    display_keys = {
                        "elevation": "Elevation (m)",
                        "depth_max": "Max allowed depth (m)",
                        "count_100m": "Boreholes within 100m",
                        "nearest_borehole_dist": "Nearest borehole dist (m)"
                    }

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Elevation (m)", value=f"{features['elevation']:.2f}")
                        st.metric(label="Boreholes within 100m", value=features['count_100m'])

                    with col2:
                        st.metric(label="Max allowed depth (m)", value=f"{features['depth_max']:.2f}")
                        st.metric(label="Nearest borehole dist (m)", value=f"{features['nearest_borehole_dist']:.2f}")
                    
                    ### Energy yield prediction block ###
                    st.markdown("---")
                    st.markdown("### 🔋 Energy Yield Estimation")

                    selected_depth = st.slider(
                        "Select probe depth (Sondentiefe in m)",
                        min_value=10,
                        max_value=int(features["depth_max"]),
                        value=min(150, int(features["depth_max"])),  # default to 150 or less than max
                        step=5,
                        help="Maximum allowed depth for probes based on location restrictions."
                    )

                    gesamtsondenzahl = st.slider(
                        "Select number of probes",
                        min_value=1,
                        max_value=6,
                        value=3,
                        step=1,
                        help="Total number of geothermal probes to be installed at this location."
                    )

                    if st.button("⚡ Estimate Energy Yield"):
                        bottom_elevation = features.get("elevation") - selected_depth

                        # Create new dictionary with required features for the model
                        features_for_model = {
                            "Gesamtsondenzahl": gesamtsondenzahl,
                            "count_100m": features.get("count_100m"),
                            "nearest_borehole_dist": features.get("nearest_borehole_dist"),
                            "Sondentiefe": selected_depth,
                            "bottom_elevation": bottom_elevation
                        }

                        with st.spinner("⏳ Processing result..."):
                            prediction = predict_energy_yield(features_for_model)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(label="Estimated Energy Yield (kW)", value=f"{prediction:.0f}")

                            with col2:
                                st.metric(label="\# Probes", value=round(gesamtsondenzahl))


                else:
                    st.error(restriction_status)
    else:
        st.info("Click on the map to select a location.")


# ───────────────────────────────
# UI Layout – Bottom Info
# ───────────────────────────────

tab1, tab2 = st.tabs(["🧭 About", "⚠️ Limitations"])

with tab1:
    st.markdown("""
    **GeoWatt ZH** is an interactive tool for assessing shallow geothermal potential in the canton of Zürich.
    Users can select a location to view drilling permissions, estimated borehole depth, elevation, potential energy yield,
    and eligibility for public financial incentives.

    The tool focuses on **Erdwärmesonden (EWS)** systems—vertical borehole heat exchangers used to extract heat from the ground.
    Suitability is estimated using official spatial data and borehole records from the [Kanton Zürich Wärmenutzungsatlas](https://maps.zh.ch/?offlayers=bezirkslabels&scale=320000&srid=2056&topic=AwelGSWaermewwwZH&x=2692500&y=1252500).

    The tool is intended as a **proof of concept**, designed to simplify access to public datasets and make geothermal planning more accessible to a broader audience.
    """)


with tab2:
    st.markdown("""
    While **GeoWatt ZH** provides helpful spatial insights, it is subject to some limitations:
    - It only includes data **within the boundaries of the canton of Zürich**; boreholes in adjacent cantons or regions are not taken into account.
    - Subsurface complexity — including **geological variability, thermal regeneration, and hydrogeological dynamics** — is **not modeled** in this version.
    - The dataset includes both **installed and approved boreholes** without distinguishing between the two, which may affect interpretation of thermal density.
    - Legal regulations and zoning restrictions are subject to change. Users should **always consult official cantonal authorities** before making planning decisions.
    """)