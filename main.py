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
import io

from scripts.depth_query import get_depth_info
from scripts.features import compute_features  # <- import your utility function
from scripts.predict_energy import predict_energy_yield
from scripts.geocode import reverse_geocode

from scipy.stats import percentileofscore

def load_svg_icon(path: str) -> str:
    with open(path, "r") as file:
        return file.read()
    
### SYSTEM COMPARISON ###

def show_performance_comparison(pred_kw, depth, sondenzahl, zh_geothermal_probes_gdf):
    from scipy.stats import percentileofscore
    import matplotlib.pyplot as plt

    if depth is None or sondenzahl is None:
        st.warning("Missing input values for comparison.")
        return

    similar_sites = zh_geothermal_probes_gdf[
        (zh_geothermal_probes_gdf["Sondentiefe"].between(depth - 20, depth + 20)) &
        (zh_geothermal_probes_gdf["Gesamtsondenzahl"] == sondenzahl)
    ].copy()

    if similar_sites.empty:
        st.info("No comparable boreholes found in canton records.")
        return

    percentile = percentileofscore(similar_sites["Waermeentnahme"], pred_kw)

    st.markdown(f"""
    Your predicted yield of **{pred_kw:.1f} kW** outperforms **{percentile:.0f}%**
    of installations in Zürich with **{sondenzahl} probes** and **depths similar (±20 m) to {depth}  m**.
    """)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(similar_sites["Waermeentnahme"], bins=25, color="grey", edgecolor="white", alpha=0.7)
    ax.axvline(pred_kw, color=(91/255, 144/255, 247/255), linewidth=2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("Heat Yield (kW)")
    ax.set_ylabel("Number of Sites")
    ax.grid(True, color="#dddddd", linestyle="-", linewidth=0.6, axis="y")
    ax.set_axisbelow(True)
    ax.set_title("")
    ax.legend().set_visible(False)
    
    # Save as SVG in memory and show
    svg_buffer = io.StringIO()
    fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
    svg_data = svg_buffer.getvalue()
    plt.close(fig)
    st.components.v1.html(f"<div style='text-align:center'>{svg_data}</div>", height=500)

### HEAT YIELD UI ###
@st.fragment
def show_yield_ui(features):
    st.markdown("### 🔋 Heat Yield Estimation")

    st.slider(
        "Select probe depth (m)",
        min_value=10,
        max_value=int(features["depth_max"]),
        value=st.session_state.selected_depth,
        step=5,
        key="selected_depth"
    )

    st.slider(
        "Select number of probes",
        min_value=1,
        max_value=10,
        value=st.session_state.gesamtsondenzahl,
        step=1,
        key="gesamtsondenzahl"
    )

    if st.button("⚡ Estimate Heat Yield"):
        st.session_state.run_yield = True

    if st.session_state.get("run_yield", False):
        bottom_elevation = features.get("elevation") - st.session_state.selected_depth
        features_for_model = {
            "Gesamtsondenzahl": st.session_state.gesamtsondenzahl,
            "count_100m": features.get("count_100m"),
            "nearest_borehole_dist": features.get("nearest_borehole_dist"),
            "Sondentiefe": st.session_state.selected_depth,
            "bottom_elevation": bottom_elevation
        }

        with st.spinner("⏳ Processing result..."):
            prediction = predict_energy_yield(features_for_model)
            st.session_state.prediction = prediction
            
            # Save parameters from the prediction:
            st.session_state.prediction_sondenzahl = st.session_state.gesamtsondenzahl
            st.session_state.prediction_depth = st.session_state.selected_depth

            # Reset yield process to avoid reloading UI
            st.session_state.run_yield = False

    if "prediction" in st.session_state and st.session_state.prediction is not None:
        conversion_option = st.selectbox(
            "Select unit for estimated yield:",
            options=[
                "Instantaneous Power (kW)",
                "Daily Energy (kWh/day)",
                "Annual Energy (kWh/year)",
                "Annual Energy (MWh/year)"
            ],
            index=0
        )

        pred_kw = st.session_state.get("prediction", 0)
        full_load_hours = 2000
        if conversion_option == "Instantaneous Power (kW)":
            converted = f"{pred_kw:,.1f} kW"
        elif conversion_option == "Daily Energy (kWh/day)":
            converted = f"{pred_kw * 24:,.1f} kWh"
        elif conversion_option == "Annual Energy (kWh/year)":
            converted = f"{pred_kw * full_load_hours:,.0f} kWh"
        else:
            converted = f"{(pred_kw * full_load_hours) / 1000:,.1f} MWh"

        # Results
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Yield", value=converted)
        col2.metric(label="Depth", value=f"{st.session_state.get('prediction_depth', 0)} m")
        col3.metric(label="# Probes", value=st.session_state.get("prediction_sondenzahl", 0))

        with st.expander("📊 Performance Comparison", expanded=False):
            show_performance_comparison(
                pred_kw=st.session_state.prediction,
                depth=st.session_state.prediction_depth,
                sondenzahl=st.session_state.prediction_sondenzahl,
                zh_geothermal_probes_gdf=zh_geothermal_probes_gdf
            )

        with st.expander("💰 Financial Incentives", expanded=False):
            st.markdown("Based on the [2025 Förderprogramm](https://www.zh.ch/de/umwelt-tiere/energie/energiefoerderung.html) for Erdwärmesonden:")

            # Assume prediction exists
            pred_kw = st.session_state.prediction

            # User toggle for bonus eligibility
            include_bonus = st.checkbox("Include optional bonus for frost-free regeneration system?", value=True)

            # Base subsidy calculation
            if pred_kw <= 15:
                base_subsidy = 6800
            else:
                base_subsidy = 6800 + 420 * (pred_kw - 15)

            base_subsidy = round(base_subsidy)

            # Optional bonus
            bonus_subsidy = 0
            if include_bonus:
                capped_kw = min(pred_kw, 70)
                bonus_subsidy = 3000 + 100 * capped_kw
                bonus_subsidy = round(bonus_subsidy)

            total = base_subsidy + bonus_subsidy

            # Display nicely
            st.markdown(f"""
            - **Base subsidy:** CHF **{base_subsidy:,}**
            {"- **Bonus (no frost protection):** CHF **" + f"{bonus_subsidy:,}**" if include_bonus else ""}
            - **Estimated total:** CHF **{total:,}**

            _Exact amount subject to approval. Values based on official canton incentives as of 2025._
            """)

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
    st.subheader("Shallow Geothermal Potential in the Canton of Zürich")

st.markdown("---")

# ───────────────────────────────
# Load Data
# ───────────────────────────────

@st.cache_resource
def load_boundary():
    return gpd.read_file("data/raw/zh_boundary.geojson").to_crs(epsg=4326)

@st.cache_data
def load_restrictions():
    restrictions_gdf = gpd.read_file("data/transformed/zh_combined_restrictions.geojson").to_crs(epsg=2056)
    color_mapping = {
        "Allowed": [0, 200, 0, 100],
        "Allowed with conditions": [255, 200, 0, 100],
        "Not allowed": [200, 0, 0, 100]
    }
    restrictions_gdf["color"] = restrictions_gdf["restrictions"].map(color_mapping)
    return restrictions_gdf

@st.cache_data
def load_geothermal_probes():
    return gpd.read_file("data/transformed/zh_geothermal_probes_with_density_elevation.geojson").set_crs(epsg=2056, allow_override=True)

@st.cache_resource
def load_borehole_tree():
    return joblib.load("data/borehole_tree.pkl")

@st.cache_data
def load_hex_layer():
    hex_gdf = gpd.read_file("data/transformed/hex_inverted_density_potential.geojson")
    return hex_gdf.to_crs(epsg=2056)

# Load all cached data
with st.spinner('⏳ Loading data...'):
    boundary = load_boundary()

    restrictions_gdf = load_restrictions()

    restrictions_map = restrictions_gdf
    restrictions_map = restrictions_gdf.copy()
    restrictions_map["geometry"] = restrictions_map["geometry"].simplify(tolerance=1.5, preserve_topology=True)
    restrictions_map["geometry"] = restrictions_map.buffer(0)

    zh_geothermal_probes_gdf = load_geothermal_probes()
    borehole_tree = load_borehole_tree()
    hex_gdf = load_hex_layer()
    hex_gdf["potential_score"] = hex_gdf["potential_score"].round(2)
    hex_gdf["color"] = hex_gdf["potential_score"].apply(lambda p: [255 * (1-p), 255 * p, 100, 140])

# Function to render probes nearby
def filter_boreholes_near_location(lat, lon, probes_gdf, distance_threshold_m=1000):
    """Return probes within 'distance_threshold_m' meters of a given lat/lon."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    x, y = transformer.transform(lon, lat)

    # Calculate distance in projected coordinates
    probes_gdf["distance_m"] = probes_gdf.geometry.distance(Point(x, y))

    # Filter nearby probes
    nearby = probes_gdf[probes_gdf["distance_m"] <= distance_threshold_m].copy()

    # Reproject to WGS84
    nearby = nearby.to_crs(epsg=4326)

    # Extract lat/lon from geometry
    nearby["lon"] = nearby.geometry.x
    nearby["lat"] = nearby.geometry.y

    return nearby

# Function to render only nearby hexagons
def filter_hexes_near_location(lat, lon, hex_gdf, distance_threshold_m=1000):
    """Return hexagons whose centroids are within 'distance_threshold_m' meters of a lat/lon."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    x, y = transformer.transform(lon, lat)

    # Calculate distance between centroids and clicked point
    hex_gdf["distance_m"] = hex_gdf.centroid.distance(Point(x, y))

    # Filter hexes
    nearby = hex_gdf[hex_gdf["distance_m"] <= distance_threshold_m].copy()

    # Very tiny simplify to speed up rendering
    nearby["geometry"] = nearby["geometry"].simplify(tolerance=1, preserve_topology=True)

    return nearby

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

    st.markdown("", unsafe_allow_html=True)

    # Load Zurich boundary in WGS84 for pydeck
    boundary_geo = boundary.__geo_interface__  # GeoJSON format

    # Get current point (if any)
    lat, lon = st.session_state.clicked_coords
 
    with st.expander("📚 Map Layers", expanded=False):
        show_canton = st.checkbox("Cantonal Border", value=True)
        show_boreholes = st.checkbox("Approved Installations (within 1 km)", value=False)
        show_hex = st.checkbox("Potential by Density (within 1 km)", value=False)
        show_restrictions = st.checkbox("Drilling Restrictions", value=False)

    # Create pydeck layers
    layers = []
    tooltip = None
    
    if show_restrictions:
        restrictions_layer = pdk.Layer(
            "GeoJsonLayer",
            data=restrictions_map.to_crs(epsg=4326),
            get_fill_color="color",
            pickable=False,
            stroked=False,
            filled=True,
            auto_highlight=False
        )
        layers.append(restrictions_layer)

    # HEX Layer
    if show_hex:
        nearby_hexes = filter_hexes_near_location(lat, lon, hex_gdf).to_crs(epsg=4326)
        nearby_hexes["Waermeentnahme"] = "-"
        nearby_hexes["Sondentiefe"] = "-"
        nearby_hexes["Gesamtsondenzahl"] = "-"
        hex_layer = pdk.Layer(
            "GeoJsonLayer",
            data=nearby_hexes,
            get_fill_color="color",
            pickable=True,
            stroked=False,
            filled=True,
            auto_highlight=True,
        )
        layers.append(hex_layer)

    # Borehole Layer
    if show_boreholes:
        nearby_probes = filter_boreholes_near_location(lat, lon, zh_geothermal_probes_gdf)
        nearby_probes["potential_score"] = "-"
        nearby_probes["color"] = nearby_probes["Gesamtsondenzahl"].apply(
            lambda x: [
                min(50 + 20 * x, 255),
                100,
                200,
                160
            ]
        )
        borehole_layer = pdk.Layer(
            "ScatterplotLayer",
            data=nearby_probes,
            get_position='[lon, lat]',
            get_fill_color='color',
            radius_min_pixels=3,
            radius_max_pixels=6,
            pickable=True,
            radius_scale=1
        )
        layers.append(borehole_layer)

    if show_boreholes and show_hex:
        tooltip = {
            "html": """
                <b>EWS Installation:</b> {Waermeentnahme} kW / {Sondentiefe} m / {Gesamtsondenzahl} probes<br/>
                <b>Density Score:</b> {potential_score}
            """,
            "style": {
                "backgroundColor": "rgba(30, 30, 30, 0.9)",
                "color": "white"
            },
            "null_value": "-"
        }
    elif show_boreholes:
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
    elif show_hex:
        tooltip = {
            "html": """
            <b>Approved Installations:</b> {borehole_density}<br/>
            <b>Density Score:</b> {potential_score}
            """,
            "style": {
                "backgroundColor": "rgba(30, 30, 30, 0.9)",
                "color": "white"
            }
        }
    else:
        tooltip = None

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
            zoom=14,
            pitch=0,
            max_pitch=0
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
                    
                    # Reset entire state
                    st.session_state.trigger_analysis = False
                    st.session_state.prediction = None
                    st.session_state.run_yield = False
                    st.session_state.prediction_sondenzahl = None
                    st.session_state.prediction_depth = None

                    st.rerun()  # refresh map and UI
            else:
                st.warning("No matching location found. Try a more specific name.")

        except Exception as e:
            st.error(f"⚠️ Geocoding error: {str(e)}")

    # Add coordinate selector below the map
    with st.expander("🧭 Adjust Coordinates"):
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

            # Reset downstream session state
            st.session_state.prediction = None
            st.session_state.run_yield = False
            st.session_state.prediction_sondenzahl = None
            st.session_state.prediction_depth = None
            st.session_state.selected_depth = None
            st.session_state.gesamtsondenzahl = None

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
                        st.metric(label="Elevation (m)", value=f"{features['elevation']:,.1f}")
                        st.metric(label="Boreholes within 100m", value=features['count_100m'])

                    with col2:
                        st.metric(label="Max allowed depth (m)", value=f"{features['depth_max']:,.1f}")
                        st.metric(label="Nearest installation (m)", value=f"{features['nearest_borehole_dist']:,.1f}")
                    
                    # Energy yield prediction block
                    st.session_state.selected_depth = int(features["depth_max"]/2)
                    st.session_state.gesamtsondenzahl = 5

                    show_yield_ui(features)
                        
                else:
                    st.error(restriction_status)
    else:
        st.info("Click on the map to select a location.")

# ───────────────────────────────
# UI Layout – Bottom Info
# ───────────────────────────────

tab1, tab2, tab3 = st.tabs(["ℹ️ About", "⚠️ Limitations", "🧮 Unit Conversions"])

with tab1:
    st.markdown("""
    **GeoWatt ZH** is an interactive tool for estimating shallow geothermal potential in the canton of Zürich.
    Users can select a location to view drilling permissions, estimated borehole depth, elevation, potential energy yield,
    and additional information.

    The tool focuses on geothermal borehole heat exchangers (**Erdwärmesonden**), used to extract energy from the ground for building heating.
    Suitability is estimated with a machine learning algorithm using open spatial data and official records from the [Kanton Zürich Wärmenutzungsatlas](https://maps.zh.ch/?offlayers=bezirkslabels&scale=320000&srid=2056&topic=AwelGSWaermewwwZH&x=2692500&y=1252500).

    The tool is intended as a **proof of concept**, designed to simplify access to public datasets and make geothermal planning more accessible to a broader audience, and does not replace case-specific technical assessments.
    """)


with tab2:
    st.markdown("""
    While **GeoWatt ZH** provides helpful spatial insights, it is subject to some limitations:
    - It only includes data **within the boundaries of the Canton of Zürich**; boreholes in adjacent cantons or regions are not taken into account.
    - Subsurface variables, including **geological variability, thermal regeneration, and hydrogeological dynamics**, is **not modelled** in this version.
    - The source dataset includes both **installed and approved boreholes** without distinguishing between the two, which may affect interpretation of thermal density.
    - Legal regulations and zoning restrictions are subject to change. Users should **always consult official cantonal authorities** before making planning decisions.
    """)

with tab3:
    st.markdown("""
    The estimated **heat extraction power** (kW) is converted into daily and annual energy yields using standard assumptions:
                
    - **Daily Energy (kWh/day)**, assumes continuous operation for 24 hours: Power (kW) × 24  
    - **Annual Energy (kWh/year)**, assumes 2000 full-load hours per year: Power (kW) × 2000  
    - **Annual Energy (MWh/year)**, simple conversion to megawatt-hours: Annual Energy (kWh) ÷ 1000  
      
    The value of **2000 full-load hours per year** is based on the Swiss standard **SIA 384/6**, which recommends typical operating times for geothermal heat extraction systems in Switzerland.
""")