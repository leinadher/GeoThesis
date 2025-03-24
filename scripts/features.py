import numpy as np
from shapely.geometry import Point

def compute_features(lat, lon, to_lv95, restrictions_gdf, borehole_tree, zh_geothermal_probes_gdf, get_depth_info):
    """Compute model-relevant features for a selected location."""

    # Convert coordinates to Swiss LV95
    lv95_x, lv95_y = to_lv95.transform(lon, lat)
    user_point = Point(lv95_x, lv95_y)

    # Check if the point is in an allowed area
    restriction_status = restrictions_gdf[restrictions_gdf.contains(user_point)]
    if restriction_status.empty:
        return "❌ Invalid location, must be within Zürich canton limits!", None

    restriction_value = restriction_status.iloc[0]["restrictions"]
    if restriction_value not in ["Allowed", "Allowed with conditions"]:
        return f"⛔ Drilling not allowed.", None

    # Elevation & depth
    elevation, depth_max = get_depth_info(lat, lon)

    # Count boreholes within 100m
    neighbors_within_100m = borehole_tree.query_ball_point([lv95_x, lv95_y], 100)
    count_100m = len(neighbors_within_100m)

    # Always get nearest borehole
    distances, _ = borehole_tree.query([lv95_x, lv95_y], k=1)
    nearest_borehole_dist = distances
    
    # Normalize values
    count_100m_norm = count_100m / zh_geothermal_probes_gdf["count_100m"].max()
    nearest_borehole_dist_norm = nearest_borehole_dist / zh_geothermal_probes_gdf["nearest_borehole_dist"].max()

    # Estimate bottom elevation
    sondentiefe = depth_max
    bottom_elevation = elevation - sondentiefe

    features = {
        "elevation": elevation,
        "depth_max": depth_max,
        "count_100m": count_100m,
        "nearest_borehole_dist": nearest_borehole_dist,
        "count_100m_norm": count_100m_norm,
        "nearest_borehole_dist_norm": nearest_borehole_dist_norm,
        "bottom_elevation": bottom_elevation
    }

    return restriction_value, features