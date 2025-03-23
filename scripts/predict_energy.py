import joblib
import streamlit as st
import numpy as np

@st.cache_resource
def load_energy_model():
    return joblib.load("models/xgb_zh_energy_yield.pkl")

def predict_energy_yield(features_dict):
    model = load_energy_model()

    # Define required features and their order
    required_features = [
        "Gesamtsondenzahl",
        "count_100m",
        "nearest_borehole_dist",
        "Sondentiefe",
        "bottom_elevation"
    ]

    # Ensure all required features are present
    try:
        feature_array = np.array([[features_dict[feat] for feat in required_features]])
    except KeyError as e:
        raise ValueError(f"Missing required feature: {e}")

    # Predict using XGBRegressor
    prediction = model.predict(feature_array)[0]
    return prediction
