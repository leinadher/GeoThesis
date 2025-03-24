import requests
import streamlit as st

@st.cache_data
def reverse_geocode(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "zoom": 18,
        "addressdetails": 1
    }
    headers = {
        "User-Agent": "geowatt_zh"
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        address = data.get("address", {})

        house_number = address.get("house_number", "")
        road = address.get("road", "")
        postcode = address.get("postcode", "")
        city = address.get("city", "") or address.get("town", "") or address.get("village", "")
        country = address.get("country", "")

        if country == "Schweiz/Suisse/Svizzera/Svizra":
            country = "Switzerland"

        # Combine road + number
        street_line = f"{road} {house_number}".strip()
        city_line = f"{postcode} {city}".strip()

        lines = [street_line, city_line, country]
        return ", ".join([line for line in lines if line])

    except Exception as e:
        return f"Reverse geocoding failed: {e}"