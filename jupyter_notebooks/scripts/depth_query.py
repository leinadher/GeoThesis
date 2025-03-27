import requests
import pyproj
import re
from bs4 import BeautifulSoup

def get_depth_info(lat, lon):
    """Fetches elevation and maximum allowed depth from Zürich maps API given WGS84 coordinates."""
    
    # Convert WGS84 to LV95 (Swiss Coordinate System)
    wgs84_to_lv95 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    lv95_x, lv95_y = wgs84_to_lv95.transform(lon, lat)
    
    # Construct the query URL with LV95 coordinates
    url = f"https://maps.zh.ch/topics/query?_dc=1716541069669&infoQuery=%7B%22queryTopics%22%3A%5B%7B%22level%22%3A%22main%22%2C%22topic%22%3A%22AwelGSWaermewwwZH%22%2C%22divCls%22%3A%22legmain%22%2C%22layers%22%3A%22bohrtiefenbegrenzung-oeffentlich%22%7D%5D%7D&scale=3702.64&srid=2056&bbox={lv95_x}%2C{lv95_y}%2C{lv95_x}%2C{lv95_y}"

    # Make the GET request
    response = requests.get(url)
    
    if response.status_code != 200:
        print("⚠️ Error fetching data from Zürich maps.")
        return None, None  # Return None for both values if the request fails

    # Parse HTML response
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()

    # Extract elevation (Höhe: 416.1 m)
    elevation_match = re.search(r"Höhe:\s*([\d.]+)\s*m", text)
    elevation = float(elevation_match.group(1)) if elevation_match else None

    # Extract max depth (e.g., "244 Meter ab Terrain")
    depth_match = re.search(r"(\d+)\s*Meter ab Terrain", text)
    depth_max = float(depth_match.group(1)) if depth_match else None  

    # Logic for default depth assignment
    if elevation is None:
        depth_max = None  # Outside Zürich → depth should be None
    elif depth_max is None:
        depth_max = 400  # Inside Zürich but no depth found → default to 400

    return elevation, depth_max