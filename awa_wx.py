import re
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import io
from datetime import datetime, timezone
from astral import LocationInfo
from astral.sun import sun
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, HPacker, TextArea
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# import discord # <-- REMOVED
# from discord.ext import commands  # <-- REMOVED
# import asyncio # <-- REMOVED
import matplotlib.patheffects as pe # Import the path effects library
from scipy.interpolate import griddata 
from descartes import PolygonPatch 
# from metpy.plots import USCOUNTIES # <-- NOTE: This is US-ONLY, cannot be used for WA
from mpl_toolkits.axes_grid1 import make_axes_locatable # <-- IMPORT FOR ROBUST COLORBAR
# --- IMPORTS for robust clipping ---
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from cartopy.mpl.patch import geos_to_path
import matplotlib.patches as patches
import traceback # Added for error handling

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64 x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json'
}

# --- BOT BOILERPLATE (REMOVED) ---

# --- Weather Icon Generation (No changes needed) ---
def create_icon_base(fig_size=(1,1), dpi=100):
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return fig, ax

def create_sunny_icon():
    fig, ax = create_icon_base()
    ax.add_patch(patches.Circle((0.5, 0.5), 0.4, color='yellow'))
    for angle in range(0, 360, 45):
        x = 0.5 + 0.45 * np.cos(np.radians(angle))
        y = 0.5 + 0.45 * np.sin(np.radians(angle))
        ax.plot([0.5, x], [0.5, y], color='yellow')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

def create_moonlight_icon():
    fig, ax = create_icon_base()
    ax.add_patch(patches.Circle((0.5, 0.5), 0.4, color='lightgray'))
    ax.add_patch(patches.Circle((0.6, 0.6), 0.35, color='darkgray'))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

def create_partially_cloudy_sun_icon():
    fig, ax = create_icon_base()
    # Sun
    ax.add_patch(patches.Circle((0.3, 0.3), 0.3, color='yellow'))
    for angle in range(0, 360, 45):
        x = 0.3 + 0.35 * np.cos(np.radians(angle))
        y = 0.3 + 0.35 * np.sin(np.radians(angle))
        ax.plot([0.3, x], [0.3, y], color='yellow')
    # Cloud
    ax.add_patch(patches.Ellipse((0.7, 0.7), 0.6, 0.4, color='white'))
    ax.add_patch(patches.Circle((0.6, 0.65), 0.2, color='white'))
    ax.add_patch(patches.Circle((0.8, 0.65), 0.2, color='white'))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

def create_partially_cloudy_moon_icon():
    fig, ax = create_icon_base()
    # Moon
    ax.add_patch(patches.Circle((0.3, 0.3), 0.3, color='lightgray'))
    ax.add_patch(patches.Circle((0.4, 0.4), 0.25, color='darkgray'))
    # Cloud
    ax.add_patch(patches.Ellipse((0.7, 0.7), 0.6, 0.4, color='white'))
    ax.add_patch(patches.Circle((0.6, 0.65), 0.2, color='white'))
    ax.add_patch(patches.Circle((0.8, 0.65), 0.2, color='white'))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

def create_cloudy_icon():
    fig, ax = create_icon_base()
    ax.add_patch(patches.Ellipse((0.5, 0.5), 0.8, 0.5, color='gray'))
    ax.add_patch(patches.Circle((0.4, 0.55), 0.3, color='gray'))
    ax.add_patch(patches.Circle((0.6, 0.55), 0.3, color='gray'))
    ax.add_patch(patches.Circle((0.3, 0.45), 0.25, color='gray'))
    ax.add_patch(patches.Circle((0.7, 0.45), 0.25, color='gray'))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

def create_mostly_cloudy_rain_icon():
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(patches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
    ax.add_patch(patches.Circle((0.4, 0.75), 0.3, color='gray'))
    ax.add_patch(patches.Circle((0.6, 0.75), 0.3, color='gray'))
    # Rain
    for i in range(3):
        ax.plot([0.3 + i*0.2, 0.25 + i*0.2], [0.4, 0.2], color='blue')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

def create_mostly_cloudy_storming_icon():
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(patches.Ellipse((0.5, 0.7), 0.8, 0.5, color='darkgray'))
    ax.add_patch(patches.Circle((0.4, 0.75), 0.3, color='darkgray'))
    ax.add_patch(patches.Circle((0.6, 0.75), 0.3, color='darkgray'))
    # Lightning
    ax.add_patch(patches.Polygon([[0.4, 0.4], [0.5, 0.6], [0.6, 0.4], [0.5, 0.5]], color='yellow'))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

def create_fog_or_mist_icon():
    fig, ax = create_icon_base()
    for y in [0.3, 0.5, 0.7]:
        ax.plot([0.1, 0.9], [y, y], color='lightgray', linewidth=5, alpha=0.7)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

def create_snowing_icon():
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(patches.Ellipse((0.5, 0.8), 0.8, 0.4, color='gray'))
    # Snowflakes
    for pos in [(0.3,0.5), (0.5,0.4), (0.7,0.5), (0.4,0.3), (0.6,0.3)]:
        ax.text(pos[0], pos[1], '*', color='white', fontsize=20)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

def create_wintry_mix_icon():
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(patches.Ellipse((0.5, 0.8), 0.8, 0.4, color='gray'))
    # Rain and Snow
    ax.plot([0.3, 0.25], [0.5, 0.3], color='blue')
    ax.plot([0.5, 0.45], [0.5, 0.3], color='blue')
    ax.text(0.7, 0.4, '*', color='white', fontsize=20)
    ax.text(0.6, 0.3, '*', color='white', fontsize=20)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

def create_rain_icon():
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(patches.Ellipse((0.5, 0.8), 0.8, 0.4, color='gray'))
    # Rain
    for i in range(5):
        ax.plot([0.2 + i*0.15, 0.15 + i*0.15], [0.6, 0.4], color='blue')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

WEATHER_ICONS = {
    'Sunny': create_sunny_icon(),
    'Moonlight': create_moonlight_icon(),
    'Partially Cloudy (Sun)': create_partially_cloudy_sun_icon(),
    'Partially Cloudy (Moon)': create_partially_cloudy_moon_icon(),
    'Cloudy': create_cloudy_icon(),
    'Mostly Cloudy & Rain': create_mostly_cloudy_rain_icon(),
    'Mostly Cloudy & Storming': create_mostly_cloudy_storming_icon(),
    'Fog or Mist': create_fog_or_mist_icon(),
    'Snowing': create_snowing_icon(),
    'Wintry Mix': create_wintry_mix_icon(),
    'Rain': create_rain_icon(),
}

# --- NOTE: Configuration updated for Western Australia ---
MAP_EXTENT = [112.5, 129.5, -35.5, -13.5] # [lon_min, lon_max, lat_min, lat_max]

# --- Main plotted cities (use METAR) ---
CITIES = {
    "Perth": (-31.94, 115.965, "YPPH"),
    "Broome": (-17.945, 122.235, "YBRM"),
    "Kalgoorlie": (-30.789, 121.462, "YPKG"),
    "Port Hedland": (-20.378, 118.626, "YPPD"),
    "Kununurra": (-15.778, 128.708, "YPKU"),
    "Learmonth": (-22.236, 114.089, "YPLM"),
    "Geraldton": (-28.796, 114.706, "YGEL"),
    "Busselton": (-33.687, 115.400, "YBLN"),
    "Jandakot": (-32.097, 115.881, "YPJT"),
    "Curtin": (-17.581, 123.828, "YCIN"),
    "Forrest": (-30.838, 128.100, "YFRT"),
    "Newman": (-23.418, 119.803, "YNWN"),
    "Paraburdoo": (-23.171, 117.746, "YPBO"),
    
    # --- Added for gradient interpolation ---
    "Alice Springs": (-23.807, 133.903, "YBAS"),
    "Woomera": (-31.144, 136.817, "YPWR"),
}

# --- Expanded stations for better interpolation (using BoM JSON) ---
# These stations do not use METAR, but provide a public JSON feed.
BoM_STATIONS = {
    "Wiluna": (-26.617, 120.225, "95439"),
    "Warburton": (-26.128, 126.583, "94457"),
    "Wyndham": (-15.487, 128.123, "95214"),
    "Derby": (-17.370, 123.661, "95205"),
    "Laverton": (-28.613, 122.424, "94449"),
    "Meekatharra": (-26.612, 118.548, "94430"),
    "Eucla": (-31.707, 128.875, "94647"),
}


# --- Offsets for the main plotted cities ---
CITY_OFFSETS = {
    # SW Corner - Aggressive diagonal separation
    "Perth": (-1.05, 0.30),       # Far Left, slightly up
    "Jandakot": (0.95, 0.1),     # Far Right, slightly up
    "Busselton": (-0.2, -0.7),  # Centered below, far down
    "Geraldton": (-0.3, -0.15),
    
    # North West
    "Learmonth": (-0.2, -0.15),
    "Port Hedland": (0.1, 0.15),
    
    # North (Broome/Derby/Curtin) - Spread horizontally and vertically
    "Broome": (-0.3, 0.15),       # Left
    "Derby": (0.0, 0.95),         # Center, but higher
    "Curtin": (0.3, 0.15),       # Right
    
    # North-East (Wyndham/Kununurra) - Spread horizontally
    "Wyndham": (-1.00, 0.15),      # Left
    "Kununurra": (0.2, 0.05),     # Right
    
    # Mid-East (Meekatharra/Wiluna) - Spread horizontally
    "Meekatharra": (-0.4, 0.15),
    "Wiluna": (0.4, 0.15),
    
    # East (Forrest/Eucla) - Spread vertically
    "Forrest": (0, 0.2),         # Up
    "Eucla": (-0.35, -0.65),        # Down

    # Central/Other
    "Kalgoorlie": (0, 0.15),
    "Warburton": (0, 0.15),
    "Laverton": (0, 0.15),
    "Newman": (0.2, 0.15),
    "Paraburdoo": (-0.2, 0.15),

    # Interpolation Stations
    "Alice Springs": (0, 0.15),
    "Woomera": (0, 0.15),
}

# ---
FIG_BG_COLOR = '#b0c4de'
LAND_COLOR = '#f5f5dc'
OCEAN_COLOR = '#b0e0e6'
STATE_BORDER_COLOR = '#8b4513'
COUNTY_BORDER_COLOR = '#27522b' # Kept for if you add shires
MAJOR_ROAD_COLOR = '#696969'
SECONDARY_ROAD_COLOR = '#a9a9a9'
BBOX_STYLE = dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.85, edgecolor='lightgray', linewidth=0.5)
FONT_SIZE_CITY = 11
FONT_SIZE_DATA = 9
FONT_SIZE_WEATHER = 9
FONT_SIZE_ICAO = 6 
FONT_SIZE_TITLE = 20
FONT_SIZE_LEGEND = 10
FONT_FAMILY = 'serif'
ALERT_COLORS = {
    "Extreme": "#FF0000",
    "Severe": "#FFA500",
    "Moderate": "#FFFF00",
    "Minor": "#00FFFF",
    "Unknown": "#808080"
}
ALERT_ALPHA = 0.25

plt.rcParams['font.family'] = FONT_FAMILY

def get_metar(icao, hoursback=0, format='json'):
    try:
        # NOTE: This API is international and should work for Australian ICAOs
        metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format={format}&hours={hoursback}'
        # --- ADDED HEADERS ---
        response = requests.get(metar_url, timeout=10, headers=HEADERS)
        src = response.content.decode('utf-8')
        if not src.strip():
            raise ValueError("Empty response from METAR API")
        json_data = json.loads(src)
        
        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")

        # ... (rest of function is unchanged) ...
        raw_metar = json_data[0]['rawOb']
        main_body = raw_metar.split('RMK')[0]

        temp_c, dew_point_c = None, None
        temp_dew_pattern = re.search(r'\b(M?\d{2})/(M?\d{2})\b', main_body)
        if temp_dew_pattern:
            temp_str, dew_str = temp_dew_pattern.groups()
            temp_c = int(temp_str.replace('M', '-'))
            dew_point_c = int(dew_str.replace('M', '-'))

        wind_direction, wind_speed = None, None
        wind_pattern = re.search(r'(\d{3}|VRB)(\d{2})(G\d{2})?KT', main_body)
        if wind_pattern:
            wind_dir_str = wind_pattern.group(1)
            wind_direction = 0 if wind_dir_str == 'VRB' else int(wind_dir_str)
            wind_speed = int(wind_pattern.group(2))
        
        altimeter_inHg = None
        alt_pattern_us = re.search(r'\bA(\d{4})\b', main_body)
        if alt_pattern_us:
            alt_raw = alt_pattern_us.group(1)
            altimeter_inHg = float(alt_raw) / 100.0
        else:
            # NOTE: Australian METARs often use QNH (HPa)
            alt_pattern_q = re.search(r'\bQ(\d{4})\b', main_body)
            if alt_pattern_q:
                alt_raw = alt_pattern_q.group(1)
                alt_hpa = float(alt_raw)
                altimeter_inHg = round(alt_hpa * 0.02953, 2)
                
        return raw_metar, temp_c, dew_point_c, wind_direction, wind_speed, altimeter_inHg

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching METAR data for {icao}: {e}")
    except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
        raise Exception(f"Error parsing METAR data for {icao}: {e}")

def get_bom_json(wmo_id):
    """
    Fetches the latest observation data from the BoM's public JSON feed.
    These stations do not use METAR but are part of the WA observation network.
    """
    try:
        # This is the public JSON feed for WA state observations.
        url = f"http://www.bom.gov.au/fwo/IDW60801/IDW60801.{wmo_id}.json"
        # --- ADDED HEADERS ---
        response = requests.get(url, timeout=10, headers=HEADERS)
        response.raise_for_status() # Raise error for bad responses (404, 500, etc.)
        
        data = response.json()
        
        # Get the *latest* observation from the list
        obs = data["observations"]["data"][0]
        
        raw_ob = f"BoM JSON: {obs.get('local_date_time_full', 'N/A')}"
        
        # ... (rest of function is unchanged) ...
        temp_c = obs.get("air_temp")
        dew_point_c = obs.get("dewpt")
        
        wind_dir_str = obs.get("wind_dir")
        wind_direction_deg = obs.get("wind_dir_deg")
        if wind_direction_deg is None:
            wind_direction = 0 # Handle null direction
        else:
            wind_direction = 0 if wind_dir_str == 'CALM' else int(wind_direction_deg)
            
        # Convert km/h (JSON default) to knots (METAR default)
        wind_spd_kmh = obs.get("wind_spd_kmh", 0)
        if wind_spd_kmh is None: wind_spd_kmh = 0
        wind_speed = round(wind_spd_kmh * 0.539957) # km/h to knots
        
        # Get QNH (pressure at sea level)
        alt_hpa = obs.get("press_qnh") 
        altimeter_inHg = None
        if alt_hpa is not None:
            altimeter_inHg = round(alt_hpa * 0.02953, 2)
            
        # These stations don't provide cloud/weather phenomena in this JSON
        cloud_info = {"low": [], "mid": [], "high": [], "vertical_visibility": None}
        weather_codes = []

        return (raw_ob, temp_c, dew_point_c, wind_direction, 
                wind_speed, altimeter_inHg, cloud_info, weather_codes)

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching BoM JSON for {wmo_id}: {e}")
    except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
        raise Exception(f"Error parsing BoM JSON for {wmo_id}: {e}")


def extract_cloud_info(metar):
    cloud_levels = {"low": [], "mid": [], "high": [], "vertical_visibility": None}
    main_body = metar.split('RMK')[0]
    cloud_pattern = re.compile(r'(FEW|SCT|BKN|OVC)(\d{3})|VV(\d{3})')
    cloud_matches = re.findall(cloud_pattern, main_body)
    for match in cloud_matches:
        if match[0]:
            cover = match[0]
            altitude_hundreds = int(match[1])
            altitude_ft = altitude_hundreds * 100
            if altitude_ft <= 6500:
                cloud_levels["low"].append((cover, altitude_ft))
            elif 6500 < altitude_ft <= 20000:
                cloud_levels["mid"].append((cover, altitude_ft))
            else:
                cloud_levels["high"].append((cover, altitude_ft))
        if match[2]:
            vv_hundreds = int(match[2])
            cloud_levels["vertical_visibility"] = vv_hundreds * 100
    return cloud_levels

def extract_weather_phenomena(metar):
    main_body = metar.split('RMK')[0]
    weather_pattern = re.compile(
        r'(-|\+|VC)?(TS|SH|FZ|DR|BL|MI|BC|PR)?(DZ|RA|SN|SG|IC|PL|GR|GS|UP)?(BR|FG|FU|VA|DU|SA|HZ|PY)?(PO|SQ|FC|SS|DS)?'
    )
    parts = main_body.split()
    weather_conditions = []
    
    for part in parts:
        if re.match(r'^(FEW|SCT|BKN|OVC|VV)\d{3}$', part) or \
           re.match(r'^\d{4}$', part) or \
           re.match(r'^M?\d{2}/M?\d{2}$', part) or \
           re.match(r'^[AQ]\d{4}$', part) or \
           re.match(r'^\d+SM$', part):
            continue
        
        match = re.match(weather_pattern, part)
        if match and any(match.groups()):
            weather_conditions.append(part)
            
    return weather_conditions

def map_metar_weather_to_condition(weather_codes, cloud_layers, is_day):
    condition = None
    
    if any('TS' in code for code in weather_codes):
        condition = 'Mostly Cloudy & Storming'
    elif any('FZRA' in code or 'PL' in code or 'PE' in code for code in weather_codes):
        condition = 'Wintry Mix'
    elif any('SN' in code for code in weather_codes):
        condition = 'Snowing'
    elif any('RA' in code or 'SHRA' in code or 'DZ' in code for code in weather_codes):
        condition = 'Mostly Cloudy & Rain' if cloud_layers else 'Rain'
    elif any('FG' in code or 'BR' in code for code in weather_codes):
        condition = 'Fog or Mist'
    elif cloud_layers:
        covers = [layer[0] for layer in cloud_layers]
        if any(cover in ['BKN', 'OVC'] for cover in covers):
            condition = 'Cloudy'
        elif any(cover in ['FEW', 'SCT'] for cover in covers):
            condition = 'Partially Cloudy (Sun)' if is_day else 'Partially Cloudy (Moon)'
        else:
            condition = 'Sunny' if is_day else 'Moonlight'
    else:
        condition = 'Sunny' if is_day else 'Moonlight'
        
    return condition

def is_daytime(lat, lon):
    loc = LocationInfo(latitude=lat, longitude=lon)
    try:
        s = sun(loc.observer, date=datetime.now(timezone.utc))
        now = datetime.now(timezone.utc)
        return s['sunrise'] <= now <= s['sunset']
    except Exception:
        return True 

def get_bom_alerts():
    """
    Fetches active weather warnings for Western Australia from BoM's JSON feed.
    Uses a more reliable URL and robust, correctly-ordered error handling.
    """
    try:
        url = 'http://www.bom.gov.au/wa/warnings/data.json'
        response = requests.get(url, timeout=10, headers=HEADERS)
        
        # Check for HTTP errors (like 403, 404, 500)
        response.raise_for_status() 

        # Check if response text is empty *before* trying to parse
        if not response.text:
            print("BoM Alerts: Received empty response. Assuming no alerts.")
            return [] 

        # This is the line that can raise JSONDecodeError
        data = response.json()
        
        alerts = []
        
        if isinstance(data, list):
            for warning_item in data:
                warning_info = warning_item.get('warning_info', {})
                if not warning_info:
                    continue
                
                headline = warning_info.get('headline', 'Unknown Warning')
                event_type = warning_info.get('type', 'Unknown')
                
                affected_areas_str = "Western Australia"
                area_desc = warning_info.get('area_desc', '')
                if area_desc:
                    match = re.search(r'for\s(.*?)\.', area_desc, re.IGNORECASE)
                    if match:
                        affected_areas_str = match.group(1)

                severity = warning_info.get('severity', 'Unknown').capitalize()
                if severity not in ALERT_COLORS:
                    severity = 'Unknown'
                
                color = ALERT_COLORS.get(severity, ALERT_COLORS['Unknown'])
                
                polygons = []
                
                alerts.append({
                    "headline": headline,
                    "event": event_type,
                    "severity": severity,
                    "polygons": polygons,
                    "color": color,
                    "affected_areas": affected_areas_str
                })
        
        return alerts
        
    # --- RE-ORDERED EXCEPTION BLOCKS ---
    except json.JSONDecodeError as e:
        # CATCH THIS FIRST! This is the "Expecting value" error.
        # This happens if the server returns HTML (like a 503 page) instead of JSON.
        print(f"Error fetching BoM alerts (JSONDecodeError): {e}. Response was not valid JSON.")
        # We know why this fails, no need to print the HTML anymore.
        return []
        
    except requests.exceptions.RequestException as e:
        # This catches network errors (timeout, DNS, 4xx/5xx HTTP errors)
        print(f"Error fetching BoM alerts (RequestException): {e}")
        return []
    except Exception as e:
        # Catch any other unexpected error
        print(f"Error fetching BoM alerts (Unexpected): {e}")
        return []
    
# --- *** MAIN MAP FUNCTION (HEAVILY UPDATED) *** ---

# @bot.command(name="awawx", help="Generates a detailed weather map for Western Australia.") # <-- REMOVED
# async def awawx(ctx: commands.Context): # <-- REPLACED
def generate_wa_weather_map():
    """
    Generate and save the Western Australia weather map to a file.
    """
    
    # await ctx.typing() # <-- REMOVED
    print("Fetching weather data...")
    
    # --- NEW: COMBINED STATION LOGIC ---
    
    # 1. Create one master list of all stations to plot
    # (ICAO, is_metar_station)
    ALL_STATIONS = {}
    
    # Add CITIES (which use METAR)
    for city, (lat, lon, icao) in CITIES.items():
        ALL_STATIONS[city] = (lat, lon, icao, True) # True = use get_metar
        
    # Add BoM_STATIONS (which use JSON)
    for city, (lat, lon, wmo_id) in BoM_STATIONS.items():
        ALL_STATIONS[city] = (lat, lon, wmo_id, False) # False = use get_bom_json

    # Dictionaries to hold all data
    all_temperatures = {}
    all_dew_points = {}
    all_wind_directions = {}
    all_wind_speeds = {}
    all_altimeters = {}
    all_weather_conditions = {}
    
    # --- NEW: UNIVERSAL DATA FETCHING LOOP ---
    for city, (lat, lon, station_id, is_metar) in ALL_STATIONS.items():
        try:
            if is_metar:
                # This is an airport, use the METAR function
                (raw_metar, temp_c, dew_point_c, wind_dir, wind_spd, alt_inHg
                ) = get_metar(station_id) # station_id is an ICAO
                
                cloud_info = extract_cloud_info(raw_metar)
                weather_codes = extract_weather_phenomena(raw_metar)
            
            else:
                # This is a BoM station, use the new JSON function
                (raw_ob, temp_c, dew_point_c, wind_dir, wind_spd, alt_inHg, 
                 cloud_info, weather_codes
                ) = get_bom_json(station_id) # station_id is a WMO ID
            
            # --- Standard processing for all stations ---
            day_flag = is_daytime(lat, lon)
            combined_cloud_layers = cloud_info['low'] + cloud_info['mid'] + cloud_info['high']
            condition = map_metar_weather_to_condition(weather_codes, combined_cloud_layers, day_flag)
            
            all_temperatures[city] = temp_c
            all_dew_points[city] = dew_point_c
            all_wind_directions[city] = wind_dir
            all_wind_speeds[city] = wind_spd
            all_altimeters[city] = alt_inHg
            all_weather_conditions[city] = condition

        except Exception as e:
            print(f"Error retrieving data for {city} ({station_id}): {e}")
            all_temperatures[city] = None
            all_dew_points[city] = None
            all_wind_directions[city] = None
            all_wind_speeds[city] = None
            all_altimeters[city] = None
            all_weather_conditions[city] = None

    # Fetch alerts using the new (stubbed) function
    print("Fetching alerts...")
    alerts = get_bom_alerts()
    
    points = [] 
    temp_values = []
    u_values = []
    v_values = []

    # --- NEW: Loop over the combined ALL_STATIONS list ---
    for city, temp in all_temperatures.items():
        lat, lon, _, _ = ALL_STATIONS[city] # Get coords from our master list
        
        if temp is not None:
            points.append((lon, lat))
            temp_values.append(temp)
        
            wind_dir = all_wind_directions.get(city)
            wind_spd = all_wind_speeds.get(city)
            
            if wind_dir is not None and wind_spd is not None and wind_spd > 0:
                rad_dir = np.radians(270 - wind_dir)
                u = wind_spd * np.cos(rad_dir)
                v = wind_spd * np.sin(rad_dir)
                u_values.append(u)
                v_values.append(v)
            else:
                u_values.append(0)
                v_values.append(0)

    grid_z = None
    grid_u = None
    grid_v = None
    grid_lon, grid_lat = None, None 

    # NOTE: Interpolation over a massive, sparse area like WA may
    # produce strange-looking contours. More data points (stations) will help.
    if len(temp_values) >= 4:
        try:
            print("Interpolating data...")
            grid_lon = np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 100)
            grid_lat = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 100)
            grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
            
            grid_z_cubic = griddata(points, temp_values, (grid_x, grid_y), method='cubic')
            grid_u_cubic = griddata(points, u_values, (grid_x, grid_y), method='cubic')
            grid_v_cubic = griddata(points, v_values, (grid_x, grid_y), method='cubic')

            grid_z_nearest = griddata(points, temp_values, (grid_x, grid_y), method='nearest')
            grid_u_nearest = griddata(points, u_values, (grid_x, grid_y), method='nearest')
            grid_v_nearest = griddata(points, v_values, (grid_x, grid_y), method='nearest')

            grid_z = np.where(np.isnan(grid_z_cubic), grid_z_nearest, grid_z_cubic)
            grid_u = np.where(np.isnan(grid_u_cubic), grid_u_nearest, grid_u_cubic)
            grid_v = np.where(np.isnan(grid_v_cubic), grid_v_nearest, grid_v_cubic)
            
        except Exception as e:
            print(f"Error during interpolation: {e}")
            grid_z, grid_u, grid_v = None, None, None
    else:
        print("Not enough data to create contour map. Skipping.")

    # --- Plotting (Enhanced) ---
    print("Generating map plot...")
    fig = plt.figure(figsize=(14, 14), facecolor=FIG_BG_COLOR) 
    
    text_outline = [pe.withStroke(linewidth=3, foreground='white')]
    
    current_time = datetime.now().strftime('%Y-%m-%d %I:%M %p') 
    fig.suptitle(f"Western Australia - Detailed Weather Map\nGenerated: {current_time}",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold", y=0.91, color='navy',
                 path_effects=text_outline) 

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    # --- NOTE: Robust STATE CLIP PATCH (Changed to 'Western Australia') ---
    clip_patch = None
    try:
        # NOTE: This name must match the 'name' attribute in the shapefile
        target_states = ['Western Australia'] 
        state_geometries = []
        states_shp = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
        reader = shpreader.Reader(states_shp)
        for record in reader.records():
            state_name = record.attributes.get('name')
            if state_name in target_states:
                state_geometries.append(record.geometry)
        if not state_geometries:
            raise ValueError("Could not find Western Australia in shpreader.")
        combined_geom = unary_union(state_geometries)
        if not combined_geom:
            raise ValueError("Unary union of WA geometry failed.")
        paths = geos_to_path(combined_geom)
        if not paths:
            raise ValueError("geos_to_path returned no paths for WA.")
        combined_path = Path.make_compound_path(*paths)
        clip_patch = PathPatch(combined_path, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='none')
        if clip_patch:
            ax.add_patch(clip_patch)
        else:
            raise ValueError("PathPatch creation failed for WA.")
    except Exception as e:
        print(f"Failed to create clip patch for WA. Error: {e}")
        clip_patch = None
    if not clip_patch:
        print("Clipping patch could not be created for $wawx. Contours and barbs will not be clipped.")


    # Enhanced Map features
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor=OCEAN_COLOR)
    ax.add_feature(cfeature.LAND, zorder=0, facecolor=LAND_COLOR, edgecolor='none')

    # METAR CONTOUR PLOT (with 2C increments)
    if grid_z is not None and grid_lon is not None and grid_lat is not None:
        cmap = cm.get_cmap('jet')
        boundaries = np.arange(-10, 51, 2) # 2C INCREMENTS for WA
        norm = BoundaryNorm(boundaries=boundaries, ncolors=cmap.N, extend='both')

        contour_plot = ax.contourf(grid_lon, grid_lat, grid_z,
                              levels=boundaries, cmap=cmap, norm=norm,
                              transform=ccrs.PlateCarree(),
                              zorder=1, alpha=0.6, extend='both')
        
        if clip_patch:
            if hasattr(contour_plot, 'collections'):
                for collection in contour_plot.collections:
                    collection.set_clip_path(clip_patch)
            else:
                contour_plot.set_clip_path(clip_patch)
    
    # WIND BARB GRID (with clipping)
    if grid_u is not None and grid_v is not None and grid_lon is not None and grid_lat is not None:
        s = 7 
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
        
        barb_plot = ax.barbs(grid_lon_mesh[::s, ::s], grid_lat_mesh[::s, ::s],
                 grid_u[::s, ::s], grid_v[::s, ::s],
                 color='#4b0082', length=6, linewidth=1.0,
                 transform=ccrs.PlateCarree(), zorder=6, alpha=0.7)
        
        if clip_patch and hasattr(barb_plot, 'set_clip_path'):
            barb_plot.set_clip_path(clip_patch)

    # Add other map features ON TOP of the clipped contours/barbs
    ax.add_feature(cfeature.COASTLINE, edgecolor='navy', linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.STATES, zorder=4, linestyle="-", edgecolor=STATE_BORDER_COLOR, linewidth=1.5)
    ax.add_feature(cfeature.BORDERS, zorder=4, linestyle="-", edgecolor=STATE_BORDER_COLOR, linewidth=1.5)
    
    # --- Add Australian LGAs/Shires (Western Australia only) ---
    try:
        # !!! IMPORTANT !!!
        # This file path MUST be correct for the environment running the bot.
        # This is the most likely reason your boundaries are not appearing.
        # If the shapefiles are in the same folder as the script, just use:
        # lga_shpfilename = "AUS_2021_AUST_GDA2020.shp"
        lga_shpfilename = "/media/desoxyn/Main/AUS_2021_AUST_SHP_GDA2020/AUS_2021_AUST_GDA2020.shp"
        
        lga_reader = shpreader.Reader(lga_shpfilename)
        
        wa_lgas = []
        for record in lga_reader.records():
            state_code = record.attributes.get('STE_CODE21')  # WA = 5
            if state_code == '5':
                wa_lgas.append(record.geometry)
        
        # --- CORRECTED LOGIC ---
        # 1. Capture the single collection returned by add_geometries
        geom_collection = ax.add_geometries(wa_lgas, ccrs.PlateCarree(), facecolor='none',
                        edgecolor=COUNTY_BORDER_COLOR, linewidth=0.5, zorder=2, alpha=0.6)
        
        # 2. Apply the clip_patch to that single collection
        if clip_patch:
            geom_collection.set_clip_path(clip_patch)
            
    except Exception as e:
        # Check your console for this error! It will tell you if the file is missing.
        print(f"Error loading LGAs: {e}")

    # Enhanced Road plotting (with clipping)
    try:
        roads_shpfilename = shpreader.natural_earth(resolution='10m', category='cultural', name='roads')
        roads_reader = shpreader.Reader(roads_shpfilename)
        
        relevant_roads = []
        map_bounds = MAP_EXTENT
        
        for record in roads_reader.records():
            road_type = record.attributes.get('type', 'Unknown')
            if road_type in ['Major Highway', 'Secondary Highway']:
                geom = record.geometry
                if geom.bounds[0] <= map_bounds[1] and geom.bounds[2] >= map_bounds[0] and \
                   geom.bounds[1] <= map_bounds[3] and geom.bounds[3] >= map_bounds[2]:
                    relevant_roads.append((geom, road_type))

        for geometry, road_type in relevant_roads:
            color = MAJOR_ROAD_COLOR if road_type == 'Major Highway' else SECONDARY_ROAD_COLOR
            linewidth = 1.0 if road_type == 'Major Highway' else 0.8
            geom_patch = ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='none',
                              edgecolor=color, linestyle='-', zorder=3, alpha=0.7, linewidth=linewidth)
            if clip_patch:
                geom_patch.set_clip_path(clip_patch)
    except Exception as e:
        print(f"Error loading roads: {e}")
    
    # Add subtle gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'path_effects': text_outline}
    gl.ylabel_style = {'size': 10, 'path_effects': text_outline}

    # Colormap (with 2C increments)
    cmap = cm.get_cmap('jet')
    boundaries = np.arange(-10, 51, 2) # 2C INCREMENTS for WA
    norm = BoundaryNorm(boundaries=boundaries, ncolors=cmap.N, extend='both')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # --- *** UPDATED PLOTTING LOOP *** ---
    print("Plotting stations...")
    for city, temp in all_temperatures.items():
        base_lat, base_lon, station_id, is_metar = ALL_STATIONS[city]
        
        # --- NEW PLOT LOGIC ---
        # Plot full labels for EITHER CITIES or BoM_STATIONS
        # Any other stations (like YBAS/YPWR if not in CITIES) would be "silent"
        if city in CITIES or city in BoM_STATIONS:
            # --- FULL PLOT LOGIC ---
            lon_offset, lat_offset = CITY_OFFSETS.get(city, (0, 0.15)) # Get offset, default to 'up'
            plot_lon = base_lon + lon_offset
            plot_lat = base_lat + lat_offset
            
            # --- Check if we are plotting *outside* the map bounds ---
            # This skips drawing labels for YBAS/YPWR which are off-map
            if (plot_lon < MAP_EXTENT[0] or plot_lon > MAP_EXTENT[1] or
                plot_lat < MAP_EXTENT[2] or plot_lat > MAP_EXTENT[3]):
                
                # Still plot the *base* scatter point if it's inside
                if (temp is not None and
                    base_lon >= MAP_EXTENT[0] and base_lon <= MAP_EXTENT[1] and
                    base_lat >= MAP_EXTENT[2] and base_lat <= MAP_EXTENT[3]):
                    
                    color = cmap(norm(temp))
                    scatter_point = ax.scatter(base_lon, base_lat, color=color, s=180, edgecolor='darkblue', linewidth=1,
                               transform=ccrs.PlateCarree(), zorder=10, alpha=0.9)
                    if clip_patch: scatter_point.set_clip_path(clip_patch)
                continue # Skip the rest of the label plotting

            # --- Plot as normal if inside bounds ---
            if lon_offset != 0 or lat_offset != 0:
                line = ax.plot([base_lon, plot_lon], [base_lat, plot_lat],
                        color='black', linewidth=0.7, linestyle=':',
                        transform=ccrs.PlateCarree(), zorder=9)[0]
                if clip_patch: line.set_clip_path(clip_patch)
            
            # --- (A) City Name ---
            city_text = ax.text(plot_lon, plot_lat + 0.15, city, transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_CITY,
                    ha="center", va="bottom", color='black', fontweight='bold', zorder=16,
                    path_effects=text_outline)
            if clip_patch: city_text.set_clip_path(clip_patch)
            
            if temp is not None:
                color = cmap(norm(temp))
                scatter_point = ax.scatter(base_lon, base_lat, color=color, s=180, edgecolor='darkblue', linewidth=1,
                           transform=ccrs.PlateCarree(), zorder=10, alpha=0.9)
                if clip_patch: scatter_point.set_clip_path(clip_patch)
                
                # --- (B) Temperature (Line 1) ---
                temp_text = ax.text(plot_lon, plot_lat - 0.15, f"{temp:.0f}°C", color='navy',
                        bbox=BBOX_STYLE, transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_DATA,
                        fontweight='bold', ha="center", va="top", zorder=12,
                        path_effects=text_outline)
                if clip_patch: temp_text.set_clip_path(clip_patch)

            dew_point = all_dew_points.get(city)
            if dew_point is not None:
                # --- (C) Dew Point (Line 2, Left) ---
                dp_text = ax.text(plot_lon - 0.07, plot_lat - 0.45, f"DP: {dew_point:.0f}°C", color='#228b22',
                        bbox=BBOX_STYLE, transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_DATA,
                        ha="right", va="top", zorder=12,
                        path_effects=text_outline)
                if clip_patch: dp_text.set_clip_path(clip_patch)
                
            alt_inHg = all_altimeters.get(city)
            if alt_inHg is not None:
                # --- (C) Altimeter (Line 2, Right) ---
                alt_text = ax.text(plot_lon + 0.07, plot_lat - 0.45, f"{alt_inHg:.2f}\"Hg", color='#4169e1',
                        bbox=BBOX_STYLE, transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_DATA,
                        ha="left", va="top", zorder=12,
                        path_effects=text_outline)
                if clip_patch: alt_text.set_clip_path(clip_patch)

            # Combined Icon and Weather Condition Plotting
            condition = all_weather_conditions.get(city)
            if condition:
                cond_text = condition.split(' & ')[0] if ' & ' in condition else condition

                if condition in WEATHER_ICONS:
                    try:
                        img = WEATHER_ICONS[condition]
                        im = OffsetImage(img, zoom=0.23)
                        
                        text_area = TextArea(
                            f" {cond_text}", 
                            textprops=dict(
                                color='darkred',
                                fontsize=FONT_SIZE_WEATHER,
                                fontweight='bold',
                                va='center'
                            )
                        )
                        
                        packer = HPacker(
                            children=[im, text_area],
                            sep=2, 
                            align="center",
                            pad=0 
                        )
                        
                        # --- (D) Weather Box (Line 3) ---
                        ab = AnnotationBbox(
                            packer,
                            (plot_lon, plot_lat - 0.80), 
                            xybox=(0, 0), 
                            xycoords='data',
                            boxcoords="offset points",
                            frameon=True, 
                            box_alignment=(0.5, 1.0), 
                            zorder=20,
                            bboxprops=dict(
                                boxstyle="round,pad=0.3", 
                                facecolor='lightyellow', 
                                alpha=0.8,
                                edgecolor='gray', 
                                linewidth=0.5
                            )
                        )
                        
                        if clip_patch: ab.set_clip_path(clip_patch)
                        ax.add_artist(ab)
                    
                    except Exception as e:
                        print(f"Error loading icon or creating HPacker for {condition}: {e}")
                        # Fallback to just text
                        cond_text_obj = ax.text(plot_lon, plot_lat - 0.80, cond_text, color='darkred',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
                                transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_WEATHER,
                                ha="center", va="top", zorder=12, fontweight='bold',
                                path_effects=text_outline)
                        if clip_patch: cond_text_obj.set_clip_path(clip_patch)
                
                else: # No icon
                    cond_text_obj = ax.text(plot_lon, plot_lat - 0.80, cond_text, color='darkred',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
                            transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_WEATHER,
                            ha="center", va="top", zorder=12, fontweight='bold',
                            path_effects=text_outline)
                    if clip_patch: cond_text_obj.set_clip_path(clip_patch)
        
        else:
            # --- SILENT PLOT LOGIC ---
            # This will now catch stations used *only* for interpolation
            # (e.g., if you added YBAS but not to the CITIES dict)
            # We only plot them if they are inside the map extent
            if (temp is not None and
                base_lon >= MAP_EXTENT[0] and base_lon <= MAP_EXTENT[1] and
                base_lat >= MAP_EXTENT[2] and base_lat <= MAP_EXTENT[3]):
                
                color = cmap(norm(temp))
                scatter_point = ax.scatter(base_lon, base_lat, color=color, s=25, edgecolor='black', linewidth=0.5,
                                           transform=ccrs.PlateCarree(), zorder=7, alpha=0.8)
                if clip_patch: scatter_point.set_clip_path(clip_patch)
                
                text_obj = ax.text(base_lon, base_lat + 0.05, station_id, transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_ICAO,
                                   ha="center", va="bottom", color='black', fontweight='normal', zorder=8,
                                   path_effects=[pe.withStroke(linewidth=1, foreground='white')])
                if clip_patch: text_obj.set_clip_path(clip_patch)

    # Enhanced Alert Plotting
    alert_handles = {}
    for alert in alerts:
        for polygon_coords in alert['polygons']:
            lons, lats = zip(*polygon_coords)
            fill_poly = ax.fill(lons, lats, color=alert['color'], alpha=ALERT_ALPHA,
                    transform=ccrs.PlateCarree(), zorder=5, hatch='//')[0] 
            line_poly = ax.plot(lons, lats, color=alert['color'], linewidth=1.5,
                    transform=ccrs.PlateCarree(), zorder=5)[0] 
            
            if clip_patch:
                fill_poly.set_clip_path(clip_patch)
                line_poly.set_clip_path(clip_patch)

            if alert['event'] not in alert_handles:
                alert_handles[alert['event']] = mpatches.Patch(
                    color=alert['color'], alpha=0.4, label=alert['event'], hatch='//'
                )

    # Robust Colorbar
    print("Adding colorbar and legend...")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    
    cbar.set_label('Temperature (°C)', fontsize=FONT_SIZE_LEGEND, fontweight='bold', color='navy',
                   path_effects=text_outline)
    cbar.set_ticks(np.arange(-10, 51, 10)) 
    for label in cbar.ax.get_yticklabels():
        label.set_path_effects(text_outline)
        label.set_color('navy')

    # --- MODIFIED LEGEND (removed county line) ---
    legend_handles = [
        mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=12,
                      markeredgecolor='darkblue', markeredgewidth=1, label='City/Temp'),
        mlines.Line2D([], [], color='#228b22', marker='_', linestyle='None', markersize=12,
                      mew=3, label='Dew Point (°C)'),
        mlines.Line2D([], [], color='#4169e1', marker='_', linestyle='None', markersize=12,
                      mew=3, label='Altimeter (inHg)'),
        mlines.Line2D([], [], color='#4b0082', marker='|', linestyle='None', markersize=12,
                      mew=3, label='Wind Barb (Grid)'),
        mpatches.Patch(color='lightyellow', alpha=0.8, label='Weather Condition'),
        mlines.Line2D([], [], color='black', linestyle=':', linewidth=0.7, label='Offset Leader'),
        mpatches.Patch(color=MAJOR_ROAD_COLOR, label='Major Highway', alpha=0.7),
        mpatches.Patch(color=SECONDARY_ROAD_COLOR, label='Secondary Highway', alpha=0.7),
    ]
    legend_handles.extend(alert_handles.values())

    # --- MODIFIED: Changed legend location ---
    leg = ax.legend(handles=legend_handles, loc='upper left', fontsize=FONT_SIZE_LEGEND,
                    bbox_to_anchor=(0.01, 0.99), facecolor='white', framealpha=0.9,
                    edgecolor='lightgray', fancybox=True, ncol=2)
    leg.get_frame().set_linewidth(0.5)

    for text in leg.get_texts():
        text.set_path_effects(text_outline)

    # --- NEW: Console and File Output ---
    print("\n" + "="*50)
    wwa_text = "--- 🔔 Active Weather Alerts in Western Australia ---\n"
    if alerts:
        event_groups = {}
        for alert in alerts:
            evt = alert['event']
            if evt not in event_groups:
                event_groups[evt] = {"severity": alert['severity'], "areas": set()}
            event_groups[evt]["areas"].update(
                [area.strip() for area in alert['affected_areas'].split(',')]
            )
        
        for event, data in event_groups.items():
            # Use console-friendly formatting
            wwa_text += f"\n{event.upper()} ({data['severity']}):\n"
            wwa_text += f"  > {', '.join(sorted(list(data['areas'])))}\n"
    else:
        wwa_text = "--- ✅ No active weather alerts for Western Australia at this time. ---"

    # Print the alert text to the console
    print(wwa_text)
    print("="*50 + "\n")

    plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.9)
    
    output_filename = "wa_detailed_weather.png"
    print(f"Saving map to {output_filename}...")
    
    # Save the figure directly to a file
    plt.savefig(output_filename, format='png', facecolor=fig.get_facecolor(), dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    # await ctx.send(...) # <-- REMOVED



# --- NEW: Main execution block ---
if __name__ == "__main__":
    """
    This block runs when you execute the script directly
    (e.g., from your terminal: `python your_script_name.py`)
    """
    print("Starting Western Australia weather map generator...")
    try:
        generate_wa_weather_map()
        print("\nSuccessfully generated and saved 'wa_detailed_weather.png'.")
    except Exception as e:
        print(f"\nAn error occurred during map generation: {e}")
        traceback.print_exc() # Print the full error for debugging