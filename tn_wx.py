#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Tennessee (TN) Weather Map Generator.

This script fetches METAR data from aviationweather.gov (asynchronously for speed),
active NWS alerts from weather.gov (with special winter & freeze alert coloring),
a GOES satellite overlay, and a NEXRAD radar image to generate a detailed
weather map for the state of Tennessee.

The final map is saved to a specified output directory with a dynamic timestamp.
"""

import re
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.colors import BoundaryNorm
import matplotlib.cm
import io
from datetime import datetime, timezone, timedelta
from astral import LocationInfo
from astral.sun import sun
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, HPacker, TextArea
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import asyncio 
import aiohttp 
import matplotlib.patheffects as pe 
from scipy.interpolate import griddata 
from descartes import PolygonPatch 
from metpy.plots import USCOUNTIES
from mpl_toolkits.axes_grid1 import make_axes_locatable # <-- IMPORT FOR ROBUST COLORBAR
# --- IMPORTS for robust clipping ---
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from cartopy.mpl.path import shapely_to_path # <-- UPDATED IMPORT
import matplotlib.patches as patches

# --- IMPORTS FOR SATELLITE & RADAR (FROM GA_WX) ---
from io import BytesIO
import logging
import time
import xarray as xr
from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore
from datetime import datetime as dt 
import metpy
from netCDF4 import num2date
from metpy.plots import ctables 
from PIL import Image # <-- NEW IMPORT FOR RADAR IMAGE HANDLING

# --- NEW IMPORTS FOR CACHING ---
import pickle
import time

# --- SCRIPT CONFIGURATION ---
# Configure logging for console output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- CACHING CONFIGURATION ---
CACHE_DURATION_SECONDS = 900  # 15 minutes
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, ".wx_cache")

# Create cache directory if it doesn't exist
try:
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logging.info(f"Created cache directory: {CACHE_DIR}")
except Exception as e:
    logging.warning(f"Could not create cache directory: {e}. Caching will be disabled.")
    CACHE_DIR = None

# --- Caching Helper Functions ---
def is_cache_valid(filepath, max_age_seconds):
    """Checks if a cached file exists and is not stale."""
    if not os.path.exists(filepath):
        return False
    try:
        file_mod_time = os.path.getmtime(filepath)
        age_seconds = time.time() - file_mod_time
        return age_seconds < max_age_seconds
    except Exception as e:
        logging.warning(f"Error checking cache validity for {filepath}: {e}")
        return False

def save_json_cache(filename, data):
    """Saves data as a JSON file in the cache directory."""
    if not CACHE_DIR: return
    filepath = os.path.join(CACHE_DIR, filename)
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f)
        logging.info(f"Saved JSON cache: {filename}")
    except Exception as e:
        logging.warning(f"Failed to save JSON cache to {filepath}: {e}")

def load_json_cache(filename, max_age_seconds):
    """Loads data from a JSON cache file if it's valid."""
    if not CACHE_DIR: return None
    filepath = os.path.join(CACHE_DIR, filename)
    if is_cache_valid(filepath, max_age_seconds):
        try:
            with open(filepath, 'r') as f:
                logging.info(f"Using cached JSON: {filename}")
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to read JSON cache from {filepath}: {e}")
    return None

def save_pickle_cache(filename, data):
    """Saves data as a pickle file in the cache directory."""
    if not CACHE_DIR: return
    filepath = os.path.join(CACHE_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Saved pickle cache: {filename}")
    except Exception as e:
        logging.warning(f"Failed to save pickle cache to {filepath}: {e}")

def load_pickle_cache(filename, max_age_seconds):
    """Loads data from a pickle cache file if it's valid."""
    if not CACHE_DIR: return None
    filepath = os.path.join(CACHE_DIR, filename)
    if is_cache_valid(filepath, max_age_seconds):
        try:
            with open(filepath, 'rb') as f:
                logging.info(f"Using cached pickle: {filename}")
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to read pickle cache from {filepath}: {e}")
    return None

# --- Weather Icon Generation ---
def create_icon_base(fig_size=(1,1), dpi=100):
    """Creates a base 1x1 matplotlib figure for icons."""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return fig, ax

def create_sunny_icon():
    """Generates a sunny icon."""
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
    """Generates a clear night icon."""
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
    """Generates a partially cloudy day icon."""
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
    """Generates a partially cloudy night icon."""
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
    """Generates a cloudy icon."""
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
    """Generates a rainy icon."""
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
    """Generates a thunderstorm icon."""
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
    """Generates a fog/mist icon."""
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
    """Generates a snowing icon."""
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
    """Generates a wintry mix (ice/snow/rain) icon."""
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
    """Generates a generic rain icon."""
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

logging.info("Generating weather icons...")
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
logging.info("Weather icons generated.")

# --- Configuration (TENNESSEE) ---
MAP_EXTENT = [-90.5, -81.5, 34.8, 36.8] # [lon_min, lon_max, lat_min, lat_max]
CITIES = {
    "Nashville": (36.1627, -86.7816, "KBNA"),
    "Memphis": (35.1495, -90.0490, "KMEM"),
    "Knoxville": (35.9606, -83.9207, "KTYS"),
    "Chattanooga": (35.0456, -85.2672, "KCHA"),
    "Clarksville": (36.5284, -87.3567, "KCKV"),
    "Jackson": (35.6145, -88.8162, "KMKL"),
    "Tri-Cities": (36.4752, -82.4074, "KTRI"),
    "Crossville": (35.9525, -85.0833, "KCSV"),
}
CITY_OFFSETS = {
    "Nashville": (0.1, 0.1),
    "Memphis": (0.1, 0.1),
    "Knoxville": (-0.1, -0.1),
    "Tri-Cities": (0.1, 0.1),
    "Chattanooga": (0.1, -0.1),
}
SILENT_STATIONS = {
    # Border stations for "pinning" the edges
    "Bowling Green (KY)": (36.9650, -86.4183, "KBWG"),
    "Asheville (NC)": (35.4361, -82.5417, "KAVL"),
    "Huntsville (AL)": (34.6433, -86.7750, "KHSV"),
    "Tupelo (MS)": (34.2694, -88.7694, "KTUP"),
    "Jonesboro (AR)": (35.8322, -90.6458, "KJBR"),
    "Paducah (KY)": (37.0603, -88.7733, "KPAH"),
}
# ---
FIG_BG_COLOR = '#b0c4de'
LAND_COLOR = '#f5f5dc'
OCEAN_COLOR = '#b0e0e6'
STATE_BORDER_COLOR = '#8b4513'
COUNTY_BORDER_COLOR = '#27522b' # Visible Gray
MAJOR_ROAD_COLOR = '#000000' # <-- Changed to Black
SECONDARY_ROAD_COLOR = '#333333' # <-- Changed to Darker Gray
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

# --- NEW: Winter Alert Colors ---
WINTER_ALERT_COLORS = {
    "Warning": "#8A2BE2",  # BlueViolet
    "Watch": "#4682B4",    # SteelBlue
    "Advisory": "#AFEEEE", # PaleTurquoise
}
# --- END NEW ---

# --- NEW: Temperature-Specific Alert Colors ---
TEMP_ALERT_COLORS = {
    "Freeze Warning": "#00008B",  # DarkBlue
    "Freeze Watch": "#1E90FF",    # DodgerBlue
    "Frost Advisory": "#B0E0E6",  # PowderBlue
}
# --- END NEW ---

plt.rcParams['font.family'] = FONT_FAMILY

# --- SATELLITE OVERLAY CONFIGURATION (OLD S3 LOGIC REMOVED) ---
# --- CACHE LOGIC (REMOVED, Siphon handles caching) ---

# --- NEW SATELLITE FUNCTION (replaces fetch_latest_band13_url, etc.) ---
def fetch_satellite_data():
    """
    Fetches the latest GOES-16 Band 13 (IR) data from UCAR THREDDS.
    Tries today (UTC) first, then yesterday if today fails.
    Returns brightness temp (C), x/y coords, and projection.
    """
    logging.info("Attempting to fetch satellite data from UCAR THREDDS...")
    
    # --- NEW: Create a list of dates to try (today, then yesterday) ---
    dates_to_try = [dt.now(timezone.utc), dt.now(timezone.utc) - timedelta(days=1)]
    
    for date_to_try in dates_to_try:
        try:
            # Connect to the UCAR THREDDS catalog for GOES-East CONUS Channel 13
            catalog_url = (
                'https://thredds.ucar.edu/thredds/catalog/satellite/goes/east/products/'
                'CloudAndMoistureImagery/CONUS/Channel13/{:%Y%m%d}/'
                'catalog.xml'.format(date_to_try)
            )
            logging.info(f"Checking for data in: {catalog_url}")
            cat = TDSCatalog(catalog_url)
            
            # Get the latest dataset
            if not cat.datasets:
                logging.warning(f"No datasets found for date: {date_to_try.strftime('%Y-%m-%d')}")
                continue # Try the next date in the loop
                
            dataset = cat.datasets[-1] # Get the most recent dataset
            logging.info(f"Found latest dataset: {dataset.name}")
            
            # Access the data via OPENDAP
            ds = dataset.remote_access(service='OPENDAP')
            ds = NetCDF4DataStore(ds)
            ds = xr.open_dataset(ds)
            
            # Parse the data using MetPy
            dat = ds.metpy.parse_cf('Sectorized_CMI')
            sat_proj = dat.metpy.cartopy_crs
            x_coords = dat['x']
            y_coords = dat['y']
            
            # Get data (in Kelvin) and convert to Celsius
            bt_kelvin = dat.values
            bt_celsius = bt_kelvin - 273.15
            
            logging.info("Successfully fetched and processed satellite data.")
            # --- If successful, return the data and exit the function ---
            return bt_celsius, x_coords, y_coords, sat_proj

        except Exception as e:
            logging.error(f"--- SATELLITE DATA FAILED for {date_to_try.strftime('%Y-%m-%d')} ---")
            print(f"--- SATELLITE DATA FAILED for {date_to_try.strftime('%Y-%m-%d')} ---")
            logging.error(f"Error: {e}")
            print(f"Error: {e}")
            continue # Try the next date

    # --- If the loop finishes without returning, all dates have failed ---
    logging.error("--- SATELLITE DATA FAILED ---")
    print("--- SATELLITE DATA FAILED ---")
    logging.error("Failed to find valid satellite data after checking all sources.")
    print("Failed to find valid satellite data after checking all sources.")
    print("The map will be generated without the satellite overlay.")
    return None, None, None, None


def add_satellite_overlay(fig, ax, clip_patch, text_outline, bt, x_coords, y_coords, sat_proj):
    """
    Adds the satellite IR data (as filled contours) and contour lines as an overlay.
    """
    
    # --- NEW: Check if data is valid ---
    if bt is None or x_coords is None or y_coords is None or sat_proj is None:
        logging.error("Cannot plot satellite overlay: Data is missing (download likely failed).")
        print("--- SATELLITE PLOTTING SKIPPED (Data was missing) ---")
        return
    # --- END NEW BLOCK ---

    try:
        logging.info("Plotting satellite overlay...")
        levels = np.arange(-80, 41, 5)
        # --- *** FIX 1: Using 'gray' colormap per user request *** ---
        cmap_bt = matplotlib.cm.get_cmap('gray')
        
        # --- This is the B&W overlay (zorder 1.0) ---
        contour_bt = ax.contourf(x_coords, y_coords, bt, levels=levels, cmap=cmap_bt, 
                                 alpha=0.7, # Reduced alpha to 0.7
                                 transform=sat_proj, zorder=1.0, extend='both')
        
        # --- These are the contour lines (zorder 1.1) ---
        contour_lines = ax.contour(x_coords, y_coords, bt, levels=levels, 
                                   colors='dimgrey', linewidths=0.7,
                                   alpha=0.6, transform=sat_proj, zorder=1.1)
        
        if clip_patch:
            # We don't clip this layer, we clip the temp layer on top
            pass
                
        # --- *** FIX 2: Draw colorbar using make_axes_locatable *** ---
        divider = make_axes_locatable(ax)
        cax_bt = divider.append_axes("bottom", size="3%", pad=0.5, axes_class=plt.Axes)
        cbar_bt = fig.colorbar(contour_bt, cax=cax_bt, orientation='horizontal')
        cbar_bt.set_label('IR Brightness Temp (°C)', fontsize=FONT_SIZE_LEGEND, fontweight='bold', color='navy',
                          path_effects=text_outline)
        for label in cbar_bt.ax.get_xticklabels():
            label.set_path_effects(text_outline)
            label.set_color('navy')
        logging.info("Satellite overlay plotted successfully.")
            
    except Exception as e:
        logging.error(f"--- SATELLITE PLOTTING FAILED ---")
        print(f"--- SATELLITE PLOTTING FAILED ---")
        logging.error(f"An error occurred while drawing the satellite contours: {e}")
        print(f"An error occurred while drawing the satellite contours: {e}")
        
# --- END SATELLITE FUNCTIONS ---

# --- *** NEW RADAR FUNCTIONS *** ---
def add_radar_image_overlay(ax, cache_duration):
    """
    Fetches the Iowa State radar PNG and overlays it with transparency.
    Uses caching to avoid re-downloading.
    """
    logging.info("Attempting to fetch radar image overlay...")
    
    cache_file = "radar_cache.png"
    filepath = os.path.join(CACHE_DIR, cache_file) if CACHE_DIR else None
    img_data = None

    # Try loading from cache
    if filepath and is_cache_valid(filepath, cache_duration):
        try:
            with open(filepath, 'rb') as f:
                img_data = BytesIO(f.read())
            logging.info("Using cached radar image.")
        except Exception as e:
            logging.warning(f"Failed to read cached radar image: {e}")
            img_data = None

    # If cache failed or is stale, fetch new data
    if img_data is None:
        try:
            logging.info("Fetching new radar image...")
            url = 'https://mesonet.agron.iastate.edu/data/gis/images/4326/USCOMP/n0r_0.png'
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logging.error(f"Failed to download radar image. Status: {response.status_code}")
                return
            
            img_data = BytesIO(response.content)
            
            # Save to cache
            if filepath:
                try:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    logging.info("Saved new radar image to cache.")
                except Exception as e:
                    logging.warning(f"Failed to save radar image to cache: {e}")
                    
        except Exception as e:
            logging.error(f"--- RADAR DOWNLOAD FAILED ---: {e}")
            return # Don't proceed if download failed

    # Process and plot the image data
    try:
        img_data.seek(0) # Ensure buffer is at the start
        img = Image.open(img_data).convert("RGBA")
        img_array = np.array(img)

        alpha = np.full(img_array.shape[:2], 255, dtype=np.uint8)
        near_black_pixels = (img_array[:, :, 0] < 5) & \
                            (img_array[:, :, 1] < 5) & \
                            (img_array[:, :, 2] < 5)
        alpha[near_black_pixels] = 0
        alpha[~near_black_pixels] = int(255 * 0.4) # 0.4 opacity
        img_array[:, :, 3] = alpha
        
        extent = [-126, -66, 24, 50]
        
        ax.imshow(img_array, origin='upper', extent=extent,
                  transform=ccrs.PlateCarree(),
                  zorder=1.8, # On top of temp, under counties
                 )
        logging.info("Radar image overlay plotted successfully.")
        
    except Exception as e:
        logging.error(f"--- RADAR PLOTTING FAILED ---")
        print(f"--- RADAR PLOTTING FAILED ---")
        logging.error(f"An error occurred while drawing the radar image: {e}")
        print(f"An error occurred while drawing the radar image: {e}")
# --- *** END NEW RADAR FUNCTIONS *** ---


# --- *** NEW ASYNC METAR FUNCTIONS FOR SPEED *** ---

async def fetch_one_metar(session, icao):
    """Asynchronously fetches METAR data for a single station."""
    metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format=json&hours=0'
    try:
        async with session.get(metar_url, timeout=10) as response:
            if response.status != 200:
                logging.warning(f"Failed to fetch METAR for {icao}: Status {response.status}")
                return icao, None
            src = await response.text()
            if not src.strip():
                logging.warning(f"Empty METAR response for {icao}")
                return icao, None
            
            json_data = json.loads(src)
            return icao, json_data
    except Exception as e:
        logging.error(f"Error fetching METAR for {icao}: {e}")
        return icao, None

def parse_metar_json(json_data, icao, lat, lon):
    """Parses the JSON data from a METAR report."""
    try:
        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")

        raw_metar = json_data[0]['rawOb']
        main_body = raw_metar.split('RMK')[0]

        temp_f, dew_point_f = None, None
        temp_dew_pattern = re.search(r'\b(M?\d{2})/(M?\d{2})\b', main_body)
        if temp_dew_pattern:
            temp_str, dew_str = temp_dew_pattern.groups()
            temp_c = int(temp_str.replace('M', '-'))
            dew_point_c = int(dew_str.replace('M', '-'))
            temp_f = round((temp_c * 9 / 5) + 32)
            dew_point_f = round((dew_point_c * 9 / 5) + 32)

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
            alt_pattern_q = re.search(r'\bQ(\d{4})\b', main_body)
            if alt_pattern_q:
                alt_raw = alt_pattern_q.group(1)
                alt_hpa = float(alt_raw)
                altimeter_inHg = round(alt_hpa * 0.02953, 2)
        
        # --- Weather Condition Logic ---
        cloud_info = extract_cloud_info(raw_metar)
        weather_codes = extract_weather_phenomena(raw_metar)
        day_flag = is_daytime(lat, lon)
        combined_cloud_layers = cloud_info['low'] + cloud_info['mid'] + cloud_info['high']
        condition = map_metar_weather_to_condition(weather_codes, combined_cloud_layers, day_flag)

        return {
            "temp_f": temp_f, "dew_point_f": dew_point_f,
            "wind_dir": wind_direction, "wind_spd": wind_speed,
            "alt_inHg": altimeter_inHg, "condition": condition
        }

    except Exception as e:
        logging.warning(f"Error parsing METAR data for {icao}: {e}")
        return {
            "temp_f": None, "dew_point_f": None,
            "wind_dir": None, "wind_spd": None,
            "alt_inHg": None, "condition": None
        }

async def fetch_all_metars_async(station_dict):
    """Creates and runs all METAR fetch tasks concurrently."""
    tasks = []
    async with aiohttp.ClientSession() as session:
        for city, (lat, lon, icao) in station_dict.items():
            tasks.append(asyncio.create_task(fetch_one_metar(session, icao)))
        
        results = await asyncio.gather(*tasks)
    
    # Process results
    parsed_data = {}
    for icao, json_data in results:
        # Find the city name from the ICAO
        city_name = None
        lat, lon = None, None
        for city, (c_lat, c_lon, c_icao) in station_dict.items():
            if c_icao == icao:
                city_name = city
                lat, lon = c_lat, c_lon
                break
        
        if city_name:
            parsed_data[city_name] = parse_metar_json(json_data, icao, lat, lon)
            
    return parsed_data

# --- End of Async METAR Functions ---


# --- Helper Functions (non-METAR) ---
def extract_cloud_info(metar):
    """Extracts cloud layers from a METAR string."""
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
    """Extracts weather phenomena codes (like 'TS' or 'RA') from a METAR."""
    main_body = metar.split('RMK')[0]
    weather_pattern = re.compile(
        r'(-|\+|VC)?(TS|SH|FZ|DR|BL|MI|BC|PR)?(DZ|RA|SN|SG|IC|PL|GR|GS|UP)?(BR|FG|FU|VA|DU|SA|HZ|PY)?(PO|SQ|FC|SS|DS)?'
    )
    parts = main_body.split()
    weather_conditions = []
    
    for part in parts:
        # Skip parts that are definitely not weather phenomena
        if re.match(r'^(FEW|SCT|BKN|OVC|VV)\d{3}$', part) or \
           re.match(r'^\d{4}$', part) or \
           re.match(r'^M?\d{2}/M?\d{2}$', part) or \
           re.match(r'^[AQ]\d{4}$', part) or \
           re.match(r'^\d+SM$', part):
            continue
        
        match = re.match(weather_pattern, part)
        # Check if it's a valid weather code
        if match and any(match.groups()):
            weather_conditions.append(part)
            
    return weather_conditions

def map_metar_weather_to_condition(weather_codes, cloud_layers, is_day):
    """Maps raw METAR codes to a simplified weather condition string."""
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
        # Default to clear if no clouds or weather
        condition = 'Sunny' if is_day else 'Moonlight'
        
    return condition

def is_daytime(lat, lon):
    """Determines if it's daytime at a given lat/lon."""
    loc = LocationInfo(latitude=lat, longitude=lon)
    try:
        s = sun(loc.observer, date=datetime.now(timezone.utc))
        now = datetime.now(timezone.utc)
        return s['sunrise'] <= now <= s['sunset']
    except Exception:
        # Default to day if sun calculation fails
        return True 

def get_nws_alerts(state='TN'): # <-- UPDATED
    """
    Fetches NWS alerts for a given state (e.g., 'TN').
    *** FIX: Uses the /alerts/active endpoint to avoid retrieving archived/future alerts. ***
    """
    
    # --- PART 1: The dedicated ACTIVE alerts URL ---
    url = f"https://api.weather.gov/alerts/active?area={state}"
    
    headers = {
        # NWS API requires a User-Agent
        "User-Agent": ("My-Standalone-Weather-Script (contact@example.com) "
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ")
    }
    
    all_features = []

    try:
        # --- PART 2: Pagination loop for active alerts ---
        while url:
            logging.info(f"Fetching NWS alert data from: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an error for bad status codes
            data = response.json()

            if "features" in data:
                all_features.extend(data["features"])
            
            # Check for the 'next' page URL
            url = data.get("pagination", {}).get("next")
        # --- END OF PAGINATION LOOP ---
            
        if not all_features:
            logging.info("No 'features' found in NWS active alert data.")
            return []
            
        alerts = []
        # --- PART 3: Process and filter features ---
        for feature in all_features:
            props = feature.get("properties", {})
            geometry = feature.get("geometry", {})
            
            if not props or not geometry:
                continue

            event = props.get("event", "Unknown Event")
            logging.info(f"--- NWS DEBUG: Found event: '{event}' ---")
            severity = props.get("severity", "Unknown")
            
            # --- The filter logic: Only keep Severe/Winter/Freeze alerts ---
            color = None
            is_winter_alert = "Freeze" in event or "Frost" in event or \
                              any(term in event for term in ["Winter", "Snow", "Ice", "Blizzard", "Wintry", "Wind Chill"])
                              
            is_severe_alert = "Tornado Warning" in event or "Severe Thunderstorm Warning" in event
            
            if is_winter_alert:
                if "Freeze Warning" in event:
                    color = TEMP_ALERT_COLORS["Freeze Warning"]
                elif "Freeze Watch" in event:
                    color = TEMP_ALERT_COLORS["Freeze Watch"]
                elif "Frost Advisory" in event:
                    color = TEMP_ALERT_COLORS["Frost Advisory"]
                elif "Warning" in event:
                    color = WINTER_ALERT_COLORS["Warning"]
                elif "Watch" in event:
                    color = WINTER_ALERT_COLORS["Watch"]
                elif "Advisory" in event:
                    color = WINTER_ALERT_COLORS["Advisory"]
                else:
                    color = ALERT_COLORS.get(severity, ALERT_COLORS["Unknown"])
            
            elif is_severe_alert:
                color = ALERT_COLORS.get(severity, ALERT_COLORS["Unknown"])
            
            else:
                # Explicitly skip all other types (e.g., Special Weather Statement, Flood, Wind)
                continue
            
            if color is None:
                continue 

            # --- Process the geometry ---
            coords_list = []
            geom_type = geometry.get("type")
            geom_coords = geometry.get("coordinates")

            if geom_type == "Polygon":
                coords_list = geom_coords 
            elif geom_type == "MultiPolygon":
                coords_list = [poly[0] for poly in geom_coords] 

            if not coords_list:
                continue

            alerts.append({
                "headline": props.get("headline", "No headline"),
                "event": event, 
                "severity": severity,
                "polygons": coords_list,
                "color": color,
                "affected_areas": props.get("areaDesc", "Unknown Area").replace(';', ',')
            })
        
        if not alerts:
            logging.info("Active alerts were fetched, but none matched the filter (Winter/Severe).")
            
        return alerts
        
    except requests.RequestException as e:
        logging.error(f"Error fetching NWS alerts: {e}")
        return []

# --- Main Map Generation Task ---

def generate_tennessee_weather_map(): # <-- [CHANGED]
    """Generate and save the Tennessee weather map."""
    
    logging.info("Generating Tennessee weather map...")
    
    # --- Combine CITIES and SILENT_STATIONS ---
    ALL_STATIONS_TN = CITIES.copy()
    ALL_STATIONS_TN.update(SILENT_STATIONS)
    
    all_temperatures = {}
    all_dew_points = {}
    all_wind_directions = {}
    all_wind_speeds = {}
    all_altimeters = {}
    all_weather_conditions = {}

    # --- *** ASYNC METAR FETCHING WITH CACHE AND RE-FETCH LOGIC *** ---
    metar_cache_file = "metar_tn_cache.json" # <-- UPDATED
    metar_results = load_json_cache(metar_cache_file, CACHE_DURATION_SECONDS)
    
    # 1. Check if cached data is usable (i.e., enough valid temperatures)
    valid_temp_count = 0
    if metar_results is not None:
        valid_temp_count = sum(1 for data in metar_results.values() if data and data.get("temp_f") is not None)

    if metar_results is None or valid_temp_count < 4:
        if valid_temp_count < 4 and metar_results is not None:
            logging.warning(f"Cached METAR data has only {valid_temp_count} valid temperatures. Forcing fresh fetch.")
        else:
            logging.info("Fetching new METAR data for all stations (asynchronously)...")
            
        metar_results = asyncio.run(fetch_all_metars_async(ALL_STATIONS_TN)) # <-- UPDATED
        
        # Re-check valid count after fresh fetch
        valid_temp_count = sum(1 for data in metar_results.values() if data and data.get("temp_f") is not None)

        if valid_temp_count >= 4:
            save_json_cache(metar_cache_file, metar_results)
            logging.info(f"Fresh METAR data saved to cache ({valid_temp_count} valid points).")
        else:
            logging.warning(f"Fresh METAR fetch yielded only {valid_temp_count} valid points. Using minimal data.")
    else:
        logging.info("Using cached METAR data.")
    
    logging.info("Parsing METAR data...")
    # Now, populate the dictionaries from the (already-fetched) results
    for city, data in metar_results.items():
        all_temperatures[city] = data["temp_f"]
        all_dew_points[city] = data["dew_point_f"]
        all_wind_directions[city] = data["wind_dir"]
        all_wind_speeds[city] = data["wind_spd"]
        all_altimeters[city] = data["alt_inHg"]
        all_weather_conditions[city] = data["condition"]
    # --- *** END METAR RE-FETCH BLOCK *** ---

    # --- NWS ALERTS WITH CACHE AND FALLBACK ---
    alert_cache_file = "alerts_tn_cache.json" # <-- UPDATED
    alerts = None
    
    # 1. Try to load from cache
    cached_alerts = load_json_cache(alert_cache_file, CACHE_DURATION_SECONDS)
    
    if cached_alerts is not None:
        try:
            # Quick check to ensure at least one alert has a 'color' key, preventing the crash
            if any('color' in alert for alert in cached_alerts):
                 alerts = cached_alerts
                 logging.info("Using cached NWS alerts.")
            else:
                 raise KeyError("Cached alerts missing 'color' key. Forcing refresh.")
        except Exception as e:
            logging.warning(f"Error validating cached alerts ({e}). Forcing fresh fetch.")
            alerts = None # Force refresh
            
    # 2. Fetch fresh data if cache failed or was stale
    if alerts is None:
        logging.info("Fetching NWS alerts for Tennessee (Fresh Download)...") # <-- UPDATED
        alerts = get_nws_alerts('TN') # Now uses /alerts/active internally # <-- UPDATED
        # Only save if the fetch was successful and returned non-empty list
        if alerts:
            save_json_cache(alert_cache_file, alerts)
        else:
            logging.warning("Fresh alert fetch returned no data.")
    # --- END MODIFIED ALERT BLOCK ---

    # --- Prepare data for interpolation using ALL METAR stations ---
    points = [] 
    temp_values = []
    u_values = []
    v_values = []

    for city, temp in all_temperatures.items():
        lat, lon, _ = ALL_STATIONS_TN[city] # <-- UPDATED
        
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

    if len(temp_values) >= 4:
        logging.info("Interpolating gridded data for contours...")
        try:
            grid_lon = np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 100)
            grid_lat = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 100)
            grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
            
            # Two-Pass Interpolation
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
            logging.error(f"Error during interpolation: {e}")
            grid_z, grid_u, grid_v = None, None, None
    else:
        logging.warning("Not enough data to create contour map. Skipping.")

    # --- Plotting (Enhanced) ---
    logging.info("Beginning map generation...")
    # --- *** LAYOUT FIX 1: Changed figsize to 16:9 aspect ratio *** ---
    fig = plt.figure(figsize=(16, 9), facecolor=FIG_BG_COLOR) # TN is wide, changed from (16, 12)
    
    text_outline = [pe.withStroke(linewidth=3, foreground='white')]
    
    current_time = datetime.now().strftime('%Y-%m-%d %I:%M %p') 
    # --- *** LAYOUT FIX 2: Moved title 'y' position up to pack tighter *** ---
    fig.suptitle(f"Tennessee - Detailed Weather Map\nGenerated: {current_time}", # <-- UPDATED
                 fontsize=FONT_SIZE_TITLE, fontweight="bold", y=0.95, color='navy', # <-- CHANGED y=0.91 to y=0.95
                 path_effects=text_outline) 

    # --- *** FIX 2: Reverting to make_axes_locatable for robust layout *** ---
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    
    # --- NEW: Robust STATE CLIP PATCH ---
    clip_patch = None
    try:
        logging.info("Creating state clipping path for Tennessee...") # <-- UPDATED
        target_states = ['Tennessee'] # <-- UPDATED
        state_geometries = []
        # --- *** FIX 1: Changed resolution to 50m for speed *** ---
        states_shp = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_1_states_provinces')
        reader = shpreader.Reader(states_shp)
        for record in reader.records():
            state_name = record.attributes.get('name')
            if state_name in target_states:
                state_geometries.append(record.geometry)
        if not state_geometries:
            raise ValueError("Could could not find Tennessee in shpreader.") # <-- UPDATED
        combined_geom = unary_union(state_geometries)
        if not combined_geom:
            raise ValueError("Unary union of Tennessee geometry failed.") # <-- UPDATED
        paths = shapely_to_path(combined_geom) # <-- UPDATED
        if not isinstance(paths, (list, tuple)): # <-- ADDED
            paths = [paths] # <-- ADDED
        combined_path = Path.make_compound_path(*paths)
        clip_patch = PathPatch(combined_path, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='none')
        if clip_patch:
            ax.add_patch(clip_patch)
        else:
            raise ValueError("PathPatch creation failed for Tennessee.") # <-- UPDATED
    except Exception as e:
        logging.error(f"Failed to create clip patch for Tennessee. Error: {e}") # <-- UPDATED
        clip_patch = None
    if not clip_patch:
        logging.warning("Clipping patch could not be created. Contours and barbs will not be clipped.")


    # Enhanced Map features (zorder 0)
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor=OCEAN_COLOR)
    ax.add_feature(cfeature.LAND, zorder=0, facecolor=LAND_COLOR, edgecolor='none')

    # --- SATELLITE OVERLAY (with caching) ---
    try:
        # Re-use the same satellite cache file as other scripts, very efficient
        sat_cache_file = "satellite_east_cache.pkl" 
        sat_data = load_pickle_cache(sat_cache_file, CACHE_DURATION_SECONDS)
        
        if sat_data:
            logging.info("Using cached satellite data.")
            # Note: We reconstruct the correct projection in the plotting function
            bt, x_coords_raw, y_coords_raw = sat_data
            x_coords = x_coords_raw # Re-assign for clarity
            y_coords = y_coords_raw
            # Must re-create proj because it's non-serializable in all cases
            sat_proj = ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0, sweep_axis='x')
        else:
            logging.info("Fetching satellite overlay...")
            bt, x_coords, y_coords, sat_proj = fetch_satellite_data()
            if bt is not None: # Only cache if fetch was successful
                # Note: We save only the array data, and reconstruct the projection
                save_pickle_cache(sat_cache_file, (bt, x_coords, y_coords))
        
        # Pass the main 'ax' to the function
        add_satellite_overlay(fig, ax, clip_patch, text_outline, bt, x_coords, y_coords, sat_proj)
    except Exception as e:
        logging.error(f"--- SATELLITE OVERLAY FAILED IN MAIN SCRIPT ---", exc_info=True)
        print(f"--- SATELLITE OVERLAY FAILED IN MAIN SCRIPT: {e} ---")

    # --- RADAR OVERLAY (with caching) ---
    try:
        # Pass cache duration to the radar function
        add_radar_image_overlay(ax, CACHE_DURATION_SECONDS)
    except Exception as e:
        logging.error(f"--- RADAR OVERLAY FAILED IN MAIN SCRIPT ---", exc_info=True)
        print(f"--- RADAR OVERLAY FAILED IN MAIN SCRIPT: {e} ---")


    # --- METAR CONTOUR PLOT (zorder 1.5) ---
    if grid_z is not None and grid_lon is not None and grid_lat is not None:
        cmap = matplotlib.cm.get_cmap('jet') 
        boundaries = np.arange(-30, 122, 2) # 2F INCREMENTS
        norm = BoundaryNorm(boundaries=boundaries, ncolors=cmap.N, extend='both')

        contour_plot = ax.contourf(grid_lon, grid_lat, grid_z,
                              levels=boundaries, cmap=cmap, norm=norm,
                              transform=ccrs.PlateCarree(),
                              zorder=1.5, alpha=0.6, extend='both') # <-- ZORDER 1.5
        
        if clip_patch:
            if hasattr(contour_plot, 'collections'):
                for collection in contour_plot.collections:
                    collection.set_clip_path(clip_patch)
            else:
                contour_plot.set_clip_path(clip_patch)
    
    
    # WIND BARB GRID (with clipping) (zorder 6)
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
    
    # --- COUNTY BORDERS USING METPY (with clipping) (zorder 2) ---
    try:
        counties_feature = USCOUNTIES.with_scale('5m')
        county_patches = ax.add_feature(counties_feature, 
                       edgecolor=COUNTY_BORDER_COLOR,
                       facecolor='none',
                       linewidth=1.2,
                       linestyle='-.',
                       zorder=2, # <-- zorder 2 (between sat/radar and roads)
                       alpha=0.8)
        if clip_patch:
            county_patches.set_clip_path(clip_patch)
    except Exception as e:
        logging.warning(f"Error loading counties with MetPy: {e}")

    # Enhanced Road plotting (with clipping) (zorder 3)
    try:
        # --- Using 10m roads via shpreader ---
        roads_shpfilename = shpreader.natural_earth(resolution='10m', category='cultural', name='roads')
        roads_reader = shpreader.Reader(roads_shpfilename)
        
        relevant_roads = []
       
        # Get bounds from the script's main MAP_EXTENT
        map_lon_min = MAP_EXTENT[0]
        map_lon_max = MAP_EXTENT[1]
        map_lat_min = MAP_EXTENT[2]
        map_lat_max = MAP_EXTENT[3]
       
        for record in roads_reader.records():
            road_type = record.attributes.get('type', 'Unknown')
            if road_type in ['Major Highway', 'Secondary Highway']:
                geom = record.geometry
               
                # Use Shapely's (minx, miny, maxx, maxy) bounds
                geom_lon_min, geom_lat_min, geom_lon_max, geom_lat_max = geom.bounds
               
                # Check for BBOX overlap
                if (geom_lon_min <= map_lon_max and geom_lon_max >= map_lon_min and
                    geom_lat_min <= map_lat_max and geom_lat_max >= map_lat_min):
                    relevant_roads.append((geom, road_type))
        
        for geometry, road_type in relevant_roads:
            color = MAJOR_ROAD_COLOR if road_type == 'Major Highway' else SECONDARY_ROAD_COLOR
            linewidth = 1.5 if road_type == 'Major Highway' else 1.0
           
            geom_patch = ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='none',
                              edgecolor=color, linestyle='-',
                              zorder=3, # Kept zorder=3 to be under state/county lines
                              alpha=1.0,
                              linewidth=linewidth)
            if clip_patch:
                geom_patch.set_clip_path(clip_patch)
               
    except Exception as e:
        logging.warning(f"Error loading roads: {e}")
    
    # Add subtle gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.5, linestyle='-.')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'path_effects': text_outline}
    gl.ylabel_style = {'size': 10, 'path_effects': text_outline}

    # Colormap (with 2F increments)
    cmap = matplotlib.cm.get_cmap('jet') 
    boundaries = np.arange(-30, 122, 2) # 2F INCREMENTS
    norm = BoundaryNorm(boundaries=boundaries, ncolors=cmap.N, extend='both')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # --- *** FIX 3: Use make_axes_locatable for the main temp cbar *** ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    
    cbar.set_label('Temperature (°F)', fontsize=FONT_SIZE_LEGEND, fontweight='bold', color='navy',
                   path_effects=text_outline)
    cbar.set_ticks(np.arange(-30, 121, 10)) 
    for label in cbar.ax.get_yticklabels():
        label.set_path_effects(text_outline)
        label.set_color('navy')

    # --- NEW: Enhanced Station Data Plotting Loop ---
    logging.info("Plotting station data...")
    for city, temp in all_temperatures.items():
        base_lat, base_lon, icao = ALL_STATIONS_TN[city] # <-- UPDATED
        
        # Check if this is a "full plot" city or a "silent" one
        if city in CITIES:
            # --- FULL PLOT LOGIC ---
            lon_offset, lat_offset = CITY_OFFSETS.get(city, (0, 0))
            plot_lon = base_lon + lon_offset
            plot_lat = base_lat + lat_offset

            if lon_offset != 0 or lat_offset != 0:
                line = ax.plot([base_lon, plot_lon], [base_lat, plot_lat],
                        color='black', linewidth=0.7, linestyle=':',
                        transform=ccrs.PlateCarree(), zorder=9)[0]
                if clip_patch: line.set_clip_path(clip_patch)
            
            city_text = ax.text(plot_lon, plot_lat + 0.08, city, transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_CITY,
                    ha="center", va="bottom", color='black', fontweight='bold', zorder=16,
                    path_effects=text_outline)
            if clip_patch: city_text.set_clip_path(clip_patch)
            
            if temp is not None:
                color = cmap(norm(temp))
                scatter_point = ax.scatter(base_lon, base_lat, color=color, s=180, edgecolor='darkblue', linewidth=1,
                           transform=ccrs.PlateCarree(), zorder=10, alpha=0.9)
                if clip_patch: scatter_point.set_clip_path(clip_patch)
                
                temp_text = ax.text(plot_lon, plot_lat - 0.04, f"{temp:.0f}°F", color='navy',
                        bbox=BBOX_STYLE, transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_DATA,
                        fontweight='bold', ha="center", va="top", zorder=12,
                        path_effects=text_outline)
                if clip_patch: temp_text.set_clip_path(clip_patch)

            dew_point = all_dew_points.get(city)
            if dew_point is not None:
                dp_text = ax.text(plot_lon - 0.07, plot_lat - 0.16, f"DP: {dew_point:.0f}°F", color='#228b22',
                        bbox=BBOX_STYLE, transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_DATA,
                        ha="right", va="top", zorder=12,
                        path_effects=text_outline)
                if clip_patch: dp_text.set_clip_path(clip_patch)
                
            alt_inHg = all_altimeters.get(city)
            if alt_inHg is not None:
                alt_text = ax.text(plot_lon + 0.07, plot_lat - 0.16, f"{alt_inHg:.2f}\"Hg", color='#4169e1',
                        bbox=BBOX_STYLE, transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_DATA,
                        ha="left", va="top", zorder=12,
                        path_effects=text_outline)
                if clip_patch: alt_text.set_clip_path(clip_patch)

            # --- MODIFIED BLOCK: Combined Icon and Weather Condition Plotting ---
            condition = all_weather_conditions.get(city)
            if condition:
                cond_text = condition.split(' & ')[0] if ' & ' in condition else condition

                if condition in WEATHER_ICONS:
                    try:
                        # 1. Create the Image Box
                        img = WEATHER_ICONS[condition]
                        im = OffsetImage(img, zoom=0.23)
                        
                        # 2. Create the Text Box
                        text_area = TextArea(
                            f" {cond_text}", # Add a space for padding
                            textprops=dict(
                                color='darkred',
                                fontsize=FONT_SIZE_WEATHER,
                                fontweight='bold',
                                va='center' # Vertical alignment
                            )
                        )
                        
                        # 3. Pack them together horizontally
                        packer = HPacker(
                            children=[im, text_area],
                            sep=2, # Separation in points
                            align="center",
                            pad=0 
                        )
                        
                        # 4. Create the final AnnotationBbox with a frame
                        ab = AnnotationBbox(
                            packer,
                            (plot_lon, plot_lat - 0.28), # Anchor point on the map
                            xybox=(0, 0), # No offset from anchor
                            xycoords='data',
                            boxcoords="offset points",
                            frameon=True, # **** TURN ON THE FRAME ****
                            box_alignment=(0.5, 1.0), # (center, top)
                            zorder=20,
                            bboxprops=dict(
                                boxstyle="round,pad=0.3", 
                                facecolor='lightyellow', 
                                alpha=0.8,
                                edgecolor='gray', # Add a subtle edge
                                linewidth=0.5
                            )
                        )
                        
                        if clip_patch: ab.set_clip_path(clip_patch)
                        ax.add_artist(ab)
                    
                    except Exception as e:
                        logging.warning(f"Error loading icon or creating HPacker for {condition}: {e}")
                        # Fallback to just text if HPacker fails
                        cond_text_obj = ax.text(plot_lon, plot_lat - 0.28, cond_text, color='darkred',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
                                transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_WEATHER,
                                ha="center", va="top", zorder=12, fontweight='bold',
                                path_effects=text_outline)
                        if clip_patch: cond_text_obj.set_clip_path(clip_patch)
                
                else: # No icon for this condition, plot text as before
                    cond_text_obj = ax.text(plot_lon, plot_lat - 0.28, cond_text, color='darkred',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
                            transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_WEATHER,
                            ha="center", va="top", zorder=12, fontweight='bold',
                            path_effects=text_outline)
                    if clip_patch: cond_text_obj.set_clip_path(clip_patch)
            # --- END OF MODIFIED BLOCK ---
        
        else:
            # --- NEW: SILENT PLOT LOGIC ---
            if temp is not None:
                color = cmap(norm(temp))
                # Plot a small, temp-colored dot
                scatter_point = ax.scatter(base_lon, base_lat, color=color, s=25, edgecolor='black', linewidth=0.5,
                                           transform=ccrs.PlateCarree(), zorder=7, alpha=0.8)
                if clip_patch: scatter_point.set_clip_path(clip_patch)
                # Plot the tiny ICAO label above it
                text_obj = ax.text(base_lon, base_lat + 0.05, icao, transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_ICAO,
                                   ha="center", va="bottom", color='black', fontweight='normal', zorder=8,
                                   path_effects=[pe.withStroke(linewidth=1, foreground='white')])
                if clip_patch: text_obj.set_clip_path(clip_patch)
                
    # --- MODIFIED: Define all known static alert event names ---
    STATIC_ALERT_EVENTS = []
    # --- END MODIFIED ---

    # Enhanced Alert Plotting (zorder 5)
    logging.info("Plotting NWS alerts...")
    alert_handles = {}
    
    # --- ADDED TRY/EXCEPT BLOCK FOR ROBUSTNESS ---
    for alert in alerts:
        try:
            for polygon_coords in alert['polygons']:
                lons, lats = zip(*polygon_coords)
                fill_poly = ax.fill(lons, lats, color=alert['color'], alpha=ALERT_ALPHA,
                        transform=ccrs.PlateCarree(), zorder=5, hatch='//')[0] # zorder 5
                line_poly = ax.plot(lons, lats, color=alert['color'], linewidth=1.5,
                        transform=ccrs.PlateCarree(), zorder=5)[0] # zorder 5
                
                if clip_patch:
                    fill_poly.set_clip_path(clip_patch)
                    line_poly.set_clip_path(clip_patch)

            # --- MODIFIED: Only add legend handles for non-static, non-winter alerts ---
            event_name = alert['event']
            
            # Check if the event name is one of our static ones
            is_static = False
            for static_name in STATIC_ALERT_EVENTS:
                if static_name in event_name: 
                    is_static = True
                    break
            
            if not is_static and event_name not in alert_handles:
                alert_handles[event_name] = mpatches.Patch(
                    color=alert['color'], alpha=0.4, label=event_name, hatch='//'
                )
        except KeyError as e:
            logging.error(f"Alert object missing key '{e}'. Skipping malformed alert: {alert.get('event', 'Unknown Event')}")
            continue # Skip this single malformed alert
        except Exception as e:
            logging.error(f"Error plotting alert: {e}")
            continue
    # --- END TRY/EXCEPT BLOCK ---


    # --- MODIFIED LEGEND (to match new county lines) ---
    legend_handles = [
        mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=12,
                      markeredgecolor='darkblue', markeredgewidth=1, label='City/Temp'),
        mlines.Line2D([], [], color='#228b22', marker='_', linestyle='None', markersize=12,
                      mew=3, label='Dew Point (°F)'),
        mlines.Line2D([], [], color='#4169e1', marker='_', linestyle='None', markersize=12,
                      mew=3, label='Altimeter (inHg)'),
        mlines.Line2D([], [], color='#4b0082', marker='|', linestyle='None', markersize=12,
                      mew=3, label='Wind Barb (Grid)'),
        mpatches.Patch(color='lightyellow', alpha=0.8, label='Weather Condition'),
        mlines.Line2D([], [], color='black', linestyle=':', linewidth=0.7, label='Offset Leader'),
        mlines.Line2D([], [], color=COUNTY_BORDER_COLOR, linestyle='-.', linewidth=0.8, label='County Border'),
        mpatches.Patch(color=MAJOR_ROAD_COLOR, label='Major Highway', alpha=0.7),
        mpatches.Patch(color=SECONDARY_ROAD_COLOR, label='Secondary Highway', alpha=0.7),
    ]
    
    # --- MODIFIED: This now only adds *other* dynamic alerts (like Flood, Severe T-Storm)
    legend_handles.extend(alert_handles.values())

    # --- *** FIX 3: Placed legend in the top left of the main 'ax' *** ---
    leg = ax.legend(handles=legend_handles, loc='lower right', fontsize=FONT_SIZE_LEGEND,
                    bbox_to_anchor=(1.0, 0.0), facecolor='white', framealpha=0.9,
                    edgecolor='lightgray', fancybox=True, ncol=2)
    leg.get_frame().set_linewidth(0.5)

    for text in leg.get_texts():
        text.set_path_effects(text_outline)

    # --- Console and File Output ---
    wwa_text = "### 🔔 Active Weather Alerts in Tennessee:\n" # <-- UPDATED
    if alerts:
        event_groups = {}
        for alert in alerts:
            # Added a check here to ensure keys exist for the console output
            try:
                evt = alert['event']
                sev = alert['severity']
                if evt not in event_groups:
                    event_groups[evt] = {"severity": sev, "areas": set()}
                event_groups[evt]["areas"].update(
                    [area.strip() for area in alert['affected_areas'].split(',')]
                )
            except KeyError as e:
                logging.warning(f"Skipping malformed alert in console output: Missing key {e}")
                continue
        
        for event, data in event_groups.items():
            wwa_text += f"**{event}** ({data['severity']}):\n"
            wwa_text += f"> {', '.join(sorted(list(data['areas'])))}\n"
    else:
        wwa_text = "✅ No active weather alerts for Tennessee at this time." # <-- UPDATED

    # Print the text output to the console
    print("\n" + "="*30)
    # Clean markdown for console
    print(wwa_text.replace("### 🔔 ", "").replace("**", "").replace("> ", "")) 
    print("="*30 + "\n")

# --- *** NEW DUAL-SAVE LOGIC *** ---

    # Get the project root (assuming this script is in a subfolder like 'state maps')
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    OUTPUT_DIR_ARCHIVE = os.path.join(PROJECT_ROOT, "output")

    # --- Create filenames ---
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dynamic_filename = f"tn_detailed_weather_{timestamp_str}.png" # <-- UPDATED
    static_filename = "tn_detailed_weather.png" # For the GUI preview # <-- UPDATED

    # --- Define file paths ---
    # Dynamic/archive file goes to the 'output' directory
    dynamic_filepath = os.path.join(OUTPUT_DIR_ARCHIVE, dynamic_filename)
    # Static file goes to the PROJECT ROOT (where the GUI expects it)
    static_filepath = os.path.join(PROJECT_ROOT, static_filename)

    # --- Ensure the archive directory exists ---
    try:
        if not os.path.exists(OUTPUT_DIR_ARCHIVE):
            os.makedirs(OUTPUT_DIR_ARCHIVE)
            logging.info(f"Created output directory: {OUTPUT_DIR_ARCHIVE}")
    except Exception as e:
        logging.error(f"Could not create output directory: {e}. Defaulting to project root.")
        # Fallback for dynamic path if 'output' dir fails
        dynamic_filepath = os.path.join(PROJECT_ROOT, dynamic_filename)

    # --- Save the plot to a memory buffer ---
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=200, bbox_inches='tight')
    buf.seek(0)
    
    # --- Write the buffer to BOTH files ---
    try:
        # 1. Save the dynamic/archive copy
        with open(dynamic_filepath, 'wb') as f:
            f.write(buf.getbuffer())
        logging.info(f"Archive map successfully saved to {dynamic_filepath}")
        
        # 2. Save the static/preview copy
        buf.seek(0) # Rewind buffer
        with open(static_filepath, 'wb') as f:
            f.write(buf.getbuffer())
        logging.info(f"Preview map successfully saved to {static_filepath}")

    except Exception as e:
        logging.error(f"Error saving map to file: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory


if __name__ == "__main__":
    """
    Main execution entry point for the script.
    """
    # This is needed to run the map generation function
    try:
        generate_tennessee_weather_map() # <-- UPDATED
    except KeyboardInterrupt:
        logging.info("Map generation cancelled by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"--- A CRITICAL UNEXPECTED ERROR OCCURRED ---")
        print(f"{e}")