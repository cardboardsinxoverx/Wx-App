#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Georgia (GA) Forecast Map Generator.
This script fetches forecast data from the NDFD
to generate High and Low temperature maps for Georgia and neighboring regions.

*** V65 Update (BROADCAST THEME APPLICATION):
    1. THEME: Applied 'ga_wx.py' Dark Gray/Broadcast theme.
    2. HEADER: Added TV-style header bar.
    3. STYLING: Updated City Badges (Black/White) and Data Boxes (Cyan Borders).
    4. ROADS: Changed to Dark Red signature style.
"""
import matplotlib
matplotlib.use('Agg') # Force headless mode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from datetime import datetime, timedelta, timezone
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from metpy.plots import USCOUNTIES, ctables
from metpy.units import units
from metpy.calc import wind_components
import logging
import xarray as xr
from siphon.catalog import TDSCatalog
import warnings
import io
import pandas as pd
import requests.exceptions
import gzip 
import unlzw3
import concurrent.futures
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from metpy.plots import ColdFront, WarmFront, StationaryFront, OccludedFront
import importlib.util

# --- OFFSETBOX IMPORTS ---
from matplotlib.offsetbox import (
    AnnotationBbox, HPacker, OffsetImage, VPacker, TextArea, PaddedBox
)

# --- HERBIE IMPORT ---
try:
    from herbie import Herbie
    HERBIE_AVAILABLE = True
except ImportError:
    HERBIE_AVAILABLE = False

# --- IMPORTS FOR CLIPPING ---
from shapely.geometry import Polygon, LineString 
from shapely.ops import unary_union
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from cartopy.mpl.path import shapely_to_path

import pickle
import time
from xarray.backends import NetCDF4DataStore

# --- SCRIPT CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- CACHING CONFIGURATION ---
CACHE_DURATION_SECONDS = 3600
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    logging.warning("__file__ not defined. Using current working directory for cache.")
    SCRIPT_DIR = os.getcwd()
    
CACHE_DIR = os.path.join(SCRIPT_DIR, ".wx_cache")

try:
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logging.info(f"Created cache directory: {CACHE_DIR}")
except Exception as e:
    logging.warning(f"Could not create cache directory: {e}. Caching will be disabled.")
    CACHE_DIR = None

# Try to load a local RTTOV helper if present (loads `rttov_helper.py` from script dir)
rttov_helper = None
try:
    helper_path = os.path.join(SCRIPT_DIR, "rttov_helper.py")
    if os.path.exists(helper_path):
        spec = importlib.util.spec_from_file_location("rttov_helper", helper_path)
        rttov_helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rttov_helper)
        logging.info("Loaded rttov_helper module.")
    else:
        rttov_helper = None
except Exception as _err:
    logging.info(f"rttov_helper load failed: {_err}")
    rttov_helper = None

# Try to load a local CRTM helper if present
crtm_helper = None
try:
    helper_path = os.path.join(SCRIPT_DIR, "crtm_helper.py")
    if os.path.exists(helper_path):
        spec = importlib.util.spec_from_file_location("crtm_helper", helper_path)
        crtm_helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crtm_helper)
        logging.info("Loaded crtm_helper module.")
    else:
        crtm_helper = None
except Exception as _err:
    logging.info(f"crtm_helper load failed: {_err}")
    crtm_helper = None

# --- MAP CONFIGURATION ---
MAP_EXTENT = [-86.8, -80.7, 30.2, 35.8]

# --- UPDATED CITY LIST (Includes Neighbors) ---
CITIES = {
    # Georgia
    "Atlanta":       (33.7490,  -84.3880),
    "Savannah":      (32.0809,  -81.0912),
    "Athens":        (33.9519,  -83.3576),
    "Rome":          (34.2570,  -85.1647),
    "Dalton":        (34.7698,  -84.9702),
    "Gainesville":   (34.2979,  -83.8241),
    "Clayton":       (34.8781,  -83.4011),
    "Valdosta":      (30.8327,  -83.2785),
    "Albany":        (31.5785,  -84.1557),
    "Waycross":      (31.2136,  -82.3549),
    "Augusta":       (33.3699,  -81.9625),
    "Macon":         (32.6934,  -83.6489),
    "Columbus":      (32.5161,  -84.9389),
    "Brunswick":     (31.1518,  -81.3911),
    "Marietta":      (33.9653,  -84.5498),
    "Milton":        (34.1322,  -84.3000),
    "LaGrange":      (33.0368,  -85.0319),
    "Statesboro":    (32.4452,  -81.7791),
    "Milledgeville": (33.0876,  -83.2332),
    "Helen":         (34.7020,  -83.7274),
    "Americus":      (32.0741,  -84.2330),
    "Thomasville":   (30.8366,  -83.9821),
    "Dublin":        (32.5404,  -82.9038),
    # Neighbors
    "Chattanooga":   (35.0456, -85.3097), # TN
    "Tallahassee":   (30.4383, -84.2807), # FL
    "Greenville":    (34.8526, -82.3940), # SC
    "Dothan":        (31.2232, -85.3905), # AL
    "Auburn":        (32.6099, -85.4808), # AL
    "Asheville":     (35.5951, -82.5515), # NC
}

# --- UPDATED LAYOUTS (Includes Neighbors) ---
CUSTOM_LAYOUTS = {
    # Georgia Layouts
    "Dalton":        {'pos': (-84.8, 34.9),  'ha': 'left'},
    "Clayton":       {'pos': (-83.4, 35.0),  'ha': 'left'},
    "Rome":          {'pos': (-85.35, 34.35), 'ha': 'right'},
    "Gainesville":   {'pos': (-83.7, 34.4),  'ha': 'left'},
    "Helen":         {'pos': (-83.5, 34.75), 'ha': 'left'},
    "Marietta":      {'pos': (-84.7, 34.0), 'ha': 'right'},
    "Milton":        {'pos': (-84.1, 34.25), 'ha': 'left'},
    "Atlanta":       {'pos': (-84.2, 33.6), 'ha': 'left'},
    "Athens":        {'pos': (-83.2, 34.05), 'ha': 'left'},
    "Augusta":       {'pos': (-81.8, 33.4),  'ha': 'left'},
    "Statesboro":    {'pos': (-81.6, 32.55), 'ha': 'left'},
    "Savannah":      {'pos': (-81.8, 32.1),  'ha': 'left'},
    "Macon":         {'pos': (-83.45, 32.8),  'ha': 'left'},
    "Milledgeville": {'pos': (-83.0, 33.15), 'ha': 'left'},
    "Dublin":        {'pos': (-82.7, 32.6), 'ha': 'left'},
    "LaGrange":      {'pos': (-85.2, 33.1), 'ha': 'right'},
    "Columbus":      {'pos': (-85.15, 32.6), 'ha': 'right'},
    "Americus":      {'pos': (-84.4, 32.15), 'ha': 'right'},
    "Albany":        {'pos': (-84.35, 31.65), 'ha': 'right'},
    "Valdosta":      {'pos': (-83.1, 30.7),  'ha': 'left'},
    "Thomasville":   {'pos': (-83.8, 30.75), 'ha': 'left'},
    "Waycross":      {'pos': (-82.4, 31.15), 'ha': 'left'},
    "Brunswick":     {'pos': (-81.5, 31.0),  'ha': 'left'},
    
    # Neighbor Layouts (Optimized for border visibility)
    "Chattanooga":   {'pos': (-85.2, 35.3), 'ha': 'left'},
    "Tallahassee":   {'pos': (-84.1, 30.3), 'ha': 'left'},
    "Greenville":    {'pos': (-82.2, 35.0), 'ha': 'left'},
    "Dothan":        {'pos': (-85.6, 31.2), 'ha': 'right'},
    "Auburn":        {'pos': (-85.7, 32.6), 'ha': 'right'},
    "Asheville":     {'pos': (-82.4, 35.7), 'ha': 'left'},
}

# --- STYLING CONFIGURATION (THEME APPLIED) ---
FIG_BG_COLOR = '#333333'       # Dark Gray background
LAND_COLOR = '#E6E6E6'         # Very Light Gray (High Contrast)
OCEAN_COLOR = '#B0C4DE'        # Muted Blue-Gray
STATE_BORDER_COLOR = 'black'   # Sharp Black
COUNTY_BORDER_COLOR = '#999999'# Medium Gray
MAJOR_ROAD_COLOR = '#8B0000'   # *** DARK RED (Signature look) ***
SECONDARY_ROAD_COLOR = '#A52A2A' # Lighter Red/Brown

FONT_FAMILY = 'DejaVu Sans'
FONT_SIZE_CITY = 11
FONT_SIZE_TEMP = 14
FONT_SIZE_DATA = 10
FONT_SIZE_TITLE = 20
FONT_SIZE_LEGEND = 10

# Header Colors
HEADER_MAIN_BG = 'white'
HEADER_SUB_BG = 'black'
HEADER_ACCENT = '#00AEEF' # Cyan/Blue

# The "City Badge" style (Black box, White text, Square corners)
CITY_BADGE_STYLE = dict(boxstyle="square,pad=0.3", facecolor='black', 
                        edgecolor='none', alpha=1.0) 
# The "Data Box" style (White box, Cyan border, Square corners)
DATA_BOX_STYLE = dict(boxstyle="square,pad=0.4", facecolor='white', 
                      edgecolor='#00AEEF', linewidth=2, alpha=0.9)

plt.rcParams['font.family'] = FONT_FAMILY

# --- CATALOG URLs ---
NDFD_CATALOG_URL = "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/NDFD/NWS/CONUS/CONDUIT/catalog.xml"

# --- BROADCAST HEADER FUNCTION ---
# --- BROADCAST HEADER FUNCTION (GLOSSY UPDATE) ---
def draw_broadcast_header(fig, main_title, sub_title, right_badge_text="LIVE", 
                          title_bg_color='white', title_text_color='black'):
    """
    Draws a TV-style header with AGGRESSIVE glossy effects (Hotdog Style).
    """
    # Define Coordinates
    logo_coords = [0.005, 0.875, 0.12, 0.12]
    header_x = 0.13
    header_w = 0.87
    title_y = 0.94
    title_h = 0.06
    sub_y = 0.88
    sub_h = 0.06

    # --- 1. The Logo Wrapper (Bezel) ---
    frame_color = "#C0C0C0" if title_bg_color == 'white' or title_bg_color == HEADER_MAIN_BG else title_bg_color
    
    # Outer Bezel
    logo_bg_rect = mpatches.FancyBboxPatch(
        (logo_coords[0], logo_coords[1]), logo_coords[2], logo_coords[3],
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="black", edgecolor=frame_color, linewidth=3,
        transform=fig.transFigure, zorder=19
    )
    fig.add_artist(logo_bg_rect)

    # Logo Image Axes
    logo_ax = fig.add_axes(logo_coords, zorder=20)
    logo_ax.axis('off')
    
    logo_path = "/home/desoxyn/frostbyte/frostbyte_project/boxlogo2.png"
    try:
        logo_img = plt.imread(logo_path)
        logo_ax.imshow(logo_img)
    except Exception as e:
        logo_ax.remove()
        fig.text(logo_coords[0] + 0.06, logo_coords[1] + 0.06, "GA\nWX", 
                 color='white', fontsize=20, fontweight='bold', 
                 ha='center', va='center', transform=fig.transFigure, zorder=21)

    # --- 2. Main Title Bar (Hotdog Gloss) ---
    ax_title = fig.add_axes([header_x, title_y, header_w, title_h], zorder=9)
    ax_title.set_axis_off()
    
    gloss_grad = create_aggressive_gloss(title_bg_color)
    ax_title.imshow(gloss_grad, aspect='auto', extent=[0, 1, 0, 1])

    # Top "Shine" Line
    ax_title.plot([0, 1], [0.98, 0.98], color='white', alpha=0.6, linewidth=1, transform=ax_title.transAxes)

    # Title Text
    outline_color = 'white' if title_text_color == 'black' else 'black'
    fig.text(header_x + 0.02, title_y + (title_h/2), main_title.upper(), 
             color=title_text_color, fontsize=26, fontweight='bold', 
             ha='left', va='center', transform=fig.transFigure, zorder=12,
             path_effects=[pe.withStroke(linewidth=2, foreground=outline_color)])

    # --- 3. Sub-Title Bar (Black Gloss) ---
    ax_sub = fig.add_axes([header_x, sub_y, header_w, sub_h], zorder=9)
    ax_sub.set_axis_off()
    
    black_gloss = create_aggressive_gloss('black')
    ax_sub.imshow(black_gloss, aspect='auto', extent=[0, 1, 0, 1])
    ax_sub.plot([0, 1], [0.95, 0.95], color='white', alpha=0.3, linewidth=0.5, transform=ax_sub.transAxes)

    # Sub Text
    fig.text(header_x + 0.02, sub_y + (sub_h/2), sub_title.upper(), 
             color='white', fontsize=14, fontweight='bold', 
             ha='left', va='center', transform=fig.transFigure, zorder=12)

    # --- 4. Right Badge ---
    badge_ax = fig.add_axes([0.92, 0.89, 0.07, 0.04], zorder=13)
    badge_ax.set_axis_off()
    badge_gloss = create_aggressive_gloss(HEADER_ACCENT)
    badge_ax.imshow(badge_gloss, aspect='auto', extent=[0, 1, 0, 1])
    
    fig.text(0.955, 0.91, right_badge_text, color='white', fontsize=12, fontweight='bold',
             ha='center', va='center', transform=fig.transFigure, zorder=14)

# --- CACHING & DATA HELPERS ---
def filter_outliers(data_array, variable_name="data"):
    if data_array is None: return None
    try:
        valid_data = data_array.values[~np.isnan(data_array.values)]
        if valid_data.size < 10: return data_array
        median = np.median(valid_data)
        std_dev = np.std(valid_data)
        min_limit = median - (5 * std_dev)
        max_limit = median + (5 * std_dev)
        filtered = data_array.where((data_array >= min_limit) & (data_array <= max_limit))
        filtered.attrs = data_array.attrs
        return filtered
    except: return data_array

def wind_dir_to_str(wind_dir_deg):
    if pd.isna(wind_dir_deg) or np.isnan(wind_dir_deg): return "N/A"
    try:
        idx = int((wind_dir_deg + 11.25) / 22.5) % 16
        return ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"][idx]
    except: return "N/A"

def get_closest_time(da, target_time, tolerance_hours=6):
    if da is None or len(da.metpy.time) == 0: return None
    times = da.metpy.time.values
    time_diff = np.abs((times - np.datetime64(target_time)).astype('timedelta64[h]')).astype(float)
    if time_diff.min() > tolerance_hours: return None
    idx = time_diff.argmin()
    return da.isel({da.metpy.time.dims[0]: idx})

# --- FETCH FUNCTIONS ---

# --- CACHING HELPER FUNCTIONS ---
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
                logging.info(f"Using cached data: {filename}")
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to read pickle cache from {filepath}: {e}")
    return None

def fetch_single_ndfd_var(var_name, time_start, time_end):
    # --- HYBRID STRATEGY ---
    # Use ALL_TIMES for Max/Min AND Precip (sparse data, fixes 400 error)
    # Use TIME_RANGE for dense data (Wind, Sky, Dewpoint) to save bandwidth
    if any(x in var_name for x in ["Maximum_temperature", "Minimum_temperature", "Total_precipitation"]):
        use_all_times = True
    else:
        use_all_times = False
        
    strategy_name = "ALL_TIMES" if use_all_times else "TIME_RANGE"
    logging.info(f"    -> Downloading {var_name.split('_')[0]} via {strategy_name}...")
    
    def try_download(fmt):
        cat = TDSCatalog(NDFD_CATALOG_URL)
        ds_entry = cat.datasets['Best National Weather Service CONUS Forecast Grids (CONDUIT) Time Series']
        ncss = ds_entry.subset()
        query = ncss.query()
        
        if use_all_times:
            query.all_times()
        else:
            query.time_range(time_start, time_end)
            
        query.variables(var_name)
        
        pad = 4.0
        query.lonlat_box(
            west=MAP_EXTENT[0] - pad, 
            east=MAP_EXTENT[1] + pad, 
            south=MAP_EXTENT[2] - pad, 
            north=MAP_EXTENT[3] + pad
        )
        
        query.accept(fmt)
        data = ncss.get_data(query)
        return xr.open_dataset(xr.backends.NetCDF4DataStore(data)).metpy.parse_cf()

    try:
        ds = try_download('netcdf')
        logging.info(f"    -> FINISHED: {var_name.split('_')[0]}")
        return var_name, ds
    except Exception as e:
        logging.warning(f"NetCDF3 fetch failed for {var_name}. Retrying with NetCDF4...")
        try:
            ds = try_download('netcdf4')
            logging.info(f"    -> FINISHED (Backup): {var_name.split('_')[0]}")
            return var_name, ds
        except Exception as e2:
            logging.error(f"FINAL FAILURE for {var_name}: {e2}")
            return var_name, None

def fetch_ndfd_data():
    # 1. TRY LOADING FROM CACHE FIRST (1 Hour TTL)
    cache_file = "ndfd_cache.pkl"
    cached_data = load_pickle_cache(cache_file, 3600)
    if cached_data:
        return cached_data

    # 2. IF NO CACHE, DOWNLOAD FRESH DATA
    logging.info("Fetching NDFD Forecast Data (Safe Mode - 2 Workers)...")
    
    target_vars = [
        "Maximum_temperature_height_above_ground_12_Hour_Maximum",
        "Minimum_temperature_height_above_ground_12_Hour_Minimum",
        "Total_precipitation_surface_12_Hour_Accumulation_probability_above_0p254",
        "Dewpoint_temperature_height_above_ground",
        "Total_cloud_cover_surface",
        "Wind_speed_height_above_ground",
        "Wind_direction_from_which_blowing_height_above_ground"
    ]
    now = datetime.now(timezone.utc)
    time_start = now - timedelta(hours=12)
    time_end = now + timedelta(hours=60)
    datasets = {}
    
    # Safe Mode: 2 Workers to prevent server blocks
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_single_ndfd_var, var, time_start, time_end): var for var in target_vars}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                var_name, ds = future.result(timeout=300)
                if ds: datasets[var_name] = ds
            except concurrent.futures.TimeoutError:
                logging.error(f"Request timed out for a variable.")
            except Exception as e:
                logging.error(f"Worker failed: {e}")

    if "Maximum_temperature_height_above_ground_12_Hour_Maximum" not in datasets:
        logging.error("Critical: Max Temp data failed to download.")
        return None, None, None, None
        
    now_et = now.astimezone(timezone(timedelta(hours=-5)))
    
    target_date = now_et
    if now_et.hour >= 23:
        target_date += timedelta(days=1)

    day_instant = target_date.replace(hour=16, minute=0, second=0, microsecond=0)
    night_instant = (day_instant + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)

    day_instant_utc = day_instant.astimezone(timezone.utc).replace(tzinfo=None)
    night_instant_utc = night_instant.astimezone(timezone.utc).replace(tzinfo=None)
    
    day_period = day_instant_utc 
    night_period = night_instant_utc

    def safe_get(var, target, tol):
        if var in datasets: return get_closest_time(datasets[var][var], target, tol)
        return None
    
    # --- PROCESS DATA ---
    maxt_raw = safe_get("Maximum_temperature_height_above_ground_12_Hour_Maximum", day_period, 12)
    maxt = ((maxt_raw - 273.15) * 1.8 + 32).squeeze() if maxt_raw is not None else None
    if maxt is not None: maxt = filter_outliers(maxt, "MaxT")
    
    dew_raw = safe_get("Dewpoint_temperature_height_above_ground", day_instant_utc, 6)
    dew = ((dew_raw - 273.15) * 1.8 + 32).squeeze() if dew_raw is not None else None
    
    wspd_raw = safe_get("Wind_speed_height_above_ground", day_instant_utc, 6)
    wspd = (wspd_raw * 1.94384).squeeze() if wspd_raw is not None else None

    pop_day = safe_get("Total_precipitation_surface_12_Hour_Accumulation_probability_above_0p254", day_period, 12)
    if pop_day is not None: pop_day = pop_day.squeeze()
    
    sky_day = safe_get("Total_cloud_cover_surface", day_instant_utc, 6)
    if sky_day is not None: sky_day = sky_day.squeeze()
    
    wdir_day = safe_get("Wind_direction_from_which_blowing_height_above_ground", day_instant_utc, 6)
    if wdir_day is not None: wdir_day = wdir_day.squeeze()

    highs = {'temp': maxt, 'pop': pop_day, 'dew': dew, 'sky': sky_day, 'wind_spd': wspd, 'wind_dir': wdir_day}
    
    mint_raw = safe_get("Minimum_temperature_height_above_ground_12_Hour_Minimum", night_period, 12)
    mint = ((mint_raw - 273.15) * 1.8 + 32).squeeze() if mint_raw is not None else None
    if mint is not None: mint = filter_outliers(mint, "MinT")
    
    dew_raw_n = safe_get("Dewpoint_temperature_height_above_ground", night_instant_utc, 6)
    dew_n = ((dew_raw_n - 273.15) * 1.8 + 32).squeeze() if dew_raw_n is not None else None

    wspd_raw_n = safe_get("Wind_speed_height_above_ground", night_instant_utc, 6)
    wspd_n = (wspd_raw_n * 1.94384).squeeze() if wspd_raw_n is not None else None
    
    pop_night = safe_get("Total_precipitation_surface_12_Hour_Accumulation_probability_above_0p254", night_period, 12)
    if pop_night is not None: pop_night = pop_night.squeeze()
    
    sky_night = safe_get("Total_cloud_cover_surface", night_instant_utc, 6)
    if sky_night is not None: sky_night = sky_night.squeeze()
    
    wdir_night = safe_get("Wind_direction_from_which_blowing_height_above_ground", night_instant_utc, 6)
    if wdir_night is not None: wdir_night = wdir_night.squeeze()

    lows = {'temp': mint, 'pop': pop_night, 'dew': dew_n, 'sky': sky_night, 'wind_spd': wspd_n, 'wind_dir': wdir_night}
    
    # 3. SAVE TO CACHE
    result_tuple = (highs, lows, day_instant_utc, night_instant_utc)
    save_pickle_cache(cache_file, result_tuple)
    
    return result_tuple

# --- NEW: GLOSS GENERATOR (HOTDOG STYLE) ---
def create_aggressive_gloss(base_color, n_points=256):
    """
    Creates a 'Hard Gel' / 'Glassy' button effect.
    Orientation: Horizontal (Hotdog Style) via Transpose.
    """
    import colorsys 
    
    # 1. Handle "White" (Silver Car Look)
    if base_color == 'white' or base_color == '#FFFFFF' or base_color == HEADER_MAIN_BG:
        c_top        = (1.0, 1.0, 1.0)    # Pure White
        c_mid_upper  = (0.9, 0.9, 0.9)    # Light Silver
        c_mid_lower  = (0.6, 0.6, 0.6)    # Dark Grey (Horizon)
        c_bot        = (0.85, 0.85, 0.85) # Reflective Grey
    
    # 2. Handle Black/Dark Grey (Sub-bar)
    elif base_color == 'black' or base_color == '#000000' or base_color == HEADER_SUB_BG:
        c_top        = (0.4, 0.4, 0.4)    
        c_mid_upper  = (0.1, 0.1, 0.1)    
        c_mid_lower  = (0.0, 0.0, 0.0)    
        c_bot        = (0.15, 0.15, 0.15)
        
    # 3. Handle Colored Alerts (Red, Yellow, Purple)
    else:
        rgb = mcolors.to_rgb(base_color)
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        
        # Top: Brighter and less saturated (The reflection)
        c_top = colorsys.hsv_to_rgb(h, max(0, s-0.3), min(1.0, v*1.5))
        c_mid_upper = rgb
        c_mid_lower = colorsys.hsv_to_rgb(h, min(1.0, s+0.1), v*0.7)
        c_bot = colorsys.hsv_to_rgb(h, s, min(1.0, v*1.1))

    # --- Build the Gradient Array ---
    half_n = n_points // 2
    
    R = np.concatenate([np.linspace(c_top[0], c_mid_upper[0], half_n),
                        np.linspace(c_mid_lower[0], c_bot[0], half_n)])
    G = np.concatenate([np.linspace(c_top[1], c_mid_upper[1], half_n),
                        np.linspace(c_mid_lower[1], c_bot[1], half_n)])
    B = np.concatenate([np.linspace(c_top[2], c_mid_upper[2], half_n),
                        np.linspace(c_mid_lower[2], c_bot[2], half_n)])
    
    # Stack and Transpose for Hotdog Style
    gradient = np.dstack((R, G, B))
    gradient = np.transpose(gradient, (1, 0, 2))
    
    return gradient

# --- NEW: INTERSTATE DATA (Expanded) ---
INTERSTATE_LOCS = {
    # --- GEORGIA ---
    "75": [(-84.92, 34.62), (-83.75, 31.85), (-84.50, 34.10)],
    "85": [(-84.95, 33.25), (-83.45, 34.15)],
    "20": [(-85.25, 33.65), (-82.55, 33.52)],
    "16": [(-82.90, 32.55)],
    "95": [(-81.55, 31.45)],
    "285": [(-84.20, 33.95)],
    "475": [(-83.72, 32.80)],
    "520": [(-82.05, 33.45)],
    
    # --- ALABAMA / FLORIDA ---
    "10": [(-84.50, 30.55), (-86.00, 30.65)],
    "59": [(-85.50, 34.70)],
    "65": [(-86.72, 33.50), (-86.72, 31.50)],
    
    # --- TENNESSEE ---
    "24": [(-85.40, 35.02)],
    "75N": [(-84.60, 35.30)],

    # --- SOUTH CAROLINA ---
    "85SC": [(-82.50, 34.75)],
    "26":   [(-82.05, 34.90), (-81.40, 34.20)],
    "385":  [(-82.25, 34.65)],
    "20SC": [(-81.75, 33.70), (-81.15, 33.95)],
    "77":   [(-81.05, 34.40)],
    "95SC": [(-80.85, 32.80)],

    # --- NORTH CAROLINA ---
    "85NC": [(-81.30, 35.25)],
    "26NC": [(-82.45, 35.35)],
    "40":   [(-82.80, 35.55)],
}

def generate_base_shield_image():
    """
    Generates a Red/Blue Interstate Shield icon in memory.
    Returns a numpy array representing the image.
    """
    # 1. Setup a tiny figure to draw the icon
    fig = plt.figure(figsize=(1, 1), dpi=150)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 2. Define the Geometry
    left_x, right_x = 0.1, 0.9
    mid_x = 0.5
    split_y = 0.65
    bottom_y = 0.05
    top_y = 0.95
    shoulder_y = 0.85 
    
    # --- A. Draw the Blue Body (Bottom) ---
    blue_verts = [
        (left_x, split_y), (right_x, split_y), (right_x, 0.55), 
        (mid_x, bottom_y), (left_x, 0.55), (left_x, split_y)
    ]
    blue_body = mpatches.Polygon(blue_verts, closed=True, color='#003399', ec='none')
    ax.add_patch(blue_body)

    # --- B. Draw the Red Crown (Top) ---
    red_verts = [
        (left_x, split_y), (left_x, shoulder_y), (0.2, top_y), (0.8, top_y), 
        (right_x, shoulder_y), (right_x, split_y), (left_x, split_y)
    ]
    red_crown = mpatches.Polygon(red_verts, closed=True, color='#CE2029', ec='none')
    ax.add_patch(red_crown)
    
    # --- C. Add the White Border ---
    full_perimeter = [
        (left_x, split_y), (left_x, 0.55), (mid_x, bottom_y), 
        (right_x, 0.55), (right_x, split_y), 
        (right_x, shoulder_y), (0.8, top_y), (0.2, top_y), (left_x, shoulder_y),
        (left_x, split_y)
    ]
    border = mpatches.Polygon(full_perimeter, closed=True, fill=False, edgecolor='white', linewidth=4)
    ax.add_patch(border)
    
    # --- D. Add Gloss ---
    gloss = mpatches.Ellipse((0.5, 0.85), 0.8, 0.3, color='white', alpha=0.2)
    ax.add_patch(gloss)

    # 3. Save to buffer and convert to array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    return img

def draw_interstate_shields(ax):
    """
    Draws Interstate markers using the custom Red/Blue Shield shape.
    """
    # 1. Generate the Shield Image ONCE (Performance optimization)
    shield_img_array = generate_base_shield_image()
    
    # 2. Plot Loop
    for route_key, coords_list in INTERSTATE_LOCS.items():
        # Clean the text (Turn "75N" or "85SC" into just "75" or "85")
        route_label = ''.join(filter(str.isdigit, route_key))
        
        for (lon, lat) in coords_list:
            # A. Create the Image Box (The Shield Icon)
            # Zoom 0.15 matches the map scale well
            imagebox = OffsetImage(shield_img_array, zoom=0.15) 
            
            # B. Place the Image
            ab = AnnotationBbox(imagebox, (lon, lat), frameon=False, pad=0.0, zorder=3.5)
            ax.add_artist(ab)
            
            # C. The Text (The Number)
            # Nudge text slightly down (lat - 0.02) to center it in the Blue area
            ax.text(lon, lat - 0.02, route_label, 
                    color='white', fontsize=7, fontweight='bold',
                    ha='center', va='center', transform=ccrs.PlateCarree(), zorder=3.6,
                    path_effects=[pe.withStroke(linewidth=1.0, foreground="#003399")])

def fetch_and_plot_wpc_pressure_centers(ax):
    logging.info("Fetching WPC Pressure Centers...")
    
    # --- 1. Try Cache ---
    cache_file = "wpc_pressure_cache.pkl"
    cached_gdf = load_pickle_cache(cache_file, 3600)
    
    full_gdf = None
    
    if cached_gdf is not None:
        full_gdf = cached_gdf
    else:
        # --- 2. Download Fresh ---
        base_url = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/natl_fcst_wx_chart/MapServer"
        layers_to_scan = [0, 1, 2, 3] 
        all_gdfs = []
        
        for layer_id in layers_to_scan:
            try:
                query_url = f"{base_url}/{layer_id}/query?where=1%3D1&outFields=*&f=geojson&outSR=4326"
                r = requests.get(query_url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if 'features' in data and data['features']:
                        gdf_layer = gpd.GeoDataFrame.from_features(data['features'])
                        points = gdf_layer[gdf_layer.geometry.type.isin(['Point', 'MultiPoint'])]
                        if not points.empty:
                            if points.crs is None: points.set_crs("EPSG:4326", allow_override=True)
                            all_gdfs.append(points)
            except Exception: continue

        if all_gdfs:
            full_gdf = pd.concat(all_gdfs, ignore_index=True)
            # Save Cache
            save_pickle_cache(cache_file, full_gdf)
    
    if full_gdf is None or full_gdf.empty: return

    # --- 3. Plotting Logic (Same as before) ---
    xmin_d, ymin_d, xmax_d, ymax_d = full_gdf.total_bounds
    if abs(xmin_d) > 180 or abs(xmax_d) > 180:
        try:
            full_gdf.crs = "EPSG:3857"
            full_gdf = full_gdf.to_crs("EPSG:4326")
        except: pass

    xmin, xmax, ymin, ymax = MAP_EXTENT
    buffer_deg = 15.0 
    bbox_poly = Polygon([(xmin-buffer_deg, ymin-buffer_deg), (xmin-buffer_deg, ymax+buffer_deg), 
                         (xmax+buffer_deg, ymax+buffer_deg), (xmax+buffer_deg, ymin-buffer_deg)])
    
    full_gdf = full_gdf[full_gdf.geometry.intersects(bbox_poly)]
    
    for _, row in full_gdf.iterrows():
        try:
            row_text = str(row.to_dict()).upper()
            sys_type = None
            color = 'black'
            if "HIGH" in row_text or "'TYPE': 'H'" in row_text:
                sys_type = 'H'; color = 'blue'
            elif "LOW" in row_text or "'TYPE': 'L'" in row_text:
                sys_type = 'L'; color = 'red'
            
            if not sys_type: continue

            pressure_val = "U"
            potential_cols = ['Pressure', 'PRESS', 'PMsl', 'Label', 'Text', 'Text_', 'Label_Txt']
            for key in potential_cols:
                if key in row and pd.notnull(row[key]):
                    try:
                        val_str = str(row[key])
                        digits = ''.join(filter(str.isdigit, val_str))
                        if digits and 900 <= int(digits) <= 1100:
                            pressure_val = int(digits); break
                    except: continue
            
            geom = row.geometry
            if geom.geom_type == 'MultiPoint': geom = geom.centroid
            lon_sys, lat_sys = geom.x, geom.y
            plot_lon = max(xmin, min(lon_sys, xmax))
            plot_lat = max(ymin, min(lat_sys, ymax))
            
            ha, va = 'center', 'center'
            if lon_sys <= xmin: ha = 'left'
            elif lon_sys >= xmax: ha = 'right'
            if lat_sys <= ymin: va = 'bottom'
            elif lat_sys >= ymax: va = 'top'

            ax.text(plot_lon, plot_lat, f'{sys_type}\n{pressure_val}', 
                    color=color, fontsize=14, fontweight='bold',
                    ha=ha, va=va, transform=ccrs.PlateCarree(),
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, pad=2), 
                    zorder=100, clip_on=False)
        except Exception: continue


def fetch_and_plot_wpc_fronts(ax):
    logging.info("Fetching WPC Fronts...")

    # --- 1. Try Cache ---
    cache_file = "wpc_fronts_cache.pkl"
    cached_gdf = load_pickle_cache(cache_file, 3600)
    
    full_gdf = None
    if cached_gdf is not None:
        full_gdf = cached_gdf
    else:
        # --- 2. Download Fresh ---
        base_url = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/natl_fcst_wx_chart/MapServer"
        layers_to_scan = [1, 2, 3, 4, 5, 6] 
        all_gdfs = []
        for layer_id in layers_to_scan:
            try:
                query_url = f"{base_url}/{layer_id}/query?where=1%3D1&outFields=*&f=geojson&outSR=4326"
                r = requests.get(query_url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if 'features' in data and data['features']:
                        gdf_layer = gpd.GeoDataFrame.from_features(data['features'])
                        if not gdf_layer.empty:
                            if gdf_layer.crs is None: gdf_layer.set_crs("EPSG:4326", allow_override=True)
                            all_gdfs.append(gdf_layer)
            except Exception: continue

        if all_gdfs:
            full_gdf = pd.concat(all_gdfs, ignore_index=True)
            save_pickle_cache(cache_file, full_gdf)

    if full_gdf is None or full_gdf.empty: return

    # --- 3. Plotting Logic (Same as before) ---
    xmin_d, ymin_d, xmax_d, ymax_d = full_gdf.total_bounds
    if abs(xmin_d) > 180 or abs(xmax_d) > 180:
        try:
            full_gdf.crs = "EPSG:3857"
            full_gdf = full_gdf.to_crs("EPSG:4326")
        except: pass

    xmin, xmax, ymin, ymax = MAP_EXTENT
    bbox_poly = Polygon([(xmin-5, ymin-5), (xmin-5, ymax+5), (xmax+5, ymax+5), (xmax+5, ymin-5)])
    gdf_clipped = full_gdf[full_gdf.geometry.intersects(bbox_poly)]
    
    if gdf_clipped.empty: return
    
    def get_style(feature_type):
        ft = str(feature_type).upper()
        if 'COLD' in ft: return {'color': 'blue', 'lw': 2.5, 'path_effects': [ColdFront(size=6, spacing=1.0)]}
        elif 'WARM' in ft: return {'color': 'red', 'lw': 2.5, 'path_effects': [WarmFront(size=6, spacing=1.0)]}
        elif 'STAT' in ft: return {'color': 'red', 'lw': 2.5, 'path_effects': [StationaryFront(size=6, spacing=1.0)]}
        elif 'OCCL' in ft: return {'color': 'purple', 'lw': 2.5, 'path_effects': [OccludedFront(size=6, spacing=1.0)]}
        elif 'TROF' in ft or 'TROUGH' in ft: return {'color': '#FF8C00', 'lw': 2.0, 'linestyle': '--', 'dashes': (5, 5)}
        elif 'DRY' in ft: return {'color': 'brown', 'lw': 2.0, 'linestyle': '-', 'dashes': (5, 5)}
        return None

    for _, row in gdf_clipped.iterrows():
        row_text = str(row.to_dict()).upper()
        ftype = "UNKNOWN"
        if "COLD" in row_text: ftype = "COLD"
        elif "WARM" in row_text: ftype = "WARM"
        elif "STATIONARY" in row_text: ftype = "STATIONARY"
        elif "OCCLUDED" in row_text: ftype = "OCCLUDED"
        elif "TROUGH" in row_text or "TROF" in row_text: ftype = "TROUGH"
        elif "DRY" in row_text: ftype = "DRY"
        
        style = get_style(ftype)
        if style:
            if row.geometry.geom_type == 'LineString': geoms = [row.geometry]
            elif row.geometry.geom_type == 'MultiLineString': geoms = row.geometry.geoms
            else: continue
            for line in geoms:
                x, y = line.xy
                ax.plot(x, y, transform=ccrs.PlateCarree(), zorder=50, **style)
    
def fetch_model_data(target_time_utc):
    if not HERBIE_AVAILABLE:
        logging.warning("Herbie library not installed. Skipping Radar/Cloud overlay.")
        return None

    if target_time_utc.tzinfo is None:
        target_time_utc = target_time_utc.replace(tzinfo=timezone.utc)

    now_utc = datetime.now(timezone.utc)
    
    def try_fetch(model_name, product, run_dt, fxx):
        run_str = run_dt.strftime("%Y-%m-%d %H:00")
        logging.info(f"Attempting {model_name.upper()} ({product}) Run: {run_str} F{fxx:02d}...")
        
        try:
            H = Herbie(run_str, model=model_name, product=product, fxx=fxx)
            
            ds_refl = H.xarray(":REFC:entire atmosphere", verbose=False)
            if isinstance(ds_refl, list): ds_refl = ds_refl[0]
            
            ds_cld = H.xarray(":TCDC:entire atmosphere", verbose=False)
            if isinstance(ds_cld, list): ds_cld = ds_cld[0]

            ds_mslp = H.xarray(":(?:MSLMA|PRMSL):mean sea level", verbose=False)
            if isinstance(ds_mslp, list): ds_mslp = ds_mslp[0]

            # Diagnostic: log available variable names in fetched datasets
            try:
                if ds_refl is not None and hasattr(ds_refl, 'data_vars'):
                    logging.info(f"ds_refl variables: {list(ds_refl.data_vars.keys())}")
            except Exception:
                pass
            try:
                if ds_cld is not None and hasattr(ds_cld, 'data_vars'):
                    logging.info(f"ds_cld variables: {list(ds_cld.data_vars.keys())}")
            except Exception:
                pass
            try:
                if ds_mslp is not None and hasattr(ds_mslp, 'data_vars'):
                    logging.info(f"ds_mslp variables: {list(ds_mslp.data_vars.keys())}")
            except Exception:
                pass

            # Try to fetch any model-equivalent Brightness Temperature (TB/TBB) fields
            ds_tb = None
            tb_candidates = [":TBB:entire atmosphere", ":TB:entire atmosphere", ":Brightness_temperature:entire atmosphere", ":BT:entire atmosphere"]
            for patt in tb_candidates:
                try:
                    tmp = H.xarray(patt, verbose=False)
                    if isinstance(tmp, list): tmp = tmp[0]
                    if tmp is not None and hasattr(tmp, 'data_vars') and len(tmp.data_vars) > 0:
                        ds_tb = tmp
                        break
                except Exception:
                    ds_tb = None
            
            def manual_crop(ds):
                if ds is None: return None
                if np.any(ds.longitude > 180):
                    ds = ds.assign_coords(longitude = (((ds.longitude + 180) % 360) - 180))
                return ds

            ds_refl = manual_crop(ds_refl)
            ds_cld = manual_crop(ds_cld)
            ds_mslp = manual_crop(ds_mslp)
            ds_tb = manual_crop(ds_tb)

            # If an RT radiative-transfer helper is available (RTTOV or CRTM), attempt to build profiles and compute model-equivalent TB
            helper_module = None
            try:
                if (rttov_helper is not None) and (hasattr(rttov_helper, 'is_available') and rttov_helper.is_available()):
                    helper_module = rttov_helper
                elif (crtm_helper is not None) and (hasattr(crtm_helper, 'is_available') and crtm_helper.is_available()):
                    helper_module = crtm_helper
            except Exception:
                helper_module = None

            if helper_module is not None and ds_tb is None:
                try:
                    logging.info(f"{getattr(helper_module,'__name__','RT helper')} available — attempting to fetch 3D profiles for TB computation...")
                    # Fetch temperature, specific humidity, and pressure profiles
                    ds_tprof = H.xarray(":TMP:entire atmosphere", verbose=False)
                    ds_qprof = H.xarray(":SPFH:entire atmosphere", verbose=False)
                    ds_pprof = H.xarray(":PRES:entire atmosphere", verbose=False)
                    # Surface/skin temperature and surface pressure
                    try:
                        ds_ts = H.xarray(":TS:surface", verbose=False)
                    except Exception:
                        ds_ts = None
                    try:
                        ds_ps = H.xarray(":PS:surface", verbose=False)
                    except Exception:
                        ds_ps = None

                    if ds_tprof is not None and ds_qprof is not None and ds_pprof is not None:
                        # Pull variable names and arrays
                        try:
                            t_var = next(iter(ds_tprof.data_vars))
                            q_var = next(iter(ds_qprof.data_vars))
                            p_var = next(iter(ds_pprof.data_vars))
                            t_da = ds_tprof[t_var].squeeze()
                            q_da = ds_qprof[q_var].squeeze()
                            p_da = ds_pprof[p_var].squeeze()

                            # Normalize shapes: remove leading time dim if present
                            def _normalize_da(da):
                                arr = da.values
                                dims = list(da.dims)
                                if arr.ndim == 4:
                                    arr = arr[0]
                                    # adjust dims
                                # If level axis is not first, try to move it to axis 0
                                level_axes = [d for d in dims if d.lower() in ('level', 'vertical', 'isobaric', 'pressure')]
                                if level_axes:
                                    lvl = level_axes[0]
                                    li = list(dims).index(lvl)
                                    if li != 0:
                                        arr = np.moveaxis(arr, li, 0)
                                return arr, da

                            t_arr, t_ref = _normalize_da(t_da)
                            q_arr, q_ref = _normalize_da(q_da)
                            p_arr, p_ref = _normalize_da(p_da)

                            # Surface fields
                            ts_arr = None
                            ps_arr = None
                            if ds_ts is not None:
                                try:
                                    ts_var = next(iter(ds_ts.data_vars))
                                    ts_arr = ds_ts[ts_var].squeeze().values
                                except Exception:
                                    ts_arr = None
                            if ds_ps is not None:
                                try:
                                    ps_var = next(iter(ds_ps.data_vars))
                                    ps_arr = ds_ps[ps_var].squeeze().values
                                except Exception:
                                    ps_arr = None

                            profiles = {'p': p_arr, 't': t_arr, 'q': q_arr, 'ts': ts_arr, 'ps': ps_arr}

                            # Call RTTOV helper (channel selection left simple — assume ABI band 13-like index 13)
                            try:
                                bt_k = rttov_helper.compute_bt_for_channel(profiles, channel=13)
                                if bt_k is not None:
                                    # Build xarray DataArray for TB (Kelvin) using the horizontal coords from reference arrays
                                    # Attempt to get lon/lat coords from one of the references
                                    try:
                                        lon = None
                                        lat = None
                                        if hasattr(t_ref, 'longitude') and hasattr(t_ref, 'latitude'):
                                            lon = t_ref.longitude
                                            lat = t_ref.latitude
                                        elif hasattr(t_ref, 'x') and hasattr(t_ref, 'y'):
                                            lon = t_ref.x
                                            lat = t_ref.y
                                        else:
                                            # Fallback: try p_ref coords
                                            lon = p_ref.longitude if hasattr(p_ref, 'longitude') else None
                                            lat = p_ref.latitude if hasattr(p_ref, 'latitude') else None
                                    except Exception:
                                        lon = None; lat = None

                                    try:
                                        import xarray as _xr
                                        if lon is not None and lat is not None:
                                            tb_da = _xr.DataArray(bt_k, coords={"y": lat.values, "x": lon.values}, dims=("y","x"))
                                        else:
                                            tb_da = _xr.DataArray(bt_k)
                                        out_tb_kelvin = tb_da
                                        out['tb'] = out_tb_kelvin
                                        logging.info(f"Computed model-equivalent TB via RTTOV for {model_name.upper()} {run_str} F{fxx:02d}")
                                    except Exception as _ex:
                                        logging.warning(f"Failed to build TB DataArray: {_ex}")
                            except Exception as _ex2:
                                logging.warning(f"RTTOV helper compute call failed: {_ex2}")
                        except Exception as _e:
                            logging.info(f"Failed to prepare profiles for RTTOV: {_e}")
                except Exception as _err:
                    logging.info(f"RTTOV profile fetch/compute attempt failed: {_err}")

            # If no model-equivalent TB was found, try to fetch a near-surface temperature
            ds_temp = None
            temp_patterns = [
                ":TMP:2 m above ground",
                ":TMP:2 m",
                ":Temperature_height_above_ground",
                ":Temperature_surface",
                ":t2m:entire atmosphere",
                ":2 m temperature",
            ]
            for patt in temp_patterns:
                try:
                    tmp = H.xarray(patt, verbose=False)
                    if isinstance(tmp, list): tmp = tmp[0]
                    if tmp is not None and hasattr(tmp, 'data_vars') and len(tmp.data_vars) > 0:
                        ds_temp = tmp
                        logging.info(f"Found candidate temp dataset for pattern '{patt}' with vars: {list(ds_temp.data_vars.keys())}")
                        break
                except Exception:
                    continue
            
            refl = ds_refl['refc'] if 'refc' in ds_refl else (ds_refl['REFC'] if 'REFC' in ds_refl else None)
            cld = ds_cld['tcdc'] if 'tcdc' in ds_cld else (ds_cld['TCDC'] if 'TCDC' in ds_cld else None)
            
            mslp_var = None
            if ds_mslp:
                for v in ds_mslp.data_vars:
                    if 'msl' in v.lower() or 'prmsl' in v.lower():
                        mslp_var = ds_mslp[v]
                        break

            if refl is not None:
                logging.info(f"SUCCESS: Fetched {model_name.upper()} {run_str} F{fxx:02d}")
                out = {'refl': refl, 'clouds': cld, 'mslp': mslp_var, 'source': f"{model_name.upper()}"}
                # Prefer true model TB if present
                if ds_tb is not None:
                    try:
                        tb_var = next(iter(ds_tb.data_vars))
                        out['tb'] = ds_tb[tb_var]
                        logging.info(f"Found TB variable in {model_name.upper()} dataset: {tb_var}")
                    except Exception as ex:
                        logging.warning(f"Failed to extract TB variable from dataset: {ex}")
                # Fallback: use near-surface model temperature as a TB surrogate
                if 'tb' not in out and ds_temp is not None:
                    try:
                        temp_var = next(iter(ds_temp.data_vars))
                        out['tb'] = ds_temp[temp_var]
                        logging.info(f"Using surface temperature variable as TB surrogate: {temp_var}")
                    except Exception as ex:
                        logging.warning(f"Failed to extract temp variable for TB surrogate: {ex}")
                return out
            
        except Exception as e:
            pass
        return None

    # STRATEGY 1: HRRR
    hrrr_attempts = []
    latest_hrrr_run = now_utc - timedelta(hours=1) 
    latest_hrrr_run = latest_hrrr_run.replace(minute=0, second=0, microsecond=0)
    
    fxx_latest = int((target_time_utc - latest_hrrr_run).total_seconds() / 3600)
    
    if 0 <= fxx_latest <= 18:
        hrrr_attempts.append((latest_hrrr_run, fxx_latest))
    
    for hours_back in range(0, 24):
        past_run = latest_hrrr_run - timedelta(hours=hours_back)
        if past_run.hour % 6 == 0: 
            fxx_extended = int((target_time_utc - past_run).total_seconds() / 3600)
            if 0 <= fxx_extended <= 48:
                hrrr_attempts.append((past_run, fxx_extended))
                
    for run_time, fxx in hrrr_attempts:
        res = try_fetch('hrrr', 'sfc', run_time, fxx)
        if res: return res

    # STRATEGY 2: RAP (Try RAP model-equivalent fields if available)
    logging.info("HRRR not available for this target. Trying RAP model next...")
    try:
        latest_rap_run = now_utc - timedelta(hours=1)
        latest_rap_run = latest_rap_run.replace(minute=0, second=0, microsecond=0)
        fxx_rap = int((target_time_utc - latest_rap_run).total_seconds() / 3600)
        if 0 <= fxx_rap <= 18:
            res = try_fetch('rap', 'sfc', latest_rap_run, fxx_rap)
            if res: return res
        # try a few past RAP hourly runs (up to 6 hours back)
        for hours_back in range(1, 7):
            past = latest_rap_run - timedelta(hours=hours_back)
            fxx_past = int((target_time_utc - past).total_seconds() / 3600)
            if 0 <= fxx_past <= 48:
                res = try_fetch('rap', 'sfc', past, fxx_past)
                if res: return res
    except Exception:
        logging.info("RAP attempt encountered an error; continuing to NAM fallback.")

    # STRATEGY 3: NAM 3km
    logging.info("Falling back to NAM 3km...")
    latest_nam = now_utc - timedelta(hours=3)
    run_hour_nam = (latest_nam.hour // 6) * 6
    nam_run_dt = latest_nam.replace(hour=run_hour_nam, minute=0, second=0, microsecond=0)
    
    fxx_nam = int((target_time_utc - nam_run_dt).total_seconds() / 3600)
    
    if 0 <= fxx_nam <= 60:
        res = try_fetch('nam', 'conusnest.hiresf', nam_run_dt, fxx_nam)
        if res: return res
        
        prev_nam = nam_run_dt - timedelta(hours=6)
        fxx_nam_prev = int((target_time_utc - prev_nam).total_seconds() / 3600)
        res = try_fetch('nam', 'conusnest.hiresf', prev_nam, fxx_nam_prev)
        if res: return res

    logging.error("All Model Fetches Failed.")
    return None

def _resolve_text_collisions(fig, ax, items, padding_px=5, max_iter=500, pixel_step=10):
    if not items:
        return

    # 1. Get renderer ONCE (Huge speedup)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # 2. Setup transforms
    inv = ax.transData.inverted()

    # 3. Helper to get boxes efficiently
    def get_box(item):
        # We must use the renderer we grabbed earlier
        return item.get_window_extent(renderer=renderer).expanded(1.0, 1.0)

    def boxes_overlap(b1, b2, pad=0):
        return not (b1.x1 + pad < b2.x0 or b1.x0 - pad > b2.x1 or b1.y1 + pad < b2.y0 or b1.y0 - pad > b2.y1)

    flattened = [{'item': it['item']} for it in items]

    for _ in range(max_iter):
        moved = False
        # Cache boxes for this iteration
        current_boxes = [get_box(f['item']) for f in flattened]

        for i in range(len(flattened)):
            for j in range(i + 1, len(flattened)):
                b1 = current_boxes[i]
                b2 = current_boxes[j]
                
                if boxes_overlap(b1, b2, pad=padding_px):
                    # Calculate centers
                    c1 = ((b1.x0 + b1.x1) / 2.0, (b1.y0 + b1.y1) / 2.0)
                    c2 = ((b2.x0 + b2.x1) / 2.0, (b2.y0 + b2.y1) / 2.0)
                    
                    dx = c1[0] - c2[0]
                    dy = c1[1] - c2[1]
                    
                    if abs(dx) < 1e-3 and abs(dy) < 1e-3: dx, dy = 1.0, 1.0
                    
                    # Move logic
                    mag = pixel_step
                    norm = (dx**2 + dy**2)**0.5
                    ux, uy = dx / norm, dy / norm
                    move_disp = (ux * mag, uy * mag)

                    # Transform pixel move to data coords
                    c1_data = inv.transform(c1)
                    c1_moved_data = inv.transform((c1[0] + move_disp[0], c1[1] + move_disp[1]))
                    
                    diff_x = c1_moved_data[0] - c1_data[0]
                    diff_y = c1_moved_data[1] - c1_data[1]

                    item_artist = flattened[i]['item']
                    
                    # Handle AnnotationBbox vs Text
                    if isinstance(item_artist, AnnotationBbox):
                        old_x, old_y = item_artist.xy
                        new_pos = (old_x + diff_x, old_y + diff_y)
                        item_artist.xy = new_pos
                    else:
                        old_x, old_y = item_artist.get_position()
                        new_pos = (old_x + diff_x, old_y + diff_y)
                        item_artist.set_position(new_pos)
                    
                    # Update Leader Line
                    if hasattr(item_artist, 'leader_line'):
                        leader = item_artist.leader_line
                        xdata, ydata = leader.get_xdata(), leader.get_ydata()
                        leader.set_xdata([xdata[0], new_pos[0]])
                        leader.set_ydata([ydata[0], new_pos[1]])
                        
                    moved = True
        
        if not moved:
            break

# --- 1. WEATHER ICON GENERATORS (Copied from ga_wx.py) ---
def create_icon_base(fig_size=(1,1), dpi=100):
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    return fig, ax

def _fig_to_img(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0); img = plt.imread(buf); plt.close(fig); buf.close()
    return img

def create_sunny_icon():
    fig, ax = create_icon_base()
    ax.add_patch(mpatches.Circle((0.5, 0.5), 0.4, color='yellow'))
    for angle in range(0, 360, 45):
        x = 0.5 + 0.45 * np.cos(np.radians(angle)); y = 0.5 + 0.45 * np.sin(np.radians(angle))
        ax.plot([0.5, x], [0.5, y], color='yellow')
    return _fig_to_img(fig)

def create_moonlight_icon():
    fig, ax = create_icon_base()
    ax.add_patch(mpatches.Circle((0.5, 0.5), 0.4, color='lightgray'))
    ax.add_patch(mpatches.Circle((0.6, 0.6), 0.35, color='darkgray'))
    return _fig_to_img(fig)

def create_mostly_sunny_icon():
    fig, ax = create_icon_base()
    ax.add_patch(mpatches.Circle((0.5, 0.5), 0.4, color='yellow'))
    for angle in range(0, 360, 45):
        x = 0.5 + 0.45 * np.cos(np.radians(angle)); y = 0.5 + 0.45 * np.sin(np.radians(angle))
        ax.plot([0.5, x], [0.5, y], color='yellow')
    ax.add_patch(mpatches.Circle((0.75, 0.25), 0.2, color='white', alpha=0.9))
    return _fig_to_img(fig)

def create_partially_cloudy_sun_icon():
    fig, ax = create_icon_base()
    ax.add_patch(mpatches.Circle((0.3, 0.3), 0.3, color='yellow')) # Sun
    ax.add_patch(mpatches.Ellipse((0.7, 0.7), 0.6, 0.4, color='white')) # Cloud
    ax.add_patch(mpatches.Circle((0.6, 0.65), 0.2, color='white'))
    ax.add_patch(mpatches.Circle((0.8, 0.65), 0.2, color='white'))
    return _fig_to_img(fig)

def create_partially_cloudy_moon_icon():
    fig, ax = create_icon_base()
    ax.add_patch(mpatches.Circle((0.3, 0.3), 0.3, color='lightgray'))
    ax.add_patch(mpatches.Circle((0.4, 0.4), 0.25, color='darkgray'))
    ax.add_patch(mpatches.Ellipse((0.7, 0.7), 0.6, 0.4, color='white'))
    ax.add_patch(mpatches.Circle((0.6, 0.65), 0.2, color='white'))
    ax.add_patch(mpatches.Circle((0.8, 0.65), 0.2, color='white'))
    return _fig_to_img(fig)

def create_cloudy_icon():
    fig, ax = create_icon_base()
    ax.add_patch(mpatches.Ellipse((0.5, 0.5), 0.8, 0.5, color='gray'))
    ax.add_patch(mpatches.Circle((0.4, 0.55), 0.3, color='gray'))
    ax.add_patch(mpatches.Circle((0.6, 0.55), 0.3, color='gray'))
    return _fig_to_img(fig)

def create_rain_icon():
    fig, ax = create_icon_base()
    ax.add_patch(mpatches.Ellipse((0.5, 0.8), 0.8, 0.4, color='gray'))
    for i in range(5): ax.plot([0.2 + i*0.15, 0.15 + i*0.15], [0.6, 0.4], color='blue')
    return _fig_to_img(fig)

def create_snowing_icon():
    fig, ax = create_icon_base()
    ax.add_patch(mpatches.Ellipse((0.5, 0.8), 0.8, 0.4, color='gray'))
    for pos in [(0.3,0.5), (0.5,0.4), (0.7,0.5)]: ax.text(pos[0], pos[1], '*', color='white', fontsize=20)
    return _fig_to_img(fig)

def create_wintry_mix_icon():
    fig, ax = create_icon_base()
    ax.add_patch(mpatches.Ellipse((0.5, 0.8), 0.8, 0.4, color='gray'))
    ax.plot([0.3, 0.25], [0.5, 0.3], color='blue')
    ax.text(0.7, 0.4, '*', color='white', fontsize=20)
    return _fig_to_img(fig)

def create_mostly_cloudy_storming_icon():
    fig, ax = create_icon_base()
    ax.add_patch(mpatches.Ellipse((0.5, 0.7), 0.8, 0.5, color='darkgray'))
    ax.add_patch(mpatches.Polygon([[0.4, 0.4], [0.5, 0.6], [0.6, 0.4], [0.5, 0.5]], color='yellow'))
    return _fig_to_img(fig)

# --- 2. FORECAST LOGIC (Converts NDFD data to Icons) ---
def get_forecast_icon_key(sky_cover, pop, temp_f, is_day=True):
    """Determines the correct icon key based on NDFD forecast variables."""
    # 1. Precip Logic (Priority)
    if pop is not None and pop >= 30:
        if temp_f is not None and temp_f <= 35:
            if temp_f <= 32: return 'Snow'
            return 'Wintry Mix'
        if pop >= 60: return 'Storms' # Assume storms for high PoP
        return 'Rain'

    # 2. Sky Condition Logic
    if sky_cover is None: return 'Sunny' if is_day else 'Moonlight'
    
    if sky_cover < 10: return 'Sunny' if is_day else 'Moonlight'
    elif sky_cover < 30: return 'Mostly Sunny' if is_day else 'Moonlight'
    elif sky_cover < 60: return 'Partly Cloudy (Sun)' if is_day else 'Partly Cloudy (Moon)'
    elif sky_cover < 90: return 'Cloudy'
    else: return 'Overcast'

def _resolve_text_collisions(fig, ax, items, padding_px=5, max_iter=500, pixel_step=10):
    if not items:
        return

    # 1. Get renderer ONCE (Huge speedup)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # 2. Setup transforms
    inv = ax.transData.inverted()

    # 3. Helper to get boxes efficiently
    def get_box(item):
        # We must use the renderer we grabbed earlier
        return item.get_window_extent(renderer=renderer).expanded(1.0, 1.0)

    def boxes_overlap(b1, b2, pad=0):
        return not (b1.x1 + pad < b2.x0 or b1.x0 - pad > b2.x1 or b1.y1 + pad < b2.y0 or b1.y0 - pad > b2.y1)

    flattened = [{'item': it['item']} for it in items]

    for _ in range(max_iter):
        moved = False
        # Cache boxes for this iteration
        current_boxes = [get_box(f['item']) for f in flattened]

        for i in range(len(flattened)):
            for j in range(i + 1, len(flattened)):
                b1 = current_boxes[i]
                b2 = current_boxes[j]
                
                if boxes_overlap(b1, b2, pad=padding_px):
                    # Calculate centers
                    c1 = ((b1.x0 + b1.x1) / 2.0, (b1.y0 + b1.y1) / 2.0)
                    c2 = ((b2.x0 + b2.x1) / 2.0, (b2.y0 + b2.y1) / 2.0)
                    
                    dx = c1[0] - c2[0]
                    dy = c1[1] - c2[1]
                    
                    if abs(dx) < 1e-3 and abs(dy) < 1e-3: dx, dy = 1.0, 1.0
                    
                    # Move logic
                    mag = pixel_step
                    norm = (dx**2 + dy**2)**0.5
                    ux, uy = dx / norm, dy / norm
                    move_disp = (ux * mag, uy * mag)

                    # Transform pixel move to data coords
                    c1_data = inv.transform(c1)
                    c1_moved_data = inv.transform((c1[0] + move_disp[0], c1[1] + move_disp[1]))
                    
                    diff_x = c1_moved_data[0] - c1_data[0]
                    diff_y = c1_moved_data[1] - c1_data[1]

                    item_artist = flattened[i]['item']
                    
                    # Handle AnnotationBbox vs Text
                    if isinstance(item_artist, AnnotationBbox):
                        old_x, old_y = item_artist.xy
                        new_pos = (old_x + diff_x, old_y + diff_y)
                        item_artist.xy = new_pos
                    else:
                        old_x, old_y = item_artist.get_position()
                        new_pos = (old_x + diff_x, old_y + diff_y)
                        item_artist.set_position(new_pos)
                    
                    # Update Leader Line
                    if hasattr(item_artist, 'leader_line'):
                        leader = item_artist.leader_line
                        xdata, ydata = leader.get_xdata(), leader.get_ydata()
                        leader.set_xdata([xdata[0], new_pos[0]])
                        leader.set_ydata([ydata[0], new_pos[1]])
                        
                    moved = True
        
        if not moved:
            break

def plot_forecast_map(title, forecast_data_dict, nam_overlays, radar_time_str, filename):
    logging.info(f"Generating map: {title}...")
    warnings.filterwarnings("ignore")
    
    # --- 1. ICON SETUP ---
    WEATHER_ICONS = {
        'Sunny': create_sunny_icon(),
        'Moonlight': create_moonlight_icon(),
        'Mostly Sunny': create_mostly_sunny_icon(),
        'Partly Cloudy (Sun)': create_partially_cloudy_sun_icon(),
        'Partly Cloudy (Moon)': create_partially_cloudy_moon_icon(),
        'Cloudy': create_cloudy_icon(),
        'Overcast': create_cloudy_icon(),
        'Rain': create_rain_icon(),
        'Storms': create_mostly_cloudy_storming_icon(),
        'Snow': create_snowing_icon(),
        'Wintry Mix': create_wintry_mix_icon(),
    }

    # 2. SETUP FIGURE
    fig = plt.figure(figsize=(16, 16), facecolor=FIG_BG_COLOR)
    text_outline = [pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()]
    model_src = nam_overlays.get('source', 'Model') if nam_overlays else "Model"
    
    # 3. HEADER
    draw_broadcast_header(fig, title.upper(), f"VALID: {radar_time_str} | SOURCE: {model_src}", "FORECAST")
    
    # --- MATCHED LAYOUT (Exact same as ga_wx.py) ---
    map_x = 0.08  
    map_y = 0.05  
    map_w = 0.84  
    map_h = 0.82  

    ax = fig.add_axes([map_x, map_y, map_w, map_h], projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    # Clip Path logic
    clip_patch = None
    try:
        states_shp = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_1_states_provinces')
        reader = shpreader.Reader(states_shp)
        target_states = ['Georgia', 'Alabama', 'Florida', 'Tennessee', 'North Carolina', 'South Carolina']
        state_geoms = [r.geometry for r in reader.records() if r.attributes.get('name') in target_states]
        if state_geoms:
            combined_geom = unary_union(state_geoms)
            extent_poly = Polygon([(MAP_EXTENT[0], MAP_EXTENT[2]), (MAP_EXTENT[0], MAP_EXTENT[3]), (MAP_EXTENT[1], MAP_EXTENT[3]), (MAP_EXTENT[1], MAP_EXTENT[2])])
            visible_geom = combined_geom.intersection(extent_poly)
            paths = shapely_to_path(visible_geom)
            if not isinstance(paths, list): paths = [paths]
            combined_path = Path.make_compound_path(*paths)
            clip_patch = PathPatch(combined_path, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='none')
            ax.add_patch(clip_patch)
    except Exception as e: logging.warning(f"Clipping failed: {e}")

    # --- BACKGROUND ---
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor=OCEAN_COLOR)
    ax.add_feature(cfeature.LAND, zorder=0, facecolor=LAND_COLOR, edgecolor='none')

    def safe_clip(artist):
        if not clip_patch: return
        try: artist.set_clip_path(clip_patch)
        except: 
            try: 
                for c in artist.collections: c.set_clip_path(clip_patch)
            except: pass

    # --- OVERLAYS ---
    if nam_overlays:
        try:
            logging.info(f"Overlay keys available: {list(nam_overlays.keys())}")
        except Exception:
            pass
        if nam_overlays.get('clouds') is not None:
            clouds = nam_overlays['clouds'].squeeze()
            try: transform = clouds.metpy.cartopy_crs; x, y = clouds.x, clouds.y
            except: transform = ccrs.PlateCarree(); x, y = clouds.longitude, clouds.latitude
            cf = ax.contourf(x, y, clouds, levels=np.arange(10, 101, 10), cmap='gray', alpha=0.4, zorder=1.1, transform=transform)
            safe_clip(cf)

        if nam_overlays.get('refl') is not None:
            refl = nam_overlays['refl'].squeeze()
            refl.values = gaussian_filter(refl.values, sigma=1.5)
            try: transform = refl.metpy.cartopy_crs; x, y = refl.x, refl.y
            except: transform = ccrs.PlateCarree(); x, y = refl.longitude, refl.latitude
            pm = ax.contourf(x, y, refl, levels=np.arange(5, 80, 5), cmap=ctables.registry.get_colortable('NWSReflectivity'), alpha=0.6, zorder=1.2, transform=transform, extend='max')
            safe_clip(pm)

        if nam_overlays.get('mslp') is not None:
            mslp = nam_overlays['mslp'].squeeze()
            mslp.values = gaussian_filter(mslp.values / 100.0, sigma=2.0)
            try: transform = mslp.metpy.cartopy_crs; x, y = mslp.x, mslp.y
            except: transform = ccrs.PlateCarree(); x, y = mslp.longitude, mslp.latitude
            cs_iso = ax.contour(x, y, mslp, levels=np.arange(900, 1100, 2), colors="#0E2914", linewidths=2.2, alpha=0.8, zorder=1.6, transform=transform)
            safe_clip(cs_iso)
            clabels = ax.clabel(cs_iso, inline=True, fontsize=9, fmt='%1.0f hPa', colors='white', use_clabeltext=True, zorder=1.7)
            for label in clabels: label.set_path_effects([pe.withStroke(linewidth=2.5, foreground="#0E2914")])

        # --- BRIGHTNESS TEMP (model-equivalent IR) ---
        if nam_overlays.get('tb') is not None:
            tb = nam_overlays['tb'].squeeze()
            try:
                transform = tb.metpy.cartopy_crs; x, y = tb.x, tb.y
            except Exception:
                transform = ccrs.PlateCarree(); x, y = tb.longitude, tb.latitude

            # Convert units: if values look like Kelvin (>200), convert to C
            tb_vals = tb.values
            if np.nanmax(tb_vals) > 200:
                tb_c = tb_vals - 273.15
            else:
                tb_c = tb_vals

            # Plot as grayscale (IR style)
            bt = ax.contourf(x, y, tb_c, levels=np.linspace(np.nanmin(tb_c), np.nanmax(tb_c), 64), cmap=plt.cm.gray_r, alpha=0.6, zorder=1.05, transform=transform)
            safe_clip(bt)

            # Add grayscale colorbar below the map, matching `ga_wx.py` placement
            try:
                cax_bt = fig.add_axes([0.08, 0.055, 0.84, 0.015])
                sm = plt.cm.ScalarMappable(cmap=plt.cm.gray_r, norm=Normalize(vmin=np.nanmin(tb_c), vmax=np.nanmax(tb_c)))
                sm.set_array([])
                cbar_bt = fig.colorbar(sm, cax=cax_bt, orientation='horizontal', extend='both')
                cbar_bt.set_label('IR Brightness Temp (°C)', fontsize=10, fontweight='bold', color='white', path_effects=text_outline)
                for label in cbar_bt.ax.get_xticklabels():
                    label.set_path_effects(text_outline)
                    label.set_color('white')
                    label.set_fontsize(9)
            except Exception:
                # Fallback: vertical bar aligned to map if horizontal placement fails
                pos = ax.get_position()
                cbar_width = 0.02
                cbar_pad = 0.01
                right_x = pos.x1 + cbar_pad + cbar_width + 0.01
                cax_bt = fig.add_axes([right_x, pos.y0, cbar_width, pos.height])
                cbar_bt = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.gray_r, norm=Normalize(vmin=np.nanmin(tb_c), vmax=np.nanmax(tb_c))), cax=cax_bt, orientation='vertical', extend='both')
                cbar_bt.set_label('IR Brightness Temp (°C)', fontsize=10, fontweight='bold', color='white', path_effects=text_outline)
                for l in cbar_bt.ax.get_yticklabels(): l.set_path_effects(text_outline); l.set_color('white'); l.set_fontsize(9)
        else:
            logging.info("No 'tb' key present in nam_overlays; skipping TB overlay.")

    # --- TEMPERATURE ---
    temp_data = forecast_data_dict.get('temp')
    if temp_data is not None:
        temp_data = temp_data.squeeze() 
        color_stops = [(-100, "#F0F0F0"), (-80, "#E0B0FF"), (-60, "#DA70D6"), (-40, "#800080"), (-20, "#4B0082"), (-10, "#00008B"), (-5, "#0000FF"), (0, "#0000FF"), (5, "#1E90FF"), (10, "#00BFFF"), (15, "#87CEFA"), (20, "#B0E0E6"), (25, "#AFEEEE"), (30, "#00FFFF"), (35, "#00FF00"), (40, "#32CD32"), (45, "#7CFC00"), (50, "#ADFF2F"), (55, "#FFFF00"), (60, "#FFD700"), (65, "#FFA500"), (70, "#FF8C00"), (75, "#FF4500"), (80, "#FF0000"), (85, "#DC143C"), (90, "#B22222"), (95, "#8B0000"), (100, "#A52A2A"), (105, "#8B4513"), (110, "#A0522D"), (115, "#CD853F"), (120, "#D2B48C"), (125, "#E0E0E0"), (130, "#FFFFFF")]
        stop_vals = [x[0] for x in color_stops]
        cmap = mcolors.LinearSegmentedColormap.from_list('custom', [((v - min(stop_vals)) / (max(stop_vals) - min(stop_vals)), c) for v, c in color_stops])
        norm = Normalize(min(stop_vals), max(stop_vals))
        
        t_min, t_max = int(np.nanmin(temp_data)) - 2, int(np.nanmax(temp_data)) + 2
        cf = ax.contourf(temp_data.x, temp_data.y, temp_data, levels=np.arange(t_min, t_max, 2), cmap=cmap, norm=norm, alpha=0.6, zorder=1.5, transform=temp_data.metpy.cartopy_crs)
        safe_clip(cf)
        
        # --- COLORBARS (Aligned to map Axes bounding box like `ga_wx.py`) ---
        pos = ax.get_position()
        cbar_width = 0.02
        cbar_pad = 0.01
        cbar_left_shift = 0.02

        # 1. RADAR (Left Gutter) - align to exact map height (match ga_wx.py)
        left_x = pos.x0 - (cbar_width + cbar_pad + cbar_left_shift)
        cax_radar = fig.add_axes([left_x, pos.y0, cbar_width, pos.height])
        cbar_radar = fig.colorbar(plt.cm.ScalarMappable(cmap=ctables.registry.get_colortable('NWSReflectivity'), norm=Normalize(vmin=5, vmax=80)), cax=cax_radar, orientation='vertical', extend='max')
        cax_radar.yaxis.set_ticks_position('left'); cax_radar.yaxis.set_label_position('left')
        cbar_radar.set_label('Reflectivity (dBZ)', fontsize=10, fontweight='bold', color='white', path_effects=text_outline)
        for l in cbar_radar.ax.get_yticklabels(): l.set_path_effects(text_outline); l.set_color('white'); l.set_fontsize(9)

        # 2. TEMP (Right Gutter) - align to exact map height (match ga_wx.py)
        right_x = pos.x1 + cbar_pad
        cax_temp = fig.add_axes([right_x, pos.y0, cbar_width, pos.height])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax_temp, extend='both', orientation='vertical')
        cb.set_label('Temperature (°F)', color='white', fontweight='bold', fontsize=12, path_effects=text_outline)
        cax_temp.yaxis.set_tick_params(color='white', labelcolor='white')
        for l in cb.ax.yaxis.get_ticklabels(): l.set_path_effects(text_outline); l.set_color('white'); l.set_fontsize(9)

    # --- MAP LAYERS ---
    ax.add_feature(cfeature.COASTLINE, edgecolor='navy', linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.STATES, edgecolor=STATE_BORDER_COLOR, linewidth=1.5, zorder=4)
    try: ax.add_feature(USCOUNTIES.with_scale('20m'), edgecolor=COUNTY_BORDER_COLOR, linewidth=0.8, linestyle='-.', zorder=2.5, alpha=0.8).set_clip_path(clip_patch)
    except: pass
    try:
        roads = shpreader.Reader(shpreader.natural_earth(resolution='10m', category='cultural', name='roads'))
        for r in roads.records():
            if r.attributes['type'] in ['Major Highway', 'Secondary Highway']:
                c = MAJOR_ROAD_COLOR if r.attributes['type'] == 'Major Highway' else SECONDARY_ROAD_COLOR
                ax.add_geometries([r.geometry], ccrs.PlateCarree(), facecolor='none', edgecolor=c, linewidth=0.5, zorder=3).set_clip_path(clip_patch)
    except: pass

    try: draw_interstate_shields(ax)
    except: pass

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=1.0, color="#3932A0", alpha=0.8, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'path_effects': text_outline, 'fontweight': 'bold', 'color': 'white'}
    gl.ylabel_style = {'size': 10, 'path_effects': text_outline, 'fontweight': 'bold', 'color': 'white'}

    # Fronts & Pressure
    try: fetch_and_plot_wpc_pressure_centers(ax)
    except: pass
    try: fetch_and_plot_wpc_fronts(ax)
    except: pass
    
    # --- STATION PLOTTING ---
    _collision_items = []
    
    # Text Styles
    style_temp = dict(color='black', fontsize=12, fontweight='bold', path_effects=[pe.withStroke(linewidth=2.5, foreground="#FF0000")])
    style_pop = dict(color='black', fontsize=10, fontweight='bold', path_effects=[pe.withStroke(linewidth=2, foreground="#6495ED")]) # Blue for Rain
    style_cond = dict(color='black', fontsize=10, fontweight='bold', path_effects=[pe.withStroke(linewidth=1.5, foreground="#AAAAAA")])
    style_wind = dict(color='black', fontsize=10, fontweight='bold', path_effects=[pe.withStroke(linewidth=2, foreground="goldenrod")])

    if temp_data is not None:
        data_crs = temp_data.metpy.cartopy_crs
        
        for city, (lat, lon) in CITIES.items():
            if city not in CUSTOM_LAYOUTS: continue
            if not (MAP_EXTENT[0] < lon < MAP_EXTENT[1] and MAP_EXTENT[2] < lat < MAP_EXTENT[3]): continue
            
            # Data extraction
            pt = data_crs.transform_point(lon, lat, ccrs.PlateCarree())
            x_idx = (np.abs(temp_data.x - pt[0])).argmin()
            y_idx = (np.abs(temp_data.y - pt[1])).argmin()
            t = temp_data.isel(y=y_idx, x=x_idx).item()
            if np.isnan(t): continue

            def get_val(key, yi, xi, default=0):
                if forecast_data_dict.get(key) is not None:
                    try: return forecast_data_dict[key].isel(y=yi, x=xi).item()
                    except: return default
                return default

            pop = get_val('pop', y_idx, x_idx, default=0)
            w_spd = get_val('wind_spd', y_idx, x_idx, default=0)
            w_dir = get_val('wind_dir', y_idx, x_idx, default=0)
            sky = get_val('sky', y_idx, x_idx, default=0)

            layout = CUSTOM_LAYOUTS[city]
            bx, by = layout['pos']; ha = layout['ha']
            
            leader, = ax.plot([lon, bx], [lat, by], 'k:', lw=1.5, zorder=9)
            if clip_patch: leader.set_clip_path(clip_patch)
            ax.scatter(lon, lat, facecolor='white', edgecolor='black', s=80, lw=1.0, zorder=10, transform=ccrs.PlateCarree())
            
            u_wind, v_wind = wind_components(w_spd * units.knots, w_dir * units.degrees)
            ax.barbs(lon, lat, u_wind.m, v_wind.m, length=7, pivot='middle', zorder=20, color='black', linewidth=2.5, transform=ccrs.PlateCarree())
            
            # --- 1. Weather Condition Logic ---
            is_day_map = "High" in title
            icon_key = get_forecast_icon_key(sky, pop, t, is_day=is_day_map)
            icon_img = WEATHER_ICONS.get(icon_key)
            
            # Format Condition Text
            cond_str = icon_key.replace(" (Sun)", "").replace(" (Moon)", "")
            cond_str = cond_str.replace("Mostly", "Msly").replace("Partly", "Ptly").replace("Cloudy", "Cldy")
            
            # --- 2. City Badge ---
            name_y = by + 0.12
            city_txt = ax.text(bx, name_y, city.upper(), transform=ccrs.PlateCarree(), fontsize=FONT_SIZE_CITY,
                               ha=ha, va='bottom', fontweight='bold', color='white', zorder=16, 
                               bbox=CITY_BADGE_STYLE)

            # --- 3. Build the 2x2 Grid ---
            
            # Top-Left: Temperature
            t_area = TextArea(f" {t:.0f}° ", textprops=style_temp)
            t_box = PaddedBox(t_area, pad=0, draw_frame=True, patch_attrs=DATA_BOX_STYLE)

            # Top-Right: Condition Text + Icon
            cond_area = TextArea(f" {cond_str} ", textprops=style_cond)
            cond_children = [cond_area]
            if icon_img is not None:
                cond_children.append(OffsetImage(icon_img, zoom=0.15))
            
            cond_packer = HPacker(children=cond_children, align="center", pad=0, sep=2)
            cond_box = PaddedBox(cond_packer, pad=0, draw_frame=True, patch_attrs=DATA_BOX_STYLE)

            # Bottom-Left: PoP (Precip Chance)
            pop_area = TextArea(f" P:{pop:.0f}% ", textprops=style_pop)
            pop_box = PaddedBox(pop_area, pad=0, draw_frame=True, patch_attrs=DATA_BOX_STYLE)
            
            # Bottom-Right: Wind (safe handling for NaN wind speeds)
            wd_label = wind_dir_to_str(w_dir)
            try:
                if pd.isna(w_spd):
                    wind_str = f" {wd_label} -- "
                else:
                    wind_str = f" {wd_label} {int(w_spd)} "
            except Exception:
                wind_str = f" {wd_label} -- "
            wind_area = TextArea(wind_str, textprops=style_wind)
            wind_box = PaddedBox(wind_area, pad=0, draw_frame=True, patch_attrs=DATA_BOX_STYLE)

            # --- 4. Assemble Grid ---
            grid_packer = HPacker(children=[
                VPacker(children=[t_box, pop_box], align=ha, pad=0, sep=0),
                VPacker(children=[cond_box, wind_box], align=ha, pad=0, sep=0)
            ], align="top", pad=0, sep=0)
            
            grid_ab = AnnotationBbox(grid_packer, (bx, by), xybox=(0, 0), xycoords='data', boxcoords='offset points', frameon=False, pad=0.0, zorder=12)
            grid_ab.leader_line = leader
            ax.add_artist(grid_ab)

            _collision_items.append({'item': city_txt, 'anchor_point': (bx, name_y)})
            _collision_items.append({'item': grid_ab, 'anchor_point': (bx, by)})

    _resolve_text_collisions(fig, ax, _collision_items)
    
    if clip_patch:
        for it in _collision_items:
            try: it['item'].set_clip_path(clip_patch)
            except: pass

    # Data Source Badge
    ax.text(0.01, 0.01, f"Data Sources:\n • NDFD Forecast (Temps)\n • {model_src} (Forecast Radar/Isobars)",
            transform=ax.transAxes, fontsize=10, fontweight='bold', color='navy',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'), zorder=20)
            
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, facecolor=FIG_BG_COLOR)
    buf.seek(0)
    plt.close(fig)
    return buf

# --- EXECUTION BLOCK (Must be unindented to run!) ---
if __name__ == "__main__":
    try:
        logging.info("--- Starting Regional Forecast Map Gen (V65 - Broadcast Theme) ---")
        highs_data, lows_data, day_time, night_time = fetch_ndfd_data()
        
        # --- UPDATED DATE FORMATTER ---
        def to_et_str(dt_utc):
            """Converts UTC datetime to 'FRI NOV 28 4 PM ET' format."""
            if dt_utc is None: return "N/A"
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            
            # Convert to Eastern Time
            et_dt = dt_utc.astimezone(timezone(timedelta(hours=-5)))
            
            # Format: Day Month Date (e.g., "FRI NOV 28")
            date_part = et_dt.strftime('%a %b %d').upper()
            
            # Format: Time (e.g., "4 PM ET"), removing leading zero
            time_part = et_dt.strftime('%I %p ET').lstrip('0')
            
            return f"{date_part} {time_part}"
        # ------------------------------

        radar_day_str = to_et_str(day_time)
        radar_night_str = to_et_str(night_time)
        
        nam_day, nam_night = None, None
        if highs_data and highs_data.get('temp') is not None:
            # ... (rest of the fetching logic remains the same) ...
            with concurrent.futures.ThreadPoolExecutor() as executor:
                f_day = executor.submit(fetch_model_data, day_time)
                f_night = executor.submit(fetch_model_data, night_time)
                nam_day = f_day.result()
                nam_night = f_night.result()
            
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            try:
                SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
                PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
            except: PROJECT_ROOT = os.getcwd()
            OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
            if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

            if highs_data.get('temp') is not None:
                # The title will now read: "VALID: FRI NOV 28 4 PM ET | SOURCE: ..."
                buf = plot_forecast_map("Today's Highs", highs_data, nam_day, radar_day_str, "highs.png")
                with open(os.path.join(OUTPUT_DIR, f"ga_forecast_highs_{ts}.png"), 'wb') as f: f.write(buf.getbuffer())
                with open(os.path.join(PROJECT_ROOT, "ga_forecast_highs.png"), 'wb') as f: f.write(buf.getbuffer())
                logging.info("Highs map saved.")
                
            if lows_data.get('temp') is not None:
                buf = plot_forecast_map("Tonight's Lows", lows_data, nam_night, radar_night_str, "lows.png")
                with open(os.path.join(OUTPUT_DIR, f"ga_forecast_lows_{ts}.png"), 'wb') as f: f.write(buf.getbuffer())
                with open(os.path.join(PROJECT_ROOT, "ga_forecast_lows.png"), 'wb') as f: f.write(buf.getbuffer())
                logging.info("Lows map saved.")
        else:
            logging.error("NDFD Data Fetch failed. Maps not generated.")
        logging.info("Done.")

    except KeyboardInterrupt:
        logging.info("Map generation cancelled by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"--- A CRITICAL UNEXPECTED ERROR OCCURRED ---")
        print(f"{e}")