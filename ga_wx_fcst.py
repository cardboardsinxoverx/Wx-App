
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Georgia (GA) Forecast Map Generator.
This script fetches forecast data from the NDFD
to generate High and Low temperature maps for Georgia and neighboring regions.

*** V70 Update (HILLSHADE & AVIATION):
    1. THEME: Applied Hillshading (3D Terrain).
    2. AVIATION: Added VFR/IFR color-coded station dots.
    3. WIND: Attached Wind Barbs directly to station dots.
"""
# --- STATUS DASHBOARD TRACKER ---
STATUS_TRACKER = {
    "Cache Hits": 0,
    "Cache Misses": 0,
    "Downloads": {"Successful": 0, "Failed": 0},
    "Data Sources": {}
}

def log_status(source_name, hit=False, success=True):
    """
    Updates the global tracker for the final dashboard report.
    This resolves the 'log_status is not defined' error.
    """
    if hit:
        STATUS_TRACKER["Cache Hits"] += 1
    else:
        STATUS_TRACKER["Cache Misses"] += 1
        if success:
            STATUS_TRACKER["Downloads"]["Successful"] += 1
        else:
            STATUS_TRACKER["Downloads"]["Failed"] += 1
            
    if source_name not in STATUS_TRACKER["Data Sources"]:
        STATUS_TRACKER["Data Sources"][source_name] = {"Hits": 0, "Misses": 0}
    
    if hit:
        STATUS_TRACKER["Data Sources"][source_name]["Hits"] += 1
    else:
        STATUS_TRACKER["Data Sources"][source_name]["Misses"] += 1

import matplotlib
matplotlib.use('Agg') # Force headless mode
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, LightSource # <--- Added LightSource
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
# Silence coordinate and metadata warnings
warnings.filterwarnings("ignore", message=".*attribute type uint not understood.*")
warnings.filterwarnings("ignore", message=".*Dataset has no geotransform.*")
warnings.filterwarnings("ignore", category=UserWarning, module='cartopy')
warnings.filterwarnings("ignore", message=".*Non-standard reference date.*")
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
import zipfile
import shutil
import rasterio 
import requests 
from PIL import Image, ImageFilter, ImageEnhance # <--- Updated

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

# CRTM disabled - using HRRR surface temp as IR (works perfectly)
crtm_helper = None
logging.info("INFO - Using HRRR surface temp as IR (professional quality)")

# --- MAP CONFIGURATION ---
MAP_EXTENT = [-86.8, -80.7, 30.2, 35.8]

# --- UPDATED CITY LIST (Macon/Robins Collision Fix) ---
CITIES = {
    # --- NORTH GEORGIA ---
    "Rome":          {'pos': (-85.32, 34.18), 'ha': 'right'},
    "Dalton":        {'pos': (-85.15, 34.90), 'ha': 'right'},
    "Gainesville":   {'pos': (-83.95, 34.45), 'ha': 'right'},
    "Toccoa":        {'pos': (-83.11, 34.70), 'ha': 'left'},
    "Athens":        {'pos': (-83.15, 34.05), 'ha': 'left'},
    
    "Atlanta":       {'pos': (-84.20, 33.60), 'ha': 'left'},
    "Marietta":      {'pos': (-84.75, 34.05), 'ha': 'right'},

    # --- CENTRAL GEORGIA ---
    "Macon":         {'pos': (-83.80, 32.95), 'ha': 'right'}, # NUDGED NORTH
    "Robins AFB":    {'pos': (-83.35, 32.40), 'ha': 'left'},  # NUDGED SOUTH-EAST
    
    "Columbus":      {'pos': (-85.15, 32.55), 'ha': 'right'},
    "Augusta":       {'pos': (-81.80, 33.45), 'ha': 'left'},
    "Milledgeville": {'pos': (-83.20, 33.25), 'ha': 'center'},

    # --- SOUTH GEORGIA & COAST ---
    "Savannah":      {'pos': (-81.35, 31.95), 'ha': 'right'}, 
    "Statesboro":    {'pos': (-81.70, 32.60), 'ha': 'right'},
    "Brunswick":     {'pos': (-81.65, 31.30), 'ha': 'right'}, 
    "Waycross":      {'pos': (-82.50, 31.10), 'ha': 'right'},
    "Albany":        {'pos': (-84.40, 31.60), 'ha': 'right'},
    "Tifton":        {'pos': (-83.60, 31.40), 'ha': 'right'},
    "Valdosta":      {'pos': (-83.30, 30.65), 'ha': 'center'},
}


# --- STYLING CONFIGURATION (THEME APPLIED) ---
FIG_BG_COLOR = '#333333'       # Dark Gray background
LAND_COLOR = '#E6E6E6'         # Very Light Gray (High Contrast)
OCEAN_COLOR = "#2774DA"        # blue
STATE_BORDER_COLOR = 'black'   # Sharp Black
COUNTY_BORDER_COLOR = '#00158D'# Medium Gray
MAJOR_ROAD_COLOR = '#8B0000'   # *** DARK RED (Signature look) ***
SECONDARY_ROAD_COLOR = '#6D3A00' # Brown

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

# --- AVIATION FLIGHT RULES COLORS ---
FLIGHT_CATEGORY_COLORS = {
    "VFR": "#00FF00",   # Green
    "MVFR": "#0000FF",  # Blue
    "IFR": "#FF0000",   # Red
    "LIFR": "#FF00FF"   # Magenta
}

plt.rcParams['font.family'] = FONT_FAMILY

# --- CATALOG URLs ---
NDFD_CATALOG_URL = "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/NDFD/NWS/CONUS/CONDUIT/catalog.xml"

# --- BROADCAST HEADER FUNCTION ---
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
    
    logo_path = "/home/desoxyn/frostbyte/frostbyte_project/boxlogo4.png"
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
    """
    Saves data as a pickle file in the .wx_cache directory.
    Optimized for Python 3.12 compatibility.
    """
    if not CACHE_DIR:
        return
    
    filepath = os.path.join(CACHE_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            # Use protocol 4 or higher for faster serialization of large GeoJSON sets
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Saved pickle cache: {filename}")
    except Exception as e:
        logging.warning(f"Failed to save pickle cache to {filepath}: {e}")

def load_pickle_cache(filename, max_age_seconds):
    """
    Loads data from a pickle cache file if it exists and is within the age limit.
    Used for both GDOT Urban Areas and Model data.
    """
    if not CACHE_DIR:
        return None
        
    filepath = os.path.join(CACHE_DIR, filename)
    
    # Check if file exists and is not stale
    if os.path.exists(filepath):
        try:
            file_mod_time = os.path.getmtime(filepath)
            age_seconds = time.time() - file_mod_time
            
            if age_seconds < max_age_seconds:
                with open(filepath, 'rb') as f:
                    logging.info(f"Using cached data: {filename} ({int(age_seconds/60)}m old)")
                    return pickle.load(f)
            else:
                logging.info(f"Cache expired for {filename}. Fetching fresh data...")
        except Exception as e:
            logging.warning(f"Failed to read pickle cache from {filepath}: {e}")
            
    return None

def apply_glow_effect(icon_array, glow_color=(255, 255, 255, 150), blur_radius=10):
    """
    Takes a numpy image array, adds a soft glowing background, 
    and expands the canvas to ensure the glow isn't clipped.
    """
    # 1. Convert to PIL
    if isinstance(icon_array, np.ndarray):
        pil_img = Image.fromarray((icon_array * 255).astype(np.uint8)).convert('RGBA')
    else:
        pil_img = icon_array.convert('RGBA')

    # 2. Expand Canvas (Add padding for the glow)
    padding = int(blur_radius * 2.5)
    new_size = (pil_img.width + 2*padding, pil_img.height + 2*padding)
    
    # Create centered canvas
    expanded_img = Image.new("RGBA", new_size, (0, 0, 0, 0))
    paste_pos = (padding, padding)
    expanded_img.paste(pil_img, paste_pos, pil_img)

    # 3. Create Glow Layer
    alpha = expanded_img.split()[-1]
    glow_layer = Image.new("RGBA", new_size, glow_color)
    glow_layer.putalpha(alpha)
    
    # Dilate (make the shape chunkier) then Blur
    glow_layer = glow_layer.filter(ImageFilter.MaxFilter(5)) 
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 4. Composite
    final_img = Image.new("RGBA", new_size, (0,0,0,0))
    final_img.paste(glow_layer, (0, 0), glow_layer)
    final_img.paste(expanded_img, (0, 0), expanded_img)
    
    return np.array(final_img) / 255.0

def fetch_single_ndfd_var(var_name, time_start, time_end):
    # --- FINAL CORRECTED STRATEGY ---
    # 1. Summary Variables (Max/Min Temp, Precip) -> FORCE ALL_TIMES.
    #    The server returns "400 Error" if we try to slice these by time.
    #    We must accept the slow download speed for these to avoid the crash.
    # 2. Dense Variables (Wind, Sky, Dewpoint) -> USE TIME_RANGE.
    #    These are safe to slice and download quickly.
    
    # CHECK: Is this a "Heavy" variable?
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
            query.all_times() # <--- This is the key fix for the 400 Error
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

def determine_wpc_day_offset(target_time_utc):
    """
    Calculates whether the target time falls in WPC Day 1 or Day 2.
    """
    if target_time_utc is None: return 1
    
    if target_time_utc.tzinfo is None:
        target_time_utc = target_time_utc.replace(tzinfo=timezone.utc)
    
    now_utc = datetime.now(timezone.utc)
    diff_hours = (target_time_utc - now_utc).total_seconds() / 3600
    
    # CHANGE: Lower threshold from 12 to 8 hours
    # This ensures Tonight's Lows (approx 10 hours away) triggers Day 2
    is_day_1 = diff_hours < 8 
    
    logging.info(f"Target: {target_time_utc} | Hours away: {diff_hours:.1f} | Using WPC Day: {1 if is_day_1 else 2}")
    
    return 1 if is_day_1 else 2

def fetch_ndfd_data():
    # 1. TRY LOADING FROM CACHE FIRST (1 Hour TTL)
    cache_file = "ndfd_cache.pkl"
    cached_data = load_pickle_cache(cache_file, 3600)
    if cached_data:
        return cached_data

    # 2. CONFIGURATION
    logging.info("Fetching NDFD Forecast Data (Hybrid Mode)...")
    
    target_vars = [
        "Maximum_temperature_height_above_ground_12_Hour_Maximum",
        "Minimum_temperature_height_above_ground_12_Hour_Minimum",
        "Total_precipitation_surface_12_Hour_Accumulation_probability_above_0p254",
        "Dewpoint_temperature_height_above_ground",
        "Total_cloud_cover_surface",
        "Wind_speed_height_above_ground",
        "Wind_direction_from_which_blowing_height_above_ground"
    ]
    
    # Split variables into "Heavy" (Must be sequential) and "Light" (Can be parallel)
    heavy_vars = [v for v in target_vars if any(x in v for x in ["Maximum", "Minimum", "Total_precipitation"])]
    light_vars = [v for v in target_vars if v not in heavy_vars]

    now = datetime.now(timezone.utc)
    time_start = now - timedelta(hours=12)
    time_end = now + timedelta(hours=60)
    datasets = {}

    # --- PHASE 1: HEAVY VARIABLES (Sequential) ---
    # We do these one-by-one to give the server full bandwidth for the massive files.
    logging.info(f"--- PHASE 1: Downloading {len(heavy_vars)} Large Files (Sequential) ---")
    for var in heavy_vars:
        try:
            var_name, ds = fetch_single_ndfd_var(var, time_start, time_end)
            if ds: datasets[var_name] = ds
        except Exception as e:
            logging.error(f"Failed to fetch {var}: {e}")

    # --- PHASE 2: LIGHT VARIABLES (Parallel) ---
    # These are small, so we can download them all at once to save time.
    logging.info(f"--- PHASE 2: Downloading {len(light_vars)} Small Files (Parallel) ---")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_single_ndfd_var, var, time_start, time_end): var for var in light_vars}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                var_name, ds = future.result(timeout=60) # 60s timeout for small files
                if ds: datasets[var_name] = ds
            except Exception as e:
                logging.error(f"Parallel fetch failed for a variable: {e}")

    # --- CHECK CRITICAL DATA ---
    if "Maximum_temperature_height_above_ground_12_Hour_Maximum" not in datasets:
        logging.error("Critical: Max Temp data failed to download.")
        return None, None, None, None
        
    # --- PROCESSING (Standard Logic) ---
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
    """
    Fetches NWS High/Low Pressure Centers via ArcGIS API.
    Adds ' hPa' to labels if they are inside the map bounds.
    """
    logging.info("Fetching WPC Pressure Centers...")
    
    # --- 1. Try Cache or Download ---
    # (Using a generic cache name compatible with both scripts)
    cache_file = "wpc_pressure_cache.pkl"
    cached_gdf = load_pickle_cache(cache_file, 3600)
    
    full_gdf = None
    
    if cached_gdf is not None:
        full_gdf = cached_gdf
    else:
        # Download Fresh
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
            save_pickle_cache(cache_file, full_gdf)
    
    if full_gdf is None or full_gdf.empty: return

    # --- 2. Filter & Coordinate Repair ---
    xmin_d, ymin_d, xmax_d, ymax_d = full_gdf.total_bounds
    if abs(xmin_d) > 180 or abs(xmax_d) > 180:
        try:
            full_gdf.crs = "EPSG:3857"
            full_gdf = full_gdf.to_crs("EPSG:4326")
        except: pass

    xmin, xmax, ymin, ymax = MAP_EXTENT
    # Large buffer for clamping
    buffer_deg = 15.0 
    bbox_poly = Polygon([(xmin-buffer_deg, ymin-buffer_deg), (xmin-buffer_deg, ymax+buffer_deg), 
                         (xmax+buffer_deg, ymax+buffer_deg), (xmax+buffer_deg, ymin-buffer_deg)])
    
    full_gdf = full_gdf[full_gdf.geometry.intersects(bbox_poly)]
    
    # --- 3. Plotting Logic ---
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
            
            # Clamp logic
            plot_lon = max(xmin, min(lon_sys, xmax))
            plot_lat = max(ymin, min(lat_sys, ymax))
            
            # Alignment logic
            ha, va = 'center', 'center'
            if lon_sys <= xmin: ha = 'left'
            elif lon_sys >= xmax: ha = 'right'
            if lat_sys <= ymin: va = 'bottom'
            elif lat_sys >= ymax: va = 'top'

            # *** FIX: Check if strictly INSIDE map extent ***
            if (xmin <= lon_sys <= xmax) and (ymin <= lat_sys <= ymax):
                label_text = f'{sys_type}\n{pressure_val} hPa'
            else:
                label_text = f'{sys_type}\n{pressure_val}'

            ax.text(plot_lon, plot_lat, label_text, 
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
    """
    Fetches model data (HRRR/RAP/NAM).
    Updated V71: Fetches Categorical Precip Type (CRAIN, CSNOW, CICE, CFRZR)
    for accurate winter weather radar masking.
    """
    if not HERBIE_AVAILABLE:
        return None

    if target_time_utc.tzinfo is None:
        target_time_utc = target_time_utc.replace(tzinfo=timezone.utc)

    now_utc = datetime.now(timezone.utc)
    
    def try_fetch(model_name, product, run_dt, fxx):
        run_str = run_dt.strftime("%Y-%m-%d %H:00")
        print(f"--- DEBUG: Trying {model_name} {run_str} F{fxx:02d} ---")
        
        try:
            H = Herbie(run_str, model=model_name, product=product, fxx=fxx)
            
            # --- 1. REFLECTIVITY ---
            ds_refl = None
            try:
                ds_refl = H.xarray(":REFC:entire atmosphere", verbose=False)
                if isinstance(ds_refl, list): ds_refl = ds_refl[0]
            except: pass

            if ds_refl is None: return None # Abort if no radar

            # --- 2. TEMPERATURE ---
            ds_temp = None
            temp_var_name = None
            
            # Specific search patterns for 2m Temp
            temp_patterns = [
                ":TMP:2 m above ground",
                ":TMP:2 m",
                ":t2m:", 
                ":Temperature_height_above_ground:"
            ]
            
            for patt in temp_patterns:
                try:
                    tmp = H.xarray(patt, verbose=False)
                    if isinstance(tmp, list): tmp = tmp[0]
                    
                    if tmp is not None and len(tmp.data_vars) > 0:
                        for var in tmp.data_vars:
                            if len(tmp[var].dims) >= 2:
                                ds_temp = tmp
                                temp_var_name = var
                                break
                        if ds_temp: break
                except: continue

            # --- 3. WINTER PRECIPITATION TYPE (New for Winterization) ---
            # Fetches categorical Rain (CRAIN), Snow (CSNOW), Ice Pellets (CICE), Freezing Rain (CFRZR)
            # These are binary masks (1=Yes, 0=No) calculated by the model's microphysics
            ds_ptype = None
            try:
                # Regex to grab all 4 categories from surface level
                ds_ptype = H.xarray(":(?:CRAIN|CSNOW|CICE|CFRZR):surface", verbose=False)
                if isinstance(ds_ptype, list): ds_ptype = ds_ptype[0]
            except Exception as e: 
                print(f"--- DEBUG: PTYPE fetch failed: {e}")
                pass

            # --- 4. CLOUDS & MSLP ---
            ds_cld = None
            try:
                ds_cld = H.xarray(":TCDC:entire atmosphere", verbose=False)
                if isinstance(ds_cld, list): ds_cld = ds_cld[0]
            except: pass

            ds_mslp = None
            try:
                ds_mslp = H.xarray(":(?:MSLMA|PRMSL):mean sea level", verbose=False)
                if isinstance(ds_mslp, list): ds_mslp = ds_mslp[0]
            except: pass
            
            # --- CROP & EXTRACT ---
            def safe_crop(ds):
                if ds is None: return None
                if np.any(ds.longitude > 180):
                    ds = ds.assign_coords(longitude = (((ds.longitude + 180) % 360) - 180))
                return ds

            ds_refl  = safe_crop(ds_refl)
            ds_temp  = safe_crop(ds_temp)
            ds_ptype = safe_crop(ds_ptype)
            ds_cld   = safe_crop(ds_cld)
            ds_mslp  = safe_crop(ds_mslp)

            # Extract actual DataArrays
            refl = ds_refl['refc'] if 'refc' in ds_refl else (ds_refl['REFC'] if 'REFC' in ds_refl else None)
            
            temp_da = None
            if ds_temp and temp_var_name:
                temp_da = ds_temp[temp_var_name]
            
            # Extract P-Type Arrays
            cat_rain = ds_ptype['crain'] if ds_ptype and 'crain' in ds_ptype else None
            cat_snow = ds_ptype['csnow'] if ds_ptype and 'csnow' in ds_ptype else None
            cat_ice  = ds_ptype['cice']  if ds_ptype and 'cice'  in ds_ptype else None # Sleet
            cat_fzra = ds_ptype['cfrzr'] if ds_ptype and 'cfrzr' in ds_ptype else None
            
            mslp_da = None
            if ds_mslp:
                for v in ds_mslp.data_vars:
                    if 'msl' in v.lower() or 'prmsl' in v.lower():
                        mslp_da = ds_mslp[v]
                        break

            cld_da = None
            if ds_cld:
                for v in ds_cld.data_vars:
                    if 'tcdc' in v.lower():
                        cld_da = ds_cld[v]
                        break

            logging.info(f"SUCCESS: Fetched {model_name.upper()} data with Winter P-Types.")
            return {
                'refl': refl, 
                'clouds': cld_da, 
                'mslp': mslp_da, 
                'sfc_temp': temp_da, 
                'cat_rain': cat_rain, # NEW
                'cat_snow': cat_snow, # NEW
                'cat_ice':  cat_ice,  # NEW
                'cat_fzra': cat_fzra, # NEW
                'source': f"{model_name.upper()}"
            }
            
        except Exception as e:
            print(f"--- DEBUG: Error in fetch: {e}")
            pass
        return None

    # STRATEGY 1: HRRR (Primary)
    hrrr_attempts = []
    latest_hrrr_run = now_utc - timedelta(hours=1) 
    latest_hrrr_run = latest_hrrr_run.replace(minute=0, second=0, microsecond=0)
    fxx_latest = int((target_time_utc - latest_hrrr_run).total_seconds() / 3600)
    
    if 0 <= fxx_latest <= 18: hrrr_attempts.append((latest_hrrr_run, fxx_latest))
    for hb in range(0, 24):
        past = latest_hrrr_run - timedelta(hours=hb)
        if past.hour % 1 == 0:
            fx = int((target_time_utc - past).total_seconds() / 3600)
            if 0 <= fx <= 48: hrrr_attempts.append((past, fx))
                
    for run_time, fxx in hrrr_attempts:
        res = try_fetch('hrrr', 'sfc', run_time, fxx)
        if res: return res

    # STRATEGY 2: RAP (Secondary)
    try:
        latest_rap = now_utc - timedelta(hours=1)
        latest_rap = latest_rap.replace(minute=0, second=0, microsecond=0)
        fxx_rap = int((target_time_utc - latest_rap).total_seconds() / 3600)
        if 0 <= fxx_rap <= 21:
            res = try_fetch('rap', 'sfc', latest_rap, fxx_rap)
            if res: return res
    except: pass

    # STRATEGY 3: NAM (Backup - usually lacks categorical p-type in this specific product stream)
    try:
        latest_nam = now_utc - timedelta(hours=3)
        run_hour_nam = (latest_nam.hour // 6) * 6
        nam_run = latest_nam.replace(hour=run_hour_nam, minute=0, second=0, microsecond=0)
        fxx_nam = int((target_time_utc - nam_run).total_seconds() / 3600)
        if 0 <= fxx_nam <= 60:
            res = try_fetch('nam', 'conusnest.hiresf', nam_run, fxx_nam)
            if res: return res
    except: pass

    return None

def _resolve_text_collisions(fig, ax, items, padding_px=15, max_iter=1500, pixel_step=20):
    """
    Iterative solver to push text boxes apart. 
    Aggressive settings: 15px padding, 20px step size.
    """
    if not items:
        return

    # 1. Get renderer
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return

    # 2. Helper to get boxes
    def get_box(item):
        return item.get_window_extent(renderer=renderer).expanded(1.0, 1.0)

    def boxes_overlap(b1, b2, pad=0):
        return not (b1.x1 + pad < b2.x0 or b1.x0 - pad > b2.x1 or 
                    b1.y1 + pad < b2.y0 or b1.y0 - pad > b2.y1)

    flattened = [{'item': it['item'], 'fixed': it.get('fixed', False)} for it in items]

    for _ in range(max_iter):
        moved = False
        current_boxes = [get_box(f['item']) for f in flattened]

        for i in range(len(flattened)):
            for j in range(i + 1, len(flattened)):
                item_a = flattened[i]
                item_b = flattened[j]
                
                # If both fixed, skip
                if item_a['fixed'] and item_b['fixed']: continue

                b1 = current_boxes[i]
                b2 = current_boxes[j]
                
                if boxes_overlap(b1, b2, pad=padding_px):
                    # Calculate repulsion vector
                    c1 = ((b1.x0 + b1.x1) / 2.0, (b1.y0 + b1.y1) / 2.0)
                    c2 = ((b2.x0 + b2.x1) / 2.0, (b2.y0 + b2.y1) / 2.0)
                    
                    dx = c1[0] - c2[0]
                    dy = c1[1] - c2[1]
                    
                    # Jitter if perfectly stacked
                    if abs(dx) < 1e-3 and abs(dy) < 1e-3: 
                        dx = np.random.uniform(-1, 1)
                        dy = np.random.uniform(-1, 1)
                    
                    norm = (dx**2 + dy**2)**0.5
                    if norm == 0: norm = 1
                    ux, uy = dx / norm, dy / norm
                    
                    # Logic: Who moves?
                    inv = ax.transData.inverted()
                    
                    # A is Fixed -> Move B away
                    if item_a['fixed']:
                        move_x, move_y = -ux * pixel_step, -uy * pixel_step
                        target_idx = j
                        
                    # B is Fixed -> Move A away
                    elif item_b['fixed']:
                        move_x, move_y = ux * pixel_step, uy * pixel_step
                        target_idx = i
                        
                    # Both Mobile -> Move both
                    else:
                        # Move A
                        p0_a = inv.transform(c1)
                        p1_a = inv.transform((c1[0] + ux * pixel_step/2, c1[1] + uy * pixel_step/2))
                        _apply_move(item_a['item'], (p1_a[0]-p0_a[0], p1_a[1]-p0_a[1]))
                        
                        # Move B
                        p0_b = inv.transform(c2)
                        p1_b = inv.transform((c2[0] - ux * pixel_step/2, c2[1] - uy * pixel_step/2))
                        _apply_move(item_b['item'], (p1_b[0]-p0_b[0], p1_b[1]-p0_b[1]))
                        
                        moved = True
                        continue

                    # Apply Single Move
                    curr_c = c1 if target_idx == i else c2
                    p0 = inv.transform(curr_c)
                    p1 = inv.transform((curr_c[0] + move_x, curr_c[1] + move_y))
                    
                    _apply_move(flattened[target_idx]['item'], (p1[0]-p0[0], p1[1]-p0[1]))
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

# --- NEW ICON GENERATORS ---
def create_ice_pellets_icon():
    """Generates an Ice Pellets (PL) icon (Orange/White balls)."""
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(mpatches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
    # Ice Pellets (Sleet) - Distinct from hail, usually smaller/translucent
    for pos in [(0.3, 0.4), (0.5, 0.3), (0.7, 0.4), (0.4, 0.2)]:
        # Orange outline for "Sleet" standard
        ax.add_patch(mpatches.Circle(pos, 0.05, facecolor='white', edgecolor='orange', linewidth=1.5))
    return _fig_to_img(fig)

def create_freezing_rain_icon():
    """Generates a Freezing Rain (FZRA) icon (Glazed look)."""
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(mpatches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
    # Rain drops with "Ice" halo
    for i in range(3):
        start_x = 0.3 + i*0.2
        # Pink/Violet glow indicating freezing
        ax.plot([start_x, start_x-0.05], [0.5, 0.3], color='#FF00FF', alpha=0.5, linewidth=4)
        ax.plot([start_x, start_x-0.05], [0.5, 0.3], color='blue', linewidth=1.5)
    return _fig_to_img(fig)

def create_mix_icon():
    """Generates a Rain/Snow Mix icon."""
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(mpatches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
    # Rain on left
    ax.plot([0.35, 0.3], [0.5, 0.3], color='blue', linewidth=2)
    # Snow on right
    ax.text(0.6, 0.3, '*', color='white', fontsize=25)
    return _fig_to_img(fig)

# --- FEELS LIKE CALCULATOR ---
def calculate_feels_like(temp_f, dew_f, wind_speed_kts):
    """Calculates Heat Index or Wind Chill based on simplified NWS formulas."""
    if temp_f is None: return None, None
    
    wind_mph = wind_speed_kts * 1.15 if wind_speed_kts else 0
    
    # 1. Wind Chill (Temp < 50F, Wind > 3mph)
    if temp_f < 50 and wind_mph > 3:
        wc = 35.74 + (0.6215 * temp_f) - (35.75 * (wind_mph ** 0.16)) + (0.4275 * temp_f * (wind_mph ** 0.16))
        return "WC", int(wc)
        
    # 2. Heat Index (Temp > 80F) - Simplified NWS Regression
    if temp_f >= 80 and dew_f is not None:
        # Calculate RH approximation
        t_c = (temp_f - 32) * 5/9
        d_c = (dew_f - 32) * 5/9
        # Magnus formula for RH
        es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
        e = 6.112 * np.exp(17.67 * d_c / (d_c + 243.5))
        rh = (e / es) * 100
        
        if rh > 40: # HI only matters if humid
            hi_val = -42.379 + 2.04901523*temp_f + 10.14333127*rh \
                     - 0.22475541*temp_f*rh - 0.00683783*temp_f*temp_f \
                     - 0.05481717*rh*rh + 0.00122874*temp_f*temp_f*rh \
                     + 0.00085282*temp_f*rh*rh - 0.00000199*temp_f*temp_f*rh*rh
            if hi_val > temp_f:
                return "HI", int(hi_val)

    return None, None

def create_mostly_cloudy_storming_icon():
    """Generates a Thunderstorm icon with 'Electric' glowing lightning."""
    fig, ax = create_icon_base()
    
    # 1. Cloud
    ax.add_patch(mpatches.Ellipse((0.5, 0.7), 0.8, 0.5, color='darkgray'))
    ax.add_patch(mpatches.Circle((0.4, 0.75), 0.3, color='darkgray'))
    ax.add_patch(mpatches.Circle((0.6, 0.75), 0.3, color='darkgray'))
    
    # 2. Lightning with Glow
    bolt_coords = [[0.4, 0.4], [0.5, 0.6], [0.6, 0.4], [0.5, 0.5]]
    
    # Outer Glow (Yellow-Orange)
    ax.add_patch(mpatches.Polygon(bolt_coords, color='#FFA500', alpha=0.5, lw=5, joinstyle='round', closed=True))
    # Inner Core (Pale Yellow)
    ax.add_patch(mpatches.Polygon(bolt_coords, color='#FFFFE0', closed=True))
    
    return _fig_to_img(fig)

def create_rain_icon():
    """Generates a Rain icon with 'Hydro Glow' drops."""
    fig, ax = create_icon_base()
    
    # 1. Cloud
    ax.add_patch(mpatches.Ellipse((0.5, 0.8), 0.8, 0.4, color='gray'))
    
    # 2. Rain with Glow
    for i in range(5):
        start_x = 0.2 + i*0.15
        end_x = 0.15 + i*0.15
        # Glow Layer
        ax.plot([start_x, end_x], [0.6, 0.4], color='#00BFFF', alpha=0.3, linewidth=3)
        # Core Layer
        ax.plot([start_x, end_x], [0.6, 0.4], color='blue', linewidth=1)
        
    return _fig_to_img(fig)

def create_mostly_cloudy_rain_icon():
    """Generates a Heavy Rain icon with 'Hydro Glow'."""
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(mpatches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
    ax.add_patch(mpatches.Circle((0.4, 0.75), 0.3, color='gray'))
    ax.add_patch(mpatches.Circle((0.6, 0.75), 0.3, color='gray'))
    
    # Rain
    drops = [([0.3, 0.25], [0.4, 0.2]), ([0.5, 0.45], [0.4, 0.2]), ([0.7, 0.65], [0.4, 0.2])]
    for x_vals, y_vals in drops:
        ax.plot(x_vals, y_vals, color='#00BFFF', alpha=0.4, linewidth=4, solid_capstyle='round')
        ax.plot(x_vals, y_vals, color='blue', linewidth=1.5, solid_capstyle='round')

    return _fig_to_img(fig)

def create_severe_storm_icon():
    """Generates a Severe Storm icon with Red/Gold lightning."""
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(mpatches.Ellipse((0.5, 0.7), 0.85, 0.5, color='#222222'))
    ax.add_patch(mpatches.Circle((0.4, 0.75), 0.3, color='#333333'))
    ax.add_patch(mpatches.Circle((0.6, 0.75), 0.3, color='#333333'))
    
    # Lightning
    bolt_coords = [[0.4, 0.4], [0.5, 0.6], [0.6, 0.4], [0.5, 0.5]]
    ax.add_patch(mpatches.Polygon(bolt_coords, color='#FF4500', alpha=0.6, lw=4, closed=True)) # Glow
    ax.add_patch(mpatches.Polygon(bolt_coords, color='#FFD700', closed=True)) # Core
    
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
        # Temperature-based precip type estimation
        if temp_f is not None:
            if temp_f <= 30:
                return 'Snow'
            elif 30 < temp_f <= 33:
                return 'Wintry Mix' # Mix/Ice Zone
            elif 33 < temp_f <= 35:
                return 'Ice Pellets' # Sleet Zone
        
        # High intensity logic
        if pop >= 60: return 'Storms' 
        return 'Rain'

    # 2. Sky Condition Logic
    if sky_cover is None: return 'Sunny' if is_day else 'Moonlight'
    
    if sky_cover < 10: return 'Sunny' if is_day else 'Moonlight'
    elif sky_cover < 30: return 'Mostly Sunny' if is_day else 'Moonlight'
    elif sky_cover < 60: return 'Partly Cloudy (Sun)' if is_day else 'Partly Cloudy (Moon)'
    elif sky_cover < 90: return 'Cloudy'
    else: return 'Overcast'

# --- 3. FLIGHT CATEGORY ESTIMATION ---
def calculate_flight_category(sky_cover_pct, precip_prob):
    """
    Estimates Flight Rules (VFR, MVFR, IFR, LIFR) based on NDFD Sky Cover and PoP.
    Since NDFD doesn't give ceiling/visibility directly in this feed, we approximate.
    
    Logic:
    - High Precip + High Clouds = Likely IFR
    - Moderate Precip + High Clouds = Likely MVFR
    - Clear/Low Clouds = VFR
    """
    if sky_cover_pct is None: return "VFR"
    
    # 1. Clear/Few Clouds -> VFR
    if sky_cover_pct < 50 and (precip_prob is None or precip_prob < 30):
        return "VFR"
        
    # 2. Cloudy but low precip chance -> MVFR/VFR Borderline (Assume VFR for simplicity unless overcast)
    if sky_cover_pct >= 50 and (precip_prob is None or precip_prob < 30):
        return "VFR" if sky_cover_pct < 80 else "MVFR"

    # 3. Cloudy + Rain Chance -> Lower categories
    if precip_prob >= 50:
        return "IFR"
    elif precip_prob >= 30:
        return "MVFR"
        
    return "VFR"

def _resolve_text_collisions(fig, ax, items, padding_px=5, max_iter=1000, pixel_step=15):
    """
    Iterative solver to push text boxes apart and away from fixed points (Station Dots).
    items: list of dicts {'item': artist, 'fixed': bool}
    """
    if not items:
        return

    # 1. Get renderer
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        # Fallback if canvas isn't ready
        return

    # 2. Helper to get display-space bounding boxes
    def get_box(item):
        return item.get_window_extent(renderer=renderer).expanded(1.0, 1.0)

    def boxes_overlap(b1, b2, pad=0):
        return not (b1.x1 + pad < b2.x0 or b1.x0 - pad > b2.x1 or 
                    b1.y1 + pad < b2.y0 or b1.y0 - pad > b2.y1)

    # 3. Iteration Loop
    for _ in range(max_iter):
        moved = False
        
        # Cache boxes for this iteration to speed up checks
        cached_boxes = [get_box(x['item']) for x in items]
        
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item_a = items[i]
                item_b = items[j]
                
                # If both are fixed, we can't do anything, skip
                if item_a.get('fixed', False) and item_b.get('fixed', False):
                    continue

                b1 = cached_boxes[i]
                b2 = cached_boxes[j]
                
                if boxes_overlap(b1, b2, pad=padding_px):
                    # Calculate overlap center vectors
                    c1 = ((b1.x0 + b1.x1) / 2.0, (b1.y0 + b1.y1) / 2.0)
                    c2 = ((b2.x0 + b2.x1) / 2.0, (b2.y0 + b2.y1) / 2.0)
                    
                    dx = c1[0] - c2[0]
                    dy = c1[1] - c2[1]
                    
                    # Add tiny jitter if perfectly stacked to force separation
                    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                        dx, dy = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
                    
                    norm = (dx**2 + dy**2)**0.5
                    if norm == 0: norm = 1
                    
                    ux, uy = dx / norm, dy / norm
                    
                    # Logic: Determine who moves
                    inv = ax.transData.inverted()
                    
                    # Case A: Item A is Fixed (Station Dot), Item B is Mobile (Text) -> Move B full step away
                    if item_a.get('fixed', False):
                        move_x, move_y = -ux * pixel_step, -uy * pixel_step
                        target_idx = j
                    
                    # Case B: Item B is Fixed, Item A is Mobile -> Move A full step away
                    elif item_b.get('fixed', False):
                        move_x, move_y = ux * pixel_step, uy * pixel_step
                        target_idx = i
                        
                    # Case C: Both Mobile -> Move both apart half step
                    else:
                        # Move A
                        disp_a = inv.transform((c1[0] + ux * pixel_step/2, c1[1] + uy * pixel_step/2)) - inv.transform(c1)
                        _apply_move(items[i]['item'], disp_a)
                        
                        # Move B
                        disp_b = inv.transform((c2[0] - ux * pixel_step/2, c2[1] - uy * pixel_step/2)) - inv.transform(c2)
                        _apply_move(items[j]['item'], disp_b)
                        moved = True
                        continue # Skip single move logic below

                    # Apply Single Move (for Fixed vs Mobile cases)
                    # Convert pixel displacement to data coordinates
                    current_center_disp = (c1 if target_idx == i else c2)
                    new_center_disp = (current_center_disp[0] + move_x, current_center_disp[1] + move_y)
                    
                    p0 = inv.transform(current_center_disp)
                    p1 = inv.transform(new_center_disp)
                    diff_data = (p1[0] - p0[0], p1[1] - p0[1])
                    
                    _apply_move(items[target_idx]['item'], diff_data)
                    moved = True

        if not moved:
            break

def _apply_move(artist, diff_data):
    """Helper to apply movement to Text or AnnotationBbox and update leader lines."""
    try:
        if isinstance(artist, AnnotationBbox):
            old_x, old_y = artist.xy
            new_pos = (old_x + diff_data[0], old_y + diff_data[1])
            artist.xy = new_pos
        else:
            old_x, old_y = artist.get_position()
            new_pos = (old_x + diff_data[0], old_y + diff_data[1])
            artist.set_position(new_pos)

        # *** LEADER LINE UPDATE ***
        # Checks if the artist has a 'leader_line' attribute attached
        if hasattr(artist, 'leader_line'):
            line = artist.leader_line
            # Update the END point of the line (index 1) to the new box position
            # The START point (index 0) remains at the station lat/lon
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            line.set_xdata([xdata[0], new_pos[0]])
            line.set_ydata([ydata[0], new_pos[1]])
            
    except Exception as e:
        pass

def fetch_and_plot_topography(ax, extent):
    """
    Fetches and plots Natural Earth High-Res (10m) Data.
    Applies HILLSHADING (3D Effect) using LightSource.
    """
    logging.info("Adding Topography/Terrain overlay with Hillshading...")

    # --- 1. CONFIG: Correct 10m URL ---
    SR_URL = "https://naturalearth.s3.amazonaws.com/10m_raster/NE1_HR_LC_SR_W_DR.zip"
    SR_FILENAME = "NE1_HR_LC_SR_W_DR.tif"
    
    if CACHE_DIR:
        zip_path = os.path.join(CACHE_DIR, "NE1_HR_LC_SR_W_DR.zip")
        tif_path = os.path.join(CACHE_DIR, SR_FILENAME)
    else:
        return 

    # --- 2. DOWNLOAD (If needed) ---
    if not os.path.exists(tif_path):
        logging.info("Downloading High-Res Terrain (160MB)... This may take 1-2 mins...")
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(SR_URL, headers=headers, stream=True, timeout=120)
            if r.status_code == 200:
                with open(zip_path, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
                
                logging.info("Extracting High-Res data...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        if file.endswith(".tif"):
                            with open(tif_path, 'wb') as target, zip_ref.open(file) as source:
                                shutil.copyfileobj(source, target)
                            break
                logging.info("Topography ready.")
            else:
                logging.error(f"Download failed: {r.status_code}")
                return
        except Exception as e:
            logging.error(f"Error fetching topography: {e}")
            return

    # --- 3. PLOTTING WITH HILLSHADE ---
    try:
        if os.path.exists(tif_path):
            with rasterio.open(tif_path) as src:
                # Windowing to map extent
                west, east, south, north = extent
                window = src.window(west, south, east, north)
                
                # READ ALL 3 BANDS (R, G, B) to calculate grayscale "height" map
                data = src.read((1, 2, 3), window=window)
                win_transform = src.window_transform(window)
                
                if data.size == 0: return

                # --- A. RGB TO GRAYSCALE ("ELEVATION") ---
                r, g, b = data[0], data[1], data[2]
                grayscale = (0.299 * r + 0.587 * g + 0.114 * b)
                
                # --- B. CONTRAST STRETCH (Pre-processing) ---
                # We clip the data to 50-200 to crush blacks/whites for drama
                # before calculating shadows, so the "peaks" are sharper.
                grayscale = np.clip(grayscale, 50, 200)
                
                # Normalize to 0-1 for LightSource
                grayscale_norm = (grayscale - 50) / (200 - 50)

                # --- C. APPLY LIGHTSOURCE (HILLSHADE) ---
                # azdeg=315 (Northwest light), altdeg=45 (Standard afternoon sun angle)
                ls = LightSource(azdeg=315, altdeg=45)
                
                # blend_mode='soft' mixes the shading gently with the colormap
                # vert_exag=1.5 exaggerates the height for more 3D "pop"
                shaded_rgb = ls.shade(grayscale_norm, cmap=plt.cm.gray, vert_exag=1.5, blend_mode='soft')

                extent_img = (win_transform[2], win_transform[2] + win_transform[0] * grayscale.shape[1],
                              win_transform[5] + win_transform[4] * grayscale.shape[0], win_transform[5])
                
                # --- PLOT ---
                # Note: We do NOT pass cmap/vmin/vmax here because 'shaded_rgb' is already a fully colored image.
                ax.imshow(shaded_rgb, origin='upper', extent=extent_img, 
                          transform=ccrs.PlateCarree(), 
                          zorder=0.5, alpha=1.0, 
                          interpolation='lanczos')
                
                logging.info("Hillshaded Terrain plotted successfully.")
    except Exception as e:
        logging.warning(f"Failed to plot topography: {e}")

def add_rivers_and_lakes(ax, extent):
    """
    Adds Rivers and Lakes to the map.
    - plots 'ne_10m_lakes' (Polygons)
    - plots 'ne_10m_rivers_lake_centerlines' (Lines)
    - Labels major rivers dynamically.
    """
    logging.info("Adding Rivers and Lakes...")

    # --- 1. LAKES (Visual Only) ---
    # We plot lakes first so rivers can flow into/through them
    try:
        # Scale 10m is high res
        lakes_feature = cfeature.NaturalEarthFeature(
            category='physical', name='lakes', scale='10m',
            facecolor='#4682B4', edgecolor='none', alpha=0.6
        )
        # Zorder 1.6 puts it just above the Temperature Fill (1.5)
        ax.add_feature(lakes_feature, zorder=1.6)
    except Exception as e:
        logging.warning(f"Error adding lakes: {e}")

    # --- 2. RIVERS (Visual + Labels) ---
    try:
        shp_path = shpreader.natural_earth(resolution='10m', category='physical', name='rivers_lake_centerlines')
        reader = shpreader.Reader(shp_path)
        
        # Rivers to look for
        target_rivers = [
            "Chattahoochee", "Savannah", "Flint", "Altamaha", "Ocmulgee", 
            "Oconee", "Coosa", "St. Marys", "Etowah", "Tallapoosa", "Suwannee", 
            "Tennessee", "Chattooga"
        ]
        
        count = 0
        for record in reader.records():
            geo = record.geometry
            bounds = geo.bounds # (minx, miny, maxx, maxy)
            
            # Spatial Filter
            if (bounds[2] < extent[0] or bounds[0] > extent[1] or 
                bounds[3] < extent[2] or bounds[1] > extent[3]):
                continue

            name = record.attributes.get('name_en', record.attributes.get('name', ''))
            scalerank = record.attributes.get('scalerank', 10)
            
            # Filter tiny creeks
            if scalerank > 8: continue

            # Style: Major rivers thicker
            lw = 1.0 if scalerank <= 5 else 0.5
            color = '#4682B4' 
            
            # Plot Line
            ax.add_geometries([geo], ccrs.PlateCarree(), 
                              facecolor='none', edgecolor=color, linewidth=lw, 
                              zorder=1.7, alpha=0.8)

            # Label Logic
            should_label = (name and (name in target_rivers or scalerank <= 4))
            
            if should_label:
                # Find midpoint
                midpoint = geo.interpolate(0.5, normalized=True)
                
                # Check if visible
                if (extent[0] < midpoint.x < extent[1] and 
                    extent[2] < midpoint.y < extent[3]):
                    
                    # Add Text with Glow
                    ax.text(midpoint.x, midpoint.y, name.upper(),
                            transform=ccrs.PlateCarree(), 
                            fontsize=7, color='#0f3d6e', fontweight='bold',
                            ha='center', va='center', zorder=1.75, clip_on=True,
                            path_effects=[pe.withStroke(linewidth=2, foreground='white', alpha=0.7)])
            
            count += 1

        logging.info(f"Plotted {count} river segments.")

    except Exception as e:
        logging.warning(f"Error plotting rivers: {e}")

def fetch_and_plot_forecast_pressure(ax, model_data=None, day_offset=1):
    """
    Fetches WPC Forecast Pressure Centers.
    - Day 1: Layer 0
    - Day 2: Layer 3
    - Day 3: Layer 6
    """
    logging.info(f"Fetching WPC Forecast Pressure Centers (Day {day_offset})...")
    
    # --- CONFIG: ICON PATHS ---
    ICON_PATHS = {
        'H': "/home/desoxyn/frostbyte/frostbyte_project/icons/high.png",
        'L': "/home/desoxyn/frostbyte/frostbyte_project/icons/low.png"
    }
    ICON_ZOOM = 0.15

    # --- CORRECT LAYER MATH ---
    # WPC MapServer Structure:
    # Day 1: 0=Press, 1=Fronts, 2=Troughs
    # Day 2: 3=Press, 4=Fronts, 5=Troughs
    # Day 3: 6=Press, 7=Fronts, 8=Troughs
    layer_id = (day_offset - 1) * 3
    
    base_url = f"https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/natl_fcst_wx_chart/MapServer/{layer_id}"
    full_gdf = None

    try:
        query_url = f"{base_url}/query?where=1%3D1&outFields=*&f=geojson&outSR=4326"
        r = requests.get(query_url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if 'features' in data and data['features']:
                full_gdf = gpd.GeoDataFrame.from_features(data['features'])
    except Exception as e:
        logging.warning(f"Error fetching forecast pressure: {e}")

    if full_gdf is None or full_gdf.empty: return

    # Filter bounds
    xmin, xmax, ymin, ymax = MAP_EXTENT
    buffer_deg = 5.0 
    bbox_poly = Polygon([(xmin-buffer_deg, ymin-buffer_deg), (xmin-buffer_deg, ymax+buffer_deg), 
                         (xmax+buffer_deg, ymax+buffer_deg), (xmax+buffer_deg, ymin-buffer_deg)])
    
    full_gdf = full_gdf[full_gdf.geometry.intersects(bbox_poly)]

    # --- BUILD MODEL LOOKUP TREE (NAM/HRRR MSLP) ---
    model_tree = None
    model_vals = None
    
    if model_data and 'mslp' in model_data and model_data['mslp'] is not None:
        try:
            mslp_da = model_data['mslp'].squeeze()
            
            # Extract coordinates safely
            try: lons, lats = mslp_da.longitude, mslp_da.latitude
            except: 
                try: lons, lats = mslp_da.grid_lon, mslp_da.grid_lat
                except: lons, lats = mslp_da.x, mslp_da.y
            
            # Flatten arrays
            if lons.ndim == 1:
                mx, my = np.meshgrid(lons, lats)
                pts = np.column_stack((mx.ravel(), my.ravel()))
            else:
                pts = np.column_stack((lons.values.ravel(), lats.values.ravel()))
            
            # Handle Units (Pa to hPa)
            vals = mslp_da.values.ravel()
            if np.nanmax(vals) > 2000: # Likely Pascals
                vals = vals / 100.0
                
            model_vals = vals
            
            from scipy.spatial import cKDTree
            model_tree = cKDTree(pts)
            logging.info("Built Model MSLP Lookup Tree.")
        except Exception as e:
            logging.warning(f"Failed to build MSLP tree: {e}")

    # --- PLOTTING LOOP ---
    count = 0
    for _, row in full_gdf.iterrows():
        try:
            row_text = str(row.to_dict()).upper()
            sys_type = None; color = 'black'
            
            if "HIGH" in row_text or "'TYPE': 'H'" in row_text: sys_type = 'H'; color = 'blue'
            elif "LOW" in row_text or "'TYPE': 'L'" in row_text: sys_type = 'L'; color = 'red'
            
            if not sys_type: continue

            geom = row.geometry
            if geom.geom_type == 'MultiPoint': geom = geom.centroid
            lon_sys, lat_sys = geom.x, geom.y
            
            # Clamp Logic
            plot_lon = max(xmin, min(lon_sys, xmax))
            plot_lat = max(ymin, min(lat_sys, ymax))

            pressure_val = "U"
            
            # 1. PRIORITY: Model Sampling
            if model_tree:
                dist, idx = model_tree.query([lon_sys, lat_sys])
                if dist < 1.0: # Within ~60 miles
                    pressure_val = int(round(model_vals[idx]))

            # 2. FALLBACK: Text Scan
            if pressure_val == "U":
                potential_cols = ['Pressure', 'PRESS', 'PMsl', 'Label']
                for key in potential_cols:
                    if key in row and pd.notnull(row[key]):
                        val_str = str(row[key])
                        import re
                        match = re.search(r'\b(9\d{2}|10\d{2}|1100)\b', val_str)
                        if match:
                            pressure_val = int(match.group(0))
                            break

            # --- PLOT (Icon + Text Stack - EXACT GA_WX STYLE) ---
            label_text = f"{pressure_val} hPa" if pressure_val != "U" else "U"
            
            # Define Text Area
            text_area = TextArea(label_text, textprops=dict(
                color='black', fontsize=11, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
            ))
            
            # Try to load Custom Icon
            icon_path = ICON_PATHS.get(sys_type)
            final_artist = None
            
            if icon_path and os.path.exists(icon_path):
                try:
                    img_arr = plt.imread(icon_path)
                    image_box = OffsetImage(img_arr, zoom=ICON_ZOOM)
                    # Stack: Image Top, Text Bottom
                    packed_box = VPacker(children=[image_box, text_area], align="center", pad=0, sep=2)
                    final_artist = AnnotationBbox(packed_box, (plot_lon, plot_lat), frameon=False, pad=0.0, zorder=16)
                except: pass
            
            # Fallback
            if final_artist is None:
                # Align based on map edge
                ha, va = 'center', 'center'
                if lon_sys <= xmin: ha = 'left'
                elif lon_sys >= xmax: ha = 'right'
                if lat_sys <= ymin: va = 'bottom'
                elif lat_sys >= ymax: va = 'top'

                ax.text(plot_lon, plot_lat, f'{sys_type}\n{label_text}', 
                        color=color, fontsize=14, fontweight='bold',
                        ha=ha, va=va, transform=ccrs.PlateCarree(),
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, pad=2), 
                        zorder=16, clip_on=False)
            else:
                ax.add_artist(final_artist)
            
            count += 1

        except Exception as e: continue
    logging.info(f"Plotted {count} Pressure Centers.")

def fetch_and_plot_forecast_fronts(ax, day_offset=1):
    """
    Fetches WPC Forecast Fronts (Day 1, 2, or 3).
    """
    logging.info(f"Fetching WPC Forecast Fronts (Day {day_offset})...")
    
    # --- CORRECT LAYER LOGIC ---
    # Day 1: Fronts=1, Troughs=2
    # Day 2: Fronts=4, Troughs=5
    # Day 3: Fronts=7, Troughs=8
    start_layer = ((day_offset - 1) * 3) + 1
    layers_to_scan = [start_layer, start_layer + 1]

    base_url = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/natl_fcst_wx_chart/MapServer"
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

    if not all_gdfs: return

    full_gdf = pd.concat(all_gdfs, ignore_index=True)
    
    # Clip to Map
    xmin, xmax, ymin, ymax = MAP_EXTENT
    bbox_poly = Polygon([(xmin-10, ymin-10), (xmin-10, ymax+10), 
                         (xmax+10, ymax+10), (xmax+10, ymin-10)])
    gdf_clipped = full_gdf[full_gdf.geometry.intersects(bbox_poly)]
    
    if gdf_clipped.empty: return
    
    def get_style(feature_type):
        ft = str(feature_type).upper()
        base = {'zorder': 15} # Match ga_wx zorder
        if 'COLD' in ft: return {**base, 'color': 'blue', 'lw': 2.5, 'path_effects': [ColdFront(size=6, spacing=1.0)]}
        elif 'WARM' in ft: return {**base, 'color': 'red', 'lw': 2.5, 'path_effects': [WarmFront(size=6, spacing=1.0)]}
        elif 'STAT' in ft: return {**base, 'color': 'red', 'lw': 2.5, 'path_effects': [StationaryFront(size=6, spacing=1.0)]}
        elif 'OCCL' in ft: return {**base, 'color': 'purple', 'lw': 2.5, 'path_effects': [OccludedFront(size=6, spacing=1.0)]}
        elif 'TROF' in ft or 'TROUGH' in ft: return {**base, 'color': '#FF8C00', 'lw': 2.0, 'linestyle': '--', 'dashes': (5, 5)}
        elif 'DRY' in ft: return {**base, 'color': 'brown', 'lw': 2.0, 'linestyle': '-', 'dashes': (5, 5)}
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
                ax.plot(x, y, transform=ccrs.PlateCarree(), **style)

def fetch_and_plot_gdot_urban_areas(ax):
    """
    Fetches and plots official GDOT Adjusted Urban Areas from the production server.
    This layer represents densely developed territory (cities) in Georgia.
    Includes 30-day on-disk caching and integration with the Status Dashboard.
    """
    cache_filename = "gdot_urban_areas_production.pkl"
    source_label = "GDOT Urban Areas"
    
    # 1. Check local cache (30-day persistence for static census boundaries)
    # This prevents downloading a 16MB+ file on every single script run.
    cached_features = load_pickle_cache(cache_filename, 2592000)
    
    if cached_features:
        logging.info(f"Using cached {source_label}.")
        features = cached_features
        # Log a cache HIT for the final performance dashboard
        log_status(source_label, hit=True)
    else:
        logging.info(f"Fetching live {source_label} from production server...")
        # Official production endpoint for Adjusted Urban Area (rnhp server)
        base_url = "https://rnhp.dot.ga.gov/hosting/rest/services/TPro/AdjustUrbanArea/MapServer/0/query"
        
        params = {
            'where': '1=1',
            'outFields': 'NAME10', # We only need the name field for geometry identification
            'f': 'geojson',
            'outSR': '4326'        # Ensure data is in standard Lat/Lon coordinates
        }
        
        try:
            # Extended timeout (25s) because the production dataset is large and detailed
            r = requests.get(base_url, params=params, timeout=25)
            
            if r.status_code == 200:
                features = r.json().get('features', [])
                if features:
                    # Save to local .wx_cache folder for future runs
                    save_pickle_cache(cache_filename, features)
                    log_status(source_label, hit=False, success=True)
                else:
                    logging.warning(f"{source_label}: Production server returned no features.")
                    log_status(source_label, hit=False, success=False)
                    return
            else:
                logging.warning(f"{source_label} fetch failed: HTTP {r.status_code}")
                log_status(source_label, hit=False, success=False)
                return
        except Exception as e:
            logging.warning(f"Could not connect to GDOT production server for {source_label}: {e}")
            log_status(source_label, hit=False, success=False)
            return

    # 2. Plotting Style
    # face_color: Deep charcoal 'footprint' that shows through hillshading
    # alpha_val: Set to 0.35 to keep it subtle but visible
    # z_order: 0.6 places it above terrain but below all weather/text data
    face_color = '#111111'
    alpha_val = 0.35
    z_order = 0.6           
    
    count = 0
    for feat in features:
        geom = feat.get('geometry')
        if not geom:
            continue
        
        poly_type = geom.get('type')
        coords = geom.get('coordinates')
        
        # 3. Handle GeoJSON coordinate nesting for Polygons vs MultiPolygons
        if poly_type == 'Polygon':
            polys = [coords]
        elif poly_type == 'MultiPolygon':
            polys = coords
        else:
            continue
            
        for poly_coords in polys:
            # ArcGIS GeoJSON usually wraps the exterior ring as the first item in the array
            ring = poly_coords[0]
            
            # Create the visual patch for the map
            poly_patch = mpatches.Polygon(
                ring, 
                closed=True, 
                transform=ccrs.PlateCarree(),
                facecolor=face_color, 
                edgecolor='none', # No edge color creates a cleaner "stain" effect on the terrain
                alpha=alpha_val,
                zorder=z_order  
            )
            ax.add_patch(poly_patch)
            count += 1
            
    logging.info(f"Successfully plotted {count} production {source_label} segments.")

def plot_forecast_map(title, forecast_data_dict, nam_overlays, radar_time_str, filename, target_dt=None):
    logging.info(f"Generating map: {title}...")
    warnings.filterwarnings("ignore")
    
    # ==========================================
    # 1. LOCAL VARIABLE INITIALIZATION
    # ==========================================
    GLOW_CITIES = ["Atlanta", "Savannah", "Augusta", "Columbus", "Macon"]
    _collision_items = [] 
    
    ICON_DIR = "/home/desoxyn/frostbyte/frostbyte_project/icons/new icons"
    
    # --- 2. SETUP FIGURE ---
    fig = plt.figure(figsize=(16, 16), facecolor=FIG_BG_COLOR)
    text_outline = [pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()]
    model_src = nam_overlays.get('source', 'Model') if nam_overlays else "Model"
    
    draw_broadcast_header(fig, title.upper(), f"VALID: {radar_time_str} | SOURCE: {model_src}", "FORECAST")
    
    # --- LAYOUT ---
    map_x = 0.09; map_y = 0.12; map_w = 0.83; map_h = 0.73  
    ax = fig.add_axes([map_x, map_y, map_w, map_h], projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    # --- CLIP PATH ---
    clip_patch = None
    try:
        shp = shpreader.natural_earth('50m', 'cultural', 'admin_1_states_provinces')
        geoms = [r.geometry for r in shpreader.Reader(shp).records() if r.attributes['name'] in ['Georgia', 'Alabama', 'Florida', 'Tennessee', 'North Carolina', 'South Carolina']]
        if geoms:
            path = Path.make_compound_path(*shapely_to_path(unary_union(geoms).intersection(Polygon([(MAP_EXTENT[0], MAP_EXTENT[2]), (MAP_EXTENT[0], MAP_EXTENT[3]), (MAP_EXTENT[1], MAP_EXTENT[3]), (MAP_EXTENT[1], MAP_EXTENT[2])]))))
            clip_patch = PathPatch(path, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='none')
            ax.add_patch(clip_patch)
    except: pass

    def safe_clip(artist):
        if not clip_patch: return
        try: artist.set_clip_path(clip_patch)
        except: 
            try: 
                for c in artist.collections: c.set_clip_path(clip_patch)
            except: pass

    # --- BACKGROUND ---
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor=OCEAN_COLOR)
    ax.add_feature(cfeature.LAND, zorder=0, facecolor=LAND_COLOR, alpha=0.0, edgecolor='none')
    fetch_and_plot_topography(ax, MAP_EXTENT)
    add_rivers_and_lakes(ax, MAP_EXTENT)
    fetch_and_plot_gdot_urban_areas(ax)

    # --- DEFINE TEMPERATURE COLORMAP (Used for fill and legend) ---
    color_stops = [
        (-100, "#F0F0F0"), (-80, "#000058"), (-60, "#0202BB"), (-40, "#15589B"), (-20, "#4180BB"), 
        (-10, "#AD006B"), (-5, "#B6009E"), (0, "#FF01FF"), (5, "#7F1BB1"), (10, "#BA55D3"), (15, "#DA70D6"), (20, "#E0B0FF"), 
        (25, "#AFEEEE"), (30, "#00FFFF"), (35, "#00FF00"), (40, "#32CD32"), (45, "#7CFC00"), (50, "#ADFF2F"), (55, "#FFFF00"), 
        (60, "#FFD700"), (65, "#FFA500"), (70, "#FF8C00"), (75, "#FF4500"), (80, "#FF0000"), (85, "#DC143C"), (90, "#B22222"), 
        (95, "#8B0000"), (100, "#A52A2A"), (105, "#8B4513"), (110, "#A0522D"), (115, "#CD853F"), (120, "#D2B48C"), (125, "#FFDDC1"), (130, "#FFFFFF")
    ]
    stop_vals = [x[0] for x in color_stops]
    temp_cmap = mcolors.LinearSegmentedColormap.from_list('custom', [((v - min(stop_vals))/(max(stop_vals)-min(stop_vals)), c) for v, c in color_stops])
    temp_norm = Normalize(min(stop_vals), max(stop_vals))

    # --- NAM/MODEL OVERLAYS ---
    rain_cmap = ctables.registry.get_colortable('NWSReflectivity')
    
    if nam_overlays:
        if nam_overlays.get('clouds') is not None:
            cld = nam_overlays['clouds'].squeeze()
            try: x, y = cld.longitude, cld.latitude
            except: x, y = cld.x, cld.y
            safe_clip(ax.contourf(x, y, cld, levels=np.arange(10, 101, 10), cmap='gray', alpha=0.3, zorder=1.1, transform=ccrs.PlateCarree()))

        if nam_overlays.get('refl') is not None:
            refl = nam_overlays['refl'].squeeze()
            try: lons, lats = refl.longitude, refl.latitude
            except: lons, lats = refl.x, refl.y
            refl_vals = np.where(gaussian_filter(np.nan_to_num(refl.values), sigma=1.5) < 5, np.nan, gaussian_filter(np.nan_to_num(refl.values), sigma=1.5))
            safe_clip(ax.contourf(lons, lats, refl_vals, levels=np.arange(5, 80, 5), cmap=rain_cmap, alpha=0.6, zorder=2.0, transform=ccrs.PlateCarree(), extend='max'))

        if nam_overlays.get('mslp') is not None:
            mslp = nam_overlays['mslp'].squeeze()
            mslp.values = gaussian_filter(mslp.values/100.0, sigma=1.0)
            try: x, y = mslp.longitude, mslp.latitude
            except: x, y = mslp.x, mslp.y
            cs_fine = ax.contour(x, y, mslp, levels=np.arange(900, 1100, 0.5), colors="#0E2914", linewidths=0.7, alpha=0.6, zorder=1.6, transform=ccrs.PlateCarree())
            safe_clip(cs_fine)

    # --- TEMPERATURE FILL ---
    if forecast_data_dict.get('temp') is not None:
        t_data = forecast_data_dict['temp'].squeeze()
        sfc_crs = t_data.metpy.cartopy_crs
        
        # Plot fill
        cf = ax.contourf(t_data.x, t_data.y, t_data, levels=np.arange(int(t_data.min()), int(t_data.max()), 2), 
                         cmap=temp_cmap, norm=temp_norm, alpha=0.50, zorder=1.5, transform=sfc_crs)
        safe_clip(cf)
        
        # Plot lines
        temp_lines = ax.contour(t_data.x, t_data.y, t_data, levels=np.arange(-100, 130, 10), 
                                colors='white', linewidths=0.5, alpha=0.75, zorder=1.55, transform=sfc_crs)
        safe_clip(temp_lines)
        
        t_labels = ax.clabel(temp_lines, inline=True, fmt='%d', fontsize=10, colors='white', use_clabeltext=True, zorder=1.6)
        for l in t_labels: l.set_path_effects(text_outline)

    # --- MAP FEATURES ---
    ax.add_feature(cfeature.COASTLINE, edgecolor='navy', linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.STATES, edgecolor=STATE_BORDER_COLOR, linewidth=1.5, zorder=4)
    try: ax.add_feature(USCOUNTIES.with_scale('20m'), edgecolor=COUNTY_BORDER_COLOR, linewidth=0.8, linestyle='-.', zorder=2.5, alpha=0.8).set_clip_path(clip_patch)
    except: pass
    try:
        roads = shpreader.Reader(shpreader.natural_earth('10m', 'cultural', 'roads'))
        for r in roads.records():
            if r.attributes['type'] in ['Major Highway', 'Secondary Highway']:
                c = MAJOR_ROAD_COLOR if r.attributes['type'] == 'Major Highway' else SECONDARY_ROAD_COLOR
                ax.add_geometries([r.geometry], ccrs.PlateCarree(), facecolor='none', edgecolor=c, linewidth=0.5, zorder=3).set_clip_path(clip_patch)
    except: pass
    draw_interstate_shields(ax)
    
    wpc_day_offset = determine_wpc_day_offset(target_dt)
    fetch_and_plot_forecast_pressure(ax, model_data=nam_overlays, day_offset=wpc_day_offset)
    fetch_and_plot_forecast_fronts(ax, day_offset=wpc_day_offset)

    # ==========================================
    # RESTORED COLORBARS (FROM GA_WX_OLD)
    # ==========================================
    pos = ax.get_position()
    
    # 1. Cloud/IR Satellite Colorbar (Horizontal Bottom)
    cax_ir = fig.add_axes([pos.x0, 0.04, pos.width, 0.04]) 
    cbar_ir = fig.colorbar(plt.cm.ScalarMappable(cmap='gray', norm=Normalize(vmin=-90, vmax=50)), cax=cax_ir, orientation='horizontal', extend='both')
    cbar_ir.set_label('Cloud Cover / IR Overlay (Grayscale)', fontsize=12, fontweight='bold', color='white', path_effects=text_outline)
    cbar_ir.ax.tick_params(labelsize=10, colors='white')
    for label in cbar_ir.ax.get_xticklabels(): label.set_path_effects(text_outline)

    # 2. Stacked Precip Colorbars (Left Side)
    # Re-define cmaps here to be sure
    rain_cmap = ctables.registry.get_colortable('NWSReflectivity')
    fzra_cmap = mcolors.LinearSegmentedColormap.from_list("fzra", ['#FFC0CB', '#FF69B4', '#FF1493', '#C71585', '#8B0000'])
    ice_cmap  = mcolors.LinearSegmentedColormap.from_list("ice", ['#E6E6FA', '#D8BFD8', '#BA55D3', '#9400D3', '#4B0082'])
    snow_cmap = mcolors.LinearSegmentedColormap.from_list("snow_map", ['#F0FFFF', '#E0FFFF', '#00BFFF', '#1E90FF', '#0000FF', '#00008B'])

    total_h = pos.height; gap = 0.01; bar_h = (total_h - (3 * gap)) / 4; bar_x = 0.02; bar_w = 0.015                
    def add_stacked_cbar(index, cmap_in, norm_in, label):
        y_pos = pos.y0 + (index * (bar_h + gap))
        cax = fig.add_axes([bar_x, y_pos, bar_w, bar_h])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_in, norm=norm_in), cax=cax, orientation='vertical', extend='max')
        cb.set_label(label, fontsize=8, color='white', fontweight='bold', path_effects=text_outline)
        cax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=7)
        for l in cax.yaxis.get_ticklabels(): l.set_path_effects(text_outline)
    add_stacked_cbar(0, rain_cmap, Normalize(5,80), 'Rain')
    add_stacked_cbar(1, fzra_cmap, Normalize(10,60), 'Frz Rain')
    add_stacked_cbar(2, ice_cmap,  Normalize(10,60), 'Ice/Sleet')
    add_stacked_cbar(3, snow_cmap, Normalize(5,50), 'Snow')

    # 3. Temp Colorbar (Right Side)
    right_x = pos.x1 + 0.01
    cax_temp = fig.add_axes([right_x, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=temp_cmap, norm=temp_norm), cax=cax_temp, orientation='vertical', extend='both')
    cbar.set_label('Temperature (°F)', fontsize=10, fontweight='bold', color='white', path_effects=text_outline)
    cbar.set_ticks(np.arange(-100, 131, 10))
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    for l in cbar.ax.get_yticklabels(): l.set_path_effects(text_outline); l.set_color('white'); l.set_fontsize(9)
    # ==========================================

    # --- STATION LABELS ---
    def get_text_props(color_hex, fsize=12, is_bold=True):
        return dict(
            color='black', fontsize=fsize, fontweight='bold' if is_bold else 'normal',
            path_effects=[
                pe.SimplePatchShadow(offset=(1.5, -1.5), shadow_rgbFace='black', alpha=0.4),
                pe.withStroke(linewidth=2.5, foreground=color_hex), pe.Normal()
            ],
            bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.2", facecolor='white', edgecolor=color_hex, linewidth=2.5, alpha=0.85)
        )
    
    props_temp = get_text_props('#FF0000', 15)   # Red Ring
    props_pop  = get_text_props('#1E90FF', 12)   # Blue Ring
    props_wind = get_text_props('goldenrod', 12) # Gold Ring
    props_cond = get_text_props('#555555', 12, is_bold=False) # Gray Ring
    props_footer = dict(
        color='white', fontsize=11, fontweight='bold', 
        path_effects=[pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()], 
        bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.8", facecolor='#555555', edgecolor='none', alpha=0.9)
    )

    if forecast_data_dict.get('temp') is not None:
        t_da = forecast_data_dict['temp'].squeeze()
        data_crs = t_da.metpy.cartopy_crs
        
        for city, layout_data in CITIES.items():
            lon, lat = layout_data['pos']
            if not (MAP_EXTENT[0]-0.5 < lon < MAP_EXTENT[1]+0.5 and MAP_EXTENT[2]-0.5 < lat < MAP_EXTENT[3]+0.5): 
                continue
            
            try:
                pt_x, pt_y = data_crs.transform_point(lon, lat, ccrs.PlateCarree())
                xi = (np.abs(t_da.x - pt_x)).argmin().item()
                yi = (np.abs(t_da.y - pt_y)).argmin().item()
            except: continue

            def get_val(k): 
                if forecast_data_dict.get(k) is not None: 
                    return forecast_data_dict[k].isel(y=yi, x=xi).item()
                return 0
            
            t = get_val('temp')
            pop = get_val('pop')
            w_spd = get_val('wind_spd')
            w_dir = get_val('wind_dir')
            sky = get_val('sky')
            dew = get_val('dew')
            bx, by = layout_data['pos']
            ha = layout_data['ha']
            
            leader_line, = ax.plot([lon, bx], [lat, by], color='black', ls=':', lw=1.5, zorder=25, 
                                   path_effects=[pe.withStroke(linewidth=3, foreground='white', alpha=0.6)])
            safe_clip(leader_line)
            
            f_cat = calculate_flight_category(sky, pop)
            cat_color = FLIGHT_CATEGORY_COLORS.get(f_cat, "white")
            circle = ax.scatter(lon, lat, facecolor=cat_color, edgecolor='black', s=175, lw=1.5, zorder=20, transform=ccrs.PlateCarree())
            safe_clip(circle)
            
            try:
                u, v = wind_components(w_spd * units.knots, w_dir * units.degrees)
                safe_clip(ax.barbs(lon, lat, u.m, v.m, length=7, pivot='tip', zorder=21, color='#3500C7', linewidth=2.0, transform=ccrs.PlateCarree()))
            except: pass
            
            blocker = mpatches.Rectangle((lon-0.125, lat-0.125), 0.25, 0.25, transform=ccrs.PlateCarree(), alpha=0)
            ax.add_patch(blocker)
            _collision_items.append({'item': blocker, 'fixed': True})

            if city in GLOW_CITIES:
                name_props = dict(color='white', fontsize=13, fontweight='bold', path_effects=[pe.SimplePatchShadow(offset=(2, -2), shadow_rgbFace='black', alpha=0.5), pe.withStroke(linewidth=5, foreground="#FF0044", alpha=0.8), pe.withStroke(linewidth=1.5, foreground='black'), pe.Normal()], bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.2", facecolor='black', edgecolor='#FF0044', linewidth=2.5, alpha=1.0))
            else:
                name_props = dict(color='white', fontsize=13, fontweight='bold', path_effects=[pe.SimplePatchShadow(offset=(1.5, -1.5), shadow_rgbFace='black', alpha=0.5), pe.withStroke(linewidth=3, foreground='black'), pe.Normal()], bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.2", facecolor='black', edgecolor='none', linewidth=0, alpha=1.0))
            
            name_box = TextArea(city.upper(), textprops=name_props)
            temp_box = TextArea(f" {t:.0f}° ", textprops=props_temp)
            pop_box  = TextArea(f" P:{pop:.0f}% ", textprops=props_pop)
            wind_box = TextArea(f" {wind_dir_to_str(w_dir)} {int(w_spd) if not pd.isna(w_spd) else '--'} ", textprops=props_wind)
            
            cond_key = get_forecast_icon_key(sky, pop, t, is_day="High" in title)
            cond_box = TextArea(f" {cond_key.replace('Mostly','Msly').replace('Partly','Ptly')} ", textprops=props_cond)

            col_left = VPacker(children=[temp_box, cond_box], align="center", pad=2, sep=12)
            col_right = VPacker(children=[pop_box, wind_box], align="center", pad=2, sep=12)
            grid_packer = HPacker(children=[col_left, col_right], align="center", pad=2, sep=45)

            fl_lbl, fl_v = calculate_feels_like(t, dew, w_spd)
            if fl_lbl:
                footer_box = TextArea(f" {fl_lbl} {fl_v}° ", textprops=props_footer)
                data_assembly = VPacker(children=[grid_packer, footer_box], align="center", pad=1, sep=10)
            else:
                data_assembly = grid_packer

            master_packer = VPacker(children=[name_box, data_assembly], align="center", pad=0, sep=10)

            grid_ab = AnnotationBbox(master_packer, (bx, by), xybox=(0,0), xycoords='data', 
                                     boxcoords='offset points', 
                                     box_alignment={'left':(0,0.5), 'right':(1,0.5), 'center':(0.5,0.5)}.get(ha, (0.5,0.5)), 
                                     frameon=False, pad=0.0, zorder=26)
            grid_ab.leader_line = leader_line
            ax.add_artist(grid_ab)
            _collision_items.append({'item': grid_ab, 'fixed': False})

    # --- COLLISION RESOLUTION ---
    try: _resolve_text_collisions(fig, ax, _collision_items, padding_px=10, pixel_step=5)
    except Exception as e: logging.warning(f"Collision resolution failed: {e}")

    # --- FLIGHT LEGEND ---
    flight_handles = [mlines.Line2D([], [], color=c, marker='o', ls='None', ms=12, mec='black', mfc=c, label=l) for l, c in [('VFR (Clear)', '#00FF00'), ('MVFR (Marginal)', '#0000FF'), ('IFR (Low Clouds/Vis)', '#FF0000'), ('LIFR (Very Low)', '#FF00FF')]]
    flight_handles.append(mlines.Line2D([], [], color='black', marker='>', ls='None', ms=12, mec='black', mfc='black', label='Wind Direction'))
    leg_flight = ax.legend(handles=flight_handles, loc='lower left', fontsize=10, title="Station Flight Rules", bbox_to_anchor=(0.02, 0.02), facecolor='white', framealpha=0.9, edgecolor=HEADER_ACCENT, fancybox=True)
    leg_flight.set_zorder(100)
    leg_flight.get_title().set_color(HEADER_ACCENT)
    leg_flight.get_title().set_fontweight('bold')
    leg_flight.get_title().set_path_effects(text_outline)
    for text in leg_flight.get_texts(): text.set_color(HEADER_ACCENT); text.set_path_effects([pe.withStroke(linewidth=1.0, foreground='black'), pe.Normal()])
    ax.add_artist(leg_flight)

    # --- STATIC LEGEND ---
    legend_handles = [mlines.Line2D([], [], color='black', marker='o', mfc='white', mec='red', mew=2.0, ms=8, ls='None', label='Temperature'), mlines.Line2D([], [], color='black', marker='o', mfc='white', mec='#1E90FF', mew=2.0, ms=8, ls='None', label='Precip %'), mlines.Line2D([], [], color='#222222', ls='-', lw=0.6, label='MSLP (NAM)'), mlines.Line2D([], [], color='blue', lw=2.5, label='Cold Front', path_effects=[ColdFront(size=6, spacing=1.0)]), mlines.Line2D([], [], color='red', lw=2.5, label='Warm Front', path_effects=[WarmFront(size=6, spacing=1.0)])]
    leg = ax.legend(handles=legend_handles, loc='upper left', fontsize=10.5, bbox_to_anchor=(0.0, 1.0), facecolor='white', framealpha=0.85, edgecolor=HEADER_ACCENT, fancybox=True, ncol=2, columnspacing=0.8, handletextpad=0.4, borderpad=0.4)
    leg.set_zorder(100)
    for t in leg.get_texts(): t.set_color(HEADER_ACCENT); t.set_fontweight('bold'); t.set_path_effects([pe.withStroke(linewidth=2.0, foreground='black'), pe.Normal()])
    ax.add_artist(leg)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, facecolor=FIG_BG_COLOR)
    buf.seek(0); plt.close(fig)
    return buf

# --- EXECUTION BLOCK (Must be unindented to run!) ---
if __name__ == "__main__":
    try:
        logging.info("--- Starting Regional Forecast Map Gen (V70 - Hillshade & Aviation) ---")
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
                # buf = plot_forecast_map("Today's Highs", highs_data, nam_day, radar_day_str, "highs.png")
                # NEW LINE (Add target_dt=day_time):
                buf = plot_forecast_map("Today's Highs", highs_data, nam_day, radar_day_str, "highs.png", target_dt=day_time)
                with open(os.path.join(OUTPUT_DIR, f"ga_forecast_highs_{ts}.png"), 'wb') as f: f.write(buf.getbuffer())
                with open(os.path.join(PROJECT_ROOT, "ga_forecast_highs.png"), 'wb') as f: f.write(buf.getbuffer())
                logging.info("Highs map saved.")
                
            if lows_data.get('temp') is not None:
                # buf = plot_forecast_map("Tonight's Lows", lows_data, nam_night, radar_night_str, "lows.png")
                buf = plot_forecast_map("Tonight's Lows", lows_data, nam_night, radar_night_str, "lows.png", target_dt=night_time)
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