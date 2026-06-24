#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Georgia (GA) Weather Map Generator.

This script fetches METAR data, NWS alerts, GOES satellite,
and a NEXRAD radar image to generate a detailed weather map.

*** V27 Update: DUAL LEGEND FIX & FIRE SUPPORT ***
    - Fixed Matplotlib behavior where creating the Alert legend
      deleted the Static legend. Now both appear simultaneously.
    - Removed all "nanny filters" on alerts. If NWS issues it,
      it appears on the map and in the top-right legend.
    - Added specific color logic for 'Fire' and 'Red Flag' events.
"""

import matplotlib
import pandas as pd
from scipy import ndimage
matplotlib.use('Agg')
import re
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.colors import BoundaryNorm, Normalize
import matplotlib.cm
import matplotlib.colors as mcolors
import io
from datetime import datetime, timezone, timedelta
from astral import LocationInfo
from astral.sun import sun
# --- V17: Added PaddedBox
from matplotlib.offsetbox import (
    OffsetImage, AnnotationBbox, HPacker, VPacker, TextArea, PaddedBox
)
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import asyncio 
import aiohttp 
import matplotlib.patheffects as pe 
from scipy.interpolate import griddata 
from scipy.ndimage import gaussian_filter
from descartes import PolygonPatch 
from metpy.plots import USCOUNTIES
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LightSource
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from cartopy.mpl.path import shapely_to_path
import matplotlib.patches as patches
import metpy.plots as mpplots
import gzip
import unlzw3
import rasterio # <-- *** For GeoTIFF ***
import geopandas as gpd # 
from shapely.geometry import LineString, Point # <--- Add this
from scipy.ndimage import generate_binary_structure, binary_dilation
import tarfile
import shutil
from metpy.plots import ColdFront, WarmFront, StationaryFront, OccludedFront
from PIL import Image, ImageFilter, ImageEnhance # <--- Add ImageFilter, ImageEnhance

# --- IMPORTS FOR SATELLITE & RADAR ---
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
from PIL import Image
import rasterio # <-- *** NEW: For GeoTIFF ***
from siphon.radarserver import TDSCatalog as RadarCatalog
# --- NEW IMPORTS FOR CACHING ---
import pickle
import time
import zipfile
from scipy.interpolate import NearestNDInterpolator

# LOCAL

# --- SCRIPT CONFIGURATION ---
# Configure logging for console output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- NEW: API Key for Synoptic Data ---
SYNOPTIC_API_KEY = "037f0678acd746408bfc146064df2f91"

# --- CACHING CONFIGURATION ---
CACHE_DURATION_SECONDS = 3600  # 60 minutes
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    logging.warning("__file__ not defined. Using current working directory for cache.")
    SCRIPT_DIR = os.getcwd()
    
CACHE_DIR = os.path.join(SCRIPT_DIR, ".wx_cache")
ALERT_CACHE_DURATION = 120  # 2 minutes for alerts

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

# --- Weather Icon Generation (Kept for now, but not used by Synoptic) ---
def create_icon_base(fig_size=(1,1), dpi=100):
    """Creates a base 1x1 matplotlib figure for icons."""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return fig, ax

def create_ice_pellets_icon():
    """Generates an Ice Pellets (PL) icon (Orange/White balls)."""
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(patches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
    # Ice Pellets (Sleet) - Distinct from hail, usually smaller/translucent
    for pos in [(0.3, 0.4), (0.5, 0.3), (0.7, 0.4), (0.4, 0.2)]:
        # Orange outline for "Sleet" standard
        ax.add_patch(patches.Circle(pos, 0.05, facecolor='white', edgecolor='orange', linewidth=1.5))
    return _fig_to_img(fig)

def create_freezing_rain_icon():
    """Generates a Freezing Rain (FZRA) icon (Glazed look)."""
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(patches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
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
    ax.add_patch(patches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
    # Rain on left
    ax.plot([0.35, 0.3], [0.5, 0.3], color='blue', linewidth=2)
    # Snow on right
    ax.text(0.6, 0.3, '*', color='white', fontsize=25)
    return _fig_to_img(fig)

def create_severe_storm_icon():
    """Generates a Severe Storm icon with 'Electric' layered lightning."""
    fig, ax = create_icon_base()
    
    # 1. Dark, Ominous Cloud
    # Two layers: Dark Gray base, slightly lighter top for depth
    ax.add_patch(patches.Ellipse((0.5, 0.7), 0.85, 0.5, color='#222222')) # Almost Black
    ax.add_patch(patches.Circle((0.4, 0.75), 0.3, color='#333333'))
    ax.add_patch(patches.Circle((0.6, 0.75), 0.3, color='#333333'))
    
    # 2. "Glow" Lightning (The background Red/Orange halo)
    bolt_coords = [[0.4, 0.4], [0.5, 0.6], [0.6, 0.4], [0.5, 0.5]]
    ax.add_patch(patches.Polygon(bolt_coords, color='#FF4500', alpha=0.6, lw=4, closed=True)) # OrangeRed Glow
    
    # 3. "Core" Lightning (Bright Red/Yellow center)
    ax.add_patch(patches.Polygon(bolt_coords, color='#FFD700', closed=True)) # Gold Center

    # 4. Secondary Bolt (Smaller)
    bolt2_coords = [[0.3, 0.3], [0.35, 0.45], [0.4, 0.3], [0.35, 0.35]]
    ax.add_patch(patches.Polygon(bolt2_coords, color='#FF4500', alpha=0.6, lw=3, closed=True))
    ax.add_patch(patches.Polygon(bolt2_coords, color='yellow', closed=True))

def create_hail_icon():
    """Generates a Hail (GR) icon."""
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(patches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
    # Hail Stones (White Circles with black edge)
    for pos in [(0.3,0.4), (0.5,0.3), (0.7,0.4), (0.4,0.2), (0.6,0.2)]:
        ax.add_patch(patches.Circle(pos, 0.06, facecolor='white', edgecolor='#DDDDDD'))
    return _fig_to_img(fig)

def create_funnel_cloud_icon():
    """Generates a Funnel Cloud / Tornado (FC) icon."""
    fig, ax = create_icon_base()
    # Cloud Base
    ax.add_patch(patches.Ellipse((0.5, 0.8), 0.9, 0.4, color='#333333'))
    # Funnel (Triangle pointing down)
    funnel = patches.Polygon([[0.35, 0.7], [0.65, 0.7], [0.5, 0.1]], closed=True, color='#333333')
    ax.add_patch(funnel)
    return _fig_to_img(fig)

def create_mostly_sunny_icon():
    """Generates a Mostly Sunny/FEW clouds icon."""
    fig, ax = create_icon_base()
    # Big Sun
    ax.add_patch(patches.Circle((0.5, 0.5), 0.4, color='yellow'))
    for angle in range(0, 360, 45):
        x = 0.5 + 0.45 * np.cos(np.radians(angle))
        y = 0.5 + 0.45 * np.sin(np.radians(angle))
        ax.plot([0.5, x], [0.5, y], color='yellow')
    # Small Cloud bottom right
    ax.add_patch(patches.Circle((0.75, 0.25), 0.2, color='white', alpha=0.9))
    ax.add_patch(patches.Circle((0.6, 0.25), 0.15, color='white', alpha=0.9))
    return _fig_to_img(fig)

# Helper to reduce copy-paste in save logic
def _fig_to_img(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()
    return img

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

def create_mostly_cloudy_storming_icon():
    """Generates a Thunderstorm icon with 'Electric' lightning."""
    fig, ax = create_icon_base()
    # Cloud
    ax.add_patch(patches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
    ax.add_patch(patches.Circle((0.4, 0.75), 0.3, color='gray'))
    ax.add_patch(patches.Circle((0.6, 0.75), 0.3, color='gray'))
    
    # Lightning with Glow
    bolt_coords = [[0.4, 0.4], [0.5, 0.6], [0.6, 0.4], [0.5, 0.5]]
    
    # Outer Glow (Yellow-Orange)
    ax.add_patch(patches.Polygon(bolt_coords, color='#FFA500', alpha=0.5, lw=5, joinstyle='round', closed=True))
    # Inner Core (Bright Yellow)
    ax.add_patch(patches.Polygon(bolt_coords, color='#FFFFE0', closed=True)) # Light Yellow
    
    return _fig_to_img(fig)

def create_hail_icon():
    """Generates a Hail (GR) icon."""
    fig, ax = create_icon_base()
    # 1. Dark Cloud
    ax.add_patch(patches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
    ax.add_patch(patches.Circle((0.4, 0.75), 0.3, color='gray'))
    ax.add_patch(patches.Circle((0.6, 0.75), 0.3, color='gray'))
    
    # 2. Hail Stones (White balls with grey outline)
    # Different positions to look scattered
    positions = [(0.3, 0.4), (0.5, 0.3), (0.7, 0.4), (0.4, 0.2), (0.6, 0.2)]
    for pos in positions:
        ax.add_patch(patches.Circle(pos, 0.06, facecolor='white', edgecolor='lightgray', linewidth=1))
        
    return _fig_to_img(fig)

def create_funnel_cloud_icon():
    """Generates a Funnel Cloud / Tornado (FC) icon."""
    fig, ax = create_icon_base()
    
    # 1. Dark/Ominous Cloud Base
    ax.add_patch(patches.Ellipse((0.5, 0.8), 0.9, 0.4, color='#222222')) # Almost black
    ax.add_patch(patches.Circle((0.3, 0.85), 0.25, color='#222222'))
    ax.add_patch(patches.Circle((0.7, 0.85), 0.25, color='#222222'))
    
    # 2. The Funnel (Triangle pointing down)
    # Points: [Top-Left, Top-Right, Bottom-Tip]
    funnel = patches.Polygon([[0.35, 0.7], [0.65, 0.7], [0.5, 0.1]], closed=True, color='#222222')
    ax.add_patch(funnel)
    
    return _fig_to_img(fig)

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

def create_mostly_cloudy_rain_icon():
    """
    Generates a Rainy icon with 'Hydro Glow' drops.
    Features a dark cloud with glowing blue rain.
    """
    fig, ax = create_icon_base()
    
    # 1. Cloud Body (Darker Gray for rain)
    ax.add_patch(patches.Ellipse((0.5, 0.7), 0.8, 0.5, color='gray'))
    ax.add_patch(patches.Circle((0.4, 0.75), 0.3, color='gray'))
    ax.add_patch(patches.Circle((0.6, 0.75), 0.3, color='gray'))
    
    # 2. Rain Drops with Glow
    # We draw 3 distinct drops
    # (x_start, x_end), (y_start, y_end)
    drops = [
        ([0.3, 0.25], [0.4, 0.2]), # Left
        ([0.5, 0.45], [0.4, 0.2]), # Middle
        ([0.7, 0.65], [0.4, 0.2])  # Right
    ]
    
    for x_vals, y_vals in drops:
        # Glow Layer (Thicker, semi-transparent light blue)
        ax.plot(x_vals, y_vals, color='#00BFFF', alpha=0.4, linewidth=4, solid_capstyle='round')
        # Core Layer (Solid Blue)
        ax.plot(x_vals, y_vals, color='blue', linewidth=1.5, solid_capstyle='round')

    return _fig_to_img(fig)

def create_rain_icon():
    """
    Generates a generic Rain icon (Light/Moderate) with glow.
    """
    fig, ax = create_icon_base()
    
    # 1. Cloud (Standard Gray)
    ax.add_patch(patches.Ellipse((0.5, 0.8), 0.8, 0.4, color='gray'))
    
    # 2. Rain Shower (5 small drops)
    for i in range(5):
        # Calculate positions
        start_x = 0.2 + i*0.15
        end_x = 0.15 + i*0.15
        
        # Glow Layer
        ax.plot([start_x, end_x], [0.6, 0.4], color='#00BFFF', alpha=0.3, linewidth=3)
        # Core Layer
        ax.plot([start_x, end_x], [0.6, 0.4], color='blue', linewidth=1)
        
    return _fig_to_img(fig)

# --- NEW: Cache Weather Icons ---
def load_weather_icons():
    if not CACHE_DIR: return None
    filepath = os.path.join(CACHE_DIR, "weather_icons.pkl")
    if is_cache_valid(filepath, CACHE_DURATION_SECONDS * 24):  # Cache icons for 24 hours
        try:
            with open(filepath, 'rb') as f:
                logging.info("Using cached weather icons.")
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to load cached icons: {e}")
    return None

def save_weather_icons(icons):
    if not CACHE_DIR: return
    filepath = os.path.join(CACHE_DIR, "weather_icons.pkl")
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(icons, f)
        logging.info("Saved weather icons to cache.")
    except Exception as e:
        logging.warning(f"Failed to save weather icons: {e}")

logging.info("Generating weather icons...")
icons_cache = load_weather_icons()
if icons_cache:
    WEATHER_ICONS = icons_cache
else:
    WEATHER_ICONS = {
        'Sunny': create_sunny_icon(),
        'Moonlight': create_moonlight_icon(),
        'Mostly Sunny': create_mostly_sunny_icon(),
        'Partly Cloudy (Sun)': create_partially_cloudy_sun_icon(),
        'Partly Cloudy (Moon)': create_partially_cloudy_moon_icon(),
        'Cloudy': create_cloudy_icon(),
        'Overcast': create_cloudy_icon(),
        'Rain': create_rain_icon(),
        'Heavy Rain': create_mostly_cloudy_rain_icon(),
        'Storms': create_mostly_cloudy_storming_icon(),
        'Severe Storms': create_severe_storm_icon(),
        
        # --- NEW ONES ---
        'Hail': create_hail_icon(),
        'Tornado': create_funnel_cloud_icon(),
        # ----------------
        
        'Fog': create_fog_or_mist_icon(),
        'Snow': create_snowing_icon(),
        'Wintry Mix': create_wintry_mix_icon(),
        'Ice Pellets': create_ice_pellets_icon(),
        'Freezing Rain': create_freezing_rain_icon(),
        'Wintry Mix': create_mix_icon(),
        'Snow': create_snowing_icon(),
        'Rain': create_rain_icon(),
    }
    save_weather_icons(WEATHER_ICONS)
logging.info("Weather icons ready.")

# --- BROADCAST THEME CONFIGURATION ---
FIG_BG_COLOR = '#333333'       # Dark Gray background for the "TV Screen"
LAND_COLOR = '#E6E6E6'         # Very Light Gray (High Contrast)
OCEAN_COLOR = '#B0C4DE'        # Muted Blue-Gray
STATE_BORDER_COLOR = 'black'   # Sharp Black
COUNTY_BORDER_COLOR = "#00158D"# Medium Gray
MAJOR_ROAD_COLOR = '#8B0000'   # *** DARK RED (The signature look) ***
SECONDARY_ROAD_COLOR = "#6D3A00" # Lighter Red/Brown

# Text & Box Styles
FONT_FAMILY = 'DejaVu Sans'
# The "City Badge" style (Black box, White text, Square corners)
CITY_BADGE_STYLE = dict(boxstyle="square,pad=0.3", facecolor='black', 
                        edgecolor='none', alpha=1.0) 
# The "Data Box" style (White box, Cyan border, Square corners)
DATA_BOX_STYLE = dict(boxstyle="square,pad=0.4", facecolor='white', 
                      edgecolor='#00AEEF', linewidth=2, alpha=0.9)

# Header Colors
HEADER_MAIN_BG = 'white'
HEADER_SUB_BG = 'black'
HEADER_ACCENT = '#00AEEF' # Cyan/Blue

BBOX_STYLE = dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.85, edgecolor='black', linewidth=0.5)
FONT_SIZE_CITY = 11
FONT_SIZE_TEMP = 14
FONT_SIZE_DATA = 10
FONT_SIZE_WEATHER = 10
FONT_SIZE_ICAO = 6 
FONT_SIZE_TITLE = 20
FONT_SIZE_LEGEND = 10
ALERT_COLORS = {
    "Extreme": "#9900FF",
    "Severe": "#A50000",
    "Moderate": "#C08300",
    "Minor": "#4776F5",
    "Unknown": "#808080"
}
ALERT_ALPHA = 0.25

WINTER_ALERT_COLORS = {
    "Warning": "#DE03E6",
    "Watch": "#617DFA",
    "Advisory": "#7EFFFF",
}

TEMP_ALERT_COLORS = {
    "Freeze Warning": "#00008B",
    "Freeze Watch": "#1E90FF",
    "Frost Advisory": "#B0E0E0",
}
plt.rcParams['font.family'] = FONT_FAMILY

# --- NEW HELPER: Highway Shield Data ---
# --- UPDATED: Full Regional Interstate Data ---
INTERSTATE_LOCS = {
    # --- GEORGIA (Existing) ---
    "75": [(-84.92, 34.62), (-83.75, 31.85), (-84.50, 34.10)], # Dalton, Cordele, Acworth
    "85": [(-84.95, 33.25), (-83.45, 34.15)], # Newnan, Commerce
    "20": [(-85.25, 33.65), (-82.55, 33.52)], # Bremen, Thomson
    "16": [(-82.90, 32.55)],                  # Dublin
    "95": [(-81.55, 31.45)],                  # GA Coast
    "285": [(-84.20, 33.95)],                 # ATL Bypass
    "475": [(-83.72, 32.80)],                 # Macon Bypass
    "520": [(-82.05, 33.45)],                 # Augusta Bypass
    
    # --- ALABAMA / FLORIDA (Border) ---
    "10": [(-84.50, 30.55), (-86.00, 30.65)], # FL Panhandle
    "59": [(-85.50, 34.70)],                  # AL/GA Corner
    "65": [(-86.72, 33.50), (-86.72, 31.50)], # AL (Birmingham/Montgomery)
    
    # --- TENNESSEE (Border) ---
    "24": [(-85.40, 35.02)],                  # Chattanooga West
    "75N": [(-84.60, 35.30)],                 # Cleveland TN (Distinct from GA 75)

    # --- SOUTH CAROLINA (New) ---
    "85SC": [(-82.50, 34.75)],                # Greenville/Anderson
    "26":   [(-82.05, 34.90), (-81.40, 34.20)], # Spartanburg, Newberry
    "385":  [(-82.25, 34.65)],                # Greenville Loop
    "20SC": [(-81.75, 33.70), (-81.15, 33.95)], # Aiken, Columbia West
    "77":   [(-81.05, 34.40)],                # Rock Hill / Columbia North
    "95SC": [(-80.85, 32.80)],                # Walterboro area

    # --- NORTH CAROLINA (New - Southern Edge) ---
    "85NC": [(-81.30, 35.25)],                # Gastonia
    "26NC": [(-82.45, 35.35)],                # Hendersonville
    "40":   [(-82.80, 35.55)],                # Canton/Waynesville
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
    # The shield shape is complex, we use specific vertices to approximate it.
    
    # Common Widths
    left_x, right_x = 0.1, 0.9
    mid_x = 0.5
    
    # The "Horizon" line where Red meets Blue
    split_y = 0.65
    
    # Bottom Tip
    bottom_y = 0.05
    
    # Top Peak
    top_y = 0.95
    shoulder_y = 0.85 # Where the curve starts
    
    # --- A. Draw the Blue Body (Bottom) ---
    # Triangle-ish shape from the split line down to the tip
    blue_verts = [
        (left_x, split_y),
        (right_x, split_y),
        (right_x, 0.55), # Slight vertical drop before curving
        (mid_x, bottom_y), # The Tip
        (left_x, 0.55),
        (left_x, split_y)
    ]
    blue_body = mpatches.Polygon(blue_verts, closed=True, color='#003399', ec='none')
    ax.add_patch(blue_body)

    # --- B. Draw the Red Crown (Top) ---
    red_verts = [
        (left_x, split_y),
        (left_x, shoulder_y),
        (0.2, top_y), # Curve in
        (0.8, top_y), # Curve in
        (right_x, shoulder_y),
        (right_x, split_y),
        (left_x, split_y)
    ]
    red_crown = mpatches.Polygon(red_verts, closed=True, color='#CE2029', ec='none')
    ax.add_patch(red_crown)
    
    # --- C. Add the White Border ---
    # We re-trace the entire outer perimeter with a thick white line
    full_perimeter = [
        (left_x, split_y), (left_x, 0.55), (mid_x, bottom_y), # Left side down
        (right_x, 0.55), (right_x, split_y), # Right side up
        (right_x, shoulder_y), (0.8, top_y), (0.2, top_y), (left_x, shoulder_y), # Top curve
        (left_x, split_y) # Close
    ]
    border = mpatches.Polygon(full_perimeter, closed=True, fill=False, edgecolor='white', linewidth=4)
    ax.add_patch(border)
    
    # --- D. Add Gloss (Horizontal Hotdog Shine) ---
    # A semi-transparent white ellipse across the top
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
            # Zoom controls the size. 0.15 is roughly "Road Sign" size.
            imagebox = OffsetImage(shield_img_array, zoom=0.15) 
            
            # B. Place the Image
            ab = AnnotationBbox(imagebox, (lon, lat), frameon=False, pad=0.0, zorder=3.5)
            ax.add_artist(ab)
            
            # C. The Text (The Number)
            # We draw the number in White, Bold, centered on the shield
            # We nudge the text slightly down (y-0.02) to center it in the "Blue" part visually
            ax.text(lon, lat - 0.02, route_label, 
                    color='white', fontsize=7, fontweight='bold',
                    ha='center', va='center', transform=ccrs.PlateCarree(), zorder=3.6,
                    path_effects=[pe.withStroke(linewidth=1.0, foreground="#003399")])

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

def determine_headline(alerts, atlanta_temp=None):
    """Determines the broadcast title and colors based on active threats."""
    # Defaults
    title = "Georgia - RTMA - Mesonet Surface Analysis"
    bg_color = HEADER_MAIN_BG # White
    text_color = "black"
    
    # Extract event types from alerts
    alert_types = [a.get('event', '') for a in alerts]
    
    # Priority 1: Tornado
    if any("Tornado" in t for t in alert_types):
        return "TORNADO WARNING", "#CC0000", "white" # Red / White
        
    # Priority 2: Severe Thunderstorm
    if any("Severe Thunderstorm" in t for t in alert_types):
        return "SEVERE WEATHER ALERT", "#FFEE00", "black" # Yellow / Black
        
    # Priority 3: Winter
    if any(x in str(alert_types) for x in ["Winter Storm", "Ice Storm", "Blizzard"]):
        return "WINTER STORM MODE", "#8C00FF", "white" # Violet / White
        
    # Priority 4: Heat
    if any("Heat" in t for t in alert_types) or (atlanta_temp and atlanta_temp > 95):
        return "EXTREME HEAT ALERT", "#FF7B00", "white" # Orange-Red / White

    # Priority 5: Tropical
    if any("Hurricane" in t for t in alert_types) or any("Tropical" in t for t in alert_types):
        return "HURRICANE ALERT", "#00198B", "white" # Dark Red / White
        
    return title, bg_color, text_color

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
        # We clamp using min(1.0, ...) to prevent the WARNING
        c_top = colorsys.hsv_to_rgb(h, max(0, s-0.3), min(1.0, v*1.5))
        
        # Mid Upper: The base color
        c_mid_upper = rgb
        
        # Horizon: Darker and more saturated
        c_mid_lower = colorsys.hsv_to_rgb(h, min(1.0, s+0.1), v*0.7)
        
        # Bottom: Slight reflection (Clamped to avoid 1.03 warning)
        c_bot = colorsys.hsv_to_rgb(h, s, min(1.0, v*1.1))

    # --- Build the Gradient Array ---
    half_n = n_points // 2
    
    R = np.concatenate([np.linspace(c_top[0], c_mid_upper[0], half_n),
                        np.linspace(c_mid_lower[0], c_bot[0], half_n)])
    G = np.concatenate([np.linspace(c_top[1], c_mid_upper[1], half_n),
                        np.linspace(c_mid_lower[1], c_bot[1], half_n)])
    B = np.concatenate([np.linspace(c_top[2], c_mid_upper[2], half_n),
                        np.linspace(c_mid_lower[2], c_bot[2], half_n)])
    
    # Stack into an image (1 pixel wide, N pixels tall)
    gradient = np.dstack((R, G, B))
    
    # *** HOTDOG STYLE FIX ***
    # We transpose axes (1, 0, 2) to turn it sideways.
    # It creates a Wide, Short image (N pixels wide, 1 pixel tall).
    # Matplotlib stretches this vertically to fill the bar, creating horizontal bands.
    gradient = np.transpose(gradient, (1, 0, 2))
    
    return gradient

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
    # Matches the alert color, or Silver if default
    frame_color = "#C0C0C0" if title_bg_color == 'white' or title_bg_color == HEADER_MAIN_BG else title_bg_color
    
    # Outer Bezel Box
    logo_bg_rect = mpatches.FancyBboxPatch(
        (logo_coords[0], logo_coords[1]), logo_coords[2], logo_coords[3],
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="black", edgecolor=frame_color, linewidth=3,
        transform=fig.transFigure, zorder=19
    )
    fig.add_artist(logo_bg_rect)

    # Logo Image
    logo_ax = fig.add_axes(logo_coords, zorder=20)
    logo_ax.axis('off')
    logo_path = "/home/desoxyn/frostbyte/frostbyte_project/boxlogo2.png"
    try:
        logo_img = plt.imread(logo_path)
        logo_ax.imshow(logo_img)
    except Exception:
        logo_ax.remove()
        fig.text(logo_coords[0] + 0.06, logo_coords[1] + 0.06, "GA\nWX", 
                 color='white', fontsize=20, fontweight='bold', 
                 ha='center', va='center', transform=fig.transFigure, zorder=21)

    # --- 2. Main Title Bar (Hotdog Gloss) ---
    ax_title = fig.add_axes([header_x, title_y, header_w, title_h], zorder=9)
    ax_title.set_axis_off()
    
    gloss_grad = create_aggressive_gloss(title_bg_color)
    # aspect='auto' is CRITICAL here to stretch the 1-pixel-tall gradient to fill the box
    ax_title.imshow(gloss_grad, aspect='auto', extent=[0, 1, 0, 1])

    # Top "Shine" Line (Horizontal)
    ax_title.plot([0, 1], [0.98, 0.98], color='white', alpha=0.6, linewidth=1, transform=ax_title.transAxes)

    # Title Text
    # If using dark alert colors (Red/Purple), make text White with Black outline
    # If using light alert colors (Yellow/White), make text Black with White outline
    # (The logic passed in usually handles this, but outlining ensures readability)
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
    # Top shine
    ax_sub.plot([0, 1], [0.95, 0.95], color='white', alpha=0.3, linewidth=0.5, transform=ax_sub.transAxes)

    # Sub Text
    fig.text(header_x + 0.02, sub_y + (sub_h/2), sub_title.upper(), 
             color='white', fontsize=14, fontweight='bold', 
             ha='left', va='center', transform=fig.transFigure, zorder=12)

    # --- 4. Right Badge ("LIVE") ---
    badge_ax = fig.add_axes([0.92, 0.89, 0.07, 0.04], zorder=13)
    badge_ax.set_axis_off()
    badge_gloss = create_aggressive_gloss(HEADER_ACCENT) # Blue gloss
    badge_ax.imshow(badge_gloss, aspect='auto', extent=[0, 1, 0, 1])
    
    fig.text(0.955, 0.91, right_badge_text, color='white', fontsize=12, fontweight='bold',
             ha='center', va='center', transform=fig.transFigure, zorder=14)

# --- Configuration (GEORGIA) ---
# MAP_EXTENT = [-86.8, -80.7, 30.2, 35.8] # [lon_min, lon_max, lat_min, lat_max]
# MAP_EXTENT = [-87.5, -73.0, 27.5, 37.5] # too smol
MAP_EXTENT = [-94.5, -73.0, 23.5, 37.5]
# --- CITIES is now used to identify which stations to plot with full details ---
# === MODIFICATION: Moved Peachtree City from SILENT to CITIES ===
# --- CONFIGURATION: CITIES & LAYOUTS ---

# MIX: Core Georgia + Gulf Coast + TN + Specific NC/SC Military Fields
# --- CONFIGURATION: CITIES & LAYOUTS ---

# MIX: Core Georgia + Gulf Coast + TN + Specific NC/SC Military Fields
# Format: "City": (Lat, Lon, ICAO)
# --- CONFIGURATION: CITIES & LAYOUTS ---

# Format: "City Name": (Latitude, Longitude, "ICAO_ID")
CITIES = {
    # --- GEORGIA (The "Big 7" Only) ---
    "Rome":          (34.3525, -85.1636, "KRMG"),
    "Atlanta":       (33.6407, -84.4277, "KATL"),
    "Athens":        (33.9480, -83.3260, "KAHN"),
    "Columbus":      (32.5163, -84.9389, "KCSG"),
    "Robins AFB":    (32.6400, -83.5917, "KWRB"), # Replaces Macon
    "Savannah":      (32.1276, -81.2021, "KSAV"),
    "Valdosta":      (30.7825, -83.2767, "KVLD"),

    # --- TENNESSEE ---
    "Nashville":     (36.1263, -86.6774, "KBNA"),
    "Knoxville":     (35.8124, -83.9928, "KTYS"),
    "Columbia":      (35.5539, -87.1793, "KMRC"), 
    "Tri-Cities":    (36.4752, -82.4074, "KTRI"),

    # --- ALABAMA ---
    "Birmingham":    (33.5629, -86.7535, "KBHM"),
    "Montgomery":    (32.3006, -86.3939, "KMGM"),
    "Mobile":        (30.6914, -88.2428, "KMOB"),

    # --- GULF COAST ---
    "Gulfport":      (30.4073, -89.0701, "KGPT"),
    "Keesler AFB":   (30.4100, -88.9200, "KBIX"), 
    "New Orleans":   (29.9911, -90.2592, "KMSY"),
    
    # --- FLORIDA ---
    "Tallahassee":   (30.3930, -84.3533, "KTLH"),
    "Jacksonville":  (30.4941, -81.6879, "KJAX"),
    "Tampa":         (27.9772, -82.5311, "KTPA"),
    "Miami":         (25.7933, -80.2906, "KMIA"),
    "Key West":      (24.5551, -81.7800, "KEYW"),
    
    # --- SOUTH CAROLINA ---
    "Charleston":    (32.8998, -80.0512, "KCHS"),
    "Beaufort MCAS": (32.4774, -80.7231, "KNBC"),
    "Columbia SC":   (33.9388, -81.1195, "KCAE"),
    "Greenville":    (34.8960, -82.2189, "KGSP"),

    # --- NORTH CAROLINA ---
    "Wilmington":    (34.2706, -77.9026, "KILM"),
    "New River MCAS":(34.7084, -77.4437, "KNCA"), 
    "Cherry Point":  (34.9009, -76.8807, "KNKT"), 
    "Charlotte":     (35.2140, -80.9431, "KCLT"),
}

# Layouts tweaked to prevent overlap on the crowded coast
# --- CUSTOM LAYOUTS (Cleaned Up for GA) ---
# --- CUSTOM LAYOUTS (Spaced Out for Leader Lines) ---
# 'pos': (Longitude, Latitude) of the TEXT BOX
# 'ha': Alignment relative to the dot
# --- CUSTOM LAYOUTS (Exploded View: Pushed Deep into Oceans) ---
# 'pos': (Longitude, Latitude) of the TEXT BOX
# 'ha': 'left' (text extends right), 'right' (text extends left), 'center'
CUSTOM_LAYOUTS = {
    # --- GEORGIA (Central - Keep Tight) ---
    "Rome":          {'pos': (-85.5, 34.8), 'ha': 'left'},   # Up/Right
    "Athens":        {'pos': (-83.0, 34.4), 'ha': 'left'},   # Right
    "Atlanta":       {'pos': (-84.6, 34.1), 'ha': 'right'},  # Left (Clear of Athens)
    "Macon":         {'pos': (-83.2, 32.5), 'ha': 'left'},   # SE
    "Columbus":      {'pos': (-85.8, 32.5), 'ha': 'right'},  # West
    "Robins AFB":    {'pos': (-83.0, 32.2), 'ha': 'left'},   # Below Macon
    "Augusta":       {'pos': (-81.5, 33.6), 'ha': 'left'},   # East
    "Savannah":      {'pos': (-82.6, 32.0), 'ha': 'right'},  # Inland (To clear coast)
    "Valdosta":      {'pos': (-83.3, 30.3), 'ha': 'center'}, # South

    # --- TENNESSEE (Top Edge) ---
    "Tri-Cities":    {'pos': (-81.5, 37.0), 'ha': 'left'},   # Far NE
    "Knoxville":     {'pos': (-83.5, 36.8), 'ha': 'left'},   # Up
    "Nashville":     {'pos': (-86.8, 36.8), 'ha': 'center'}, # Up
    "Columbia":      {'pos': (-88.0, 35.8), 'ha': 'right'},  # West

    # --- ALABAMA (Left Flank) ---
    "Birmingham":    {'pos': (-87.8, 33.8), 'ha': 'right'},  # NW
    "Montgomery":    {'pos': (-87.2, 32.0), 'ha': 'right'},  # SW

    # --- GULF COAST (Fanned Out into Gulf) ---
    # Mobile -> Down/Center
    "Mobile":        {'pos': (-88.0, 29.5), 'ha': 'center'}, 
    # Keesler -> Pushed East/Down
    "Keesler AFB":   {'pos': (-88.2, 29.0), 'ha': 'left'},   
    # Gulfport -> Pushed West/Down
    "Gulfport":      {'pos': (-90.0, 29.8), 'ha': 'right'},  
    # New Orleans -> Deep SW
    "New Orleans":   {'pos': (-91.5, 28.5), 'ha': 'right'}, 

    # --- FLORIDA (Pushed Out to Sea) ---
    "Tallahassee":   {'pos': (-84.3, 29.2), 'ha': 'center'}, # Deep South
    "Jacksonville":  {'pos': (-80.0, 30.3), 'ha': 'left'},   # Far East (Atlantic)
    "Tampa":         {'pos': (-84.5, 27.5), 'ha': 'right'},  # Far West (Gulf)
    "Miami":         {'pos': (-78.5, 25.8), 'ha': 'left'},   # Far East (Atlantic)
    "Key West":      {'pos': (-82.5, 23.8), 'ha': 'right'},  # West/Bottom

    # --- CAROLINAS (The "Atlantic Wall" - Fanned Out) ---
    "Greenville":    {'pos': (-82.5, 35.3), 'ha': 'right'},  # Inland
    "Columbia SC":   {'pos': (-80.5, 33.8), 'ha': 'right'},  # Inland
    "Charlotte":     {'pos': (-80.5, 35.8), 'ha': 'center'}, # Up
    
    # Coastal Spread (Top to Bottom)
    "Cherry Point":  {'pos': (-74.5, 35.5), 'ha': 'left'},   # Far East
    "New River MCAS":{'pos': (-75.0, 34.8), 'ha': 'left'},   # East/Down
    "Wilmington":    {'pos': (-75.5, 33.8), 'ha': 'left'},   # SE
    "Charleston":    {'pos': (-77.5, 32.2), 'ha': 'left'},   # SE (Deep Water)
    "Beaufort MCAS": {'pos': (-79.0, 31.2), 'ha': 'left'},   # South (Deep Water)
}

# --- AVIATION CONSTANTS ---
FLIGHT_CATEGORY_COLORS = {
    "VFR": "#00FF00",   # Green (Clear)
    "MVFR": "#0000FF",  # Blue (Marginal)
    "IFR": "#FF0000",   # Red (Instrument)
    "LIFR": "#FF00FF",  # Magenta (Low Instrument)
    "N/A": "white"
}

def get_flight_category(metar_str):
    """
    Determines flight category based on METAR string using simple regex.
    VFR:  Ceiling > 3000ft AND Vis > 5sm
    MVFR: Ceiling 1000-3000ft OR Vis 3-5sm
    IFR:  Ceiling 500-1000ft OR Vis 1-3sm
    LIFR: Ceiling < 500ft OR Vis < 1sm
    """
    if not metar_str: return "N/A"
    
    # 1. Parse Visibility (e.g., "10SM", "1/2SM", "3SM")
    vis_val = 10.0 # Default to unrestricted
    vis_match = re.search(r'\b(\d+(?:/\d+)?|M1/4)SM', metar_str)
    if vis_match:
        val_str = vis_match.group(1)
        if '/' in val_str:
            n, d = val_str.split('/')
            vis_val = float(n) / float(d)
        elif val_str == 'M1/4':
            vis_val = 0.25
        else:
            try: vis_val = float(val_str)
            except: pass

    # 2. Parse Ceiling (Lowest BKN or OVC or VV)
    ceil_val = 12000 # Default to high
    cloud_pattern = re.compile(r'(BKN|OVC|VV)(\d{3})')
    clouds = cloud_pattern.findall(metar_str)
    
    if clouds:
        # Find lowest broken/overcast layer
        # Match is tuple: ('OVC', '007') -> height is 700
        heights = [int(c[1]) * 100 for c in clouds]
        ceil_val = min(heights)
    elif 'CLR' in metar_str or 'SKC' in metar_str or 'FEW' in metar_str or 'SCT' in metar_str:
        ceil_val = 12000
    
    # 3. Determine Category
    if vis_val < 1 or ceil_val < 500:
        return "LIFR"
    elif vis_val < 3 or ceil_val < 1000:
        return "IFR"
    elif vis_val <= 5 or ceil_val <= 3000:
        return "MVFR"
    else:
        return "VFR"
    
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
            x_coords = dat['x'].values
            y_coords = dat['y'].values
            
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
    if bt is None or x_coords is None or y_coords is None or sat_proj is None:
        logging.error("Cannot plot satellite overlay: Data is missing.")
        return

    try:
        logging.info("Plotting satellite overlay...")
        levels = np.arange(-80, 41, 5)
        cmap_bt = matplotlib.colormaps['gray']
        # norm_bt = Normalize(vmin=-80, vmax=40) # Unused if we remove the colorbar below
        
        x, y = np.meshgrid(x_coords, y_coords)
        
        # Plot B&W Layer
        contour_bt = ax.contourf(x, y, bt, levels=levels, cmap=cmap_bt, 
                                 alpha=0.7, transform=sat_proj, zorder=1.0, extend='both', transform_first=True)
        
        # Plot Contour Lines
        contour_lines = ax.contour(x, y, bt, levels=levels, 
                                   colors='dimgrey', linewidths=0.7,
                                   alpha=0.6, transform=sat_proj, zorder=1.1)
        
        # --- REMOVED DUPLICATE COLORBAR CODE FROM HERE ---
        # The main function handles the colorbar generation now.
            
    except Exception as e:
        logging.error(f"Satellite plotting error: {e}")
        
# --- END SATELLITE FUNCTIONS ---

def add_radar_image_overlay(fig, ax, cache_duration, text_outline, clip_patch=None, rtma_data=None, hrrr_data=None):
    """
    Fetches the Iowa State High-Res Radar (N0Q) GeoTIFF.
    Uses HRRR (Vertical Column Analysis) to mask radar into Snow/Ice/Freezing Rain.
    Falls back to RTMA surface temp if HRRR is unavailable.
    """
    logging.info("Attempting to fetch High-Res Radar (N0Q) GeoTIFF overlay...")
    
    # --- RADAR DOWNLOAD & CACHE ---
    RADAR_CACHE_TTL = 600
    cache_file_gtif = "radar_cache_n0q.tif.Z" 
    filepath_gtif = os.path.join(CACHE_DIR, cache_file_gtif) if CACHE_DIR else None
    
    raw_bytes = None

    # 1. Cache Check
    if filepath_gtif and is_cache_valid(filepath_gtif, RADAR_CACHE_TTL):
        try:
            with open(filepath_gtif, 'rb') as f: raw_bytes = f.read()
            logging.info("Using cached radar file.")
        except: raw_bytes = None

    # 2. Download
    if raw_bytes is None:
        url_gtif = 'https://mesonet.agron.iastate.edu/data/gis/images/4326/USCOMP/n0q_0.tif.Z'
        try:
            r = requests.get(url_gtif, timeout=20)
            if r.status_code == 200:
                if filepath_gtif:
                    with open(filepath_gtif, 'wb') as f: f.write(r.content)
                raw_bytes = r.content
        except: pass

    if raw_bytes is None: return 

    # 3. Process GeoTIFF
    try:
        if raw_bytes[:2] == b'\x1f\x8b': decompressed_data = gzip.decompress(raw_bytes)
        else: decompressed_data = unlzw3.unlzw(raw_bytes)
        
        with io.BytesIO(decompressed_data) as memfile:
            with rasterio.open(memfile) as src:
                # Handle Bounds
                is_identity = src.transform.is_identity or src.bounds.left == 0.0
                if is_identity:
                    raw_data = src.read(1)
                    height, width = raw_data.shape
                    west, south, east, north = -126.0, 23.0, -66.0, 50.0
                    lons = np.linspace(west, east, width)
                    lats = np.linspace(north, south, height)
                    radar_lons, radar_lats = np.meshgrid(lons, lats)
                    precise_extent = [west, east, south, north]
                else:
                    west, east, south, north = MAP_EXTENT
                    window = src.window(west-1, south-1, east+1, north+1)
                    raw_data = src.read(1, window=window)
                    win_transform = src.window_transform(window)
                    height, width = raw_data.shape
                    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                    xs, ys = rasterio.transform.xy(win_transform, rows, cols, offset='center')
                    radar_lons = np.array(xs)
                    radar_lats = np.array(ys)
                    precise_extent = [radar_lons.min(), radar_lons.max(), radar_lats.min(), radar_lats.max()]

        # Convert to dBZ
        dbz_data = (raw_data.astype(np.float32) * 0.5) - 32.5
        dbz_data = gaussian_filter(dbz_data, sigma=1.5) # Smooth it

        # Mask Clear Air
        mask_clear = dbz_data < 5
        
        # --- PRECIPITATION TYPE LOGIC ---
        is_snow = np.zeros_like(dbz_data, dtype=bool)
        is_ice  = np.zeros_like(dbz_data, dtype=bool)
        is_fzra = np.zeros_like(dbz_data, dtype=bool)
        
        # PRIORITY 1: HRRR (Vertical Physics)
        if hrrr_data and hrrr_data[2] is not None:
            logging.info("Applying HRRR Microphysics mask to Radar...")
            h_lat, h_lon, cat_snow, cat_ice, cat_fzra = hrrr_data
            
            # Interpolate HRRR grid to Radar grid (Nearest Neighbor for binary flags)
            points = np.column_stack((h_lon.flatten(), h_lat.flatten()))
            
            # --- OPTIMIZED HRRR INTERPOLATION (cKDTree) ---
            from scipy.spatial import cKDTree

            try:
                logging.info("Building spatial tree for HRRR/Radar interpolation (cKDTree)...")
                
                # 1. Flatten coordinates
                # Source (HRRR)
                src_coords = np.column_stack((h_lon.ravel(), h_lat.ravel()))
                # Target (Radar)
                tgt_coords = np.column_stack((radar_lons.ravel(), radar_lats.ravel()))
                
                # 2. Build Tree & Query Once
                # workers=-1 uses all available CPU cores for massive speedup
                tree = cKDTree(src_coords)
                _, indices = tree.query(tgt_coords, k=1, workers=-1)
                
                # 3. Map Data using the indices (Instantaneous)
                radar_shape = radar_lons.shape
                
                grid_snow = cat_snow.ravel()[indices].reshape(radar_shape)
                grid_ice  = cat_ice.ravel()[indices].reshape(radar_shape)
                grid_fzra = cat_fzra.ravel()[indices].reshape(radar_shape)

                # 4. Create Boolean Masks
                is_snow = grid_snow > 0.5
                is_ice  = grid_ice > 0.5
                is_fzra = grid_fzra > 0.5
                
                logging.info("HRRR interpolation complete.")

            except Exception as e:
                logging.warning(f"HRRR Interpolation failed: {e}")
                # Fallback to zeros if interpolation fails
                is_snow = np.zeros_like(dbz_data, dtype=bool)
                is_ice  = np.zeros_like(dbz_data, dtype=bool)
                is_fzra = np.zeros_like(dbz_data, dtype=bool)

        # PRIORITY 2: RTMA (Surface Temp Fallback)
        elif rtma_data and rtma_data[2] is not None:
            logging.info("HRRR missing. Falling back to RTMA Surface Temp mask...")
            r_lat, r_lon, r_temp = rtma_data
            points = np.column_stack((r_lon.flatten(), r_lat.flatten()))
            try:
                radar_temp = griddata(points, r_temp.flatten(), (radar_lons, radar_lats), method='linear')
                mask_nan = np.isnan(radar_temp)
                if np.any(mask_nan):
                    radar_temp[mask_nan] = griddata(points, r_temp.flatten(), (radar_lons[mask_nan], radar_lats[mask_nan]), method='nearest')
                
                # Simple thresholds (cannot detect ice/fzra accurately)
                is_snow = radar_temp < 34.0
                # We won't try to guess FZRA/ICE with surface temp only
            except: pass

        # Create masked arrays for each type
        # Priority of visibility: Snow > Ice > FZRA > Rain
        dbz_snow = np.ma.masked_where(mask_clear | ~is_snow, dbz_data)
        dbz_ice  = np.ma.masked_where(mask_clear | ~is_ice, dbz_data)
        dbz_fzra = np.ma.masked_where(mask_clear | ~is_fzra, dbz_data)
        dbz_rain = np.ma.masked_where(mask_clear | is_snow | is_ice | is_fzra, dbz_data)
        
        # --- PLOTTING ---
        radar_proj = ccrs.PlateCarree()

        # 1. RAIN (Standard NWS Reflectivity)
        rain_cmap = ctables.registry.get_colortable('NWSReflectivity')
        ax.imshow(dbz_rain, origin='upper', extent=precise_extent, transform=radar_proj,
                  cmap=rain_cmap, vmin=5, vmax=80, zorder=4.0, alpha=0.75)
        
        # 2. FREEZING RAIN (Purple - Lavender to Indigo)
        fzra_cmap = mcolors.LinearSegmentedColormap.from_list("fzra", ['#E6E6FA', '#D8BFD8', '#BA55D3', '#9400D3', '#4B0082'])
        ax.imshow(dbz_fzra, origin='upper', extent=precise_extent, transform=radar_proj,
                  cmap=fzra_cmap, vmin=5, vmax=50, zorder=4.1, alpha=0.75)
        # 3. ICE PELLETS / SLEET (Pink - Pink to Dark Red)
        ice_cmap = mcolors.LinearSegmentedColormap.from_list("ice", ['#FFC0CB', '#FF69B4', '#FF1493', '#C71585', '#8B0000'])
        ax.imshow(dbz_ice, origin='upper', extent=precise_extent, transform=radar_proj,
                  cmap=ice_cmap, vmin=5, vmax=50, zorder=4.2, alpha=0.75)
        
        # 4. SNOW (Blue)
        snow_cmap = mcolors.LinearSegmentedColormap.from_list("snow_map", ['#F0FFFF', '#E0FFFF', '#00BFFF', '#1E90FF', '#0000FF', '#00008B'])
        ax.imshow(dbz_snow, origin='upper', extent=precise_extent, transform=radar_proj,
                  cmap=snow_cmap, vmin=5, vmax=50, zorder=4.3, alpha=0.75)
            
        logging.info("Radar GeoTIFF overlay plotted successfully.")

    except Exception as e:
        logging.error(f"Radar Error: {e}")

# --- *** NEW ASYNC METAR FUNCTIONS FOR SPEED *** ---
# ==========================================
# IEM / MADIS DATA FUNCTIONS (COMPLETE)
# ==========================================

def process_iem_data(json_data):
    """
    Parses the IEM JSON response into the format expected by the plotting script.
    Extracts raw METARs to determine Flight Category and Weather Conditions.
    """
    parsed_data = {}
    
    if not json_data or 'data' not in json_data:
        logging.warning("IEM data is empty or missing 'data' key.")
        return parsed_data
        
    for station in json_data['data']:
        try:
            # IEM uses "KATL" or "ATL". We map this ID later.
            station_id = station.get("station")
            if not station_id: continue
            
            # --- 1. Basic Variables ---
            temp_f = station.get("tmpf")
            dew_point_f = station.get("dwpf")
            wind_spd = station.get("sknt")
            wind_dir = station.get("drct")
            alt_inHg = station.get("alti")
            
            # Ensure types are correct (float/int)
            if temp_f is not None: temp_f = float(temp_f)
            if dew_point_f is not None: dew_point_f = float(dew_point_f)
            if wind_spd is not None: wind_spd = int(float(wind_spd))
            if wind_dir is not None: wind_dir = int(float(wind_dir))
            
            # --- 2. Advanced Logic (Conditions & Flight Cat) ---
            raw_metar = station.get("raw", "")
            condition = " "
            flight_cat = "N/A"
            press_arrow = ""
            
            if raw_metar:
                # Helper: need lat/lon to determine day/night icon
                lat = station.get("lat")
                lon = station.get("lon")
                
                # Check that we have valid location data
                if lat and lon:
                    is_day = is_daytime(lat, lon)
                    cloud_info = extract_cloud_info(raw_metar)
                    weather_codes = extract_weather_phenomena(raw_metar)
                    
                    # Combine cloud layers for the condition mapper
                    combined_clouds = cloud_info['low'] + cloud_info['mid'] + cloud_info['high']
                    if cloud_info['vertical_visibility']:
                         combined_clouds.append(('OVC', cloud_info['vertical_visibility']))
                         
                    condition = map_metar_weather_to_condition(weather_codes, combined_clouds, is_day)
                    flight_cat = get_flight_category(raw_metar)
                    
                    # Pressure Tendency Logic (from Remarks)
                    remarks = raw_metar.split('RMK')[1] if 'RMK' in raw_metar else ""
                    tendency_pattern = re.search(r'\b5([0-8])(\d{3})\b', remarks)
                    if tendency_pattern:
                        char = int(tendency_pattern.group(1))
                        if 0 <= char <= 3: press_arrow = "↗"
                        elif char == 4: press_arrow = "→"
                        elif 5 <= char <= 8: press_arrow = "↘"

            parsed_data[station_id] = {
                "name": station.get("name", station_id),
                "lat": station.get("lat"),
                "lon": station.get("lon"),
                "temp_f": temp_f, 
                "dew_point_f": dew_point_f,
                "wind_dir": wind_dir, 
                "wind_spd": wind_spd,
                "alt_inHg": alt_inHg, 
                "press_arrow": press_arrow,
                "condition": condition,
                "flight_cat": flight_cat
            }

        except Exception as e:
            continue
            
    return parsed_data

async def fetch_state_data(session, state_code):
    """
    Helper to fetch a single state's data from IEM.
    """
    iem_url = "https://mesonet.agron.iastate.edu/api/1/currents.json"
    try:
        # Fetch just one state at a time to be safe
        async with session.get(iem_url, params={'state': state_code}, timeout=20) as response:
            if response.status == 200:
                return await response.json()
            else:
                logging.warning(f"Failed to fetch {state_code}: HTTP {response.status}")
                return None
    except Exception as e:
        logging.warning(f"Error fetching {state_code}: {e}")
        return None

async def get_all_georgia_weather_data_iem():
    """
    Fetches GA + Surrounding States concurrently.
    Aggregates all results into a single data dictionary using process_iem_data.
    """
    # The list of states to fetch
    states = ['GA', 'AL', 'FL', 'SC', 'NC', 'TN']
    
    # Placeholder for the combined data
    combined_json = {'data': []}
    
    logging.info(f"Starting multi-state fetch for: {states}")
    
    async with aiohttp.ClientSession() as session:
        # 1. Create a task for every state
        tasks = [fetch_state_data(session, state) for state in states]
        
        # 2. Run them all at once (Parallel)
        results = await asyncio.gather(*tasks)
        
        # 3. Combine the results
        for i, result in enumerate(results):
            if result and 'data' in result:
                combined_json['data'].extend(result['data'])
            else:
                logging.warning(f"No data returned for {states[i]}")

    total_stations = len(combined_json['data'])
    logging.info(f"Total stations fetched across all states: {total_stations}")
    
    # 4. Process the massive combined list using the function defined above
    return process_iem_data(combined_json)

# --- Helper Functions (non-METAR) ---
def extract_cloud_info(metar):
    """
    Extracts cloud layers from a METAR string.
    FIXED: Removed double escaping on regex \\d which caused cloud layers to be ignored.
    """
    cloud_levels = {"low": [], "mid": [], "high": [], "vertical_visibility": None}
    main_body = metar.split('RMK')[0]
    
    # --- FIX: Changed \\d to \d ---
    # In a raw string (r''), \d matches a digit. \\d matches a literal backslash + d.
    cloud_pattern = re.compile(r'(FEW|SCT|BKN|OVC)(\d{3})|VV(\d{3})')
    
    cloud_matches = re.findall(cloud_pattern, main_body)
    
    for match in cloud_matches:
        # Match 0: Type (FEW/SCT/BKN/OVC), Match 1: Height (Digits)
        if match[0] and match[1]:
            cover = match[0]
            try:
                altitude_hundreds = int(match[1])
                altitude_ft = altitude_hundreds * 100
                
                if altitude_ft <= 6500:
                    cloud_levels["low"].append((cover, altitude_ft))
                elif 6500 < altitude_ft <= 20000:
                    cloud_levels["mid"].append((cover, altitude_ft))
                else:
                    cloud_levels["high"].append((cover, altitude_ft))
            except ValueError:
                continue

        # Match 2: Vertical Visibility (VV) -> Treat as Low Cloud / Obscuration
        if match[2]:
            try:
                vv_hundreds = int(match[2])
                cloud_levels["vertical_visibility"] = vv_hundreds * 100
            except ValueError:
                continue
                
    return cloud_levels

def extract_weather_phenomena(metar):
    """Extracts weather phenomena codes (like 'TS' or 'RA') from a METAR."""
    main_body = metar.split('RMK')[0]
    
    # Regex to catch standard weather codes (Qualifiers + Phenomena)
    weather_pattern = re.compile(
        r'^(-|\+|VC)?(TS|SH|FZ|DR|BL|MI|BC|PR)?(DZ|RA|SN|SG|IC|PL|GR|GS|UP)?(BR|FG|FU|VA|DU|SA|HZ|PY)?(PO|SQ|FC|SS|DS)?$'
    )
    
    parts = main_body.split()
    weather_conditions = []
    
    for part in parts:
        # Strip maintenance indicator ($) if present
        clean_part = part.rstrip('$')

        # 1. CRITICAL FIX: Skip Station IDs (e.g., KFFC, KAGS, KCSG)
        # Prevents KFFC from reading as "FC" (Tornado)
        # Prevents KAGS from reading as "GS" (Hail)
        if re.match(r'^K[A-Z0-9]{3}$', clean_part):
            continue

        # 2. Skip standard non-weather parts
        if re.match(r'^(FEW|SCT|BKN|OVC|VV)\d{3}$', clean_part) or \
           re.match(r'^\d{4}$', clean_part) or \
           re.match(r'^A\d{4}$', clean_part) or \
           re.match(r'^Q\d{4}$', clean_part) or \
           re.match(r'^\d{5}KT$', clean_part) or \
           re.match(r'^\d{5}G\d{2}KT$', clean_part) or \
           re.match(r'^\d{5}\d{2}$', clean_part) or \
           re.match(r'^\d{2}/\d{2}$', clean_part) or \
           re.match(r'^\d{2}/M\d{2}$', clean_part) or \
           re.match(r'^M\d{2}/\d{2}$', clean_part) or \
           re.match(r'^M\d{2}/M\d{2}$', clean_part) or \
           re.match(r'^RMK', clean_part) or \
           re.match(r'^NOSPECI', clean_part) or \
           re.match(r'^AUTO', clean_part) or \
           re.match(r'^COR', clean_part):
            continue
            
        # 3. Check for valid weather code
        # We use re.match to ensure it matches the WHOLE string, not just a substring
        match = weather_pattern.match(clean_part)
        if match:
            # Filter out empty matches (e.g. if the regex matches an empty string)
            if clean_part in ['-', '+', 'VC']: continue 
            weather_conditions.append(clean_part)

    return weather_conditions

# --- UPDATED MAPPING FUNCTION ---
def map_metar_weather_to_condition(weather_codes, cloud_layers, is_day):
    """
    Priority: Tornado > Hail > Ice/FZ > Snow > Storm > Rain > Fog > Clouds
    """
    raw_codes = " ".join(weather_codes)
    
    # 1. TORNADO / FUNNEL
    if 'FC' in raw_codes or '+FC' in raw_codes: return 'Tornado'

    # 2. HAIL
    if 'GR' in raw_codes or 'GS' in raw_codes: return 'Hail'
    
    # 3. WINTER PRECIP (Explicit Types)
    if 'PL' in raw_codes: return 'Ice Pellets' # Sleet
    if 'FZRA' in raw_codes or 'FZDZ' in raw_codes: return 'Freezing Rain'
    if 'ZR' in raw_codes: return 'Freezing Rain'
    
    # Mix Logic: If Snow AND Rain appear
    if 'SN' in raw_codes and 'RA' in raw_codes: return 'Wintry Mix'
    
    if 'SN' in raw_codes or 'SG' in raw_codes: return 'Snow'
    if 'IC' in raw_codes: return 'Ice Pellets'

    # 4. THUNDERSTORMS
    if 'TS' in raw_codes:
        if '+TS' in raw_codes or '+TSRA' in raw_codes: return 'Severe Storms'
        return 'Storms'

    # 5. RAIN
    if 'RA' in raw_codes or 'SHRA' in raw_codes:
        if '+RA' in raw_codes: return 'Heavy Rain'
        return 'Rain'
    
    # 6. OBSCURATION
    if any(c in raw_codes for c in ['FG', 'BR', 'HZ', 'FU', 'VA', 'DZ']): return 'Fog'

    # 7. SKY CONDITION
    if any(c == 'OVC' for c, _ in cloud_layers): return 'Overcast'
    if any(c == 'BKN' for c, _ in cloud_layers): return 'Cloudy' 
    
    valid_layers = [c for c, _ in cloud_layers if c in ['FEW', 'SCT']]
    if len(valid_layers) >= 2: return 'Partly Cloudy (Sun)' if is_day else 'Partly Cloudy (Moon)'
    elif len(valid_layers) == 1:
        if valid_layers[0] == 'SCT': return 'Partly Cloudy (Sun)' if is_day else 'Partly Cloudy (Moon)'
        return 'Mostly Sunny' if is_day else 'Moonlight'
            
    return 'Sunny' if is_day else 'Moonlight'

def is_daytime(lat, lon):
    """Determines if it's daytime based on sunrise/sunset times."""
    location = LocationInfo(latitude=lat, longitude=lon, timezone='UTC')
    s = sun(location.observer, date=datetime.now(timezone.utc).date())
    now = datetime.now(timezone.utc)
    return s['sunrise'] < now < s['sunset']

def load_nws_zone_lookup(state="GA"):
    """
    Fetches and caches NWS Zones (UGC) using the Iowa Environmental Mesonet (IEM).
    This is critical for Fire Weather Watches which cover many counties.
    """
    cache_file = f"nws_zones_{state}_iem.pkl"
    cached_data = load_pickle_cache(cache_file, 86400 * 30) # Cache for 30 days
    
    if cached_data:
        logging.info(f"Using cached NWS zones ({len(cached_data)} loaded).")
        return cached_data
    
    logging.info("Fetching bulk NWS Zones from Iowa Mesonet...")
    zone_lookup = {}
    
    # IEM provides a single clean GeoJSON for all zones in the state
    url = f"https://mesonet.agron.iastate.edu/geojson/ugc.py?state={state}"
    
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            count = 0
            for feature in data.get('features', []):
                try:
                    props = feature.get('properties', {})
                    code = props.get('code') # e.g., GAZ001
                    geo = feature.get('geometry')
                    
                    if code and geo:
                        if geo['type'] == 'Polygon':
                            poly = Polygon(geo['coordinates'][0])
                            zone_lookup[code] = poly
                            count += 1
                        elif geo['type'] == 'MultiPolygon':
                            polys = [Polygon(p[0]) for p in geo['coordinates']]
                            zone_lookup[code] = MultiPolygon(polys)
                            count += 1
                except: continue
            
            logging.info(f"Loaded {count} zones from IEM.")
            save_pickle_cache(cache_file, zone_lookup)
            return zone_lookup
            
    except Exception as e:
        logging.error(f"Failed to fetch IEM zones: {e}")

    return {}

def get_nws_alerts(state, nws_zone_lookup):
    """
    Fetches active NWS alerts. 
    INCLUDES 'LAZY LOADING': If a Zone ID (GAZxxx) is in an alert but missing 
    from our lookup, we fetch it individually on the fly.
    """
    url = 'https://api.weather.gov/alerts/active?area=' + state
    
    # --- Helper to fetch missing zones on demand (With Retries) ---
    def fetch_single_zone(zone_id):
        z_url = f"https://api.weather.gov/zones/forecast/{zone_id}"
        # Retry up to 3 times
        for attempt in range(1, 4):
            try:
                # Increased timeout to 15 seconds
                r = requests.get(z_url, headers={"User-Agent": "(my_weather_map, contact@email.com)"}, timeout=15)
                if r.status_code == 200:
                    geo = r.json().get('geometry')
                    if geo:
                        if geo['type'] == 'Polygon':
                            return Polygon(geo['coordinates'][0])
                        elif geo['type'] == 'MultiPolygon':
                            return MultiPolygon([Polygon(p[0]) for p in geo['coordinates']])
                    return None # Success but empty geometry
                elif r.status_code == 429:
                    # Rate limit hit, wait a bit
                    time.sleep(1)
            except Exception as e:
                # Log warning only on last attempt
                if attempt == 3:
                    logging.warning(f"Could not fetch zone {zone_id} after 3 attempts: {e}")
                time.sleep(0.5) # Short pause before retry
        return None
    # -----------------------------------------------

    try:
        logging.info(f"Fetching active alerts from {url}...")
        response = requests.get(url, timeout=20)
        data = response.json()
        features = data.get('features', [])
        alerts = []
        
        dir_map = {'north': 0, 'northeast': 45, 'east': 90, 'southeast': 135, 'south': 180, 'southwest': 225, 'west': 270, 'northwest': 315}
        regex_strict = re.compile(r'From (\d+) degrees? .+? at (\d+) KTS? .+? (\d+ \d+)')
        regex_natural = re.compile(r'(north|south|east|west|northeast|northwest|southeast|southwest) at (\d+) MPH')
        
        for feature in features:
            try:
                props = feature.get('properties', {})
                event = props.get("event", "Unknown Event")
                severity = props.get("severity", "Unknown")
                description = props.get("description", "").replace("\n", " ") 
                headline = props.get("headline", "")
                
                polygons = [] 
                api_geometry = feature.get('geometry')
                
                # 1. Prefer Explicit Polygon (Warnings)
                if api_geometry is not None:
                    coords_list = []
                    if api_geometry['type'] == 'Polygon':
                        coords_list = [api_geometry['coordinates'][0]]
                    elif api_geometry['type'] == 'MultiPolygon':
                        coords_list = [poly[0] for poly in api_geometry['coordinates']]
                    for coords in coords_list:
                        polygons.append([(lon, lat) for lon, lat in coords])
                
                # 2. Fallback to Zone Lookup (Watches/Advisories/SPS)
                else:
                    zone_codes = props.get('geocode', {}).get('UGC', [])
                    zone_geometries = []
                    
                    for code in zone_codes:
                        # --- LAZY LOAD MISSING ZONES ---
                        if code not in nws_zone_lookup:
                            logging.info(f"Zone {code} missing from cache. Fetching on-demand...")
                            poly = fetch_single_zone(code)
                            if poly:
                                nws_zone_lookup[code] = poly # Update the dict in memory
                        # -------------------------------
                        
                        if code in nws_zone_lookup:
                            zone_geometries.append(nws_zone_lookup[code])
                    
                    if zone_geometries:
                        combined_geometry = unary_union(zone_geometries)
                        if combined_geometry.geom_type == 'Polygon':
                            polygons = [list(combined_geometry.exterior.coords)]
                        elif combined_geometry.geom_type == 'MultiPolygon':
                            polygons = [list(p.exterior.coords) for p in combined_geometry.geoms]

                if not polygons:
                    continue

                # --- TRACKING (Centroid/Regex) ---
                storm_track = None
                match_strict = regex_strict.search(description)
                if match_strict:
                    try:
                        dir_from = int(match_strict.group(1))
                        speed_kts = int(match_strict.group(2))
                        coords_raw = match_strict.group(3).strip().split()
                        if len(coords_raw) >= 2:
                            lat_raw = int(coords_raw[0]); lon_raw = int(coords_raw[1])
                            start_lat = lat_raw / 100.0; start_lon = (lon_raw / 100.0) * -1.0
                            storm_track = {"start": (start_lon, start_lat), "dir_from": dir_from, "speed_kts": speed_kts}
                    except: pass

                if not storm_track:
                    match_nat = regex_natural.search(description)
                    if match_nat:
                        try:
                            direction_str = match_nat.group(1).lower()
                            speed_mph = int(match_nat.group(2))
                            speed_kts = int(speed_mph / 1.15)
                            dir_towards = dir_map.get(direction_str, 0)
                            dir_from = (dir_towards + 180) % 360
                            if len(polygons) > 0:
                                poly_obj = Polygon(polygons[0])
                                centroid = poly_obj.centroid
                                storm_track = {"start": (centroid.x, centroid.y), "dir_from": dir_from, "speed_kts": speed_kts}
                        except: pass

                # --- COLOR LOGIC ---
                is_fire_event = "Fire" in event or "Red Flag" in event or "Fire Danger" in headline or "Fire Danger" in description
                
                if event in ["Tornado Warning", "Hurricane Warning", "Tsunami Warning"]: color = ALERT_COLORS["Extreme"]
                elif "Severe" in event or "Hurricane" in event: color = ALERT_COLORS["Severe"]
                elif "Flood" in event or "Winter Storm" in event:
                    if "Warning" in event: color = WINTER_ALERT_COLORS["Warning"] if "Winter" in event else ALERT_COLORS["Moderate"]
                    elif "Watch" in event: color = WINTER_ALERT_COLORS["Watch"] if "Winter" in event else ALERT_COLORS["Moderate"]
                    else: color = WINTER_ALERT_COLORS["Advisory"] if "Winter" in event else ALERT_COLORS["Minor"]
                elif event in TEMP_ALERT_COLORS: color = TEMP_ALERT_COLORS[event]
                elif is_fire_event: color = ALERT_COLORS["Moderate"] 
                else: color = ALERT_COLORS.get(severity, ALERT_COLORS["Unknown"])
                
                alerts.append({
                    "headline": headline, "event": event, "severity": severity,
                    "polygons": polygons, "color": color,
                    "affected_areas": props.get("areaDesc", "Unknown Area").replace(';', ','),
                    "storm_track": storm_track 
                })
            
            except Exception as e:
                continue
        
        logging.info(f"Processed {len(alerts)} alerts.")
        return alerts
        
    except requests.RequestException as e:
        logging.error(f"Error fetching NWS alerts: {e}")
        return []

def fetch_and_plot_spc_outlooks(ax, clip_patch=None):
    """
    Fetches SPC Day 1 Convective and Fire Weather Outlooks.
    """
    logging.info("Fetching SPC Outlooks (Convective & Fire)...")
    
    active_labels = set() 

    SPC_COLORS = {
        "TSTM": "#C1E9C1", "MRGL": "#006400", "SLGT": "#FFD700",
        "ENH":  "#FFA500", "MDT":  "#FF0000", "HIGH": "#FF00FF",
        "ELEV": "#FFB90F", "CRIT": "#FF0000", "EXT":  "#FF00FF",
        "IDRT": "#A0522D", "SDRT": "#CD853F"
    }

    # *** FIX: Added '_cat' to the fire URL to get the actual risk areas ***
    endpoints = [
        ("https://www.spc.noaa.gov/products/outlook/day1otlk_cat.lyr.geojson", "Convective"),
        ("https://www.spc.noaa.gov/products/fire_wx/day1firewx_cat.lyr.geojson", "Fire") 
    ]

    count = 0
    for url, cat_type in endpoints:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code != 200: continue
            
            data = r.json()
            for feat in data.get('features', []):
                props = feat.get('properties', {})
                label = props.get('LABEL') 
                if not label: continue
                
                geom = feat.get('geometry')
                if not geom: continue
                
                active_labels.add(label)
                color = SPC_COLORS.get(label, 'gray')
                lw = 2.5 if label in ['ENH', 'MDT', 'HIGH', 'CRIT', 'EXT', 'ELEV'] else 1.5
                
                coords_list = []
                if geom['type'] == 'Polygon':
                    coords_list = [geom['coordinates'][0]]
                elif geom['type'] == 'MultiPolygon':
                    coords_list = [p[0] for p in geom['coordinates']]
                
                for coords in coords_list:
                    poly_points = [(x, y) for x, y in coords]
                    poly_patch = mpatches.Polygon(
                        poly_points, closed=True, transform=ccrs.PlateCarree(),
                        facecolor='none', edgecolor=color, hatch='///',
                        linewidth=lw, linestyle='-', zorder=4.5, label=f"SPC {label}"
                    )
                    ax.add_patch(poly_patch)
                    if clip_patch: poly_patch.set_clip_path(clip_patch)
                    count += 1
                    
        except Exception as e:
            logging.warning(f"Error fetching SPC {cat_type}: {e}")

    return active_labels

# --- Text Collision Resolver ---
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

def fetch_hrrr_ptype(extent):
    """
    Fetches HRRR 'Categorical' precipitation types.
    Interprets vertical columns of air to distinguish Snow, Sleet, and Freezing Rain.
    """
    logging.info("Fetching HRRR Precipitation Type Data (Vertical Column Analysis)...")
    try:
        # Buffer the extent slightly to ensure we cover the edges
        buffer = 0.5
        bbox = [extent[2]-buffer, extent[3]+buffer, extent[0]-buffer, extent[1]+buffer]

        # Connect to HRRR 2.5km Surface Analysis via UCAR THREDDS
        # We use the 'latest' catalog to get the most recent run
        cat_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/HRRR/CONUS_2p5km/latest.xml'
        cat = TDSCatalog(cat_url)
        ds = cat.datasets[0] # Get latest run
        ncss = ds.subset()
        
        # We need these specific 'Categorical' variables
        # The model outputs 1.0 (Yes) or 0.0 (No) for each pixel
        query = ncss.query()
        query.lonlat_box(north=bbox[1], south=bbox[0], east=bbox[3], west=bbox[2])
        query.accept('netcdf4')
        query.add_lonlat(True)
        
        # Variable names (Standard NCEP names)
        # Note: If these change, use the fuzzy matcher logic from fetch_rtma_data
        vars_to_fetch = [
            "Categorical_snow_surface",
            "Categorical_ice_pellets_surface",
            "Categorical_freezing_rain_surface",
            "Categorical_rain_surface"
        ]
        
        # Validate variables exist in the dataset to prevent 400 Errors
        available = set(ncss.variables)
        valid_vars = [v for v in vars_to_fetch if v in available]
        
        if not valid_vars:
            logging.warning("HRRR P-Type variables not found on server.")
            return None, None, None, None, None

        query.variables(*valid_vars)
        data = ncss.get_data(query)

        # Parse Lat/Lon
        lat_var = next(v for v in data.variables if 'lat' in v.lower())
        lon_var = next(v for v in data.variables if 'lon' in v.lower())
        lats = data.variables[lat_var][:]
        lons = data.variables[lon_var][:]
        
        # Helper to safely get data or return zeros
        def get_grid(name):
            if name in data.variables:
                return data.variables[name][:].squeeze()
            return np.zeros_like(lats)

        cat_snow = get_grid("Categorical_snow_surface")
        cat_ice  = get_grid("Categorical_ice_pellets_surface")
        cat_fzra = get_grid("Categorical_freezing_rain_surface")
        
        logging.info("HRRR P-Type data fetched successfully.")
        return lats, lons, cat_snow, cat_ice, cat_fzra

    except Exception as e:
        logging.error(f"HRRR Fetch Failed: {e}")
        return None, None, None, None, None

def fetch_rtma_data(extent):
    """
    Fetches high-res (2.5km) RTMA data. 
    Includes DYNAMIC VARIABLE LOOKUP with protection against 'Error' vars.
    """
    logging.info("Fetching RTMA High-Res Model Data...")
    try:
        buffer = 0.5
        bbox = [extent[2]-buffer, extent[3]+buffer, extent[0]-buffer, extent[1]+buffer]
        
        # Connect to UCAR THREDDS
        cat_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RTMA/CONUS_2p5km/latest.xml'
        cat = TDSCatalog(cat_url)
        ds = cat.datasets[0]
        ncss = ds.subset()
        
        # --- NEW: Dynamic Variable Finder ---
        available_vars = set(ncss.variables)
        
        def get_best_match(keywords, fallback):
            # 1. Try exact matches from our known list
            for kw in keywords:
                if kw in available_vars: return kw
            
            # 2. Search for partial matches
            for var_name in available_vars:
                # *** CRITICAL FIX: Skip Error/Uncertainty variables ***
                if 'error' in var_name.lower() or 'uncertainty' in var_name.lower():
                    continue
                
                # Check if keywords match
                if all(k in var_name for k in keywords[0].split('_') if len(k)>3):
                    return var_name
            return fallback

        # Define preferred names
        temp_var = get_best_match(
            ['Temperature_height_above_ground', 'Temperature_surface', 'tmp2m'], 
            'Temperature_height_above_ground'
        )
        dew_var = get_best_match(
            ['Dewpoint_temperature_height_above_ground', 'Dewpoint_temperature_surface', 'dpt2m', 'Dewpoint_temperature'], 
            'Dewpoint_temperature_height_above_ground'
        )
        u_var = get_best_match(
            ['u-component_of_wind_height_above_ground', 'u-component_of_wind_isobaric', 'u-component_of_wind'],
            'u-component_of_wind_height_above_ground'
        )
        v_var = get_best_match(
            ['v-component_of_wind_height_above_ground', 'v-component_of_wind_isobaric', 'v-component_of_wind'],
            'v-component_of_wind_height_above_ground'
        )
        
        logging.info(f"RTMA Variables Selected: {temp_var}, {dew_var}")

        # Construct Query
        query = ncss.query()
        query.lonlat_box(north=bbox[1], south=bbox[0], east=bbox[3], west=bbox[2])
        query.accept('netcdf4')
        query.add_lonlat(True)
        query.variables(temp_var, dew_var, u_var, v_var)
        
        data = ncss.get_data(query)
        
        # Parse Coordinates
        lat_var = next((v for v in data.variables if 'lat' in v.lower()), None)
        lon_var = next((v for v in data.variables if 'lon' in v.lower()), None)
        lat = data.variables[lat_var][:]
        lon = data.variables[lon_var][:]
        
        # Parse Data
        temp_k = data.variables[temp_var][:].squeeze()
        dew_k = data.variables[dew_var][:].squeeze()
        u_ms = data.variables[u_var][:].squeeze()
        v_ms = data.variables[v_var][:].squeeze()
        
        # Convert Units
        temp_f = (temp_k - 273.15) * 9/5 + 32
        dew_f = (dew_k - 273.15) * 9/5 + 32
        u_kts = u_ms * 1.94384
        v_kts = v_ms * 1.94384
        
        logging.info(f"RTMA Fetch Success. Grid shape: {temp_f.shape}")
        return lat, lon, temp_f, dew_f, u_kts, v_kts

    except Exception as e:
        logging.error(f"RTMA Fetch Failed: {e}")
        return None, None, None, None, None, None
    
    # ==========================================
# --- NEW: NAVY / MARINE MODULE ---
# ==========================================

def calc_marine_physics(water_temp_k, water_u, water_v, air_temp_f, air_dew_f, wind_speed_kts):
    """
    Calculates Latent Heat Flux potential and Ocean Vorticity (Upwelling).
    FIXED: Correctly extracts partial derivatives for vorticity math.
    """
    # 1. Vorticity Calculation (dv/dx - du/dy)
    # np.gradient returns [gradient_y, gradient_x] for a 2D array
    
    # Get gradients of V (Northward current)
    grad_v = np.gradient(water_v)
    dv_dx = grad_v[1] # The change in V as you move East
    
    # Get gradients of U (Eastward current)
    grad_u = np.gradient(water_u)
    du_dy = grad_u[0] # The change in U as you move North
    
    # Vorticity = dv/dx - du/dy
    # Positive (Red) = Cyclonic (Counter-Clockwise) -> Potential Upwelling
    vorticity = dv_dx - du_dy

    # 2. Latent Heat Flux Potential (The "Energy Suck")
    # Convert Water to Celsius (assuming input is Kelvin)
    water_c = water_temp_k - 273.15
    
    # Convert Air Temp to Celsius
    air_c = (air_temp_f - 32) * 5/9
    
    # Simple Bulk Formula approximation: (WaterT - AirT) * WindSpeed
    # This shows where cold dry air is ripping heat off warm water
    # 0.514 converts Knots to m/s
    flux_potential = (water_c - air_c) * (wind_speed_kts * 0.514) 
    
    return vorticity, flux_potential, water_c
    
    return vorticity, flux_potential, water_c

def fetch_ncom_data(extent):
    """
    Fetches MARINE Data via NOAA RTOFS (Real-Time Ocean Forecast System).
    Replaces the retired Navy NCOM model.
    FIXED: Uses decode_times=False to bypass 'Ambiguous reference date' errors.
    FIXED: Handles Longitude wrapping (0-360 vs -180-180).
    FIXED: Uses .squeeze() to flatten dimensions for plotting.
    """
    print("\n--- MARINE DATA FETCH DEBUG (RTOFS) ---")
    logging.info("Attempting to fetch RTOFS (Global Ocean) data...")
    
    # 1. Setup Dates (Today -> Yesterday)
    today = datetime.now(timezone.utc)
    dates_to_try = [today, today - timedelta(days=1)]
    
    # 2. RTOFS OPeNDAP URL Pattern
    base_url_pattern = "https://nomads.ncep.noaa.gov/dods/rtofs/rtofs_global{date_str}/rtofs_glo_2ds_forecast_3hrly_prog"

    for date_obj in dates_to_try:
        date_str = date_obj.strftime('%Y%m%d')
        url = base_url_pattern.format(date_str=date_str)
        print(f"• Trying RTOFS for {date_str}...", end=" ")
        
        try:
            # FIX 1: decode_times=False prevents the "Ambiguous reference date" crash
            ds = xr.open_dataset(url, decode_times=False)
            
            # FIX 2: Handle Longitude Wrapping (Georgia is -80, RTOFS might be 280)
            lat_min, lat_max = extent[2] - 1.0, extent[3] + 1.0
            lon_min, lon_max = extent[0] - 1.0, extent[1] + 1.0
            
            # Check if dataset uses 0-360 Longitude format
            if ds.lon.min() >= 0:
                lon_min = (lon_min + 360) % 360
                lon_max = (lon_max + 360) % 360
                if lon_min > lon_max: lon_min, lon_max = lon_max, lon_min
            
            # FIX 3: Use .isel(time=0) to grab the first frame by INDEX
            ds_sub = ds.isel(time=0).sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            
            # 4. Extract Variables
            if 'sst' not in ds_sub:
                print("Missing 'sst' variable. ❌")
                continue
                
            lats = ds_sub.lat.values
            lons = ds_sub.lon.values
            
            # Normalize Longitudes back to -180 to 180 for plotting consistency
            lons = np.where(lons > 180, lons - 360, lons)
            
            # FIX 4: .squeeze() removes the extra dimension (1, 91, 97) -> (91, 97)
            w_temp = ds_sub['sst'].values.squeeze()
            
            # Convert Celsius to Kelvin if needed
            if np.nanmax(w_temp) < 100:
                w_temp = w_temp + 273.15
                
            w_u = ds_sub['u_velocity'].values.squeeze()
            w_v = ds_sub['v_velocity'].values.squeeze()
            
            if 'sss' in ds_sub:
                salinity = ds_sub['sss'].values.squeeze()
            else:
                salinity = np.zeros_like(w_temp)
            
            print(f"Success! ✅ (Shape: {w_temp.shape})")
            logging.info(f"RTOFS Success for {date_str}")
            return lats, lons, w_temp, w_u, w_v, salinity

        except Exception as e:
            print(f"Failed. ❌")
            continue

    print("❌ ALL RTOFS ATTEMPTS FAILED. No Marine layer will be plotted.\n")
    logging.error("RTOFS Fetch failed after trying all dates.")
    return None, None, None, None, None, None

# --- Main Map Generation Task ---
def fetch_rap_data(extent):
    """
    Fetches MSLP from the RAP model.
    Robustly handles variable name changes and forces Lat/Lon generation.
    """
    logging.info("Fetching RAP model data for Isobars...")
    try:
        buffer = 3.0
        bbox = [extent[2]-buffer, extent[3]+buffer, extent[0]-buffer, extent[1]+buffer] 
        
        cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RAP/CONUS_13km/latest.xml')
        ds = cat.datasets[0] 
        ncss = ds.subset()
        
        # 1. SMART PRESSURE DETECTION
        available_vars = ncss.variables
        candidates = ['MSLP_MAPS_System_Reduction_msl', 'Pressure_reduced_to_MSL_msl', 'MSL_pressure_surface', 'Pressure_msl']
        var_name = next((c for c in candidates if c in available_vars), None)
        
        if var_name is None:
            logging.error(f"RAP Fetch Failed: No MSLP variable found in {available_vars}")
            return None, None, None

        logging.info(f"Using RAP variable: {var_name}")

        # 2. QUERY
        query = ncss.query()
        query.lonlat_box(north=bbox[1], south=bbox[0], east=bbox[3], west=bbox[2])
        query.accept('netcdf4')
        query.variables(var_name)
        
        # --- FIX: FORCE SERVER TO SEND LAT/LON ARRAYS ---
        query.add_lonlat(True) 
        # ------------------------------------------------
        
        data = ncss.get_data(query)
        
        # 3. SMART COORDINATE DETECTION
        # Look for the new 2D coordinates that add_lonlat(True) generates
        # Usually named 'lat', 'latitude', 'grid_lat', etc.
        lat_var = next((v for v in data.variables if 'lat' in v.lower()), None)
        lon_var = next((v for v in data.variables if 'lon' in v.lower()), None)
        
        if not lat_var or not lon_var:
             logging.error(f"RAP Fetch Failed: Could not identify lat/lon variables even with add_lonlat. Available: {list(data.variables.keys())}")
             return None, None, None

        # Parse Data
        p_pa = data.variables[var_name][:].squeeze()
        lat = data.variables[lat_var][:]
        lon = data.variables[lon_var][:]
        
        # Convert Pa -> mb
        p_mb = p_pa / 100.0
        p_mb = gaussian_filter(p_mb, sigma=1.0)
        
        logging.info(f"RAP data success. Shape: {p_mb.shape}")
        return lat, lon, p_mb

    except Exception as e:
        logging.error(f"Failed to fetch RAP data: {e}")
        return None, None, None

def fetch_and_plot_wpc_pressure_centers(ax, rap_data=None):
    """
    Fetches NWS High/Low Pressure Centers via ArcGIS API.
    - Uses custom PNG icons for 'H' and 'L'.
    - Calculates pressure via RAP Model sampling (Priority 1) or Text Scan (Priority 2).
    - Adds 'hPa' suffix to the label.
    """
    logging.info("Fetching WPC Pressure Centers (ArcGIS Source)...")
    
    # --- CONFIGURATION: ICON PATHS ---
    ICON_PATHS = {
        'H': "/home/desoxyn/frostbyte/frostbyte_project/icons/high.png",
        'L': "/home/desoxyn/frostbyte/frostbyte_project/icons/low.png"
    }
    ICON_ZOOM = 0.15  # Adjust this if icons are too big/small (0.1 to 0.3 usually works)

    # Use the NWS National Forecast Chart MapServer
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

    if not all_gdfs:
        return

    full_gdf = pd.concat(all_gdfs, ignore_index=True)
    
    # Auto-repair coordinates
    xmin_d, ymin_d, xmax_d, ymax_d = full_gdf.total_bounds
    if abs(xmin_d) > 180 or abs(xmax_d) > 180:
        try:
            full_gdf.crs = "EPSG:3857"
            full_gdf = full_gdf.to_crs("EPSG:4326")
        except: pass

    # Setup Map Boundaries
    xmin, xmax, ymin, ymax = MAP_EXTENT
    buffer_deg = 15.0 
    bbox_poly = Polygon([(xmin-buffer_deg, ymin-buffer_deg), (xmin-buffer_deg, ymax+buffer_deg), 
                         (xmax+buffer_deg, ymax+buffer_deg), (xmax+buffer_deg, ymin-buffer_deg)])
    full_gdf = full_gdf[full_gdf.geometry.intersects(bbox_poly)]
    
    if full_gdf.empty:
        return

    # --- BUILD MODEL LOOKUP TREE (If RAP data provided) ---
    rap_tree = None
    rap_vals = None
    if rap_data:
        try:
            r_lat, r_lon, r_p = rap_data
            if r_lat is not None and r_p is not None:
                # Flatten arrays for KDTree
                pts = np.column_stack((r_lon.ravel(), r_lat.ravel()))
                rap_vals = r_p.ravel()
                from scipy.spatial import cKDTree
                rap_tree = cKDTree(pts)
        except Exception as e:
            logging.warning(f"Failed to build RAP lookup tree: {e}")

    count = 0
    for _, row in full_gdf.iterrows():
        try:
            # 1. IDENTIFY TYPE
            row_text = str(row.to_dict()).upper()
            sys_type = None
            color = 'black'
            
            if "HIGH" in row_text or "'TYPE': 'H'" in row_text:
                sys_type, color = 'H', 'blue'
            elif "LOW" in row_text or "'TYPE': 'L'" in row_text:
                sys_type, color = 'L', 'red'
            
            if not sys_type: continue

            # 2. GEOMETRY
            geom = row.geometry
            if geom.geom_type == 'MultiPoint': geom = geom.centroid
            lon_sys, lat_sys = geom.x, geom.y

            # Clamp Logic
            plot_lon = max(xmin, min(lon_sys, xmax))
            plot_lat = max(ymin, min(lat_sys, ymax))
            
            pressure_val = "U"
            
            # 3a. PRIORITY: CALCULATE FROM MODEL (RAP)
            if rap_tree:
                dist, idx = rap_tree.query([lon_sys, lat_sys])
                if dist < 1.5:
                    pressure_val = int(round(rap_vals[idx]))
            
            # 3b. FALLBACK: TEXT SCANNER
            if pressure_val == "U":
                row_data = row.to_dict()
                if 'geometry' in row_data: del row_data['geometry']
                for col_name, val in row_data.items():
                    if 'ID' in str(col_name).upper() and 'GUID' not in str(col_name).upper(): continue
                    val_str = str(val).strip()
                    numbers = re.findall(r'\b(\d{3,4}(?:\.\d+)?)\b', val_str)
                    for num_str in numbers:
                        try:
                            f_val = float(num_str)
                            if 850 <= f_val <= 1100:
                                pressure_val = int(round(f_val))
                                break
                        except: continue
                    if pressure_val != "U": break

            # 4. PLOT (IMAGE + TEXT STACK)
            # ----------------------------
            
            # Prepare the text string
            label_text = f"{pressure_val} hPa" if pressure_val != "U" else "U"
            
            # Define Text Style
            text_area = TextArea(label_text, textprops=dict(
                color='black', fontsize=11, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
            ))
            
            # Try to load the Custom Icon
            icon_path = ICON_PATHS.get(sys_type)
            final_artist = None
            
            if icon_path and os.path.exists(icon_path):
                try:
                    # Load Image
                    img_arr = plt.imread(icon_path)
                    image_box = OffsetImage(img_arr, zoom=ICON_ZOOM)
                    
                    # Stack Image (Top) and Text (Bottom)
                    packed_box = VPacker(children=[image_box, text_area], align="center", pad=0, sep=2)
                    
                    # Create the Annotation
                    final_artist = AnnotationBbox(packed_box, (plot_lon, plot_lat), 
                                                  frameon=False, pad=0.0, zorder=16)
                except Exception as e:
                    logging.warning(f"Error loading icon {icon_path}: {e}")
            
            # Fallback: If image fails, use the old text method
            if final_artist is None:
                # Align Text based on location relative to bounds
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

        except Exception: continue
            
    logging.info(f"Plotted {count} Pressure Centers.")

def fetch_and_plot_wpc_fronts(ax):
    """
    Fetches WPC Fronts via ArcGIS API.
    Scans multiple layers (1-6) to ensure both Fronts and Troughs are caught,
    then combines them before plotting.
    """
    logging.info("Fetching WPC Fronts (ArcGIS Source)...")
    print("\n--- FETCHING FRONTS (ARCGIS API) ---")
    
    base_url = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/natl_fcst_wx_chart/MapServer"
    
    # We scan layers 1 through 6 to catch Highs, Lows, Fronts, and Troughs.
    layers_to_scan = [1, 2, 3, 4, 5, 6] 
    all_gdfs = []
    
    for layer_id in layers_to_scan:
        try:
            print(f"• Scanning Layer {layer_id}...", end=" ")
            query_url = f"{base_url}/{layer_id}/query?where=1%3D1&outFields=*&f=geojson&outSR=4326"
            
            r = requests.get(query_url, timeout=5)
            if r.status_code != 200:
                print(f"HTTP {r.status_code}. ❌")
                continue
                
            data = r.json()
            if 'features' in data and data['features']:
                gdf_layer = gpd.GeoDataFrame.from_features(data['features'])
                if not gdf_layer.empty:
                    if gdf_layer.crs is None: gdf_layer.set_crs("EPSG:4326", allow_override=True)
                    all_gdfs.append(gdf_layer)
                    print(f"Found {len(gdf_layer)} items. ✅")
                else:
                    print("Empty. ⚪")
            else:
                print("Empty. ⚪")
        except Exception: 
            print("Error. ⚠️")
            continue

    if not all_gdfs:
        print("❌ Could not find frontal data on any scanned layer.")
        return

    full_gdf = pd.concat(all_gdfs, ignore_index=True)
    
    xmin_d, ymin_d, xmax_d, ymax_d = full_gdf.total_bounds
    if abs(xmin_d) > 180 or abs(xmax_d) > 180:
        print("• ⚠️ Converting Web Mercator to Lat/Lon...", end=" ")
        try:
            full_gdf.crs = "EPSG:3857"
            full_gdf = full_gdf.to_crs("EPSG:4326")
            print("Done. ✅")
        except: pass

    try:
        xmin, xmax, ymin, ymax = MAP_EXTENT
        bbox_poly = Polygon([(xmin-10, ymin-10), (xmin-10, ymax+10), (xmax+10, ymax+10), (xmax+10, ymin-10)])
        
        gdf_clipped = full_gdf[full_gdf.geometry.intersects(bbox_poly)]
        
        if gdf_clipped.empty:
            print(f"• No fronts found inside GA area.")
            return
        
        count = 0
        def get_style(feature_type):
            ft = str(feature_type).upper()
            # --- ZORDER SET TO 15 ---
            base_style = {'zorder': 15}
            
            if 'COLD' in ft: return {**base_style, 'color': 'blue', 'lw': 2.5, 'path_effects': [ColdFront(size=6, spacing=1.0)]}
            elif 'WARM' in ft: return {**base_style, 'color': 'red', 'lw': 2.5, 'path_effects': [WarmFront(size=6, spacing=1.0)]}
            elif 'STAT' in ft: return {**base_style, 'color': 'red', 'lw': 2.5, 'path_effects': [StationaryFront(size=6, spacing=1.0)]}
            elif 'OCCL' in ft: return {**base_style, 'color': 'purple', 'lw': 2.5, 'path_effects': [OccludedFront(size=6, spacing=1.0)]}
            elif 'TROF' in ft or 'TROUGH' in ft: return {**base_style, 'color': '#FF8C00', 'lw': 2.0, 'linestyle': '--', 'dashes': (5, 5)}
            elif 'DRY' in ft: return {**base_style, 'color': 'brown', 'lw': 2.0, 'linestyle': '-', 'dashes': (5, 5)}
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
                    count += 1

        print(f"✅ Plotted {count} frontal features.\n")

    except Exception as e:
        print(f"❌ Error plotting: {e}")

def get_glossy_badge_array(text_str, height_px=40):
    """
    Generates a Glossy Black 'Hotdog Style' button sized to fit the text.
    Returns a numpy array (H, W, 3).
    """
    # 1. Estimate Width based on character count
    # Approx 12 pixels per char + padding
    width_px = max(60, int(len(text_str) * 12) + 20)
    
    # 2. Create Vertical Gradient (Top to Bottom)
    # We want the shine to run horizontally, so colors change vertically.
    # Top: White/Grey -> Middle: Black -> Bottom: Grey
    
    # Gradient Logic (Manual implementation to ensure correct orientation for OffsetImage)
    # Top Highlight
    c_top = (0.4, 0.4, 0.4)    # Dark Grey
    c_mid_upper = (0.1, 0.1, 0.1)  # Near Black
    c_mid_lower = (0.0, 0.0, 0.0)  # Pure Black
    c_bot = (0.15, 0.15, 0.15) # Bottom Reflection
    
    half_n = height_px // 2
    
    R = np.concatenate([np.linspace(c_top[0], c_mid_upper[0], half_n),
                        np.linspace(c_mid_lower[0], c_bot[0], half_n)])
    G = np.concatenate([np.linspace(c_top[1], c_mid_upper[1], half_n),
                        np.linspace(c_mid_lower[1], c_bot[1], half_n)])
    B = np.concatenate([np.linspace(c_top[2], c_mid_upper[2], half_n),
                        np.linspace(c_mid_lower[2], c_bot[2], half_n)])
    
    # Stack to (H, 1, 3) -> A single vertical column
    col_vector = np.dstack((R, G, B)) 
    
    # 3. Tile Horizontally to create the bar width
    # Result shape: (H, W, 3)
    img_array = np.tile(col_vector, (1, width_px, 1))
    
    return img_array

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
                # CHANGED ALPHA FROM 0.5 TO 1.0 TO MATCH FORECAST SCRIPT
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
        lakes_feature = cfeature.NaturalEarthFeature(
            category='physical', name='lakes', scale='10m',
            facecolor='#4682B4', edgecolor='none', alpha=0.6
        )
        # Zorder 1.6 puts it just above the Temperature Fill (1.5) but below others
        ax.add_feature(lakes_feature, zorder=1.6)
    except Exception as e:
        logging.warning(f"Error adding lakes: {e}")

    # --- 2. RIVERS (Visual + Labels) ---
    # We use ShapeReader to get attributes (names) for labeling
    try:
        shp_path = shpreader.natural_earth(resolution='10m', category='physical', name='rivers_lake_centerlines')
        reader = shpreader.Reader(shp_path)
        
        # List of rivers we specifically want to label if found
        # (Combined with scalerank check to filter out tiny creeks)
        target_rivers = [
            "Chattahoochee", "Savannah", "Flint", "Altamaha", "Ocmulgee", 
            "Oconee", "Coosa", "St. Marys", "Etowah", "Tallapoosa", "Suwannee", 
            "Tennessee", "Chattooga"
        ]
        
        count = 0
        for record in reader.records():
            geo = record.geometry
            bounds = geo.bounds # (minx, miny, maxx, maxy)
            
            # 1. Spatial Filter: Skip if completely outside map extent
            if (bounds[2] < extent[0] or bounds[0] > extent[1] or 
                bounds[3] < extent[2] or bounds[1] > extent[3]):
                continue

            # 2. Extract Info
            name = record.attributes.get('name_en', record.attributes.get('name', ''))
            scalerank = record.attributes.get('scalerank', 10)
            
            # 3. Filter: Only plot rivers with Rank <= 8 (avoids too much clutter)
            if scalerank > 8: continue

            # 4. Style Logic
            # Major rivers get thicker lines
            lw = 1.0 if scalerank <= 5 else 0.5
            color = '#4682B4' # Steel Blue
            
            # Plot the River Line
            # Zorder 1.7 ensures it sits on top of lakes and temp contours
            ax.add_geometries([geo], ccrs.PlateCarree(), 
                              facecolor='none', edgecolor=color, linewidth=lw, 
                              zorder=1.7, alpha=0.8)

            # 5. Labeling Logic
            # We only label if it's a "Target River" OR a very major river (Rank <= 4)
            # AND if we haven't labeled it too many times (rudimentary check omitted for simplicity)
            should_label = (name and (name in target_rivers or scalerank <= 4))
            
            if should_label:
                # Find a point to place the label.
                # 'interpolate(0.5)' finds the midpoint of the line segment.
                # We check if that specific point is visible on the map.
                midpoint = geo.interpolate(0.5, normalized=True)
                
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
    
def create_glossy_square_icon(hex_color, size=30):
    """
    Generates a shiny, glass-effect square icon with a black border.
    Returns a numpy array (size, size, 4) suitable for OffsetImage.
    """
    # 1. Base Color
    r, g, b = mcolors.to_rgb(hex_color)
    
    # Initialize array (RGBA)
    icon = np.zeros((size, size, 4))
    
    # 2. Fill Base Color (Solid)
    icon[:, :, 0] = r
    icon[:, :, 1] = g
    icon[:, :, 2] = b
    icon[:, :, 3] = 1.0 # Alpha
    
    # 3. Create Gradients for "Glass/Gel" Effect
    # Vertical indices
    y_indices = np.linspace(0, 1, size).reshape(-1, 1)
    
    # -- Top Half (Reflection) --
    # Fades from White (strong) to Transparent
    # We apply this to the top 45% of pixels
    horizon = int(size * 0.45)
    
    # White highlight strength (0.0 to 1.0)
    top_highlight = np.linspace(0.6, 0.1, horizon).reshape(-1, 1)
    
    # Apply white blend to RGB channels
    # NewColor = OldColor * (1 - Alpha) + White * Alpha
    for i in range(3):
        icon[:horizon, :, i] = icon[:horizon, :, i] * (1 - top_highlight) + 1.0 * top_highlight

    # -- Horizon Line (Sharp Darkening) --
    # A dark line right at the middle to simulate the "horizon" reflection
    mid_start = horizon
    mid_end = int(size * 0.55)
    icon[mid_start:mid_end, :, :3] *= 0.85 # Darken slightly
    
    # -- Bottom Half (Shadow) --
    # Fades from Base to Darker Base
    bottom_h = size - mid_end
    shadow_grad = np.linspace(0.0, 0.3, bottom_h).reshape(-1, 1)
    
    # Darken RGB channels
    for i in range(3):
        icon[mid_end:, :, i] *= (1 - shadow_grad)

    # 4. Black Border (1px)
    border_color = [0, 0, 0, 1.0]
    icon[0, :] = border_color  # Top
    icon[-1, :] = border_color # Bottom
    icon[:, 0] = border_color  # Left
    icon[:, -1] = border_color # Right
    
    return icon

def generate_georgia_weather_map():
    """Generate and save the Georgia weather map using Mesonet + RTMA + HRRR + METARs."""
    
    # Import locally to ensure availability
    from scipy.interpolate import NearestNDInterpolator
    from matplotlib.offsetbox import AnnotationBbox, TextArea, PaddedBox, HPacker, VPacker, OffsetImage
    import matplotlib.patheffects as pe

    logging.info("Generating Georgia weather map (Mesonet + RTMA + HRRR Mode)...")

    # ==========================================
    # 1. DATA FETCHING
    # ==========================================
    logging.info("Fetching Surface Data via Synoptic/IEM...")
    iem_results = asyncio.run(get_all_georgia_weather_data_iem())
    
    all_station_data = {}  
    mesonet_stations = []  
    city_ids_found = set()

    for city_name, (lat, lon, icao) in CITIES.items():
        short_id = icao[1:] if icao.startswith('K') else icao
        station_blob = iem_results.get(icao) or iem_results.get(short_id)
        if station_blob: station_blob['station_id'] = icao
        if station_blob and station_blob.get('temp_f') is not None:
            merged = station_blob.copy(); merged['name'] = city_name
            all_station_data[icao] = merged
            city_ids_found.add(icao); city_ids_found.add(short_id)

    for s_id, data in iem_results.items():
        if s_id not in city_ids_found:
            if data.get('temp_f') is not None and data.get('lat') and data.get('lon'):
                data['station_id'] = s_id
                mesonet_stations.append(data)

    # --- FETCH MODEL DATA ---
    rtma_lat, rtma_lon, rtma_temp, rtma_dew, rtma_u, rtma_v = fetch_rtma_data(MAP_EXTENT)
    rap_lat, rap_lon, rap_p = fetch_rap_data(MAP_EXTENT)
    hrrr_lat, hrrr_lon, cat_snow, cat_ice, cat_fzra = fetch_hrrr_ptype(MAP_EXTENT)

    # --- FETCH NAVY DATA ---
    ncom_lat, ncom_lon, ncom_temp_k, ncom_u, ncom_v, ncom_sal = fetch_ncom_data(MAP_EXTENT)
    
    nws_zone_lookup = load_nws_zone_lookup()
    cached_alerts = load_json_cache("alerts_ga_cache.json", CACHE_DURATION_SECONDS)
    alerts = cached_alerts if cached_alerts else get_nws_alerts('GA', nws_zone_lookup)
    if alerts and not cached_alerts: save_json_cache("alerts_ga_cache.json", alerts)

    # ==========================================
    # 1.5 DATA SANITIZATION
    # ==========================================
    STATION_INTERP_BLACKLIST = {"0500W", "C4217", "KWDR", "KVIH", "KBGE", "K9A1", "KGVL1", "K19A"} 
    SANE_TEMP_EXEMPTION_LIST = {"BRBG1", "KSSI", "CING1", "CSAG1", "NTC8679598", "KBMG1", "FPKG1", "NDBCFPKG1", "E7541"}
    
    valid_obs_for_contours = [] 
    valid_mesonet_for_plot = [] 
    
    print("\n" + "="*60); print("       STATION DATA DEBUG LOG"); print("="*60)

    if rtma_temp is not None:
        rtma_points = np.column_stack((rtma_lon.flatten(), rtma_lat.flatten()))
        rtma_temps_flat = rtma_temp.flatten()
        rtma_validator = NearestNDInterpolator(rtma_points, rtma_temps_flat)
        DEVIATION_LIMIT = 5.0
        
        for icao, data in all_station_data.items():
            t = data.get('temp_f')
            sid = data.get('station_id', icao)
            if sid in STATION_INTERP_BLACKLIST: continue
            model_t = rtma_validator(data['lon'], data['lat'])
            if abs(t - model_t) > 15.0: print(f"⚠️  CITY WARNING: {sid} is off model. Keeping.")
            valid_obs_for_contours.append([data['lon'], data['lat'], t])

        blocked_count = 0
        for data in mesonet_stations:
            t = data.get('temp_f'); sid = data.get('station_id')
            if sid in STATION_INTERP_BLACKLIST: blocked_count += 1; continue
            if sid in SANE_TEMP_EXEMPTION_LIST:
                valid_obs_for_contours.append([data['lon'], data['lat'], t])
                valid_mesonet_for_plot.append(data); continue
            model_t = rtma_validator(data['lon'], data['lat'])
            if abs(t - model_t) > DEVIATION_LIMIT: blocked_count += 1; continue
            valid_obs_for_contours.append([data['lon'], data['lat'], t])
            valid_mesonet_for_plot.append(data)
        print(f"Blocked {blocked_count} stations."); print("="*60 + "\n")
    else:
        print("⚠️  RTMA MISSING. Using Simple Range.")
        for data in mesonet_stations:
            t = data.get('temp_f')
            if -20 < t < 120:
                valid_obs_for_contours.append([data['lon'], data['lat'], t])
                valid_mesonet_for_plot.append(data)

    # ==========================================
    # 2. GRID SETUP
    # ==========================================
    grid_z, grid_u, grid_v, grid_x, grid_y = None, None, None, None, None
    grid_adv = None
    
    if rtma_temp is not None:
        grid_x = rtma_lon; grid_y = rtma_lat; base_grid = rtma_temp 
        if valid_obs_for_contours:
            obs_array = np.array(valid_obs_for_contours)
            obs_points = np.column_stack((obs_array[:, 0], obs_array[:, 1]))
            obs_vals = obs_array[:, 2]
            if len(obs_points) > 5:
                rtma_vals = NearestNDInterpolator(list(zip(grid_x.flatten(), grid_y.flatten())), base_grid.flatten())(obs_points)
                correction = griddata(obs_points, obs_vals - rtma_vals, (grid_x, grid_y), method='linear', fill_value=0)
                if np.any(np.isnan(correction)): correction[np.isnan(correction)] = 0
                correction = gaussian_filter(correction, sigma=9.0)
                grid_z = gaussian_filter(base_grid + correction, sigma=0.5)
            else: grid_z = gaussian_filter(base_grid, sigma=0.5)
        grid_u = gaussian_filter(rtma_u, sigma=0.5); grid_v = gaussian_filter(rtma_v, sigma=0.5)
    else:
        if valid_obs_for_contours and len(valid_obs_for_contours) > 10:
            try:
                obs_array = np.array(valid_obs_for_contours)
                grid_lon = np.linspace(MAP_EXTENT[0]-1, MAP_EXTENT[1]+1, 200)
                grid_lat = np.linspace(MAP_EXTENT[2]-1, MAP_EXTENT[3]+1, 200)
                grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
                grid_z = gaussian_filter(griddata(obs_array[:, 0:2], obs_array[:, 2], (grid_x, grid_y), method='linear'), sigma=3.0)
            except: pass

    # ==========================================
    # 2.5 PRESSURE
    # ==========================================
    grid_p, grid_px, grid_py = None, None, None
    if rap_p is not None:
        rap_pts = np.column_stack((rap_lon.flatten(), rap_lat.flatten())); rap_flat = rap_p.flatten()
        if grid_x is None:
            grid_px, grid_py = np.meshgrid(np.linspace(MAP_EXTENT[0]-2, MAP_EXTENT[1]+2, 200), np.linspace(MAP_EXTENT[2]-2, MAP_EXTENT[3]+2, 200))
        else: grid_px, grid_py = grid_x, grid_y
        try:
            grid_p_rap = griddata(rap_pts, rap_flat, (grid_px, grid_py), method='linear')
            if np.any(np.isnan(grid_p_rap)): grid_p_rap[np.isnan(grid_p_rap)] = griddata(rap_pts, rap_flat, (grid_px[np.isnan(grid_p_rap)], grid_py[np.isnan(grid_p_rap)]), method='nearest')
            base_p = gaussian_filter(grid_p_rap, sigma=1.0)
            valid_p = []
            rap_val = NearestNDInterpolator(rap_pts, rap_flat)
            for s in mesonet_stations + list(all_station_data.values()):
                alt = s.get('alt_inHg')
                if alt and 28 < alt < 31:
                    hpa = alt * 33.8639
                    if abs(hpa - rap_val(s['lon'], s['lat'])) < 4.0: valid_p.append([s['lon'], s['lat'], hpa])
            if len(valid_p) > 5:
                parr = np.array(valid_p)
                corr = griddata(parr[:, 0:2], parr[:, 2] - rap_val(parr[:, 0:2]), (grid_px, grid_py), method='linear', fill_value=0)
                if np.any(np.isnan(corr)): corr[np.isnan(corr)] = 0
                grid_p = gaussian_filter(base_p + gaussian_filter(corr, sigma=8.0), sigma=1.0)
            else: grid_p = base_p
        except: grid_p = None

    # ==========================================
    # 3. PLOTTING
    # ==========================================
    logging.info("Beginning map generation...")
    
    fig = plt.figure(figsize=(16, 16), facecolor=FIG_BG_COLOR)
    text_outline = [pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()]
    current_time = datetime.now().strftime('%Y-%m-%d %I:%M %p') 
    
    atl_temp = None
    for icao, d in all_station_data.items():
        if d['name'] == 'Atlanta': atl_temp = d.get('temp_f')

    head_title, head_bg, head_text = determine_headline(alerts, atl_temp)
    draw_broadcast_header(fig, head_title, f"Generated: {current_time}", 
                          title_bg_color=head_bg, title_text_color=head_text)

    map_x = 0.09; map_y = 0.12; map_w = 0.83; map_h = 0.73
    ax = fig.add_axes([map_x, map_y, map_w, map_h], projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    rect_poly = Polygon([(MAP_EXTENT[0], MAP_EXTENT[2]), (MAP_EXTENT[0], MAP_EXTENT[3]), 
                         (MAP_EXTENT[1], MAP_EXTENT[3]), (MAP_EXTENT[1], MAP_EXTENT[2])])
    clip_patch = PathPatch(shapely_to_path(rect_poly), transform=ccrs.PlateCarree(), facecolor='none', edgecolor='none')
    ax.add_patch(clip_patch) 

    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor=OCEAN_COLOR)
    ax.add_feature(cfeature.LAND, zorder=0.4, facecolor=LAND_COLOR, alpha=0.0) 
    fetch_and_plot_topography(ax, MAP_EXTENT)
    add_rivers_and_lakes(ax, MAP_EXTENT)

    try:
        sat_data = load_pickle_cache("satellite_east_cache.pkl", CACHE_DURATION_SECONDS)
        if not sat_data:
            bt, x_coords, y_coords, sat_proj = fetch_satellite_data()
            if bt is not None: save_pickle_cache("satellite_east_cache.pkl", (bt, x_coords, y_coords))
        else: bt, x_coords, y_coords = sat_data; sat_proj = ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0, sweep_axis='x')
        add_satellite_overlay(fig, ax, clip_patch, text_outline, bt, x_coords, y_coords, sat_proj)
        for collection in ax.collections:
            if collection.get_zorder() == 1.0: collection.set_alpha(0.35) 
    except: pass

    try: 
        add_radar_image_overlay(fig, ax, CACHE_DURATION_SECONDS, text_outline, clip_patch, 
                                rtma_data=(rtma_lat, rtma_lon, rtma_temp), hrrr_data=(hrrr_lat, hrrr_lon, cat_snow, cat_ice, cat_fzra))
    except: pass

    # ==========================================
    # 4. COLORMAPS
    # ==========================================
    color_stops_data = [
    (-100, "#F0F0F0"), # Gray/White (Unchanged)
    
    # Swapped to BLUES (Previously Purple)
    (-80, "#000058"),  # Dark Blue
    (-60, "#0202BB"),  # Blue
    (-40, "#15589B"),  # Dodger Blue
    (-20, "#4180BB"),  # Deep Sky Blue
    
    # Swapped to PURPLES (Previously Blue)
    (-10, "#AD006B"),  # Indigo
    (-5, "#B6009E"),   # Blue Violet
    (0, "#FF01FF"),    # Purple
    (5, "#7F1BB1"),    # Dark Orchid
    (10, "#BA55D3"),   # Medium Orchid
    (15, "#DA70D6"),   # Orchid
    (20, "#E0B0FF"),   # Mauve
    
    # Remaining colors (Unchanged)
    (25, "#AFEEEE"), (30, "#00FFFF"), (35, "#00FF00"), 
    (40, "#32CD32"), (45, "#7CFC00"), (50, "#ADFF2F"), (55, "#FFFF00"), (60, "#FFD700"), 
    (65, "#FFA500"), (70, "#FF8C00"), (75, "#FF4500"), (80, "#FF0000"), (85, "#DC143C"), 
    (90, "#B22222"), (95, "#8B0000"), (100, "#A52A2A"), (105, "#8B4513"), (110, "#A0522D"), 
    (115, "#CD853F"), (120, "#D2B48C"), (125, "#FFDDC1"), (130, "#FFFFFF")
]
    min_val, max_val = -100, 130
    normalized_stops = [((v - min_val) / (max_val - min_val), c) for v, c in color_stops_data]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', normalized_stops)
    norm = Normalize(vmin=min_val, vmax=max_val)
    boundaries = np.arange(-100, 131, 2)

    if grid_z is not None:
        contour_plot = ax.contourf(grid_x, grid_y, grid_z, levels=boundaries, cmap=cmap, norm=norm, 
                                   transform=ccrs.PlateCarree(), zorder=1.5, alpha=0.50, extend='both')
        contour_plot.set_clip_path(clip_patch)
        temp_lines = ax.contour(grid_x, grid_y, grid_z, levels=np.arange(-100, 131, 10), 
                                colors='white', linewidths=0.5, alpha=0.75, transform=ccrs.PlateCarree(), zorder=1.55)
        temp_lines.set_clip_path(clip_patch)
        t_labels = ax.clabel(temp_lines, inline=True, fmt='%d', fontsize=10, colors='white', use_clabeltext=True, zorder=1.6)
        for l in t_labels: l.set_path_effects(text_outline)
    
    # ==========================================
    # --- MARINE / NAVY LAYER (Clean "Gulf Stream" Look) ---
    # ==========================================
    if ncom_lat is None:
        print(">> Marine Layer Skipped: No NCOM Data Available.")
    else:
        print(f">> Plotting Marine Layer... (Streamlines by Speed)")
        
    if ncom_lat is not None and grid_z is not None:
        try:
            logging.info("Plotting Marine Streamlines...")
            
            # 1. Calculate Current Speed (Magnitude)
            # This is what makes the Gulf Stream pop out!
            # U and V are in m/s. 
            magnitude = np.sqrt(ncom_u**2 + ncom_v**2)

            # 2. STREAMLINES (Colored by Speed)
            # cmap='plasma': Dark Blue (Slow) -> Bright Yellow (Fast)
            # linewidth: We make faster currents slightly thicker
            lw_stream = 0.6 + (magnitude / np.nanmax(magnitude)) * 1.5
            
            strm = ax.streamplot(ncom_lon, ncom_lat, ncom_u, ncom_v, 
                                 color=magnitude,      # Color lines by Speed
                                 cmap='plasma',        # Glowing look
                                 norm=Normalize(0, 1.5), # Clamp speed 0 to 1.5 m/s (Gulf Stream is ~1-2 m/s)
                                 density=2.0,          # How many lines (2.0 is dense)
                                 linewidth=lw_stream,  # Variable thickness
                                 arrowsize=0.8,
                                 transform=ccrs.PlateCarree(), 
                                 zorder=1.58)
            
            # 3. Clip to Map Boundaries
            if clip_patch:
                strm.lines.set_clip_path(clip_patch)
                strm.arrows.set_clip_path(clip_patch)
                
            # Optional: Add a subtle label on the map so you know what it is
            # ax.text(-76, 29, "OCEAN CURRENTS", transform=ccrs.PlateCarree(),
            #         color='yellow', fontsize=8, fontweight='bold', zorder=20,
            #         path_effects=text_outline)

        except Exception as e:
            logging.error(f"Marine Plotting Error: {e}")

    if grid_p is not None:
        try:
            cs_iso_fine = ax.contour(grid_px, grid_py, grid_p, levels=np.arange(900, 1100, 0.5), 
                                     colors="#0E2914", linewidths=1.5, alpha=0.9, zorder=1.6, transform=ccrs.PlateCarree())
            cs_iso_fine.set_clip_path(clip_patch)
            cs_iso_label = ax.contour(grid_px, grid_py, grid_p, levels=np.arange(900, 1100, 2.0), alpha=0, transform=ccrs.PlateCarree())
            clabels_iso = ax.clabel(cs_iso_label, inline=True, fontsize=10, fmt='%1.0f', colors='white', use_clabeltext=True, zorder=1.7)
            for label in clabels_iso: label.set_path_effects([pe.withStroke(linewidth=2.5, foreground="#0E2914")])
        except: pass

    try: 
        rap_pack = (rap_lat, rap_lon, rap_p)
        fetch_and_plot_wpc_pressure_centers(ax, rap_data=rap_pack)
    except: fetch_and_plot_wpc_pressure_centers(ax)
    try: fetch_and_plot_wpc_fronts(ax)
    except: pass
    active_spc_risks = set() 
    try: active_spc_risks = fetch_and_plot_spc_outlooks(ax, clip_patch)
    except: pass

    if grid_u is not None:
        barb_plot = ax.barbs(grid_x[::30, ::30], grid_y[::30, ::30], grid_u[::30, ::30], grid_v[::30, ::30], 
                             color="#8000A7", length=6, linewidth=2.0, transform=ccrs.PlateCarree(), zorder=6, alpha=0.7)
        barb_plot.set_clip_path(clip_patch)

    ax.add_feature(cfeature.COASTLINE, edgecolor='navy', linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.STATES, zorder=4, linestyle="-", edgecolor=STATE_BORDER_COLOR, linewidth=1.5)
    try:
        counties_feature = USCOUNTIES.with_scale('20m')
        county_patches = ax.add_feature(counties_feature, edgecolor=COUNTY_BORDER_COLOR, facecolor="none", linewidth=1.2, linestyle='-.', zorder=2, alpha=0.8)
        if clip_patch: county_patches.set_clip_path(clip_patch)
    except: pass
    try:
        roads_shp = shpreader.natural_earth(resolution='10m', category='cultural', name='roads')
        reader = shpreader.Reader(roads_shp)
        for rec in reader.records():
            if rec.attributes.get('type') in ['Major Highway', 'Secondary Highway']:
                b = rec.geometry.bounds
                if (b[0] <= MAP_EXTENT[1] and b[2] >= MAP_EXTENT[0] and b[1] <= MAP_EXTENT[3] and b[3] >= MAP_EXTENT[2]):
                    color = MAJOR_ROAD_COLOR if rec.attributes.get('type') == 'Major Highway' else SECONDARY_ROAD_COLOR
                    lw = 1.5 if rec.attributes.get('type') == 'Major Highway' else 1.0
                    p = ax.add_geometries([rec.geometry], ccrs.PlateCarree(), facecolor='none', edgecolor=color, linewidth=lw, zorder=3, alpha=1.0)
                    if clip_patch: p.set_clip_path(clip_patch)
    except: pass
    try: draw_interstate_shields(ax)
    except: pass

    gl = ax.gridlines(draw_labels=True, linewidth=1.0, color="#3932A0", alpha=0.8, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'path_effects': text_outline, 'fontweight': 'bold', 'color': 'white'}
    gl.ylabel_style = {'size': 10, 'path_effects': text_outline, 'fontweight': 'bold', 'color': 'white'}

    # --- COLORBARS ---
    pos = ax.get_position()
    cax_ir = fig.add_axes([pos.x0, 0.04, pos.width, 0.04])

    # ==========================================
    # --- NEW: OCEAN CURRENT COLORBAR (Top) ---
    # ==========================================
    # Placed at y=0.09 (Just above the Satellite bar which is at 0.04)
    # Height 0.015 (Thin strip)
    if ncom_lat is not None:
        cax_ocean = fig.add_axes([pos.x0, 0.09, pos.width, 0.015]) 
        
        # Must match streamplot settings: cmap='plasma', norm=0 to 1.8
        cb_ocean = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1.8), cmap='plasma'), 
                                cax=cax_ocean, orientation='horizontal', extend='max')
        
        cb_ocean.set_label('Ocean Current Speed (m/s)', fontsize=9, fontweight='bold', color='white', path_effects=text_outline)
        cb_ocean.ax.tick_params(labelsize=8, colors='white')
        
        # Add outlines to text so it pops against map
        for label in cb_ocean.ax.get_xticklabels():
            label.set_path_effects(text_outline)

    cbar_ir = fig.colorbar(plt.cm.ScalarMappable(cmap='gray', norm=Normalize(vmin=-90, vmax=50)), cax=cax_ir, orientation='horizontal', extend='both')
    cbar_ir.set_label('IR Satellite Overlay (Grayscale)', fontsize=12, fontweight='bold', color='white', path_effects=text_outline)
    cbar_ir.ax.tick_params(labelsize=10, colors='white')
    for label in cbar_ir.ax.get_xticklabels(): label.set_path_effects(text_outline)

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

    right_x = pos.x1 + 0.01
    cax_temp = fig.add_axes([right_x, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax_temp, orientation='vertical', extend='both')
    cbar.set_label('Temperature (°F)', fontsize=10, fontweight='bold', color='white', path_effects=text_outline)
    cbar.set_ticks(np.arange(-100, 131, 10))
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    for l in cbar.ax.get_yticklabels(): l.set_path_effects(text_outline); l.set_color('white'); l.set_fontsize(9)

    # ==========================================
    # 5. PLOT MESONET SQUARES
    # ==========================================
    logging.info(f"Plotting {len(valid_mesonet_for_plot)} sanitized mesonet stations...")
    MESO_ICON_SIZE_PX = 12; MESO_ZOOM = 0.5; glossy_icon_cache = {}
    main_city_locs = [(d['lat'], d['lon']) for d in all_station_data.values()]
    for station in valid_mesonet_for_plot:
        try:
            lat = station.get('lat'); lon = station.get('lon'); temp_val = station.get('temp_f')
            if lat is None or lon is None or temp_val is None: continue
            if not (MAP_EXTENT[0] < lon < MAP_EXTENT[1] and MAP_EXTENT[2] < lat < MAP_EXTENT[3]): continue
            too_close = False
            for m_lat, m_lon in main_city_locs:
                if abs(lat - m_lat) < 0.15 and abs(lon - m_lon) < 0.15: too_close = True; break
            if too_close: continue
            color_rgba = cmap(norm(temp_val)); color_hex = mcolors.to_hex(color_rgba)
            temp_int = int(round(temp_val))
            if temp_int not in glossy_icon_cache:
                icon_arr = create_glossy_square_icon(color_hex, size=MESO_ICON_SIZE_PX)
                glossy_icon_cache[temp_int] = OffsetImage(icon_arr, zoom=MESO_ZOOM)
            ab = AnnotationBbox(glossy_icon_cache[temp_int], (lon, lat), frameon=False, pad=0.0, zorder=19)
            ax.add_artist(ab)
        except: continue

    # ==========================================
    # 6. MAIN STATION LABELS (CENTERED ICON + 2x2 GRID + FLIGHT COLOR GLOW)
    # ==========================================
    print("\n--- PLOTTING BROADCAST STYLE LABELS (STACKED VPACKER FIX) ---")
    logging.info("Plotting station text boxes...")
    _collision_items = []
    GLOW_CITIES = ["Atlanta", "Savannah", "Augusta", "Columbus", "Macon"]

    # --- HELPERS FOR COLOR GLOW TEXT ---
    def get_text_props(color_hex, fsize=10, is_bold=True): # Default size reduced
        return dict(
            color='black', 
            fontsize=fsize, 
            fontweight='bold' if is_bold else 'normal',
            path_effects=[pe.withStroke(linewidth=2.5, foreground=color_hex), pe.Normal()],
            bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.2", facecolor='white', edgecolor=color_hex, linewidth=2.0, alpha=0.95)
        )
    
    # REDUCED FONT SIZES HERE:
    props_temp = get_text_props('#FF0000', 12) # Temp (Main) - Was 15
    props_dew  = get_text_props('#32CD32', 10) # Dew - Was 12
    props_alt  = get_text_props('#4682B4', 10) # Pressure - Was 12
    props_cond = get_text_props('#555555', 8, is_bold=False) # Condition - Was 12 (Smallest)
    
    # Footer props (Pill)
    pill_bbox = dict(boxstyle="round,pad=0.15,rounding_size=0.8", facecolor='#555555', edgecolor='none', alpha=0.9)
    props_footer = dict(color='white', fontsize=9, fontweight='bold', 
                        path_effects=[pe.withStroke(linewidth=2.0, foreground='black'), pe.Normal()], bbox=pill_bbox)

    from metpy.calc import wind_components
    from metpy.units import units

    for icao, data in all_station_data.items():
        city_name = data.get("name")
        if not city_name or city_name not in CUSTOM_LAYOUTS: continue
        
        # --- 1. COORDINATES & LEADER LINES ---
        layout = CUSTOM_LAYOUTS[city_name]
        box_x, box_y = layout['pos']; box_ha = layout['ha']
        base_lat, base_lon = data["lat"], data["lon"]
        
        # Draw Leader Line (Dotted Black with White Outline)
        leader_line, = ax.plot([base_lon, box_x], [base_lat, box_y], 
                               color='black', ls=':', lw=2.5, zorder=25,
                               path_effects=[pe.withStroke(linewidth=4, foreground='white', alpha=0.7)])
        
        if clip_patch: leader_line.set_clip_path(clip_patch)
        
        # --- 2. STATION DOT (FLIGHT CATEGORY) ---
        f_cat = data.get("flight_cat", "N/A")
        cat_color = FLIGHT_CATEGORY_COLORS.get(f_cat, "white")
        
        circle = ax.scatter(base_lon, base_lat, facecolor=cat_color, edgecolor='black', s=175, lw=1.5, 
                            transform=ccrs.PlateCarree(), zorder=20, alpha=1.0, marker='o')
        if clip_patch: circle.set_clip_path(clip_patch)

        # Invisible blocker for collision detection at the station point
        blocker = mpatches.Rectangle((base_lon - 0.1, base_lat - 0.1), 0.2, 0.2, transform=ccrs.PlateCarree(), alpha=0)
        ax.add_patch(blocker)
        _collision_items.append({'item': blocker, 'fixed': True})

        # --- 3. WIND BARBS ---
        current_wind_spd = data.get("wind_spd") or 0 
        w_dir = data.get("wind_dir")
        
        if current_wind_spd is not None and w_dir is not None:
            try:
                u, v = wind_components(current_wind_spd * units.knots, w_dir * units.degrees)
                b = ax.barbs(base_lon, base_lat, u.m, v.m, length=7, pivot='tip', color='#3500C7', linewidth=2.0, transform=ccrs.PlateCarree(), zorder=21)
                if clip_patch: b.set_clip_path(clip_patch)
            except: pass

        # =========================================================
        # --- 4. ASSEMBLE TEXT BOXES (VPACKER FIX) ---
        # =========================================================

        # --- A. CITY NAME BOX ---
        if city_name in GLOW_CITIES:
            name_props = dict(
                color='white', fontsize=11, fontweight='bold', # Reduced to 11
                path_effects=[pe.withStroke(linewidth=4, foreground="#FF0044", alpha=0.8), 
                              pe.withStroke(linewidth=1.0, foreground='black'), pe.Normal()],
                bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.2", facecolor='black', 
                          edgecolor='#FF0044', linewidth=2.0, alpha=1.0)
            )
        else:
            name_props = dict(
                color='white', fontsize=11, fontweight='bold', # Reduced to 11
                path_effects=[pe.withStroke(linewidth=3, foreground='black'), pe.Normal()],
                bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.2", facecolor='black', 
                          edgecolor='none', linewidth=0, alpha=1.0)
            )
        
        name_box = TextArea(city_name.upper(), textprops=name_props)

        # --- B. DATA BOX COMPONENTS ---
        temp_val = data.get("temp_f")
        temp_str = f" {temp_val:.0f}° "
        temp_box = TextArea(temp_str, textprops=props_temp)
        
        dp_box = TextArea(f" DP {data.get('dew_point_f'):.0f}° ", textprops=props_dew)
        
        alt = data.get("alt_inHg"); tend_arrow = data.get("press_arrow", "")
        alt_box = TextArea(f" {alt:.2f}\" {tend_arrow} ", textprops=props_alt)

        # Sky Condition
        cond = data.get("condition", " "); cond_short = cond.replace("Partially", "Partly").replace("Mostly", "Msly").replace("Storming", "Storms")
        cond_box = TextArea(f" {cond_short} ", textprops=props_cond)

        # Center Icon
        icon_img = WEATHER_ICONS.get(cond)
        icon_box = None
        if icon_img is not None:
            cat_rgb = mcolors.to_rgb(cat_color)
            cat_rgba_int = (int(cat_rgb[0]*255), int(cat_rgb[1]*255), int(cat_rgb[2]*255), 220)
            glowing_icon = apply_glow_effect(icon_img, glow_color=cat_rgba_int, blur_radius=12)
            icon_box = OffsetImage(glowing_icon, zoom=0.38)
        else:
            from matplotlib.offsetbox import AuxTransformBox
            icon_box = TextArea(" ", textprops=dict(alpha=0)) 

        # --- C. BUILD THE DATA GRID ---
        # Left Col (Temp / Cond) - Right aligned
        col1 = VPacker(children=[temp_box, cond_box], align="right", pad=0, sep=5)
        
        # Right Col (Dew / Alt) - Left aligned
        col_right = VPacker(children=[dp_box, alt_box], align="left", pad=0, sep=5)
        
        # Row (Left Col + Icon + Right Col)
        grid_packer = HPacker(children=[col1, icon_box, col_right], align="center", pad=0, sep=5)
        
        # Add Footer if needed (Wind/Heat Index)
        data_assembly = grid_packer
        extras = []
        fl_label, fl_val = calculate_feels_like(temp_val, data.get("dew_point_f"), current_wind_spd)
        if fl_label: extras.append(f"{fl_label} {fl_val}°")
        if current_wind_spd and current_wind_spd >= 5: extras.append(f"W {current_wind_spd}kt")
        
        if extras:
            footer_box = TextArea(f" {' | '.join(extras)} ", textprops=props_footer)
            data_assembly = VPacker(children=[grid_packer, footer_box], align="center", pad=0, sep=5)

        # --- D. MASTER PACKER (NAME ON TOP OF DATA) ---
        # sep=2 puts a 2 pixel gap between the name and the data box
        master_packer = VPacker(children=[name_box, data_assembly], align="center", pad=0, sep=2)

        # --- E. DRAW ANNOTATION ---
        grid_ab = AnnotationBbox(master_packer, (box_x, box_y), xybox=(0,0), xycoords='data', 
                                 boxcoords='offset points', frameon=False, pad=0.0, zorder=26)
        grid_ab.leader_line = leader_line
        ax.add_artist(grid_ab)
        
        # Add to collision detection
        _collision_items.append({'item': grid_ab, 'fixed': False})

    # Run collision resolution after the loop finishes
    # try:
    #     _resolve_text_collisions(fig, ax, _collision_items, padding_px=10, pixel_step=20) 
    #     if clip_patch: 
    #         for it in _collision_items: 
    #             try: it['item'].set_clip_path(clip_patch)
    #             except: pass
    # except: pass

    # ==========================================
    # 7. LEGENDS & SAVE
    # ==========================================
    alert_handles = {}
    if alerts:
        for alert in alerts:
            try:
                base_color = alert['color']
                if alert['event'] not in alert_handles:
                    alert_handles[alert['event']] = mpatches.Patch(facecolor=mcolors.to_rgba(base_color, 0.25), edgecolor='black', alpha=0.8, hatch='//', label=alert['event'])
                for poly_coords in alert.get('polygons', []):
                    p_fill = mpatches.Polygon(poly_coords, closed=True, facecolor=base_color, edgecolor='none', alpha=0.25, hatch=None, zorder=5, transform=ccrs.PlateCarree())
                    ax.add_patch(p_fill)
                    if clip_patch: p_fill.set_clip_path(clip_patch)
                    p_hatch = mpatches.Polygon(poly_coords, closed=True, facecolor='none', edgecolor='black', lw=1.0, alpha=0.7, hatch='//', zorder=5, transform=ccrs.PlateCarree())
                    ax.add_patch(p_hatch)
                    if clip_patch: p_hatch.set_clip_path(clip_patch)
            except: continue

    if alert_handles:
        sorted_alerts = sorted(alert_handles.values(), key=lambda x: x.get_label())
        leg_alerts = ax.legend(handles=sorted_alerts, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                               title="Active Alerts", fontsize=8, facecolor='white', framealpha=0.9, 
                               edgecolor='red', fancybox=True, ncol=1)
        leg_alerts.set_zorder(101)
        leg_alerts.get_title().set_path_effects(text_outline)
        leg_alerts.get_title().set_color("red") 
        leg_alerts.get_title().set_fontweight("bold")
        for t in leg_alerts.get_texts(): t.set_color("black")
        ax.add_artist(leg_alerts)

    legend_handles = [
        mlines.Line2D([], [], color='black', marker='*', ls='None', ms=9, mec='darkblue', label='City Dot'),
        mlines.Line2D([], [], color='white', marker='s', mfc='white', mec='red', mew=1.5, ms=8, ls='None', label='Temperature (°F)'),
        mlines.Line2D([], [], color='white', marker='s', mfc='white', mec='green', mew=1.5, ms=8, ls='None', label='Dew Pt / Feels Like'),
        mlines.Line2D([], [], color='white', marker='s', mfc='white', mec='#6495ED', mew=1.5, ms=8, ls='None', label='Pressure / Wind'),
        mlines.Line2D([], [], color='white', marker='s', mfc='white', mec='gray', mew=1.5, ms=8, ls='None', label='Condition'),
        mlines.Line2D([], [], color='#222222', ls='-', lw=0.6, label='Isobar (RAP Model)'),
        mlines.Line2D([], [], color="#750099", marker='|', ls='None', ms=10, mew=2, label='Wind Barb (RTMA)'),
        mpatches.Patch(color=MAJOR_ROAD_COLOR, label='Major Highway', alpha=0.7),
    ]

    if active_spc_risks:
        def add_spc_leg(label, color, lw=1.5):
            legend_handles.append(mpatches.Patch(facecolor='none', edgecolor=color, hatch='///', linewidth=lw, label=label))
        if 'TSTM' in active_spc_risks: add_spc_leg('SPC Gen Thunder', '#C1E9C1')
        if 'MRGL' in active_spc_risks: add_spc_leg('SPC Marginal', '#006400')
        if 'SLGT' in active_spc_risks: add_spc_leg('SPC Slight', '#FFD700', 2.0)
        if 'ENH' in active_spc_risks:  add_spc_leg('SPC Enhanced', '#FFA500', 2.0)
        if 'MDT' in active_spc_risks or 'CRIT' in active_spc_risks: add_spc_leg('SPC Mod/Crit', '#FF0000', 2.0)
        if 'HIGH' in active_spc_risks or 'EXT' in active_spc_risks: add_spc_leg('SPC High/Ext', '#FF00FF', 2.0)

    leg = ax.legend(handles=legend_handles, loc='upper left', fontsize=8, 
                    bbox_to_anchor=(0.0, 1.0), facecolor='white', framealpha=0.9, 
                    edgecolor='lightgray', fancybox=True, ncol=2,
                    handletextpad=0.5, borderpad=0.4, columnspacing=1.0)
    leg.set_zorder(100)
    for t in leg.get_texts(): t.set_path_effects(text_outline); t.set_color("#FFFFFF")

    flight_handles = [
        mpatches.Patch(color=FLIGHT_CATEGORY_COLORS["VFR"], label='VFR (Clear)'),
        mpatches.Patch(color=FLIGHT_CATEGORY_COLORS["MVFR"], label='MVFR (Marginal)'),
        mpatches.Patch(color=FLIGHT_CATEGORY_COLORS["IFR"], label='IFR (Low Clouds/Vis)'),
        mpatches.Patch(color=FLIGHT_CATEGORY_COLORS["LIFR"], label='LIFR (Very Low)'),
        mlines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=10, label='Wind Barb (Knots)'),
    ]

    leg_flight = ax.legend(handles=flight_handles, loc='lower left', fontsize=9, title="Station Flight Rules",
                        bbox_to_anchor=(0.02, 0.02), facecolor='white', framealpha=0.9, edgecolor='black')
    leg_flight.set_zorder(100)

    white_stroke = [pe.withStroke(linewidth=3, foreground='white'), pe.Normal()]
    leg_flight.get_title().set_color('black')
    leg_flight.get_title().set_fontweight('bold')
    leg_flight.get_title().set_path_effects(white_stroke)
    for text in leg_flight.get_texts(): text.set_color('black'); text.set_path_effects(white_stroke)

    ax.add_artist(leg) 
    ax.add_artist(leg_flight)

    try: PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    except: PROJECT_ROOT = os.getcwd()
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=300)
    buf.seek(0)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    with open(os.path.join(OUTPUT_DIR, f"ga_hrrr_weather_{timestamp}.png"), 'wb') as f: f.write(buf.getbuffer())
    buf.seek(0)
    with open(os.path.join(PROJECT_ROOT, "ga_detailed_weather.png"), 'wb') as f: f.write(buf.getbuffer())
    plt.close(fig)
    print("\n" + "="*30 + "\n✅ Map Generation Complete (Mesonet+HRRR+RTMA Mode).\n" + "="*30)

# --- EXECUTION BLOCK (Must be unindented to run!) ---
if __name__ == "__main__":
    try:
        generate_georgia_weather_map()
    except KeyboardInterrupt:
        logging.info("Map generation cancelled by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"--- A CRITICAL UNEXPECTED ERROR OCCURRED ---")
        print(f"{e}")