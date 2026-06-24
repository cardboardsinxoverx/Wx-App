import os
import logging
import aiohttp
import asyncio
import gzip
from io import BytesIO, StringIO
import pandas as pd
import numpy as np
from tropycal import realtime as realtime_obj, tracks
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError
from datetime import datetime, timedelta, timezone
import requests
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from scipy.optimize import brentq
import re
from PIL import Image as PILImage
import math
import xarray as xr
import netCDF4 as nc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Optional
import zipfile
import rasterio
from rasterio.plot import show
import warnings
import tempfile 

# Suppress DtypeWarning globally for ATCF parsing
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Output directory & Theme Constants
OUTPUT_DIR = 'outputs'
BG_COLOR = '#333333'
AX_COLOR = '#2B2B2B'
TEXT_COLOR = 'white'
LAND_COLOR = '#444444'

ATLANTIC_BASIN_LOCATIONS = [
    {"name": "St. John's, NL", "lat": 47.56, "lon": -52.71},
    {"name": "Sydney, NS", "lat": 46.1, "lon": -60.2},
    {"name": "Halifax, NS", "lat": 44.64, "lon": -63.57},
    {"name": "Saint John, NB", "lat": 45.3, "lon": -66.0},
    {"name": "Portland, ME", "lat": 43.7, "lon": -70.3},
    {"name": "Portsmouth, NH", "lat": 43.1, "lon": -70.8},
    {"name": "Boston, MA", "lat": 42.36, "lon": -71.05},
    {"name": "Cape Cod, MA", "lat": 41.8, "lon": -70.5},
    {"name": "New Bedford, MA", "lat": 41.7, "lon": -70.9},
    {"name": "Nantucket, MA", "lat": 41.3, "lon": -70.1},
    {"name": "New York, NY", "lat": 40.71, "lon": -74.00},
    {"name": "Atlantic City, NJ", "lat": 39.4, "lon": -74.4},
    {"name": "Ocean City, MD", "lat": 38.4, "lon": -75.0},
    {"name": "Norfolk, VA", "lat": 36.85, "lon": -76.28},
    {"name": "Virginia Beach, VA", "lat": 36.7, "lon": -76.0},
    {"name": "Cape Hatteras, NC", "lat": 35.25, "lon": -75.52},
    {"name": "Cape Lookout, NC", "lat": 34.6, "lon": -76.5},
    {"name": "Jacksonville, NC", "lat": 34.8, "lon": -77.4},
    {"name": "Wilmington, NC", "lat": 34.22, "lon": -77.94},
    {"name": "Cape Fear, NC", "lat": 33.8, "lon": -78.0},
    {"name": "Myrtle Beach, SC", "lat": 33.7, "lon": -78.9},
    {"name": "Charleston, SC", "lat": 32.77, "lon": -79.93},
    {"name": "Savannah, GA", "lat": 32.08, "lon": -81.09},
    {"name": "Brunswick, GA", "lat": 31.1, "lon": -81.5},
    {"name": "Jacksonville, FL", "lat": 30.33, "lon": -81.65},
    {"name": "Daytona Beach, FL", "lat": 29.2, "lon": -81.1},
    {"name": "Melbourne, FL", "lat": 28.1, "lon": -80.6},
    {"name": "Cape Canaveral, FL", "lat": 28.4, "lon": -80.6},
    {"name": "Fort Pierce, FL", "lat": 27.4, "lon": -80.3},
    {"name": "West Palm Beach, FL", "lat": 26.7, "lon": -80.1},
    {"name": "Boca Raton, FL", "lat": 26.4, "lon": -80.1},
    {"name": "Fort Lauderdale, FL", "lat": 26.1, "lon": -80.1},
    {"name": "Miami, FL", "lat": 25.76, "lon": -80.19},
    {"name": "Key West, FL", "lat": 24.55, "lon": -81.78},
    {"name": "Tampa, FL", "lat": 27.95, "lon": -82.45},
    {"name": "St. Petersburg, FL", "lat": 27.8, "lon": -82.6},
    {"name": "Sarasota, FL", "lat": 27.3, "lon": -82.5},
    {"name": "Port Charlotte, FL", "lat": 27.0, "lon": -82.1},
    {"name": "Fort Myers, FL", "lat": 26.6, "lon": -81.9},
    {"name": "Cape Coral, FL", "lat": 26.6, "lon": -82.0},
    {"name": "Naples, FL", "lat": 26.14, "lon": -81.79},
    {"name": "Pensacola, FL", "lat": 30.44, "lon": -87.19},
    {"name": "Panama City, FL", "lat": 30.17, "lon": -85.67},
    {"name": "Apalachicola, FL", "lat": 29.73, "lon": -84.99},
    {"name": "Perry, FL", "lat": 30.1, "lon": -83.6},
    {"name": "Mobile, AL", "lat": 30.68, "lon": -88.09},
    {"name": "Pascagoula, MS", "lat": 30.37, "lon": -88.55},
    {"name": "Biloxi, MS", "lat": 30.42, "lon": -88.93},
    {"name": "Gulfport, MS", "lat": 30.39, "lon": -89.07},
    {"name": "New Orleans, LA", "lat": 29.95, "lon": -90.07},
    {"name": "Houma, LA", "lat": 29.58, "lon": -90.71},
    {"name": "Buras, LA", "lat": 29.34, "lon": -89.50},
    {"name": "Port Arthur, TX", "lat": 29.83, "lon": -93.93},
    {"name": "Galveston, TX", "lat": 29.23, "lon": -94.89},
    {"name": "Freeport, TX", "lat": 28.95, "lon": -95.36},
    {"name": "Lake Jackson, TX", "lat": 29.0, "lon": -95.45},
    {"name": "Port O'Connor, TX", "lat": 28.23, "lon": -96.64},
    {"name": "Corpus Christi, TX", "lat": 27.71, "lon": -97.29},
    {"name": "Port Aransas, TX", "lat": 27.83, "lon": -97.08},
    {"name": "Brownsville, TX", "lat": 25.93, "lon": -97.48},
    {"name": "Altamira, Mexico", "lat": 22.39, "lon": -97.94},
    {"name": "Tampico, Mexico", "lat": 22.23, "lon": -97.86},
    {"name": "Progreso, Mexico", "lat": 21.28, "lon": -89.66},
    {"name": "Cancun, Mexico", "lat": 21.16, "lon": -86.85},
    {"name": "Veracruz, Mexico", "lat": 19.18, "lon": -96.14},
    {"name": "Coatzacoalcos, Mexico", "lat": 18.14, "lon": -94.45},
    {"name": "San Juan, PR", "lat": 18.46, "lon": -66.10},
    {"name": "Santo Domingo, DR", "lat": 18.48, "lon": -69.91},
    {"name": "Kingston, Jamaica", "lat": 17.97, "lon": -76.79},
    {"name": "Havana, Cuba", "lat": 23.11, "lon": -82.36},
    {"name": "Nassau, Bahamas", "lat": 25.04, "lon": -77.33},
    {"name": "Freeport, Bahamas", "lat": 26.53, "lon": -78.70},
    {"name": "Marsh Harbour, Bahamas", "lat": 26.54, "lon": -77.06},
    {"name": "Alice Town, Bahamas", "lat": 25.73, "lon": -79.30},
    {"name": "George Town, Bahamas", "lat": 23.52, "lon": -75.79},
    {"name": "Matthew Town, Bahamas", "lat": 20.95, "lon": -73.67},
    {"name": "Hamilton, Bermuda", "lat": 32.29, "lon": -64.78},
    {"name": "Bridgetown, Barbados", "lat": 13.10, "lon": -59.62},
    {"name": "Port of Spain, T&T", "lat": 10.65, "lon": -61.52},
]

# --- PROJECTION SETUP ---
map_proj = ccrs.PlateCarree()
goes_proj = ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0)
sat_height = 35786023.0
half_angle = 0.151872
x_max = sat_height * math.tan(half_angle)
y_max = x_max
y_min = -x_max
x_min = -x_max
pe_stroke = [pe.withStroke(linewidth=2, foreground='black')]

# --- HELPER FUNCTIONS ---
def fetch_ocean_currents(extent):
    print("\n--- MARINE DATA FETCH (RTOFS Direct Download with Cache) ---")
    import time
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    temp_file = os.path.join(cache_dir, "rtofs_today.nc")
    download_needed = True

    if os.path.exists(temp_file):
        file_age = time.time() - os.path.getmtime(temp_file)
        file_size = os.path.getsize(temp_file)
        if file_age < 43200 and file_size > 10000000:
            print(f"• Found valid cached RTOFS data ({int(file_age/60)} mins old). Skipping download.", flush=True)
            download_needed = False
        else:
            print("• Cached file is old or corrupted. Redownloading...", flush=True)
            try: os.remove(temp_file)
            except: pass

    if download_needed:
        logging.info("Attempting to fetch RTOFS data via direct HTTPS...")
        today = datetime.now(timezone.utc)
        dates_to_try = [today, today - timedelta(days=1), today - timedelta(days=2)]
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 Frostbyte/1.0'})
        files_to_try = ["rtofs_glo_2ds_f000_prog.nc", "rtofs_glo_2ds_f024_prog.nc", "rtofs_glo_2ds.f000.nc", "rtofs_glo_2ds.f024.nc"]

        success = False
        for date_obj in dates_to_try:
            if success: break
            date_str = date_obj.strftime('%Y%m%d')
            base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod/rtofs.{date_str}"

            for filename in files_to_try:
                url = f"{base_url}/{filename}"
                print(f"• Trying {filename} for {date_str}...", end=" ", flush=True)
                try:
                    head = session.head(url, timeout=10)
                    if head.status_code != 200:
                        print("Not found.")
                        continue
                    print("Downloading (this takes a moment but only runs ONCE)...", end=" ", flush=True)
                    r = session.get(url, stream=True, timeout=30)
                    if r.status_code == 200:
                        with open(temp_file, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk: f.write(chunk)
                        success = True
                        print("Done.")
                        break
                    else:
                        print("Failed.")
                except Exception as e:
                    print(f"Error ({e}).")
                    if os.path.exists(temp_file):
                        try: os.remove(temp_file)
                        except: pass
                    continue
        
        if not success:
            print("• All direct downloads failed.")
            return None, None, None, None, None, None

    print("Extracting...", end=" ", flush=True)
    try:
        ds = nc.Dataset(temp_file, 'r')
        lat_var = next((v for v in ['lat', 'Latitude', 'latitude'] if v in ds.variables), None)
        lon_var = next((v for v in ['lon', 'Longitude', 'longitude'] if v in ds.variables), None)

        lats_raw = ds.variables[lat_var][:]
        lons_raw = ds.variables[lon_var][:]
        lons_raw = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
        lat_min, lat_max = extent[2] - 2, extent[3] + 2
        lon_min, lon_max = extent[0] - 2, extent[1] + 2

        if lats_raw.ndim == 2:
            mask = (lats_raw >= lat_min) & (lats_raw <= lat_max) & (lons_raw >= lon_min) & (lons_raw <= lon_max)
            y_idxs, x_idxs = np.where(mask)
        else:
            y_mask = (lats_raw >= lat_min) & (lats_raw <= lat_max)
            x_mask = (lons_raw >= lon_min) & (lons_raw <= lon_max)
            y_idxs = np.where(y_mask)[0]
            x_idxs = np.where(x_mask)[0]

        if len(y_idxs) == 0 or len(x_idxs) == 0:
            print("Extent not in grid.")
            ds.close()
            return None, None, None, None, None, None

        y_min_idx, y_max_idx = y_idxs.min(), y_idxs.max()
        x_min_idx, x_max_idx = x_idxs.min(), x_idxs.max()

        if lats_raw.ndim == 2:
            lats = lats_raw[y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
            lons = lons_raw[y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
        else:
            lats = lats_raw[y_min_idx:y_max_idx+1]
            lons = lons_raw[x_min_idx:x_max_idx+1]

        sst_var = next((v for v in ['sst', 'SST', 'temperature', 'temp', 'sea_surface_temperature'] if v in ds.variables), None)
        u_var   = next((v for v in ['u_velocity', 'u', 'water_u', 'U', 'u_comp', 'velocity_u', 'surf_u'] if v in ds.variables), None)
        v_var   = next((v for v in ['v_velocity', 'v', 'water_v', 'V', 'v_comp', 'velocity_v', 'surf_v'] if v in ds.variables), None)
        sal_var = next((v for v in ['sss', 'SSS', 'salinity', 'salt', 'Salinity'] if v in ds.variables), None)

        def safe_slice(vname):
            if not vname: return None
            var_obj = ds.variables[vname]
            ndim = len(var_obj.shape)
            if ndim == 2: return var_obj[y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
            if ndim == 3: return var_obj[0, y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
            if ndim == 4: return var_obj[0, 0, y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
            return None

        w_temp = safe_slice(sst_var)
        w_temp = np.asarray(w_temp, dtype=np.float32) if w_temp is not None else np.full(lats.shape, np.nan, dtype=np.float32)
        w_u = safe_slice(u_var)
        w_u = np.asarray(w_u, dtype=np.float32) if w_u is not None else np.full_like(w_temp, np.nan)
        w_v = safe_slice(v_var)
        w_v = np.asarray(w_v, dtype=np.float32) if w_v is not None else np.full_like(w_temp, np.nan)
        salinity = safe_slice(sal_var)
        salinity = np.asarray(salinity, dtype=np.float32) if salinity is not None else np.full_like(w_temp, np.nan)

        valid_temp = w_temp[np.isfinite(w_temp) & (w_temp > -50) & (w_temp < 100)]
        if len(valid_temp) > 10 and np.nanmax(valid_temp) < 100:
            w_temp = np.where(np.isfinite(w_temp), w_temp + 273.15, w_temp)

        ocean_mask = (np.isfinite(w_temp) & (w_temp > 270) & (w_temp < 310) & np.isfinite(w_u) & np.isfinite(w_v)) 
        w_temp = np.where(ocean_mask, w_temp, np.nan)
        w_u    = np.where(ocean_mask, w_u,    np.nan)
        w_v    = np.where(ocean_mask, w_v,    np.nan)
        salinity = np.where(ocean_mask, salinity, np.nan)

        print("Success! ✅ (ocean masked)")
        ds.close()
        return lats, lons, w_temp, w_u, w_v, salinity

    except Exception as e:
        print(f"Error ({e}).")
        if 'ds' in locals():
            try: ds.close()
            except: pass
        return None, None, None, None, None, None

async def fetch_satellite_image(session):
    current_utc = datetime.now(timezone.utc)
    minutes = current_utc.minute
    rounded_min = (minutes // 15) * 15
    fetch_time = current_utc.replace(minute=rounded_min, second=0, microsecond=0)

    for delta in range(0, 45, 15):
        try_time = fetch_time - timedelta(minutes=delta)
        year = try_time.strftime('%Y')
        julian_day = try_time.strftime('%j')
        hour = try_time.strftime('%H')
        minute = try_time.strftime('%M').zfill(2)
        timestamp = f"{year}{julian_day}{hour}{minute}"

        url = f"https://cdn.star.nesdis.noaa.gov/GOES19/ABI/FD/GEOCOLOR/{timestamp}_GOES19-ABI-FD-GEOCOLOR-1808x1808.jpg"
        logging.info(f"Trying direct fetch: {url}")
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    if response.headers.get('Content-Type', '').startswith('image/'):
                        return PILImage.open(BytesIO(content)).convert('RGB')
        except Exception: pass

    base_url = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/FD/GEOCOLOR/"
    pattern = r'(\d{11}_GOES19-ABI-FD-GEOCOLOR-1808x1808\.jpg)'
    try:
        async with session.get(base_url) as response:
            if response.status == 200:
                html_content = await response.text()
                available_files = re.findall(pattern, html_content)
                if available_files:
                    available_times = []
                    for filename in available_files:
                        try:
                            ts = datetime.strptime(filename.split('_')[0], '%Y%j%H%M').replace(tzinfo=timezone.utc)
                            if ts <= current_utc: available_times.append((ts, filename))
                        except ValueError: continue
                    if available_times:
                        available_times.sort(reverse=True)
                        fallback_url = f"{base_url}{available_times[0][1]}"
                        async with session.get(fallback_url) as resp:
                            if resp.status == 200:
                                return PILImage.open(BytesIO(await resp.read())).convert('RGB')
    except Exception: pass

    rammb_url = "https://rammb-data.cira.colostate.edu/tc_realtime/products/general/atl/geocolor/latest.jpg"
    try:
        async with session.get(rammb_url) as response:
            if response.status == 200:
                return PILImage.open(BytesIO(await response.read())).convert('RGB')
    except Exception: pass
    return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return (R * c) / 1.852

def parse_direction(dir_str):
    if not dir_str: return None
    dirs = {'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5}
    return dirs.get(dir_str.upper(), None)

def get_category_color(vmax_kt):
    if vmax_kt < 34: return '#00BFFF', '^' 
    elif vmax_kt < 64: return '#00FA9A', 'o' 
    elif vmax_kt < 83: return '#FFD700', 'o' 
    elif vmax_kt < 96: return '#FFA500', 'o' 
    elif vmax_kt < 113: return '#FF4500', 'o' 
    elif vmax_kt < 130: return '#DA70D6', 'o' 
    else: return '#FF00FF', 'o' 

def estimate_wind_radii(V_th_kt, vmax_kt, mslp_mb, lat_deg, motion_dir_deg, motion_speed_kt):
    """
    Estimate 34/64-kt wind radii (NM) in 4 quadrants (NE, SE, SW, NW order)
    using a Holland (1980) parametric wind profile + storm motion asymmetry.
    RMW is now *always* computed dynamically from intensity (deltaP) + latitude
    for physical consistency across current position and all forecast points.
    """
    V_th_ms = V_th_kt * 0.514444
    vmax_ms = vmax_kt * 0.514444
    rho = 1.15
    Pn_mb = 1010.0
    if np.isnan(mslp_mb) or mslp_mb >= Pn_mb or mslp_mb <= 0:
        deltaP_mb = (vmax_ms ** 2 * rho * np.e) / 100.0
        mslp_mb = Pn_mb - deltaP_mb
    deltaP_mb = Pn_mb - mslp_mb
    deltaP_Pa = deltaP_mb * 100.0
    f = 2 * 7.292e-5 * np.sin(np.deg2rad(lat_deg))

    # Dynamically estimate RMW (km) from pressure deficit and latitude.
    # This replaces the previous fixed 15 NM placeholder and makes radii
    # consistent with the forecast intensity at each point in time.
    deltaP_mb_safe = max(deltaP_mb, 1.0)
    rmw_km = np.exp(2.635 - 0.00005086 * deltaP_mb_safe ** 2 + 0.0394899 * lat_deg)
    rmw_nm = rmw_km / 1.852
    rmw_nm = float(np.clip(rmw_nm, 5.0, 200.0))  # reasonable bounds

    rmw_km = rmw_nm * 1.852
    rmw_m = rmw_km * 1000.0
    B = np.clip((vmax_ms ** 2 * rho * np.e) / deltaP_Pa, 0.8, 2.5)

    def dP_dr(r):
        if r <= 0:
            return 0.0
        A = (rmw_m / r) ** B
        return deltaP_Pa * (B / r) * A * np.exp(-A)

    def V_sym_ms(r):
        if r <= 0:
            return 0.0
        term1 = (r * dP_dr(r)) / rho
        term2 = (f * r / 2) ** 2
        return - (f * r / 2) + np.sqrt(term1 + term2)

    quad_azims = [45, 135, 225, 315]
    quad_radii_nm = []
    has_motion = (motion_dir_deg is not None) and (motion_speed_kt > 0)
    motion_speed_ms = motion_speed_kt * 0.514444

    for quad in range(4):
        azim_deg = quad_azims[quad]
        asym_ms = motion_speed_ms * np.sin(np.deg2rad(azim_deg - motion_dir_deg)) if has_motion else 0.0
        V_th_eff_ms = V_th_ms - asym_ms
        if V_th_eff_ms <= 0:
            r_nm = 300.0
        else:
            def func(r):
                return V_sym_ms(r) - V_th_eff_ms
            try:
                r_sol_m = brentq(func, rmw_m * 1.01, 2000e3, rtol=1e-4)
                r_nm = (r_sol_m / 1000.0) / 1.852
            except ValueError:
                r_nm = 0.0
        quad_radii_nm.append(max(r_nm, 0.0))

    return quad_radii_nm

def plot_wind_field(ax, center_lon, center_lat, quad_radii, color, label=None, alpha=0.4, transform=ccrs.PlateCarree(), zorder=4.7):
    if not quad_radii or all(r == 0 for r in quad_radii):
        return
    angles = np.linspace(0, 360, 361)
    lons, lats = [], []
    for angle in angles:
        quad = int(angle // 90) % 4
        rad = quad_radii[quad]
        dlat = (rad / 60.0) * np.cos(np.deg2rad(angle))
        dlon = (rad / 60.0) * np.sin(np.deg2rad(angle)) / np.cos(np.deg2rad(center_lat))
        lons.append(center_lon + dlon)
        lats.append(center_lat + dlat)

    if label:
        ax.fill(lons, lats, color=color, alpha=alpha, edgecolor=None, transform=transform, clip_on=True, label=label, zorder=zorder)
    else:
        ax.fill(lons, lats, color=color, alpha=alpha, edgecolor=None, transform=transform, clip_on=True, zorder=zorder)

    ax.plot(lons + [lons[0]], lats + [lats[0]], color='black', linewidth=1.5, transform=transform, clip_on=True, zorder=zorder + 0.1)

def fetch_and_plot_storm_surge(ax, storm_id, extent, transform):
    surge_url = f"https://www.nhc.noaa.gov/gis/inundation/forecasts/{storm_id.lower()}_inundation_latest.zip"
    try:
        response = requests.get(surge_url)
        if response.status_code != 200: return
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            tif_name = [f for f in z.namelist() if f.endswith('.tif')][0]
            with z.open(tif_name) as f:
                with rasterio.open(BytesIO(f.read())) as src:
                    surge_data = src.read(1)
                    surge_lons, surge_lats = src.xy(*np.indices(src.shape), index=0)
                    min_lon, max_lon, min_lat, max_lat = extent
                    levels = [0, 3, 6, 9, 12, 18, np.inf]
                    colors = ['#E6FFE6', '#CCFFCC', '#99FF99', '#66FF66', '#33FF33', 'red']
                    cf = ax.contourf(surge_lons, surge_lats, surge_data, levels=levels, colors=colors, alpha=0.6, extend='max', transform=transform, zorder=2)
                    cbar = plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.02)
                    cbar.set_label('Storm Surge Inundation (ft)', color=TEXT_COLOR, path_effects=pe_stroke)
                    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
                    for label in cbar.ax.get_yticklabels():
                        label.set_color(TEXT_COLOR)
                        label.set_path_effects(pe_stroke)
    except Exception as e:
        logging.error(f"Error fetching storm surge: {e}")

async def fetch_nhc_advisory(storm_id, session):
    tcm_text = None
    cache_buster = int(datetime.now().timestamp())
    for i in range(1, 6):
        url = f"https://www.nhc.noaa.gov/text/MIATCMAT{i}.shtml?v={cache_buster}"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    if storm_id[:4].upper() in text:
                        tcm_text = text
                        break
        except Exception:
            continue

    if not tcm_text:
        return None, {}, {}, None, None, None

    storm_name = None
    name_match = re.search(r'(?:TROPICAL DEPRESSION|TROPICAL STORM|HURRICANE|POTENTIAL TROPICAL CYCLONE|POST-TROPICAL CYCLONE|REMNANTS OF)\s+([A-Z0-9\-\s]+?)\s+FORECAST', tcm_text, re.IGNORECASE)
    if name_match:
        storm_name = name_match.group(1).strip().title()

    adv_num = None
    adv_match = re.search(r'ADVISORY NUMBER\s+(\d+[A-Z]?)', tcm_text, re.IGNORECASE)
    if adv_match:
        adv_num = adv_match.group(1).zfill(3) if adv_match.group(1).isdigit() else adv_match.group(1)[:-1].zfill(3) + adv_match.group(1)[-1]

    init_time = None
    time_match = re.search(r'(\d{4})\s+UTC\s+[A-Z]{3}\s+([A-Z]{3})\s+(\d{1,2})\s+(\d{4})', tcm_text, re.IGNORECASE)
    if time_match:
        try:
            time_str = f"{time_match.group(4)} {time_match.group(2)} {time_match.group(3).zfill(2)} {time_match.group(1)}"
            init_time = datetime.strptime(time_str, "%Y %b %d %H%M").replace(tzinfo=timezone.utc)
        except Exception:
            pass

    track_data = []
    radii_34, radii_64 = {}, {}
    init_dt = init_time or datetime.now(timezone.utc)
    current_fhr = 0

    for line in tcm_text.splitlines():
        # Forgiving coordinate parser
        pt_match = re.search(r'(\d{2})/(\d{2})(\d{2})Z\s+(\d+\.?\d*)[NS]\s+(\d+\.?\d*)[EW]\s+(\d+)\s+KT', line, re.IGNORECASE)
        if pt_match:
            day, hr = int(pt_match.group(1)), int(pt_match.group(2))
            
            if 'INITIAL' in line.upper():
                init_dt = init_dt.replace(day=day, hour=hr)
                current_fhr = 0
            else:
                valid_dt = (init_dt + timedelta(days=15)).replace(day=day, hour=hr) if day < init_dt.day else init_dt.replace(day=day, hour=hr)
                current_fhr = int((valid_dt - init_dt).total_seconds() / 3600)
                
            lat = float(pt_match.group(4))
            if 'S' in line.upper().split(pt_match.group(4))[1][:2]: lat = -lat
            lon = float(pt_match.group(6))
            if 'E' not in line.upper().split(pt_match.group(6))[1][:2]: lon = -lon
            
            if current_fhr not in [td[0] for td in track_data]:
                track_data.append([current_fhr, f"{day:02d}/{hr:02d}00Z", lat, lon, int(pt_match.group(8)), np.nan])

        # Bulletproof radii parser (Ignores extra dots, spaces, and casing)
        rad_match = re.search(r'(34|50|64)\s*KT.*?(\d+)\s*[Nn][Ee].*?(\d+)\s*[Ss][Ee].*?(\d+)\s*[Ss][Ww].*?(\d+)\s*[Nn][Ww]', line)
        if rad_match:
            rad_type = int(rad_match.group(1))
            ne, se, sw, nw = map(int, rad_match.groups()[1:])
            if rad_type == 34: radii_34[current_fhr] = [ne, se, sw, nw]
            if rad_type == 64: radii_64[current_fhr] = [ne, se, sw, nw]

    df_nhc = pd.DataFrame(track_data, columns=['Hour', 'Time', 'Lat', 'Lon', 'Vmax', 'MSLP'])
    mslp_match = re.search(r'MINIMUM CENTRAL PRESSURE\s+(\d+)\s+MB', tcm_text, re.IGNORECASE)
    if mslp_match and not df_nhc.empty:
        df_nhc.loc[df_nhc['Hour'] == 0, 'MSLP'] = int(mslp_match.group(1))

    return df_nhc, radii_34, radii_64, init_time, adv_num, storm_name

async def fetch_and_plot_wwa(ax, storm_id, adv_num, session, transform):
    from shapely.geometry import shape  # Cartopy dependency, guaranteed to be available
    
    # --- 1. FETCH LIVE INLAND POLYGONS (US NWS ZONES) ---
    # The NHC doesn't put inland polygons in their basic 5-day ZIP. 
    # Those are issued by local NWS offices, so we grab them live from the API.
    nws_url = "https://api.weather.gov/alerts/active?event=Hurricane%20Warning,Hurricane%20Watch,Tropical%20Storm%20Warning,Tropical%20Storm%20Watch"
    try:
        async with session.get(nws_url, headers={'User-Agent': 'FrostByte/1.0'}) as response:
            if response.status == 200:
                data = await response.json()
                for feature in data.get('features', []):
                    event = feature['properties'].get('event', '')
                    geom = feature.get('geometry')
                    if not geom: continue
                    
                    try:
                        poly = shape(geom)
                        
                        # Layer them with a clean alpha
                        if 'Hurricane Warning' in event:
                            ax.add_geometries([poly], transform, facecolor='red', edgecolor='black', linewidth=0.5, alpha=0.4, zorder=3.7)
                        elif 'Tropical Storm Warning' in event:
                            ax.add_geometries([poly], transform, facecolor='blue', edgecolor='black', linewidth=0.5, alpha=0.4, zorder=3.8)
                        elif 'Hurricane Watch' in event:
                            ax.add_geometries([poly], transform, facecolor='magenta', edgecolor='black', linewidth=0.5, alpha=0.4, zorder=3.5)
                        elif 'Tropical Storm Watch' in event:
                            ax.add_geometries([poly], transform, facecolor='yellow', edgecolor='black', linewidth=0.5, alpha=0.4, zorder=3.6)
                    except Exception:
                        continue
    except Exception as e:
        logging.error(f"NWS API Error: {e}")

    # --- 2. FETCH NHC COASTAL LINES (INTERNATIONAL / FALLBACK) ---
    # We still fetch the NHC zip to get the coastal breakpoints, which is crucial 
    # for international landmasses (Mexico, Bahamas, etc.) not covered by the NWS API.
    urls_to_try = []
    if adv_num:
        urls_to_try.append(f"https://www.nhc.noaa.gov/gis/forecast/archive/{storm_id.lower()}_5day_{adv_num}.zip")
        if not adv_num.isdigit():
            base_adv = ''.join(filter(str.isdigit, adv_num)).zfill(3)
            urls_to_try.append(f"https://www.nhc.noaa.gov/gis/forecast/archive/{storm_id.lower()}_5day_{base_adv}.zip")
            
    urls_to_try.append(f"https://www.nhc.noaa.gov/gis/forecast/archive/{storm_id.lower()}_5day_latest.zip")
    
    for url in urls_to_try:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    with tempfile.TemporaryDirectory() as tmpdir:
                        with zipfile.ZipFile(BytesIO(content)) as z:
                            z.extractall(tmpdir)
                        
                        all_files = os.listdir(tmpdir)
                        ww_candidates = [f for f in all_files if 'ww' in f.lower() and f.endswith('.shp')]
                        if not ww_candidates: continue
                        
                        shp_path = None
                        
                        # Actively open candidates and check their geometry type
                        for f in ww_candidates:
                            test_path = os.path.join(tmpdir, f)
                            try:
                                test_reader = shpreader.Reader(test_path)
                                geoms = list(test_reader.geometries())
                                if geoms and geoms[0].geom_type in ['Polygon', 'MultiPolygon']:
                                    shp_path = test_path
                                    break
                            except Exception:
                                continue
                                
                        # Fallback to lines
                        if not shp_path:
                            lin_files = [f for f in ww_candidates if 'lin' in f.lower()]
                            shp_path = os.path.join(tmpdir, lin_files[0]) if lin_files else os.path.join(tmpdir, ww_candidates[0])
                        
                        reader = shpreader.Reader(shp_path)
                        records_geoms = list(zip(reader.records(), reader.geometries()))
                        
                        thick_outline = [pe.Stroke(linewidth=8.0, foreground='black'), pe.Normal()]
                        thin_outline = [pe.Stroke(linewidth=5.5, foreground='black'), pe.Normal()]

                        # Watches first (underneath) - Adjusted Z-orders below the track (4.0)
                        for record, geom in records_geoms:
                            tcww = record.attributes.get('TCWW') or record.attributes.get('WARN')
                            is_poly = geom.geom_type in ['Polygon', 'MultiPolygon']
                            
                            if tcww == 'HWA':
                                if is_poly:
                                    ax.add_geometries([geom], transform, facecolor='magenta', edgecolor='black', linewidth=0.5, alpha=0.4, zorder=3.5)
                                else:
                                    ax.add_geometries([geom], transform, facecolor='none', edgecolor='magenta', linewidth=6.0, path_effects=thick_outline, zorder=3.5)
                            elif tcww == 'TWA':
                                if is_poly:
                                    ax.add_geometries([geom], transform, facecolor='yellow', edgecolor='black', linewidth=0.5, alpha=0.4, zorder=3.6)
                                else:
                                    ax.add_geometries([geom], transform, facecolor='none', edgecolor='yellow', linewidth=3.5, path_effects=thin_outline, zorder=3.6)
                                    
                        # Warnings next (on top) - Adjusted Z-orders below the track (4.0)
                        for record, geom in records_geoms:
                            tcww = record.attributes.get('TCWW') or record.attributes.get('WARN')
                            is_poly = geom.geom_type in ['Polygon', 'MultiPolygon']
                            
                            if tcww == 'HWR':
                                if is_poly:
                                    ax.add_geometries([geom], transform, facecolor='red', edgecolor='black', linewidth=0.5, alpha=0.4, zorder=3.7)
                                else:
                                    ax.add_geometries([geom], transform, facecolor='none', edgecolor='red', linewidth=6.0, path_effects=thick_outline, zorder=3.7)
                            elif tcww == 'TWR':
                                if is_poly:
                                    ax.add_geometries([geom], transform, facecolor='blue', edgecolor='black', linewidth=0.5, alpha=0.4, zorder=3.8)
                                else:
                                    ax.add_geometries([geom], transform, facecolor='none', edgecolor='blue', linewidth=3.5, path_effects=thin_outline, zorder=3.8)
                    return
        except Exception as e:
            continue

def get_category_label(vmax_kt):
    if vmax_kt < 34: return 'D'
    elif vmax_kt < 64: return 'S'
    elif vmax_kt < 83: return '1'
    elif vmax_kt < 96: return '2'
    elif vmax_kt < 113: return '3'
    elif vmax_kt < 130: return '4'
    else: return '5'

async def create_plot(tracks, title_suffix, session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, adv_num=None, is_ensemble=False, color_by_mslp=False, show_cone_and_forecast=False, show_surge=False, ensemble_name=None, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    # Added 0 to properly close the gap to the starting location
    cone_radii = {0: 15, 12: 26, 24: 39, 36: 52, 48: 67, 60: 83, 72: 100, 96: 142, 120: 213}
    
    initial_vmax = float(vmax) if vmax != 'N/A' else 35.0
    storm_type = "Tropical Storm" if (vmax == 'N/A' or float(vmax) < 64) else "Hurricane"
    motion_dir_deg = parse_direction(storm_dir_str) or 0
    storm_speed_kt = float(storm_speed_mph) * 0.868976 if storm_speed_mph else 0.0

    # --- DYNAMIC SQUARE EXTENT CALCULATION ---
    all_lons, all_lats = [], []
    if not np.isnan(lon) and not np.isnan(lat):
        all_lons.append(lon)
        all_lats.append(lat)
    if official_forecast:
        all_lons.extend(official_forecast['lon'])
        all_lats.extend(official_forecast['lat'])
    for track in tracks.values():
        all_lons.extend(track['lon'])
        all_lats.extend(track['lat'])

    if all_lons and all_lats:
        pad = 8.0 # Generous 8-degree pad so the storm isn't crushed against the walls
        min_lon, max_lon = min(all_lons) - pad, max(all_lons) + pad
        min_lat, max_lat = min(all_lats) - pad, max(all_lats) + pad
        
        width = max_lon - min_lon
        height = max_lat - min_lat
        max_dim = max(width, height)
        
        center_lon = (max_lon + min_lon) / 2.0
        center_lat = (max_lat + min_lat) / 2.0
        
        EXTENT_LON_MIN = center_lon - max_dim / 2.0
        EXTENT_LON_MAX = center_lon + max_dim / 2.0
        EXTENT_LAT_MIN = center_lat - max_dim / 2.0
        EXTENT_LAT_MAX = center_lat + max_dim / 2.0
    else:
        EXTENT_LON_MIN, EXTENT_LON_MAX = -100, -15
        EXTENT_LAT_MIN, EXTENT_LAT_MAX = 5, 50

    extent_list = [EXTENT_LON_MIN, EXTENT_LON_MAX, EXTENT_LAT_MIN, EXTENT_LAT_MAX]

    # --- EXPLICIT MARGIN CALCULATIONS FOR LEGEND ---
    fig_width = 13.0 
    left_margin_in = 0.55
    right_margin_in = 1.15 # Perfectly balance the left/right gray background margins
    top_margin_in = 1.2
    bottom_margin_in = 0.55
    
    ax_width_in = fig_width - left_margin_in - right_margin_in
    map_aspect = 1.0 # Forced square
    ax_height_in = ax_width_in / map_aspect
    
    fig_height = ax_height_in + top_margin_in + bottom_margin_in

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=BG_COLOR)
    
    ax_x0 = left_margin_in / fig_width
    ax_y0 = bottom_margin_in / fig_height
    ax_w = ax_width_in / fig_width
    ax_h = ax_height_in / fig_height
    
    cax_x0 = (left_margin_in + ax_width_in + 0.15) / fig_width
    cax_y0 = ax_y0
    cax_w = 0.15 / fig_width
    cax_h = ax_h
    
    ax = fig.add_axes([ax_x0, ax_y0, ax_w, ax_h], projection=map_proj)
    ax.set_facecolor(AX_COLOR)
    ax.set_extent(extent_list, crs=ccrs.PlateCarree())

    text_outline = [pe.Stroke(linewidth=2.5, foreground='black'), pe.Normal()]

    # --- OVERLAYS ---
    img = await fetch_satellite_image(session)
    lats_oc, lons_oc, w_temp, w_u, w_v, salinity = await asyncio.to_thread(fetch_ocean_currents, extent_list)

    try:
        color_stops_data = [(-100, "#F0F0F0"), (-80, "#000058"), (-60, "#0202BB"), (-40, "#15589B"), (-20, "#4180BB"), (-10, "#AD006B"), (-5, "#B6009E"), (0, "#FF01FF"), (5, "#7F1BB1"), (10, "#BA55D3"), (15, "#DA70D6"), (20, "#E0B0FF"), (25, "#AFEEEE"), (30, "#00FFFF"), (35, "#00FF00"), (40, "#32CD32"), (45, "#7CFC00"), (50, "#ADFF2F"), (55, "#FFFF00"), (60, "#FFD700"), (65, "#FFA500"), (70, "#FF8C00"), (75, "#FF4500"), (80, "#FF0000"), (85, "#DC143C"), (90, "#B22222"), (95, "#8B0000"), (100, "#A52A2A"), (105, "#8B4513"), (110, "#A0522D"), (115, "#CD853F"), (120, "#D2B48C"), (125, "#FFDDC1"), (130, "#FFFFFF")]
        sst_cmap = mcolors.LinearSegmentedColormap.from_list('ga_temp_cmap', [((v - -100) / (130 - -100), c) for v, c in color_stops_data])
        sst_norm = mcolors.Normalize(vmin=-73.333, vmax=54.444)
        sst_url = f"https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{datetime.now(timezone.utc).year}.nc"
        with xr.open_dataset(sst_url) as ds:
            sst = ds['sst'].isel(time=-1).sortby('lat').sortby('lon')
            sst = sst.sel(lat=slice(EXTENT_LAT_MIN - 5, EXTENT_LAT_MAX + 5), lon=slice(360 + EXTENT_LON_MIN - 5, 360 + EXTENT_LON_MAX + 5))
            levels = np.arange(-10, 35, 1)
            ax.contourf(sst.lon, sst.lat, sst, transform=ccrs.PlateCarree(), levels=levels, cmap=sst_cmap, norm=sst_norm, alpha=1.0, zorder=0.5)
            sst_contours = ax.contour(sst.lon, sst.lat, sst, transform=ccrs.PlateCarree(), levels=levels, cmap=sst_cmap, norm=sst_norm, linewidths=1.5, alpha=0.9, zorder=2.5)
            for label in ax.clabel(sst_contours, inline=True, fontsize=9, fmt='%1.0f°C'):
                label.set_path_effects(text_outline)
                label.set_fontweight('bold')
    except Exception as e: pass

    if img is not None:
        ax.imshow(img, origin='upper', extent=(x_min, x_max, y_min, y_max), transform=goes_proj, zorder=1.0, interpolation='bilinear', alpha=0.8)
    else:
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor='black', zorder=1.0)
        ax.add_feature(cfeature.OCEAN, facecolor=AX_COLOR, zorder=1.0)

    strm = None
    if lats_oc is not None and w_u is not None and w_temp is not None:
        try:
            stream_norm = mcolors.Normalize(vmin=0, vmax=35)
            strm = ax.streamplot(lons_oc[::4, ::4], lats_oc[::4, ::4], w_u[::4, ::4], w_v[::4, ::4], color=(w_temp - 273.15)[::4, ::4], cmap='jet', norm=stream_norm, density=1.2, linewidth=1.0, arrowsize=1.2, transform=ccrs.PlateCarree(), zorder=1.8)
            strm.lines.set_path_effects([pe.Stroke(linewidth=2.5, foreground='black'), pe.Normal()])
        except Exception: pass

    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.5, zorder=2.0)
    ax.add_feature(cfeature.BORDERS, edgecolor='white', linestyle=':', zorder=2.0)
    ax.add_feature(cfeature.STATES, edgecolor='white', linewidth=0.5, zorder=2.0)

    gl = ax.gridlines(draw_labels=True, alpha=0.6, color='#3932A0', linestyle='--', zorder=3.0, path_effects=[pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()])
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'white', 'path_effects': text_outline}
    gl.ylabel_style = {'size': 10, 'color': 'white', 'path_effects': text_outline}

    # Ensure radii arrays contain initial estimates if missing
    if 0 not in radii_34 and not np.isnan(lat):
        radii_34[0] = estimate_wind_radii(34, initial_vmax, float(mslp) if mslp != 'N/A' else 1010, lat, motion_dir_deg, storm_speed_kt)
    if 0 not in radii_64 and not np.isnan(lat):
        radii_64[0] = estimate_wind_radii(64, initial_vmax, float(mslp) if mslp != 'N/A' else 1010, lat, motion_dir_deg, storm_speed_kt)

    # --- PAST TRACK WITH HISTORICAL DOTS ---
    if len(obs_lats) > 0 and len(obs_lons) > 0:
        if obs_lats[-1] != lat or obs_lons[-1] != lon:
            plot_obs_lats = np.append(obs_lats, lat)
            plot_obs_lons = np.append(obs_lons, lon)
        else:
            plot_obs_lats, plot_obs_lons = obs_lats, obs_lons
            
        ax.plot(plot_obs_lons, plot_obs_lats, color='white', linewidth=2.5, path_effects=text_outline, transform=ccrs.PlateCarree(), zorder=4.0, clip_on=True)
        ax.scatter(plot_obs_lons[:-1], plot_obs_lats[:-1], color='white', s=20, edgecolors='black', transform=ccrs.PlateCarree(), zorder=4.1, clip_on=True)

    # --- OFFICIAL FORECAST WITH DYNAMIC WIND FIELDS ---
    if show_cone_and_forecast and official_forecast:
        fc_lats, fc_lons, fc_fhrs = official_forecast['lat'], official_forecast['lon'], official_forecast['fhr']
        fc_vmaxs = official_forecast.get('vmax', np.full(len(fc_fhrs), initial_vmax))
        fc_mslps = official_forecast.get('mslp', np.full(len(fc_fhrs), float(mslp) if mslp != 'N/A' else 1010))
        
        # Drop fhr == 0 (the INITIAL point) so the cone initializes purely at the current `lat, lon`
        valid_idx = fc_fhrs > 0
        fc_lats = fc_lats[valid_idx]
        fc_lons = fc_lons[valid_idx]
        fc_fhrs = fc_fhrs[valid_idx]
        fc_vmaxs = fc_vmaxs[valid_idx]
        fc_mslps = fc_mslps[valid_idx]
        
        # PREPEND CURRENT LOCATION FOR CONE & LINE ANCHORING
        if not np.isnan(lat) and not np.isnan(lon):
            cone_lats = np.insert(fc_lats, 0, lat)
            cone_lons = np.insert(fc_lons, 0, lon)
            cone_fhrs = np.insert(fc_fhrs, 0, 0)
        else:
            cone_lats, cone_lons, cone_fhrs = fc_lats, fc_lons, fc_fhrs
            
        # Reverted back to a dashed red line!
        ax.plot(cone_lons, cone_lats, color='red', linestyle='--', linewidth=2.5, path_effects=text_outline, transform=ccrs.PlateCarree(), zorder=5.0, clip_on=True)
        
        points = []
        for i, fhr in enumerate(cone_fhrs):
            if fhr in cone_radii:
                radius = cone_radii[fhr] / 60.0
                theta = np.linspace(0, 2 * np.pi, 100)
                points.extend(list(zip(cone_lons[i] + radius * np.cos(theta) / np.cos(np.deg2rad(cone_lats[i])), cone_lats[i] + radius * np.sin(theta))))
        if points:
            try:
                hull_points = np.array([points[i] for i in ConvexHull(points).vertices])
                ax.fill(hull_points[:, 0], hull_points[:, 1], color='white', alpha=0.4, transform=ccrs.PlateCarree(), zorder=4.5, clip_on=True)
                hull_lons = np.append(hull_points[:, 0], hull_points[0, 0])
                hull_lats = np.append(hull_points[:, 1], hull_points[0, 1])
                ax.plot(hull_lons, hull_lats, color='white', linewidth=2.0, path_effects=text_outline, transform=ccrs.PlateCarree(), zorder=4.6, clip_on=True)
            except QhullError: pass
            
        # Draw Wind Radii, Dots, and Labels at Forecast Points
        for i, (f_lat, f_lon, f_vmax, f_fhr) in enumerate(zip(fc_lats, fc_lons, fc_vmaxs, fc_fhrs)):
            if f_fhr in cone_radii:
                q34 = radii_34.get(f_fhr) or estimate_wind_radii(34, f_vmax, fc_mslps[i], f_lat, motion_dir_deg, storm_speed_kt)
                q64 = radii_64.get(f_fhr) or estimate_wind_radii(64, f_vmax, fc_mslps[i], f_lat, motion_dir_deg, storm_speed_kt)
                plot_wind_field(ax, f_lon, f_lat, q64, 'red', alpha=0.45, transform=ccrs.PlateCarree(), zorder=4.9)
                plot_wind_field(ax, f_lon, f_lat, q34, 'yellow', alpha=0.45, transform=ccrs.PlateCarree(), zorder=4.8)
                
                # Render the stamped dot for forecast points
                color, marker = get_category_color(f_vmax)
                cat_label = get_category_label(f_vmax)
                ax.scatter(f_lon, f_lat, color=color, marker=marker, s=140, edgecolors='black', linewidths=1, transform=ccrs.PlateCarree(), zorder=6.0, clip_on=True)
                ax.text(f_lon, f_lat, cat_label, color='black', fontsize=7, fontweight='bold', ha='center', va='center', transform=ccrs.PlateCarree(), zorder=6.1, clip_on=True)
                
                if 'init' in official_forecast:
                    valid_time = official_forecast['init'] + timedelta(hours=int(f_fhr))
                    label_str = valid_time.strftime('%d %b %Hz')
                else:
                    label_str = f'{int(f_fhr)}h'
                
                ax.text(f_lon + 0.4, f_lat + 0.3, label_str, color='white', fontsize=9, fontweight='bold', path_effects=text_outline, transform=ccrs.PlateCarree(), zorder=7.0, clip_on=True)

        # Draw Active Coastal Watches & Warnings Shapefiles Natively
        await fetch_and_plot_wwa(ax, storm_id, adv_num, session, ccrs.PlateCarree())

    # --- CURRENT POSITION WIND FIELDS (Hour 0) ---
    if 0 in radii_64: plot_wind_field(ax, lon, lat, radii_64[0], 'red', alpha=0.55, transform=ccrs.PlateCarree(), zorder=4.9)
    if 0 in radii_34: plot_wind_field(ax, lon, lat, radii_34[0], 'yellow', alpha=0.55, transform=ccrs.PlateCarree(), zorder=4.8)

    # --- CURRENT POSITION MARKER ---
    if not np.isnan(lat) and not np.isnan(lon):
        color, marker = get_category_color(initial_vmax)
        cat_label = get_category_label(initial_vmax)
        ax.scatter(lon, lat, color=color, marker=marker, s=200, edgecolors='black', linewidths=1.5, transform=ccrs.PlateCarree(), zorder=10.0, clip_on=True)
        ax.text(lon, lat, cat_label, color='black', fontsize=9, fontweight='heavy', ha='center', va='center', transform=ccrs.PlateCarree(), zorder=10.1, clip_on=True)

    if show_surge: fetch_and_plot_storm_surge(ax, storm_id, extent_list, ccrs.PlateCarree())

    # --- GEFS MSLP TRACING OR STANDARD MODELS ---
    colors = plt.cm.tab20(np.linspace(0, 1, len(tracks)))
    
    if is_ensemble:
        gefs_norm = mcolors.Normalize(vmin=940, vmax=1020)
        gefs_cmap = plt.cm.turbo_r
        
    for i, (model, track) in enumerate(tracks.items()):
        color = colors[i % len(colors)]
        plot_lon, plot_lat = np.array(track['lon']), np.array(track['lat'])
        plot_mslp = np.array(track['mslp'])
        
        if len(plot_lon) > 0 and not np.isnan(lat) and not np.isnan(lon):
            if haversine(lat, lon, plot_lat[0], plot_lon[0]) < 400:
                if track['fhr'][0] == 0: 
                    plot_lon[0], plot_lat[0] = lon, lat
                    if np.isnan(plot_mslp[0]) and mslp != 'N/A': plot_mslp[0] = float(mslp)
                else: 
                    plot_lon, plot_lat = np.insert(plot_lon, 0, lon), np.insert(plot_lat, 0, lat)
                    plot_mslp = np.insert(plot_mslp, 0, float(mslp) if mslp != 'N/A' else np.nan)
        
        if is_ensemble:
            s = pd.Series(plot_mslp).interpolate().bfill().ffill()
            plot_mslp_clean = s.values
            if np.isnan(plot_mslp_clean).all():
                plot_mslp_clean = np.full_like(plot_mslp_clean, 1000.0)
            
            pts = np.array([plot_lon, plot_lat]).T.reshape(-1, 1, 2)
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            
            lc_bg = LineCollection(segments, colors='black', linewidths=3.0, transform=ccrs.PlateCarree(), zorder=4.9, clip_on=True)
            ax.add_collection(lc_bg)
            
            lc = LineCollection(segments, cmap=gefs_cmap, norm=gefs_norm, transform=ccrs.PlateCarree(), zorder=5.0, clip_on=True)
            lc.set_array(plot_mslp_clean[:-1])
            lc.set_linewidth(1.5)
            ax.add_collection(lc)
            
        else:
            ax.plot(plot_lon, plot_lat, color=color, linewidth=1.5, path_effects=[pe.withStroke(linewidth=3.5, foreground='black'), pe.Normal()], transform=ccrs.PlateCarree(), label=model, zorder=5.0, clip_on=True)
            if len(plot_lon) > 0:
                label_x = plot_lon[-1] + 0.5 if plot_lon[-1] < EXTENT_LON_MAX - 1 else plot_lon[-1] - 1
                if EXTENT_LON_MIN <= label_x <= EXTENT_LON_MAX and EXTENT_LAT_MIN <= plot_lat[-1] <= EXTENT_LAT_MAX:
                    ax.text(label_x, plot_lat[-1], model, color=color, fontsize=9, fontweight='bold', path_effects=text_outline, transform=ccrs.PlateCarree(), zorder=6.0, clip_on=True)

    # --- TOP LEFT DATE STAMP BOX ---
    time_str_box = datetime.now(timezone.utc).strftime("%H00 UTC %d %b %Y")
    if official_forecast and 'init' in official_forecast:
        time_str_box = official_forecast['init'].strftime("%H%M UTC %d %b %Y")
    
    ax.text(0.02, 0.98, time_str_box, transform=ax.transAxes, fontsize=11, fontweight='bold', color='white', va='top', ha='left', path_effects=text_outline, bbox=dict(facecolor=AX_COLOR, alpha=0.8, edgecolor='black', pad=5), zorder=12)

    # --- DYNAMIC COLORBAR HANDLING ---
    if is_ensemble:
        cax = fig.add_axes([cax_x0, cax_y0, cax_w, cax_h])
        sm = plt.cm.ScalarMappable(cmap=gefs_cmap, norm=gefs_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical', extend='both')
        cbar.set_label('Minimum MSLP (mb)', color='white', weight='bold', path_effects=text_outline)
        cbar.ax.tick_params(labelcolor='white', labelsize=10)
        for t in cbar.ax.get_yticklabels():
            t.set_path_effects(text_outline)
            t.set_fontweight('bold')
    elif strm is not None:
        cax = fig.add_axes([cax_x0, cax_y0, cax_w, cax_h])
        cbar = fig.colorbar(strm.lines, cax=cax, orientation='vertical', extend='both')
        cbar.set_label('Ocean Current Temp (°C)', color='white', weight='bold', path_effects=text_outline)
        cbar.ax.tick_params(labelcolor='white', labelsize=10)
        for t in cbar.ax.get_yticklabels():
            t.set_path_effects(text_outline)
            t.set_fontweight('bold')

    # --- CLEAN BANNER TITLE ---
    time_str = datetime.now().strftime("%I:%M %p EDT %a %b %d %Y")
    title_str = f'{title_suffix} for {storm_name} ({storm_id})\nNational Hurricane Center Miami, FL\nValid: {time_str}'
    
    if official_forecast and 'init' in official_forecast:
        init_time = official_forecast['init']
        title_str += f' | Initialized: {init_time.strftime("%Hz %b %d %Y")}'
        
        if show_cone_and_forecast:
            summary_title = f"SUMMARY OF {datetime.now(timezone.utc).strftime('%H%M UTC %d %b %Y')}Z...INFORMATION"
            separator = "----------------------------------------------"
            lat_str = f"{lat:.1f}N" if lat >= 0 else f"{abs(lat):.1f}S"
            lon_str = f"{abs(lon):.1f}W" if lon < 0 else f"{lon:.1f}E"
            nearest_loc = min(ATLANTIC_BASIN_LOCATIONS, key=lambda loc: haversine(lat, lon, loc['lat'], loc['lon']))
            dist_mi = int(haversine(lat, lon, nearest_loc['lat'], nearest_loc['lon']))
            speed_mph_val = int(storm_speed_mph) if storm_speed_mph else 0
            mslp_val = int(float(mslp)) if mslp != 'N/A' else 1000
            
            initial_conditions = (
                f"{summary_title}\n"
                f"{separator}\n"
                f"LOCATION...{lat_str} {lon_str}\n"
                f"ABOUT {dist_mi} MI...{int(dist_mi * 1.852)} KM {nearest_loc['name']}\n"
                f"MAXIMUM SUSTAINED WINDS...{int(initial_vmax * 1.15078)} MPH\n"
                f"PRESENT MOVEMENT...{storm_dir_str or 'NNW'} OR {int(motion_dir_deg)} DEGREES AT {speed_mph_val} MPH\n"
                f"MINIMUM CENTRAL PRESSURE...{mslp_val} MB"
            )
            
            ax.text(0.01, 0.01, initial_conditions, transform=ax.transAxes, color='white', fontsize=8.0, va='bottom', ha='left', family='monospace', bbox=dict(facecolor=BG_COLOR, alpha=0.8, edgecolor='black', pad=5), path_effects=text_outline, zorder=12)

    title_y = ax_y0 + ax_h + (top_margin_in / 2) / fig_height
    fig.text(ax_x0, title_y, title_str, fontsize=14, fontweight='bold', color='white', va='center', ha='left', path_effects=text_outline)

    # --- COMPREHENSIVE CUSTOM LEGEND ALIGNED TO BOTTOM RIGHT ---
    if show_cone_and_forecast:
        custom_handles = []
        custom_labels = []

        custom_handles.append(Line2D([0], [0], color='white', linewidth=2.5, path_effects=text_outline))
        custom_labels.append('Observed Track')
        
        # Reverted back to a dashed red line!
        custom_handles.append(Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, path_effects=text_outline))
        custom_labels.append('Official Forecast')
        
        custom_handles.append(Patch(facecolor='white', alpha=0.4, edgecolor='black'))
        custom_labels.append('NHC Cone')

        # --- COASTAL WARNINGS ---
        custom_handles.append(Line2D([0], [0], color='yellow', linewidth=3.5, path_effects=text_outline))
        custom_labels.append('TS Watch')

        custom_handles.append(Line2D([0], [0], color='blue', linewidth=3.5, path_effects=text_outline))
        custom_labels.append('TS Warning')

        custom_handles.append(Line2D([0], [0], color='magenta', linewidth=6.0, path_effects=text_outline))
        custom_labels.append('Hurr Watch')

        custom_handles.append(Line2D([0], [0], color='red', linewidth=6.0, path_effects=text_outline))
        custom_labels.append('Hurr Warning')

        # Tuple for Mix Warning (Blue solid over thick Magenta solid)
        mix_base = Line2D([0], [0], color='magenta', linewidth=6.0, path_effects=text_outline)
        mix_top = Line2D([0], [0], color='blue', linewidth=3.5)
        custom_handles.append((mix_base, mix_top))
        custom_labels.append('TS Warn / Hurr Watch')
        
        # --- STORM CATEGORIES ---
        cat_legend_items = [
            ('TD', '#00BFFF', '^'),
            ('TS', '#00FA9A', 'o'),
            ('Cat 1', '#FFD700', 'o'),
            ('Cat 2', '#FFA500', 'o'),
            ('Cat 3', '#FF4500', 'o'),
            ('Cat 4', '#DA70D6', 'o'),
            ('Cat 5', '#FF00FF', 'o')
        ]
        
        for label, color, marker in cat_legend_items:
            custom_handles.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markeredgecolor='black', markersize=8, linestyle='None'))
            custom_labels.append(label)

        num_cols = (len(custom_handles) + 2) // 3

        legend = ax.legend(
            handles=custom_handles, 
            labels=custom_labels,
            loc='lower right', 
            fontsize=8, 
            facecolor=AX_COLOR, 
            edgecolor='white', 
            framealpha=1.0, 
            ncol=num_cols,
            labelspacing=0.4, 
            columnspacing=1.0, 
            borderpad=0.5, 
            handletextpad=0.5, 
            handlelength=2.0
        )
        
        legend.set_zorder(20) 
        
        for text in legend.get_texts(): 
            text.set_color('white')
            text.set_path_effects(text_outline)

    # --- DYNAMIC TIMESTAMP FILE NAMING ---
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    map_image_path = os.path.join(output_dir, f'{storm_id}_{title_suffix.lower().replace(" ", "_")}_{timestamp_str}.png')
    plt.savefig(map_image_path, dpi=150, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close()
    print(f"{title_suffix} plot generated for {storm_name} ({storm_id}). Image saved to: {map_image_path}")

async def create_intensity_plot(tracks, title_suffix, storm_name, storm_id, vmax, mslp, official_forecast=None, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    # Increased figure height slightly to give the legend more breathing room
    fig = plt.figure(figsize=(12, 9), facecolor=BG_COLOR)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(AX_COLOR)
    
    initial_vmax = float(vmax) if vmax != 'N/A' else 35.0
    categories = {'Cat 5': (137, 200, '#DA70D6', 0.2), 'Cat 4': (113, 136, '#FF00FF', 0.2), 'Cat 3': (96, 112, '#FF4500', 0.2), 'Cat 2': (83, 95, '#FFA500', 0.2), 'Cat 1': (64, 82, '#FFD700', 0.2), 'TS': (34, 63, '#00FA9A', 0.2)}
    for cat, (ymin, ymax, color, alpha) in categories.items(): ax.axhspan(ymin, ymax, facecolor=color, alpha=alpha)
    for b in [64, 83, 96, 113, 137]: ax.axhline(b, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    for cat, y_pos in {'Cat 5': 137, 'Cat 4': 113, 'Cat 3': 96, 'Cat 2': 83, 'Cat 1': 64, 'TS': 34}.items(): ax.text(2, y_pos + 1, cat, color='white', va='bottom', ha='left', fontsize=10, fontweight='bold', path_effects=pe_stroke)
        
    ax.set_xlim(0, 168); ax.set_ylim(0, 210)
    ax.set_xlabel('Forecast Hour', fontsize=12, color='white', fontweight='bold'); ax.xaxis.label.set_path_effects(pe_stroke)
    ax.set_ylabel('Wind Speed (kt)', fontsize=12, color='white', fontweight='bold'); ax.yaxis.label.set_path_effects(pe_stroke)
    ax.tick_params(axis='both', colors='white')
    for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_color('white'); label.set_path_effects(pe_stroke)
    for spine in ax.spines.values(): spine.set_color('white')
    ax.grid(True, linestyle='--', color='gray', alpha=0.4)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(tracks)))
    for i, (model, track) in enumerate(tracks.items()):
        color = colors[i % len(colors)]
        plot_fhr, plot_vmax = np.array(track['fhr']), np.maximum(np.array(track['vmax']), 0)
        if len(plot_fhr) > 0:
            if plot_fhr[0] == 0: plot_vmax[0] = initial_vmax
            else: plot_fhr, plot_vmax = np.insert(plot_fhr, 0, 0), np.insert(plot_vmax, 0, initial_vmax)
        ax.plot(plot_fhr, plot_vmax, color=color, linewidth=2, path_effects=pe_stroke, label=model)
        if len(plot_fhr) > 0: ax.text(plot_fhr[-1] + 1 if plot_fhr[-1] < 168 else 169, max(plot_vmax[-1], 0), model, color=color, fontsize=10, fontweight='bold', va='center', path_effects=pe_stroke)
            
    if tracks:
        common_fhrs = np.arange(0, 169, 6)
        all_vmax = np.full((len(tracks), len(common_fhrs)), np.nan)
        for i, track in enumerate(tracks.values()):
            plot_fhr, plot_vmax = np.array(track['fhr']), np.maximum(np.array(track['vmax']), 0)
            if len(plot_fhr) > 0:
                if plot_fhr[0] == 0: plot_vmax[0] = initial_vmax
                else: plot_fhr, plot_vmax = np.insert(plot_fhr, 0, 0), np.insert(plot_vmax, 0, initial_vmax)
            all_vmax[i] = np.interp(common_fhrs, plot_fhr, plot_vmax, left=np.nan, right=np.nan)
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_vmax, q25, q75 = np.nanmean(all_vmax, axis=0), np.nanpercentile(all_vmax, 25, axis=0), np.nanpercentile(all_vmax, 75, axis=0)
        ax.fill_between(common_fhrs, q25, q75, color='gray', alpha=0.4, label='IQR (25-75%)')
        ax.plot(common_fhrs, mean_vmax, color='white', linestyle='--', linewidth=3, path_effects=pe_stroke, label='Consensus (Mean)')
        
    ax.scatter(0, initial_vmax, color='cyan', edgecolors='black', s=100, zorder=10, label='Initial Intensity')
    
    # --- DYNAMIC HORIZONTAL BOTTOM LEGEND ---
    # Calculate a clean number of columns based on how many tracks are rendered
    total_legend_items = len(tracks) + (3 if tracks else 1) # Tracks + IQR, Consensus, Initial
    num_cols = min(6, max(3, total_legend_items // 5))
    
    legend = ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), # Anchors it nicely below the x-axis label
        ncol=num_cols,               # Spreads the list horizontally
        facecolor=BG_COLOR, 
        edgecolor='white',
        fontsize=9
    )
    for text in legend.get_texts(): text.set_color('white'); text.set_path_effects(pe_stroke)
        
    ax.set_title(f"{title_suffix} for {storm_name} ({storm_id})", color='white', fontsize=16, fontweight='bold', pad=10, path_effects=pe_stroke)
    if official_forecast: ax.text(0.02, 0.98, f"Initialized at {official_forecast['init'].strftime('%Hz %b %d %Y')}", color='white', transform=ax.transAxes, fontsize=12, va='top', path_effects=pe_stroke)
        
    # --- DYNAMIC TIMESTAMP FILE NAMING ---
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(output_dir, f'{storm_id}_{title_suffix.lower().replace(" ", "_")}_intensity_{timestamp_str}.png')
    plt.savefig(image_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"{title_suffix} intensity plot generated for {storm_name} ({storm_id}). Image saved to: {image_path}")

async def spaghetti_ATL(storm_id: str, show_surge: bool = False):
    try: os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e: print(f"Error creating output directory: {e}"); return
        
    async with aiohttp.ClientSession() as session:
        try:
            storm_id = storm_id.upper()
            if not storm_id.startswith('AL') or len(storm_id) != 8 or not storm_id[-4:].isdigit():
                print("Invalid storm ID. Please use format like 'AL082026'."); return
                
            rt = realtime_obj.Realtime()
            storm = None
            try: storm = rt.get_storm(storm_id)
            except Exception: print(f"\n[INFO] {storm_id} is no longer in the active NHC feed. Falling back to raw ATCF data...")

            atcf_url = f"https://ftp.nhc.noaa.gov/atcf/aid_public/a{storm_id.lower()}.dat.gz"
            try:
                async with session.get(atcf_url) as response:
                    if response.status != 200: raise Exception(f"Failed to fetch ATCF: {response.status}")
                    with gzip.GzipFile(fileobj=BytesIO(await response.read())) as gz: atcf_data = gz.read().decode('utf-8')
            except Exception:
                atcf_data = None
                print(f"Could not fetch ATCF forecast data for {storm_id}. It may be completely inactive.")
                if not storm: return
                
            df = pd.DataFrame()
            if atcf_data:
                lines = atcf_data.splitlines()
                max_fields = max(len(line.split(',')) for line in lines)
                column_names = ['basin', 'number', 'date', 'technum', 'model', 'fhr', 'lat', 'lon', 'vmax', 'mslp', 'type', 'rad', 'windcode', 'rad1', 'rad2', 'rad3', 'rad4', 'pouter', 'router', 'rmw', 'gust', 'eye', 'subregion', 'maxseas', 'initials', 'dir', 'speed', 'stormname', 'depth', 'seas', 'seacode', 'seas1', 'seas2', 'seas3', 'seas4']
                if max_fields > len(column_names): column_names.extend([f'extra_{i}' for i in range(len(column_names), max_fields)])
                df = pd.read_csv(StringIO(atcf_data), header=None, names=column_names[:max_fields], skipinitialspace=True, low_memory=False)
                
                def parse_coord(x):
                    try:
                        s = str(x).strip()
                        if not s or s == 'nan': return np.nan
                        val = float(s[:-1]) / 10.0
                        if s[-1] in ['S', 'W']: val = -val
                        return val
                    except: return np.nan

                df['lat'] = df['lat'].apply(parse_coord)
                df['lon'] = df['lon'].apply(parse_coord)
                df['fhr'] = pd.to_numeric(df['fhr'], errors='coerce')
                df['vmax'] = pd.to_numeric(df['vmax'], errors='coerce')
                df['mslp'] = pd.to_numeric(df['mslp'], errors='coerce')

            obs_lats, obs_lons, lat, lon = [], [], np.nan, np.nan
            vmax, mslp = 'N/A', 'N/A'
            storm_name = f"INVEST {storm_id[2:4]}" if storm_id.startswith('AL9') else storm_id
            storm_dir_str, storm_speed_mph, official_forecast, radii_34, radii_64 = None, None, None, {}, {}

            if storm:
                obs_lats = np.array(storm.lat) if isinstance(storm.lat, (list, np.ndarray)) else np.array([storm.lat])
                obs_lons = np.array(storm.lon) if isinstance(storm.lon, (list, np.ndarray)) else np.array([storm.lon])
                obs_vmaxs = np.array(storm.vmax) if isinstance(storm.vmax, (list, np.ndarray)) else np.array([storm.vmax])
                lat = obs_lats[-1] if len(obs_lats) > 0 else np.nan
                lon = obs_lons[-1] if len(obs_lons) > 0 else np.nan
                vmax = obs_vmaxs[-1] if len(obs_vmaxs) > 0 else storm.vmax
                mslp = storm.mslp[-1] if isinstance(storm.mslp, (list, np.ndarray)) else storm.mslp
                storm_name = storm.name
                try:
                    info = storm.get_realtime_info(source="public_advisory")
                    storm_dir_str = info.get('motion_direction')
                    storm_speed_mph = info.get('motion_mph')
                except: pass
                try:
                    official_forecast = storm.get_forecast_realtime()
                    if official_forecast:
                        for k in ['lat', 'lon', 'vmax', 'mslp', 'fhr']: official_forecast[k] = np.asarray(official_forecast[k])
                except: pass
            else:
                if not df.empty:
                    init_df = df[df['fhr'] == 0].sort_values('date')
                    if not init_df.empty:
                        lat, lon = init_df['lat'].iloc[-1], init_df['lon'].iloc[-1]
                        vmax = init_df['vmax'].iloc[-1] if pd.notna(init_df['vmax'].iloc[-1]) else 'N/A'
                        mslp = init_df['mslp'].iloc[-1] if pd.notna(init_df['mslp'].iloc[-1]) else 'N/A'
                    else:
                        lat, lon = df['lat'].iloc[-1], df['lon'].iloc[-1]
                    obs_lats, obs_lons = np.array([lat]), np.array([lon])
            
            storm_type = "Tropical Storm" if (vmax == 'N/A' or float(vmax) < 64) else "Hurricane"

            nhc_df, parsed_radii_34, parsed_radii_64, init_time_nhc, adv_num, tcm_storm_name = await fetch_nhc_advisory(storm_id, session)
            
            if tcm_storm_name:
                storm_name = tcm_storm_name
                
            if nhc_df is not None and not nhc_df.empty:
                radii_34.update(parsed_radii_34)
                radii_64.update(parsed_radii_64)
                
                init_row = nhc_df[nhc_df['Hour'] == 0]
                if not init_row.empty:
                    lat = init_row['Lat'].iloc[0]
                    lon = init_row['Lon'].iloc[0]
                    vmax = init_row['Vmax'].iloc[0]
                    mslp_val = init_row['MSLP'].iloc[0]
                    if not pd.isna(mslp_val):
                        mslp = mslp_val
                        
                official_forecast = {
                    'lat': np.asarray(nhc_df['Lat'].values),
                    'lon': np.asarray(nhc_df['Lon'].values),
                    'fhr': np.asarray(nhc_df['Hour'].values),
                    'vmax': np.asarray(nhc_df['Vmax'].values),
                    'mslp': np.asarray(nhc_df['MSLP'].values),
                    'radii_34': radii_34,
                    'radii_64': radii_64,
                    'init': init_time_nhc or datetime.now(timezone.utc)
                }
            
            if vmax != 'N/A':
                if float(vmax) < 34: storm_type = "Tropical Depression"
                elif float(vmax) < 64: storm_type = "Tropical Storm"
                else: storm_type = "Hurricane"
                
            # Initialize separate dictionaries for each ensemble suite
            deterministic_tracks, gefs_tracks, geps_tracks, eps_tracks, intensity_tracks = {}, {}, {}, {}, {}
            
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_numeric(df['date'].astype(str).str.strip(), errors='coerce')
                
                ai_models = ['AEMN', 'AIFS', 'GCAI', 'GCFS', 'PANG', 'FOUR']
                deterministic_models = ['GFSO', 'HMON', 'HWRF', 'CTCX', 'AVNO', 'UKX', 'CMC', 'ECMF', 'NVGM'] + ai_models
                
                # --- PARSE ENSEMBLE MEMBERS BY ATCF PREFIX ---
                gefs_members = [m for m in df['model'].unique() if str(m).startswith(('GE', 'AP'))]
                geps_members = [m for m in df['model'].unique() if str(m).startswith(('CP', 'CC'))]
                eps_members  = [m for m in df['model'].unique() if str(m).startswith(('EP', 'EE'))]
                
                target_models = deterministic_models + gefs_members + geps_members + eps_members
                df = df[df['model'].isin(target_models)].copy()
                
                if not df.empty:
                    latest_dates = df.groupby('model')['date'].transform('max')
                    df = df[df['date'] == latest_dates]
                    df = df.sort_values(['model', 'fhr']).drop_duplicates(subset=['model', 'fhr'], keep='last')
                    
                    for model in df['model'].unique():
                        model_df = df[df['model'] == model]
                        track = {'lat': model_df['lat'].values, 'lon': model_df['lon'].values, 'fhr': model_df['fhr'].values, 'vmax': model_df['vmax'].values, 'mslp': model_df['mslp'].values}
                        
                        intensity_tracks[model] = track # Add all models to the intensity spaghetti plot
                        
                        # Sort into correct map products
                        if model in deterministic_models: deterministic_tracks[model] = track
                        elif model in gefs_members: gefs_tracks[model] = track
                        elif model in geps_members: geps_tracks[model] = track
                        elif model in eps_members: eps_tracks[model] = track
                    
            # Create plots for each type of track
            if deterministic_tracks: 
                await create_plot(deterministic_tracks, f"{storm_type} {storm_name} Model Track Guidance", session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, adv_num=adv_num, is_ensemble=False, show_surge=show_surge, output_dir=OUTPUT_DIR)
            else:
                print(f"[INFO] No Deterministic track data available for {storm_id} at this time.")

            if gefs_tracks: 
                await create_plot(gefs_tracks, f"{storm_type} {storm_name} GEFS Tracks", session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, adv_num=adv_num, is_ensemble=True, ensemble_name='GEFS', show_surge=show_surge, output_dir=OUTPUT_DIR)
            else:
                print(f"[INFO] No GEFS data available for {storm_id} at this time.")

            if geps_tracks: 
                await create_plot(geps_tracks, f"{storm_type} {storm_name} GEPS Tracks", session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, adv_num=adv_num, is_ensemble=True, ensemble_name='GEPS', show_surge=show_surge, output_dir=OUTPUT_DIR)
            else:
                print(f"[INFO] No GEPS (Canadian Ensemble) data available for {storm_id} at this time.")

            if eps_tracks: 
                await create_plot(eps_tracks, f"{storm_type} {storm_name} EPS Tracks", session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, adv_num=adv_num, is_ensemble=True, ensemble_name='EPS', show_surge=show_surge, output_dir=OUTPUT_DIR)
            else:
                print(f"[INFO] No EPS (European Ensemble) data available for {storm_id} at this time.")
            
            if official_forecast: 
                await create_plot({}, f"NHC Cone Forecast", session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, adv_num=adv_num, show_cone_and_forecast=True, show_surge=show_surge, output_dir=OUTPUT_DIR)
            
            if intensity_tracks: 
                await create_intensity_plot(intensity_tracks, f"{storm_type} {storm_name} Model Intensity Guidance", storm_name, storm_id, vmax, mslp, official_forecast=official_forecast, output_dir=OUTPUT_DIR)
            
            if not any([deterministic_tracks, gefs_tracks, geps_tracks, eps_tracks, official_forecast, intensity_tracks]): 
                print(f"No data available to plot for {storm_name} ({storm_id}).")
        except Exception as e: print(f"An error occurred while processing spaghetti maps for {storm_id}: {e}")

async def marine_key_messages_ATL(storm_id: Optional[str] = None):
    if storm_id is None: print("Please provide a storm ID."); return
    try: os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e: print(f"Error creating output directory: {e}"); return
        
    storm_id = storm_id.upper()
    if not storm_id.startswith('AL') or len(storm_id) != 8 or not storm_id[-4:].isdigit(): print("Invalid storm ID."); return
        
    async with aiohttp.ClientSession() as session:
        try:
            rt = realtime_obj.Realtime()
            storm = rt.get_storm(storm_id)
            if not storm: print(f"No active storm found with ID {storm_id}."); return
            
            storm_name = storm.name if storm else f"INVEST {storm_id[2:4]}"
            obs_lats = np.array(storm.lat) if isinstance(storm.lat, (list, np.ndarray)) else np.array([storm.lat])
            obs_lons = np.array(storm.lon) if isinstance(storm.lon, (list, np.ndarray)) else np.array([storm.lon])
            lat, lon = obs_lats[-1] if len(obs_lats) > 0 else 25.0, obs_lons[-1] if len(obs_lons) > 0 else -75.0
            
            official_forecast = storm.get_forecast_realtime()
            if official_forecast is None: official_forecast = {'lat': np.linspace(lat, lat + 5, 5), 'lon': np.linspace(lon, lon - 2, 5), 'fhr': np.arange(0, 120, 24), 'vmax': np.linspace(50, 80, 5), 'mslp': np.full(5, 990.0)}
            fc_lons, fc_lats, fc_fhrs = np.array(official_forecast['lon']), np.array(official_forecast['lat']), np.array(official_forecast['fhr'])
            
            try:
                async with session.get("https://www.nhc.noaa.gov/text/MIAHSFAT2.shtml") as response: marine_text = await response.text()
            except Exception: marine_text = ""
                
            storm_section = (re.search(rf'(Tropical Storm|Hurricane)\s+{re.escape(storm_name)}\s+.*?(?=\n\n[A-Z]{{4}}|\Z)', marine_text, re.DOTALL | re.IGNORECASE) or type('obj', (object,), {'group': lambda self, i: ""})()).group(0)
            current_winds_kt = int((re.search(r'maximum sustained winds of (\d+) KT', storm_section) or type('obj', (object,), {'group': lambda self, i: "50"})()).group(1))
            current_seas_ft = round(float((re.search(r'seas.*?up to ([\d.]+) M', storm_section) or type('obj', (object,), {'group': lambda self, i: "7.0"})()).group(1)) * 3.28084)
            forecast_24h_winds = int((re.search(r'24-hour forecast.*?winds of (\d+) KT', storm_section) or type('obj', (object,), {'group': lambda self, i: "70"})()).group(1))
            
            try: info = storm.get_realtime_info(source="public_advisory"); motion_dir_str, motion_speed_mph = info.get('motion_direction', 'N'), info.get('motion_mph', 8)
            except: motion_dir_str, motion_speed_mph = 'N', 8
                
            now = datetime.now()
            key_messages = [f'Winds this afternoon are {current_winds_kt} kt with seas to {current_seas_ft} ft.', f"{storm_name}'s wind field and seas will expand as it moves {motion_dir_str.lower() if motion_dir_str else 'north'} at {motion_speed_mph * 0.868976:.0f} kt into {(now + timedelta(days=1)).strftime('%A').lower()}; forecast to {'become a hurricane' if forecast_24h_winds >= 64 else 'intensify further'}", f"{storm_name} should then accelerate toward the open Atlantic into {(now + timedelta(days=3)).strftime('%A').lower()}, but very hazardous seas will persist.", 'See latest NHC advisory at www.hurricanes.gov']
            
            async def get_latest_gfswave_url(session):
                utcnow = datetime.now(timezone.utc)
                for delta_h in range(0, 72, 6):
                    dt = utcnow - timedelta(hours=delta_h)
                    if dt.hour not in [0, 6, 12, 18]: continue
                    url = f"https://ftpprd.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs.{dt.strftime('%Y%m%d')}/{dt.hour:02d}/gfswave.t{dt.hour:02d}z.atlocn.0p16.grib2"
                    try:
                        async with session.head(url) as resp:
                            if resp.status == 200: return url
                    except Exception: pass
                return None
                
            url = await get_latest_gfswave_url(session)
            hsig_atl = None
            if url:
                try:
                    async with session.get(url) as resp: content = await resp.read()
                    import cfgrib
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp: tmp.write(content); tmp_path = tmp.name
                    try:
                        ds = xr.open_dataset(tmp_path, engine='cfgrib')
                        hsig_atl = (ds['htsgw'].sel(step=pd.Timedelta(hours=120)) * 3.28084).squeeze().sel(longitude=slice(-100, -20), latitude=slice(5, 60), method='nearest')
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                except Exception: hsig_atl = None
                    
            fig = plt.figure(figsize=(16, 10), facecolor=BG_COLOR)
            gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.05)
            ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
            ax_map.set_facecolor(AX_COLOR)
            ax_map.set_extent([-85, -60, 20, 40], crs=ccrs.PlateCarree())
            ax_map.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1.5); ax_map.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor='black'); ax_map.add_feature(cfeature.OCEAN, facecolor=AX_COLOR); ax_map.add_feature(cfeature.BORDERS, edgecolor='black', linestyle=':')
            
            if len(obs_lats) > 1: ax_map.plot(obs_lons, obs_lats, color='white', linewidth=2.5, path_effects=pe_stroke, transform=ccrs.PlateCarree(), label='Observed Track')
            ax_map.plot(fc_lons, fc_lats, color='red', linestyle='--', linewidth=2.5, path_effects=pe_stroke, transform=ccrs.PlateCarree(), label='Forecast Track')
            ax_map.scatter(lon, lat, marker='o', color='red', s=150, edgecolor='black', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
            ax_map.text(lon + 0.5, lat, f'{storm_name}\n{current_winds_kt} kt', color='white', path_effects=pe_stroke, fontweight='bold', transform=ccrs.PlateCarree(), fontsize=12)
            
            if len(fc_fhrs) > 4:
                idx_120 = np.argmin(np.abs(fc_fhrs - 120))
                ax_map.scatter(fc_lons[idx_120], fc_lats[idx_120], marker='*', color='yellow', s=300, edgecolor='black', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
                ax_map.text(fc_lons[idx_120] + 0.5, fc_lats[idx_120], '120h Forecast', color='white', path_effects=pe_stroke, fontweight='bold', transform=ccrs.PlateCarree(), fontsize=11)
                
            if hsig_atl is not None:
                cf = ax_map.contourf(hsig_atl.longitude, hsig_atl.latitude, hsig_atl, levels=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36], colors=['#00FFFF', '#00CCFF', '#0099FF', '#0066FF', '#0033FF', '#0000FF', '#9900FF', '#CC00FF', 'red'], extend='max', transform=ccrs.PlateCarree(), alpha=0.7)
                ax_map.text(lon + 1, lat + 1, 'Significant Wave Heights (ft)', color='white', path_effects=pe_stroke, transform=ccrs.PlateCarree(), fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor=BG_COLOR, alpha=0.8, edgecolor='white'))
            else: ax_map.text(lon + 1, lat + 1, 'Wave data unavailable', color='white', path_effects=pe_stroke, transform=ccrs.PlateCarree(), fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor=BG_COLOR, alpha=0.8, edgecolor='red'))
                
            scale_x, y_scale = [-84, -63], 21
            for i, (h, col) in enumerate(zip([4, 8, 12, 16, 20, 24, 28, 32, 36], ['#00FFFF', '#00CCFF', '#0099FF', '#0066FF', '#0033FF', '#0000FF', '#9900FF', '#CC00FF', 'red'])):
                x_start = scale_x[0] + i * ((scale_x[1] - scale_x[0]) / 9)
                ax_map.add_patch(Rectangle((x_start, y_scale), ((scale_x[1] - scale_x[0]) / 9), 0.3, facecolor=col, edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree()))
                ax_map.text(x_start + ((scale_x[1] - scale_x[0]) / 9)/2, y_scale + 0.5, f'{h}', color='white', path_effects=pe_stroke, fontweight='bold', ha='center', transform=ccrs.PlateCarree(), fontsize=10)
            ax_map.text((scale_x[0] + scale_x[1])/2, y_scale - 1, 'Wave Heights (ft)', color='white', path_effects=pe_stroke, ha='center', transform=ccrs.PlateCarree(), fontsize=12, fontweight='bold')
            ax_map.set_title('120h Forecast Seas', color='white', fontsize=14, fontweight='bold', pad=10, path_effects=pe_stroke)
            
            ax_text = fig.add_subplot(gs[1])
            ax_text.set_facecolor(BG_COLOR); ax_text.axis('off'); ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
            ax_text.text(0.5, 0.95, f'Marine Key Messages\n{now.strftime("%I%M %p EDT %b %d %Y")}', color='white', ha='center', va='top', fontsize=16, fontweight='bold', path_effects=pe_stroke)
            for i, msg in enumerate(key_messages): ax_text.text(0.05, 0.82 - i*0.18, f'• {msg}', color='white', ha='left', va='top', fontsize=13, wrap=True, bbox=dict(facecolor=AX_COLOR, alpha=0.5, edgecolor='none', pad=5))
            ax_text.text(0.5, 0.05, 'For more marine information visit www.hurricanes.gov/marine\nwww.nhc.noaa.gov', color='gray', ha='center', va='bottom', fontsize=10, style='italic')
            
            # --- DYNAMIC TIMESTAMP FILE NAMING ---
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            map_image_path = os.path.join(OUTPUT_DIR, f'marine_{storm_id}_{timestamp_str}.png')
            plt.savefig(map_image_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
            plt.close()
            print(f"Marine key messages plot generated for {storm_name} ({storm_id}). Image saved to: {map_image_path}")
        except Exception as e: print(f"An error occurred: {e}")

if __name__ == "__main__":
    async def main():
        print("Tropical Atlantic Command-Line Tool")
        import sys
        if len(sys.argv) < 2: print("No command provided."); return
        command, storm_id = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
        show_surge = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
        if command == "spaghetti_atl": await spaghetti_ATL(storm_id, show_surge)
        elif command == "marine_key_messages_atl": await marine_key_messages_ATL(storm_id)
        else: print(f"Unknown command: {command}")
    asyncio.run(main())