import os
import logging
import aiohttp
import asyncio
import gzip
from io import BytesIO, StringIO
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
from typing import Optional
import warnings
import tempfile 
import ssl
import zipfile
import xml.etree.ElementTree as ET

# --- SSL Patch for DoD Servers & Tropycal ---
def create_unverified_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl._create_default_https_context = create_unverified_context

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Output directory & Theme Constants
OUTPUT_DIR = 'outputs'
BG_COLOR = '#333333'
AX_COLOR = '#2B2B2B'
TEXT_COLOR = 'white'
LAND_COLOR = '#444444'

WEST_PAC_LOCATIONS = [
    {"name": "Tokyo, Japan", "lat": 35.68, "lon": 139.65},
    {"name": "Osaka, Japan", "lat": 34.69, "lon": 135.50},
    {"name": "Fukuoka, Japan", "lat": 33.59, "lon": 130.40},
    {"name": "Sapporo, Japan", "lat": 43.06, "lon": 141.35},
    {"name": "Naha, Okinawa", "lat": 26.21, "lon": 127.68},
    {"name": "Taipei, Taiwan", "lat": 25.03, "lon": 121.56},
    {"name": "Kaohsiung, Taiwan", "lat": 22.62, "lon": 120.31},
    {"name": "Manila, Philippines", "lat": 14.59, "lon": 120.98},
    {"name": "Cebu City, Philippines", "lat": 10.31, "lon": 123.89},
    {"name": "Davao City, Philippines", "lat": 7.19, "lon": 125.42},
    {"name": "Hanoi, Vietnam", "lat": 21.02, "lon": 105.83},
    {"name": "Ho Chi Minh City, Vietnam", "lat": 10.82, "lon": 106.62},
    {"name": "Da Nang, Vietnam", "lat": 16.05, "lon": 108.20},
    {"name": "Hong Kong", "lat": 22.31, "lon": 114.16},
    {"name": "Macau", "lat": 22.19, "lon": 113.54},
    {"name": "Shanghai, China", "lat": 31.23, "lon": 121.47},
    {"name": "Guangzhou, China", "lat": 23.12, "lon": 113.26},
    {"name": "Shenzhen, China", "lat": 22.54, "lon": 114.05},
    {"name": "Fuzhou, China", "lat": 26.07, "lon": 119.29},
    {"name": "Seoul, South Korea", "lat": 37.56, "lon": 126.97},
    {"name": "Busan, South Korea", "lat": 35.10, "lon": 129.03},
    {"name": "Jeju City, South Korea", "lat": 33.49, "lon": 126.53},
    {"name": "Guam (US)", "lat": 13.44, "lon": 144.79},
    {"name": "Saipan (US)", "lat": 15.19, "lon": 145.74},
    {"name": "Koror, Palau", "lat": 7.34, "lon": 134.47},
    {"name": "Vladivostok, Russia", "lat": 43.11, "lon": 131.87},
]

# --- PROJECTION SETUP (Himawari-9) ---
map_proj = ccrs.PlateCarree()
goes_proj = ccrs.Geostationary(central_longitude=140.7, satellite_height=35786023.0)
sat_height = 35786023.0
half_angle = 0.151872
x_max = sat_height * math.tan(half_angle)
y_max = x_max
y_min = -x_max
x_min = -x_max
pe_stroke = [pe.withStroke(linewidth=2, foreground='black')]

# --- HELPER FUNCTIONS ---
def plot_jtwc_kmz(kmz_path, ax):
    """ Extracts and plots the JTWC forecast track and cone of uncertainty from a KMZ file. """
    try:
        with zipfile.ZipFile(kmz_path, 'r') as kmz:
            kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
            if not kml_files:
                print(f"[WARNING] No KML file found inside {kmz_path}")
                return
            kml_data = kmz.read(kml_files[0]).decode('utf-8')
            
        # Strip all KML namespaces globally to completely clean the XML element structure
        kml_data = re.sub(r'\sxmlns[^=]*="[^"]*"', '', kml_data)
        kml_data = re.sub(r'<([/]?)[a-zA-Z0-9_-]+:', r'<\1', kml_data)
        root = ET.fromstring(kml_data)

        # 3. Parse and Plot the Cone of Uncertainty (Polygons)
        for polygon in root.findall('.//Polygon'):
            coords_text = polygon.find('.//coordinates')
            if coords_text is not None and coords_text.text:
                coords = coords_text.text.strip().split()
                lon_lat = []
                for c in coords:
                    if ',' in c:
                        parts = c.split(',')
                        if len(parts) >= 2:
                            try:
                                lon_lat.append((float(parts[0]), float(parts[1])))
                            except ValueError:
                                pass
                if lon_lat:
                    lons, lats = zip(*lon_lat)
                    ax.fill(lons, lats, color='white', alpha=0.25, 
                            transform=ccrs.PlateCarree(), zorder=4.2, 
                            edgecolor='white', linestyle='--', linewidth=1.5)

        # 4. Parse and Plot the Track (LineStrings)
        for linestring in root.findall('.//LineString'):
            coords_text = linestring.find('.//coordinates')
            if coords_text is not None and coords_text.text:
                coords = coords_text.text.strip().split()
                lon_lat = []
                for c in coords:
                    if ',' in c:
                        parts = c.split(',')
                        if len(parts) >= 2:
                            try:
                                lon_lat.append((float(parts[0]), float(parts[1])))
                            except ValueError:
                                pass
                if lon_lat:
                    lons, lats = zip(*lon_lat)
                    ax.plot(lons, lats, color='red', linewidth=3, 
                            transform=ccrs.PlateCarree(), zorder=5.0,
                            path_effects=[pe.withStroke(linewidth=5, foreground='black'), pe.Normal()])

        # 5. Parse and Plot the Forecast Points (Points inside Placemarks)
        for placemark in root.findall('.//Placemark'):
            point = placemark.find('.//Point')
            if point is not None:
                coords_text = point.find('.//coordinates')
                if coords_text is not None and coords_text.text:
                    parts = coords_text.text.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            lon, lat = float(parts[0]), float(parts[1])
                            ax.scatter(lon, lat, color='red', edgecolors='black', s=50, 
                                       transform=ccrs.PlateCarree(), zorder=6.0)
                        except ValueError:
                            pass
                    
        print(f"[DEBUG] Successfully mapped KMZ track/cone from {kmz_path}")

    except Exception as e:
        print(f"[ERROR] Failed to parse or plot KMZ file: {e}")

def fetch_ocean_currents(extent):
    print("\n[DEBUG] --- MARINE DATA FETCH (RTOFS Direct Download with Cache) ---")
    import time
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    temp_file = os.path.join(cache_dir, "rtofs_today.nc")
    download_needed = True

    if os.path.exists(temp_file):
        file_age = time.time() - os.path.getmtime(temp_file)
        file_size = os.path.getsize(temp_file)
        if file_age < 43200 and file_size > 10000000:
            print(f"[DEBUG] Found valid cached RTOFS data ({int(file_age/60)} mins old). Skipping download.", flush=True)
            download_needed = False
        else:
            print("[DEBUG] Cached file is old or corrupted. Redownloading...", flush=True)
            try: os.remove(temp_file)
            except: pass

    if download_needed:
        print("[DEBUG] Attempting to fetch RTOFS data via direct HTTPS...")
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
                print(f"[DEBUG] Trying {url} ...", end=" ", flush=True)
                try:
                    head = session.head(url, timeout=10)
                    if head.status_code != 200:
                        print(f"Not found (Status: {head.status_code}).")
                        continue
                    print("Downloading...", end=" ", flush=True)
                    r = session.get(url, stream=True, timeout=30)
                    if r.status_code == 200:
                        with open(temp_file, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk: f.write(chunk)
                        success = True
                        print("Done.")
                        break
                    else:
                        print(f"Failed (Status: {r.status_code}).")
                except Exception as e:
                    print(f"Error ({e}).")
                    if os.path.exists(temp_file):
                        try: os.remove(temp_file)
                        except: pass
                    continue
        
        if not success:
            print("[DEBUG] All direct RTOFS downloads failed.")
            return None, None, None, None, None, None

    print("[DEBUG] Extracting RTOFS data...", end=" ", flush=True)
    try:
        ds = nc.Dataset(temp_file, 'r')
        lat_var = next((v for v in ['lat', 'Latitude', 'latitude'] if v in ds.variables), None)
        lon_var = next((v for v in ['lon', 'Longitude', 'longitude'] if v in ds.variables), None)

        lats_raw = ds.variables[lat_var][:]
        lons_raw = ds.variables[lon_var][:]
        
        # Safe fallback if extent crosses 180 (antimeridian)
        lon_min, lon_max = extent[0] - 2, extent[1] + 2
        if lon_max < lon_min: lon_min, lon_max = -180, 180

        lons_raw = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
        lat_min, lat_max = extent[2] - 2, extent[3] + 2

        if lats_raw.ndim == 2:
            mask = (lats_raw >= lat_min) & (lats_raw <= lat_max) & (lons_raw >= lon_min) & (lons_raw <= lon_max)
            y_idxs, x_idxs = np.where(mask)
        else:
            y_mask = (lats_raw >= lat_min) & (lats_raw <= lat_max)
            x_mask = (lons_raw >= lon_min) & (lons_raw <= lon_max)
            y_idxs = np.where(y_mask)[0]
            x_idxs = np.where(x_mask)[0]

        if len(y_idxs) == 0 or len(x_idxs) == 0:
            print("[DEBUG] Extent not in grid.")
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

async def fetch_himawari_s3():
    print("[DEBUG] Fetching Himawari-9 Band 14 (IR) via s3fs AWS Bucket...")
    def _fetch():
        try:
            import s3fs
            import xarray as xr
            import re
        except ImportError:
            print("[DEBUG] s3fs or xarray not installed. Cannot fetch Himawari via AWS.")
            return None
            
        fs = s3fs.S3FileSystem(anon=True)
        base_paths = ["noaa-himawari9/AHI-L2-FLDK-ISatSS", "noaa-himawari9/AHI-L1b-FLDK"]
        now_utc = datetime.now(timezone.utc)

        for base_path in base_paths:
            for i in range(12):
                current_time = now_utc - timedelta(minutes=i * 10)
                rounded_minute = (current_time.minute // 10) * 10
                current_time = current_time.replace(minute=rounded_minute, second=0, microsecond=0)
                s3_path = f"{base_path}/{current_time:%Y/%m/%d/%H%M}/"

                try:
                    if fs.exists(s3_path):
                        files = fs.glob(f"{s3_path}*.nc")
                        if files:
                            # 1. STRICT FILTER: Only grab Clean IR Window (Band 13) 
                            # NOAA Himawari files use 'C13' (CMI) or 'B13' (L1b)
                            target_files = [f for f in files if ('C13' in f or 'B13' in f)]
                            
                            # 2. GROUP BY EXACT SCAN TIME to avoid mixing different timestamps
                            scan_groups = {}
                            for f in target_files:
                                # Find the start time block (e.g., _s20261721610000_)
                                match = re.search(r'_s(\d+)_', f)
                                if match:
                                    scan_groups.setdefault(match.group(1), []).append(f)
                            
                            if scan_groups:
                                # Grab the latest complete scan
                                best_scan = max(scan_groups.keys())
                                candidate_files = sorted(scan_groups[best_scan]) # Sorting orders T001 to T010 seamlessly
                                candidate_paths = [f"s3://{f}" for f in candidate_files]

                                print(f"[DEBUG] Found {len(candidate_paths)} candidate tiles for scan {best_scan}. Verifying shapes before stitching...")

                                shape_groups = {}
                                for p in candidate_paths:
                                    try:
                                        f_obj = fs.open(p, 'rb')
                                        with xr.open_dataset(f_obj, engine="h5netcdf", chunks={}) as probe:
                                            shape_key = tuple(sorted(probe.sizes.items()))
                                        shape_groups.setdefault(shape_key, []).append(p)
                                    except Exception as e:
                                        print(f"[DEBUG] Could not probe {p}: {e}")

                                if not shape_groups:
                                    print(f"[DEBUG] No readable tiles for scan {best_scan}. Trying earlier scan...")
                                    continue

                                best_shape, final_paths = max(shape_groups.items(), key=lambda kv: len(kv[1]))
                                discarded = len(candidate_paths) - len(final_paths)
                                print(f"[DEBUG] Using {len(final_paths)} tiles with consistent shape {dict(best_shape)} for scan {best_scan}" + (f" (discarded {discarded} mismatched tiles)." if discarded else "."))

                                if len(final_paths) > 1:
                                    print("[DEBUG] Stitching tiles...")
                                    s3_file_objs = [fs.open(p, 'rb') for p in final_paths]
                                    ds = xr.open_mfdataset(
                                        s3_file_objs, 
                                        engine="h5netcdf", 
                                        combine='by_coords',
                                        compat='override',     
                                        coords='minimal',      # <-- PREVENTS XARRAY CONFLICT 
                                        combine_attrs='override',
                                        join='outer',          
                                        data_vars='minimal'    
                                    ).load()
                                    return ds
                                else:
                                    print(f"[DEBUG] Loading single file: {final_paths[0]}")
                                    with fs.open(final_paths[0], 'rb') as f:
                                        ds = xr.open_dataset(f, engine="h5netcdf").load()
                                    return ds
                except Exception as e:
                    print(f"[DEBUG] Error reading {s3_path}: {e}")
                    pass
                
        print("[DEBUG] Failed to find Himawari data on S3 after 120-minute lookback.")
        return None
        
    return await asyncio.to_thread(_fetch)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return (R * c) / 1.852

def calculate_bearing_and_speed(lat1, lon1, lat2, lon2, hours=6):
    if np.isnan(lat1) or np.isnan(lat2): return 0, 0
    dLon = math.radians(lon2 - lon1)
    y = math.sin(dLon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dLon)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
    dist_nm = haversine(lat1, lon1, lat2, lon2)
    speed_kt = dist_nm / hours if hours > 0 else 0
    return bearing, speed_kt

def get_category_color(vmax_kt):
    if vmax_kt < 34: return '#00BFFF', '^' 
    elif vmax_kt < 64: return '#00FA9A', 'o' 
    elif vmax_kt < 83: return '#FFD700', 'o' 
    elif vmax_kt < 96: return '#FFA500', 'o' 
    elif vmax_kt < 113: return '#FF4500', 'o' 
    elif vmax_kt < 130: return '#DA70D6', 'o' 
    else: return '#FF00FF', 'o' 

def estimate_wind_radii(V_th_kt, vmax_kt, mslp_mb, lat_deg, motion_dir_deg, motion_speed_kt):
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

    deltaP_mb_safe = max(deltaP_mb, 1.0)
    rmw_km = np.exp(2.635 - 0.00005086 * deltaP_mb_safe ** 2 + 0.0394899 * lat_deg)
    rmw_nm = rmw_km / 1.852
    rmw_nm = float(np.clip(rmw_nm, 5.0, 200.0))

    rmw_km = rmw_nm * 1.852
    rmw_m = rmw_km * 1000.0
    B = np.clip((vmax_ms ** 2 * rho * np.e) / deltaP_Pa, 0.8, 2.5)

    def dP_dr(r):
        if r <= 0: return 0.0
        A = (rmw_m / r) ** B
        return deltaP_Pa * (B / r) * A * np.exp(-A)

    def V_sym_ms(r):
        if r <= 0: return 0.0
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
    if not quad_radii or all(r == 0 for r in quad_radii): return
    angles = np.linspace(0, 360, 361)
    lons, lats = [], []
    for angle in angles:
        quad = int(angle // 90) % 4
        rad = quad_radii[quad]
        dlat = (rad / 60.0) * np.cos(np.deg2rad(angle))
        dlon = (rad / 60.0) * np.sin(np.deg2rad(angle)) / np.cos(np.deg2rad(center_lat))
        lons.append(center_lon + dlon)
        lats.append(center_lat + dlat)

    if label: ax.fill(lons, lats, color=color, alpha=alpha, edgecolor=None, transform=transform, clip_on=True, label=label, zorder=zorder)
    else: ax.fill(lons, lats, color=color, alpha=alpha, edgecolor=None, transform=transform, clip_on=True, zorder=zorder)
    ax.plot(lons + [lons[0]], lats + [lats[0]], color='black', linewidth=1.5, transform=transform, clip_on=True, zorder=zorder + 0.1)

async def fetch_jtwc_atcf(storm_id, session):
    print(f"\n[DEBUG] --- ATCF PARSING INITIATED FOR {storm_id} ---")
    obs_lats, obs_lons, lat, lon, vmax, mslp, storm_name = [], [], np.nan, np.nan, 'N/A', 'N/A', 'INVEST'
    kmz_file = None
    
    # 1. TROPYCAL B-DECK FETCH
    try:
        from tropycal import realtime
        print("[DEBUG] Using Tropycal to fetch observations...")
        dataset = await asyncio.to_thread(realtime.Realtime, jtwc=True)
        storm = await asyncio.to_thread(dataset.get_storm, storm_id)
        if storm:
            obs_lats = np.array(storm.lat)
            obs_lons = np.array(storm.lon)
            lat = obs_lats[-1] if len(obs_lats) > 0 else np.nan
            lon = obs_lons[-1] if len(obs_lons) > 0 else np.nan
            vmax = storm.vmax[-1] if len(storm.vmax) > 0 else 'N/A'
            mslp = storm.mslp[-1] if len(storm.mslp) > 0 else 'N/A'
            storm_name = storm.name.upper() if storm.name and storm.name.upper() != 'UNNAMED' else 'INVEST'
            print(f"[DEBUG] Tropycal fetch successful: {storm_name} at {lat}N, {lon}E")
    except Exception as e:
        print(f"[DEBUG] Tropycal fetch failed or storm not active: {e}")

    # 2. MANUAL A-DECK FETCH
    print(f"\n[DEBUG] Starting Model A-Deck Fetch...")
    def _fetch_adeck(s_id):
        import requests, gzip, urllib3, subprocess
        from io import BytesIO
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        basin, num, year = s_id[:2].lower(), s_id[2:4], s_id[4:]
        
        urls = [
            f"http://hurricanes.ral.ucar.edu/repository/data/adecks_active/a{basin}{num}{year}.dat",
            f"http://hurricanes.ral.ucar.edu/repository/data/adecks_open/a{basin}{num}{year}.dat",
            f"https://www.metoc.navy.mil/jtwc/products/atcf/{basin}/a{basin}{num}{year}.dat",
            f"https://ftp.nhc.noaa.gov/atcf/aid_public/a{basin}{num}{year}.dat.gz",
            f"https://www.nrlmry.navy.mil/atcf_web/docs/current_storms/a{basin}{num}{year}.dat",
            f"http://hurricanes.ral.ucar.edu/repository/data/adecks_open/{year}/a{basin}{num}{year}.dat",
            f"https://ftp.nhc.noaa.gov/atcf/archive/{year}/a{basin}{num}{year}.dat.gz"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Accept': 'text/plain,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Connection': 'keep-alive'
        }
        
        for url in urls:
            print(f"[DEBUG] Requesting A-Deck: {url}")
            try:
                r = requests.get(url, headers=headers, verify=False, timeout=15)
                if r.status_code == 403 and 'navy.mil' in url:
                    try:
                        res = subprocess.run(['curl', '-s', '-A', headers['User-Agent'], '-k', url], capture_output=True, text=True, timeout=15)
                        if res.returncode == 0 and res.stdout and '<html' not in res.stdout.lower() and res.stdout.count(',') > 10:
                            return res.stdout
                    except: pass
                
                if r.status_code == 200:
                    if url.endswith('.gz'):
                        with gzip.GzipFile(fileobj=BytesIO(r.content)) as gz: text_data = gz.read().decode('utf-8', errors='ignore')
                    else:
                        text_data = r.text
                    if '<html' not in text_data.lower() and text_data.count(',') > 10:
                        return text_data
            except: continue
        return None

    atcf_a_data = await asyncio.to_thread(_fetch_adeck, storm_id)
    df_a = pd.DataFrame()
    official_forecast = None
    radii_34, radii_64 = {}, {}
    
    if atcf_a_data:
        lines = atcf_a_data.splitlines()
        column_names = ['basin', 'number', 'date', 'technum', 'model', 'fhr', 'lat', 'lon', 'vmax', 'mslp', 'type', 'rad', 'windcode', 'rad1', 'rad2', 'rad3', 'rad4', 'pouter', 'router', 'rmw', 'gust', 'eye', 'subregion', 'maxseas', 'initials', 'dir', 'speed', 'stormname', 'depth', 'seas', 'seacode', 'seas1', 'seas2', 'seas3', 'seas4']
        max_fields = max(len(line.split(',')) for line in lines) if lines else len(column_names)
        if max_fields > len(column_names): column_names.extend([f'extra_{i}' for i in range(len(column_names), max_fields)])
        df_a = pd.read_csv(StringIO(atcf_a_data), header=None, names=column_names[:max_fields], skipinitialspace=True, low_memory=False)
        
        def parse_coord(x):
            try:
                s = str(x).strip()
                if not s or s == 'nan': return np.nan
                val = float(s[:-1]) / 10.0
                if s[-1] in ['S', 'W']: val = -val
                return val
            except: return np.nan
        
        df_a['lat'] = df_a['lat'].apply(parse_coord)
        df_a['lon'] = df_a['lon'].apply(parse_coord)
        df_a['fhr'] = pd.to_numeric(df_a['fhr'], errors='coerce')
        df_a['vmax'] = pd.to_numeric(df_a['vmax'], errors='coerce')
        df_a['mslp'] = pd.to_numeric(df_a['mslp'], errors='coerce')
        df_a['model'] = df_a['model'].astype(str).str.strip()
        
        df_a = df_a[(df_a['lat'].notna()) & (df_a['lon'].notna())]
        df_a = df_a[(df_a['lat'] != 0.0) | (df_a['lon'] != 0.0)]
        
        if 'stormname' in df_a.columns and storm_name == 'INVEST':
            valid_names = df_a['stormname'].dropna().astype(str).str.strip().unique()
            for n in valid_names:
                if n.upper() not in ['INVEST', 'NONAME', 'NN', '']:
                    storm_name = n.upper()
                    break

        jtwc_df = df_a[(df_a['model'].isin(['JTWC', 'OFCL'])) & (df_a['fhr'] >= 0)]
        if not jtwc_df.empty:
            latest_date = jtwc_df['date'].max()
            ofcl = jtwc_df[jtwc_df['date'] == latest_date].sort_values('fhr')
            try: init_dt = datetime.strptime(str(int(latest_date)), '%Y%m%d%H').replace(tzinfo=timezone.utc)
            except: init_dt = datetime.now(timezone.utc)
            
            official_forecast = {
                'lat': ofcl['lat'].values, 'lon': ofcl['lon'].values, 
                'fhr': ofcl['fhr'].values, 'vmax': ofcl['vmax'].values, 
                'mslp': ofcl['mslp'].values, 'init': init_dt
            }
            print(f"[DEBUG] Successfully extracted Official JTWC Forecast Cone from A-Deck.")
        else:
            print("[DEBUG] Official forecast not in A-Deck. Attempting direct JTWC Warning fetch...")
            import subprocess
            import re
            
            b_n = storm_id[:4].lower()
            yy = storm_id[6:8]
            
            warning_urls = [
                f"https://www.metoc.navy.mil/jtwc/products/warnings/{b_n}{yy}web.txt",
                f"https://www.metoc.navy.mil/jtwc/products/warnings/{b_n}22web.txt",
                f"https://www.metoc.navy.mil/jtwc/products/warnings/{b_n}23web.txt",
                f"https://www.metoc.navy.mil/jtwc/products/warnings/{b_n}24web.txt"
            ]
            
            headers_curl = ['-A', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36']
            success_cone = False
            
            for w_url in warning_urls:
                if success_cone: break
                try:
                    w_res = subprocess.run(['curl', '-s', '-k'] + headers_curl + [w_url], capture_output=True, text=True, timeout=10)
                    if w_res.returncode == 0 and w_res.stdout and 'MAX SUSTAINED WINDS' in w_res.stdout.upper():
                        print(f"[DEBUG] Found JTWC Advisory at: {w_url}")
                        track_matches = re.findall(r'(INITIAL|\d{2}H?)\s+(\d{2}/\d{2}Z?)\s+(\d+\.?\d*)N?\s+(\d+\.?\d*)E?\s+(\d+)KT', w_res.stdout, re.IGNORECASE | re.MULTILINE)
                        if track_matches:
                            fc_lats = [float(t[2]) for t in track_matches]
                            fc_lons = [float(t[3]) for t in track_matches]
                            fc_fhrs = [0 if t[0].upper() == 'INITIAL' else int(t[0].upper().replace('H', '')) for t in track_matches]
                            fc_vmax = [float(t[4]) for t in track_matches]
                            
                            official_forecast = {
                                'lat': np.array(fc_lats), 'lon': np.array(fc_lons), 
                                'fhr': np.array(fc_fhrs), 'vmax': np.array(fc_vmax), 
                                'mslp': np.full(len(fc_fhrs), 1010.0), 'init': datetime.now(timezone.utc)
                            }
                            print("[DEBUG] JTWC Official Forecast parsed successfully via direct text URL.")
                            success_cone = True
                except Exception: pass
                
            if not success_cone:
                print(f"[DEBUG] Could not fetch official forecast from A-Deck or direct JTWC URL.")
                try:
                    kmz_url = f"https://www.weather.gov/source/gum/tropical/{storm_id}_CONE_latest.kmz"
                    print(f"[DEBUG] Attempting to download official JTWC cone KMZ from: {kmz_url}")
                    kmz_resp = requests.get(kmz_url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
                    if kmz_resp.status_code == 200 and len(kmz_resp.content) > 1000:
                        kmz_path = os.path.join(OUTPUT_DIR, f"{storm_id}_JTWC_official_cone.kmz")
                        with open(kmz_path, 'wb') as f:
                            f.write(kmz_resp.content)
                        kmz_file = kmz_path
                        print(f"[DEBUG] ✅ Saved official JTWC cone KMZ to: {kmz_path}")
                except Exception as e:
                    print(f"[DEBUG] Could not download official KMZ fallback: {e}")

        if np.isnan(lat):
            init_rows = df_a[df_a['fhr'] == 0]
            if not init_rows.empty:
                lat = init_rows['lat'].mode()[0] if not init_rows['lat'].mode().empty else init_rows['lat'].iloc[0]
                lon = init_rows['lon'].mode()[0] if not init_rows['lon'].mode().empty else init_rows['lon'].iloc[0]
                vmax = init_rows['vmax'].max() 
                mslp = init_rows['mslp'].min() 

        print(f"[DEBUG] Successfully parsed A-Deck models.")
    else:
        print("[DEBUG] Failed to fetch A-Deck. Models will not be plotted.")

    print("[DEBUG] --- ATCF PARSING COMPLETE ---\n")
    return np.array(obs_lats), np.array(obs_lons), lat, lon, vmax, mslp, storm_name, df_a, official_forecast, radii_34, radii_64, kmz_file

def get_category_label(vmax_kt):
    if vmax_kt < 34: return 'D'
    elif vmax_kt < 64: return 'S'
    elif vmax_kt < 83: return '1'
    elif vmax_kt < 96: return '2'
    elif vmax_kt < 113: return '3'
    elif vmax_kt < 130: return '4'
    else: return '5'

async def create_plot(tracks, title_suffix, session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, is_ensemble=False, show_cone_and_forecast=False, ensemble_name=None, output_dir=OUTPUT_DIR, kmz_file=None):
    os.makedirs(output_dir, exist_ok=True)
    cone_radii = {0: 15, 12: 26, 24: 39, 36: 52, 48: 67, 60: 83, 72: 100, 96: 142, 120: 213}
    
    initial_vmax = float(vmax) if vmax != 'N/A' else 35.0
    motion_dir_deg = 0
    if storm_dir_str and storm_dir_str != 'N/A':
        dirs = {'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5}
        motion_dir_deg = dirs.get(storm_dir_str.upper(), 0)
    storm_speed_kt = float(storm_speed_mph) * 0.868976 if storm_speed_mph else 0.0

    all_lons, all_lats = [], []
    if not np.isnan(lon) and not np.isnan(lat):
        all_lons.append(lon); all_lats.append(lat)
    if official_forecast:
        all_lons.extend(official_forecast['lon']); all_lats.extend(official_forecast['lat'])
    for track in tracks.values():
        all_lons.extend(track['lon']); all_lats.extend(track['lat'])

    if all_lons and all_lats:
        if not np.isnan(lat) and not np.isnan(lon):
            valid_lons, valid_lats = [], []
            for lo, la in zip(all_lons, all_lats):
                lon_diff = min(abs(lo - lon), 360 - abs(lo - lon))
                if lon_diff < 40 and abs(la - lat) < 35: 
                    valid_lons.append(lo)
                    valid_lats.append(la)
            if valid_lons:
                all_lons, all_lats = valid_lons, valid_lats

        all_lons_rad = np.deg2rad(all_lons)
        all_lons_unwrapped = np.rad2deg(np.unwrap(all_lons_rad))
        
        pad = 8.0 
        min_lon, max_lon = min(all_lons_unwrapped) - pad, max(all_lons_unwrapped) + pad
        min_lat, max_lat = min(all_lats) - pad, max(all_lats) + pad
        
        width = max_lon - min_lon
        height = max_lat - min_lat
        max_dim = max(width, height)
        
        center_lon = (max_lon + min_lon) / 2.0
        center_lat = (max_lat + min_lat) / 2.0
        
        EXTENT_LON_MIN = (center_lon - max_dim / 2.0 + 180) % 360 - 180
        EXTENT_LON_MAX = (center_lon + max_dim / 2.0 + 180) % 360 - 180
        EXTENT_LAT_MIN = center_lat - max_dim / 2.0
        EXTENT_LAT_MAX = center_lat + max_dim / 2.0
    else:
        EXTENT_LON_MIN, EXTENT_LON_MAX = 110, 160
        EXTENT_LAT_MIN, EXTENT_LAT_MAX = 5, 50

    extent_list = [EXTENT_LON_MIN, EXTENT_LON_MAX, EXTENT_LAT_MIN, EXTENT_LAT_MAX]

    fig_width = 13.0 
    left_margin_in = 0.55
    right_margin_in = 1.15 
    top_margin_in = 1.2
    bottom_margin_in = 0.55
    
    ax_width_in = fig_width - left_margin_in - right_margin_in
    map_aspect = 1.0 
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

    ds_sat = await fetch_himawari_s3()
    lats_oc, lons_oc, w_temp, w_u, w_v, salinity = await asyncio.to_thread(fetch_ocean_currents, extent_list)

    try:
        color_stops_data = [(-100, "#F0F0F0"), (-80, "#000058"), (-60, "#0202BB"), (-40, "#15589B"), (-20, "#4180BB"), (-10, "#AD006B"), (-5, "#B6009E"), (0, "#FF01FF"), (5, "#7F1BB1"), (10, "#BA55D3"), (15, "#DA70D6"), (20, "#E0B0FF"), (25, "#AFEEEE"), (30, "#00FFFF"), (35, "#00FF00"), (40, "#32CD32"), (45, "#7CFC00"), (50, "#ADFF2F"), (55, "#FFFF00"), (60, "#FFD700"), (65, "#FFA500"), (70, "#FF8C00"), (75, "#FF4500"), (80, "#FF0000"), (85, "#DC143C"), (90, "#B22222"), (95, "#8B0000"), (100, "#A52A2A"), (105, "#8B4513"), (110, "#A0522D"), (115, "#CD853F"), (120, "#D2B48C"), (125, "#FFDDC1"), (130, "#FFFFFF")]
        sst_cmap = mcolors.LinearSegmentedColormap.from_list('ga_temp_cmap', [((v - -100) / (130 - -100), c) for v, c in color_stops_data])
        sst_norm = mcolors.Normalize(vmin=-73.333, vmax=54.444)
        sst_url = f"https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{datetime.now(timezone.utc).year}.nc"
        with xr.open_dataset(sst_url) as ds:
            sst = ds['sst'].isel(time=-1).sortby('lat').sortby('lon')
            lon_slice_min = (EXTENT_LON_MIN - 5) % 360
            lon_slice_max = (EXTENT_LON_MAX + 5) % 360
            
            if lon_slice_max < lon_slice_min:
                sst1 = sst.sel(lat=slice(EXTENT_LAT_MIN - 5, EXTENT_LAT_MAX + 5), lon=slice(lon_slice_min, 360))
                sst2 = sst.sel(lat=slice(EXTENT_LAT_MIN - 5, EXTENT_LAT_MAX + 5), lon=slice(0, lon_slice_max))
                sst_region = xr.concat([sst1, sst2], dim='lon')
            else:
                sst_region = sst.sel(lat=slice(EXTENT_LAT_MIN - 5, EXTENT_LAT_MAX + 5), lon=slice(lon_slice_min, lon_slice_max))
                
            levels = np.arange(-10, 35, 1)
            ax.contourf(sst_region.lon, sst_region.lat, sst_region, transform=ccrs.PlateCarree(), levels=levels, cmap=sst_cmap, norm=sst_norm, alpha=1.0, zorder=0.5)
            sst_contours = ax.contour(sst_region.lon, sst_region.lat, sst_region, transform=ccrs.PlateCarree(), levels=levels, cmap=sst_cmap, norm=sst_norm, linewidths=1.5, alpha=0.9, zorder=2.5)
            for label in ax.clabel(sst_contours, inline=True, fontsize=9, fmt='%1.0f°C'):
                label.set_path_effects(text_outline)
                label.set_fontweight('bold')
    except Exception as e: pass

    if ds_sat is not None:
        try:
            lon_0 = 140.7
            if 'nominal_satellite_subpoint_lon' in ds_sat.attrs: lon_0 = float(ds_sat.attrs['nominal_satellite_subpoint_lon'])
            elif 'nominal_satellite_subpoint_lon' in ds_sat.variables: lon_0 = float(ds_sat['nominal_satellite_subpoint_lon'].values[0])
                
            h = 35785863.0
            if 'nominal_satellite_height' in ds_sat.attrs: h = float(ds_sat.attrs['nominal_satellite_height'])
            elif 'nominal_satellite_height' in ds_sat.variables: h = float(ds_sat['nominal_satellite_height'].values[0])
            
            sweep = 'x'
            if 'sweep_angle_axis' in ds_sat.attrs: sweep = ds_sat.attrs['sweep_angle_axis']
            elif 'sweep_angle_axis' in ds_sat.variables:
                val = ds_sat['sweep_angle_axis'].values
                sweep = val.item() if hasattr(val, 'item') else val[0]
            if isinstance(sweep, bytes): sweep = sweep.decode('utf-8')

            # Fix: Prioritize 1D scanner angles with Geostationary lookup to bypass Cartopy meshgrid bugs
            if 'x' in ds_sat and 'y' in ds_sat:
                x_val = ds_sat['x'].values
                y_val = ds_sat['y'].values
                if np.nanmax(np.abs(x_val)) < 1.0: 
                    x_sat = x_val * h
                    y_sat = y_val * h
                    sat_proj = ccrs.Geostationary(central_longitude=lon_0, satellite_height=h, sweep_axis=sweep)
                elif np.nanmax(np.abs(x_val)) <= 360.0:
                    x_sat = x_val
                    y_sat = y_val
                    sat_proj = ccrs.PlateCarree()
                else:
                    x_sat = x_val
                    y_sat = y_val
                    sat_proj = ccrs.Geostationary(central_longitude=lon_0, satellite_height=h, sweep_axis=sweep)
            elif 'longitude' in ds_sat and 'latitude' in ds_sat:
                x_sat = ds_sat['longitude'].values
                y_sat = ds_sat['latitude'].values
                sat_proj = ccrs.PlateCarree()
            elif 'lon' in ds_sat and 'lat' in ds_sat:
                x_sat = ds_sat['lon'].values
                y_sat = ds_sat['lat'].values
                sat_proj = ccrs.PlateCarree()
            else:
                raise ValueError("Could not locate spatial coordinates in dataset.")

            arr_sat = None
            var_candidates = ['tbb', 'tb', 'CldTopTemp', 'CMI', 'temp', 'Rad', 'Sectorized_CMI']
            for var in var_candidates:
                if var in ds_sat.data_vars:
                    data = ds_sat[var].squeeze().values.astype(float)
                    if len(data.shape) != 2: continue
                    
                    if var == 'Rad':
                        c1 = 1.191042e-5; c2 = 1.4387752; nu = 892.4 
                        data = np.where(data <= 0.01, np.nan, data)
                        arr_sat = ((c2 * nu) / np.log((c1 * (nu ** 3)) / data + 1)) - 273.15
                    else:
                        arr_sat = data - 273.15 if np.nanmax(data) > 150 else data
                    print(f"[DEBUG] Rendered satellite variable: {var} (Max Temp: {np.nanmax(arr_sat):.1f}°C)")
                    break
            
            if arr_sat is None:
                for var in ds_sat.data_vars:
                    data = ds_sat[var].squeeze().values.astype(float)
                    if len(data.shape) == 2:
                        arr_sat = data - 273.15 if np.nanmax(data) > 150 else data
                        print(f"[DEBUG] Fallback rendered variable: {var}")
                        break

            C_WHITE = (1.0, 1.0, 1.0, 1.0)
            C_BLACK = (0.0, 0.0, 0.0, 1.0)
            C_RED = (1.0, 0.0, 0.0, 1.0)
            C_ORANGE = (1.0, 0.5, 0.0, 1.0)
            C_GREEN = (0.0, 1.0, 0.0, 1.0)
            C_BLUE = (0.0, 0.0, 1.0, 1.0)
            C_PURPLE = (0.5, 0.0, 0.5, 1.0)
            C_TRANS1 = (0.7, 0.7, 0.7, 0.3)
            C_TRANS2 = (0.3, 0.3, 0.3, 0.0)

            node_levels = np.array([-110, -75, -65, -55, -45, -28, -20, -15, 0, 10, 56])
            node_colors = [C_WHITE, C_BLACK, C_RED, C_ORANGE, C_GREEN, C_BLUE, C_PURPLE, C_WHITE, C_TRANS1, C_TRANS2, C_TRANS2]

            span = 56 - (-110)
            normalized_nodes = (node_levels - (-110)) / span
            cmap_data = list(zip(normalized_nodes, node_colors))
            ir_cmap = mcolors.LinearSegmentedColormap.from_list('custom_ir_cmap_trans', cmap_data)
            ir_norm = mcolors.Normalize(vmin=-110, vmax=56)

            skip = 2
            x_plot = x_sat.squeeze()
            y_plot = y_sat.squeeze()
            
            # Fix: Safely map slices across either 1D projection markers or 2D meshgrids
            if x_plot.ndim == 2:
                x_p = x_plot[::skip, ::skip]
                y_p = y_plot[::skip, ::skip]
            else:
                x_p = x_plot[::skip]
                y_p = y_plot[::skip]

            ax.pcolormesh(x_p, y_p, arr_sat[::skip, ::skip],
                          cmap=ir_cmap, norm=ir_norm, transform=sat_proj, zorder=2.0, shading='auto')

            ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', zorder=3.0)
            ax.add_feature(cfeature.OCEAN, facecolor='none', edgecolor='none', zorder=3.0)

        except Exception as e:
            print(f"[DEBUG] Failed to plot Himawari IR: {e}")
            ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor='black', zorder=1.0)
            ax.add_feature(cfeature.OCEAN, facecolor=AX_COLOR, zorder=1.0)
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

    gl = ax.gridlines(draw_labels=True, alpha=0.6, color='#3932A0', linestyle='--', zorder=3.0, path_effects=[pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()])
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'white', 'path_effects': text_outline}
    gl.ylabel_style = {'size': 10, 'color': 'white', 'path_effects': text_outline}

    if 0 not in radii_34 and not np.isnan(lat): radii_34[0] = estimate_wind_radii(34, initial_vmax, float(mslp) if mslp != 'N/A' else 1010, lat, motion_dir_deg, storm_speed_kt)
    if 0 not in radii_64 and not np.isnan(lat): radii_64[0] = estimate_wind_radii(64, initial_vmax, float(mslp) if mslp != 'N/A' else 1010, lat, motion_dir_deg, storm_speed_kt)

    if len(obs_lats) > 0 and len(obs_lons) > 0:
        if obs_lats[-1] != lat or obs_lons[-1] != lon:
            plot_obs_lats, plot_obs_lons = np.append(obs_lats, lat), np.append(obs_lons, lon)
        else:
            plot_obs_lats, plot_obs_lons = obs_lats, obs_lons
            
        ax.plot(plot_obs_lons, plot_obs_lats, color='white', linewidth=2.5, path_effects=text_outline, transform=ccrs.PlateCarree(), zorder=4.0, clip_on=True)
        ax.scatter(plot_obs_lons[:-1], plot_obs_lats[:-1], color='white', s=20, edgecolors='black', transform=ccrs.PlateCarree(), zorder=4.1, clip_on=True)

    if show_cone_and_forecast:
        if kmz_file and os.path.exists(kmz_file):
            plot_jtwc_kmz(kmz_file, ax)
            
        if official_forecast:
            fc_lats, fc_lons, fc_fhrs = official_forecast['lat'], official_forecast['lon'], official_forecast['fhr']
            fc_vmaxs = official_forecast.get('vmax', np.full(len(fc_fhrs), initial_vmax))
            fc_mslps = official_forecast.get('mslp', np.full(len(fc_fhrs), float(mslp) if mslp != 'N/A' else 1010))
            
            valid_idx = fc_fhrs > 0
            fc_lats, fc_lons, fc_fhrs, fc_vmaxs, fc_mslps = fc_lats[valid_idx], fc_lons[valid_idx], fc_fhrs[valid_idx], fc_vmaxs[valid_idx], fc_mslps[valid_idx]
            
            if not np.isnan(lat) and not np.isnan(lon):
                cone_lats, cone_lons, cone_fhrs = np.insert(fc_lats, 0, lat), np.insert(fc_lons, 0, lon), np.insert(fc_fhrs, 0, 0)
            else:
                cone_lats, cone_lons, cone_fhrs = fc_lats, fc_lons, fc_fhrs
                
            ax.plot(cone_lons, cone_lats, color='red', linestyle='--', linewidth=2.5, path_effects=text_outline, transform=ccrs.PlateCarree(), zorder=5.0, clip_on=True)
            
            from scipy.interpolate import interp1d
            
            cone_radii_base = {0: 15, 12: 26, 24: 39, 36: 52, 48: 67, 60: 83, 72: 100, 96: 142, 120: 213, 168: 300}
            interp_cone = interp1d(list(cone_radii_base.keys()), list(cone_radii_base.values()), fill_value="extrapolate")

            points = []
            for i, fhr in enumerate(cone_fhrs):
                if fhr <= 168:
                    radius_nm = interp_cone(fhr)
                    radius_deg = radius_nm / 60.0
                    theta = np.linspace(0, 2 * np.pi, 100)
                    
                    lons_circle = cone_lons[i] + radius_deg * np.cos(theta) / np.cos(np.deg2rad(cone_lats[i]))
                    lats_circle = cone_lats[i] + radius_deg * np.sin(theta)
                    points.extend(list(zip(lons_circle, lats_circle)))
            if points:
                try:
                    hull_points = np.array([points[i] for i in ConvexHull(points).vertices])
                    ax.fill(hull_points[:, 0], hull_points[:, 1], color='white', alpha=0.4, transform=ccrs.PlateCarree(), zorder=4.5, clip_on=True)
                    hull_lons = np.append(hull_points[:, 0], hull_points[0, 0])
                    hull_lats = np.append(hull_points[:, 1], hull_points[0, 1])
                    ax.plot(hull_lons, hull_lats, color='white', linewidth=2.0, path_effects=text_outline, transform=ccrs.PlateCarree(), zorder=4.6, clip_on=True)
                except QhullError: pass
                
            for i, (f_lat, f_lon, f_vmax, f_fhr) in enumerate(zip(fc_lats, fc_lons, fc_vmaxs, fc_fhrs)):
                if f_fhr in cone_radii:
                    q34 = radii_34.get(f_fhr) or estimate_wind_radii(34, f_vmax, fc_mslps[i], f_lat, motion_dir_deg, storm_speed_kt)
                    q64 = radii_64.get(f_fhr) or estimate_wind_radii(64, f_vmax, fc_mslps[i], f_lat, motion_dir_deg, storm_speed_kt)
                    plot_wind_field(ax, f_lon, f_lat, q64, 'red', alpha=0.45, transform=ccrs.PlateCarree(), zorder=4.9)
                    plot_wind_field(ax, f_lon, f_lat, q34, 'yellow', alpha=0.45, transform=ccrs.PlateCarree(), zorder=4.8)
                    
                    color, marker = get_category_color(f_vmax)
                    cat_label = get_category_label(f_vmax)
                    ax.scatter(f_lon, f_lat, color=color, marker=marker, s=140, edgecolors='black', linewidths=1, transform=ccrs.PlateCarree(), zorder=6.0, clip_on=True)
                    ax.text(f_lon, f_lat, cat_label, color='black', fontsize=7, fontweight='bold', ha='center', va='center', transform=ccrs.PlateCarree(), zorder=6.1, clip_on=True)
                    
                    if 'init' in official_forecast:
                        valid_time = official_forecast['init'] + timedelta(hours=int(f_fhr))
                        label_str = valid_time.strftime('%d %b %Hz')
                    else: label_str = f'{int(f_fhr)}h'
                    ax.text(f_lon + 0.4, f_lat + 0.3, label_str, color='white', fontsize=9, fontweight='bold', path_effects=text_outline, transform=ccrs.PlateCarree(), zorder=7.0, clip_on=True)

    if 0 in radii_64: plot_wind_field(ax, lon, lat, radii_64[0], 'red', alpha=0.55, transform=ccrs.PlateCarree(), zorder=4.9)
    if 0 in radii_34: plot_wind_field(ax, lon, lat, radii_34[0], 'yellow', alpha=0.55, transform=ccrs.PlateCarree(), zorder=4.8)

    if not np.isnan(lat) and not np.isnan(lon):
        color, marker = get_category_color(initial_vmax)
        cat_label = get_category_label(initial_vmax)
        ax.scatter(lon, lat, color=color, marker=marker, s=200, edgecolors='black', linewidths=1.5, transform=ccrs.PlateCarree(), zorder=10.0, clip_on=True)
        ax.text(lon, lat, cat_label, color='black', fontsize=9, fontweight='heavy', ha='center', va='center', transform=ccrs.PlateCarree(), zorder=10.1, clip_on=True)

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
            if np.isnan(plot_mslp_clean).all(): plot_mslp_clean = np.full_like(plot_mslp_clean, 1000.0)
            
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

    time_str_box = datetime.now(timezone.utc).strftime("%H00 UTC %d %b %Y")
    if official_forecast and 'init' in official_forecast:
        time_str_box = official_forecast['init'].strftime("%H%M UTC %d %b %Y")
    
    ax.text(0.02, 0.98, time_str_box, transform=ax.transAxes, fontsize=11, fontweight='bold', color='white', va='top', ha='left', path_effects=text_outline, bbox=dict(facecolor=AX_COLOR, alpha=0.8, edgecolor='black', pad=5), zorder=12)

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

    time_str = datetime.now(timezone.utc).strftime("%H%M UTC %a %b %d %Y")
    title_str = f'{title_suffix} for {storm_name} ({storm_id})\nJoint Typhoon Warning Center Pearl Harbor, HI\nValid: {time_str}'
    
    if official_forecast and 'init' in official_forecast:
        init_time = official_forecast['init']
        title_str += f' | Initialized: {init_time.strftime("%Hz %b %d %Y")}'
        
        if show_cone_and_forecast:
            summary_title = f"SUMMARY OF {datetime.now(timezone.utc).strftime('%H%M UTC %d %b %Y')}Z...INFORMATION"
            separator = "----------------------------------------------"
            lat_str = f"{lat:.1f}N" if lat >= 0 else f"{abs(lat):.1f}S"
            lon_str = f"{abs(lon):.1f}W" if lon < 0 else f"{lon:.1f}E"
            nearest_loc = min(WEST_PAC_LOCATIONS, key=lambda loc: haversine(lat, lon, loc['lat'], loc['lon']))
            dist_mi = int(haversine(lat, lon, nearest_loc['lat'], nearest_loc['lon']))
            speed_mph_val = int(storm_speed_mph) if storm_speed_mph else 0
            mslp_val = int(float(mslp)) if mslp != 'N/A' else 1000
            
            initial_conditions = (
                f"{summary_title}\n"
                f"{separator}\n"
                f"LOCATION...{lat_str} {lon_str}\n"
                f"ABOUT {dist_mi} MI...{int(dist_mi * 1.852)} KM {nearest_loc['name']}\n"
                f"MAXIMUM SUSTAINED WINDS...{int(initial_vmax * 1.15078)} MPH\n"
                f"PRESENT MOVEMENT...{storm_dir_str or 'N/A'} OR {int(motion_dir_deg)} DEGREES AT {speed_mph_val} MPH\n"
                f"MINIMUM CENTRAL PRESSURE...{mslp_val} MB"
            )
            ax.text(0.01, 0.01, initial_conditions, transform=ax.transAxes, color='white', fontsize=8.0, va='bottom', ha='left', family='monospace', bbox=dict(facecolor=BG_COLOR, alpha=0.8, edgecolor='black', pad=5), path_effects=text_outline, zorder=12)

    title_y = ax_y0 + ax_h + (top_margin_in / 2) / fig_height
    fig.text(ax_x0, title_y, title_str, fontsize=14, fontweight='bold', color='white', va='center', ha='left', path_effects=text_outline)

    if show_cone_and_forecast:
        custom_handles = [
            Line2D([0], [0], color='white', linewidth=2.5, path_effects=text_outline),
            Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, path_effects=text_outline),
            Patch(facecolor='white', alpha=0.4, edgecolor='black')
        ]
        custom_labels = ['Observed Track', 'Official Forecast', 'Forecast Cone']
        
        cat_legend_items = [
            ('TD', '#00BFFF', '^'), ('TS', '#00FA9A', 'o'), ('Cat 1', '#FFD700', 'o'),
            ('Cat 2', '#FFA500', 'o'), ('Cat 3', '#FF4500', 'o'), ('Cat 4', '#DA70D6', 'o'),
            ('Cat 5', '#FF00FF', 'o')
        ]
        for label, color, marker in cat_legend_items:
            custom_handles.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markeredgecolor='black', markersize=8, linestyle='None'))
            custom_labels.append(label)

        legend = ax.legend(handles=custom_handles, labels=custom_labels, loc='lower right', fontsize=8, facecolor=AX_COLOR, edgecolor='white', framealpha=1.0, ncol=3, labelspacing=0.4, columnspacing=1.0, borderpad=0.5, handletextpad=0.5, handlelength=2.0)
        legend.set_zorder(20) 
        for text in legend.get_texts(): 
            text.set_color('white')
            text.set_path_effects(text_outline)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    map_image_path = os.path.join(output_dir, f'{storm_id}_{title_suffix.lower().replace(" ", "_")}_{timestamp_str}.png')
    plt.savefig(map_image_path, dpi=150, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close()
    print(f"[DEBUG] {title_suffix} plot successfully generated. Saved to: {map_image_path}")

async def create_intensity_plot(tracks, title_suffix, storm_name, storm_id, vmax, mslp, official_forecast=None, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
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
    
    num_cols = min(6, max(3, (len(tracks) + 3) // 5))
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=num_cols, facecolor=BG_COLOR, edgecolor='white', fontsize=9)
    for text in legend.get_texts(): text.set_color('white'); text.set_path_effects(pe_stroke)
        
    ax.set_title(f"{title_suffix} for {storm_name} ({storm_id})", color='white', fontsize=16, fontweight='bold', pad=10, path_effects=pe_stroke)
    if official_forecast: ax.text(0.02, 0.98, f"Initialized at {official_forecast['init'].strftime('%Hz %b %d %Y')}", color='white', transform=ax.transAxes, fontsize=12, va='top', path_effects=pe_stroke)
        
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(output_dir, f'{storm_id}_{title_suffix.lower().replace(" ", "_")}_intensity_{timestamp_str}.png')
    plt.savefig(image_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[DEBUG] {title_suffix} intensity plot successfully generated. Saved to: {image_path}")

async def spaghetti_WPAC(storm_id: str):
    try: os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e: print(f"Error creating output directory: {e}"); return
        
    async with aiohttp.ClientSession() as session:
        try:
            print(f"\n[DEBUG] === STARTING SPAGHETTI PLOT ROUTINE FOR {storm_id} ===")
            storm_id = storm_id.upper()
            if not storm_id.startswith('WP') or len(storm_id) != 8 or not storm_id[-4:].isdigit():
                print("[DEBUG] Invalid storm ID. Please use format like 'WP012026'."); return
                
            obs_lats, obs_lons, lat, lon, vmax, mslp, storm_name, df, official_forecast, radii_34, radii_64, kmz_file = await fetch_jtwc_atcf(storm_id, session)

            motion_dir_deg, storm_speed_kt = 0.0, 0.0
            storm_dir_str, storm_speed_mph = 'N/A', 0.0
            if len(obs_lats) >= 2:
                motion_dir_deg, storm_speed_kt = calculate_bearing_and_speed(obs_lats[-2], obs_lons[-2], obs_lats[-1], obs_lons[-1], 6)
                storm_speed_mph = storm_speed_kt * 1.15078
                dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
                storm_dir_str = dirs[int((motion_dir_deg / 22.5) + 0.5) % 16]

            storm_type = "Typhoon" if (vmax != 'N/A' and float(vmax) >= 64) else "Tropical Storm" if (vmax != 'N/A' and float(vmax) >= 34) else "Tropical Depression"
            
            deterministic_tracks, gefs_tracks, geps_tracks, eps_tracks, intensity_tracks = {}, {}, {}, {}, {}
            
            if not df.empty and 'date' in df.columns:
                print("[DEBUG] Filtering ATCF data for models...")
                df['date'] = pd.to_numeric(df['date'].astype(str).str.strip(), errors='coerce')
                
                ai_models = ['AEMN', 'AIFS', 'GCAI', 'GCFS', 'PANG', 'FOUR']
                deterministic_models = ['GFSO', 'HMON', 'HWRF', 'CTCX', 'AVNO', 'UKX', 'CMC', 'ECMF', 'NVGM', 'JGSM'] + ai_models
                
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
                        
                        intensity_tracks[model] = track 
                        
                        if model in deterministic_models: deterministic_tracks[model] = track
                        elif model in gefs_members: gefs_tracks[model] = track
                        elif model in geps_members: geps_tracks[model] = track
                        elif model in eps_members: eps_tracks[model] = track
                    
                    print(f"[DEBUG] Model Parsing Complete -> Deterministic: {len(deterministic_tracks)}, GEFS: {len(gefs_tracks)}, GEPS: {len(geps_tracks)}, EPS: {len(eps_tracks)}")
                else:
                    print("[DEBUG] Targeted models (GFSO, ECMF, etc.) were not found in the filtered data.")
            else:
                print("[DEBUG] The A-Deck DataFrame was empty or lacked the 'date' column. Cannot extract models.")
                    
            if deterministic_tracks: 
                print("[DEBUG] Dispatching Deterministic Plot creation...")
                await create_plot(deterministic_tracks, f"{storm_type} {storm_name} Model Track Guidance", session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, is_ensemble=False, output_dir=OUTPUT_DIR, kmz_file=kmz_file)
            
            if gefs_tracks: 
                print("[DEBUG] Dispatching GEFS Plot creation...")
                await create_plot(gefs_tracks, f"{storm_type} {storm_name} GEFS Tracks", session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, is_ensemble=True, ensemble_name='GEFS', output_dir=OUTPUT_DIR, kmz_file=kmz_file)
            
            if geps_tracks: 
                print("[DEBUG] Dispatching GEPS Plot creation...")
                await create_plot(geps_tracks, f"{storm_type} {storm_name} GEPS Tracks", session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, is_ensemble=True, ensemble_name='GEPS', output_dir=OUTPUT_DIR, kmz_file=kmz_file)
            
            if eps_tracks: 
                print("[DEBUG] Dispatching EPS Plot creation...")
                await create_plot(eps_tracks, f"{storm_type} {storm_name} EPS Tracks", session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, is_ensemble=True, ensemble_name='EPS', output_dir=OUTPUT_DIR, kmz_file=kmz_file)
            
            if official_forecast or kmz_file: 
                print("[DEBUG] Dispatching Official Cone Plot creation...")
                await create_plot({}, f"JTWC Cone Forecast", session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, official_forecast, radii_34, radii_64, storm_dir_str, storm_speed_mph, show_cone_and_forecast=True, output_dir=OUTPUT_DIR, kmz_file=kmz_file)
            
            if intensity_tracks: 
                print("[DEBUG] Dispatching Intensity Plot creation...")
                await create_intensity_plot(intensity_tracks, f"{storm_type} {storm_name} Model Intensity Guidance", storm_name, storm_id, vmax, mslp, official_forecast=official_forecast, output_dir=OUTPUT_DIR)
            
            if not any([deterministic_tracks, gefs_tracks, geps_tracks, eps_tracks, official_forecast, intensity_tracks, kmz_file]): 
                if len(obs_lats) > 0:
                    print(f"[DEBUG] No models found, but historical track exists. Dispatching Observed Track plot...")
                    await create_plot({}, f"{storm_type} {storm_name} Observed Track", session, storm_name, storm_id, vmax, mslp, obs_lats, obs_lons, lat, lon, None, radii_34, radii_64, storm_dir_str, storm_speed_mph, is_ensemble=False, show_cone_and_forecast=False, output_dir=OUTPUT_DIR)
                else:
                    print(f"[ERROR] No data available to plot for {storm_name} ({storm_id}). The lists were empty after parsing.")
        except Exception as e: 
            print(f"[CRITICAL] An error occurred while processing spaghetti maps for {storm_id}: {e}")

async def marine_key_messages_WPAC(storm_id: Optional[str] = None):
    if storm_id is None: print("Please provide a storm ID."); return
    try: os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e: print(f"Error creating output directory: {e}"); return
        
    storm_id = storm_id.upper()
    if not storm_id.startswith('WP') or len(storm_id) != 8 or not storm_id[-4:].isdigit(): print("Invalid storm ID."); return
        
    async with aiohttp.ClientSession() as session:
        try:
            print(f"\n[DEBUG] === STARTING MARINE ROUTINE FOR {storm_id} ===")
            obs_lats, obs_lons, lat, lon, vmax, mslp, storm_name, df, official_forecast, r34, r64, kmz_file = await fetch_jtwc_atcf(storm_id, session)
            
            if len(obs_lats) == 0:
                print(f"[DEBUG] No active storm found with ID {storm_id}. Exiting marine routine."); return
            
            motion_dir_deg, storm_speed_kt = 0.0, 0.0
            storm_dir_str, storm_speed_mph = 'N/A', 0.0
            if len(obs_lats) >= 2:
                motion_dir_deg, storm_speed_kt = calculate_bearing_and_speed(obs_lats[-2], obs_lons[-2], obs_lats[-1], obs_lons[-1], 6)
                storm_speed_mph = storm_speed_kt * 1.15078
                dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
                storm_dir_str = dirs[int((motion_dir_deg / 22.5) + 0.5) % 16]
            
            if official_forecast is None: 
                print("[DEBUG] No official forecast object found. Applying synthetic forecast defaults.")
                official_forecast = {'lat': np.linspace(lat, lat + 5, 5), 'lon': np.linspace(lon, lon - 2, 5), 'fhr': np.arange(0, 120, 24), 'vmax': np.linspace(50, 80, 5), 'mslp': np.full(5, 990.0)}
            fc_lons, fc_lats, fc_fhrs = np.array(official_forecast['lon']), np.array(official_forecast['lat']), np.array(official_forecast['fhr'])
            
            current_winds_kt = int(vmax) if vmax != 'N/A' else 35
            forecast_24h_winds = int(official_forecast['vmax'][min(1, len(official_forecast['vmax'])-1)]) if official_forecast else 50
                
            now = datetime.now(timezone.utc)
            key_messages = [
                f"Maximum sustained winds are {current_winds_kt} kt.",
                f"{storm_name} is moving {storm_dir_str.lower() if storm_dir_str != 'N/A' else 'generally west'} at {int(storm_speed_kt)} kt.",
                f"Intensity is forecast to reach {forecast_24h_winds} kt by {(now + timedelta(days=1)).strftime('%A')}." if official_forecast else "Monitor JTWC for intensity updates.",
                "See latest Joint Typhoon Warning Center advisory at metoc.navy.mil/jtwc"
            ]
            
            async def get_latest_gfswave_url(session):
                print("[DEBUG] Locating latest GFS wave model run...")
                utcnow = datetime.now(timezone.utc)
                for delta_h in range(0, 72, 6):
                    dt = utcnow - timedelta(hours=delta_h)
                    if dt.hour not in [0, 6, 12, 18]: continue
                    url = f"https://ftpprd.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs.{dt.strftime('%Y%m%d')}/{dt.hour:02d}/gfswave.t{dt.hour:02d}z.atlocn.0p16.grib2"
                    try:
                        async with session.head(url) as resp:
                            if resp.status == 200: 
                                print(f"[DEBUG] Found GFS Wave run: {url}")
                                return url
                    except Exception as e: 
                        print(f"[DEBUG] Failed to access URL {url}: {e}")
                return None
                
            url = await get_latest_gfswave_url(session)
            hsig_wpac = None
            if url:
                try:
                    print(f"[DEBUG] Fetching GFS Wave Data from URL...")
                    async with session.get(url) as resp: content = await resp.read()
                    import cfgrib
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp: tmp.write(content); tmp_path = tmp.name
                    try:
                        print(f"[DEBUG] Decoding GRIB2 file...")
                        ds = xr.open_dataset(tmp_path, engine='cfgrib')
                        hsig_wpac = (ds['htsgw'].sel(step=pd.Timedelta(hours=120)) * 3.28084).squeeze().sel(longitude=slice(100, 180), latitude=slice(0, 60), method='nearest')
                        print(f"[DEBUG] Successfully extracted significant wave height data.")
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                except Exception as e: 
                    print(f"[DEBUG] Failed to process wave data: {e}")
                    hsig_wpac = None
                    
            print(f"[DEBUG] Rendering marine base map...")
            fig = plt.figure(figsize=(16, 10), facecolor=BG_COLOR)
            gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.05)
            ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
            ax_map.set_facecolor(AX_COLOR)
            
            ax_map.set_extent([100, 160, 5, 40], crs=ccrs.PlateCarree())
            ax_map.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1.5); ax_map.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor='black'); ax_map.add_feature(cfeature.OCEAN, facecolor=AX_COLOR); ax_map.add_feature(cfeature.BORDERS, edgecolor='black', linestyle=':')
            
            if len(obs_lats) > 1: ax_map.plot(obs_lons, obs_lats, color='white', linewidth=2.5, path_effects=pe_stroke, transform=ccrs.PlateCarree(), label='Observed Track')
            ax_map.plot(fc_lons, fc_lats, color='red', linestyle='--', linewidth=2.5, path_effects=pe_stroke, transform=ccrs.PlateCarree(), label='Forecast Track')
            ax_map.scatter(lon, lat, marker='o', color='red', s=150, edgecolor='black', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
            ax_map.text(lon + 0.5, lat, f'{storm_name}\n{current_winds_kt} kt', color='white', path_effects=pe_stroke, fontweight='bold', transform=ccrs.PlateCarree(), fontsize=12)
            
            if len(fc_fhrs) > 4:
                idx_120 = np.argmin(np.abs(fc_fhrs - 120))
                ax_map.scatter(fc_lons[idx_120], fc_lats[idx_120], marker='*', color='yellow', s=300, edgecolor='black', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
                ax_map.text(fc_lons[idx_120] + 0.5, fc_lats[idx_120], '120h Forecast', color='white', path_effects=pe_stroke, fontweight='bold', transform=ccrs.PlateCarree(), fontsize=11)
                
            if hsig_wpac is not None:
                cf = ax_map.contourf(hsig_wpac.longitude, hsig_wpac.latitude, hsig_wpac, levels=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36], colors=['#00FFFF', '#00CCFF', '#0099FF', '#0066FF', '#0033FF', '#0000FF', '#9900FF', '#CC00FF', 'red'], extend='max', transform=ccrs.PlateCarree(), alpha=0.7)
                ax_map.text(101, 38, 'Significant Wave Heights (ft)', color='white', path_effects=pe_stroke, transform=ccrs.PlateCarree(), fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor=BG_COLOR, alpha=0.8, edgecolor='white'))
            else: ax_map.text(101, 38, 'Wave data unavailable', color='white', path_effects=pe_stroke, transform=ccrs.PlateCarree(), fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor=BG_COLOR, alpha=0.8, edgecolor='red'))
                
            scale_x, y_scale = [102, 123], 6
            for i, (h, col) in enumerate(zip([4, 8, 12, 16, 20, 24, 28, 32, 36], ['#00FFFF', '#00CCFF', '#0099FF', '#0066FF', '#0033FF', '#0000FF', '#9900FF', '#CC00FF', 'red'])):
                x_start = scale_x[0] + i * ((scale_x[1] - scale_x[0]) / 9)
                ax_map.add_patch(Rectangle((x_start, y_scale), ((scale_x[1] - scale_x[0]) / 9), 0.5, facecolor=col, edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree()))
                ax_map.text(x_start + ((scale_x[1] - scale_x[0]) / 9)/2, y_scale + 0.8, f'{h}', color='white', path_effects=pe_stroke, fontweight='bold', ha='center', transform=ccrs.PlateCarree(), fontsize=10)
            ax_map.text((scale_x[0] + scale_x[1])/2, y_scale - 1.5, 'Wave Heights (ft)', color='white', path_effects=pe_stroke, ha='center', transform=ccrs.PlateCarree(), fontsize=12, fontweight='bold')
            ax_map.set_title('120h Forecast Seas', color='white', fontsize=14, fontweight='bold', pad=10, path_effects=pe_stroke)
            
            print(f"[DEBUG] Rendering marine text block...")
            ax_text = fig.add_subplot(gs[1])
            ax_text.set_facecolor(BG_COLOR); ax_text.axis('off'); ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
            ax_text.text(0.5, 0.95, f'Marine Key Messages\n{now.strftime("%I%M %p UTC %b %d %Y")}', color='white', ha='center', va='top', fontsize=16, fontweight='bold', path_effects=pe_stroke)
            for i, msg in enumerate(key_messages): ax_text.text(0.05, 0.82 - i*0.18, f'• {msg}', color='white', ha='left', va='top', fontsize=13, wrap=True, bbox=dict(facecolor=AX_COLOR, alpha=0.5, edgecolor='none', pad=5))
            ax_text.text(0.5, 0.05, 'For more marine information visit metoc.navy.mil/jtwc\nwww.nhc.noaa.gov', color='gray', ha='center', va='bottom', fontsize=10, style='italic')
            
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            map_image_path = os.path.join(OUTPUT_DIR, f'marine_{storm_id}_{timestamp_str}.png')
            plt.savefig(map_image_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
            plt.close()
            print(f"[DEBUG] Marine key messages plot generated for {storm_name} ({storm_id}). Saved to: {map_image_path}")
        except Exception as e: print(f"[CRITICAL] An error occurred in marine routine: {e}")

if __name__ == "__main__":
    async def main():
        print("Western Pacific Tropical Command-Line Tool")
        import sys
        if len(sys.argv) < 2: print("No command provided."); return
        command, storm_id = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
        if command == "spaghetti_wpac": await spaghetti_WPAC(storm_id)
        elif command == "marine_key_messages_wpac": await marine_key_messages_WPAC(storm_id)
        else: print(f"Unknown command: {command}")
    asyncio.run(main())