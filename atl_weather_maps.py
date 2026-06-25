#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone GFS Map Generator for the Atlantic (Modernized)

Fetches GFS data from the UCAR THREDDS server and generates various
meteorological maps for the CONUS East Coast, Caribbean, and Atlantic.

Upgraded with High-Contrast Dark Theme, Dynamic Colorbars, Native WPC Fronts (Local File),
Tiered Isodrosotherms, Marine RTOFS Currents, and True GOES East Full Disk Water Vapor Overlays.
"""

import logging
import io
import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
import argparse

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import xarray as xr
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import requests

from scipy import ndimage
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import ColdFront, WarmFront, StationaryFront, OccludedFront, StationPlot
from metpy.plots.wx_symbols import sky_cover
from siphon.catalog import TDSCatalog

# ====================================================================
# 1. LOGGING & CONSTANTS
# ====================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

EARTH_RADIUS = 6371000  # meters
OMEGA = 7.292e-5  # Earth's angular velocity (rad/s)
REGION = "Atlantic"
EXTENT = [-100, -30, 10, 50]

# Standardized Layout Colors and Styles (FrostByte Weather Aesthetic)
FIG_BG_COLOR = '#333333'
OCEAN_COLOR = '#1A2A3A'
LAND_COLOR = '#2B2B2B'
GRIDLINE_COLOR = '#3932A0'
TEXT_OUTLINE = [pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()]
WPC_FRONTS_SHP = './shapefiles/wpc_fronts.shp'

LABEL_STYLE = {
    'size': 11, 
    'fontweight': 'bold', 
    'color': 'white', 
    'path_effects': TEXT_OUTLINE
}

cities = [
    {'name': 'Miami', 'lat': 25.7617, 'lon': -80.1918},
    {'name': 'San Juan', 'lat': 18.4663, 'lon': -66.1057},
    {'name': 'Havana', 'lat': 23.1136, 'lon': -82.3666},
    {'name': 'Nassau', 'lat': 25.0582, 'lon': -77.3431},
    {'name': 'Kingston', 'lat': 17.9970, 'lon': -76.7936},
    {'name': 'Port-au-Prince', 'lat': 18.5944, 'lon': -72.3074},
    {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
    {'name': 'Boston', 'lat': 42.3601, 'lon': -71.0589},
    {'name': 'Washington', 'lat': 38.9072, 'lon': -77.0369},
    {'name': 'Charleston', 'lat': 32.7765, 'lon': -79.9311},
    {'name': 'Jacksonville', 'lat': 30.3322, 'lon': -81.6557},
    {'name': 'New Orleans', 'lat': 29.9511, 'lon': -90.0715},
    {'name': 'Bermuda', 'lat': 32.3078, 'lon': -64.7505},
]

def get_ga_temp_cmap():
    color_stops_data = [
        (-100, "#F0F0F0"), (-80, "#000058"), (-60, "#0202BB"), (-40, "#15589B"), 
        (-20, "#4180BB"), (-10, "#AD006B"), (-5, "#B6009E"), (0, "#FF01FF"), 
        (5, "#7F1BB1"), (10, "#BA55D3"), (15, "#DA70D6"), (20, "#E0B0FF"), 
        (25, "#AFEEEE"), (30, "#00FFFF"), (35, "#00FF00"), (40, "#32CD32"), 
        (45, "#7CFC00"), (50, "#ADFF2F"), (55, "#FFFF00"), (60, "#FFD700"), 
        (65, "#FFA500"), (70, "#FF8C00"), (75, "#FF4500"), (80, "#FF0000"), 
        (85, "#DC143C"), (90, "#B22222"), (95, "#8B0000"), (100, "#A52A2A"), 
        (105, "#8B4513"), (110, "#A0522D"), (115, "#CD853F"), (120, "#D2B48C"), 
        (125, "#FFDDC1"), (130, "#FFFFFF")
    ]
    min_val, max_val = -100, 130
    normalized_stops = [((v - min_val) / (max_val - min_val), c) for v, c in color_stops_data]
    return mcolors.LinearSegmentedColormap.from_list('ga_temp_cmap', normalized_stops)

# ====================================================================
# 2. METEOROLOGICAL PHYSICS MATH
# ====================================================================

def compute_wind_speed(u, v):
    return np.sqrt(u**2 + v**2) * 1.94384

def compute_vorticity(u, v, lat, lon):
    lat_rad = np.deg2rad(lat)
    dlon, dlat = lon[1] - lon[0], lat[1] - lat[0]
    dlon_rad, dlat_rad = np.deg2rad(dlon), np.deg2rad(dlat)
    dx = EARTH_RADIUS * np.cos(lat_rad) * dlon_rad
    dy = EARTH_RADIUS * np.abs(dlat_rad)
    v_x = np.full_like(v, np.nan)
    v_x[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx[:, None])
    u_y = np.full_like(u, np.nan)
    u_y[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * dy)
    zeta_r = v_x - u_y
    return zeta_r * 1e5

def compute_advection(phi, u, v, lat, lon):
    lat_rad = np.deg2rad(lat)
    dlon, dlat = lon[1] - lon[0], lat[1] - lat[0]
    dlon_rad, dlat_rad = np.deg2rad(dlon), np.deg2rad(dlat)
    dx = EARTH_RADIUS * np.cos(lat_rad) * dlon_rad
    dy = EARTH_RADIUS * np.abs(dlat_rad)
    phi_x = np.full_like(phi, np.nan)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx[:, None])
    phi_y = np.full_like(phi, np.nan)
    phi_y[1:-1, :] = (phi[:-2, :] - phi[2:, :]) / (2 * dy)
    return u * phi_x + v * phi_y

def compute_dewpoint(T, rh):
    T_C = T - 273.15
    rh = np.clip(rh, 1e-10, 100)
    ln_rh = np.log(rh / 100.0)
    a, b = 17.67, 243.5
    gamma = (a * T_C) / (b + T_C)
    return (b * (ln_rh + gamma)) / (a - ln_rh - gamma)

def compute_divergence(u, v, lat, lon):
    lat_rad = np.deg2rad(lat)
    dlon, dlat = lon[1] - lon[0], lat[1] - lat[0]
    dlon_rad, dlat_rad = np.deg2rad(dlon), np.deg2rad(dlat)
    dx = EARTH_RADIUS * np.cos(lat_rad)[:, None] * dlon_rad
    dy = EARTH_RADIUS * dlat_rad
    u_x = np.full_like(u, np.nan)
    v_y = np.full_like(v, np.nan)
    u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    v_y[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
    return (u_x + v_y) * 1e5

def frontogenesis_700hPa(T, u, v, lat, lon):
    try:
        lat_rad = np.deg2rad(lat)
        dlon, dlat = lon[1] - lon[0], lat[1] - lat[0]
        dlon_rad, dlat_rad = np.deg2rad(dlon), np.deg2rad(dlat)
        dx = EARTH_RADIUS * np.cos(lat_rad)[:, None] * dlon_rad
        dy = EARTH_RADIUS * np.abs(dlat_rad)
        dT_dx = np.full_like(T, np.nan); dT_dy = np.full_like(T, np.nan)
        dT_dx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * dx)
        dT_dy[1:-1, :] = (T[:-2, :] - T[2:, :]) / (2 * dy)
        du_dx = np.full_like(u, np.nan); du_dy = np.full_like(u, np.nan)
        dv_dx = np.full_like(v, np.nan); dv_dy = np.full_like(v, np.nan)
        du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        du_dy[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * dy)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
        dv_dy[1:-1, :] = (v[:-2, :] - v[2:, :]) / (2 * dy)
        mag_grad_T = np.sqrt(dT_dx**2 + dT_dy**2)
        mag_grad_T_safe = np.where(mag_grad_T < 1e-10, 1e-10, mag_grad_T)
        F = (1 / mag_grad_T_safe) * (- (dT_dx**2) * du_dx - dT_dx * dT_dy * (dv_dx + du_dy) - (dT_dy**2) * dv_dy)
        return F * 1e5 * 10800
    except Exception as e:
        logging.error(f"Error computing frontogenesis: {e}")
        return None

# ====================================================================
# 3. DATA FETCHING (GFS, GOES, & RTOFS)
# ====================================================================

def fetch_ocean_currents(extent):
    """Fetches RTOFS marine layer data (ocean currents & SST). Includes local caching."""
    print("\n--- MARINE DATA FETCH (RTOFS Direct Download with Cache) ---")
    logging.info("Attempting to fetch RTOFS data via direct HTTPS...")

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
        today = datetime.now(timezone.utc)
        dates_to_try = [today, today - timedelta(days=1), today - timedelta(days=2)]
        
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 BoxWeather/1.0'})

        files_to_try = [
            "rtofs_glo_2ds_f000_prog.nc", 
            "rtofs_glo_2ds_f024_prog.nc", 
            "rtofs_glo_2ds.f000.nc",       
            "rtofs_glo_2ds.f024.nc"        
        ]

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
                        print("Failed.")
                        continue
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
            mask = (lats_raw >= lat_min) & (lats_raw <= lat_max) & \
                   (lons_raw >= lon_min) & (lons_raw <= lon_max)
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

        sst_candidates = ['sst', 'SST', 'temperature', 'temp', 'sea_surface_temperature', 'Temperature']
        u_candidates   = ['u_velocity', 'u', 'water_u', 'U', 'u_comp', 'velocity_u', 'surf_u', 'eastward_vel']
        v_candidates   = ['v_velocity', 'v', 'water_v', 'V', 'v_comp', 'velocity_v', 'surf_v', 'northward_vel']
        sal_candidates = ['sss', 'SSS', 'salinity', 'salt', 'Salinity', 'sea_surface_salinity']

        sst_var = next((v for v in sst_candidates if v in ds.variables), None)
        u_var   = next((v for v in u_candidates   if v in ds.variables), None)
        v_var   = next((v for v in v_candidates   if v in ds.variables), None)
        sal_var = next((v for v in sal_candidates if v in ds.variables), None)

        def safe_slice(vname):
            if not vname: return None
            var_obj = ds.variables[vname]
            ndim = len(var_obj.shape)
            if ndim == 2: return var_obj[y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
            if ndim == 3: return var_obj[0, y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
            if ndim == 4: return var_obj[0, 0, y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
            return None

        w_temp = safe_slice(sst_var)
        if w_temp is None: w_temp = np.full(lats.shape, np.nan, dtype=np.float32)
        else: w_temp = np.asarray(w_temp, dtype=np.float32)

        w_u = safe_slice(u_var)
        if w_u is None: w_u = np.full_like(w_temp, np.nan)
        else: w_u = np.asarray(w_u, dtype=np.float32)

        w_v = safe_slice(v_var)
        if w_v is None: w_v = np.full_like(w_temp, np.nan)
        else: w_v = np.asarray(w_v, dtype=np.float32)

        salinity = safe_slice(sal_var)
        if salinity is None: salinity = np.full_like(w_temp, np.nan)
        else: salinity = np.asarray(salinity, dtype=np.float32)

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

def fetch_goes_ch9():
    """Fetches real GOES East Ch 9 Water Vapor for the background overlay."""
    logging.info("Fetching real GOES East Ch 9 Water Vapor (Full Disk) for background overlay...")
    try:
        import metpy
        cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/satellite/goes/east/products/CloudAndMoistureImagery/FullDisk/Channel09/current/catalog.xml')
        datasets = sorted(list(cat.datasets.values()), key=lambda d: d.name)
        latest_url = datasets[-1].access_urls['OPeNDAP']
        
        ds = xr.open_dataset(latest_url)
        ds = ds.metpy.parse_cf()
        var = ds['Sectorized_CMI'] if 'Sectorized_CMI' in ds else ds['CMI']
        logging.info("✅ GOES East Full Disk Data Fetched Successfully.")
        return ds, var
    except Exception as e:
        logging.error(f"Failed to fetch GOES data: {e}")
        return None, None

def get_latest_run_date():
    now = datetime.now(timezone.utc)
    run_hours = [0, 6, 12, 18]
    check_time = now - timedelta(hours=6)
    for run_hour in sorted(run_hours, reverse=True):
        if check_time.hour >= run_hour:
            return check_time.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    return (check_time - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)

def get_time_dimension(ds, run_date):
    try:
        run_hour = run_date.hour
        time_dim_map = {
            0: ['reftime', 'time', 'validtime'],
            6: ['reftime1', 'validtime1', 'reftime', 'time'],
            12: ['reftime2', 'validtime2', 'reftime', 'time'],
            18: ['reftime3', 'validtime3', 'reftime', 'time']
        }
        possible_dims = time_dim_map.get(run_hour, ['reftime', 'time', 'validtime'])
        selected_dims = {}
        for dim in possible_dims:
            if dim in ds.dims:
                selected_dims[dim] = 0
                break
        if 'time' in ds.dims and 'time' not in selected_dims:
            selected_dims['time'] = 0
        return selected_dims
    except Exception as e:
        logging.error(f"Error in get_time_dimension: {e}")
        raise

def get_gfs_data_for_level(level, forecast_hour=0):
    run_date = get_latest_run_date()
    valid_time = run_date + timedelta(hours=forecast_hour)
    logging.info(f"Fetching GFS data for {level} Pa at +{forecast_hour}h")

    catalog_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml'
    cat = TDSCatalog(catalog_url)
    latest_dataset = list(cat.datasets.values())[0]
    ncss = latest_dataset.subset()

    query = ncss.query()
    query.accept('netcdf4')
    query.time(valid_time)
    query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric',
                    'v-component_of_wind_isobaric', 'Relative_humidity_isobaric', 'Temperature_isobaric')
    query.vertical_level([level])
    query.lonlat_box(north=EXTENT[3], south=EXTENT[2], east=EXTENT[1], west=EXTENT[0])

    try:
        data = ncss.get_data(query)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data)).metpy.parse_cf()
        return ds, run_date, valid_time
    except Exception as e:
        logging.error(f"Error fetching data for level {level}: {e}")
        return None, None, None

def get_gfs_surface_data(forecast_hour=0):
    run_date = get_latest_run_date()
    valid_time = run_date + timedelta(hours=forecast_hour)
    logging.info(f"Fetching GFS surface data at +{forecast_hour}h")

    catalog_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml'
    try:
        cat = TDSCatalog(catalog_url)
        latest_dataset = list(cat.datasets.values())[0]
        ncss = latest_dataset.subset()

        query1 = ncss.query()
        query1.accept('netcdf4')
        query1.time(valid_time)
        query1.variables('Temperature_surface', 'Pressure_surface', 'Geopotential_height_surface',
                         'Dewpoint_temperature_height_above_ground')
        query1.lonlat_box(north=EXTENT[3], south=EXTENT[2], east=EXTENT[1], west=EXTENT[0])
        
        data1 = ncss.get_data(query1)
        ds1 = xr.open_dataset(xr.backends.NetCDF4DataStore(data1)).metpy.parse_cf()

        query2 = ncss.query()
        query2.accept('netcdf4')
        query2.time(valid_time)
        query2.variables('u-component_of_wind_height_above_ground', 'v-component_of_wind_height_above_ground')
        query2.lonlat_box(north=EXTENT[3], south=EXTENT[2], east=EXTENT[1], west=EXTENT[0])
        query2.vertical_level(10)
        
        data2 = ncss.get_data(query2)
        ds2 = xr.open_dataset(xr.backends.NetCDF4DataStore(data2)).metpy.parse_cf()

        def drop_height_dims(dataset):
            h_dims = [d for d in dataset.dims if 'height' in d.lower()]
            if h_dims:
                return dataset.squeeze(dim=h_dims, drop=True)
            return dataset

        ds = xr.merge([drop_height_dims(ds1), drop_height_dims(ds2)], compat='override')
        
        rename_dict = {
            'Temperature_surface': 't2m', 
            'Pressure_surface': 'sp',
            'Geopotential_height_surface': 'orog', 
            'u-component_of_wind_height_above_ground': 'u10',
            'v-component_of_wind_height_above_ground': 'v10',
            'Dewpoint_temperature_height_above_ground': 'dpt2m'
        }
        rename_safe = {k: v for k, v in rename_dict.items() if k in ds.variables}
        ds = ds.rename(rename_safe)
        
        return ds, run_date, valid_time
    except Exception as e:
        logging.error(f"Error in get_gfs_surface_data: {e}")
        return None, None, None

# ====================================================================
# 4. PLOTTING UTILITIES
# ====================================================================

def plot_background(ax):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1.4, zorder=3)
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linestyle=':', linewidth=1.4, zorder=3)
    ax.add_feature(cfeature.STATES, edgecolor='black', linestyle=':', linewidth=0.9, zorder=3)
    
    gl = ax.gridlines(draw_labels=True, linewidth=1.0, color=GRIDLINE_COLOR, alpha=0.4, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = LABEL_STYLE; gl.ylabel_style = LABEL_STYLE

def add_cities(ax):
    for city in cities:
        ax.plot(city['lon'], city['lat'], marker='o', color='white', markeredgecolor='black', markersize=5, transform=ccrs.PlateCarree(), zorder=5)
        ax.text(city['lon'] + 0.5, city['lat'] + 0.5, city['name'], color='white', fontsize=11, fontweight='bold', path_effects=TEXT_OUTLINE, transform=ccrs.PlateCarree(), zorder=6)

def add_clean_colorbar(fig, ax, cf, label):
    if cf is not None:
        pos = ax.get_position()
        cb_x0 = pos.x0 + 0.004
        cb_width = pos.width - 0.008
        
        cax = fig.add_axes([cb_x0, 0.095, cb_width, 0.025])
        cb = fig.colorbar(cf, cax=cax, orientation='horizontal', extend='both', extendfrac=0.04)
        
        cb.set_label(label, fontsize=13, fontweight='bold', color='white', labelpad=10)
        cb.ax.xaxis.label.set_path_effects(TEXT_OUTLINE)
        
        if 'Temperature' in label or 'Dewpoint' in label: ticks = np.arange(-40, 131, 10)
        elif 'Relative Humidity' in label: ticks = np.arange(10, 101, 10)
        elif 'Wind Speed' in label: ticks = np.arange(20, 201, 20)
        elif 'Advection' in label or 'Vorticity' in label or 'Divergence' in label: ticks = np.arange(-40, 41, 10)
        else: ticks = cb.get_ticks()
        
        vmin, vmax = cf.get_clim()
        valid_ticks = [t for t in ticks if vmin <= t <= vmax]
        
        cb.set_ticks(valid_ticks)
        cb.ax.tick_params(axis='x', length=0, colors='white')
        
        cb.ax.set_xticklabels([str(int(t)) for t in valid_ticks], fontsize=11, fontweight='bold', color='white')
        for tick_label in cb.ax.get_xticklabels():
            tick_label.set_path_effects(TEXT_OUTLINE)

def draw_isodrosotherms(ax, grid_x, grid_y, dp_f, interval=5):
    smoothed_dp = gaussian_filter(dp_f, sigma=2.0)
    
    levels_blue = np.arange(-40, 65, interval)
    levels_green = np.arange(65, 75, interval)
    levels_orange = np.arange(75, 91, interval)
    
    def plot_tier(levels, color):
        if len(levels) == 0: return
        cs = ax.contour(grid_x, grid_y, smoothed_dp, levels=levels, 
                        colors=color, linestyles='--', linewidths=1.5, 
                        zorder=2.3, transform=ccrs.PlateCarree())
        cs.set_path_effects([pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
        
        clabels = ax.clabel(cs, fmt='%d°F', inline=True, inline_spacing=8, fontsize=10, colors=color)
        for label in clabels:
            label.set_path_effects([pe.withStroke(linewidth=3, foreground='black'), pe.Normal()])
            label.set_fontweight('bold')

    plot_tier(levels_blue, '#87CEFA')   
    plot_tier(levels_green, '#32CD32')  
    plot_tier(levels_orange, '#FFA500') 

def plot_wpc_fronts(ax, shapefile_path=WPC_FRONTS_SHP):
    """Plots Native MetPy fronts from a local WPC shapefile."""
    if not os.path.exists(shapefile_path):
        logging.info(f"WPC Front shapefile not found at '{shapefile_path}'. Skipping fronts.")
        return
        
    logging.info(f"Applying WPC Fronts from {shapefile_path}...")
    try:
        reader = shpreader.Reader(shapefile_path)
        for record in reader.records():
            geom = record.geometry
            fnt_type = str(record.attributes.get('type', record.attributes.get('TYPE', record.attributes.get('Feature', '')))).lower()
            
            def draw_front_line(x, y, f_type):
                if 'cold' in f_type:
                    ColdFront(x, y, ax=ax, transform=ccrs.PlateCarree(), zorder=5)
                elif 'warm' in f_type:
                    WarmFront(x, y, ax=ax, transform=ccrs.PlateCarree(), zorder=5)
                elif 'stat' in f_type:
                    StationaryFront(x, y, ax=ax, transform=ccrs.PlateCarree(), zorder=5)
                elif 'occl' in f_type:
                    OccludedFront(x, y, ax=ax, transform=ccrs.PlateCarree(), zorder=5)
                elif 'trof' in f_type or 'trough' in f_type:
                    ax.plot(x, y, color='orange', linestyle='--', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
            
            if geom.geom_type == 'LineString':
                x, y = geom.xy
                draw_front_line(x, y, fnt_type)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    x, y = line.xy
                    draw_front_line(x, y, fnt_type)
    except Exception as e:
        logging.error(f"Failed to plot WPC fronts: {e}")

# ====================================================================
# 5. GENERATORS
# ====================================================================

def generate_map(ds, init_time, valid_time, level, variable, cmap, title, cb_label, levels=None, forecast_hour=0, wv_background=False, plot_vort_max=False, sat_ds=None, sat_var=None):
    try:
        time_dims = get_time_dimension(ds, init_time)
        ds = ds.isel(**time_dims)
        
        lon, lat = ds.longitude.values, ds.latitude.values
        lon = np.where(lon > 180, lon - 360, lon)
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        
        heights = ds['Geopotential_height_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
        heights_smooth = ndimage.gaussian_filter(heights / 10.0, sigma=3, order=0) # DAM

        u_wind = ds['u-component_of_wind_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
        v_wind = ds['v-component_of_wind_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()

        if variable == 'wind_speed': data = compute_wind_speed(u_wind, v_wind)
        elif variable == 'vorticity': data = compute_vorticity(u_wind, v_wind, lat, lon)
        elif variable == 'relative_humidity': data = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
        elif variable == 'temp_advection':
            temp = ds['Temperature_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            data = compute_advection(temp, u_wind, v_wind, lat, lon) * 3600
        elif variable == 'moisture_advection':
            rh = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            data = compute_advection(rh, u_wind, v_wind, lat, lon) * 1e4
        elif variable == 'dewpoint':
            temp = ds['Temperature_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            rh = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            data = compute_dewpoint(temp, rh)
        elif variable == 'divergence': data = compute_divergence(u_wind, v_wind, lat, lon)
        elif variable == 'frontogenesis':
            temp = ds['Temperature_isobaric'].sel(isobaric=level, method='nearest').squeeze().metpy.convert_units('degC').metpy.dequantify().values.copy()
            data = frontogenesis_700hPa(temp, u_wind, v_wind, lat, lon)
            
        fig = plt.figure(figsize=(16, 10), facecolor=FIG_BG_COLOR)
        ax = fig.add_axes([0.06, 0.16, 0.88, 0.75], projection=ccrs.PlateCarree())
        
        plot_background(ax)
        add_cities(ax)

        main_cf = None
        if wv_background and sat_ds is not None and sat_var is not None:
            sat_proj = sat_var.metpy.cartopy_crs
            x = sat_ds['x'].values
            y = sat_ds['y'].values
            
            grid_mapping = sat_var.attrs.get('grid_mapping')
            if grid_mapping and grid_mapping in sat_ds:
                proj_var = sat_ds[grid_mapping]
                if getattr(proj_var, 'grid_mapping_name', '') == 'geostationary' and sat_ds['x'].attrs.get('units') == 'rad':
                    height = getattr(proj_var, 'perspective_point_height', 35786023.0)
                    x = x * height
                    y = y * height

            sat_data = sat_var.metpy.magnitude
            if getattr(sat_var.metpy, 'units', None) == 'K' or np.nanmax(sat_data) > 150:
                sat_data = sat_data - 273.15

            wv_levels = np.array([0, -15.5, -30, -47, -75, -100, -109])
            C_RED, C_YELLOW, C_BLUE = (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0)
            C_WHITE, C_GREEN, C_TEAL, C_LB = (1.0, 1.0, 1.0), (0.0, 0.5, 0.0), (0.0, 1.0, 1.0), (0.5, 0.5, 1.0)

            span = wv_levels.max() - wv_levels.min()
            nodes = np.flip((wv_levels - wv_levels.min()) / span)
            node_colors = [C_RED, C_YELLOW, C_BLUE, C_WHITE, C_GREEN, C_TEAL, C_LB][::-1]
            nodes[0] = 0.0; nodes[-1] = 1.0
            wv_cmap = LinearSegmentedColormap.from_list('custom_wv_cmap', list(zip(nodes, node_colors)))
            norm = mcolors.Normalize(vmin=-109, vmax=0)
            
            main_cf = ax.pcolormesh(x, y, sat_data, cmap=wv_cmap, norm=norm, transform=sat_proj, shading='auto', zorder=1.5, alpha=0.90)

            # Match dynamic width of standard colorbar
            pos = ax.get_position()
            cb_x0 = pos.x0 + 0.004
            cb_width = pos.width - 0.008
            cax = fig.add_axes([cb_x0, 0.095, cb_width, 0.025])
            
            cbar = fig.colorbar(main_cf, cax=cax, orientation='horizontal', extend='both')
            cbar.set_label('Temp (°C)', fontsize=13, fontweight='bold', color='white', labelpad=10)
            cbar.ax.xaxis.label.set_path_effects(TEXT_OUTLINE)
            cb_ticks = np.sort(wv_levels)
            cbar.set_ticks(cb_ticks)
            cbar.ax.set_xticklabels([f"{t:g}" for t in cb_ticks], fontsize=11, fontweight='bold', color='white')
            for tick in cbar.ax.get_xticklabels(): tick.set_path_effects(TEXT_OUTLINE)
            
        elif wv_background:
            logging.warning("Satellite overlay skipped. Falling back to RH simulation...")
            try:
                rh = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
                rh_points = [0, 15, 30, 50, 70, 85, 100]
                temp_points = [0, -15.5, -30, -47, -75, -100, -109]
                sim_wv_temp = np.interp(np.clip(rh, 0, 100), rh_points, temp_points)

                wv_levels = np.array([0, -15.5, -30, -47, -75, -100, -109])
                C_RED, C_YELLOW, C_BLUE = (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0)
                C_WHITE, C_GREEN, C_TEAL, C_LB = (1.0, 1.0, 1.0), (0.0, 0.5, 0.0), (0.0, 1.0, 1.0), (0.5, 0.5, 1.0)
                span = wv_levels.max() - wv_levels.min()
                nodes = np.flip((wv_levels - wv_levels.min()) / span)
                node_colors = [C_RED, C_YELLOW, C_BLUE, C_WHITE, C_GREEN, C_TEAL, C_LB][::-1]
                nodes[0] = 0.0; nodes[-1] = 1.0
                wv_cmap = LinearSegmentedColormap.from_list('custom_wv_cmap', list(zip(nodes, node_colors)))
                norm = mcolors.Normalize(vmin=-109, vmax=0)
                
                main_cf = ax.pcolormesh(lon_2d, lat_2d, sim_wv_temp, cmap=wv_cmap, norm=norm, alpha=0.85, transform=ccrs.PlateCarree(), shading='auto', zorder=1.5)

                # Match dynamic width of standard colorbar
                pos = ax.get_position()
                cb_x0 = pos.x0 + 0.004
                cb_width = pos.width - 0.008
                cax = fig.add_axes([cb_x0, 0.095, cb_width, 0.025])
                
                cbar = fig.colorbar(main_cf, cax=cax, orientation='horizontal', extend='both')
                cbar.set_label('Simulated WV Temp (°C)', fontsize=13, fontweight='bold', color='white', labelpad=10)
                cbar.ax.xaxis.label.set_path_effects(TEXT_OUTLINE)
                cb_ticks = np.sort(wv_levels)
                cbar.set_ticks(cb_ticks)
                cbar.ax.set_xticklabels([f"{t:g}" for t in cb_ticks], fontsize=11, fontweight='bold', color='white')
                for tick in cbar.ax.get_xticklabels(): tick.set_path_effects(TEXT_OUTLINE)
            except Exception as e:
                logging.error(f"Fallback RH sim failed: {e}")
        else:
            if levels is None:
                cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=ccrs.PlateCarree(), extend='both', alpha=0.85)
            else:
                cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=ccrs.PlateCarree(), levels=levels, extend='both', alpha=0.85)
            add_clean_colorbar(fig, ax, cf, cb_label)

        h_levels = np.arange(480, 620, 6) if level == 50000 else np.arange(90, 1100, 4)
        c = ax.contour(lon_2d, lat_2d, heights_smooth, levels=h_levels, colors='white', linewidths=2, transform=ccrs.PlateCarree())
        c.set_path_effects([pe.Stroke(linewidth=3.5, foreground='black'), pe.Normal()])
        clabels = ax.clabel(c, fmt='%i dam', inline=True, inline_spacing=12, fontsize=10, colors='white')
        for label in clabels:
            label.set_path_effects([pe.withStroke(linewidth=3, foreground='black'), pe.Normal()])
            label.set_fontweight('bold')

        if plot_vort_max and variable == 'vorticity':
            from scipy.ndimage import maximum_filter
            vort_max = maximum_filter(data, size=15)
            local_max = (data == vort_max) & (data > 10)
            y_idx, x_idx = np.where(local_max)
            
            for i in range(len(y_idx)):
                lon_val = lon_2d[y_idx[i], x_idx[i]]
                lat_val = lat_2d[y_idx[i], x_idx[i]]
                
                if (EXTENT[0] <= lon_val <= EXTENT[1]) and (EXTENT[2] <= lat_val <= EXTENT[3]):
                    ax.plot(lon_val, lat_val, 'X',
                            color='yellow', markersize=14, markeredgecolor='black', 
                            markeredgewidth=2, zorder=6, transform=ccrs.PlateCarree(), clip_on=True)
                    ax.text(lon_val + 0.4, lat_val + 0.4,
                            f"{data[y_idx[i], x_idx[i]]:.0f}", color='yellow', fontsize=10,
                            fontweight='bold', path_effects=TEXT_OUTLINE, zorder=7, 
                            transform=ccrs.PlateCarree(), clip_on=True)

        u_wind_knots, v_wind_knots = u_wind * 1.94384, v_wind * 1.94384
        skip = 16 if plot_vort_max else 5
        ax.barbs(lon_2d[::skip, ::skip], lat_2d[::skip, ::skip], u_wind_knots[::skip, ::skip], v_wind_knots[::skip, ::skip], transform=ccrs.PlateCarree(), length=6, color='black', zorder=4)

        ax.text(0.0, 1.02, title, fontsize=20, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, transform=ax.transAxes, ha='left', va='bottom')
        ax.text(1.0, 1.02, valid_time.strftime('%d %B %Y %H:%MZ') + f" (+{forecast_hour}h)", fontsize=16, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, transform=ax.transAxes, ha='right', va='bottom')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        logging.error(f"Error in generate_map: {e}", exc_info=True)
        return None

def generate_mslp_temp_map(forecast_hour=0):
    try:
        ds, init_time, valid_time = get_gfs_surface_data(forecast_hour)
        if ds is None: raise ValueError("Failed to retrieve surface data.")
        
        ncom_lat, ncom_lon, ncom_temp_k, ncom_u, ncom_v, ncom_sal = fetch_ocean_currents(EXTENT)

        time_dims = get_time_dimension(ds, init_time)
        ds = ds.isel(**time_dims)

        lon, lat = ds.longitude.values, ds.latitude.values
        lon = np.where(lon > 180, lon - 360, lon)
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        temp_surface_c = ds["t2m"].squeeze().metpy.convert_units("degC").metpy.dequantify()
        temp_surface_f = (temp_surface_c * 9/5) + 32
        
        dpt_f = None
        if "dpt2m" in ds.variables:
            dpt_k = ds["dpt2m"].squeeze().metpy.dequantify()
            dpt_f = ((dpt_k - 273.15) * 9/5) + 32

        surface_pressure = ds["sp"].squeeze().metpy.convert_units("Pa").metpy.dequantify()
        elevation = ds["orog"].squeeze().metpy.dequantify()
        u_wind = ds["u10"].squeeze().metpy.dequantify()
        v_wind = ds["v10"].squeeze().metpy.dequantify()

        temp_kelvin = temp_surface_c + 273.15
        mslp = (surface_pressure * np.exp((9.80665 * elevation) / (287.05 * temp_kelvin))) / 100
        mslp = np.where((mslp >= 850) & (mslp <= 1100), mslp, np.nan)
        mslp_smooth = ndimage.gaussian_filter(mslp, sigma=3, order=0)

        fig = plt.figure(figsize=(16, 10), facecolor=FIG_BG_COLOR)
        ax = fig.add_axes([0.06, 0.16, 0.88, 0.75], projection=ccrs.PlateCarree())
        
        plot_background(ax)

        temp_cmap = get_ga_temp_cmap()
        temp_levels = np.arange(-40, 121, 2.0)
        cf = ax.contourf(lon_2d, lat_2d, temp_surface_f, levels=temp_levels, cmap=temp_cmap, transform=ccrs.PlateCarree(), extend='both', alpha=0.85, zorder=1.5)

        # Plot Marine Layer (RTOFS) mapped securely onto the GFS 2D Grid
        if (ncom_lat is not None and ncom_u is not None and np.any(np.isfinite(ncom_u))):
            print("• Plotting Marine/Ocean Layer...", end=" ", flush=True)
            try:
                if ncom_lat.ndim == 1:
                    nc_lon_2d, nc_lat_2d = np.meshgrid(ncom_lon, ncom_lat)
                else:
                    nc_lon_2d, nc_lat_2d = ncom_lon, ncom_lat
                    
                valid_mask = np.isfinite(ncom_u) & np.isfinite(ncom_v) & np.isfinite(ncom_temp_k)
                rtofs_lon = nc_lon_2d[valid_mask]
                rtofs_lat = nc_lat_2d[valid_mask]
                rtofs_u = ncom_u[valid_mask]
                rtofs_v = ncom_v[valid_mask]
                rtofs_t = (ncom_temp_k - 273.15)[valid_mask]
                
                pts = np.column_stack((rtofs_lon, rtofs_lat))
                
                u_interp = griddata(pts, rtofs_u, (lon_2d, lat_2d), method='linear')
                v_interp = griddata(pts, rtofs_v, (lon_2d, lat_2d), method='linear')
                t_interp = griddata(pts, rtofs_t, (lon_2d, lat_2d), method='linear')
                
                strm = ax.streamplot(lon, lat, 
                                     u_interp, v_interp, 
                                     color=t_interp, cmap='jet',
                                     density=1.2, linewidth=1.5, arrowsize=1.5,
                                     transform=ccrs.PlateCarree(), zorder=2.0)
                strm.lines.set_path_effects([pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
                
                cax_marine = fig.add_axes([0.02, 0.164, 0.008, 0.742]) 
                cbar_m = fig.colorbar(strm.lines, cax=cax_marine, extend='both')
                cbar_m.ax.yaxis.set_label_position('left')
                cbar_m.ax.yaxis.set_ticks_position('left')
                
                cbar_m.set_label('Ocean Temp (°C)', fontsize=9, fontweight='bold', color='white', labelpad=2)
                cbar_m.ax.yaxis.label.set_path_effects(TEXT_OUTLINE)
                
                cbar_m.ax.tick_params(axis='y', colors='white', pad=2, labelsize=8) 
                for tick_label in cbar_m.ax.get_yticklabels():
                    tick_label.set_path_effects(TEXT_OUTLINE)
                    tick_label.set_fontweight('bold')
                print("Done. ✅")
            except Exception as e:
                logging.error(f"Marine Plotting Error: {e}")

        isobar_levels = np.arange(950, 1050, 2)
        c = ax.contour(lon_2d, lat_2d, mslp_smooth, levels=isobar_levels, colors='white', linewidths=2, transform=ccrs.PlateCarree(), zorder=2.0)
        c.set_path_effects([pe.Stroke(linewidth=3.5, foreground='black'), pe.Normal()])
        clabels = ax.clabel(c, fmt='%d mb', inline=True, inline_spacing=12, fontsize=10, colors='white')
        for label in clabels:
            label.set_path_effects([pe.withStroke(linewidth=3, foreground='black'), pe.Normal()])
            label.set_fontweight('bold')

        if dpt_f is not None:
            draw_isodrosotherms(ax, lon_2d, lat_2d, dpt_f, interval=5)

        plot_wpc_fronts(ax)

        u_wind_knots, v_wind_knots = u_wind * 1.94384, v_wind * 1.94384
        ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind_knots[::5, ::5], v_wind_knots[::5, ::5], transform=ccrs.PlateCarree(), length=6, color='black', zorder=6.0)

        add_cities(ax)

        # --- CUSTOM LEGEND ---
        isobar_line = mlines.Line2D([], [], color='white', linewidth=1.8, label='MSLP (mb)', 
                                    path_effects=[pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
        dp_blue = mlines.Line2D([], [], color='#87CEFA', linestyle='--', linewidth=1.5, label='Td < 65°F',
                                path_effects=[pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
        dp_green = mlines.Line2D([], [], color='#32CD32', linestyle='--', linewidth=1.5, label='Td 65-74°F',
                                 path_effects=[pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
        dp_orange = mlines.Line2D([], [], color='#FFA500', linestyle='--', linewidth=1.5, label='Td ≥ 75°F',
                                  path_effects=[pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
        
        leg = fig.legend(handles=[isobar_line, dp_blue, dp_green, dp_orange], 
                         loc='lower left', bbox_to_anchor=(0.01, 0.015),
                         facecolor=LAND_COLOR, edgecolor='black', labelcolor='white',
                         fontsize=10, framealpha=1.0, borderpad=0.6,
                         handlelength=1.6, handletextpad=0.6)
        leg.get_frame().set_linewidth(1.5)

        ax.text(0.0, 1.02, "FrostByte Weather Atlantic: Surface Analysis", fontsize=20, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, transform=ax.transAxes, ha='left', va='bottom')
        ax.text(1.0, 1.02, valid_time.strftime('%d %B %Y %H:%MZ') + f" (+{forecast_hour}h)", fontsize=16, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, transform=ax.transAxes, ha='right', va='bottom')
        add_clean_colorbar(fig, ax, cf, 'Surface Temperature (°F)')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf, init_time, valid_time
    except Exception as e:
        logging.error(f"Error in generate_mslp_temp_map: {e}", exc_info=True)
        return None, None, None

# ====================================================================
# 6. SCRIPT RUNNERS
# ====================================================================

OUTPUT_DIR = "/media/evan/Main/frostbyte/frostbyte_project/output/Atlantic Maps"

def save_map_to_disk(image_bytes, filename):
    if image_bytes:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        full_path = os.path.join(OUTPUT_DIR, filename)
        
        with open(full_path, 'wb') as f: 
            f.write(image_bytes.getbuffer())
            
        logging.info(f"✅ Successfully saved {full_path}")
        image_bytes.close()

def run_wind300(forecast_hour=0):
    ds, init_time, valid_time = get_gfs_data_for_level(30000, forecast_hour)
    if ds: save_map_to_disk(generate_map(ds, init_time, valid_time, 30000, 'wind_speed', 'cool', 'FrostByte Weather: 300-hPa Wind Speeds and Heights', 'Wind Speed (knots)', np.arange(20, 201, 10), forecast_hour), f'Awind300_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

def run_wind500(forecast_hour=0):
    ds, init_time, valid_time = get_gfs_data_for_level(50000, forecast_hour)
    if ds: save_map_to_disk(generate_map(ds, init_time, valid_time, 50000, 'wind_speed', 'YlOrBr', 'FrostByte Weather: 500-hPa Wind Speeds and Heights', 'Wind Speed (knots)', np.arange(20, 181, 10), forecast_hour), f'Awind500_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

def run_vort500(forecast_hour=0):
    ds, init_time, valid_time = get_gfs_data_for_level(50000, forecast_hour)
    if ds: save_map_to_disk(generate_map(ds, init_time, valid_time, 50000, 'vorticity', 'seismic', 'FrostByte Weather: 500-hPa Relative Vorticity', r'Vorticity ($10^{-5}$ s$^{-1}$)', np.linspace(-20, 20, 41), forecast_hour, plot_vort_max=True), f'Avort500_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

def run_500wv(forecast_hour=0):
    ds, init_time, valid_time = get_gfs_data_for_level(50000, forecast_hour)
    sat_ds, sat_var = fetch_goes_ch9()
    if ds: save_map_to_disk(generate_map(ds, init_time, valid_time, 50000, 'vorticity', 'RdBu_r', 'FrostByte Weather: 500mb Analysis — Water Vapor / Heights / Vort Max', 'Vorticity (10⁻⁵ s⁻¹)', np.linspace(-25, 25, 51), forecast_hour, wv_background=True, plot_vort_max=True, sat_ds=sat_ds, sat_var=sat_var), f'A500wv_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

def run_rh700(forecast_hour=0):
    ds, init_time, valid_time = get_gfs_data_for_level(70000, forecast_hour)
    if ds: save_map_to_disk(generate_map(ds, init_time, valid_time, 70000, 'relative_humidity', 'BuGn', 'FrostByte Weather: 700-hPa Relative Humidity', 'Relative Humidity (%)', np.arange(10, 101, 10), forecast_hour), f'Arh700_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

def run_fronto700(forecast_hour=0):
    ds, init_time, valid_time = get_gfs_data_for_level(70000, forecast_hour)
    if ds: save_map_to_disk(generate_map(ds, init_time, valid_time, 70000, 'frontogenesis', 'RdBu_r', 'FrostByte Weather: 700-hPa Frontogenesis', 'Frontogenesis (K/100km/3hr)', np.linspace(-10, 10, 41), forecast_hour), f'Afronto700_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

def run_wind850(forecast_hour=0):
    ds, init_time, valid_time = get_gfs_data_for_level(85000, forecast_hour)
    if ds: save_map_to_disk(generate_map(ds, init_time, valid_time, 85000, 'wind_speed', 'YlOrBr', 'FrostByte Weather: 850-hPa Wind Speeds and Heights', 'Wind Speed (knots)', np.arange(20, 141, 10), forecast_hour), f'Awind850_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

def run_dew850(forecast_hour=0):
    ds, init_time, valid_time = get_gfs_data_for_level(85000, forecast_hour)
    if ds: save_map_to_disk(generate_map(ds, init_time, valid_time, 85000, 'dewpoint', 'BuGn', 'FrostByte Weather: 850-hPa Dewpoint', 'Dewpoint (°C)', np.arange(-40, 31, 2), forecast_hour), f'Adew850_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

def run_mAdv850(forecast_hour=0):
    ds, init_time, valid_time = get_gfs_data_for_level(85000, forecast_hour)
    if ds: save_map_to_disk(generate_map(ds, init_time, valid_time, 85000, 'moisture_advection', 'PRGn', 'FrostByte Weather: 850-hPa Moisture Advection', 'Moisture Advection (%/hour)', np.linspace(-20, 20, 41), forecast_hour), f'AmAdv850_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

def run_tAdv850(forecast_hour=0):
    ds, init_time, valid_time = get_gfs_data_for_level(85000, forecast_hour)
    if ds: save_map_to_disk(generate_map(ds, init_time, valid_time, 85000, 'temp_advection', 'coolwarm', 'FrostByte Weather: 850-hPa Temp Advection', 'Temp Advection (K/hour)', np.linspace(-20, 20, 41), forecast_hour), f'AtAdv850_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

def run_mslp_temp(forecast_hour=0):
    image_bytes, init_time, valid_time = generate_mslp_temp_map(forecast_hour)
    if image_bytes: save_map_to_disk(image_bytes, f'Amslp_temp_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

def run_divcon300(forecast_hour=0):
    ds, init_time, valid_time = get_gfs_data_for_level(30000, forecast_hour)
    if ds: save_map_to_disk(generate_map(ds, init_time, valid_time, 30000, 'divergence', 'RdBu_r', 'FrostByte Weather: 300-hPa Divergence/Convergence', r'Divergence ($10^{-5}$ s$^{-1}$)', np.linspace(-40, 40, 41), forecast_hour), f'Adivcon300_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png')

# ====================================================================
# 7. SCRIPT ENTRY
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Atlantic Weather Maps Generator (FrostByte Weather Aesthetic)")
    subparsers = parser.add_subparsers(dest='command', required=True)

    commands = ['wind300', 'wind500', 'vort500', '500wv', 'rh700', 'fronto700', 'wind850', 'dew850', 'mAdv850', 'tAdv850', 'mslp_temp', 'divcon300']
    for cmd in commands:
        cmd_parser = subparsers.add_parser(cmd)
        cmd_parser.add_argument('forecast_hour', type=int, nargs='?', default=0)

    args = parser.parse_args()
    
    cmd_map = {
        'wind300': run_wind300, 'wind500': run_wind500, 'vort500': run_vort500, '500wv': run_500wv,
        'rh700': run_rh700, 'fronto700': run_fronto700, 'wind850': run_wind850,
        'dew850': run_dew850, 'mAdv850': run_mAdv850, 'tAdv850': run_tAdv850,
        'mslp_temp': run_mslp_temp, 'divcon300': run_divcon300
    }
    
    if args.command in cmd_map:
        cmd_map[args.command](args.forecast_hour)

if __name__ == "__main__":
    main()
