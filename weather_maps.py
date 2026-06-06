#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced CONUS Weather Map Generator
Surface Maps: Blended HRRR/RAP with explicit high-terrain hypsometric reduction.
Upper Air Maps: RAP (Rapid Refresh) 13km High-Resolution Data.
Includes universal 4 dam upper-air intervals, dynamic colorbar locators, and optimized wind barbs.
"""

import logging
import io
import os
import sys
import pandas as pd
from datetime import datetime, timezone, timedelta
import requests
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import numpy as np
import xarray as xr

from scipy import ndimage
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from siphon.catalog import TDSCatalog

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- Constants & Layout Definitions ---
REGION = "CONUS"
EXTENT = [-125, -66.5, 20, 50]
FIG_BG_COLOR = '#333333'
OCEAN_COLOR = '#1A2A3A'
LAND_COLOR = '#2B2B2B'
GRIDLINE_COLOR = '#3932A0'
TEXT_OUTLINE = [pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()]
INTERSTATES_SHP = './shapefiles/tl_2023_us_primaryroads.shp' 
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')

# Standardized Text Styling for Axes and Labels
LABEL_STYLE = {
    'size': 11, 
    'fontweight': 'bold', 
    'color': 'white', 
    'path_effects': TEXT_OUTLINE
}

# Physics Constants
G = 9.80665          
RD = 287.05          
LAPSE_RATE = 0.0065  
EARTH_RADIUS = 6371000 
OMEGA = 7.292e-5  

# ====================================================================
# 1. COLORMAPS & STYLING
# ====================================================================

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
# 2. PHYSICS & CALCULATIONS
# ====================================================================

def calculate_adjusted_mslp(p_sfc, t_sfc_k, elevation_m):
    elevation_safe = np.where(elevation_m < 0, 0, elevation_m)
    base = 1.0 - ((LAPSE_RATE * elevation_safe) / (t_sfc_k + (LAPSE_RATE * elevation_safe)))
    exponent = - (G / (RD * LAPSE_RATE))
    return p_sfc * (base ** exponent)

def get_dlon_dlat(lat, lon):
    if lon.ndim == 2:
        dlon = np.abs(np.mean(np.diff(lon, axis=1)))
        dlat = np.abs(np.mean(np.diff(lat, axis=0)))
    else:
        dlon = np.abs(lon[1] - lon[0])
        dlat = np.abs(lat[1] - lat[0])
    if dlon == 0: dlon = 0.1
    if dlat == 0: dlat = 0.1
    return dlon, dlat

def compute_wind_speed(u, v):
    return np.sqrt(u**2 + v**2) * 1.94384

def compute_vorticity(u, v, lat, lon):
    lat_rad = np.deg2rad(lat)
    dlon, dlat = get_dlon_dlat(lat, lon)
    dx = EARTH_RADIUS * np.cos(lat_rad) * np.deg2rad(dlon)
    dy = EARTH_RADIUS * np.deg2rad(dlat)
    v_x = np.full_like(v, np.nan)
    u_y = np.full_like(u, np.nan)
    
    if dx.ndim == 2: v_x[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx[:, 1:-1])
    else: v_x[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx[:, None])
    u_y[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * dy)
    
    return (v_x - u_y) * 1e5

def compute_advection(phi, u, v, lat, lon):
    lat_rad = np.deg2rad(lat)
    dlon, dlat = get_dlon_dlat(lat, lon)
    dx = EARTH_RADIUS * np.cos(lat_rad) * np.deg2rad(dlon)
    dy = EARTH_RADIUS * np.deg2rad(dlat)
    phi_x = np.full_like(phi, np.nan)
    phi_y = np.full_like(phi, np.nan)
    
    if dx.ndim == 2: phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx[:, 1:-1])
    else: phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx[:, None])
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
    dlon, dlat = get_dlon_dlat(lat, lon)
    dx = EARTH_RADIUS * np.cos(lat_rad) * np.deg2rad(dlon)
    dy = EARTH_RADIUS * np.deg2rad(dlat)
    u_x = np.full_like(u, np.nan)
    v_y = np.full_like(v, np.nan)
    
    if dx.ndim == 2: u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx[:, 1:-1])
    else: u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx[:, None])
    v_y[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
    
    return (u_x + v_y) * 1e5

def frontogenesis_700hPa(T, u, v, lat, lon):
    try:
        lat_rad = np.deg2rad(lat)
        dlon, dlat = get_dlon_dlat(lat, lon)
        dx = EARTH_RADIUS * np.cos(lat_rad) * np.deg2rad(dlon)
        dy = EARTH_RADIUS * np.deg2rad(dlat)

        dT_dx, dT_dy = np.full_like(T, np.nan), np.full_like(T, np.nan)
        du_dx, du_dy = np.full_like(u, np.nan), np.full_like(u, np.nan)
        dv_dx, dv_dy = np.full_like(v, np.nan), np.full_like(v, np.nan)

        if dx.ndim == 2:
            dT_dx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * dx[:, 1:-1])
            du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx[:, 1:-1])
            dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx[:, 1:-1])
        else:
            dT_dx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * dx[:, None])
            du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx[:, None])
            dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx[:, None])

        dT_dy[1:-1, :] = (T[:-2, :] - T[2:, :]) / (2 * dy)
        du_dy[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * dy)
        dv_dy[1:-1, :] = (v[:-2, :] - v[2:, :]) / (2 * dy)

        mag_grad_T = np.sqrt(dT_dx**2 + dT_dy**2)
        mag_grad_T_safe = np.where(mag_grad_T < 1e-10, 1e-10, mag_grad_T)

        F = (1 / mag_grad_T_safe) * (- (dT_dx**2) * du_dx - dT_dx * dT_dy * (dv_dx + du_dy) - (dT_dy**2) * dv_dy)
        return F * 1e5 * 10800
    except Exception as e:
        logging.error(f"Error computing frontogenesis: {e}")
        return None

# ====================================================================
# 3. Model Fetchers
# ====================================================================

def fetch_rap_full(extent):
    logging.info("Fetching RAP Surface Variables & Elevation Geopotential...")
    try:
        buffer = 8.0
        bbox = [extent[2]-buffer, extent[3]+buffer, extent[0]-buffer, extent[1]+buffer] 
        cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RAP/CONUS_13km/latest.xml')
        ds = cat.datasets[0] 
        ncss = ds.subset()
        available = set(ncss.variables)
        
        def find_var(options):
            for opt in options:
                if opt in available: return opt
            return None

        temp_var = find_var(['Temperature_height_above_ground', 'Temperature_surface', 'tmp2m'])
        u_var = find_var(['u-component_of_wind_height_above_ground', 'u-component_of_wind_surface'])
        v_var = find_var(['v-component_of_wind_height_above_ground', 'v-component_of_wind_surface'])
        sfc_press_var = find_var(['Pressure_surface', 'pressure_surface', 'pres_sfc'])
        orog_var = find_var(['Geopotential_height_surface', 'orography', 'HGT_surface'])

        query = ncss.query()
        query.lonlat_box(north=bbox[1], south=bbox[0], east=bbox[3], west=bbox[2])
        query.accept('netcdf4')
        query.add_lonlat(True)
        
        vars_to_fetch = [v for v in [temp_var, u_var, v_var, sfc_press_var, orog_var] if v]
        query.variables(*vars_to_fetch)
        data = ncss.get_data(query)
        
        lat_var = next((v for v in data.variables if 'lat' in v.lower()), None)
        lon_var = next((v for v in data.variables if 'lon' in v.lower()), None)
        
        def get_val(v_name):
            val = data.variables[v_name][:].squeeze()
            if val.ndim == 3: val = val[0]
            return val

        t_k = get_val(temp_var)
        p_sfc = get_val(sfc_press_var)
        elev = get_val(orog_var) if orog_var else np.zeros_like(t_k)
        
        calculated_mslp = calculate_adjusted_mslp(p_sfc, t_k, elev) / 100.0

        return {
            'lat': data.variables[lat_var][:], 'lon': data.variables[lon_var][:],
            'temp': (t_k - 273.15) * 9/5 + 32,
            'u': get_val(u_var) * 1.94384, 'v': get_val(v_var) * 1.94384,
            'pressure': calculated_mslp
        }
    except Exception as e:
        logging.error(f"RAP Surface Fetch Failed: {e}")
        return None

def fetch_hrrr_sfc(extent):
    logging.info("Fetching HRRR Surface Data & Surface Pressure...")
    try:
        buffer = 8.0
        bbox = [extent[2]-buffer, extent[3]+buffer, extent[0]-buffer, extent[1]+buffer]
        cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/HRRR/CONUS_2p5km/latest.xml')
        ds = cat.datasets[0] 
        ncss = ds.subset()
        available = set(ncss.variables)
        
        def find(opts):
            for o in opts: 
                if o in available: return o
            return None

        t_var = find(['Temperature_height_above_ground', 'Temperature_surface'])
        u_var = find(['u-component_of_wind_height_above_ground', 'u-component_of_wind_surface'])
        v_var = find(['v-component_of_wind_height_above_ground', 'v-component_of_wind_surface'])
        sfc_p_var = find(['Pressure_surface', 'pres_sfc'])
        orog_var = find(['Geopotential_height_surface', 'HGT_surface'])

        query = ncss.query()
        query.lonlat_box(north=bbox[1], south=bbox[0], east=bbox[3], west=bbox[2])
        query.accept('netcdf4')
        query.add_lonlat(True)
        
        valid_vars = [v for v in [t_var, u_var, v_var, sfc_p_var, orog_var] if v]
        query.variables(*valid_vars)
        data = ncss.get_data(query)
        
        lat_var = next((v for v in data.variables if 'lat' in v.lower()), None)
        lon_var = next((v for v in data.variables if 'lon' in v.lower()), None)
        
        def get(name):
            if not name: return np.zeros_like(data.variables[lat_var][:])
            val = data.variables[name][:].squeeze()
            if val.ndim == 3: val = val[0]
            return val

        t_k = get(t_var)
        p_sfc = get(sfc_p_var)
        elev = get(orog_var) if orog_var else np.zeros_like(t_k)
        
        calculated_mslp = calculate_adjusted_mslp(p_sfc, t_k, elev) / 100.0

        return {
            'lat': data.variables[lat_var][:], 'lon': data.variables[lon_var][:],
            'temp': (t_k - 273.15) * 9/5 + 32,
            'u': get(u_var) * 1.94384, 'v': get(v_var) * 1.94384,
            'pressure': calculated_mslp
        }
    except Exception as e:
        logging.error(f"HRRR Surface Fetch Failed: {e}")
        return None

def blend_surface_models(rap_data, hrrr_data, rtma_data, extent):
    """
    Blends the broad RAP background, high-res HRRR forecast, 
    and ground-truth RTMA observations into a single cohesive point cloud.
    Includes a Strict Reality Filter to safely destroy -9999.0 THREDDS fill-values.
    """
    logging.info("Blending Unified High-Terrain Surface Models (RAP/HRRR/RTMA)...")
    grid_lon = np.arange(extent[0]-0.5, extent[1]+0.5, 0.1)
    grid_lat = np.arange(extent[2]-0.5, extent[3]+0.5, 0.1)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
    
    points, temps, us, vs, press = [], [], [], [], []
    
    def add_to_blend(data_dict, step):
        if not data_dict: return
        
        lon_arr = data_dict['lon'][::step, ::step].flatten()
        lat_arr = data_dict['lat'][::step, ::step].flatten()
        t_arr = data_dict['temp'][::step, ::step].flatten()
        u_arr = data_dict['u'][::step, ::step].flatten()
        v_arr = data_dict['v'][::step, ::step].flatten()
        
        has_p = data_dict.get('pressure') is not None
        
        # 1. Fill masked arrays with NaN
        t_clean = np.ma.filled(t_arr, np.nan)
        u_clean = np.ma.filled(u_arr, np.nan)
        v_clean = np.ma.filled(v_arr, np.nan)
        
        # 2. THE REALITY FILTER: Explicitly bounds the data to realistic meteorological limits.
        # This prevents THREDDS -9999.0 fill values from causing massive black tears over the oceans.
        valid = (~np.isnan(t_clean)) & (t_clean > -100) & (t_clean < 150)
        valid &= (~np.isnan(u_clean)) & (np.abs(u_clean) < 200)
        valid &= (~np.isnan(v_clean)) & (np.abs(v_clean) < 200)
        
        if has_p:
            p_arr = data_dict['pressure'][::step, ::step].flatten()
            p_clean = np.ma.filled(p_arr, np.nan)
            valid &= (~np.isnan(p_clean)) & (p_clean > 800) & (p_clean < 1150)
            
        lon_arr = np.where(lon_arr > 180, lon_arr - 360, lon_arr)
        
        # Only append data that survives the Reality Filter
        points.append(np.column_stack((lon_arr[valid], lat_arr[valid])))
        temps.append(t_clean[valid])
        us.append(u_clean[valid])
        vs.append(v_clean[valid])
        if has_p:
            press.append(p_clean[valid])

    # Combine models into a unified point cloud
    add_to_blend(rap_data, 1)
    add_to_blend(hrrr_data, 3)
    add_to_blend(rtma_data, 3)

    if not points: return None, None, None, None, None, None

    all_points = np.vstack(points)
    all_temps = np.concatenate(temps)
    all_us = np.concatenate(us)
    all_vs = np.concatenate(vs)
    all_press = np.concatenate(press) if press else None
    
    def interp_safe(vals, sigma=1.0):
        # Uses Delaunay triangulation across the unified point cloud for a perfectly smooth gradient
        res = griddata(all_points, vals, (grid_x, grid_y), method='linear')
        mask_nan = np.isnan(res)
        if np.any(mask_nan):
            res[mask_nan] = griddata(all_points, vals, (grid_x[mask_nan], grid_y[mask_nan]), method='nearest')
        return gaussian_filter(res, sigma=sigma) 

    try:
        # Standard smoothing restores the gorgeous, seamless color transitions
        g_temp = interp_safe(all_temps, sigma=1.0)
        g_u = interp_safe(all_us, sigma=1.0)
        g_v = interp_safe(all_vs, sigma=1.0)
        
        # Heavier smoothing (4.0) applies ONLY to pressure to clean up the noisy Rockies isobars
        g_p = interp_safe(all_press, sigma=4.0) if all_press is not None else None
        
        return grid_y, grid_x, g_temp, g_u, g_v, g_p
    except Exception as e:
        logging.error(f"Blending failed: {e}")
        return None, None, None, None, None, None

def fetch_rtma_sfc(extent):
    logging.info("Fetching RTMA 2.5km Surface Analysis...")
    try:
        buffer = 8.0
        bbox = [extent[2]-buffer, extent[3]+buffer, extent[0]-buffer, extent[1]+buffer]
        
        cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RTMA/CONUS_2p5km/latest.xml')
        ds = cat.datasets[0] 
        ncss = ds.subset()
        available = set(ncss.variables)
        
        # Restored the robust fuzzy matcher 
        def get_best_match(keywords, fallback):
            for kw in keywords:
                if kw in available: return kw
            for var_name in available:
                if 'error' in var_name.lower() or 'uncertainty' in var_name.lower():
                    continue
                if all(k in var_name for k in keywords[0].split('_') if len(k)>3):
                    return var_name
            return fallback

        t_var = get_best_match(['Temperature_height_above_ground', 'Temperature_surface', 'tmp2m'], 'Temperature_height_above_ground')
        u_var = get_best_match(['u-component_of_wind_height_above_ground', 'u-component_of_wind_surface'], 'u-component_of_wind_height_above_ground')
        v_var = get_best_match(['v-component_of_wind_height_above_ground', 'v-component_of_wind_surface'], 'v-component_of_wind_height_above_ground')
        sfc_p_var = get_best_match(['Pressure_surface', 'MSL_pressure_surface', 'MSLP_MAPS_System_Reduction_msl'], 'Pressure_surface')
        orog_var = get_best_match(['Geopotential_height_surface', 'HGT_surface'], 'Geopotential_height_surface')

        query = ncss.query()
        query.lonlat_box(north=bbox[1], south=bbox[0], east=bbox[3], west=bbox[2])
        query.accept('netcdf4')
        query.add_lonlat(True)
        
        valid_vars = [v for v in [t_var, u_var, v_var, sfc_p_var, orog_var] if v]
        query.variables(*valid_vars)
        data = ncss.get_data(query)
        
        lat_var = next((v for v in data.variables if 'lat' in v.lower()), None)
        lon_var = next((v for v in data.variables if 'lon' in v.lower()), None)
        
        def get(name):
            if not name: return np.zeros_like(data.variables[lat_var][:])
            val = data.variables[name][:].squeeze()
            if val.ndim == 3: val = val[0]
            return val

        t_k = get(t_var)
        p_sfc = get(sfc_p_var)
        elev = get(orog_var) if orog_var else np.zeros_like(t_k)
        
        calculated_mslp = calculate_adjusted_mslp(p_sfc, t_k, elev) / 100.0

        return {
            'lat': data.variables[lat_var][:], 'lon': data.variables[lon_var][:],
            'temp': (t_k - 273.15) * 9/5 + 32,
            'u': get(u_var) * 1.94384, 'v': get(v_var) * 1.94384,
            'pressure': calculated_mslp
        }
    except Exception as e:
        logging.error(f"RTMA Fetch Failed: {e}")
        return None
    
def get_rap_data_for_level(level):
    logging.info(f"Fetching RAP 13km Data for {level} Pa...")
    catalog_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RAP/CONUS_13km/latest.xml'
    try:
        cat = TDSCatalog(catalog_url)
        latest_dataset = cat.datasets[0]
        ncss = latest_dataset.subset()

        query = ncss.query()
        query.accept('netcdf4')
        query.add_lonlat(True)
        query.time(datetime.now(timezone.utc))
        query.variables('Geopotential_height_isobaric',
                        'u-component_of_wind_isobaric',
                        'v-component_of_wind_isobaric',
                        'Relative_humidity_isobaric',
                        'Temperature_isobaric')
        query.vertical_level([level])
        query.lonlat_box(north=55, south=15, east=-60, west=-130)

        data = ncss.get_data(query)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
        
        time_var = next((v for v in ds.variables if 'time' in v.lower()), None)
        if time_var is not None:
            time_val = ds[time_var].values
            run_date = pd.to_datetime(time_val[0]).to_pydatetime() if isinstance(time_val, np.ndarray) else pd.to_datetime(time_val).to_pydatetime()
        else:
            run_date = datetime.now(timezone.utc)
            
        return ds, run_date
    except Exception as e:
        logging.error(f"Error fetching RAP upper-air: {e}")
        return None, None

def get_dynamic_time_dims(ds):
    time_dims = {}
    for dim in ds.dims:
        if 'time' in dim.lower(): time_dims[dim] = 0
    return time_dims

# ====================================================================
# 4. OVERLAYS, LAYOUT MANAGERS, & COLORBARS
# ====================================================================

def draw_interstates_from_shapefile(ax, shapefile=INTERSTATES_SHP):
    if not os.path.exists(shapefile): return
    try:
        reader = shpreader.Reader(shapefile)
        interstate_geoms = []
        for record, geometry in zip(reader.records(), reader.geometries()):
            attrs = record.attributes if hasattr(record, 'attributes') else record.__dict__
            is_interstate = False
            for key in ('RTTYP', 'RTE_TYPE', 'MTFCC'):
                if key in attrs and str(attrs.get(key)).upper() == 'I': is_interstate = True
            if not is_interstate and 'FULLNAME' in attrs and 'INTERSTATE' in str(attrs.get('FULLNAME')).upper():
                is_interstate = True
            if is_interstate: interstate_geoms.append(geometry)
        if interstate_geoms:
            ax.add_geometries(interstate_geoms, ccrs.PlateCarree(), edgecolor='#D14900', 
                              facecolor='none', linewidth=1.2, zorder=3.5, alpha=0.9)
    except Exception as e: pass

def draw_major_cities(ax):
    cities = {
        'Seattle': (-122.33, 47.60), 'Los Angeles': (-118.24, 34.05),
        'Denver': (-104.99, 39.73), 'Chicago': (-87.62, 41.87),
        'New York': (-74.00, 40.71), 'Houston': (-95.36, 29.76),
        'Miami': (-80.19, 25.76), 'Atlanta': (-84.38, 33.74),
        'Cartersville': (-84.80, 34.16)
    }
    for name, (lon, lat) in cities.items():
        ax.plot(lon, lat, marker='o', color='white', markeredgecolor='black', markersize=5, transform=ccrs.PlateCarree(), zorder=5)
        ax.text(lon + 0.5, lat + 0.5, name, color='white', fontsize=11, fontweight='bold',
                path_effects=TEXT_OUTLINE, transform=ccrs.PlateCarree(), zorder=6)

def draw_isobars(ax, grid_x, grid_y, grid_p, levels, fmt='%d mb'):
    """
    Renders clean, high-visibility isobars.
    White center line with a persistent black 'halo' outline.
    """
    cs = ax.contour(grid_x, grid_y, grid_p, levels=levels, 
                    colors='white', linewidths=1.8, zorder=2.0, transform=ccrs.PlateCarree())
    
    # Line Outline
    cs.set_path_effects([pe.Stroke(linewidth=3.5, foreground='black'), pe.Normal()])
    
    # Text Labels
    clabels = ax.clabel(cs, fmt=fmt, inline=True, inline_spacing=12, fontsize=10, colors='white')
    
    for label in clabels:
        label.set_path_effects([pe.withStroke(linewidth=3, foreground='black'), pe.Normal()])
        label.set_fontweight('bold')
        
    return cs

def add_clean_colorbar(fig, ax, cf, label):
    """
    Renders a cleanly spaced colorbar, strictly enforcing ticks within bounds 
    and explicitly formatting labels to prevent Matplotlib ghost-stacking.
    """
    if cf is not None:
        cax = fig.add_axes([0.06, 0.08, 0.88, 0.025])
        cb = fig.colorbar(cf, cax=cax, orientation='horizontal', extend='both', extendfrac=0.04)
        
        # Colorbar Label & Outline
        cb.set_label(label, fontsize=13, fontweight='bold', color='white', labelpad=10)
        cb.ax.xaxis.label.set_path_effects(TEXT_OUTLINE)
        
        # Define standard uniform intervals
        if 'Temperature' in label or 'Dewpoint' in label: ticks = np.arange(-40, 131, 10)
        elif 'Relative Humidity' in label: ticks = np.arange(10, 101, 10)
        elif 'Wind Speed' in label: ticks = np.arange(20, 201, 20)
        elif 'Advection' in label or 'Vorticity' in label or 'Divergence' in label: ticks = np.arange(-20, 21, 5)
        else: ticks = cb.get_ticks()
        
        # OVERLAP FIX: Keep ticks strictly inside or exactly on the boundaries (vmin <= t <= vmax).
        # This brings back the Min and Max numbers on the absolute edges.
        vmin, vmax = cf.get_clim()
        valid_ticks = [t for t in ticks if vmin <= t <= vmax]
        
        cb.set_ticks(valid_ticks)
        cb.ax.tick_params(axis='x', length=0)
        
        # By forcing explicit string labels, we overwrite Matplotlib's native 'extend' ghost labels,
        # which cures the double-stacked number glitch permanently.
        cb.ax.set_xticklabels([str(int(t)) for t in valid_ticks], fontsize=11, fontweight='bold', color='white')
        
        # Apply text outline to the numbers
        for tick_label in cb.ax.get_xticklabels():
            tick_label.set_path_effects(TEXT_OUTLINE)


# ==========================================
# --- NAVY / MARINE MODULE ---
# ==========================================

def calc_marine_physics(water_temp_k, water_u, water_v, air_temp_f, air_dew_f, wind_speed_kts):
    """Calculates Latent Heat Flux potential and Ocean Vorticity (Upwelling)."""
    # NaN-safe gradients (land points are now NaN)
    grad_v = np.gradient(np.ma.masked_invalid(water_v))
    dv_dx = grad_v[1]
    
    grad_u = np.gradient(np.ma.masked_invalid(water_u))
    du_dy = grad_u[0]
    
    vorticity = dv_dx - du_dy
    water_c = water_temp_k - 273.15
    air_c = (air_temp_f - 32) * 5/9
    flux_potential = (water_c - air_c) * (wind_speed_kts * 0.514) 
    
    return vorticity, flux_potential, water_c


def fetch_ocean_currents(extent):
    """
    Fetches MARINE Data via direct HTTP download from NOAA NOMADS.
    Bypasses the retired OpenDAP service (SCN 25-81).
    CRITICAL FIX: Now properly masks land points so streamplot no longer draws
    straight lines across the continent.
    """
    import netCDF4 as nc
    print("\n--- MARINE DATA FETCH (RTOFS Direct Download) ---")
    logging.info("Attempting to fetch RTOFS data via direct HTTPS...")

    today = datetime.now(timezone.utc)
    dates_to_try = [today, today - timedelta(days=1), today - timedelta(days=2)]
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 Frostbyte/1.0'})

    files_to_try = [
        "rtofs_glo_2ds_f000_prog.nc", 
        "rtofs_glo_2ds_f024_prog.nc", 
        "rtofs_glo_2ds.f000.nc",       
        "rtofs_glo_2ds.f024.nc"        
    ]

    temp_file = "rtofs_temp.nc"

    for date_obj in dates_to_try:
        date_str = date_obj.strftime('%Y%m%d')
        base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod/rtofs.{date_str}"

        for filename in files_to_try:
            url = f"{base_url}/{filename}"
            print(f"• Trying {filename} for {date_str}...", end=" ", flush=True)

            try:
                # 1. Check if file exists
                head = session.head(url, timeout=10)
                if head.status_code != 200:
                    print("Not found.")
                    continue

                # 2. Download the file
                print("Downloading...", end=" ", flush=True)
                r = session.get(url, stream=True, timeout=30)
                if r.status_code == 200:
                    with open(temp_file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk: f.write(chunk)
                else:
                    print("Failed.")
                    continue

                print("Extracting...", end=" ", flush=True)
                
                # 3. Extract using low-level netCDF4
                ds = nc.Dataset(temp_file, 'r')

                lat_var = next((v for v in ['lat', 'Latitude', 'latitude'] if v in ds.variables), None)
                lon_var = next((v for v in ['lon', 'Longitude', 'longitude'] if v in ds.variables), None)

                lats_raw = ds.variables[lat_var][:]
                lons_raw = ds.variables[lon_var][:]
                lons_raw = np.where(lons_raw > 180, lons_raw - 360, lons_raw)

                lat_min, lat_max = extent[2] - 2, extent[3] + 2
                lon_min, lon_max = extent[0] - 2, extent[1] + 2

                # 4. Find bounding box indices
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
                    continue

                y_min_idx, y_max_idx = y_idxs.min(), y_idxs.max()
                x_min_idx, x_max_idx = x_idxs.min(), x_idxs.max()

                # Slice coordinates
                if lats_raw.ndim == 2:
                    lats = lats_raw[y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
                    lons = lons_raw[y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
                else:
                    lats = lats_raw[y_min_idx:y_max_idx+1]
                    lons = lons_raw[x_min_idx:x_max_idx+1]

                # 5. Expanded variable name candidates (robust against file changes)
                sst_candidates = ['sst', 'SST', 'temperature', 'temp', 'sea_surface_temperature', 'Temperature']
                u_candidates   = ['u_velocity', 'u', 'water_u', 'U', 'u_comp', 'velocity_u']
                v_candidates   = ['v_velocity', 'v', 'water_v', 'V', 'v_comp', 'velocity_v']
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

                # Extract raw data
                w_temp = safe_slice(sst_var)
                if w_temp is None:
                    w_temp = np.full(lats.shape, np.nan, dtype=np.float32)
                else:
                    w_temp = np.asarray(w_temp, dtype=np.float32)
                    if np.nanmax(w_temp) < 100:          # likely Celsius → convert to Kelvin
                        w_temp += 273.15

                w_u = safe_slice(u_var)
                if w_u is None:
                    w_u = np.full_like(w_temp, np.nan)
                else:
                    w_u = np.asarray(w_u, dtype=np.float32)

                w_v = safe_slice(v_var)
                if w_v is None:
                    w_v = np.full_like(w_temp, np.nan)
                else:
                    w_v = np.asarray(w_v, dtype=np.float32)

                salinity = safe_slice(sal_var)
                if salinity is None:
                    salinity = np.full_like(w_temp, np.nan)
                else:
                    salinity = np.asarray(salinity, dtype=np.float32)

                # === CRITICAL FIX: Ocean mask to kill straight-line artifacts ===
                # Only keep points with realistic ocean SST (Kelvin)
                ocean_mask = np.isfinite(w_temp) & (w_temp > 270) & (w_temp < 310)
                
                w_temp = np.where(ocean_mask, w_temp, np.nan)
                w_u    = np.where(ocean_mask, w_u,    np.nan)
                w_v    = np.where(ocean_mask, w_v,    np.nan)
                salinity = np.where(ocean_mask, salinity, np.nan)

                print("Success! ✅ (ocean masked)")
                ds.close()
                if os.path.exists(temp_file): os.remove(temp_file)
                    
                return lats, lons, w_temp, w_u, w_v, salinity

            except Exception as e:
                print(f"Error ({e}).")
                if 'ds' in locals():
                    try: ds.close()
                    except: pass
                if os.path.exists(temp_file):
                    try: os.remove(temp_file)
                    except: pass
                continue

    print("• All direct downloads failed.")
    return None, None, None, None, None, None

# ====================================================================
# 5. CORE GENERATION PLOTS 
# ====================================================================

def generate_mslp_temp_map():
    logging.info("Starting Clean 2mb Surface Analysis Generator...")
    run_date = datetime.now(timezone.utc)
    
    rap_data = fetch_rap_full(EXTENT)
    hrrr_data = fetch_hrrr_sfc(EXTENT)
    rtma_data = fetch_rtma_sfc(EXTENT)
    ncom_lat, ncom_lon, ncom_temp_k, ncom_u, ncom_v, ncom_sal = fetch_ocean_currents(EXTENT)
    
    # Heuristic: determine which source likely provided the data for logging
    try:
        if ncom_sal is not None and np.any(~np.isnan(ncom_sal) & (ncom_sal != 0)):
            _marine_source = 'RTOFS/HYCOM (salinity present)'
        else:
            _marine_source = 'IEM mesonet (station-gridded)'
    except Exception:
        _marine_source = 'unknown'
    print(f"Marine currents source: {_marine_source}")
    
    grid_y, grid_x, g_temp, g_u, g_v, g_p = blend_surface_models(rap_data, hrrr_data, rtma_data, EXTENT)

    fig = plt.figure(figsize=(18, 11), facecolor=FIG_BG_COLOR)
    ax = fig.add_axes([0.06, 0.16, 0.88, 0.75], projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1.4, zorder=3)
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linestyle=':', linewidth=1.4, zorder=3)
    ax.add_feature(cfeature.STATES, edgecolor='black', linestyle=':', linewidth=0.9, zorder=3)
    
    # Axis Gridline Styling
    gl = ax.gridlines(draw_labels=True, linewidth=1.0, color=GRIDLINE_COLOR, alpha=0.4, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = LABEL_STYLE
    gl.ylabel_style = LABEL_STYLE

    cf_temp = None
    if g_temp is not None:
        temp_cmap = get_ga_temp_cmap()
        temp_levels = np.arange(-40, 121, 2.0)
        cf_temp = ax.contourf(grid_x, grid_y, g_temp, levels=temp_levels, cmap=temp_cmap, 
                              norm=mcolors.Normalize(vmin=-40.0, vmax=120.0), 
                              transform=ccrs.PlateCarree(), extend='both', alpha=0.85, zorder=1.5)

    # === PLOT MARINE OCEAN CURRENTS ===
    if (ncom_lat is not None and 
        ncom_u is not None and 
        np.any(np.isfinite(ncom_u))):   # ← only plot if we actually have valid ocean data
        print("• Plotting Marine/Ocean Layer (Dynamic/Outlined)...", end=" ", flush=True)
        try:
            # NO MESHGRID: Use raw RTOFS 2D arrays directly
            nc_x, nc_y = ncom_lon, ncom_lat
            marine_vort, _, _ = calc_marine_physics(ncom_temp_k, ncom_u, ncom_v, 70, 60, 10)

            # 1. Vorticity contours
            ax.contour(nc_x, nc_y, marine_vort, levels=[-0.0001, -0.00005, 0.00005, 0.0001],
                       colors=['blue', 'cyan', 'orange', 'red'], linewidths=1.2,
                       alpha=0.6, transform=ccrs.PlateCarree(), zorder=1.56)

            # 2. Dynamic Streamplot (now cleanly stops at coastlines)
            stride = 6
            strm = ax.streamplot(nc_x[::stride, ::stride], nc_y[::stride, ::stride], 
                                 ncom_u[::stride, ::stride], ncom_v[::stride, ::stride], 
                                 color=ncom_temp_k[::stride, ::stride], cmap='plasma',
                                 density=1.2, linewidth=1.5, arrowsize=1.5,
                                 transform=ccrs.PlateCarree(), zorder=2.0)
            
            # Isobar-style outline
            strm.lines.set_path_effects([pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
            
            # Dedicated colorbar for ocean temperature
            cax_marine = fig.add_axes([0.02, 0.2, 0.015, 0.5])
            fig.colorbar(strm.lines, cax=cax_marine, label='Ocean Temp (K)')
            
            print("Done. ✅")
        except Exception as e:
            print(f"Marine Error: {e} ❌")
            logging.error(f"Marine Plotting Error: {e}")

    # === PLOT ISOBARS & BARBS ===
    if g_p is not None:
        draw_isobars(ax, grid_x, grid_y, g_p, np.arange(950, 1050, 2), fmt='%d mb')
        
        skip = 13
        u_kts, v_kts = g_u[::skip, ::skip], g_v[::skip, ::skip]
        ax.barbs(grid_x[::skip, ::skip], grid_y[::skip, ::skip], u_kts, v_kts, 
                 transform=ccrs.PlateCarree(), length=5.5, color='black', zorder=2.5)

    draw_interstates_from_shapefile(ax)
    draw_major_cities(ax)

    ax.set_title("CONUS Surface Analysis & High-Terrain Adjusted MSLP Gradient", fontsize=20, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, loc='left', pad=15)
    ax.set_title(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=16, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, loc='right', pad=15)

    add_clean_colorbar(fig, ax, cf_temp, 'Surface Temperature (°F)')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"mslp_temp_{run_date.strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.2)
    logging.info(f"✅ Adjusted High-Terrain Map Saved: {out_path}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    return buf, run_date

def generate_upper_air_map(ds, run_date, level, variable, cmap, title, cb_label, levels=None):
    try:
        time_dims = get_dynamic_time_dims(ds)
        ds = ds.isel(**time_dims)
        
        lat_var = next((v for v in ds.variables if 'lat' in v.lower() and 'isobaric' not in v.lower()), None)
        lon_var = next((v for v in ds.variables if 'lon' in v.lower() and 'isobaric' not in v.lower()), None)
        
        if not lat_var or not lon_var:
            logging.error("Valid Lat/Lon arrays not found in dataset.")
            return None

        lat_2d = ds[lat_var].values
        lon_2d = ds[lon_var].values
        
        if lon_2d.ndim == 1:
            lon_2d, lat_2d = np.meshgrid(lon_2d, lat_2d)
            
        lon_2d = np.where(lon_2d > 180, lon_2d - 360, lon_2d)

        iso_dim = next((d for d in ds.dims if 'isobaric' in d.lower()), 'isobaric')
        
        heights_m = ds['Geopotential_height_isobaric'].sel({iso_dim: level}, method='nearest').squeeze().values.copy()
        heights_dam = ndimage.gaussian_filter(heights_m, sigma=3, order=0) / 10.0

        u_wind = ds['u-component_of_wind_isobaric'].sel({iso_dim: level}, method='nearest').squeeze().values.copy()
        v_wind = ds['v-component_of_wind_isobaric'].sel({iso_dim: level}, method='nearest').squeeze().values.copy()

        if variable == 'wind_speed': data = compute_wind_speed(u_wind, v_wind)
        elif variable == 'vorticity': data = compute_vorticity(u_wind, v_wind, lat_2d, lon_2d)
        elif variable == 'relative_humidity': data = ds['Relative_humidity_isobaric'].sel({iso_dim: level}, method='nearest').squeeze().values.copy()
        elif variable == 'temp_advection':
            temp = ds['Temperature_isobaric'].sel({iso_dim: level}, method='nearest').squeeze().values.copy()
            data = compute_advection(temp, u_wind, v_wind, lat_2d, lon_2d) * 3600
        elif variable == 'moisture_advection':
            rh = ds['Relative_humidity_isobaric'].sel({iso_dim: level}, method='nearest').squeeze().values.copy()
            data = compute_advection(rh, u_wind, v_wind, lat_2d, lon_2d) * 1e4
        elif variable == 'dewpoint':
            temp = ds['Temperature_isobaric'].sel({iso_dim: level}, method='nearest').squeeze().values.copy()
            rh = ds['Relative_humidity_isobaric'].sel({iso_dim: level}, method='nearest').squeeze().values.copy()
            data = compute_dewpoint(temp, rh)
        elif variable == 'divergence': data = compute_divergence(u_wind, v_wind, lat_2d, lon_2d)
        elif variable == 'frontogenesis':
            temp_k = ds['Temperature_isobaric'].sel({iso_dim: level}, method='nearest').squeeze().values.copy()
            temp = temp_k - 273.15
            data = frontogenesis_700hPa(temp, u_wind, v_wind, lat_2d, lon_2d)

        fig = plt.figure(figsize=(18, 11), facecolor=FIG_BG_COLOR)
        ax = fig.add_axes([0.06, 0.16, 0.88, 0.75], projection=ccrs.PlateCarree())
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1.4, zorder=3)
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linestyle=':', linewidth=1.4, zorder=3)
        ax.add_feature(cfeature.STATES, edgecolor='black', linestyle=':', linewidth=0.9, zorder=3)
        
        # Axis Gridline Styling (Requested: Outlines for Lat/Lon)
        gl = ax.gridlines(draw_labels=True, linewidth=1.0, color=GRIDLINE_COLOR, alpha=0.4, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = LABEL_STYLE
        gl.ylabel_style = LABEL_STYLE

        draw_interstates_from_shapefile(ax)
        draw_major_cities(ax)

        if levels is None:
            cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=ccrs.PlateCarree(), extend='both', alpha=0.85)
        else:
            cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=ccrs.PlateCarree(), levels=levels, extend='both', alpha=0.85)

        if level == 30000: h_levels = np.arange(800, 1040, 4) 
        elif level == 50000: h_levels = np.arange(480, 620, 4)
        elif level == 70000: h_levels = np.arange(210, 350, 4)
        elif level == 85000: h_levels = np.arange(90, 230, 4) 
        else: h_levels = None

        if h_levels is not None:
            draw_isobars(ax, lon_2d, lat_2d, heights_dam, h_levels, fmt='%i dam')

        skip = 24
        u_wind_knots, v_wind_knots = u_wind * 1.94384, v_wind * 1.94384
        ax.barbs(lon_2d[::skip, ::skip], lat_2d[::skip, ::skip], u_wind_knots[::skip, ::skip], v_wind_knots[::skip, ::skip], 
                 transform=ccrs.PlateCarree(), length=6, color='black', zorder=4)

        ax.set_title(f"{title}", fontsize=20, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, loc='left', pad=15)
        ax.set_title(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=16, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, loc='right', pad=15)

        add_clean_colorbar(fig, ax, cf, cb_label)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        logging.error(f"Upper Air Generation Error: {e}", exc_info=True)
        return None

# ====================================================================
# 6. EXPLICIT CONTOUR LEVELS & RUNNERS
# ====================================================================

def run_wind300():
    ds, run_date = get_rap_data_for_level(30000)
    if ds:
        buf = generate_upper_air_map(ds, run_date, 30000, 'wind_speed', 'cool', 'RAP 13km: 300-hPa Wind Speeds and Heights', 'Wind Speed (knots)', levels=np.arange(20, 201, 10))
        if buf:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f'wind300_{run_date.strftime("%Y%m%d_%H%MZ")}.png'), 'wb') as f: f.write(buf.getbuffer())

def run_wind500():
    ds, run_date = get_rap_data_for_level(50000)
    if ds:
        buf = generate_upper_air_map(ds, run_date, 50000, 'wind_speed', 'YlOrBr', 'RAP 13km: 500-hPa Wind Speeds and Heights', 'Wind Speed (knots)', levels=np.arange(20, 181, 10))
        if buf:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f'wind500_{run_date.strftime("%Y%m%d_%H%MZ")}.png'), 'wb') as f: f.write(buf.getbuffer())

def run_vort500():
    ds, run_date = get_rap_data_for_level(50000)
    if ds:
        buf = generate_upper_air_map(ds, run_date, 50000, 'vorticity', 'seismic', 'RAP 13km: 500-hPa Relative Vorticity and Heights', 'Vorticity ($10^{-5}$ s$^{-1}$)', levels=np.linspace(-20, 20, 41))
        if buf:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f'vort500_{run_date.strftime("%Y%m%d_%H%MZ")}.png'), 'wb') as f: f.write(buf.getbuffer())

def run_rh700():
    ds, run_date = get_rap_data_for_level(70000)
    if ds:
        buf = generate_upper_air_map(ds, run_date, 70000, 'relative_humidity', 'BuGn', 'RAP 13km: 700-hPa Relative Humidity and Heights', 'Relative Humidity (%)', levels=np.arange(10, 101, 5))
        if buf:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f'rh700_{run_date.strftime("%Y%m%d_%H%MZ")}.png'), 'wb') as f: f.write(buf.getbuffer())

def run_fronto700():
    ds, run_date = get_rap_data_for_level(70000)
    if ds:
        buf = generate_upper_air_map(ds, run_date, 70000, 'frontogenesis', 'RdBu_r', 'RAP 13km: 700-hPa Frontogenesis and Heights', 'Frontogenesis (K/100km/3hr)', levels=np.linspace(-10, 10, 41))
        if buf:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f'fronto700_{run_date.strftime("%Y%m%d_%H%MZ")}.png'), 'wb') as f: f.write(buf.getbuffer())

def run_wind850():
    ds, run_date = get_rap_data_for_level(85000)
    if ds:
        buf = generate_upper_air_map(ds, run_date, 85000, 'wind_speed', 'YlOrBr', 'RAP 13km: 850-hPa Wind Speeds and Heights', 'Wind Speed (knots)', levels=np.arange(20, 141, 10))
        if buf:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f'wind850_{run_date.strftime("%Y%m%d_%H%MZ")}.png'), 'wb') as f: f.write(buf.getbuffer())

def run_dew850():
    ds, run_date = get_rap_data_for_level(85000)
    if ds:
        buf = generate_upper_air_map(ds, run_date, 85000, 'dewpoint', 'BuGn', 'RAP 13km: 850-hPa Dewpoint and Heights', 'Dewpoint (°C)', levels=np.arange(-40, 31, 2))
        if buf:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f'dew850_{run_date.strftime("%Y%m%d_%H%MZ")}.png'), 'wb') as f: f.write(buf.getbuffer())

def run_mAdv850():
    ds, run_date = get_rap_data_for_level(85000)
    if ds:
        buf = generate_upper_air_map(ds, run_date, 85000, 'moisture_advection', 'PRGn', 'RAP 13km: 850-hPa Moisture Advection and Heights', 'Moisture Advection (%/hour)', levels=np.linspace(-20, 20, 41))
        if buf:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f'mAdv850_{run_date.strftime("%Y%m%d_%H%MZ")}.png'), 'wb') as f: f.write(buf.getbuffer())

def run_tAdv850():
    ds, run_date = get_rap_data_for_level(85000)
    if ds:
        buf = generate_upper_air_map(ds, run_date, 85000, 'temp_advection', 'coolwarm', 'RAP 13km: 850-hPa Temperature Advection and Heights', 'Temperature Advection (K/hour)', levels=np.linspace(-20, 20, 41))
        if buf:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f'tAdv850_{run_date.strftime("%Y%m%d_%H%MZ")}.png'), 'wb') as f: f.write(buf.getbuffer())

def run_divcon300():
    ds, run_date = get_rap_data_for_level(30000)
    if ds:
        buf = generate_upper_air_map(ds, run_date, 30000, 'divergence', 'RdBu_r', 'RAP 13km: 300-hPa Divergence and Heights', 'Divergence ($10^{-5}$ s$^{-1}$)', levels=np.linspace(-15, 15, 31))
        if buf:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f'divcon300_{run_date.strftime("%Y%m%d_%H%MZ")}.png'), 'wb') as f: f.write(buf.getbuffer())

def run_mslp():
    generate_mslp_temp_map()

# ====================================================================
# SCRIPT ENTRY POINT
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Frostbyte CONUS Map Generator")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Map type target')
    
    subparsers.add_parser('mslp', help='MSLP with clean 2mb surface temperature configuration')
    subparsers.add_parser('tempgrad', help='MSLP with surface temperature gradient')
    subparsers.add_parser('wind300', help='300 hPa Wind Speed')
    subparsers.add_parser('wind500', help='500 hPa Wind Speed')
    subparsers.add_parser('vort500', help='500 hPa Relative Vorticity')
    subparsers.add_parser('rh700', help='700 hPa Relative Humidity')
    subparsers.add_parser('fronto700', help='700 hPa Frontogenesis')
    subparsers.add_parser('wind850', help='850 hPa Wind Speed')
    subparsers.add_parser('dew850', help='850 hPa Dewpoint')
    subparsers.add_parser('mAdv850', help='850 hPa Moisture Advection')
    subparsers.add_parser('tAdv850', help='850 hPa Temperature Advection')
    subparsers.add_parser('divcon300', help='300 hPa Divergence/Convergence')
    
    if len(sys.argv) == 1:
        args = parser.parse_args(['mslp'])
    else:
        args = parser.parse_args()

    command_functions = {
        'wind300': run_wind300,
        'wind500': run_wind500,
        'vort500': run_vort500,
        'rh700': run_rh700,
        'fronto700': run_fronto700,
        'wind850': run_wind850,
        'dew850': run_dew850,
        'mAdv850': run_mAdv850,
        'tAdv850': run_tAdv850,
        'mslp': run_mslp,
        'tempgrad': run_mslp,
        'divcon300': run_divcon300,
    }

    if args.command in command_functions:
        command_functions[args.command]()
    else:
        logging.error(f"Command '{args.command}' is invalid.")

if __name__ == "__main__":
    main()
