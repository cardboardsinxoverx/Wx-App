#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced CONUS Weather Map Generator
Surface Maps: Blended HRRR/RAP/RTMA with explicit high-terrain hypsometric reduction.
Upper Air Maps: Blended RAP (13km) + GFS (0.25deg) Isobaric Data.
Includes universal 4 dam upper-air intervals, dynamic colorbar locators, optimized wind barbs,
and tiered surface dew point contours.
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
from matplotlib.colors import LinearSegmentedColormap
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
# 3. Model Fetchers (Surface & Upper-Air)
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
        dp_var = find_var(['Dewpoint_temperature_height_above_ground', 'Dewpoint_temperature_surface', 'dpt2m'])
        u_var = find_var(['u-component_of_wind_height_above_ground', 'u-component_of_wind_surface'])
        v_var = find_var(['v-component_of_wind_height_above_ground', 'v-component_of_wind_surface'])
        sfc_press_var = find_var(['MSLP_MAPS_System_Reduction_msl', 'Pressure_reduced_to_MSL_msl'])
        orog_var = find_var(['Geopotential_height_surface', 'orography', 'HGT_surface'])

        query = ncss.query()
        query.lonlat_box(north=bbox[1], south=bbox[0], east=bbox[3], west=bbox[2])
        query.accept('netcdf4')
        query.add_lonlat(True)
        
        vars_to_fetch = [v for v in [temp_var, dp_var, u_var, v_var, sfc_press_var, orog_var] if v]
        query.variables(*vars_to_fetch)
        data = ncss.get_data(query)
        
        lat_var = next((v for v in data.variables if 'lat' in v.lower()), None)
        lon_var = next((v for v in data.variables if 'lon' in v.lower()), None)
        
        def get_val(v_name):
            val = data.variables[v_name][:].squeeze()
            if val.ndim == 3: val = val[0]
            return val

        t_k = get_val(temp_var)
        dp_k = get_val(dp_var) if dp_var else np.full_like(t_k, np.nan)
        p_sfc = get_val(sfc_press_var)
        elev = get_val(orog_var) if orog_var else np.zeros_like(t_k)
        
        calculated_mslp = (p_sfc / 100.0) if sfc_press_var else np.zeros_like(t_k)

        return {
            'lat': data.variables[lat_var][:], 'lon': data.variables[lon_var][:],
            'temp': (t_k - 273.15) * 9/5 + 32,
            'dewpoint': (dp_k - 273.15) * 9/5 + 32,
            'u': get_val(u_var) * 1.94384, 'v': get_val(v_var) * 1.94384,
            'pressure': calculated_mslp
        }
    except Exception as e:
        logging.error(f"RAP Surface Fetch Failed: {e}")
        return None

def fetch_goes_ch9():
    """Fetches real GOES-16 Ch 9 Water Vapor for the background overlay."""
    logging.info("Fetching real GOES-16 Ch 9 Water Vapor for background overlay...")
    try:
        import metpy
        cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/satellite/goes/east/products/CloudAndMoistureImagery/CONUS/Channel09/current/catalog.xml')
        datasets = sorted(list(cat.datasets.values()), key=lambda d: d.name)
        latest_url = datasets[-1].access_urls['OPeNDAP']
        
        ds = xr.open_dataset(latest_url)
        ds = ds.metpy.parse_cf()
        var = ds['Sectorized_CMI'] if 'Sectorized_CMI' in ds else ds['CMI']
        logging.info("✅ GOES-16 Data Fetched Successfully.")
        return ds, var
    except ImportError:
        logging.error("MetPy is required for real GOES overlays. Run: pip install metpy")
        return None, None
    except Exception as e:
        logging.error(f"Failed to fetch GOES data: {e}")
        return None, None
    
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
        dp_var = find(['Dewpoint_temperature_height_above_ground', 'Dewpoint_temperature_surface'])
        u_var = find(['u-component_of_wind_height_above_ground', 'u-component_of_wind_surface'])
        v_var = find(['v-component_of_wind_height_above_ground', 'v-component_of_wind_surface'])
        sfc_p_var = find(['MSLP_MAPS_System_Reduction_msl', 'Pressure_reduced_to_MSL_msl'])
        orog_var = find(['Geopotential_height_surface', 'HGT_surface'])

        query = ncss.query()
        query.lonlat_box(north=bbox[1], south=bbox[0], east=bbox[3], west=bbox[2])
        query.accept('netcdf4')
        query.add_lonlat(True)
        
        valid_vars = [v for v in [t_var, dp_var, u_var, v_var, sfc_p_var, orog_var] if v]
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
        dp_k = get(dp_var) if dp_var else np.full_like(t_k, np.nan)
        p_sfc = get(sfc_p_var)
        elev = get(orog_var) if orog_var else np.zeros_like(t_k)
        
        calculated_mslp = p_sfc / 100.0

        return {
            'lat': data.variables[lat_var][:], 'lon': data.variables[lon_var][:],
            'temp': (t_k - 273.15) * 9/5 + 32,
            'dewpoint': (dp_k - 273.15) * 9/5 + 32,
            'u': get(u_var) * 1.94384, 'v': get(v_var) * 1.94384,
            'pressure': calculated_mslp
        }
    except Exception as e:
        logging.error(f"HRRR Surface Fetch Failed: {e}")
        return None

def fetch_rtma_sfc(extent):
    logging.info("Fetching RTMA 2.5km Surface Analysis...")
    try:
        buffer = 8.0
        bbox = [extent[2]-buffer, extent[3]+buffer, extent[0]-buffer, extent[1]+buffer]
        
        cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RTMA/CONUS_2p5km/latest.xml')
        ds = cat.datasets[0] 
        ncss = ds.subset()
        available = set(ncss.variables)
        
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
        dp_var = get_best_match(['Dewpoint_temperature_height_above_ground', 'Dewpoint_temperature_surface', 'dpt2m'], 'Dewpoint_temperature_height_above_ground')
        u_var = get_best_match(['u-component_of_wind_height_above_ground', 'u-component_of_wind_surface'], 'u-component_of_wind_height_above_ground')
        v_var = get_best_match(['v-component_of_wind_height_above_ground', 'v-component_of_wind_surface'], 'v-component_of_wind_height_above_ground')
        sfc_p_var = get_best_match(['MSLP_MAPS_System_Reduction_msl', 'Pressure_reduced_to_MSL_msl'], None)
        orog_var = get_best_match(['Geopotential_height_surface', 'HGT_surface'], 'Geopotential_height_surface')

        query = ncss.query()
        query.lonlat_box(north=bbox[1], south=bbox[0], east=bbox[3], west=bbox[2])
        query.accept('netcdf4')
        query.add_lonlat(True)
        
        valid_vars = [v for v in [t_var, dp_var, u_var, v_var, sfc_p_var, orog_var] if v]
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
        dp_k = get(dp_var) if dp_var else np.full_like(t_k, np.nan)
        p_sfc = get(sfc_p_var)
        elev = get(orog_var) if orog_var else np.zeros_like(t_k)
        
        calculated_mslp = p_sfc / 100.0

        return {
            'lat': data.variables[lat_var][:], 'lon': data.variables[lon_var][:],
            'temp': (t_k - 273.15) * 9/5 + 32,
            'dewpoint': (dp_k - 273.15) * 9/5 + 32,
            'u': get(u_var) * 1.94384, 'v': get(v_var) * 1.94384,
            'pressure': calculated_mslp
        }
    except Exception as e:
        logging.error(f"RTMA Fetch Failed: {e}")
        return None

def fetch_upper_air_model(level):
    logging.info(f"Fetching RAP isobaric data for {level} Pa...")
    try:
        cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RAP/CONUS_13km/latest.xml')
        ds = cat.datasets[0].subset()
        query = ds.query()
        query.lonlat_box(north=55, south=15, east=-60, west=-130)
        query.accept('netcdf4')
        query.add_lonlat(True)
        query.variables('Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 
                        'v-component_of_wind_isobaric', 'Temperature_isobaric', 'Relative_humidity_isobaric')
        query.vertical_level([level])
        
        data = ds.get_data(query)
        
        def get_var(name, fallbacks=None):
            if fallbacks is None:
                fallbacks = []
            for n in [name] + fallbacks:
                try:
                    var = data.variables[n][:].squeeze()
                    if var.ndim == 3:
                        var = var[0]
                    return var
                except:
                    continue
            return None

        result = {
            'lat': get_var('lat', ['latitude']),
            'lon': get_var('lon', ['longitude']),
            'hgt': get_var('Geopotential_height_isobaric'),
            'u': get_var('u-component_of_wind_isobaric'),
            'v': get_var('v-component_of_wind_isobaric'),
            'temp': get_var('Temperature_isobaric'),
            'rh': get_var('Relative_humidity_isobaric')
        }
        
        logging.info(f"✅ Successfully fetched RAP at {level} Pa")
        return result

    except Exception as e:
        logging.error(f"RAP fetch failed at {level} Pa: {e}")
        return None

# ====================================================================
# 4. BLENDERS
# ====================================================================

def blend_surface_models(rap_data, hrrr_data, rtma_data, extent):
    from scipy.spatial import cKDTree
    logging.info("Blending Unified Surface Models (KDTree Spatial Masking)...")
    
    grid_lon = np.arange(extent[0]-0.5, extent[1]+0.5, 0.1)
    grid_lat = np.arange(extent[2]-0.5, extent[3]+0.5, 0.1)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
    
    # Flattened grid coordinates for KDTree spatial querying
    grid_points_flat = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    pts_t, vals_t = [], []
    pts_dp, vals_dp = [], []
    pts_wind, vals_u, vals_v = [], [], []
    pts_p, vals_p = [], []
    
    def add_to_blend(data_dict, step):
        if not data_dict: return
        
        lon = data_dict['lon'][::step, ::step].flatten()
        lat = data_dict['lat'][::step, ::step].flatten()
        lon = np.where(lon > 180, lon - 360, lon)
        
        def extract(var_name, vmin, vmax):
            if data_dict.get(var_name) is None: return None, None
            arr = np.ma.filled(data_dict[var_name][::step, ::step].flatten(), np.nan)
            valid = (~np.isnan(arr)) & (arr > vmin) & (arr < vmax)
            if not np.any(valid): return None, None
            return np.column_stack((lon[valid], lat[valid])), arr[valid]

        p_t, v_t = extract('temp', -100, 150)
        if p_t is not None:
            pts_t.append(p_t)
            vals_t.append(v_t)
            
        p_dp, v_dp = extract('dewpoint', -100, 150)
        if p_dp is not None:
            pts_dp.append(p_dp)
            vals_dp.append(v_dp)
            
        p_p, v_p = extract('pressure', 800, 1150)
        if p_p is not None:
            pts_p.append(p_p)
            vals_p.append(v_p)
            
        if data_dict.get('u') is not None and data_dict.get('v') is not None:
            u_arr = np.ma.filled(data_dict['u'][::step, ::step].flatten(), np.nan)
            v_arr = np.ma.filled(data_dict['v'][::step, ::step].flatten(), np.nan)
            valid_w = (~np.isnan(u_arr)) & (np.abs(u_arr) < 200) & (~np.isnan(v_arr)) & (np.abs(v_arr) < 200)
            if np.any(valid_w):
                pts_wind.append(np.column_stack((lon[valid_w], lat[valid_w])))
                vals_u.append(u_arr[valid_w])
                vals_v.append(v_arr[valid_w])

    add_to_blend(rap_data, 1)
    add_to_blend(hrrr_data, 3)
    add_to_blend(rtma_data, 3)

    def interp_var(pts_list, vals_list, sigma=1.0, mask_extrapolated=True):
        if not pts_list: return None
        pts = np.vstack(pts_list)
        vals = np.concatenate(vals_list)
        
        # 1. Standard linear interpolation (creates the bad convex hull bridge)
        res = griddata(pts, vals, (grid_x, grid_y), method='linear')
        
        if mask_extrapolated:
            # 2. Build a spatial tree of the actual valid data points
            tree = cKDTree(pts)
            # 3. Query the distance from every map pixel to the nearest real data point
            dists, _ = tree.query(grid_points_flat)
            dists = dists.reshape(grid_x.shape)
            
            # 4. If a pixel is more than 0.3 degrees (~30km) from a real point, kill it.
            # This instantly severs the convex hull bridge across the Gulf/Mexico.
            res[dists > 0.3] = np.nan
            
            mask_nan = np.isnan(res)
            if np.any(mask_nan):
                # Safe blur technique so the edges don't shrink
                nearest_fill = griddata(pts, vals, (grid_x[mask_nan], grid_y[mask_nan]), method='nearest')
                temp_fill = np.copy(res)
                temp_fill[mask_nan] = nearest_fill
                
                smoothed = gaussian_filter(temp_fill, sigma=sigma)
                smoothed[mask_nan] = np.nan # Re-apply the strict KDTree mask
                return smoothed
            return gaussian_filter(res, sigma=sigma)
        else:
            # Temperature bypasses the KDTree so it seamlessly fills the map background
            mask_nan = np.isnan(res)
            if np.any(mask_nan):
                nearest_fill = griddata(pts, vals, (grid_x[mask_nan], grid_y[mask_nan]), method='nearest')
                res[mask_nan] = nearest_fill
            return gaussian_filter(res, sigma=sigma)

    try:
        g_temp = interp_var(pts_t, vals_t, sigma=1.0, mask_extrapolated=False)
        g_dp = interp_var(pts_dp, vals_dp, sigma=1.0, mask_extrapolated=True)
        g_p = interp_var(pts_p, vals_p, sigma=4.0, mask_extrapolated=True)
        
        g_u, g_v = None, None
        if pts_wind:
            g_u = interp_var(pts_wind, vals_u, sigma=1.0, mask_extrapolated=True)
            g_v = interp_var(pts_wind, vals_v, sigma=1.0, mask_extrapolated=True)
            
        return grid_y, grid_x, g_temp, g_dp, g_u, g_v, g_p
    except Exception as e:
        logging.error(f"Blending failed: {e}")
        return None, None, None, None, None, None, None

def blend_upper_air_models(level, extent):
    """RAP only."""
    logging.info("Fetching and processing RAP Upper-Air data...")
    rap = fetch_upper_air_model(level)
    
    if not rap or rap.get('hgt') is None:
        logging.error("No valid RAP upper-air data available")
        return None, None, None, None, None, None, None
    
    grid_lon = np.arange(extent[0]-0.5, extent[1]+0.5, 0.1)
    grid_lat = np.arange(extent[2]-0.5, extent[3]+0.5, 0.1)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
    
    try:
        lon = np.where(rap['lon'] > 180, rap['lon'] - 360, rap['lon']).flatten()
        lat = rap['lat'].flatten()
        
        valid = (~np.isnan(rap['hgt'].flatten())) & (~np.isnan(rap['temp'].flatten()))
        
        if not np.any(valid):
            logging.error("No valid points in RAP data")
            return None, None, None, None, None, None, None
        
        pts = np.column_stack((lon[valid], lat[valid]))
        hgts = rap['hgt'].flatten()[valid]
        us = rap['u'].flatten()[valid]
        vs = rap['v'].flatten()[valid]
        temps = rap['temp'].flatten()[valid]
        rhs = rap['rh'].flatten()[valid] if rap.get('rh') is not None else np.zeros_like(hgts)
        
        def interp(vals, sigma=1.5):
            res = griddata(pts, vals, (grid_x, grid_y), method='linear')
            mask = np.isnan(res)
            if np.any(mask):
                res[mask] = griddata(pts, vals, (grid_x[mask], grid_y[mask]), method='nearest')
            return gaussian_filter(res, sigma=sigma)
        
        logging.info("RAP upper-air data processed successfully")
        return grid_x, grid_y, interp(hgts, 2.0), interp(us), interp(vs), interp(temps), interp(rhs)
        
    except Exception as e:
        logging.error(f"RAP Upper-Air processing failed: {e}")
        return None, None, None, None, None, None, None
    
# ==========================================
# --- NAVY / MARINE MODULE ---
# ==========================================

def calc_marine_physics(water_temp_k, water_u, water_v, air_temp_f, air_dew_f, wind_speed_kts):
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
    import netCDF4 as nc
    import time
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
        session.headers.update({'User-Agent': 'Mozilla/5.0 Frostbyte/1.0'})

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

# ====================================================================
# 5. OVERLAYS & COLORBARS
# ====================================================================

def draw_interstates_from_shapefile(ax, shapefile=INTERSTATES_SHP):
    if not os.path.exists(shapefile): 
        logging.error(f"Shapefile not found: {shapefile}")
        return
    try:
        reader = shpreader.Reader(shapefile)
        road_geoms = list(reader.geometries())
        
        if road_geoms:
            ax.add_geometries(road_geoms, ccrs.PlateCarree(), edgecolor='white', 
                              facecolor='none', linewidth=1.8, zorder=10, alpha=0.85)
            ax.add_geometries(road_geoms, ccrs.PlateCarree(), edgecolor='black', 
                              facecolor='none', linewidth=0.7, zorder=11, alpha=0.95)
    except Exception as e: 
        logging.error(f"Failed to draw interstates: {e}")

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
    smoothed_p = gaussian_filter(grid_p, sigma=3.0)
    
    cs = ax.contour(grid_x, grid_y, smoothed_p, levels=levels, 
                    colors='white', linewidths=1.8, zorder=2.0, transform=ccrs.PlateCarree())
    cs.set_path_effects([pe.Stroke(linewidth=3.5, foreground='black'), pe.Normal()])
    
    clabels = ax.clabel(cs, fmt=fmt, inline=True, inline_spacing=12, fontsize=10, colors='white')
    for label in clabels:
        label.set_path_effects([pe.withStroke(linewidth=3, foreground='black'), pe.Normal()])
        label.set_fontweight('bold')
    return cs

def draw_isotherms(ax, grid_x, grid_y, temp_c, levels_interval=2):
    levels_cold = np.arange(-90, 1, levels_interval)
    levels_warm = np.arange(levels_interval, 60, levels_interval)
    
    cs_cold = ax.contour(grid_x, grid_y, temp_c, levels=levels_cold,
                         colors='#00BFFF', linestyles='--', linewidths=1.5,
                         zorder=2.2, transform=ccrs.PlateCarree())
    cs_cold.set_path_effects([pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
    
    clabels_cold = ax.clabel(cs_cold, fmt='%d°C', inline=True, inline_spacing=8, fontsize=10, colors='#00BFFF')
    for label in clabels_cold:
        label.set_path_effects([pe.withStroke(linewidth=3, foreground='black'), pe.Normal()])
        label.set_fontweight('bold')

    cs_warm = ax.contour(grid_x, grid_y, temp_c, levels=levels_warm,
                         colors='#FF4500', linestyles='--', linewidths=1.5,
                         zorder=2.2, transform=ccrs.PlateCarree())
    cs_warm.set_path_effects([pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
    
    clabels_warm = ax.clabel(cs_warm, fmt='%d°C', inline=True, inline_spacing=8, fontsize=10, colors='#FF4500')
    for label in clabels_warm:
        label.set_path_effects([pe.withStroke(linewidth=3, foreground='black'), pe.Normal()])
        label.set_fontweight('bold')

def draw_isodrosotherms(ax, grid_x, grid_y, dp_f, interval=5):
    """Draws tiered, dashed dew point contours (isodrosotherms)."""
    smoothed_dp = gaussian_filter(dp_f, sigma=2.0)
    
    # Split levels into the requested tiers
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

    # Execute plotting for each tier with its corresponding color
    plot_tier(levels_blue, '#87CEFA')   # Light Blue
    plot_tier(levels_green, '#32CD32')  # Lime Green
    plot_tier(levels_orange, '#FFA500') # Orange

def add_clean_colorbar(fig, ax, cf, label):
    if cf is not None:
        # Dynamically grab the map's exact position on the figure
        pos = ax.get_position()
        
        # Shave off 0.4% from the edges so the triangular extensions don't stretch past the map
        cb_x0 = pos.x0 + 0.004
        cb_width = pos.width - 0.008
        
        # Bumped the Y-position up from 0.08 to 0.095 to clear the legend box
        cax = fig.add_axes([cb_x0, 0.095, cb_width, 0.025])
        cb = fig.colorbar(cf, cax=cax, orientation='horizontal', extend='both', extendfrac=0.04)
        
        cb.set_label(label, fontsize=13, fontweight='bold', color='white', labelpad=10)
        cb.ax.xaxis.label.set_path_effects(TEXT_OUTLINE)
        
        if 'Temperature' in label or 'Dewpoint' in label: ticks = np.arange(-40, 131, 10)
        elif 'Relative Humidity' in label: ticks = np.arange(10, 101, 10)
        elif 'Wind Speed' in label: ticks = np.arange(20, 201, 20)
        elif 'Advection' in label or 'Vorticity' in label or 'Divergence' in label: ticks = np.arange(-20, 21, 5)
        else: ticks = cb.get_ticks()
        
        vmin, vmax = cf.get_clim()
        valid_ticks = [t for t in ticks if vmin <= t <= vmax]
        
        cb.set_ticks(valid_ticks)
        cb.ax.tick_params(axis='x', length=0, colors='white')
        
        cb.ax.set_xticklabels([str(int(t)) for t in valid_ticks], fontsize=11, fontweight='bold', color='white')
        
        for tick_label in cb.ax.get_xticklabels():
            tick_label.set_path_effects(TEXT_OUTLINE)

# ====================================================================
# 6. CORE GENERATION PLOTS 
# ====================================================================

def generate_mslp_temp_map():
    import matplotlib.lines as mlines
    logging.info("Starting Clean 2mb Surface Analysis Generator...")
    run_date = datetime.now(timezone.utc)
    
    rap_data = fetch_rap_full(EXTENT)
    hrrr_data = fetch_hrrr_sfc(EXTENT)
    rtma_data = fetch_rtma_sfc(EXTENT)
    ncom_lat, ncom_lon, ncom_temp_k, ncom_u, ncom_v, ncom_sal = fetch_ocean_currents(EXTENT)
    
    grid_y, grid_x, g_temp, g_dp, g_u, g_v, g_p = blend_surface_models(rap_data, hrrr_data, rtma_data, EXTENT)

    fig = plt.figure(figsize=(18, 11), facecolor=FIG_BG_COLOR)
    ax = fig.add_axes([0.08, 0.16, 0.86, 0.75], projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1.4, zorder=3)
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linestyle=':', linewidth=1.4, zorder=3)
    ax.add_feature(cfeature.STATES, edgecolor='black', linestyle=':', linewidth=0.9, zorder=3)
    
    gl = ax.gridlines(draw_labels=True, linewidth=1.0, color=GRIDLINE_COLOR, alpha=0.4, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = LABEL_STYLE; gl.ylabel_style = LABEL_STYLE

    cf_temp = None
    if g_temp is not None:
        temp_cmap = get_ga_temp_cmap()
        temp_levels = np.arange(-40, 121, 2.0)
        cf_temp = ax.contourf(grid_x, grid_y, g_temp, levels=temp_levels, cmap=temp_cmap, 
                              norm=mcolors.Normalize(vmin=-40.0, vmax=120.0), 
                              transform=ccrs.PlateCarree(), extend='both', alpha=0.85, zorder=1.5)

    if (ncom_lat is not None and ncom_u is not None and np.any(np.isfinite(ncom_u))):
        print("• Plotting Marine/Ocean Layer...", end=" ", flush=True)
        try:
            nc_x, nc_y = ncom_lon, ncom_lat
            ncom_temp_c = ncom_temp_k - 273.15
            stride = 6
            strm = ax.streamplot(nc_x[::stride, ::stride], nc_y[::stride, ::stride], 
                                 ncom_u[::stride, ::stride], ncom_v[::stride, ::stride], 
                                 color=ncom_temp_c[::stride, ::stride], cmap='jet',
                                 density=1.2, linewidth=1.5, arrowsize=1.5,
                                 transform=ccrs.PlateCarree(), zorder=2.0)
            strm.lines.set_path_effects([pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
            
            cax_marine = fig.add_axes([0.03, 0.164, 0.008, 0.742]) 
            cbar = fig.colorbar(strm.lines, cax=cax_marine, extend='both')
            cbar.ax.yaxis.set_label_position('left')
            cbar.ax.yaxis.set_ticks_position('left')
            
            cbar.set_label('Ocean Temp (°C)', fontsize=9, fontweight='bold', color='white', labelpad=2)
            cbar.ax.yaxis.label.set_path_effects(TEXT_OUTLINE)
            
            cbar.ax.tick_params(axis='y', colors='white', pad=2, labelsize=8) 
            for tick_label in cbar.ax.get_yticklabels():
                tick_label.set_path_effects(TEXT_OUTLINE)
                tick_label.set_fontweight('bold')
            print("Done. ✅")
        except Exception as e:
            logging.error(f"Marine Plotting Error: {e}")

    if g_p is not None:
        draw_isobars(ax, grid_x, grid_y, g_p, np.arange(950, 1050, 2), fmt='%d mb')
        
    if g_dp is not None:
        draw_isodrosotherms(ax, grid_x, grid_y, g_dp, interval=5)
        
    if g_u is not None and g_v is not None:
        skip = 13
        u_kts, v_kts = g_u[::skip, ::skip], g_v[::skip, ::skip]
        ax.barbs(grid_x[::skip, ::skip], grid_y[::skip, ::skip], u_kts, v_kts, 
                 transform=ccrs.PlateCarree(), length=5.5, color='black', zorder=2.5)

    draw_interstates_from_shapefile(ax)
    draw_major_cities(ax)

    # --- CUSTOM LEGEND (Compacted and Nudged) ---
    isobar_line = mlines.Line2D([], [], color='white', linewidth=1.8, label='MSLP (mb)', 
                                path_effects=[pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
    dp_blue = mlines.Line2D([], [], color='#87CEFA', linestyle='--', linewidth=1.5, label='Td < 65°F',
                            path_effects=[pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
    dp_green = mlines.Line2D([], [], color='#32CD32', linestyle='--', linewidth=1.5, label='Td 65-74°F',
                             path_effects=[pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
    dp_orange = mlines.Line2D([], [], color='#FFA500', linestyle='--', linewidth=1.5, label='Td ≥ 75°F',
                              path_effects=[pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()])
    
    # Tightened box dimensions and shifted anchor to float cleanly
    leg = fig.legend(handles=[isobar_line, dp_blue, dp_green, dp_orange], 
                     loc='lower left', bbox_to_anchor=(0.01, 0.015),
                     facecolor=LAND_COLOR, edgecolor='black', labelcolor='white',
                     fontsize=10, framealpha=1.0, borderpad=0.6,
                     handlelength=1.6, handletextpad=0.6)
    leg.get_frame().set_linewidth(1.5)

    ax.text(0.0, 1.02, "CONUS Surface Analysis & High-Terrain Adjusted MSLP Gradient", 
            fontsize=20, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, 
            transform=ax.transAxes, ha='left', va='bottom')
    ax.text(1.0, 1.02, run_date.strftime('%d %B %Y %H:%MZ'), 
            fontsize=16, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, 
            transform=ax.transAxes, ha='right', va='bottom')

    add_clean_colorbar(fig, ax, cf_temp, 'Surface Temperature (°F)')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"mslp_temp_{run_date.strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    logging.info(f"✅ Adjusted High-Terrain Map Saved: {out_path}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf, run_date
def generate_upper_air_map(grid_x, grid_y, hgt, u, v, temp, rh, run_date, level, variable, cmap, title, cb_label, levels=None, wv_background=False, plot_vort_max=False, sat_ds=None, sat_var=None):
    try:
        heights_dam = hgt / 10.0
        
        if variable == 'wind_speed': data = compute_wind_speed(u, v)
        elif variable == 'vorticity': data = compute_vorticity(u, v, grid_y, grid_x)
        elif variable == 'relative_humidity': data = rh
        elif variable == 'temp_advection': data = compute_advection(temp, u, v, grid_y, grid_x) * 3600
        elif variable == 'moisture_advection': data = compute_advection(rh, u, v, grid_y, grid_x) * 1e4
        elif variable == 'dewpoint': data = compute_dewpoint(temp, rh)
        elif variable == 'divergence': data = compute_divergence(u, v, grid_y, grid_x)
        elif variable == 'frontogenesis': data = frontogenesis_700hPa(temp - 273.15, u, v, grid_y, grid_x)
        else: data = np.zeros_like(grid_x)

        fig = plt.figure(figsize=(18, 11), facecolor=FIG_BG_COLOR)
        ax = fig.add_axes([0.06, 0.16, 0.88, 0.75], projection=ccrs.PlateCarree())
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1.4, zorder=3)
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linestyle=':', linewidth=1.4, zorder=3)
        ax.add_feature(cfeature.STATES, edgecolor='black', linestyle=':', linewidth=0.9, zorder=3)
        
        gl = ax.gridlines(draw_labels=True, linewidth=1.0, color=GRIDLINE_COLOR, alpha=0.4, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = LABEL_STYLE; gl.ylabel_style = LABEL_STYLE

        draw_interstates_from_shapefile(ax)
        draw_major_cities(ax)

        if wv_background and sat_ds is not None and sat_var is not None:
            logging.info("Plotting true GOES-16 satellite overlay...")
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
            C_RED = (1.0, 0.0, 0.0); C_YELLOW = (1.0, 1.0, 0.0); C_BLUE = (0.0, 0.0, 1.0)
            C_WHITE = (1.0, 1.0, 1.0); C_GREEN = (0.0, 0.5, 0.0); C_TEAL = (0.0, 1.0, 1.0); C_LB = (0.5, 0.5, 1.0)

            span = wv_levels.max() - wv_levels.min()
            nodes = np.flip((wv_levels - wv_levels.min()) / span)
            node_colors = [C_RED, C_YELLOW, C_BLUE, C_WHITE, C_GREEN, C_TEAL, C_LB][::-1]
            nodes[0] = 0.0; nodes[-1] = 1.0
            wv_cmap = LinearSegmentedColormap.from_list('custom_wv_cmap', list(zip(nodes, node_colors)))
            norm = mcolors.Normalize(vmin=-109, vmax=0)
            
            main_cf = ax.pcolormesh(x, y, sat_data, cmap=wv_cmap, norm=norm, 
                                    transform=sat_proj, shading='auto', zorder=1.5, alpha=0.90)

            cax = fig.add_axes([0.06, 0.08, 0.88, 0.025])
            cbar = fig.colorbar(main_cf, cax=cax, orientation='horizontal', extend='both')
            cbar.set_label('Temp (°C)', fontsize=13, fontweight='bold', color='white', labelpad=10)
            cbar.ax.xaxis.label.set_path_effects(TEXT_OUTLINE)
            cb_ticks = np.sort(wv_levels)
            cbar.set_ticks(cb_ticks)
            cbar.ax.set_xticklabels([f"{t:g}" for t in cb_ticks], fontsize=11, fontweight='bold', color='white')
            for tick in cbar.ax.get_xticklabels(): tick.set_path_effects(TEXT_OUTLINE)
            
        elif wv_background and rh is not None:
            logging.warning("Satellite overlay failed. Falling back to RH simulation...")
            rh_points = [0, 15, 30, 50, 70, 85, 100]
            temp_points = [0, -15.5, -30, -47, -75, -100, -109]
            sim_wv_temp = np.interp(np.clip(rh, 0, 100), rh_points, temp_points)

            wv_levels = np.array([0, -15.5, -30, -47, -75, -100, -109])
            C_RED = (1.0, 0.0, 0.0); C_YELLOW = (1.0, 1.0, 0.0); C_BLUE = (0.0, 0.0, 1.0)
            C_WHITE = (1.0, 1.0, 1.0); C_GREEN = (0.0, 0.5, 0.0); C_TEAL = (0.0, 1.0, 1.0); C_LB = (0.5, 0.5, 1.0)
            span = wv_levels.max() - wv_levels.min()
            nodes = np.flip((wv_levels - wv_levels.min()) / span)
            node_colors = [C_RED, C_YELLOW, C_BLUE, C_WHITE, C_GREEN, C_TEAL, C_LB][::-1]
            nodes[0] = 0.0; nodes[-1] = 1.0
            wv_cmap = LinearSegmentedColormap.from_list('custom_wv_cmap', list(zip(nodes, node_colors)))
            norm = mcolors.Normalize(vmin=-109, vmax=0)
            
            main_cf = ax.pcolormesh(grid_x, grid_y, sim_wv_temp, cmap=wv_cmap, norm=norm, alpha=0.85, transform=ccrs.PlateCarree(), shading='auto', zorder=1.5)

            cax = fig.add_axes([0.06, 0.08, 0.88, 0.025])
            cbar = fig.colorbar(main_cf, cax=cax, orientation='horizontal', extend='both')
            cbar.set_label('Temp (°C)', fontsize=13, fontweight='bold', color='white', labelpad=10)
            cbar.ax.xaxis.label.set_path_effects(TEXT_OUTLINE)
            cb_ticks = np.sort(wv_levels)
            cbar.set_ticks(cb_ticks)
            cbar.ax.set_xticklabels([f"{t:g}" for t in cb_ticks], fontsize=11, fontweight='bold', color='white')
            for tick in cbar.ax.get_xticklabels(): tick.set_path_effects(TEXT_OUTLINE)
        else:
            if levels is None:
                cf = ax.contourf(grid_x, grid_y, data, cmap=cmap, transform=ccrs.PlateCarree(), extend='both', alpha=0.85)
            else:
                cf = ax.contourf(grid_x, grid_y, data, cmap=cmap, transform=ccrs.PlateCarree(), levels=levels, extend='both', alpha=0.85)
            main_cf = cf
            add_clean_colorbar(fig, ax, main_cf, cb_label)

        if level == 50000: h_levels = np.arange(480, 620, 6)
        elif level == 30000: h_levels = np.arange(800, 1040, 4) 
        elif level == 70000: h_levels = np.arange(210, 350, 4)
        elif level == 85000: h_levels = np.arange(90, 230, 4) 
        else: h_levels = None

        if h_levels is not None:
            draw_isobars(ax, grid_x, grid_y, heights_dam, h_levels, fmt='%i dam')

        if level in [85000, 70000, 50000]:
            temp_c = temp - 273.15
            draw_isotherms(ax, grid_x, grid_y, temp_c, levels_interval=5)

        if plot_vort_max and variable == 'vorticity':
            from scipy.ndimage import maximum_filter
            vort_max = maximum_filter(data, size=15)
            local_max = (data == vort_max) & (data > 10)
            y_idx, x_idx = np.where(local_max)
            
            for i in range(len(y_idx)):
                lon_val = grid_x[y_idx[i], x_idx[i]]
                lat_val = grid_y[y_idx[i], x_idx[i]]
                
                if (EXTENT[0] <= lon_val <= EXTENT[1]) and (EXTENT[2] <= lat_val <= EXTENT[3]):
                    ax.plot(lon_val, lat_val, 'X',
                            color='yellow', markersize=14, markeredgecolor='black', 
                            markeredgewidth=2, zorder=6, transform=ccrs.PlateCarree(), clip_on=True)
                    ax.text(lon_val + 0.4, lat_val + 0.4,
                            f"{data[y_idx[i], x_idx[i]]:.0f}", color='yellow', fontsize=10,
                            fontweight='bold', path_effects=TEXT_OUTLINE, zorder=7, 
                            transform=ccrs.PlateCarree(), clip_on=True)

        u_wind_knots = u * 1.94384
        v_wind_knots = v * 1.94384

        skip = 24 if not plot_vort_max else 16
        ax.barbs(grid_x[::skip, ::skip], grid_y[::skip, ::skip], u_wind_knots[::skip, ::skip], v_wind_knots[::skip, ::skip], 
                 transform=ccrs.PlateCarree(), length=6, color='black', zorder=4)

        ax.text(0.0, 1.02, f"{title}", 
                fontsize=20, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, 
                transform=ax.transAxes, ha='left', va='bottom')
        ax.text(1.0, 1.02, run_date.strftime('%d %B %Y %H:%MZ'), 
                fontsize=16, color='white', fontweight='bold', path_effects=TEXT_OUTLINE, 
                transform=ax.transAxes, ha='right', va='bottom')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        logging.error(f"Upper Air Generation Error: {e}", exc_info=True)
        return None
    
# ====================================================================
# 7. UNIFIED PIPELINE RUNNERS
# ====================================================================

def run_upper_air_pipeline(level, variable, cmap, title, cb_label, levels=None, wv_background=False, plot_vort_max=False):
    """RAP-only pipeline with actual GOES-16 overlay integration."""
    run_date = datetime.now(timezone.utc)
    
    grid_x, grid_y, hgt, u, v, temp, rh = blend_upper_air_models(level, EXTENT)
    
    sat_ds, sat_var = None, None
    if wv_background:
        sat_ds, sat_var = fetch_goes_ch9()
    
    if grid_x is not None:
        buf = generate_upper_air_map(grid_x, grid_y, hgt, u, v, temp, rh, run_date, 
                                     level, variable, cmap, title, cb_label, levels,
                                     wv_background=wv_background, plot_vort_max=plot_vort_max,
                                     sat_ds=sat_ds, sat_var=sat_var)
        if buf:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = f"{variable}_{int(level/100)}mb_{run_date.strftime('%Y%m%d_%H%MZ')}.png"
            with open(os.path.join(OUTPUT_DIR, filename), 'wb') as f: 
                f.write(buf.getbuffer())
            logging.info(f"✅ Generated {filename}")

def run_wind300():
    run_upper_air_pipeline(30000, 'wind_speed', 'cool', 'RAP Analysis: 300-hPa Wind Speeds and Heights', 'Wind Speed (knots)', np.arange(20, 201, 10))

def run_wind500():
    run_upper_air_pipeline(50000, 'wind_speed', 'YlOrBr', 'RAP Analysis: 500-hPa Wind Speeds and Heights', 'Wind Speed (knots)', np.arange(20, 181, 10))

def run_vort500():
    run_upper_air_pipeline(50000, 'vorticity', 'seismic', 'RAP Analysis: 500-hPa Relative Vorticity and Heights', 'Vorticity ($10^{-5}$ s$^{-1}$)', np.linspace(-20, 20, 41), plot_vort_max=True)

def run_500wv():
    run_upper_air_pipeline(50000, 'vorticity', 'RdBu_r', 
                           'RAP 500mb Analysis — Water Vapor / Heights / Vort Max', 
                           'Vorticity (10⁻⁵ s⁻¹)', np.linspace(-25, 25, 51),
                           wv_background=True, plot_vort_max=True)
    
def run_rh700():
    run_upper_air_pipeline(70000, 'relative_humidity', 'BuGn', 'RAP Analysis: 700-hPa Relative Humidity and Heights', 'Relative Humidity (%)', np.arange(10, 101, 5))

def run_fronto700():
    run_upper_air_pipeline(70000, 'frontogenesis', 'RdBu_r', 'RAP Analysis: 700-hPa Frontogenesis and Heights', 'Frontogenesis (K/100km/3hr)', np.linspace(-10, 10, 41))

def run_wind850():
    run_upper_air_pipeline(85000, 'wind_speed', 'YlOrBr', 'RAP Analysis: 850-hPa Wind Speeds and Heights', 'Wind Speed (knots)', np.arange(20, 141, 10))

def run_dew850():
    run_upper_air_pipeline(85000, 'dewpoint', 'BuGn', 'RAP Analysis: 850-hPa Dewpoint and Heights', 'Dewpoint (°C)', np.arange(-40, 31, 2))

def run_mAdv850():
    run_upper_air_pipeline(85000, 'moisture_advection', 'PRGn', 'RAP Analysis: 850-hPa Moisture Advection and Heights', 'Moisture Advection (%/hour)', np.linspace(-20, 20, 41))

def run_tAdv850():
    run_upper_air_pipeline(85000, 'temp_advection', 'coolwarm', 'RAP Analysis: 850-hPa Temperature Advection and Heights', 'Temperature Advection (K/hour)', np.linspace(-20, 20, 41))

def run_divcon300():
    run_upper_air_pipeline(30000, 'divergence', 'RdBu_r', 'RAP Analysis: 300-hPa Divergence and Heights', 'Divergence ($10^{-5}$ s$^{-1}$)', np.linspace(-15, 15, 31))

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
    subparsers.add_parser('500wv', help='500mb Water Vapor + Heights + Vort Max')
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
        '500wv': run_500wv,
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
