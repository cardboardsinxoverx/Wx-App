#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Section Plotting Script for Current GFS Data.
Refitted for FrostByte Dark Theme Aesthetics with Global Smart Routing and Topography Masking.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
from metpy.interpolate import cross_section
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timezone
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import requests
import scipy.ndimage as ndimage

# --- Standalone FrostByte Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('CrossSectionScript')

# --- APPLYING FROSTBYTE THEME GLOBALS ---
plt.rcParams['font.family'] = 'DejaVu Sans'

FIG_BG_COLOR = '#333333'
AXES_BG_COLOR = '#2B2B2B'
GRIDLINE_COLOR = '#3932A0'

TEXT_OUTLINE = [pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()]

plt.rcParams.update({
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': 'white',
    'figure.facecolor': FIG_BG_COLOR,
    'axes.facecolor': AXES_BG_COLOR
})

LEVELS_HPA = [1000, 850, 700, 500, 300, 250, 200, 150, 100]

AVIATION_LABELS = [
    '1000 hPa (Sfc)', '850 hPa (FL050)', '700 hPa (FL100)', 
    '500 hPa (FL180)', '300 hPa (FL300)', '250 hPa (FL340)', 
    '200 hPa (FL390)', '150 hPa (FL450)', '100 hPa (FL530)'
]

GFS_CATALOG_URL = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml'

def get_latest_gfs_ncss():
    logger.info("Accessing GFS Catalog to fetch current dataset...")
    try:
        catalog = TDSCatalog(GFS_CATALOG_URL)
        latest_dataset = list(catalog.datasets.values())[0]
        ncss_url = latest_dataset.access_urls['NetcdfSubset']
        logger.info(f"Connected to NCSS: {ncss_url}")
        return NCSS(ncss_url)
    except Exception as e:
        logger.error(f"Failed to query GFS THREDDS Catalog: {e}")
        raise

def get_optimal_route_coords(start_lon, end_lon):
    s_180 = (start_lon + 180) % 360 - 180
    e_180 = (end_lon + 180) % 360 - 180
    dist_180 = abs(e_180 - s_180)
    
    s_360 = start_lon % 360
    e_360 = end_lon % 360
    dist_360 = abs(e_360 - s_360)
    
    if dist_360 < dist_180:
        logger.info("Pacific/Dateline Route detected. Using 0-360 numerical space.")
        return s_360, e_360, True
    else:
        logger.info("Atlantic/Prime Meridian Route detected. Using -180 to 180 numerical space.")
        return s_180, e_180, False

def download_data(ncss, start_lat, calc_start_lon, end_lat, calc_end_lon, use_360):
    logger.info("Requesting cross-section flight-path bounding box from server...")
    query = ncss.query()
    
    pad = 3.0
    min_lat = min(start_lat, end_lat) - pad
    max_lat = max(start_lat, end_lat) + pad
    min_lon = min(calc_start_lon, calc_end_lon) - pad
    max_lon = max(calc_start_lon, calc_end_lon) + pad
    
    query.lonlat_box(west=min_lon, east=max_lon, south=min_lat, north=max_lat)
    query.time(datetime.now(timezone.utc))
    query.accept('netcdf4')
    query.variables('Relative_humidity_isobaric', 'Geopotential_height_isobaric',
                    'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric',
                    'Temperature_isobaric', 'Pressure_surface')
    
    try:
        data_raw = ncss.get_data(query)
        store = xr.backends.NetCDF4DataStore(data_raw)
        ds = xr.open_dataset(store).metpy.parse_cf()
        
        if use_360:
            ds = ds.assign_coords(longitude=(ds.longitude % 360))
        else:
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
            
        ds = ds.sortby('longitude')
        
        logger.info("Data successfully conditioned and parsed into memory.")
        return ds
    except Exception as e:
        logger.error(f"Data retrieval sequence failed: {e}")
        raise

def resolve_location_string(lat, lon):
    norm_lon = lon
    if norm_lon > 180:
        norm_lon -= 360
        
    lat_dir = 'N' if lat >= 0 else 'S'
    lon_dir = 'E' if norm_lon >= 0 else 'W'
    
    return f"{abs(lat):.2f}°{lat_dir}/{abs(norm_lon):.2f}°{lon_dir}"

def generate_plot(ds, start_lat, start_lon, end_lat, end_lon, calc_start_lon, calc_end_lon, steps=500, output_file='cross_section.png'):
    logger.info("Processing cross-section interpolation vectors...")
    
    start_pt = (start_lat, calc_start_lon)
    end_pt = (end_lat, calc_end_lon)
    
    cross = cross_section(ds, start_pt, end_pt, steps=steps)
    cross = cross.squeeze()
    
    # --- XARRAY NATIVE EXTRACTION ---
    vert_col = [col for col in cross.coords if 'isobaric' in col][0]
    
    isobaric_pa = cross[vert_col].values
    LAT = cross['latitude'].values
    LON = cross['longitude'].values
    
    RH = cross['Relative_humidity_isobaric'].values
    HGT = cross['Geopotential_height_isobaric'].values
    U = cross['u-component_of_wind_isobaric'].values
    V = cross['v-component_of_wind_isobaric'].values
    TMP = cross['Temperature_isobaric'].values

    num_lvls = len(isobaric_pa)
    num_pts = len(LAT)

    TMP_C = TMP - 273.15
    WIND_SPEED_KT = np.sqrt(U**2 + V**2) * 1.94384

    idx_850 = (np.abs(isobaric_pa - 85000)).argmin()
    idx_500 = (np.abs(isobaric_pa - 50000)).argmin()
    
    u_shear = U[idx_500, :] - U[idx_850, :]
    v_shear = V[idx_500, :] - V[idx_850, :]
    shear_mag = np.sqrt(u_shear**2 + v_shear**2) * 1.94384 
    
    # Drop this right here: Smooth the 1D line graph
    shear_mag = ndimage.gaussian_filter1d(shear_mag, sigma=3.0) 

    # --- DEFINE THE MESHGRIDS ---
    dist_array = np.arange(num_pts)
    X, Y = np.meshgrid(dist_array, isobaric_pa / 100.0) 

    # --- SET UP THE PRIMARY PLOT FIGURE ---
    fig = plt.figure(figsize=(16, 12))
    ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    ax1.set_facecolor(AXES_BG_COLOR)

    # --- ICING CALCULATION ---
    icing_mask = np.zeros_like(RH)
    icing_mask[(RH >= 70) & (TMP_C >= -15) & (TMP_C <= 0)] = 1
    icing_mask[(RH >= 80) & (TMP_C >= -10) & (TMP_C <= -2)] = 2
    icing_mask[(RH >= 90) & (TMP_C >= -8) & (TMP_C <= -3)] = 3

    # --- TURBULENCE / VERTICAL SHEAR CALCULATION ---
    turb_intensity = np.zeros_like(U)
    for i in range(num_pts):
        if np.isnan(HGT[:, i]).all():
            continue
            
        hgt_ft = HGT[:, i] * 3.28084
        
        sort_idx = np.argsort(hgt_ft)
        sorted_hgt = hgt_ft[sort_idx]
        sorted_U = (U[:, i] * 1.94384)[sort_idx]
        sorted_V = (V[:, i] * 1.94384)[sort_idx]
        
        du_dh = np.gradient(sorted_U, sorted_hgt)
        dv_dh = np.gradient(sorted_V, sorted_hgt)
        
        unsort_idx = np.argsort(sort_idx)
        turb_intensity[:, i] = np.sqrt(du_dh[unsort_idx]**2 + dv_dh[unsort_idx]**2) * 1000

    turb_intensity[0, :] = 0  
    
    # Drop this right here: Smooth the 2D turbulence matrix
    # sigma=[vertical_smoothing, horizontal_smoothing]
    turb_intensity = ndimage.gaussian_filter(turb_intensity, sigma=[1.0, 3.0])
    
    idx_1000 = (np.abs(isobaric_pa - 100000)).argmin()
    u_llws = (U[idx_850, :] - U[idx_1000, :]) * 1.94384
    v_llws = (V[idx_850, :] - V[idx_1000, :]) * 1.94384
    llws_mag = np.sqrt(u_llws**2 + v_llws**2)
    
    llws_mask = np.zeros_like(RH)
    llws_mask[0:2, :] = llws_mag 
        
    rh_contour = ax1.contourf(X, Y, RH, levels=np.linspace(0, 100, 11), cmap='YlGnBu', alpha=0.85)
    
    plt.rcParams['hatch.color'] = 'magenta'
    ax1.contourf(X, Y, icing_mask, levels=[0.5, 1.5, 2.5, 3.5], colors='none', hatches=['...', '///', 'xxx'])
    
    plt.rcParams['hatch.color'] = '#73fc03'
    turb_levels = [4, 8, 12, 16, 50] 
    ax1.contourf(X, Y, turb_intensity, levels=turb_levels, colors='none', hatches=['ooo', 'OO', 'xxx', '+++'])
    if np.max(turb_intensity) > 4:
        ax1.contour(X, Y, turb_intensity, levels=[4.0], colors=['#73fc03'], linewidths=2.0, zorder=6)
    
    plt.rcParams['hatch.color'] = 'red' 
    ax1.contourf(X, Y, llws_mask, levels=[25, 100], colors='none', hatches=['---'])
    if np.max(llws_mask) > 25:
        ax1.contour(X, Y, llws_mask, levels=[25.0], colors=['red'], linewidths=2.5, zorder=6)
    
    plt.rcParams['hatch.color'] = 'white'
    
    if 'Pressure_surface' in cross:
        sfc_press_pa = cross['Pressure_surface'].values
        sfc_press_hpa = sfc_press_pa / 100.0
        
        ax1.fill_between(dist_array, sfc_press_hpa, 1030, facecolor='#161616', zorder=9)
        ax1.plot(dist_array, sfc_press_hpa, color='#8B4513', linewidth=2.5, zorder=10)
    
    freezing_line = ax1.contour(X, Y, TMP_C, levels=[0.0], colors='#00FFFF', linestyles='dashed', linewidths=2.5, zorder=5)
    
    jet_contour = ax1.contour(X, Y, WIND_SPEED_KT, levels=np.arange(60, 180, 20), colors='#fc03a9', linewidths=1.2, alpha=0.7, zorder=4)
    ax1.clabel(jet_contour, inline=True, fmt='%d kt', fontsize=8, colors='#fc03a9', use_clabeltext=True)

    skip_x = max(1, num_pts // 15)
    skip_y = max(1, num_lvls // 12) 
    ax1.barbs(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x], 
              U[::skip_y, ::skip_x]*1.94384, V[::skip_y, ::skip_x]*1.94384, 
              color='white', length=5, linewidth=0.6, barbcolor='white', flagcolor='white',
              path_effects=TEXT_OUTLINE)
    
    ax1.set_yscale('log')
    ax1.set_ylim(1030, 100)
    
    ax1.get_yaxis().set_major_locator(ticker.FixedLocator(LEVELS_HPA))
    ax1.get_yaxis().set_major_formatter(ticker.FixedFormatter(AVIATION_LABELS))
    
    ax1.set_ylabel('Pressure Altitude (Flight Level)', weight='bold', path_effects=TEXT_OUTLINE)
    
    start_location = resolve_location_string(start_lat, start_lon)
    end_location = resolve_location_string(end_lat, end_lon)
    
    time_col = 'reftime' if 'reftime' in cross.coords else 'time'
    time_val = cross[time_col].values
    valid_time = pd.to_datetime(np.atleast_1d(time_val)[0])
    
    ax1.set_title(f"Atmospheric Cross-Section Profile | {start_location} ➔ {end_location}\nValid: {valid_time.strftime('%Y-%m-%d %H:%M UTC')}", 
                 weight='bold', fontsize=12, pad=10, path_effects=TEXT_OUTLINE)
    
    tick_idx = np.linspace(0, num_pts-1, 5, dtype=int)
    ax1.set_xticks(tick_idx)
    
    formatted_labels = []
    for i in tick_idx:
        norm_lon = LON[i]
        if norm_lon > 180:
            norm_lon -= 360
        lon_dir = 'E' if norm_lon >= 0 else 'W'
        formatted_labels.append(f"{LAT[i]:.1f}°N\n{abs(norm_lon):.1f}°{lon_dir}")
        
    ax1.set_xticklabels(formatted_labels, weight='bold')
    
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_path_effects(TEXT_OUTLINE)
    for text in ax1.texts:
        text.set_path_effects(TEXT_OUTLINE)
        
    ax1.grid(True, color='#555555', linestyle=':', alpha=0.5)

    ax2 = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=3, sharex=ax1)
    ax2.set_facecolor(AXES_BG_COLOR)
    
    ax2.plot(dist_array, shear_mag, color='#FF6B6B', linewidth=2.5, label='850-500 hPa Shear', path_effects=TEXT_OUTLINE)
    ax2.legend(loc='upper left', facecolor='#2B2B2B', edgecolor='white', fontsize=9)
    
    stripes = np.arange(0, num_pts, skip_x)
    for i in range(len(stripes)-1):
        if i % 2 == 0:
            ax2.axvspan(stripes[i], stripes[i+1], color='#383838', alpha=0.3, zorder=0)
            
    ax2.set_ylabel('Shear (knots)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
    ax2.set_xlabel('Flight Path Coordinates', weight='bold', path_effects=TEXT_OUTLINE)
    ax2.grid(True, color='#555555', linestyle=':', alpha=0.5)
    
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_path_effects(TEXT_OUTLINE)

    ax_inset = plt.subplot2grid((4, 4), (3, 3), rowspan=1, colspan=1, projection=ccrs.PlateCarree())
    ax_inset.set_facecolor(AXES_BG_COLOR)
    
    ax_inset.add_feature(cfeature.LAND, facecolor='#2B2B2B', edgecolor='none')
    ax_inset.add_feature(cfeature.OCEAN, facecolor='#1A2A3A', edgecolor='none')
    ax_inset.coastlines(resolution='50m', color='#555555', linewidth=0.8)
    ax_inset.add_feature(cfeature.STATES, edgecolor='#444444', linewidth=0.4)
    
    ax_inset.plot([start_lon, end_lon], [start_lat, end_lat], color='#FF6B6B', 
                  linewidth=3, marker='o', transform=ccrs.Geodetic(), path_effects=TEXT_OUTLINE)
    ax_inset.text(start_lon, start_lat, ' A', color='white', weight='bold', path_effects=TEXT_OUTLINE, transform=ccrs.Geodetic())
    ax_inset.text(end_lon, end_lat, ' B', color='white', weight='bold', path_effects=TEXT_OUTLINE, transform=ccrs.Geodetic())
    
    if abs(calc_end_lon - calc_start_lon) > 90 or abs(end_lon - start_lon) > 180:
        ax_inset.set_global()
    else:
        pad = 4.0
        min_lat = min(start_lat, end_lat) - pad
        max_lat = max(start_lat, end_lat) + pad
        min_lon = min(start_lon, end_lon) - pad
        max_lon = max(start_lon, end_lon) + pad
        ax_inset.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
    ax_inset.set_title('Flight Track Orientation', weight='bold', fontsize=11, path_effects=TEXT_OUTLINE)

    rh_colorbar = fig.colorbar(rh_contour, ax=ax1, orientation='horizontal', pad=0.12, aspect=60, shrink=1.0, extend='both')
    rh_colorbar.set_label('Relative Humidity (%)', weight='bold', path_effects=TEXT_OUTLINE)
    rh_colorbar.ax.tick_params(labelcolor='white')
    for label in rh_colorbar.ax.get_xticklabels():
        label.set_path_effects(TEXT_OUTLINE)
        label.set_fontweight('bold')

    legend_handles = [
        Line2D([0], [0], color='#00FFFF', linestyle='dashed', linewidth=2, label='0°C Freezing Line'),
        Line2D([0], [0], color='#fc03a9', linewidth=1.2, label='Jet Core')
    ]
    
    if np.max(icing_mask) > 0:
        legend_handles.extend([
            Patch(facecolor='none', edgecolor='magenta', hatch='...', label='Light Icing'),
            Patch(facecolor='none', edgecolor='magenta', hatch='///', label='Mod Icing'),
            Patch(facecolor='none', edgecolor='magenta', hatch='xxx', label='Sev Icing')
        ])  
        
    if np.max(turb_intensity) > 16:
        legend_handles.append(Patch(facecolor='none', edgecolor='#73fc03', hatch='+++', linewidth=2.0, label='Extreme Turb'))
    if np.max(turb_intensity) > 12: 
        legend_handles.append(Patch(facecolor='none', edgecolor='#73fc03', hatch='xxx', linewidth=2.0, label='Severe Turb'))
    if np.max(turb_intensity) > 8: 
        legend_handles.append(Patch(facecolor='none', edgecolor='#73fc03', hatch='OO', linewidth=2.0, label='Mod Turb'))
    elif np.max(turb_intensity) > 4: 
        legend_handles.append(Patch(facecolor='none', edgecolor='#73fc03', hatch='ooo', linewidth=2.0, label='Light Turb'))
        
    if np.max(llws_mask) > 25:
        legend_handles.append(Patch(facecolor='none', edgecolor='red', hatch='---', label='LL Wind Shear'))
    
    ice_leg = ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                         framealpha=0.9, facecolor='#2B2B2B', edgecolor='white', fontsize=9, prop={'weight':'bold'})
    
    for text in ice_leg.get_texts():
        text.set_color('white')
        text.set_path_effects(TEXT_OUTLINE)

    plt.subplots_adjust(hspace=0.5)
    plt.suptitle('BOX WEATHER METEOROLOGICAL CROSS-SECTION', fontsize=20, fontweight='bold', color='white', y=0.96, path_effects=TEXT_OUTLINE)
    
    os.makedirs('outputs', exist_ok=True)
    
    start_str = f"{abs(start_lat):.0f}{'N' if start_lat >= 0 else 'S'}_{abs(start_lon):.0f}{'E' if start_lon >= 0 else 'W'}"
    end_str = f"{abs(end_lat):.0f}{'N' if end_lat >= 0 else 'S'}_{abs(end_lon):.0f}{'E' if end_lon >= 0 else 'W'}"
    coord_label = f"{start_str}_to_{end_str}"
    
    timestamp_str = valid_time.strftime("%Y%m%d_%H%M")
    
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    final_output_path = os.path.join('outputs', f"{base_name}_{coord_label}_{timestamp_str}.png")
    
    plt.savefig(final_output_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    logger.info(f"✅ Cross-section plot saved to: {final_output_path}")
    print(f"✅ Generated {final_output_path}")
    plt.close(fig)

def parse_location(loc_string):
    loc_string = loc_string.strip()
    
    if ',' in loc_string:
        try:
            lat, lon = map(float, loc_string.split(','))
            return lat, lon
        except ValueError:
            raise ValueError(f"Invalid coordinate format: {loc_string}")
            
    logger.info(f"Attempting to resolve ICAO {loc_string.upper()} via AviationWeather API...")
    url = f"https://aviationweather.gov/api/data/stationinfo?ids={loc_string.upper()}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data and isinstance(data, list):
            lat = data[0].get('lat')
            lon = data[0].get('lon')
            if lat is not None and lon is not None:
                logger.info(f"Resolved {loc_string.upper()} to {lat}, {lon}")
                return float(lat), float(lon)
    except Exception as e:
        logger.error(f"Failed to fetch coords for {loc_string}: {e}")
        
    raise ValueError(f"Could not resolve location: {loc_string}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GFS Cross-Section Plot (Current Data)")
    parser.add_argument('start', type=str, help="Start point as 'lat,lon' or ICAO (e.g., 'KATL')")
    parser.add_argument('end', type=str, help="End point as 'lat,lon' or ICAO (e.g., 'KJFK')")
    parser.add_argument('--steps', type=int, default=500, help="Number of steps (default: 500)")
    parser.add_argument('--output', type=str, default='cross_section.png', help="Output file")
    
    args = parser.parse_args()
    
    try:
        start_lat, start_lon = parse_location(args.start)
        end_lat, end_lon = parse_location(args.end)
    except Exception as e:
        logger.error(f"Location parsing failed: {e}")
        sys.exit(1)
        
    calc_start_lon, calc_end_lon, use_360 = get_optimal_route_coords(start_lon, end_lon)
        
    ncss = get_latest_gfs_ncss()
    dataset = download_data(ncss, start_lat, calc_start_lon, end_lat, calc_end_lon, use_360)
    generate_plot(dataset, start_lat, start_lon, end_lat, end_lon, calc_start_lon, calc_end_lon, steps=args.steps, output_file=args.output)