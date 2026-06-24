#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone GFS Map Generator for the Atlantic.

Fetches GFS data from the UCAR THREDDS server and generates various
meteorological maps for the CONUS East Coast, Caribbean, and Atlantic.
"""

# Section 1: Imports
import logging
import io
import os
import sys
import json
import re
from datetime import datetime, timedelta, timezone
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests

from scipy import ndimage
from scipy.ndimage import generate_binary_structure, minimum_filter, gaussian_filter, label, binary_dilation
from scipy.interpolate import griddata
from skimage.morphology import skeletonize

import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import ColdFront, WarmFront, StationaryFront, OccludedFront, StationPlot
from metpy.plots.wx_symbols import sky_cover
from siphon.catalog import TDSCatalog
from siphon.simplewebservice.wyoming import WyomingUpperAir

# Section 2: Logging Configuration
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger('WeatherScript')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')

# Section 3: Constants
EARTH_RADIUS = 6371000  # meters
OMEGA = 7.292e-5  # Earth's angular velocity (rad/s)
REGION = "Atlantic"  # Define the region

# --- MODIFIED: Changed hardcoded paths to relative paths ---
INTERSTATES_SHP = './shapefiles/tl_2023_us_primaryroads.shp' 
LOGO_PATHS = ["./photo.jpg", "./boxlogo2.png"]
ICON_DIR = './icons'
# ---
METAR_STATIONS = [
    'KMIA', 'MUHA', 'TJSJ', 'TXKF', 'MYNN', 'MKJP', 'MTPP', 'KJFK', 'KBOS',
    'KDCA', 'KCHS', 'KJAX', 'KMSY', 'KBDA', 'MDSD', 'TNCF', 'TNCC'
]
SOUNDING_STATIONS = ['MFL', 'JAX', 'CHS', 'XMR', 'LIX'] # Miami, Jacksonville, Charleston, Cape Canaveral, Slidell

# List of major Atlantic cities
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

# Section 4: Helper Functions
def get_sounding_data(station):
    """Fetches the latest available sounding data for a given station."""
    try:
        now = datetime.now(timezone.utc)
        times_to_try = [
            now.replace(hour=12, minute=0, second=0, microsecond=0),
            now.replace(hour=0, minute=0, second=0, microsecond=0),
            (now - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
        ]
        for time in times_to_try:
            try:
                df = WyomingUpperAir.request_data(time, station)
                if not df.empty:
                    logging.info(f"Successfully fetched sounding for {station} at {time}")
                    return df
            except Exception:
                continue
        logging.warning(f"Could not fetch sounding data for {station} in the last 24 hours.")
        return None
    except Exception as e:
        logging.error(f"An error occurred fetching sounding data for {station}: {e}")
        return None

def analyze_sounding_for_fronts(df):
    """Analyzes a sounding DataFrame for frontal indicators."""
    if df is None or df.empty:
        return 0

    p = df['pressure'].values * units.hPa
    T = df['temperature'].values * units.degC
    u = df['u_wind'].values * units.knots
    v = df['v_wind'].values * units.knots

    lower_levels = p > 850 * units.hPa
    if np.any(lower_levels):
        temp_diff = np.diff(T[lower_levels])
        if np.any(temp_diff > 0.5 * units.delta_degC):
            return 1

    mid_levels = p > 700 * units.hPa
    if np.sum(mid_levels) > 1:
        try:
            shear_u, shear_v = mpcalc.bulk_shear(p[mid_levels], u[mid_levels], v[mid_levels])
            if np.sqrt(shear_u**2 + shear_v**2).m > 15:
                return 1
        except ValueError:
            pass
    return 0

def get_metar_data(stations):
    """Fetches and parses METAR data for a list of station ICAOs."""
    station_string = ','.join(stations)
    url = f"https://aviationweather.gov/api/data/metar?ids={station_string}&format=json&hours=2"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        obs_points = []
        for metar in data:
            try:
                point = {
                    'lat': metar['lat'],
                    'lon': metar['lon'],
                    'tmpc': metar.get('temp', np.nan),
                    'dwpc': metar.get('dewp', np.nan),
                    'wspd': metar.get('wspd', np.nan),
                    'wdir': metar.get('wdir', np.nan),
                    'slp': metar.get('slp', np.nan),
                    'cover': metar.get('sky_cover', 'CLR')
                }
                if not any(np.isnan(v) for k, v in point.items() if k not in ['cover', 'lat', 'lon']):
                    obs_points.append(point)
            except (KeyError, TypeError):
                continue
        return obs_points
    except (requests.RequestException, json.JSONDecodeError) as e:
        logging.error(f"Error fetching METAR data: {e}")
        return []

def plot_metar_stations(ax, metar_obs):
    """Plots METAR station data on the map."""
    if not metar_obs:
        return

    stationplot = StationPlot(ax, [o['lon'] for o in metar_obs], [o['lat'] for o in metar_obs],
                              transform=ccrs.PlateCarree(), fontsize=8)

    temps = np.array([o['tmpc'] for o in metar_obs])
    dewps = np.array([o['dwpc'] for o in metar_obs])
    slps = np.array([o['slp'] for o in metar_obs])

    sky_cover_map = {'CLR': 0, 'SKC': 0, 'FEW': 1, 'SCT': 3, 'BKN': 5, 'OVC': 8}
    covers = np.array([sky_cover_map.get(o['cover'], 0) for o in metar_obs])

    stationplot.plot_parameter('NW', temps, color='red')
    stationplot.plot_parameter('SW', dewps, color='darkgreen')
    stationplot.plot_parameter('NE', slps, formatter=lambda v: f'{v:.0f}' if not np.isnan(v) else '')
    stationplot.plot_symbol('C', covers, sky_cover)

    winds = [((o['wspd'] * units.knots).to('m/s').m, o['wdir']) for o in metar_obs]
    for (spd, dir_), (lon, lat) in zip(winds, [(o['lon'], o['lat']) for o in metar_obs]):
        if not np.isnan(spd):
            stationplot.plot_barb(spd * np.cos(np.radians(270 - dir_)),
                                  spd * np.sin(np.radians(270 - dir_)), 
                                  length=6)

def compute_wind_speed(u, v):
    """Compute wind speed from u and v components (m/s to knots)."""
    wind_speed_ms = np.sqrt(u**2 + v**2)
    return wind_speed_ms * 1.94384

def compute_vorticity(u, v, lat, lon):
    """Compute absolute vorticity (scaled by 10^5 s^-1)."""
    lat_rad = np.deg2rad(lat)
    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)

    dx = EARTH_RADIUS * np.cos(lat_rad) * dlon_rad
    dy = EARTH_RADIUS * np.abs(dlat_rad)

    v_x = np.full_like(v, np.nan)
    v_x[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx[:, None])
    u_y = np.full_like(u, np.nan)
    u_y[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * dy)

    zeta_r = v_x - u_y
    f = 2 * OMEGA * np.sin(lat_rad)[:, None]
    zeta_a = zeta_r + f
    return zeta_a * 1e5

def compute_advection(phi, u, v, lat, lon):
    """Compute advection of a scalar field."""
    lat_rad = np.deg2rad(lat)
    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)

    dx = EARTH_RADIUS * np.cos(lat_rad) * dlon_rad
    dy = EARTH_RADIUS * np.abs(dlat_rad)

    phi_x = np.full_like(phi, np.nan)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx[:, None])
    phi_y = np.full_like(phi, np.nan)
    phi_y[1:-1, :] = (phi[:-2, :] - phi[2:, :]) / (2 * dy)

    return u * phi_x + v * phi_y

def compute_dewpoint(T, rh):
    """Compute dewpoint temperature (°C) from temperature (K) and relative humidity (%)."""
    T_C = T - 273.15
    rh = np.clip(rh, 1e-10, 100)
    ln_rh = np.log(rh / 100.0)
    a = 17.67
    b = 243.5
    gamma = (a * T_C) / (b + T_C)
    dewpoint_C = (b * (ln_rh + gamma)) / (a - ln_rh - gamma)
    return dewpoint_C

def compute_divergence(u, v, lat, lon):
    """Compute horizontal divergence (s^-1) from u and v wind components."""
    lat_rad = np.deg2rad(lat)
    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)

    dx = EARTH_RADIUS * np.cos(lat_rad)[:, None] * dlon_rad
    dy = EARTH_RADIUS * dlat_rad

    u_x = np.full_like(u, np.nan)
    v_y = np.full_like(v, np.nan)
    u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    v_y[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)

    divergence = u_x + v_y
    return divergence

def frontogenesis_700hPa(T, u, v, lat, lon):
    """Compute frontogenesis at 700 hPa (K/100km/3hr)."""
    try:
        lat_rad = np.deg2rad(lat)
        dlon = lon[1] - lon[0]
        dlat = lat[1] - lat[0]
        dlon_rad = np.deg2rad(dlon)
        dlat_rad = np.deg2rad(dlat)

        dx = EARTH_RADIUS * np.cos(lat_rad)[:, None] * dlon_rad
        dy = EARTH_RADIUS * np.abs(dlat_rad)

        dT_dx = np.full_like(T, np.nan)
        dT_dy = np.full_like(T, np.nan)
        dT_dx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * dx)
        dT_dy[1:-1, :] = (T[:-2, :] - T[2:, :]) / (2 * dy)

        du_dx = np.full_like(u, np.nan)
        du_dy = np.full_like(u, np.nan)
        dv_dx = np.full_like(v, np.nan)
        dv_dy = np.full_like(v, np.nan)
        du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        du_dy[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * dy)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
        dv_dy[1:-1, :] = (v[:-2, :] - v[2:, :]) / (2 * dy)

        mag_grad_T = np.sqrt(dT_dx**2 + dT_dy**2)
        epsilon = 1e-10
        mag_grad_T_safe = np.where(mag_grad_T < epsilon, epsilon, mag_grad_T)

        F = (1 / mag_grad_T_safe) * (
            - (dT_dx**2) * du_dx
            - dT_dx * dT_dy * (dv_dx + du_dy)
            - (dT_dy**2) * dv_dy
        )

        F_scaled = F * 1e5 * 10800
        return F_scaled
    except Exception as e:
        logging.error(f"Error computing frontogenesis: {e}")
        return None

def get_latest_run_date():
    """Determines the most recent GFS run time available."""
    now = datetime.now(timezone.utc)
    run_hours = [0, 6, 12, 18]
    check_time = now - timedelta(hours=6)
    
    for run_hour in sorted(run_hours, reverse=True):
        if check_time.hour >= run_hour:
            run_date = check_time.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            return run_date
            
    run_date = (check_time - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
    return run_date

def get_time_dimension(ds, run_date):
    """Dynamically select the appropriate time dimension and index from the dataset."""
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
                logging.debug(f"Found time dimension: {dim}")
                selected_dims[dim] = 0
                break
        else:
            logging.error(f"No time dimension found. Available dimensions: {ds.dims}")
            raise ValueError("No valid time dimension found in dataset")

        if 'time' in ds.dims and 'time' not in selected_dims:
            selected_dims['time'] = 0
            logging.debug(f"Added time dimension: time")

        logging.debug(f"Selected time dimensions: {selected_dims}")
        return selected_dims
    except Exception as e:
        logging.error(f"Error in get_time_dimension: {e}")
        raise

def get_gfs_data_for_level(level, forecast_hour=0):
    """Fetches GFS data for a specific isobaric level and forecast hour."""
    run_date = get_latest_run_date()
    init_time = run_date
    valid_time = run_date + timedelta(hours=forecast_hour)

    logging.debug(f"Fetching GFS data for level {level} Pa at +{forecast_hour}h (valid: {valid_time})")

    catalog_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml'
    cat = TDSCatalog(catalog_url)
    latest_dataset = list(cat.datasets.values())[0]
    ncss = latest_dataset.subset()

    query = ncss.query()
    query.accept('netcdf4')
    query.time(valid_time)
    query.variables('Geopotential_height_isobaric',
                    'u-component_of_wind_isobaric',
                    'v-component_of_wind_isobaric',
                    'Relative_humidity_isobaric',
                    'Temperature_isobaric')
    query.vertical_level([level])
    query.lonlat_box(north=50, south=10, east=-30, west=-100)  # Atlantic region

    try:
        data = ncss.get_data(query)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
        ds = ds.metpy.parse_cf()
        logging.debug(f"Available variables for level {level}: {list(ds.variables)}")
        return ds, init_time, valid_time
    except Exception as e:
        logging.error(f"Error fetching data for level {level} at +{forecast_hour}h: {e}")
        return None, None, None

def get_gfs_surface_data(forecast_hour=0):
    """Fetches GFS surface data for a given forecast hour."""
    run_date = get_latest_run_date()
    init_time = run_date
    valid_time = run_date + timedelta(hours=forecast_hour)

    logging.debug(f"Fetching GFS surface data at +{forecast_hour}h (valid: {valid_time})")

    catalog_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml'
    try:
        cat = TDSCatalog(catalog_url)
        latest_dataset = list(cat.datasets.values())[0]
        ncss = latest_dataset.subset()

        query = ncss.query()
        query.accept('netcdf4')
        query.time(valid_time)
        query.variables(
            'Temperature_surface',
            'Pressure_surface',
            'Geopotential_height_surface',
            'u-component_of_wind_height_above_ground',
            'v-component_of_wind_height_above_ground'
        )
        query.lonlat_box(north=50, south=10, east=-30, west=-100)
        query.vertical_level(10)

        logging.info(f"Downloading GFS surface data for +{forecast_hour}h...")
        data = ncss.get_data(query)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
        ds = ds.metpy.parse_cf()
        logging.info("Successfully downloaded GFS surface data.")

        ds = ds.rename({
            'Temperature_surface': 't2m',
            'Pressure_surface': 'sp',
            'Geopotential_height_surface': 'orog',
            'u-component_of_wind_height_above_ground': 'u10',
            'v-component_of_wind_height_above_ground': 'v10'
        })

        required_vars = ['t2m', 'sp', 'orog', 'u10', 'v10']
        missing_vars = [var for var in required_vars if var not in ds.variables]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        logging.debug(f"Dataset dimensions: {ds.dims}")
        logging.debug(f"Available variables: {list(ds.variables)}")

        return ds, init_time, valid_time
    except Exception as e:
        logging.error(f"Error in get_gfs_surface_data at +{forecast_hour}h: {e}")
        return None, None, None

def plot_background(ax):
    """Adds background features to the map axes."""
    ax.set_extent([-100, -30, 10, 50], crs=ccrs.PlateCarree())  # Atlantic
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidths=2.5, edgecolor='#750b7a')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', linewidths=2, edgecolor='#750b7a')
    ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.5)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

def add_cities(ax):
    """Adds major cities to the map with markers and labels."""
    for city in cities:
        ax.plot(city['lon'], city['lat'], 'o', color='white', markeredgecolor='black', markersize=4, transform=ccrs.PlateCarree())
        ax.text(city['lon'] + 0.5, city['lat'] + 0.5, city['name'], color='black', fontsize=6, transform=ccrs.PlateCarree(),
                ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

def generate_map(ds, init_time, valid_time, level, variable, cmap, title, cb_label, levels=None, forecast_hour=0):
    """Generates a map for a specified isobaric level and variable."""
    try:
        time_dims = get_time_dimension(ds, init_time)
        ds = ds.isel(**time_dims)
        
        lon = ds.get('longitude')
        lat = ds.get('latitude')
        if lon is None or lat is None:
            raise ValueError("Longitude or latitude data is missing from the dataset.")

        lon = lon.values
        lat = lat.values
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        heights = ds['Geopotential_height_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
        heights_smooth = ndimage.gaussian_filter(heights, sigma=3, order=0)

        u_wind = ds['u-component_of_wind_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
        v_wind = ds['v-component_of_wind_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()

        if variable == 'wind_speed':
            data = compute_wind_speed(u_wind, v_wind)
        elif variable == 'vorticity':
            data = compute_vorticity(u_wind, v_wind, lat, lon)
        elif variable == 'relative_humidity':
            data = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
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
        elif variable == 'divergence':
            data = compute_divergence(u_wind, v_wind, lat, lon)
        elif variable == 'frontogenesis':
            if level != 70000:
                raise ValueError("Frontogenesis is only computed at 700 hPa.")
            temp = ds['Temperature_isobaric'].sel(isobaric=level, method='nearest').squeeze().metpy.convert_units('degC').metpy.dequantify().values.copy()
            data = frontogenesis_700hPa(temp, u_wind, v_wind, lat, lon)
            if data is None:
                raise ValueError("Failed to compute frontogenesis.")
        else:
            raise ValueError(f"Unsupported variable type: {variable}")

        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': crs})
        fig.patch.set_facecolor('lightsteelblue')

        if levels is None:
            cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs, extend='both')
        else:
            cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs, levels=levels, extend='both')

        c = ax.contour(lon_2d, lat_2d, heights_smooth, colors='black', linewidths=2, transform=crs)
        ax.clabel(c, fontsize=8, inline=1, fmt='%i')

        u_wind_knots = u_wind * 1.94384
        v_wind_knots = v_wind * 1.94384
        ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind_knots[::5, ::5], v_wind_knots[::5, ::5], transform=crs, length=6)

        ax.set_title(f"{title} | Valid: {valid_time.strftime('%d %B %Y %H:%MZ')} (+{forecast_hour}h)", fontsize=16)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03, extend='both')
        cb.set_label(cb_label, size='large')

        plot_background(ax)
        add_cities(ax)

        add_logos_to_figure(fig, LOGO_PATHS, logo_size=1.0, logo_pad=0.03)

        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.show()
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logging.error(f"Error in generate_map: {e}", exc_info=True)
        return None

def generate_mslp_temp_map(forecast_hour=0):
    """Generate a Mean Sea Level Pressure (MSLP) chart with temperature gradient and meteorological features."""
    try:
        ds, init_time, valid_time = get_gfs_surface_data(forecast_hour)
        if ds is None:
            raise ValueError("Failed to retrieve surface data.")

        time_dims = get_time_dimension(ds, init_time)
        ds = ds.isel(**time_dims)

        lon = ds.longitude.values
        lat = ds.latitude.values
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        temp_surface = ds["t2m"].squeeze().metpy.convert_units("degC").metpy.dequantify()
        surface_pressure = ds["sp"].squeeze().metpy.convert_units("Pa").metpy.dequantify()
        elevation = ds["orog"].squeeze().metpy.dequantify()
        u_wind = ds["u10"].squeeze().metpy.dequantify()
        v_wind = ds["v10"].squeeze().metpy.dequantify()

        g = 9.80665
        Rd = 287.05
        temp_kelvin = temp_surface + 273.15
        mslp = surface_pressure * np.exp((g * elevation) / (Rd * temp_kelvin))
        mslp = mslp / 100

        mslp = np.where((mslp >= 850) & (mslp <= 1100), mslp, np.nan)
        mslp_smooth = ndimage.gaussian_filter(mslp, sigma=3, order=0)

        temp_grad_x, temp_grad_y = np.gradient(temp_surface, lon[1] - lon[0], lat[1] - lat[0])
        temp_grad_mag = np.sqrt(temp_grad_x**2 + temp_grad_y**2)
        frontogenesis = compute_advection(temp_grad_mag, u_wind, v_wind, lat, lon)

        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': crs})
        fig.patch.set_facecolor('lightsteelblue')

        plot_background(ax)
        add_cities(ax)

        cf = ax.contourf(
            lon_2d, lat_2d, temp_surface,
            levels=np.linspace(np.nanmin(temp_surface), np.nanmax(temp_surface), 41),
            cmap='jet', transform=crs
        )

        mslp_min = np.floor(np.nanmin(mslp) / 2) * 2
        mslp_max = np.ceil(np.nanmax(mslp) / 2) * 2
        isobar_levels = np.arange(mslp_min, mslp_max + 2, 2)
        c = ax.contour(lon_2d, lat_2d, mslp_smooth, levels=isobar_levels, colors='black', linewidths=2, transform=crs)
        ax.clabel(c, fmt='%d hPa', inline=True, fontsize=5)

        u_wind_knots = u_wind * 1.94384
        v_wind_knots = v_wind * 1.94384
        ax.barbs(
            lon_2d[::5, ::5], lat_2d[::5, ::5],
            u_wind_knots[::5, ::5], v_wind_knots[::5, ::5],
            transform=crs, length=6, color='black'
        )

        dry_line_mask = (temp_grad_mag > np.percentile(temp_grad_mag, 90)) & (frontogenesis > -0.005) & (frontogenesis < 0.005)
        ax.contour(lon_2d, lat_2d, dry_line_mask, levels=[0.5], colors='brown', linestyles='-.', linewidths=1, transform=crs)

        main_title = f"Atlantic: MSLP with Temperature Gradient (°C) and Features"
        ax.set_title(main_title, fontsize=16)
        fig.suptitle(f"Valid: {valid_time.strftime('%d %B %Y %H:%MZ')} (+{forecast_hour}h)", fontsize=12, y=1.02)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
        cb.set_label('Temperature (°C)', size='large')

        add_logos_to_figure(fig, LOGO_PATHS, logo_size=1.0, logo_pad=0.03)
        add_cities(ax)

        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.show()
        plt.close(fig)
        buf.seek(0)

        return buf, init_time, valid_time

    except Exception as e:
        logging.error(f"Error in generate_mslp_temp_map: {e}", exc_info=True)
        return None, None, None

def add_logos_to_figure(fig, logo_paths, logo_size=1.0, logo_pad=0.03):
    """Adds logos to the figure at the top-left and top-right positions."""
    fig_width, fig_height = fig.get_size_inches()
    logo_width = logo_size / fig_width
    logo_height = logo_size / fig_height
    pad_width = logo_pad / fig_width
    pad_height = logo_pad / fig_height

    positions = [
        {'left': pad_width, 'bottom': 1 - pad_height - logo_height, 'path': logo_paths[0]},
        {'left': 1 - pad_width - logo_width, 'bottom': 1 - pad_height - logo_height, 'path': logo_paths[1]}
    ]

    for pos in positions:
        try:
            ax_logo = fig.add_axes([pos['left'], pos['bottom'], logo_width, logo_height])
            ax_logo.axis('off')
            if not os.path.exists(pos['path']):
                logging.warning(f"Logo file not found: {pos['path']}. Skipping logo.")
                continue
            logo_img = plt.imread(pos['path'])
            ax_logo.imshow(logo_img)
        except Exception as e:
            logging.error(f"Error adding logo {pos['path']}: {e}")

# --- Standalone Execution Functions ---

def save_map_to_disk(image_bytes, filename):
    """Saves an image buffer to a file."""
    if image_bytes is None:
        print('Failed to generate the map (image_bytes is None).')
        return
    try:
        with open(filename, 'wb') as f:
            f.write(image_bytes.getbuffer())
        print(f"Successfully generated and saved map to {filename}")
        logging.info(f"Successfully generated and saved map to {filename}")
    except Exception as e:
        logging.error(f"Error saving map to file: {e}")
        print(f"Error saving map to file: {e}")
    finally:
        if image_bytes:
            image_bytes.close()

def run_wind300(forecast_hour=0):
    print(f'Generating 300 hPa wind map for +{forecast_hour}h, please wait...')
    try:
        ds, init_time, valid_time = get_gfs_data_for_level(30000, forecast_hour)
        if ds is None:
            print('Failed to retrieve data for the 300 hPa wind map.')
            return
        image_bytes = generate_map(
            ds, init_time, valid_time, 30000, 'wind_speed', 'cool', '300-hPa Wind Speeds and Heights (Atlantic)', 'Wind Speed (knots)',
            forecast_hour=forecast_hour
        )
        if image_bytes is None:
            print('Failed to generate the 300 hPa wind map due to missing or invalid data.')
            return
        filename = f'Awind300_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png'
        save_map_to_disk(image_bytes, filename)
    except Exception as e:
        logging.error(f"Error generating 300 hPa wind map at +{forecast_hour}h: {e}")
        print(f'An unexpected error occurred: {e}')

def run_wind500(forecast_hour=0):
    print(f'Generating 500 hPa wind map for +{forecast_hour}h, please wait...')
    try:
        ds, init_time, valid_time = get_gfs_data_for_level(50000, forecast_hour)
        if ds is None:
            print('Failed to retrieve data for the 500 hPa wind map.')
            return
        image_bytes = generate_map(
            ds, init_time, valid_time, 50000, 'wind_speed', 'YlOrBr', '500-hPa Wind Speeds and Heights (Atlantic)', 'Wind Speed (knots)',
            forecast_hour=forecast_hour
        )
        if image_bytes is None:
            print('Failed to generate the 500 hPa wind map due to missing or invalid data.')
            return
        filename = f'Awind500_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png'
        save_map_to_disk(image_bytes, filename)
    except Exception as e:
        logging.error(f"Error generating 500 hPa wind map at +{forecast_hour}h: {e}")
        print(f'An unexpected error occurred: {e}')

def run_vort500(forecast_hour=0):
    print(f'Generating 500 hPa vorticity map for +{forecast_hour}h, please wait...')
    try:
        ds, init_time, valid_time = get_gfs_data_for_level(50000, forecast_hour)
        if ds is None:
            print('Failed to retrieve data for the 500 hPa vorticity map.')
            return
        image_bytes = generate_map(
            ds, init_time, valid_time, 50000, 'vorticity', 'seismic', '500-hPa Absolute Vorticity and Heights (Atlantic)',
            r'Vorticity ($10^{-5}$ s$^{-1}$)', levels=np.linspace(-20, 20, 41), forecast_hour=forecast_hour
        )
        if image_bytes is None:
            print('Failed to generate the 500 hPa vorticity map due to missing or invalid data.')
            return
        filename = f'Avort500_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png'
        save_map_to_disk(image_bytes, filename)
    except Exception as e:
        logging.error(f"Error generating 500 hPa vorticity map at +{forecast_hour}h: {e}")
        print(f'An unexpected error occurred: {e}')

def run_rh700(forecast_hour=0):
    print(f'Generating 700 hPa relative humidity map for +{forecast_hour}h, please wait...')
    try:
        ds, init_time, valid_time = get_gfs_data_for_level(70000, forecast_hour)
        if ds is None:
            print('Failed to retrieve data for the 700 hPa relative humidity map.')
            return
        image_bytes = generate_map(
            ds, init_time, valid_time, 70000, 'relative_humidity', 'BuGn', '700-hPa Relative Humidity and Heights (Atlantic)', 'Relative Humidity (%)',
            forecast_hour=forecast_hour
        )
        if image_bytes is None:
            print('Failed to generate the 700 hPa relative humidity map due to missing or invalid data.')
            return
        filename = f'Arh700_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png'
        save_map_to_disk(image_bytes, filename)
    except Exception as e:
        logging.error(f"Error generating 700 hPa relative humidity map at +{forecast_hour}h: {e}")
        print(f'An unexpected error occurred: {e}')

def run_fronto700(forecast_hour=0):
    print(f'Generating 700 hPa frontogenesis map for +{forecast_hour}h, please wait...')
    try:
        ds, init_time, valid_time = get_gfs_data_for_level(70000, forecast_hour)
        if ds is None:
            print('Failed to retrieve data for the 700 hPa frontogenesis map.')
            return
        image_bytes = generate_map(
            ds, init_time, valid_time, 70000, 'frontogenesis', 'RdBu_r', '700-hPa Frontogenesis and Heights (Atlantic)',
            'Frontogenesis (K/100km/3hr)', levels=np.linspace(-10, 10, 41), forecast_hour=forecast_hour
        )
        if image_bytes is None:
            print('Failed to generate the 700 hPa frontogenesis map due to missing or invalid data.')
            return
        filename = f'Afronto700_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png'
        save_map_to_disk(image_bytes, filename)
    except Exception as e:
        logging.error(f"Error in run_fronto700 at +{forecast_hour}h: {e}")
        print(f'An unexpected error occurred: {e}')

def run_wind850(forecast_hour=0):
    print(f'Generating 850 hPa wind map for +{forecast_hour}h, please wait...')
    try:
        ds, init_time, valid_time = get_gfs_data_for_level(85000, forecast_hour)
        if ds is None:
            print('Failed to retrieve data for the 850 hPa wind map.')
            return
        image_bytes = generate_map(
            ds, init_time, valid_time, 85000, 'wind_speed', 'YlOrBr', '850-hPa Wind Speeds and Heights (Atlantic)', 'Wind Speed (knots)',
            forecast_hour=forecast_hour
        )
        if image_bytes is None:
            print('Failed to generate the 850 hPa wind map due to missing or invalid data.')
            return
        filename = f'Awind850_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png'
        save_map_to_disk(image_bytes, filename)
    except Exception as e:
        logging.error(f"Error generating 850 hPa wind map at +{forecast_hour}h: {e}")
        print(f'An unexpected error occurred: {e}')

def run_dew850(forecast_hour=0):
    print(f'Generating 850 hPa dewpoint map for +{forecast_hour}h, please wait...')
    try:
        ds, init_time, valid_time = get_gfs_data_for_level(85000, forecast_hour)
        if ds is None:
            print('Failed to retrieve data for the 850 hPa dewpoint map.')
            return
        image_bytes = generate_map(
            ds, init_time, valid_time, 85000, 'dewpoint', 'BuGn', '850-hPa Dewpoint (°C) (Atlantic)', 'Dewpoint (°C)',
            forecast_hour=forecast_hour
        )
        if image_bytes is None:
            print('Failed to generate the 850 hPa dewpoint map due to missing or invalid data.')
            return
        filename = f'Adew850_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png'
        save_map_to_disk(image_bytes, filename)
    except Exception as e:
        logging.error(f"Error generating 850 hPa dewpoint map at +{forecast_hour}h: {e}")
        print(f'An unexpected error occurred: {e}')

def run_mAdv850(forecast_hour=0):
    print(f'Generating 850 hPa moisture advection map for +{forecast_hour}h, please wait...')
    try:
        ds, init_time, valid_time = get_gfs_data_for_level(85000, forecast_hour)
        if ds is None:
            print('Failed to retrieve data for the 850 hPa moisture advection map.')
            return
        image_bytes = generate_map(
            ds, init_time, valid_time, 85000, 'moisture_advection', 'PRGn', '850-hPa Moisture Advection (Atlantic)', 'Moisture Advection (%/hour)',
            forecast_hour=forecast_hour
        )
        if image_bytes is None:
            print('Failed to generate the 850 hPa moisture advection map due to missing or invalid data.')
            return
        filename = f'AmAdv850_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png'
        save_map_to_disk(image_bytes, filename)
    except Exception as e:
        logging.error(f"Error generating 850 hPa moisture advection map at +{forecast_hour}h: {e}")
        print(f'An unexpected error occurred: {e}')

def run_tAdv850(forecast_hour=0):
    print(f'Generating 850 hPa temperature advection map for +{forecast_hour}h, please wait...')
    try:
        ds, init_time, valid_time = get_gfs_data_for_level(85000, forecast_hour)
        if ds is None:
            print('Failed to retrieve data for the 850 hPa temperature advection map.')
            return
        image_bytes = generate_map(
            ds, init_time, valid_time, 85000, 'temp_advection', 'coolwarm', '850-hPa Temperature Advection (Atlantic)', 'Temperature Advection (K/hour)',
            levels=np.linspace(-20, 20, 41), forecast_hour=forecast_hour
        )
        if image_bytes is None:
            print('Failed to generate the 850 hPa temperature advection map due to missing or invalid data.')
            return
        filename = f'AtAdv850_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png'
        save_map_to_disk(image_bytes, filename)
    except Exception as e:
        logging.error(f"Error generating 850 hPa temperature advection map at +{forecast_hour}h: {e}")
        print(f'An unexpected error occurred: {e}')

def run_mslp_temp(forecast_hour=0):
    print(f'Generating MSLP with temperature gradient map for +{forecast_hour}h, please wait...')
    try:
        logging.info(f"Fetching data for MSLP with temperature gradient map at +{forecast_hour}h")
        image_bytes, init_time, valid_time = generate_mslp_temp_map(forecast_hour)
        if image_bytes is None:
            print("Failed to generate the map due to an error in data processing.")
            return
        filename = f'Amslp_temp_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png'
        save_map_to_disk(image_bytes, filename)
    except Exception as e:
        logging.error(f"Error generating MSLP with temperature gradient map at +{forecast_hour}h: {e}")
        print(f'An unexpected error occurred: {e}')

def run_divcon300(forecast_hour=0):
    print(f'Generating 300 hPa divergence map for +{forecast_hour}h, please wait...')
    try:
        logging.info(f"Fetching data for 300 hPa divergence/convergence map at +{forecast_hour}h")
        ds, init_time, valid_time = get_gfs_data_for_level(30000, forecast_hour)
        if ds is None:
            print('Failed to retrieve data for the 300 hPa divergence/convergence map.')
            return
        image_bytes = generate_map(
            ds, init_time, valid_time, 30000, 'divergence', 'RdBu_r',
            '300-hPa Divergence/Convergence (Atlantic)', r'Divergence ($10^{-5}$ s$^{-1}$)',
            levels=np.linspace(-40, 40, 41), forecast_hour=forecast_hour
        )
        if image_bytes is None:
            print('Failed to generate the 300 hPa divergence/convergence map due to missing or invalid data.')
            return
        filename = f'Adivcon300_{valid_time.strftime("%Y%m%d_%H%MZ")}_f{forecast_hour:03d}.png'
        save_map_to_disk(image_bytes, filename)
    except Exception as e:
        logging.error(f"Error generating 300 hPa divergence/convergence map at +{forecast_hour}h: {e}")
        print(f'An unexpected error occurred: {e}')

# Section 7: Main execution block
def main():
    parser = argparse.ArgumentParser(
        description="Generate meteorological maps for the Atlantic region.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='The map to generate. Example: python atl_weather_maps.py mslp_temp 12')

    cmd_wind300 = subparsers.add_parser('wind300', help='300 hPa Wind Speed and Heights')
    cmd_wind300.add_argument('forecast_hour', type=int, nargs='?', default=0, help='Forecast hour (0-384, default: 0)')

    cmd_wind500 = subparsers.add_parser('wind500', help='500 hPa Wind Speed and Heights')
    cmd_wind500.add_argument('forecast_hour', type=int, nargs='?', default=0, help='Forecast hour (0-384, default: 0)')

    cmd_vort500 = subparsers.add_parser('vort500', help='500 hPa Absolute Vorticity')
    cmd_vort500.add_argument('forecast_hour', type=int, nargs='?', default=0, help='Forecast hour (0-384, default: 0)')

    cmd_rh700 = subparsers.add_parser('rh700', help='700 hPa Relative Humidity')
    cmd_rh700.add_argument('forecast_hour', type=int, nargs='?', default=0, help='Forecast hour (0-384, default: 0)')

    cmd_fronto700 = subparsers.add_parser('fronto700', help='700 hPa Frontogenesis')
    cmd_fronto700.add_argument('forecast_hour', type=int, nargs='?', default=0, help='Forecast hour (0-384, default: 0)')

    cmd_wind850 = subparsers.add_parser('wind850', help='850 hPa Wind Speed')
    cmd_wind850.add_argument('forecast_hour', type=int, nargs='?', default=0, help='Forecast hour (0-384, default: 0)')

    cmd_dew850 = subparsers.add_parser('dew850', help='850 hPa Dewpoint')
    cmd_dew850.add_argument('forecast_hour', type=int, nargs='?', default=0, help='Forecast hour (0-384, default: 0)')

    cmd_mAdv850 = subparsers.add_parser('mAdv850', help='850 hPa Moisture Advection')
    cmd_mAdv850.add_argument('forecast_hour', type=int, nargs='?', default=0, help='Forecast hour (0-384, default: 0)')

    cmd_tAdv850 = subparsers.add_parser('tAdv850', help='850 hPa Temperature Advection')
    cmd_tAdv850.add_argument('forecast_hour', type=int, nargs='?', default=0, help='Forecast hour (0-384, default: 0)')

    cmd_mslp_temp = subparsers.add_parser('mslp_temp', help='MSLP, 2m Temp, 10m Wind, and Fronts')
    cmd_mslp_temp.add_argument('forecast_hour', type=int, nargs='?', default=0, help='Forecast hour (0-384, default: 0)')

    cmd_divcon300 = subparsers.add_parser('divcon300', help='300 hPa Divergence/Convergence')
    cmd_divcon300.add_argument('forecast_hour', type=int, nargs='?', default=0, help='Forecast hour (0-384, default: 0)')
    
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
        'mslp_temp': run_mslp_temp,
        'divcon300': run_divcon300,
    }
    
    func_to_run = command_functions.get(args.command)
    
    if func_to_run:
        try:
            func_to_run(args.forecast_hour)
        except Exception as e:
            logging.error(f"A critical error occurred while running {args.command}: {e}", exc_info=True)
            print(f"A critical error occurred: {e}")
            sys.exit(1)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()