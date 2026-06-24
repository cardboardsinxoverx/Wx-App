#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Command-Line Weather Tool.

This script provides various weather utilities, converted from a Discord bot.
It includes METAR/TAF/Alert fetchers, a meteogram generator, radar display,
NWS forecast retrieval, time conversion, and unit conversion.
"""

# Section 1: Imports
# Standard library
import argparse
import asyncio
import datetime as _dt
from datetime import datetime, timedelta, timezone
import gzip
import io
from io import BytesIO, StringIO
import json
import logging
import math
import re
import os
import random
import shutil
import signal
import sqlite3
import sys
import tempfile
import time
import traceback
from uuid import uuid4

# Third-party libraries
import aiohttp
import airportsdata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cmocean
import erddapy
from erddapy import ERDDAP
from bs4 import BeautifulSoup
import certifi
import geopandas as gpd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from matplotlib.collections import LineCollection
from matplotlib.colors import (LinearSegmentedColormap, Normalize, BoundaryNorm)
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import pytz
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
from PIL import Image
import requests
import urllib3
import xarray as xr

# Scientific / meteorology libraries
import astropy.coordinates as coord
from astropy.time import Time
import metpy
from metpy.units import units
import metpy.calc as mpcalc
from metpy.calc import parcel_profile, mixed_layer_cape_cin, dewpoint_from_relative_humidity
from metpy.plots import (
    add_metpy_logo, SkewT, Hodograph, add_timestamp,
    ColdFront, WarmFront, OccludedFront, StationaryFront, StationPlot
)
import metpy.interpolate as mpinterpolate

from siphon.simplewebservice.wyoming import WyomingUpperAir
from siphon.simplewebservice.iastate import IAStateUpperAir
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS

import scipy
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import scipy.spatial

from shapely.geometry import Point, LineString
from shapely.ops import unary_union

import ephem

import psutil
import fuzzywuzzy
from fuzzywuzzy import fuzz

# Optional visualization / utility libs
import cartopy
import matplotlib.patches as mpatches
import shapely

# Type hints
from typing import Optional


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

# Section 3: Constants & Theming
EARTH_RADIUS = 6371000  # meters
INTERSTATES_SHP = './shapefiles/tl_2023_us_primaryroads.shp' 
LOGO_PATHS = ["./photo.jpg", "./boxlogo2.png"]
ICON_DIR = './icons'

METAR_STATIONS = [
    'KMIA', 'MUHA', 'TJSJ', 'TXKF', 'MYNN', 'MKJP', 'MTPP', 'KJFK', 'KBOS',
    'KDCA', 'KCHS', 'KJAX', 'KMSY', 'KBDA', 'MDSD', 'TNCF', 'TNCC'
]
SOUNDING_STATIONS = ['MFL', 'JAX', 'CHS', 'XMR', 'LIX']

OPENWEATHERMAP_API_KEY = 'efd4f5ec6d2b16958a946b2ceb0419a6'
WEATHERBIT_API_KEY = '4759188e762b4184828c575b5d8df034'

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
    'axes.facecolor': AXES_BG_COLOR,
    'legend.facecolor': AXES_BG_COLOR,
    'legend.edgecolor': 'white'
})
# --- END THEME SETTINGS ---

def style_legend(legend):
    """Helper to automatically theme dynamic legends."""
    if not legend: return
    legend.get_frame().set_facecolor(AXES_BG_COLOR)
    legend.get_frame().set_edgecolor('white')
    legend.get_frame().set_alpha(0.9)
    if legend.get_title():
        legend.get_title().set_color("white")
        legend.get_title().set_path_effects(TEXT_OUTLINE)
        legend.get_title().set_fontweight('bold')
    for text in legend.get_texts():
        text.set_color("white")
        text.set_path_effects(TEXT_OUTLINE)
        text.set_fontweight('bold')


# Section 4: Locally Defined Commands

# --- METAR Command ---
def get_metar(icao, hoursback=0, format='json'):
    try:
        metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format={format}&hours={hoursback}'
        response = requests.get(metar_url, timeout=15)
        response.raise_for_status()
        json_data = response.json()

        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")

        raw_metars = [entry['rawOb'] for entry in json_data]
        return raw_metars

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error fetching METAR data for {icao} (Server may be down): {e}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Error parsing METAR data for {icao}: {e}")

def run_metar(args):
    """Fetches and prints METAR data."""
    airport_code = args.airport_code.upper()
    hoursback = args.hoursback
    try:
        raw_metars = get_metar(airport_code, hoursback)
        
        print(f"--- METAR for {airport_code} ---")
        if hoursback > 0:
            print(f"--- Last {hoursback} Hours ---")
            for i, raw_metar in enumerate(raw_metars):
                print(f"Observation {i+1}:\n{raw_metar}\n")
        else:
            print(f"{raw_metars[0]}\n")
        logging.info(f"METAR requested for {airport_code} (hoursback={hoursback})")
    except Exception as e:
        print(f"Error fetching METAR: {e}")

# --- Meteogram Command ---
def extract_cloud_info(metar):
    cloud_levels = {
        "low": [], "mid": [], "high": [], "vertical_visibility": None
    }
    cloud_pattern = re.compile(r'((FEW|SCT|BKN|OVC)(\d{3}))|(VV(\d{3}))')
    cloud_matches = re.findall(cloud_pattern, metar)
    for match in cloud_matches:
        if match[1]:
            cover = match[1]
            altitude_hundreds = int(match[2])
            altitude_ft = altitude_hundreds * 100
            if altitude_ft <= 6500:
                cloud_levels["low"].append((cover, altitude_ft))
            elif 6500 < altitude_ft <= 20000:
                cloud_levels["mid"].append((cover, altitude_ft))
            else:
                cloud_levels["high"].append((cover, altitude_ft))
        if match[4]:
            vv_hundreds = int(match[4])
            cloud_levels["vertical_visibility"] = vv_hundreds * 100
    return cloud_levels

def extract_wind_info(metar):
    wind_direction = np.nan
    wind_speed = np.nan
    wind_gusts = np.nan
    
    wind_match = re.search(r'(VRB|\d{3})(\d{2,3})(?:G(\d{2,3}))?KT', metar)
    if wind_match:
        dir_str = wind_match.group(1)
        wind_direction = np.nan if dir_str == 'VRB' else int(dir_str)
        wind_speed = int(wind_match.group(2))
        if wind_match.group(3):
            wind_gusts = int(wind_match.group(3))
            
    return wind_direction, wind_speed, wind_gusts

PRECIP_CODES = [
    'DZ', 'RA', 'SN', 'SG', 'IC', 'PL', 'GR', 'GS', 'UP',
    'SH', 'TS', 'FZ'
]
PRECIP_REGEX = r'(?<!VC)([+-]?(?:' + '|'.join(PRECIP_CODES) + r'))'

def process_metar_data(metar_list):
    data = {
        "time": [], "temperature": [], "dewpoint": [], "wind_direction": [],
        "wind_speed": [], "wind_gusts": [], "pressure": [], "low_clouds": [],
        "mid_clouds": [], "high_clouds": [], "vertical_visibility": [],
        "present_weather_codes": [], "hourly_precipitation_in": []
    }
    now_utc = datetime.now(timezone.utc)

    for metar in metar_list:
        try:
            time_match = re.search(r'(\d{2})(\d{2})(\d{2})Z', metar)
            if not time_match:
                raise ValueError("No valid time string found.")
            
            observation_day = int(time_match.group(1))
            observation_hour = int(time_match.group(2))
            observation_minute = int(time_match.group(3))

            temp_dt = now_utc.replace(day=observation_day,
                                     hour=observation_hour,
                                     minute=observation_minute,
                                     second=0, microsecond=0)
            if temp_dt > now_utc:
                if temp_dt.month == 1:
                    temp_dt = temp_dt.replace(year=temp_dt.year - 1, month=12)
                else:
                    temp_dt = temp_dt.replace(month=temp_dt.month - 1)
            
            data["time"].append(temp_dt)

        except (IndexError, ValueError, AttributeError):
            for key in data.keys():
                if key == "present_weather_codes": data[key].append([])
                else: data[key].append(np.nan)
            continue 

        temp_dewpoint_match = re.search(r'(?:^|\s)([M]?\d{2})/([M]?\d{2}|[X]{2})?(?=\s|$)', metar)
        if temp_dewpoint_match:
            temp_str = temp_dewpoint_match.group(1)
            dewpoint_str = temp_dewpoint_match.group(2)
            
            temp = int(temp_str.replace('M', '-')) if temp_str else np.nan
            
            if dewpoint_str and 'X' not in dewpoint_str:
                dewpoint = int(dewpoint_str.replace('M', '-'))
            else:
                dewpoint = np.nan
                
            if not np.isnan(temp) and not np.isnan(dewpoint) and dewpoint > temp:
                dewpoint = temp
                
            data["temperature"].append(temp)
            data["dewpoint"].append(dewpoint)
        else:
            data["temperature"].append(np.nan)
            data["dewpoint"].append(np.nan)

        direction, speed, gusts = extract_wind_info(metar)
        data["wind_direction"].append(direction)
        data["wind_speed"].append(speed)
        data["wind_gusts"].append(gusts)

        pressure_match = re.search(r'A(\d{4})', metar)
        pressure_inhg = float(pressure_match.group(1)) / 100 if pressure_match else np.nan
        pressure_hpa = pressure_inhg * 33.8639 if not np.isnan(pressure_inhg) else np.nan
        data["pressure"].append(pressure_hpa)

        cloud_info = extract_cloud_info(metar)
        data["low_clouds"].append(cloud_info["low"][0][1] if cloud_info["low"] else np.nan)
        data["mid_clouds"].append(cloud_info["mid"][0][1] if cloud_info["mid"] else np.nan)
        data["high_clouds"].append(cloud_info["high"][0][1] if cloud_info["high"] else np.nan)
        data["vertical_visibility"].append(cloud_info["vertical_visibility"] if cloud_info["vertical_visibility"] else np.nan)
        
        present_weather_matches = re.findall(PRECIP_REGEX, metar)
        valid_present_weather = [pw for pw in present_weather_matches if pw and any(code in pw for code in PRECIP_CODES)]
        data["present_weather_codes"].append(valid_present_weather)

        # Ensure 'P' is preceded by a space (or start of string) 
        # and the digits are followed by a space (or end of string).
        hourly_precip_match = re.search(r'(?:^|\s)P(\d{4})(?=\s|$)', metar)
        data["hourly_precipitation_in"].append(float(hourly_precip_match.group(1)) / 100 if hourly_precip_match else np.nan)

    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df.sort_values(by='time', inplace=True)
    
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['dewpoint'] = pd.to_numeric(df['dewpoint'], errors='coerce')
    df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce')

    # --- ADD THESE LINES TO FILTER SEVERE AWOS GLITCHES ---
    df.loc[df['wind_speed'] > 150, 'wind_speed'] = np.nan
    df.loc[df['wind_gusts'] > 150, 'wind_gusts'] = np.nan
    
    window_size = 5
    df['temperature_filtered'] = df['temperature'].rolling(window=window_size, center=True).median()
    df['dewpoint_filtered'] = df['dewpoint'].rolling(window=window_size, center=True).median()
    temp_spikes = (df['temperature'] - df['temperature_filtered']).abs() > 5
    dewpoint_spikes = (df['dewpoint'] - df['dewpoint_filtered']).abs() > 5
    df.loc[temp_spikes, 'temperature'] = df['temperature_filtered']
    df.loc[dewpoint_spikes, 'dewpoint'] = df['dewpoint_filtered']

    smoothing_window = 3
    df['temperature_smoothed'] = df['temperature'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    df['dewpoint_smoothed'] = df['dewpoint'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    df['temperature_smoothed'] = df['temperature_smoothed'].interpolate(method='linear')
    df['dewpoint_smoothed'] = df['dewpoint_smoothed'].interpolate(method='linear')
    df.dropna(subset=['temperature_smoothed', 'dewpoint_smoothed'], inplace=True)

    df['relative_humidity'] = mpcalc.relative_humidity_from_dewpoint(
        df['temperature_smoothed'].values * units.degC,
        df['dewpoint_smoothed'].values * units.degC
    ) * 100

    wet_bulb_values = []
    for index, row in df.iterrows():
        temp = row['temperature_smoothed']
        dewpoint = row['dewpoint_smoothed']
        pressure = row['pressure']
        if not np.isnan(temp) and not np.isnan(dewpoint) and not np.isnan(pressure):
            try:
                wet_bulb_temp = mpcalc.wet_bulb_temperature(
                    pressure * units.hPa, temp * units.degC, dewpoint * units.degC
                ).to('degF').m
                wet_bulb_values.append(wet_bulb_temp)
            except Exception:
                wet_bulb_values.append(np.nan)
        else:
            wet_bulb_values.append(np.nan)
    df['wet_bulb'] = wet_bulb_values

    return df

def to_heat_index(tempF, dewF):
    tempF = np.array(tempF)
    dewF = np.array(dewF)
    tempC = (tempF - 32) * 5 / 9
    dewC = (dewF - 32) * 5 / 9
    rh = mpcalc.relative_humidity_from_dewpoint(tempC * units.degC, dewC * units.degC) * 100
    rh = np.clip(rh.magnitude, 0, 100)
    c1, c2, c3 = -42.379, 2.04901523, 10.14333127
    c4, c5, c6 = -0.22475541, -6.83783 * (10 ** -3), -5.481717 * (10 ** -2)
    c7, c8, c9 = 1.22874 * (10 ** -3), 8.5282 * (10 ** -4), -1.99 * (10 ** -6)
    heat_index = (c1 + (c2 * tempF) + (c3 * rh) + (c4 * tempF * rh) +
                  (c5 * tempF ** 2) + (c6 * rh ** 2) + (c7 * tempF ** 2 * rh) +
                  (c8 * tempF * rh ** 2) + (c9 * tempF ** 2 * rh ** 2))
    heat_index = np.where(tempF >= 80, heat_index, np.nan)
    return heat_index

def to_wind_chill(tempF, wind_speed):
    wind_speed_mph = wind_speed * 1.15078
    wind_chill = np.where((tempF <= 50) & (wind_speed_mph > 3), 
                          35.74 + 0.6215 * tempF - 35.75 * wind_speed_mph**0.16 + 0.4275 * tempF * wind_speed_mph**0.16,
                          np.nan)
    return wind_chill

async def run_meteogram(args):
    """Generates a FrostByte-themed meteogram plot."""
    icao = args.icao.upper()
    hoursback = args.hoursback
    try:
        metar_list = get_metar(icao, hoursback)
        df = process_metar_data(metar_list)
        utc_time = datetime.now(pytz.utc)

        if df.empty:
            print("No valid METAR data available for plotting.")
            return

        df['tempF'] = (df['temperature_smoothed'] * 9 / 5) + 32
        df['dewF'] = (df['dewpoint_smoothed'] * 9 / 5) + 32
        df['heat_index'] = to_heat_index(df['tempF'].values, df['dewF'].values)
        df.loc[df['tempF'] < 80, 'heat_index'] = np.nan
        df['wind_chill'] = to_wind_chill(df['tempF'].values, df['wind_speed'].values)
        df.loc[(df['wind_speed'] <= 5) | (df['tempF'] > 50), 'wind_chill'] = np.nan

        max_temp = df['tempF'].max()
        min_temp = df['tempF'].min()
        
        # --- NEW PARAMETERS ---
        max_wet_bulb = df['wet_bulb'].max()
        min_wet_bulb = df['wet_bulb'].min()
        max_heat_index = df['heat_index'].max()
        min_heat_index = df['heat_index'].min()
        max_wind_chill = df['wind_chill'].max()
        min_wind_chill = df['wind_chill'].min()
        # ----------------------

        avg_wind_speed = df['wind_speed'].mean()
        if not df['wind_direction'].dropna().empty:
            wind_dirs_rad = np.deg2rad(df['wind_direction'].dropna())
            mean_sin = np.mean(np.sin(wind_dirs_rad))
            mean_cos = np.mean(np.cos(wind_dirs_rad))
            avg_wind_dir = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360
        else:
            avg_wind_dir = np.nan
        avg_rh = df['relative_humidity'].mean()
        max_gust = df['wind_gusts'].max() if 'wind_gusts' in df.columns and not df['wind_gusts'].isnull().all() else np.nan
        
        # --- FIXED PRECIPITATION AGGREGATION ---
        # Copy the dataframe to avoid SettingWithCopyWarning
        precip_data = df.dropna(subset=['hourly_precipitation_in']).copy()
        
        # Group by the hour and keep only the row with the maximum precipitation for that hour
        precip_data['hour'] = precip_data['time'].dt.floor('h')
        idx_to_keep = precip_data.groupby('hour')['hourly_precipitation_in'].idxmax()
        precip_data = precip_data.loc[idx_to_keep]

        total_rain = total_snow = total_freezing_rain = total_sleet = total_precip = 0
        for index, row in precip_data.iterrows():
            precip_amount = row['hourly_precipitation_in']
            weather_codes = row['present_weather_codes']
            
            if any(code in weather_codes for code in ['RA', '+RA', '-RA', 'SHRA', 'TSRA']): total_rain += precip_amount
            elif any(code in weather_codes for code in ['SN', '+SN', '-SN', 'SHSN']): total_snow += precip_amount
            elif any(code in weather_codes for code in ['FZRA', 'FZDZ']): total_freezing_rain += precip_amount
            elif any(code in weather_codes for code in ['PL', 'GS']): total_sleet += precip_amount
            total_precip += precip_amount
        # ----------------------------------------

        fig = plt.figure(figsize=(20, 20))
        fig.set_facecolor(FIG_BG_COLOR)
        
        gs = fig.add_gridspec(6, 2, width_ratios=[1, 0.03], wspace=0.05)
        axs = []
        for i in range(6):
            ax = fig.add_subplot(gs[i, 0])
            if i > 0: ax.sharex(axs[0])
            ax.set_facecolor(AXES_BG_COLOR)
            axs.append(ax)
        cax = fig.add_subplot(gs[1, 1])

        axs[0].set_title(
            f'Meteogram for {icao.upper()} - Last {hoursback} hours (Generated at: {utc_time.strftime("%Y-%m-%d %H:%M UTC")})',
            weight='bold', size='16', color='white', path_effects=TEXT_OUTLINE
        )

        # 0. Temperature Plots
        if not df[['tempF', 'dewF']].isnull().all().any():
            plt.sca(axs[0])
            plt.plot(df['time'], df['tempF'], label='Temperature (°F)', linewidth=3, color='#FF6B6B')
            plt.plot(df['time'], df['dewF'], label='Dewpoint (°F)', linewidth=3, color='#6BCB77')
            plt.plot(df['time'], df['wet_bulb'], label='Wet Bulb (°F)', linewidth=3, linestyle='dotted', color='#4D96FF')
            plt.plot(df['time'], df['heat_index'], label='Heat Index (°F)', linestyle='--', color='#C780FA')
            plt.plot(df['time'], df['wind_chill'], label='Wind Chill (°F)', linestyle='--', color='cyan')
            axs[0].set_ylabel('Temperature (°F)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
            legend0 = axs[0].legend(loc='upper left', fontsize=10, frameon=True, title='Temperature / Dewpoint')
            style_legend(legend0)
        else:
            axs[0].axhline(0, color='white')
            axs[0].set_ylabel('Temperature (°F)', color='white', weight='bold', path_effects=TEXT_OUTLINE)

        # 1. Wind Plot (Upgraded Gusts)
        valid_wind_data = df.dropna(subset=['wind_direction', 'wind_speed'], how='any')
        plot_wind = valid_wind_data[valid_wind_data['wind_speed'] >= 1]
        
        if not plot_wind.empty:
            # Filter for significant gusts
            gust_data = plot_wind.dropna(subset=['wind_gusts'])
            gust_data = gust_data[(gust_data['wind_gusts'] >= 10) & (gust_data['wind_gusts'] >= (gust_data['wind_speed'] + 5))]
            
            # Calculate max color scale across BOTH sustained and gusts so colors are accurate
            vmin = plot_wind['wind_speed'].min()
            vmax_sustained = plot_wind['wind_speed'].max()
            vmax_gusts = gust_data['wind_gusts'].max() if not gust_data.empty else vmax_sustained
            vmax = max(vmax_sustained, vmax_gusts)

            # Plot Sustained
            scatter = axs[1].scatter(plot_wind['time'], plot_wind['wind_direction'],
                                     c=plot_wind['wind_speed'], cmap='turbo', label='Wind Speed (knots)',
                                     s=150, zorder=2,
                                     vmin=vmin if vmin <= vmax else 0, vmax=vmax if vmin <= vmax else 10)
            
            # Plot Gusts
            if not gust_data.empty:
                # Color map the X to match the actual gust speed
                axs[1].scatter(gust_data['time'], gust_data['wind_direction'],
                               c=gust_data['wind_gusts'], cmap='turbo', marker='x', s=100, linewidth=2.5,
                               zorder=3,
                               vmin=vmin if vmin <= vmax else 0, vmax=vmax if vmin <= vmax else 10,
                               label='Wind Gusts (knots)')
                               
                # Add text annotation for the actual speed right next to the X
                for _, row in gust_data.iterrows():
                    y_pos = row['wind_direction']
                    # Shift text down if it's near the very top of the chart to prevent clipping
                    y_offset = -20 if y_pos > 330 else 18
                    axs[1].text(row['time'], y_pos + y_offset, f"{int(row['wind_gusts'])}",
                                color='white', fontsize=10, ha='center', va='center',
                                weight='bold', path_effects=TEXT_OUTLINE)
            
            cbar = fig.colorbar(scatter, cax=cax)
            cbar.set_label('Wind Speed (knots)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
            cbar.ax.tick_params(colors='white')
            for label in cbar.ax.get_yticklabels():
                label.set_path_effects(TEXT_OUTLINE)
                label.set_fontweight('bold')
                
            axs[1].set_yticks([0, 90, 180, 270, 360])
            axs[1].set_yticklabels(['N', 'E', 'S', 'W', 'N'])
            axs[1].set_ylim(0, 360)
            axs[1].set_ylabel('Wind Direction (°)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
            legend1 = axs[1].legend(loc='upper left', fontsize=10, frameon=True, title='Wind')
            style_legend(legend1)
        else:
            axs[1].axhline(0, color='white')
            axs[1].set_ylabel('Wind Direction (°)', color='white', weight='bold', path_effects=TEXT_OUTLINE)

        # 2. Pressure Plot
        if not df['pressure'].isnull().all():
            axs[2].plot(df['time'], df['pressure'], label='Pressure (hPa)', linewidth=3, color='#E066FF')
            axs[2].set_ylabel('Pressure (hPa)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
            axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            legend2 = axs[2].legend(loc='upper left', fontsize=10, frameon=True, title='Pressure')
            style_legend(legend2)
        else:
            axs[2].axhline(0, color='white')
            axs[2].set_ylabel('Pressure (hPa)', color='white', weight='bold', path_effects=TEXT_OUTLINE)

        # 3. Cloud Plot
        axs[3].scatter(df['time'], df['low_clouds'], color='#6BCB77', label='Low Clouds', marker='s')
        axs[3].scatter(df['time'], df['mid_clouds'], color='#4D96FF', label='Mid Clouds', marker='o')
        axs[3].scatter(df['time'], df['high_clouds'], color='#FF6B6B', label='High Clouds', marker='^')
        axs[3].scatter(df['time'], df['vertical_visibility'], color='#C780FA', label='Vertical Visibility', marker='v')
        axs[3].set_ylabel('Cloud Cover (ft)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
        legend3 = axs[3].legend(loc='upper left', fontsize=10, frameon=True, title='Cloud Cover')
        style_legend(legend3)
        axs[3].set_ylim(0)

        # 4. Precipitation Plot
        if not precip_data.empty:
            bar_width = timedelta(hours=0.8)
            for index, row in precip_data.iterrows():
                precip_amount = row['hourly_precipitation_in']
                weather_codes = row['present_weather_codes']
                timestamp = row['time']
                color = 'gray'
                if any(code in weather_codes for code in ['RA', '+RA', '-RA', 'SHRA', 'TSRA']): color = '#6BCB77'
                elif any(code in weather_codes for code in ['SN', '+SN', '-SN', 'SHSN']): color = '#4D96FF'
                elif any(code in weather_codes for code in ['FZRA', 'FZDZ']): color = '#C780FA'
                elif any(code in weather_codes for code in ['PL', 'GS']): color = '#FF6B6B'
                axs[4].bar(timestamp, precip_amount, width=bar_width, color=color, zorder=2)
            axs[4].set_ylabel('Precipitation (in)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
            axs[4].set_ylim(bottom=0)
            
            legend_elements = [
                Patch(facecolor='#6BCB77', edgecolor='white', label='Rain'),
                Patch(facecolor='#4D96FF', edgecolor='white', label='Snow'),
                Patch(facecolor='#C780FA', edgecolor='white', label='Freezing Rain'),
                Patch(facecolor='#FF6B6B', edgecolor='white', label='Sleet')
            ]
            legend4 = axs[4].legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True, title='Precipitation Types')
            style_legend(legend4)
        else:
            axs[4].axhline(0, color='white')
            axs[4].set_ylabel('Precipitation (in)', color='white', weight='bold', path_effects=TEXT_OUTLINE)

        icon_y_offset = 0.07
        icon_size = 0.07
        for index, row in df.iterrows():
            timestamp = row['time']
            weather_codes = row['present_weather_codes']
            if weather_codes and pd.notna(timestamp):
                icon_path = None
                icon_base_path = ICON_DIR 
                if any(code in weather_codes for code in ['RA', '+RA', '-RA', 'SHRA']): icon_path = os.path.join(icon_base_path, 'meteogram_rain_icon.png')
                elif any(code in weather_codes for code in ['SN', '+SN', '-SN', 'SHSN']): icon_path = os.path.join(icon_base_path, 'meteogram_snow_icon.png')
                elif any(code in weather_codes for code in ['TS', 'TSRA', '+TSRA']): icon_path = os.path.join(icon_base_path, 'meteogram_thunderstorm_icon.png')
                elif any(code in weather_codes for code in ['DZ', '+DZ', '-DZ']): icon_path = os.path.join(icon_base_path, 'meteogram_drizzle_icon.png')
                elif any(code in weather_codes for code in ['FZRA', 'FZDZ']): icon_path = os.path.join(icon_base_path, 'meteogram_freezing_rain_icon.png')

                if icon_path and os.path.exists(icon_path):
                    try:
                        img = plt.imread(icon_path)
                        im = OffsetImage(img, zoom=icon_size)
                        ab = AnnotationBbox(im, (mdates.date2num(timestamp), axs[4].get_ylim()[0] + icon_y_offset * (axs[4].get_ylim()[1] - axs[4].get_ylim()[0])),
                                            xycoords='data', frameon=False, box_alignment=(0.5, 0))
                        axs[4].add_artist(ab)
                    except Exception as img_err:
                        logging.warning(f"Error loading icon {icon_path}: {img_err}")

        # 5. Relative Humidity
        if not df['relative_humidity'].isnull().all():
            rh_array = df['relative_humidity'].values
            rh_band = 5
            rh_lower = np.maximum(0, rh_array - rh_band)
            rh_upper = np.minimum(100, rh_array + rh_band)
            axs[5].fill_between(df['time'], rh_lower, rh_upper, color='#4D96FF', alpha=0.3, label='RH Range')
            axs[5].plot(df['time'], rh_array, color='#4D96FF', label='Relative Humidity (%)', linewidth=2)
            axs[5].set_ylabel('Relative Humidity (%)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
            legend5 = axs[5].legend(loc='upper left', fontsize=10, frameon=True, title='Relative Humidity')
            style_legend(legend5)
            axs[5].set_ylim(0, 100)
        else:
            axs[5].axhline(0, color='white')
            axs[5].set_ylabel('Relative Humidity (%)', color='white', weight='bold', path_effects=TEXT_OUTLINE)

        # Summary Text Box
        if not df.empty:
            # Helper function to cleanly handle missing values (N/A) for HI/WC
            def fmt_t(val): return f"{val:.1f} °F" if pd.notna(val) else "N/A"
            
            # Widen the box by combining stats horizontally
            precip_text = (
                f"--- Meteogram Summary ---\n"
                f"Precipitation:  Sum {total_precip:.2f}\"  |  Rain {total_rain:.2f}\"  |  Snow {total_snow:.2f}\"  |  Ice {total_freezing_rain:.2f}\"  |  Sleet {total_sleet:.2f}\"\n"
                f"Temperature:  Max {fmt_t(max_temp)}  |  Min {fmt_t(min_temp)}\n"
                f"Wet Bulb:  Max {fmt_t(max_wet_bulb)}  |  Min {fmt_t(min_wet_bulb)}\n"
            )
            
            # Conditionally render Heat Index / Wind Chill lines only if they actually occurred
            if pd.notna(max_heat_index) or pd.notna(min_heat_index):
                precip_text += f"Heat Index:  Max {fmt_t(max_heat_index)}  |  Min {fmt_t(min_heat_index)}\n"
                
            if pd.notna(max_wind_chill) or pd.notna(min_wind_chill):
                precip_text += f"Wind Chill:  Max {fmt_t(max_wind_chill)}  |  Min {fmt_t(min_wind_chill)}\n"

            dir_str = f"{avg_wind_dir:.0f}°" if pd.notna(avg_wind_dir) else "N/A"
            precip_text += f"Average Wind:  {avg_wind_speed:.1f} knots @ {dir_str}"
            
            if not np.isnan(max_gust):
                precip_text += f"  |  Max Gust: {max_gust:.1f} knots"
                
            precip_text += f"\nAverage RH:  {avg_rh:.1f}%\n"

            # Moved Y coordinate down from -0.35 to -0.45 to fully clear the date tick labels
            axs[5].text(
                0.5, -0.45, precip_text, transform=axs[5].transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='center', color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=AXES_BG_COLOR, alpha=0.9, edgecolor='white'),
                path_effects=TEXT_OUTLINE
            )

        # Final Formatting
        for ax in axs:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
            ax.tick_params(axis='x', labelbottom=True, colors='white')
            ax.tick_params(axis='y', colors='white')
            plt.setp(ax.get_xticklabels(), path_effects=TEXT_OUTLINE, rotation=60, fontsize=9, ha='right', weight='bold')
            plt.setp(ax.get_yticklabels(), path_effects=TEXT_OUTLINE, weight='bold')
            
            ax.grid(True, color='#555555', linestyle=':', alpha=0.5, zorder=0)
            ymin, ymax = ax.get_ylim()
            y_ticks = ax.get_yticks()
            if len(y_ticks) > 1:
                y_ticks_filtered = [y for y in y_ticks if ymin <= y <= ymax]
                if len(y_ticks_filtered) > 1:
                    for i in range(len(y_ticks_filtered) - 1):
                        if i % 2 == 0:
                            ax.axhspan(y_ticks_filtered[i], y_ticks_filtered[i+1], color='#383838', alpha=0.4, zorder=0)

        gs.update(hspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.15)

        os.makedirs('outputs', exist_ok=True)
        filename = f'meteogram_{icao}_{utc_time.strftime("%Y%m%d_%H%M")}.png'
        out_path = os.path.join('outputs', filename)
        
        plt.savefig(out_path, format='png', dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        
        print(f"✅ Generated {out_path}")
        
        plt.close(fig)

    except Exception as e:
        print(f"An error occurred generating the meteogram: {str(e)}")
        logging.error(f"Error in meteogram command for {icao}: {traceback.format_exc()}")

# --- TAF Command ---
def get_taf_checkwx(icao):
    api_key = "c0eead33ce0a4403800f26f173"
    base_url = "https://api.checkwx.com/taf/"
    url = f"{base_url}{icao}/decoded"
    headers = {"X-API-Key": api_key}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            raw_taf = data["data"][0]["raw_text"]
            return raw_taf
        else:
            raise ValueError(f"No TAF data found for {icao}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching TAF data for {icao} from CheckWX: {e}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Error parsing TAF data for {icao} from CheckWX: {e}")

def run_taf(args):
    airport_code = args.airport_code.upper()
    try:
        raw_taf = get_taf_checkwx(airport_code)
        taf_sections = re.split(r'(BECMG|FM)', raw_taf)
        
        print(f"--- TAF for {airport_code} ---")
        current_section_name = "Initial"
        for i, section in enumerate(taf_sections):
            if section.strip():
                if section.startswith(('BECMG', 'FM')):
                    current_section_name = section.strip()
                else:
                    print(f"\n--- {current_section_name} ---")
                    print(f"{section.strip()}")
    except Exception as e:
        print(f"Sorry, there was an error fetching the TAF for {airport_code}: {e}")

# --- SIGMETs & AIRMETs Command ---
def get_aviation_weather_alerts_checkwx(icao):
    api_key = "c0eead33ce0a4403800f26f173" 
    base_url = "https://api.checkwx.com/"
    try:
        alerts = []
        for alert_type, endpoint in [("AIRMET", "airmet"), ("SIGMET", "sigmet")]:
            url = f"{base_url}{endpoint}/{icao}"
            response = requests.get(url, headers={"X-API-Key": api_key}, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                for alert in data["data"]:
                    if isinstance(alert, str):
                        alerts.append({
                            "type": alert_type, "raw_text": alert,
                            "valid_from": "N/A", "valid_to": "N/A"
                        })
                    else:
                        alerts.append({
                            "type": alert_type,
                            "raw_text": alert.get("raw_text", "N/A"),
                            "valid_from": alert.get("valid_from", "N/A"),
                            "valid_to": alert.get("valid_to", "N/A")
                        })
        return alerts
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching aviation alerts for {icao}: {e}")

def run_mets(args):
    airport_code = args.airport_code.upper()
    try:
        alerts = get_aviation_weather_alerts_checkwx(airport_code)
        if not alerts:
            print(f"No AIRMETs or SIGMETs found for {airport_code}.")
            return
        
        print(f"--- Aviation Alerts for {airport_code} ---")
        for alert in alerts:
            print(f"\n--- {alert['type']} ---")
            print(f"Valid From: {alert['valid_from']}  |  Valid To: {alert['valid_to']}")
            print(f"{alert['raw_text']}")
            print("-" * (len(alert['type']) + 4))
    except Exception as e:
        print(f"Error fetching aviation weather alerts for {airport_code}: {e}")

# --- Radar Command ---
def run_radar(args):
    region = args.region.lower()
    overlay = args.overlay.lower()
    
    try:
        valid_regions = ["chase", "ne", "se", "sw", "nw", "pr"]
        valid_overlays = ["base", "totals"]
        if region not in valid_regions:
            raise ValueError(f"Invalid region. Valid options are: {', '.join(valid_regions)}")
        if overlay not in valid_overlays:
            raise ValueError(f"Invalid overlay. Valid options are: {', '.join(valid_overlays)}")

        image_links = {
            ("chase", "base"): "https://tempest.aos.wisc.edu/radar/chase3comp.gif",
            ("chase", "totals"): "https://tempest.aos.wisc.edu/radar/chasePcomp.gif",
            ("ne", "base"): "https://tempest.aos.wisc.edu/radar/ne3comp.gif",
            ("ne", "totals"): "https://tempest.aos.wisc.edu/radar/nePcomp.gif",
            ("se", "base"): "https://tempest.aos.wisc.edu/radar/se3comp.gif",
            ("se", "totals"): "https://tempest.aos.wisc.edu/radar/sePcomp.gif",
            ("sw", "base"): "https://tempest.aos.wisc.edu/radar/sw3comp.gif",
            ("sw", "totals"): "https://tempest.aos.wisc.edu/radar/swPcomp.gif",
            ("nw", "base"): "https://tempest.aos.wisc.edu/radar/nw3comp.gif",
            ("nw", "totals"): "https://tempest.aos.wisc.edu/radar/nwPcomp.gif",
	        ("pr", "base"): "https://tempest.aos.wisc.edu/radar/pr3comp.gif",
            ("pr", "totals"): "https://tempest.aos.wisc.edu/radar/prPcomp.gif"
        }
        image_url = image_links.get((region, overlay))

        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        os.makedirs('outputs', exist_ok=True)
        filename = os.path.join('outputs', f"radar_{region}_{overlay}.gif")
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Radar image saved to {filename}")

    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error retrieving radar image: {e}")

# --- Alerts Command ---
def run_alerts(args):
    state_abbr = args.state_abbr.upper()
    try:
        alerts_url = f"https://api.weather.gov/alerts/active?area={state_abbr}"
        response = requests.get(alerts_url, timeout=10)
        response.raise_for_status()
        alerts_data = response.json()
        
        filtered_alerts = [
            alert for alert in alerts_data.get('features', [])
            if alert.get('properties') and alert['properties'].get('event') and alert['properties'].get('severity')
        ]
        
        if filtered_alerts:
            print(f"--- Active Weather Alerts for {state_abbr} ---")
            for alert in filtered_alerts:
                properties = alert['properties']
                print(f"\n--- {properties['headline']} ---")
                
                cleaned_area_desc = "".join(properties['areaDesc']).split(";")
                cleaned_area_desc = [area.strip() for area in cleaned_area_desc if area.strip()]
                area_desc = ", ".join(cleaned_area_desc)
                
                print(f"Event: {properties.get('event', 'N/A')}")
                print(f"Severity: {properties.get('severity', 'N/A')}")
                print(f"Area: {area_desc}")
                print(f"Description: {properties.get('description', 'N/A')}")
                print("-" * (len(properties['headline']) + 8))
        else:
            print(f"No weather alerts found for {state_abbr}.")
    except Exception as e:
        print(f"Error fetching alerts: {e}")

# --- Forecast Command ---
async def run_forecast(args):
    location = args.location
    try:
        lat, lon = None, None
        location_data = {
            "KATL": (33.6367, -84.4281), "KJFK": (40.6398, -73.7789),
            "KBIX": (30.4103, -88.9261), "KVQQ": (30.2264, -81.8878),
            "KVPC": (34.1561, -84.7983), "KRMG": (34.4956, -85.2214),
            "KMGE": (33.9131, -84.5197), "KGPT": (30.4075, -89.0753),
            "KPIT": (40.4915, -80.2329), "KSGJ": (34.25, -84.95),
            "KPEZ": (40.3073, -75.6192), "KNSE": (30.72247, -87.02390),
            "KTPA": (27.9755, -82.5332), "30184": (34.1561, -84.7983),
            "30303": (33.7525, -84.3922),
        }

        if '/' in location:
            try:
                lat, lon = map(float, location.split('/'))
            except ValueError:
                print("Invalid lat/lon format. Use 'lat/lon' (e.g., '34.05/-118.25').")
                return
        elif location.upper() in location_data or location in location_data:
            lat, lon = location_data.get(location.upper(), location_data.get(location))
        elif location.isdigit() and len(location) == 5:
            geonames_url = f"http://api.geonames.org/postalCodeSearchJSON?postalcode={location}&country=US&username=freeuser&maxRows=1"
            async with aiohttp.ClientSession() as session:
                async with session.get(geonames_url, headers={'User-Agent': 'CLWeatherScript/2.1'}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('postalCodes'):
                            lat = data['postalCodes'][0]['lat']
                            lon = data['postalCodes'][0]['lng']
                        else:
                            print(f"Could not find ZIP code: {location}.")
                            return
                    else:
                        print(f"Could not find ZIP code: {location}.")
                        return
        else:
            try:
                geolocator = Nominatim(user_agent="weather_script_user")
                loc = geolocator.geocode(location)
                if loc:
                    lat, lon = loc.latitude, loc.longitude
                else:
                    raise ValueError
            except Exception:
                print(f"Could not find location: {location}. Try a city name (e.g., 'Cartersville, GA'), ZIP code, ICAO code, or lat/lon.")
                return

        nws_zone_url = f"https://api.weather.gov/points/{lat},{lon}"
        async with aiohttp.ClientSession() as session:
            async with session.get(nws_zone_url, headers={'User-Agent': 'CLWeatherScript/2.1'}) as response:
                if response.status == 200:
                    nws_data = await response.json()
                    zone_id = nws_data['properties']['forecastZone'].split('/')[-1]
                    state_code = zone_id[:2].lower()
                    zone_code = zone_id.lower()
                else:
                    print("Could not retrieve weather zone for this location.")
                    return

        forecast_url = f"https://tgftp.nws.noaa.gov/data/forecasts/zone/{state_code}/{zone_code}.txt"
        response_forecast = requests.get(forecast_url, timeout=10)
        if response_forecast.status_code != 200:
            print(f"Could not retrieve forecast for zone {zone_id}.")
            return

        forecast_text = response_forecast.text
        
        print(f"--- Weather Forecast for {location.title()} (Zone: {zone_id}) ---")
        print(forecast_text)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# --- Time Command ---
def run_worldtimes(args):
    print(f"--- World Times ---")
    utc_now = pytz.utc.localize(datetime.utcnow())
    
    us_timezones = {
        "Hawaii": "Pacific/Honolulu", "Alaska": "America/Anchorage",
        "Pacific": "America/Los_Angeles", "Mountain": "America/Denver",
        "Central": "America/Chicago", "Eastern": "America/New_York"
    }
    international_timezones = {
        "London": "Europe/London", "Berlin": "Europe/Berlin",
        "Tokyo": "Asia/Tokyo", "Sydney": "Australia/Sydney",
        "Tehran (Iran)": "Asia/Tehran", "Jerusalem (Israel)": "Asia/Jerusalem",
        "Moscow": "Europe/Moscow", "Beijing": "Asia/Shanghai"
    }
    
    all_times = {}
    for region, timezone_str in us_timezones.items():
        timezone = pytz.timezone(timezone_str)
        local_time = utc_now.astimezone(timezone)
        all_times[f"{region} (US)"] = local_time.strftime('%H:%M:%S')

    for city, timezone_str in international_timezones.items():
        timezone = pytz.timezone(timezone_str)
        local_time = utc_now.astimezone(timezone)
        all_times[city] = local_time.strftime('%H:%M:%S')
    
    for name, time_str in all_times.items():
        print(f"{name:<20} {time_str}")

# --- Convert Command ---
conversion_factors = {
    "C": {"F": lambda x: (9/5) * x + 32, "K": lambda x: x + 273.15},
    "F": {"C": lambda x: (5/9) * (x - 32), "K": lambda x: (5/9) * (x - 32) + 273.15},
    "K": {"C": lambda x: x - 273.15, "F": lambda x: (9/5) * (x - 273.15) + 32},
    "M": {"KM": lambda x: x * 0.001, "FT": lambda x: x * 3.28084, "MI": lambda x: x * 0.000621371, "NM": lambda x: x * 0.000539957,
          "CM": lambda x: x * 100, "MM": lambda x: x * 1000, "IN": lambda x: x * 39.3701},
    "KM": {"M": lambda x: x * 1000, "FT": lambda x: x * 3280.84, "MI": lambda x: x * 0.621371, "NM": lambda x: x * 0.539957},
    "FT": {"M": lambda x: x * 0.3048, "KM": lambda x: x * 0.0003048, "MI": lambda x: x * 0.000189394, "NM": lambda x: x * 0.000164579,
          "CM": lambda x: x * 30.48, "MM": lambda x: x * 304.8, "IN": lambda x: x * 12},
    "MI": {"M": lambda x: x * 1609.34, "KM": lambda x: x * 1.60934, "FT": lambda x: x * 5280, "NM": lambda x: x * 0.868976},
    "NM": {"M": lambda x: x * 1852, "KM": lambda x: x * 1.852, "FT": lambda x: x * 6076.12, "MI": lambda x: x * 1.15078},
    "AU": {"KM": lambda x: x * 149597870.7, "M": lambda x: x * 149597870700},
    "LY": {"KM": lambda x: x * 9.461e+12, "M": lambda x: x * 9.461e+15},
    "PC": {"LY": lambda x: x * 3.26156, "AU": lambda x: x * 206264.806},
    "CM": {"M": lambda x: x / 100, "IN": lambda x: x / 2.54, "MM": lambda x: x * 10},
    "MM": {"M": lambda x: x / 1000, "IN": lambda x: x / 25.4, "CM": lambda x: x / 10},
    "IN": {"M": lambda x: x / 39.3701, "CM": lambda x: x * 2.54, "MM": lambda x: x * 25.4},
    "ACRE": {"M^2": lambda x: x * 4046.86, "KM^2": lambda x: x * 0.00404686, "HA": lambda x: x * 0.404686},
    "MI^2": {"KM^2": lambda x: x * 2.58999},
    "GAL_US": {"L": lambda x: x * 3.78541}, "GAL_IMP": {"L": lambda x: x * 4.54609},
    "FT^3": {"M^3": lambda x: x * 0.0283168}, "BBL": {"L": lambda x: x * 158.987, "GAL_US": lambda x: x * 42},
    "LB": {"KG": lambda x: x * 0.453592}, "OZ": {"G": lambda x: x * 28.3495},
    "TON_US": {"KG": lambda x: x * 907.185}, "TONNE": {"KG": lambda x: x * 1000},
    "N": {"LBF": lambda x: x * 0.224809}, "W": {"HP": lambda x: x * 0.00134102}, "HP": {"W": lambda x: x / 0.00134102},
    "N_M": {"LB_FT": lambda x: x * 0.737562}, "LB_FT": {"N_M": lambda x: x / 0.737562},
    "HP_TO_TORQUE": lambda hp, rpm: (hp * 5252) / rpm,
    "TORQUE_TO_HP": lambda torque, rpm: (torque * rpm) / 5252,
    "KT": {"MPH": lambda x: x * 1.15078, "KPH": lambda x: x * 1.852, "M/S": lambda x: x * 0.514444},
    "MPH": {"KT": lambda x: x / 1.15078, "KPH": lambda x: x * 1.609344, "M/S": lambda x: x * 0.44704},
    "KPH": {"MPH": lambda x: x / 1.609344, "KT": lambda x: x / 1.852, "M/S": lambda x: x * 0.277778},
    "M/S": {"MPH": lambda x: x / 0.44704, "KPH": lambda x: x / 0.277778, "KT": lambda x: x / 0.514444},
    "ATM": {"PA": lambda x: x * 101325, "KPA": lambda x: x * 101.325, "BAR": lambda x: x * 1.01325, "MB": lambda x: x * 1013.25, "INHG": lambda x: x * 29.92126},
    "INHG": {"MMHG": lambda x: x * 25.4, "PA": lambda x: x * 3386.38816, "ATM": lambda x: x / 29.92126, "MB": lambda x: x * 33.863886666667},
    "PSI": {"PA": lambda x: x * 6894.76, "KPA": lambda x: x * 6.89476},
    "MB": {"INHG": lambda x: x / 33.863886666667},
    "J": {"CAL": lambda x: x * 0.239006, "KCAL": lambda x: x * 0.000239006, "BTU": lambda x: x * 0.000947817},
    "EV": {"J": lambda x: x * 1.60218e-19},
    "DEG": {"RAD": lambda x: math.radians(x)}, "RAD": {"DEG": lambda x: math.degrees(x)},
    "SIN_DEG": {"": lambda x: math.sin(math.radians(x))},
    "COS_DEG": {"": lambda x: math.cos(math.radians(x))},
    "TAN_DEG": {"": lambda x: math.tan(math.radians(x))},
    "DERIVATIVE_AT": {"": lambda coeffs, point: sum(c * i * point**(i-1) for i, c in enumerate(coeffs) if i > 0)}
}

def run_convert(args_list):
    try:
        if len(args_list) == 3 and args_list[0] == '1':
            hp, rpm = float(args_list[1]), float(args_list[2])
            torque = conversion_factors["HP_TO_TORQUE"](hp, rpm)
            print("--- Conversion Result ---")
            print(f"Horsepower: {hp} HP")
            print(f"RPM: {rpm} RPM")
            print(f"Torque: {torque:.4f} lb-ft")
        elif len(args_list) == 3 and args_list[0] == '2':
            torque, rpm = float(args_list[1]), float(args_list[2])
            hp = conversion_factors["TORQUE_TO_HP"](torque, rpm)
            print("--- Conversion Result ---")
            print(f"Torque: {torque} lb-ft")
            print(f"RPM: {rpm} RPM")
            print(f"Horsepower: {hp:.4f} HP")
        elif len(args_list) == 3:
            value, from_unit, to_unit = args_list
            value = float(value)
            from_unit = from_unit.upper()
            to_unit = to_unit.upper()
            if from_unit in conversion_factors and to_unit in conversion_factors[from_unit]:
                converted_value = conversion_factors[from_unit][to_unit](value)
                print("--- Conversion Result ---")
                print(f"Original Value: {value} {from_unit}")
                print(f"Converted Value: {converted_value:.4f} {to_unit}")
            else:
                print("Unsupported conversion or units. Please check your input.")
        else:
            print("Invalid conversion format. Use: <value> <from_unit> <to_unit> OR 1 <hp> <rpm> OR 2 <torque> <rpm>")
    except Exception as e:
        print(f"Error during conversion: {e}")


# Section 5: Main execution
def main():
    parser = argparse.ArgumentParser(
        description="Standalone Weather Bot Script",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="The command to run.")

    metar_parser = subparsers.add_parser('metar', help='Fetches METAR for an airport.')
    metar_parser.add_argument('airport_code', type=str, help='ICAO airport code (e.g., KATL).')
    metar_parser.add_argument('--hoursback', type=int, default=0, help='Hours of previous METARs (default: 0 for latest).')

    meteo_parser = subparsers.add_parser('meteogram', help='Generates a meteogram for an airport.')
    meteo_parser.add_argument('icao', type=str, help='ICAO airport code (e.g., KFFC).')
    meteo_parser.add_argument('hoursback', type=int, help='Number of hours of history to show (e.g., 72).')

    taf_parser = subparsers.add_parser('taf', help='Fetches TAF for an airport.')
    taf_parser.add_argument('airport_code', type=str, help='ICAO airport code (e.g., KATL).')

    mets_parser = subparsers.add_parser('mets', help='Fetches AIRMETs/SIGMETs for an airport.')
    mets_parser.add_argument('airport_code', type=str, help='ICAO airport code (e.g., KATL).')

    radar_parser = subparsers.add_parser('radar', help='Fetches a radar image.')
    radar_parser.add_argument('region', type=str, nargs='?', default='chase', help="Region (chase, ne, se, sw, nw, pr). Default: chase.")
    radar_parser.add_argument('overlay', type=str, nargs='?', default='base', help="Overlay (base, totals). Default: base.")

    alerts_parser = subparsers.add_parser('alerts', help='Fetches NWS alerts for a state.')
    alerts_parser.add_argument('state_abbr', type=str, help='Two-letter state abbreviation (e.g., GA, FL, MT).')

    forecast_parser = subparsers.add_parser('forecast', help='Fetches NWS zone forecast.')
    forecast_parser.add_argument('location', type=str, help="Location query (ICAO, ZIP, 'City, ST', or 'lat/lon').")

    utc_parser = subparsers.add_parser('utc', help='Shows current times in various time zones.')

    convert_parser = subparsers.add_parser('convert', help='Converts units.')
    convert_parser.add_argument('args', nargs='+', help='Arguments for conversion.')

    args = parser.parse_args()

    try:
        if args.command == 'metar':
            run_metar(args)
        elif args.command == 'meteogram':
            asyncio.run(run_meteogram(args))
        elif args.command == 'taf':
            run_taf(args)
        elif args.command == 'mets':
            run_mets(args)
        elif args.command == 'radar':
            run_radar(args)
        elif args.command == 'alerts':
            run_alerts(args)
        elif args.command == 'forecast':
            asyncio.run(run_forecast(args))
        elif args.command == 'utc':
            run_worldtimes(args)
        elif args.command == 'convert':
            run_convert(args.args)
    except Exception as e:
        logging.error(f"Failed to run command '{args.command}': {e}", exc_info=True)
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if not os.path.exists(INTERSTATES_SHP):
        logging.warning(f"Interstate shapefile not found at: {INTERSTATES_SHP}. Interstate overlays will be skipped.")
    if not os.path.exists(ICON_DIR):
        logging.warning(f"Icon directory not found at: {ICON_DIR}. Meteogram icons will be missing.")
    
    main()