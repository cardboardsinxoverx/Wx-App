import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import requests
import pytz
import logging
from datetime import datetime, timezone, timedelta
import aiohttp
import json
import re
from geopy.geocoders import Nominatim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from bs4 import BeautifulSoup
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metpy.calc as mpcalc
from metpy.units import units
from wx_box.weatherpy.data_access import NOMADS_OPENDAP_Downloads, UCAR_THREDDS_SERVER_OPENDAP_Downloads
from wx_box.weatherpy.plotting import (
    plot_relative_humidity, plot_24_hour_relative_humidity_comparison,
    plot_temperature, plot_dry_and_gusty_areas,
    plot_relative_humidity_with_metar_obs, plot_low_relative_humidity_with_metar_obs
)
from wx_box.weatherpy.settings import get_metar_mask
from wx_box.weatherpy.utilities import OUTPUT_DIR
from wx_box.au_weather_maps import (
    au_wind300, au_wind500, au_vort500, au_fronto700, au_rh700, au_wind850, au_dew850,
    au_mAdv850, au_tAdv850, au_mslp_temp, au_divcon300, au_thermal_wind
)
from wx_box.weather_maps import (
    wind300, wind500, vort500, rh700, fronto700, wind850, dew850, mAdv850,
    tAdv850, mslp_temp, divcon300
)
from wx_box.eu_weather_maps import (
    eu_wind300, eu_wind500, eu_vort500, eu_rh700, eu_wind850, eu_dew850,
    eu_mAdv850, eu_tAdv850, eu_mslp_temp, eu_divcon300
)
from wx_box.skewt import skewt
from wx_box.utils import parse_date
from wx_box.weather_calculations import calc_mslp
import math
import ephem
from timezonefinder import TimezoneFinder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Keys
OPENWEATHERMAP_API_KEY = 'efd4f5ec6d2b16958a946b2ceb0419a6'
WEATHERBIT_API_KEY = '4759188e762b4184828c575b5d8df034'

# --- METAR Command ---
def get_metar(icao, hoursback=0, format='json'):
    try:
        metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format={format}&hours={hoursback}'
        src = requests.get(metar_url).content
        json_data = json.loads(src)
        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")
        raw_metars = [entry['rawOb'] for entry in json_data]
        return raw_metars
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching METAR data for {icao}: {e}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Error parsing METAR data for {icao}: {e}")

def metar_command(icao, hoursback=0):
    icao = icao.upper()
    try:
        raw_metars = get_metar(icao, hoursback)
        if hoursback > 0:
            print(f"METARs for {icao} (Last {hoursback} Hours):")
            for i, raw_metar in enumerate(raw_metars):
                print(f"Observation {i+1}: {raw_metar}")
        else:
            print(f"METAR for {icao}: {raw_metars[0]}")
        logger.info(f"Requested METAR for {icao} (hoursback={hoursback})")
    except Exception as e:
        print(f"Error fetching METAR: {e}")
        logger.error(f"Error fetching METAR: {e}")

# --- TAF Command ---
def get_taf(icao):
    try:
        taf_url = f'https://aviationweather.gov/api/data/taf?ids={icao}&format=json'
        src = requests.get(taf_url).content
        json_data = json.loads(src)
        if not json_data:
            raise ValueError(f"No TAF data found for {icao}.")
        raw_tafs = [entry['rawTAF'] for entry in json_data]
        return raw_tafs
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching TAF data for {icao}: {e}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Error parsing TAF data for {icao}: {e}")

def taf_command(icao):
    icao = icao.upper()
    try:
        raw_tafs = get_taf(icao)
        print(f"TAF for {icao}:\n{raw_tafs[0]}")
        logger.info(f"Requested TAF for {icao}")
    except Exception as e:
        print(f"Error fetching TAF: {e}")
        logger.error(f"Error fetching TAF: {e}")

# --- Meteogram Utility Functions ---
def extract_cloud_info(metar):
    cloud_levels = {
        "low": [],
        "mid": [],
        "high": [],
        "vertical_visibility": None
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

def convert_pressure(altimeter_str):
    return float(altimeter_str[1:]) / 100

def extract_wind_info(metar):
    wind_direction = -999
    wind_speed = -999
    wind_gusts = np.nan
    wind_match = re.search(r'(\d{3})(\d{2})(G\d{2})?KT', metar)
    if wind_match:
        wind_direction = int(wind_match.group(1))
        wind_speed = int(wind_match.group(2))
        if wind_match.group(3):
            wind_gusts = int(wind_match.group(3)[1:])
    wind_speed = max(wind_speed, 0)
    wind_gusts = max(wind_gusts, 0) if not np.isnan(wind_gusts) else np.nan
    return wind_direction, wind_speed, wind_gusts

PRECIP_CODES = ['DZ', 'RA', 'SN', 'SG', 'IC', 'PL', 'GR', 'GS', 'UP', 'SH', 'TS', 'FZ']
PRECIP_REGEX = r'(?<!VC)([+-]?(?:' + '|'.join(PRECIP_CODES) + r'))'

def process_metar_data(metar_list):
    data = {
        "time": [],
        "temperature": [],
        "dewpoint": [],
        "wind_direction": [],
        "wind_speed": [],
        "wind_gusts": [],
        "pressure": [],
        "low_clouds": [],
        "mid_clouds": [],
        "high_clouds": [],
        "vertical_visibility": [],
        "present_weather_codes": [],
        "hourly_precipitation_in": []
    }
    now_utc = datetime.now(timezone.utc)
    for metar in metar_list:
        print(f"Processing METAR: {metar}")
        parts = metar.split()
        try:
            observation_day = int(parts[1][0:2])
            observation_hour = int(parts[1][2:4])
            observation_minute = int(parts[1][4:6])
            temp_dt = now_utc.replace(day=observation_day, hour=observation_hour, minute=observation_minute, second=0, microsecond=0)
            if temp_dt > now_utc + timedelta(days=1):
                if temp_dt.month == 1:
                    temp_dt = temp_dt.replace(year=temp_dt.year - 1, month=12)
                else:
                    temp_dt = temp_dt.replace(month=temp_dt.month - 1)
            data["time"].append(temp_dt)
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse time from METAR '{metar}': {e}")
            data["time"].append(np.nan)
            data["temperature"].append(np.nan)
            data["dewpoint"].append(np.nan)
            data["wind_direction"].append(np.nan)
            data["wind_speed"].append(np.nan)
            data["wind_gusts"].append(np.nan)
            data["pressure"].append(np.nan)
            data["low_clouds"].append(np.nan)
            data["mid_clouds"].append(np.nan)
            data["high_clouds"].append(np.nan)
            data["vertical_visibility"].append(np.nan)
            data["present_weather_codes"].append([])
            data["hourly_precipitation_in"].append(np.nan)
            continue
        temp_dewpoint_match = re.search(r'([M]?\d{1,2})/([M]?\d{1,2})', metar)
        if temp_dewpoint_match:
            temp_str, dewpoint_str = temp_dewpoint_match.groups()
            temp = int(temp_str.replace('M', '-')) if temp_str else np.nan
            dewpoint = int(dewpoint_str.replace('M', '-')) if dewpoint_str else np.nan
            if not np.isnan(temp) and not np.isnan(dewpoint) and dewpoint > temp:
                dewpoint = temp
            data["temperature"].append(temp)
            data["dewpoint"].append(dewpoint)
        else:
            data["temperature"].append(np.nan)
            data["dewpoint"].append(np.nan)
        direction, speed, gusts = extract_wind_info(metar)
        data["wind_direction"].append(direction if direction != -999 else np.nan)
        data["wind_speed"].append(speed if speed != -999 else np.nan)
        data["wind_gusts"].append(gusts if gusts != -999 else np.nan)
        pressure_match = re.search(r'A(\d{4})', metar)
        pressure_inhg = float(pressure_match.group(1)) / 100 if pressure_match else np.nan
        pressure_hpa = pressure_inhg * 33.8639 if not np.isnan(pressure_inhg) else np.nan
        data["pressure"].append(pressure_hpa)
        cloud_info = extract_cloud_info(metar)
        low_clouds = cloud_info["low"][0][1] if cloud_info["low"] else np.nan
        mid_clouds = cloud_info["mid"][0][1] if cloud_info["mid"] else np.nan
        high_clouds = cloud_info["high"][0][1] if cloud_info["high"] else np.nan
        vertical_visibility = cloud_info["vertical_visibility"] if cloud_info["vertical_visibility"] else np.nan
        data["low_clouds"].append(low_clouds)
        data["mid_clouds"].append(mid_clouds)
        data["high_clouds"].append(high_clouds)
        data["vertical_visibility"].append(vertical_visibility)
        present_weather_matches = re.findall(PRECIP_REGEX, metar)
        valid_present_weather = [pw for pw in present_weather_matches if pw and any(code in pw for code in PRECIP_CODES)]
        data["present_weather_codes"].append(valid_present_weather)
        hourly_precip_match = re.search(r'P(\d{4})', metar)
        if hourly_precip_match:
            hourly_precip = float(hourly_precip_match.group(1)) / 100
            data["hourly_precipitation_in"].append(hourly_precip)
        else:
            data["hourly_precipitation_in"].append(np.nan)
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df.sort_values(by='time', inplace=True)
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['dewpoint'] = pd.to_numeric(df['dewpoint'], errors='coerce')
    df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce')
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
    print("Relative Humidity (%):", df['relative_humidity'].values)
    wet_bulb_values = []
    for index, row in df.iterrows():
        temp = row['temperature_smoothed']
        dewpoint = row['dewpoint_smoothed']
        pressure = row['pressure']
        if not np.isnan(temp) and not np.isnan(dewpoint) and not np.isnan(pressure):
            try:
                wet_bulb_temp = mpcalc.wet_bulb_temperature(
                    pressure * units.hPa,
                    temp * units.degC,
                    dewpoint * units.degC
                ).to('degF').m
                wet_bulb_values.append(wet_bulb_temp)
            except Exception as e:
                print(f"Error calculating wet bulb temperature at index {index}: {e}")
                wet_bulb_values.append(np.nan)
        else:
            wet_bulb_values.append(np.nan)
    df['wet_bulb'] = wet_bulb_values
    print("DataFrame summary before plotting:")
    print(df[['time', 'temperature', 'dewpoint', 'pressure', 'relative_humidity', 'wet_bulb', 'present_weather_codes', 'hourly_precipitation_in']])
    return df

def to_heat_index(tempF, dewF):
    tempF = np.array(tempF)
    dewF = np.array(dewF)
    tempC = (tempF - 32) * 5 / 9
    dewC = (dewF - 32) * 5 / 9
    rh = mpcalc.relative_humidity_from_dewpoint(tempC * units.degC, dewC * units.degC) * 100
    rh = np.clip(rh.magnitude, 0, 100)
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783 * (10 ** -3)
    c6 = -5.481717 * (10 ** -2)
    c7 = 1.22874 * (10 ** -3)
    c8 = 8.5282 * (10 ** -4)
    c9 = -1.99 * (10 ** -6)
    heat_index = (c1 + (c2 * tempF) + (c3 * rh) + (c4 * tempF * rh) +
                  (c5 * tempF ** 2) + (c6 * rh ** 2) + (c7 * tempF ** 2 * rh) +
                  (c8 * tempF * rh ** 2) + (c9 * tempF ** 2 * rh ** 2))
    heat_index = np.where(tempF >= 80, heat_index, np.nan)
    return heat_index

def to_wind_chill(tempF, wind_speed):
    wind_speed_mph = wind_speed * 1.15078
    wind_chill = np.where((tempF <= 50) & (wind_speed > 5),
                          35.74 + 0.6215 * tempF - 35.75 * wind_speed_mph**0.16 + 0.4275 * tempF * wind_speed_mph**0.16,
                          np.nan)
    return wind_chill

async def meteogram_command(icao, hoursback):
    try:
        metar_list = get_metar(icao, hoursback)
        df = process_metar_data(metar_list)
        utc_time = datetime.now(pytz.utc)
        if df.empty:
            print("No valid METAR data available for plotting.")
            return None
        df['tempF'] = (df['temperature_smoothed'] * 9 / 5) + 32
        df['dewF'] = (df['dewpoint_smoothed'] * 9 / 5) + 32
        df['heat_index'] = to_heat_index(df['tempF'].values, df['dewF'].values)
        df.loc[df['tempF'] < 80, 'heat_index'] = np.nan
        df['wind_chill'] = to_wind_chill(df['tempF'].values, df['wind_speed'].values)
        df.loc[(df['wind_speed'] <= 5) | (df['tempF'] > 50), 'wind_chill'] = np.nan
        if not df.empty:
            max_temp = df['tempF'].max()
            min_temp = df['tempF'].min()
            avg_wind_speed = df['wind_speed'].mean()
            if not df['wind_direction'].dropna().empty:
                wind_dirs_rad = np.deg2rad(df['wind_direction'].dropna())
                mean_sin = np.mean(np.sin(wind_dirs_rad))
                mean_cos = np.mean(np.cos(wind_dirs_rad))
                avg_wind_dir = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360
            else:
                avg_wind_dir = np.nan
            avg_rh = df['relative_humidity'].mean()
            max_gust = df['wind_gusts'].max() if 'wind_gusts' in df.columns else np.nan
            precip_data = df.dropna(subset=['hourly_precipitation_in'])
            total_rain = total_snow = total_freezing_rain = total_sleet = total_precip = 0
            for index, row in precip_data.iterrows():
                precip_amount = row['hourly_precipitation_in']
                weather_codes = row['present_weather_codes']
                if any(code in weather_codes for code in ['RA', '+RA', '-RA', 'SHRA', 'TSRA']):
                    total_rain += precip_amount
                elif any(code in weather_codes for code in ['SN', '+SN', '-SN', 'SHSN']):
                    total_snow += precip_amount
                elif any(code in weather_codes for code in ['FZRA', 'FZDZ']):
                    total_freezing_rain += precip_amount
                elif any(code in weather_codes for code in ['PL', 'GS']):
                    total_sleet += precip_amount
                total_precip += precip_amount
        else:
            max_temp = min_temp = avg_wind_speed = avg_wind_dir = avg_rh = max_gust = total_rain = total_snow = total_freezing_rain = total_sleet = total_precip = np.nan
        fig, axs = plt.subplots(6, 1, figsize=(20, 20), sharex=True)
        fig.patch.set_facecolor('lightsteelblue')
        for ax in axs:
            ax.grid(True)
        axs[0].set_title(f'Meteogram for {icao.upper()} - Last {hoursback} hours (Generated at: {utc_time.strftime("%Y-%m-%d %H:%M UTC")})', weight='bold', size='16')
        if not df[['tempF', 'dewF']].isnull().all().any():
            plt.sca(axs[0])
            plt.plot(df['time'], df['tempF'], label='Temperature (°F)', linewidth=3, color='tab:red')
            plt.plot(df['time'], df['dewF'], label='Dewpoint (°F)', linewidth=3, color='tab:green')
            plt.plot(df['time'], df['wet_bulb'], label='Wet Bulb (°F)', linewidth=3, linestyle='dotted', color='tab:blue')
            plt.plot(df['time'], df['heat_index'], label='Heat Index (°F)', linestyle='--', color='tab:orange')
            plt.plot(df['time'], df['wind_chill'], label='Wind Chill (°F)', linestyle='--', color='tab:purple')
            axs[0].set_ylabel('Temperature (°F)')
            axs[0].legend(loc='upper left', fontsize=10, frameon=True, title='Temperature / Dewpoint')
        else:
            axs[0].axhline(0, color='black')
            axs[0].set_ylabel('Temperature (°F)')
        valid_wind_data = df.dropna(subset=['wind_direction', 'wind_speed'], how='any')
        if not valid_wind_data.empty:
            vmin = valid_wind_data['wind_speed'].min()
            vmax = valid_wind_data['wind_speed'].max()
            scatter = axs[1].scatter(valid_wind_data['time'], valid_wind_data['wind_direction'],
                                     c=valid_wind_data['wind_speed'], cmap='brg', label='Wind Speed (knots)',
                                     vmin=vmin if vmin <= vmax else 0, vmax=vmax if vmin <= vmax else 10)
            gust_data = valid_wind_data.dropna(subset=['wind_gusts'])
            if not gust_data.empty:
                axs[1].scatter(gust_data['time'], gust_data['wind_direction'],
                               c=gust_data['wind_gusts'], cmap='Greys', marker='x', s=100,
                               label='Wind Gusts (knots)', vmin=vmin if vmin <= vmax else 0, vmax=vmax if vmin <= vmax else 10)
            cbar = plt.colorbar(scatter, ax=axs[1], pad=0.002)
            cbar.set_label('Wind Speed (knots)')
            axs[1].set_yticks([0, 90, 180, 270, 360])
            axs[1].set_yticklabels(['N', 'E', 'S', 'W', 'N'])
            axs[1].set_ylim(0, 360)
            axs[1].set_ylabel('Wind Direction (°)')
            axs[1].legend(loc='upper left', fontsize=10, frameon=True, title='Wind')
        else:
            axs[1].axhline(0, color='black')
            axs[1].set_ylabel('Wind Direction (°)')
        if not df['pressure'].isnull().all():
            axs[2].plot(df['time'], df['pressure'], label='Pressure (hPa)', linewidth=3, color='tab:purple')
            axs[2].set_ylabel('Pressure (hPa)')
            axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[2].legend(loc='upper left', fontsize=10, frameon=True, title='Pressure')
        else:
            axs[2].axhline(0, color='black')
            axs[2].set_ylabel('Pressure (hPa)')
        axs[3].scatter(df['time'], df['low_clouds'], color='green', label='Low Clouds', marker='s')
        axs[3].scatter(df['time'], df['mid_clouds'], color='blue', label='Mid Clouds', marker='o')
        axs[3].scatter(df['time'], df['high_clouds'], color='red', label='High Clouds', marker='^')
        axs[3].scatter(df['time'], df['vertical_visibility'], color='yellow', label='Vertical Visibility', marker='v')
        axs[3].set_ylabel('Cloud Cover / Vertical Visibility (ft)')
        axs[3].legend(loc='upper left', fontsize=10, frameon=True, title='Cloud Cover')
        axs[3].set_ylim(0)
        precip_data = df.dropna(subset=['hourly_precipitation_in'])
        if not precip_data.empty:
            bar_width = timedelta(hours=0.8)
            for index, row in precip_data.iterrows():
                precip_amount = row['hourly_precipitation_in']
                weather_codes = row['present_weather_codes']
                timestamp = row['time']
                color = 'gray'
                if any(code in weather_codes for code in ['RA', '+RA', '-RA', 'SHRA', 'TSRA']):
                    color = 'green'
                elif any(code in weather_codes for code in ['SN', '+SN', '-SN', 'SHSN']):
                    color = 'blue'
                elif any(code in weather_codes for code in ['FZRA', 'FZDZ']):
                    color = 'purple'
                elif any(code in weather_codes for code in ['PL', 'GS']):
                    color = 'pink'
                axs[4].bar(timestamp, precip_amount, width=bar_width, color=color, zorder=2)
            axs[4].set_ylabel('Precipitation (in)')
            axs[4].set_ylim(bottom=0)
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Rain'),
                Patch(facecolor='blue', label='Snow'),
                Patch(facecolor='purple', label='Freezing Rain'),
                Patch(facecolor='pink', label='Sleet')
            ]
            axs[4].legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True, title='Precipitation Types')
        else:
            axs[4].axhline(0, color='black')
            axs[4].set_ylabel('Precipitation (in)')
        icon_y_offset = 0.07
        icon_size = 0.07
        for index, row in df.iterrows():
            timestamp = row['time']
            weather_codes = row['present_weather_codes']
            if weather_codes and pd.notna(timestamp):
                icon_path = None
                if 'RA' in weather_codes or '+RA' in weather_codes or '-RA' in weather_codes:
                    icon_path = 'meteogram_rain_icon.png'
                elif 'SN' in weather_codes or '+SN' in weather_codes or '-SN' in weather_codes:
                    icon_path = 'meteogram_snow_icon.png'
                elif 'TS' in weather_codes or 'TSRA' in weather_codes:
                    icon_path = 'meteogram_thunderstorm_icon.png'
                elif 'DZ' in weather_codes:
                    icon_path = 'meteogram_drizzle_icon.png'
                elif 'FZRA' in weather_codes or 'FZDZ' in weather_codes:
                    icon_path = 'meteogram_freezing_rain_icon.png'
                if icon_path:
                    try:
                        img = plt.imread(icon_path)
                        im = OffsetImage(img, zoom=icon_size)
                        ab = AnnotationBbox(im, (timestamp, axs[4].get_ylim()[0] + icon_y_offset * (axs[4].get_ylim()[1] - axs[4].get_ylim()[0])),
                                            xycoords='data', frameon=False, box_alignment=(0.5, 0))
                        axs[4].add_artist(ab)
                    except FileNotFoundError:
                        print(f"Icon file not found: {icon_path}. Skipping icon for {weather_codes} at {timestamp}.")
        if not df['relative_humidity'].isnull().all():
            rh_array = df['relative_humidity'].values
            rh_band = 5
            rh_lower = np.maximum(0, rh_array - rh_band)
            rh_upper = np.minimum(100, rh_array + rh_band)
            axs[5].fill_between(df['time'], rh_lower, rh_upper, color='cornflowerblue', alpha=0.3, label='RH Range')
            axs[5].plot(df['time'], rh_array, color='darkblue', label='Relative Humidity (%)', linewidth=2)
            axs[5].set_ylabel('Relative Humidity (%)')
            axs[5].legend(loc='upper left', fontsize=10, frameon=True, title='Relative Humidity')
            axs[5].set_ylim(0, 100)
        else:
            axs[5].axhline(0, color='black')
            axs[5].set_ylabel('Relative Humidity (%)')
        if not df.empty:
            precip_text = (
                f"Total Precipitation:\n"
                f"  Rain: {total_rain:.2f} in\n"
                f"  Snow: {total_snow:.2f} in\n"
                f"  Freezing Rain: {total_freezing_rain:.2f} in\n"
                f"  Sleet: {total_sleet:.2f} in\n"
                f"  Sum: {total_precip:.2f} in\n\n"
                f"Temperature:\n"
                f"  Max: {max_temp:.1f} °F\n"
                f"  Min: {min_temp:.1f} °F\n\n"
                f"Average Wind:\n"
                f"  Speed: {avg_wind_speed:.1f} knots\n"
                f"  Direction: {avg_wind_dir:.0f}°\n\n"
                f"Average Relative Humidity: {avg_rh:.1f}%\n"
            )
            if not np.isnan(max_gust):
                precip_text += f"Max Wind Gust: {max_gust:.1f} knots\n"
            axs[5].text(
                0.5, -0.35, precip_text, transform=axs[5].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
            )
        for ax in axs:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
        for ax in axs[:-1]:
            ax.xaxis.set_tick_params(labelbottom=True)
        plt.setp(axs[-1].get_xticklabels(), rotation=60, fontsize=8)
        plt.setp([ax.get_xticklabels() for ax in axs[:-1]], rotation=60, fontsize=8)
        for ax in axs:
            ax.grid(True, zorder=0)
            ymin, ymax = ax.get_ylim()
            y_ticks = ax.get_yticks()
            if len(y_ticks) > 1:
                y_ticks_filtered = [y for y in y_ticks if ymin <= y <= ymax]
                if len(y_ticks_filtered) > 1:
                    for i in range(len(y_ticks_filtered) - 1):
                        if i % 2 == 0:
                            ax.axhspan(y_ticks_filtered[i], y_ticks_filtered[i+1], color='limegreen', alpha=0.3, zorder=1)
        fig.subplots_adjust(hspace=0.4, left=0.05, right=0.97, top=0.95, bottom=0.15)
        output_path = os.path.join(OUTPUT_DIR, f'meteogram_{icao}.png')
        plt.savefig(output_path, format='png', dpi=150)
        plt.close(fig)
        print(f"Meteogram saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating meteogram: {e}")
        logger.error(f"Error in meteogram_command for {icao}: {e}")
        return None

async def windrose_command(longitude, latitude, start_date, end_date):
    try:
        print("Fetching wind data and generating chart. Please wait...")
        api_url = f"https://power.larc.nasa.gov/api/application/windrose/point?longitude={longitude}&latitude={latitude}&start={start_date}&end={end_date}&format=JSON"
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    directions = data['properties']['parameter']['wind_direction']['bins']
                    speeds = data['properties']['parameter']['wind_speed']['bins']
                    frequencies = data['properties']['parameter']['frequency']
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(111, projection='polar')
                    for i in range(len(directions)):
                        ax.bar(np.deg2rad(directions[i]), frequencies[:, i],
                               width=np.pi / len(directions), bottom=0.0,
                               label=f'{speeds[i]} m/s')
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    plt.legend()
                    plt.title(f'Wind Rose for ({latitude}, {longitude}) from {start_date} to {end_date}')
                    output_path = os.path.join(OUTPUT_DIR, f'windrose_{latitude}_{longitude}_{start_date}_{end_date}.png')
                    plt.savefig(output_path)
                    plt.close()
                    print(f"Wind rose saved to {output_path}")
                    return output_path
                else:
                    print(f"Error fetching data from NASA POWER API: Status {response.status}")
                    return None
    except Exception as e:
        print(f"Error generating wind rose: {e}")
        logger.error(f"Error in windrose_command: {e}")
        return None

async def fetch_jtwc_storms(session):
    url = "https://www.metoc.navy.mil/jtwc/jtwc.html?tropical"
    storms = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 403:
                logger.error(f"403 Forbidden on {url}, retrying with delay")
                await asyncio.sleep(5)
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as retry_response:
                    retry_response.raise_for_status()
                    soup = BeautifulSoup(await retry_response.text(), 'html.parser')
            else:
                response.raise_for_status()
                soup = BeautifulSoup(await response.text(), 'html.parser')
            tables = soup.find_all('table')
            if not tables:
                logger.warning("No tables found, trying alternative selectors")
                tables = soup.find_all('div', class_='tropical-data')
                if not tables:
                    tables = soup.find_all('table', class_='table')
            for table in tables:
                for row in table.find_all('tr')[1:]:
                    cols = row.find_all('td')
                    if len(cols) > 2:
                        storm_id = cols[0].text.strip() if cols[0].text.strip() else "Unknown"
                        name = cols[1].text.strip() if len(cols) > 1 and cols[1].text.strip() else "Unnamed"
                        lat_text = cols[2].text.strip() if len(cols) > 2 and cols[2].text.strip() else ""
                        lon_text = cols[3].text.strip() if len(cols) > 3 and cols[3].text.strip() else ""
                        if lat_text and lon_text:
                            lat = float(lat_text.rstrip('NS')) if 'N' in lat_text else -float(lat_text.rstrip('NS'))
                            lon = float(lon_text.rstrip('EW')) if 'E' in lon_text else -float(lon_text.rstrip('EW'))
                            vmax = float(cols[4].text.strip()) if len(cols) > 4 and cols[4].text.strip() else np.nan
                            mslp = float(cols[5].text.strip()) if len(cols) > 5 and cols[5].text.strip() else np.nan
                            if 'Invest' in storm_id or any(c in storm_id for c in ['W', 'S', 'P']) or 'DISTURBANCE' in storm_id:
                                storms.append({"id": storm_id, "name": name, "lat": lat, "lon": lon, "vmax": vmax, "mslp": mslp})
            disturbance_text = soup.find_all(string=lambda text: "Invest" in text or "DISTURBANCE" in text or "TCFA" in text)
            for text in disturbance_text:
                invest_match = re.search(r'(Invest|DISTURBANCE) (\d+[WS])', text, re.IGNORECASE)
                if invest_match:
                    invest_id = f"{invest_match.group(1)} {invest_match.group(2)}"
                    lat_lon_match = re.search(r'(\d+\.\d+)[NS] (\d+\.\d+)[EW]', text)
                    if lat_lon_match:
                        lat = float(lat_lon_match.group(1)) if 'N' in lat_lon_match.group(0) else -float(lat_lon_match.group(1))
                        lon = float(lat_lon_match.group(2)) if 'E' in lat_lon_match.group(0) else -float(lat_lon_match.group(2))
                        storms.append({"id": invest_id, "name": "Potential Development", "lat": lat, "lon": lon, "vmax": np.nan, "mslp": np.nan})
            logger.info(f"JTWC storms and invests retrieved: {storms}")
            return storms
    except Exception as e:
        logger.error(f"Failed to fetch JTWC storms: {e}, URL: {url}")
        return []

async def fetch_nhc_invests(session):
    url = "https://www.nhc.noaa.gov/text/refresh/MIATWOAT+shtml/180300.shtml"
    invests = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            response.raise_for_status()
            soup = BeautifulSoup(await response.text(), 'html.parser')
            outlook_text = soup.find('pre').text if soup.find('pre') else ""
            invest_matches = re.findall(r'Invest (\d+[LE])[^.\n]*near (\d+\.\d+)[NS] (\d+\.\d+)[EW]', outlook_text)
            for invest_id, lat_text, lon_text in invest_matches:
                lat = float(lat_text) if 'N' in lat_text else -float(lat_text)
                lon = float(lon_text) if 'E' in lon_text else -float(lon_text)
                invests.append({"id": invest_id, "lat": lat, "lon": lon, "name": "Potential Development", "vmax": np.nan, "mslp": np.nan})
            logger.info(f"NHC invests retrieved: {invests}")
            return invests
    except Exception as e:
        logger.error(f"Failed to fetch NHC invests: {e}, URL: {url}")
        return []

async def active_storms_atl_command():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        try:
            logger.info("Fetching active storms and invests")
            jtwc_storms = await fetch_jtwc_storms(session)
            nhc_invests = await fetch_nhc_invests(session)
            all_storms = jtwc_storms + nhc_invests
            if not all_storms:
                print("No active storms or invests found in the North Atlantic basin.")
                return None
            print("\nActive Storms and Invests in North Atlantic:")
            for storm in all_storms:
                print(f"Storm ID: {storm['id']}, Name: {storm['name']}, Lat: {storm['lat']}, Lon: {storm['lon']}, Vmax: {storm['vmax']}, MSLP: {storm['mslp']}")
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([-100, -10, 0, 60], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            for storm in all_storms:
                ax.plot(storm['lon'], storm['lat'], 'ro', markersize=10, transform=ccrs.PlateCarree())
                ax.text(storm['lon'] + 1, storm['lat'], f"{storm['name']} ({storm['id']})", transform=ccrs.PlateCarree())
            ax.set_title(f"Active Storms and Invests in North Atlantic - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%MZ')}")
            output_path = os.path.join(OUTPUT_DIR, f"active_storms_atl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%MZ')}.png")
            plt.savefig(output_path, format='png', bbox_inches='tight')
            plt.close()
            print(f"Storm map saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error generating active storms map: {e}")
            logger.error(f"Error in active_storms_atl_command: {e}")
            return None

image_links = {
    'conus': {
        '01': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/01/1250x750.jpg',
        '02': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/02/1250x750.jpg',
        '03': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/03/1250x750.jpg',
        '04': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/04/1250x750.jpg',
        '05': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/05/1250x750.jpg',
        '06': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/06/1250x750.jpg',
        '07': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/07/1250x750.jpg',
        '08': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/08/1250x750.jpg',
        '09': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/09/1250x750.jpg',
        '10': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/10/1250x750.jpg',
        '11': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/11/1250x750.jpg',
        '12': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/12/1250x750.jpg',
        '13': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/13/1250x750.jpg',
        '14': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/14/1250x750.jpg',
        '15': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/15/1250x750.jpg',
        '16': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/16/1250x750.jpg'
    },
    'chase': {
        '01': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/01/1800x1080.jpg',
        '02': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/02/1800x1080.jpg',
        '03': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/03/1800x1080.jpg',
        '04': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/04/1800x1080.jpg',
        '05': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/05/1800x1080.jpg',
        '06': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/06/1800x1080.jpg',
        '07': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/07/1800x1080.jpg',
        '08': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/08/1800x1080.jpg',
        '09': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/09/1800x1080.jpg',
        '10': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/10/1800x1080.jpg',
        '11': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/11/1800x1080.jpg',
        '12': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/12/1800x1080.jpg',
        '13': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/13/1800x1080.jpg',
        '14': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/14/1800x1080.jpg',
        '15': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/15/1800x1080.jpg',
        '16': 'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/chase/16/1800x1080.jpg'
    }
}
product_names = {
    '01': 'Blue Visible Band',
    '02': 'Red Visible Band',
    '03': 'Veggie Near-IR Band',
    '04': 'Cirrus Band',
    '05': 'Snow/Ice Near-IR Band',
    '06': 'Cloud Particle Size Band',
    '07': 'Shortwave Window Band',
    '08': 'Upper-Level Water Vapor Band',
    '09': 'Mid-Level Water Vapor Band',
    '10': 'Low-Level Water Vapor Band',
    '11': 'Cloud-Top Phase Band',
    '12': 'Ozone Band',
    '13': 'Clean Longwave Infrared Window',
    '14': 'Longwave Infrared Window',
    '15': 'Dirty Longwave Infrared Window',
    '16': 'Lower-Level Water Vapor Band'
}

async def sat_command(region, product_code):
    valid_regions = ['conus', 'chase']
    if region.lower() not in valid_regions:
        print(f"Invalid region: {region}. Valid regions: {', '.join(valid_regions)}")
        return None
    if product_code not in image_links[region.lower()]:
        print(f"Invalid product code: {product_code}. Valid codes: {', '.join(image_links[region.lower()].keys())}")
        return None
    try:
        url = image_links[region.lower()][product_code]
        product_name = product_names.get(product_code, 'Unknown Product')
        print(f"Fetching {product_name} ({product_code}) for {region}...")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                image_data = await response.read()
                output_path = os.path.join(OUTPUT_DIR, f"sat_{region.lower()}_{product_code}.jpg")
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                print(f"Satellite image saved to {output_path}")
                return output_path
    except Exception as e:
        print(f"Error fetching satellite image: {e}")
        logger.error(f"Error in sat_command: {e}")
        return None

def radar_command(region='chase', overlay='base'):
    valid_regions = ['conus', 'chase']
    valid_overlays = ['base', 'tops', 'vil']
    if region.lower() not in valid_regions:
        print(f"Invalid region: {region}. Valid regions: {', '.join(valid_regions)}")
        return None
    if overlay.lower() not in valid_overlays:
        print(f"Invalid overlay: {overlay}. Valid overlays: {', '.join(valid_overlays)}")
        return None
    base_urls = {
        'conus': 'https://weather.ral.ucar.edu/radar/images',
        'chase': 'https://weather.ral.ucar.edu/radar/images/chase'
    }
    overlay_files = {
        'base': 'NEXRAD_mosaic.gif',
        'tops': 'NEXRAD_tops_mosaic.gif',
        'vil': 'NEXRAD_vil_mosaic.gif'
    }
    try:
        url = f"{base_urls[region.lower()]}/{overlay_files[overlay.lower()]}"
        print(f"Fetching {overlay} radar for {region} from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        output_path = os.path.join(OUTPUT_DIR, f"radar_{region.lower()}_{overlay.lower()}.gif")
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Radar image saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error fetching radar image: {e}")
        logger.error(f"Error in radar_command: {e}")
        return None

def parse_time(time_str):
    if not time_str:
        return datetime.now(timezone.utc)
    try:
        time_obj = datetime.strptime(time_str, "%H:%M").time()
        now = datetime.now(timezone.utc)
        return now.replace(hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0)
    except ValueError:
        raise ValueError("Invalid time format. Use HH:MM (e.g., '14:30').")

async def astro_command(location, time_str=None):
    location_data = {
        "KATL": (33.6367, -84.4281), "KJFK": (40.6398, -73.7789), "KBIX": (30.4103, -88.9261),
        "KVQQ": (30.2264, -81.8878), "KVPC": (34.1561, -84.7983), "KRMG": (34.4956, -85.2214),
        "KMGE": (33.9131, -84.5197), "KGPT": (30.4075, -89.0753), "KPIT": (40.4915, -80.2329),
        "KSGJ": (34.25, -84.95), "KPEZ": (40.3073, -75.6192), "KNSE": (30.72247, -87.02390),
        "KTPA": (27.9755, -82.5332), "30184": (34.1561, -84.7983), "30303": (33.7525, -84.3922)
    }
    try:
        lat, lon = None, None
        if '/' in location:
            lat, lon = map(float, location.split('/'))
        elif location.upper() in location_data or location in location_data:
            lat, lon = location_data.get(location.upper(), location_data.get(location))
        else:
            geolocator = Nominatim(user_agent="weather_app")
            location_obj = geolocator.geocode(location)
            if location_obj:
                lat, lon = location_obj.latitude, location_obj.longitude
            else:
                print(f"Could not find location: {location}. Try a city name, ZIP code, ICAO code, or lat/lon.")
                return None
        observer = ephem.Observer()
        observer.lat = str(lat)
        observer.lon = str(lon)
        observer.date = parse_time(time_str)
        sun = ephem.Sun()
        moon = ephem.Moon()
        sun.compute(observer)
        moon.compute(observer)
        sun_alt = float(sun.alt) * 180 / math.pi
        sun_az = float(sun.az) * 180 / math.pi
        moon_alt = float(moon.alt) * 180 / math.pi
        moon_az = float(moon.az) * 180 / math.pi
        moon_phase = moon.phase
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=lat, lng=lon)
        if not timezone_str:
            timezone_str = 'UTC'
        local_tz = pytz.timezone(timezone_str)
        local_time = observer.date.datetime().replace(tzinfo=pytz.UTC).astimezone(local_tz)
        print(f"\nAstronomical Data for {location} at {local_time.strftime('%Y-%m-%d %H:%M %Z')} (Lat: {lat}, Lon: {lon}):")
        print(f"Sun Altitude: {sun_alt:.2f}°")
        print(f"Sun Azimuth: {sun_az:.2f}°")
        print(f"Moon Altitude: {moon_alt:.2f}°")
        print(f"Moon Azimuth: {moon_az:.2f}°")
        print(f"Moon Phase: {moon_phase:.2f}%")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='polar')
        ax.scatter([sun_az * math.pi / 180], [90 - sun_alt], c='yellow', s=100, label='Sun')
        ax.scatter([moon_az * math.pi / 180], [90 - moon_alt], c='gray', s=50, label='Moon')
        sun_path = []
        moon_path = []
        start_time = observer.date.datetime()
        for i in range(24):
            observer.date = start_time + timedelta(hours=i)
            sun.compute(observer)
            moon.compute(observer)
            sun_path.append((float(sun.az) * 180 / math.pi, 90 - float(sun.alt) * 180 / math.pi))
            moon_path.append((float(moon.az) * 180 / math.pi, 90 - float(moon.alt) * 180 / math.pi))
        sun_path = np.array(sun_path)
        moon_path = np.array(moon_path)
        ax.plot(sun_path[:, 0] * math.pi / 180, sun_path[:, 1], 'y--', label='Sun Path (24h)')
        ax.plot(moon_path[:, 0] * math.pi / 180, moon_path[:, 1], 'k--', label='Moon Path (24h)')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.set_rlim(0, 90)
        ax.set_title(f"Sky Position for {location} at {local_time.strftime('%Y-%m-%d %H:%M %Z')}")
        ax.legend()
        output_path = os.path.join(OUTPUT_DIR, f"astro_plot_{location}_{local_time.strftime('%Y%m%d_%H%MZ')}.png")
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.close()
        print(f"Astronomical plot saved to {output_path}")
        planets = ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']
        planet_data = []
        for planet_name in planets:
            planet = getattr(ephem, planet_name)()
            planet.compute(observer)
            planet_alt = float(planet.alt) * 180 / math.pi
            planet_az = float(planet.az) * 180 / math.pi
            planet_data.append({
                'name': planet_name,
                'alt': planet_alt,
                'az': planet_az
            })
        print("\nPlanetary Positions:")
        for planet in planet_data:
            print(f"{planet['name']}: Altitude {planet['alt']:.2f}°, Azimuth {planet['az']:.2f}°")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='polar')
        for planet in planet_data:
            ax.scatter([planet['az'] * math.pi / 180], [90 - planet['alt']], label=planet['name'])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.set_rlim(0, 90)
        ax.set_title(f"Solar System Positions for {location} at {local_time.strftime('%Y-%m-%d %H:%M %Z')}")
        ax.legend()
        planet_output_path = os.path.join(OUTPUT_DIR, f"astro_planets_{location}_{local_time.strftime('%Y%m%d_%H%MZ')}.png")
        plt.savefig(planet_output_path, format='png', bbox_inches='tight')
        plt.close()
        print(f"Solar system plot saved to {planet_output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating astronomical data: {e}")
        logger.error(f"Error in astro_command: {e}")
        return None

async def alerts_command(state_abbr=None):
    try:
        state_abbr = state_abbr.upper() if state_abbr else None
        if not state_abbr:
            print("Please provide a two-letter state abbreviation (e.g., 'MT').")
            return None
        alerts_url = f"https://api.weather.gov/alerts/active?area={state_abbr}"
        print(f"Fetching alerts from: {alerts_url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(alerts_url, headers={'User-Agent': 'WeatherApp/1.0'}) as response:
                response.raise_for_status()
                alerts_data = await response.json()
                filtered_alerts = [
                    alert for alert in alerts_data.get('features', [])
                    if alert.get('properties') and alert['properties'].get('event') and alert['properties'].get('severity')
                ]
                if filtered_alerts:
                    print(f"\nWeather Alerts for {state_abbr}:")
                    for alert in filtered_alerts:
                        properties = alert['properties']
                        headline = properties.get('headline', 'No Headline')
                        event = properties.get('event', 'Unknown Event')
                        severity = properties.get('severity', 'Unknown Severity')
                        description = properties.get('description', 'No description available.')
                        area_desc = "".join(properties.get('areaDesc', '')).split(";")
                        area_desc = [area.strip() for area in area_desc if area.strip()]
                        area_desc = ", ".join(area_desc)
                        print(f"\n--- {headline} ---")
                        print(f"Event: {event}")
                        print(f"Severity: {severity}")
                        print(f"Area: {area_desc}")
                        print(f"Description:\n{description}")
                else:
                    print(f"No weather alerts found for {state_abbr}.")
                logger.info(f"Requested alerts for {state_abbr}")
                return True
    except Exception as e:
        print(f"Error fetching alerts: {e}")
        logger.error(f"Error in alerts_command: {e}")
        return None

async def forecast_command(location):
    try:
        lat, lon = None, None
        location_data = {
            "KATL": (33.6367, -84.4281), "KJFK": (40.6398, -73.7789), "KBIX": (30.4103, -88.9261),
            "KVQQ": (30.2264, -81.8878), "KVPC": (34.1561, -84.7983), "KRMG": (34.4956, -85.2214),
            "KMGE": (33.9131, -84.5197), "KGPT": (30.4075, -89.0753), "KPIT": (40.4915, -80.2329),
            "KSGJ": (34.25, -84.95), "KPEZ": (40.3073, -75.6192), "KNSE": (30.72247, -87.02390),
            "KTPA": (27.9755, -82.5332), "30184": (34.1561, -84.7983), "30303": (33.7525, -84.3922)
        }
        if '/' in location:
            try:
                lat, lon = map(float, location.split('/'))
                logger.info(f"Resolved {location} to coordinates: ({lat}, {lon})")
            except ValueError:
                print("Invalid lat/lon format. Use 'lat/lon' (e.g., '34.05/-118.25').")
                return None
        elif location.upper() in location_data or location in location_data:
            lat, lon = location_data.get(location.upper(), location_data.get(location))
            logger.info(f"Resolved {location} to coordinates via local data: ({lat}, {lon})")
        elif location.isdigit() and len(location) == 5:
            geonames_url = f"http://api.geonames.org/postalCodeSearchJSON?postalcode={location}&country=US&username=freeuser&maxRows=1"
            async with aiohttp.ClientSession() as session:
                async with session.get(geonames_url, headers={'User-Agent': 'WeatherApp/1.0'}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('postalCodes'):
                            lat = data['postalCodes'][0]['lat']
                            lon = data['postalCodes'][0]['lng']
                            logger.info(f"Resolved ZIP {location} to coordinates via GeoNames: ({lat}, {lon})")
                        else:
                            print(f"Could not find ZIP code: {location}. Try a city name, ZIP code, ICAO code, or lat/lon.")
                            return None
                    else:
                        print(f"Could not find ZIP code: {location}. Try a city name, ZIP code, ICAO code, or lat/lon.")
                        return None
        else:
            geocode_url = f"https://api.weather.gov/points/{location.replace(' ', '+')}"
            async with aiohttp.ClientSession() as session:
                async with session.get(geocode_url, headers={'User-Agent': 'WeatherApp/1.0'}) as response:
                    if response.status == 200:
                        data = await response.json()
                        lat = data['properties']['relativeLocation']['geometry']['coordinates'][1]
                        lon = data['properties']['relativeLocation']['geometry']['coordinates'][0]
                        logger.info(f"Resolved {location} to coordinates via NWS API: ({lat}, {lon})")
                    else:
                        print(f"Could not find location: {location}. Try a city name, ZIP code, ICAO code, or lat/lon.")
                        return None
        nws_zone_url = f"https://api.weather.gov/points/{lat},{lon}"
        async with aiohttp.ClientSession() as session:
            async with session.get(nws_zone_url, headers={'User-Agent': 'WeatherApp/1.0'}) as response:
                if response.status == 200:
                    nws_data = await response.json()
                    zone_id = nws_data['properties']['forecastZone'].split('/')[-1]
                    state_code = zone_id[:2].lower()
                    zone_code = zone_id.lower()
                    logger.info(f"Resolved coordinates ({lat}, {lon}) to NWS zone: {zone_id}")
                else:
                    print("Could not retrieve weather zone for this location.")
                    return None
        forecast_url = f"https://tgftp.nws.noaa.gov/data/forecasts/zone/{state_code}/{zone_code}.txt"
        response = requests.get(forecast_url)
        if response.status_code != 200:
            print(f"Could not retrieve forecast for zone {zone_id}. The zone may not exist or the server is down.")
            return None
        forecast_text = response.text
        issuance_match = re.search(r'ISSUED BY NATIONAL WEATHER SERVICE.*?\n.*?(\d{1,2}:\d{2} [AP]M [A-Z]{3} [A-Z]{3} \d{1,2} \d{4})', forecast_text, re.IGNORECASE)
        if issuance_match:
            try:
                issuance_time = datetime.strptime(issuance_match.group(1), '%I:%M %p %Z %b %d %Y').replace(tzinfo=timezone.utc)
                current_time = datetime.now(timezone.utc)
                if (current_time - issuance_time).total_seconds() > 12 * 3600:
                    print("The forecast data is outdated. Please try again later.")
                    return None
            except ValueError as e:
                logger.warning(f"Could not parse issuance time for zone {zone_id}: {e}")
                print("Could not verify forecast issuance time. Proceeding with available data.")
        else:
            logger.warning(f"Could not find issuance time in forecast for zone {zone_id}.")
            print("Could not verify forecast issuance time. Proceeding with available data.")
        print(f"\nWeather Forecast for {location.title()} (Zone: {zone_id}):")
        print("```\n" + forecast_text.strip() + "\n```")
        logger.info(f"Requested forecast for {location} (Zone: {zone_id})")
        return True
    except Exception as e:
        print(f"Error in forecast command: {str(e)}")
        logger.error(f"Error in forecast_command for {location}: {str(e)}")
        return None

def worldtimes_command():
    utc_now = pytz.utc.localize(datetime.utcnow())
    us_timezones = {
        "Hawaii": "Pacific/Honolulu",
        "Alaska": "America/Anchorage",
        "Pacific": "America/Los_Angeles",
        "Mountain": "America/Denver",
        "Central": "America/Chicago",
        "Eastern": "America/New_York"
    }
    international_timezones = {
        "London": "Europe/London",
        "Berlin": "Europe/Berlin",
        "Tokyo": "Asia/Tokyo",
        "Sydney": "Australia/Sydney",
        "Tehran (Iran)": "Asia/Tehran",
        "Jerusalem (Israel)": "Asia/Jerusalem",
        "Moscow": "Europe/Moscow",
        "Beijing": "Asia/Shanghai"
    }
    print("\nTime Zones:")
    print("\nUS Time Zones:")
    for region, timezone_str in us_timezones.items():
        timezone = pytz.timezone(timezone_str)
        local_time = utc_now.astimezone(timezone)
        print(f"{region} (US): {local_time.strftime('%H:%M:%S')}")
    print("\nInternational Time Zones:")
    for city, timezone_str in international_timezones.items():
        timezone = pytz.timezone(timezone_str)
        local_time = utc_now.astimezone(timezone)
        print(f"{city}: {local_time.strftime('%H:%M:%S')}")
    logger.info("Requested world times")
    return True

async def main():
    print("Weather Application - Type 'help' for available commands")
    while True:
        command = input("> ").strip().split()
        if not command:
            continue
        cmd_name = command[0].lower()
        try:
            if cmd_name == "restart":
                print("Restarting application...")
                logger.info("Application restart initiated")
                plt.close('all')
                loop = asyncio.get_running_loop()
                tasks = [task for task in asyncio.all_tasks(loop) if task is not asyncio.current_task()]
                for task in tasks:
                    task.cancel()
                os.execv(sys.executable, [sys.executable] + sys.argv)
            elif cmd_name == "exit":
                print("Exiting application...")
                break
            elif cmd_name == "help":
                print_help()
            elif cmd_name == "metar":
                if len(command) < 2:
                    print("Usage: metar <icao> [hoursback]")
                    continue
                icao = command[1]
                hoursback = int(command[2]) if len(command) > 2 else 0
                metar_command(icao, hoursback)
            elif cmd_name == "taf":
                if len(command) < 2:
                    print("Usage: taf <icao>")
                    continue
                icao = command[1]
                taf_command(icao)
            elif cmd_name == "meteogram":
                if len(command) < 3:
                    print("Usage: meteogram <icao> <hoursback>")
                    continue
                icao = command[1]
                hoursback = int(command[2])
                path = await meteogram_command(icao, hoursback)
                if path:
                    print(f"Meteogram plot saved to {path}")
                else:
                    print("Failed to generate meteogram plot.")
            elif cmd_name == "windrose":
                if len(command) != 5:
                    print("Usage: windrose <longitude> <latitude> <start_date> <end_date>")
                    continue
                longitude = float(command[1])
                latitude = float(command[2])
                start_date = command[3]
                end_date = command[4]
                path = await windrose_command(longitude, latitude, start_date, end_date)
                if path:
                    print(f"Wind rose plot saved to {path}")
                else:
                    print("Failed to generate wind rose plot.")
            elif cmd_name == "active_storms_atl":
                path = await active_storms_atl_command()
                if path:
                    print(f"Storm map saved to {path}")
                else:
                    print("Failed to generate active storms map.")
            elif cmd_name == "sat":
                if len(command) < 3:
                    print("Usage: sat <region> <product_code>")
                    continue
                region = command[1]
                product_code = command[2]
                path = await sat_command(region, product_code)
                if path:
                    print(f"Satellite image saved to {path}")
                else:
                    print("Failed to fetch satellite image.")
            elif cmd_name == "radar":
                region = command[1] if len(command) > 1 else "chase"
                overlay = command[2] if len(command) > 2 else "base"
                path = radar_command(region, overlay)
                if path:
                    print(f"Radar image saved to {path}")
                else:
                    print("Failed to fetch radar image.")
            elif cmd_name == "astro":
                if len(command) < 2:
                    print("Usage: astro <location> [time]")
                    continue
                location = command[1]
                time_str = command[2] if len(command) > 2 else None
                path = await astro_command(location, time_str)
                if path:
                    print(f"Astronomical plot saved to {path}")
                else:
                    print("Failed to generate astronomical plot.")
            elif cmd_name == "alerts":
                state_abbr = command[1] if len(command) > 1 else None
                success = await alerts_command(state_abbr)
                if not success:
                    print("Failed to fetch alerts.")
            elif cmd_name == "forecast":
                if len(command) < 2:
                    print("Usage: forecast <location>")
                    continue
                location = ' '.join(command[1:])
                success = await forecast_command(location)
                if not success:
                    print("Failed to fetch forecast.")
            elif cmd_name == "worldtimes":
                success = worldtimes_command()
                if not success:
                    print("Failed to fetch world times.")
            elif cmd_name == "skewt":
                if len(command) == 3:
                    station, sounding_time = command[1:3]
                    path = await skewt(station, sounding_time=sounding_time)
                    if path:
                        print(f"Skew-T plot saved to {path}")
                    else:
                        print("Failed to generate Skew-T plot.")
                elif len(command) == 4:
                    station, model, forecast_hour = command[1:4]
                    path = await skewt(station, model=model, forecast_hour=int(forecast_hour))
                    if path:
                        print(f"Skew-T plot saved to {path}")
                    else:
                        print("Failed to generate Skew-T plot.")
                else:
                    print("Usage: skewt <station> <time> or skewt <station> <model> <forecast_hour>")
                    continue
            elif cmd_name == "rtma_rh":
                if len(command) < 2:
                    print("Usage: rtma_rh <state>")
                    continue
                state = command[1].upper()
                utc_time = datetime.now(pytz.UTC)
                ds, rtma_time = NOMADS_OPENDAP_Downloads.RTMA_CONUS.get_RTMA_dataset(utc_time)
                path = await plot_relative_humidity(state=state, data=ds, time=rtma_time)
                if path:
                    print(f"RTMA Relative Humidity plot saved to {path}")
                else:
                    print("Failed to generate RTMA Relative Humidity plot.")
            elif cmd_name == "rtma_rh_24hr":
                if len(command) < 2:
                    print("Usage: rtma_rh_24hr <state>")
                    continue
                state = command[1].upper()
                utc_time = datetime.now(pytz.UTC)
                ds, ds_24, rtma_time, rtma_time_24 = NOMADS_OPENDAP_Downloads.RTMA_CONUS.get_RTMA_24_hour_comparison_datasets(utc_time)
                path = await plot_24_hour_relative_humidity_comparison(state=state, data=ds, data_24=ds_24, time=rtma_time, time_24=rtma_time_24)
                if path:
                    print(f"RTMA 24-Hour Relative Humidity Comparison plot saved to {path}")
                else:
                    print("Failed to generate RTMA 24-Hour Relative Humidity Comparison plot.")
            elif cmd_name == "rtma_temp":
                if len(command) < 2:
                    print("Usage: rtma_temp <state>")
                    continue
                state = command[1].upper()
                utc_time = datetime.now(pytz.UTC)
                ds, rtma_time = NOMADS_OPENDAP_Downloads.RTMA_CONUS.get_RTMA_dataset(utc_time)
                path = await plot_temperature(state=state, data=ds, time=rtma_time)
                if path:
                    print(f"RTMA Temperature plot saved to {path}")
                else:
                    print("Failed to generate RTMA Temperature plot.")
            elif cmd_name == "rtma_dry_gusty":
                if len(command) < 2:
                    print("Usage: rtma_dry_gusty <state>")
                    continue
                state = command[1].upper()
                utc_time = datetime.now(pytz.UTC)
                ds, rtma_time = NOMADS_OPENDAP_Downloads.RTMA_CONUS.get_RTMA_dataset(utc_time)
                path = await plot_dry_and_gusty_areas(state=state, data=ds, time=rtma_time)
                if path:
                    print(f"RTMA Dry and Gusty Areas plot saved to {path}")
                else:
                    print("Failed to generate RTMA Dry and Gusty Areas plot.")
            elif cmd_name == "rtma_rh_metar":
                if len(command) < 2:
                    print("Usage: rtma_rh_metar <state>")
                    continue
                state = command[1].upper()
                utc_time = datetime.now(pytz.UTC)
                mask = get_metar_mask(state, None)
                data = await UCAR_THREDDS_SERVER_OPENDAP_Downloads.METARs.RTMA_Relative_Humidity_Synced_With_METAR(utc_time, mask)
                path = await plot_relative_humidity_with_metar_obs(state=state, data=data)
                if path:
                    print(f"RTMA Relative Humidity with METAR plot saved to {path}")
                else:
                    print("Failed to generate RTMA Relative Humidity with METAR plot.")
            elif cmd_name == "rtma_low_rh_metar":
                if len(command) < 2:
                    print("Usage: rtma_low_rh_metar <state>")
                    continue
                state = command[1].upper()
                utc_time = datetime.now(pytz.UTC)
                mask = get_metar_mask(state, None)
                data = await UCAR_THREDDS_SERVER_OPENDAP_Downloads.METARs.RTMA_Relative_Humidity_Synced_With_METAR(utc_time, mask)
                path = await plot_low_relative_humidity_with_metar_obs(state=state, data=data)
                if path:
                    print(f"RTMA Low Relative Humidity with METAR plot saved to {path}")
                else:
                    print("Failed to generate RTMA Low Relative Humidity with METAR plot.")
            elif cmd_name == "au_wind300":
                path = await au_wind300()
                if path:
                    print(f"300 hPa Wind (Australia) plot saved to {path}")
                else:
                    print("Failed to generate 300 hPa Wind (Australia) plot.")
            elif cmd_name == "au_wind500":
                path = await au_wind500()
                if path:
                    print(f"500 hPa Wind (Australia) plot saved to {path}")
                else:
                    print("Failed to generate 500 hPa Wind (Australia) plot.")
            elif cmd_name == "au_vort500":
                path = await au_vort500()
                if path:
                    print(f"500 hPa Vorticity (Australia) plot saved to {path}")
                else:
                    print("Failed to generate 500 hPa Vorticity (Australia) plot.")
            elif cmd_name == "au_fronto700":
                path = await au_fronto700()
                if path:
                    print(f"700 hPa Frontogenesis (Australia) plot saved to {path}")
                else:
                    print("Failed to generate 700 hPa Frontogenesis (Australia) plot.")
            elif cmd_name == "au_rh700":
                path = await au_rh700()
                if path:
                    print(f"700 hPa Relative Humidity (Australia) plot saved to {path}")
                else:
                    print("Failed to generate 700 hPa Relative Humidity (Australia) plot.")
            elif cmd_name == "au_wind850":
                path = await au_wind850()
                if path:
                    print(f"850 hPa Wind (Australia) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Wind (Australia) plot.")
            elif cmd_name == "au_dew850":
                path = await au_dew850()
                if path:
                    print(f"850 hPa Dewpoint (Australia) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Dewpoint (Australia) plot.")
            elif cmd_name == "au_madv850":
                path = await au_mAdv850()
                if path:
                    print(f"850 hPa Moisture Advection (Australia) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Moisture Advection (Australia) plot.")
            elif cmd_name == "au_tadv850":
                path = await au_tAdv850()
                if path:
                    print(f"850 hPa Temperature Advection (Australia) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Temperature Advection (Australia) plot.")
            elif cmd_name == "au_mslp_temp":
                path = await au_mslp_temp()
                if path:
                    print(f"MSLP with Temp Gradient (Australia) plot saved to {path}")
                else:
                    print("Failed to generate MSLP with Temp Gradient (Australia) plot.")
            elif cmd_name == "au_divcon300":
                path = await au_divcon300()
                if path:
                    print(f"300 hPa Divergence/Convergence (Australia) plot saved to {path}")
                else:
                    print("Failed to generate 300 hPa Divergence/Convergence (Australia) plot.")
            elif cmd_name == "au_thermal_wind":
                path = await au_thermal_wind()
                if path:
                    print(f"Thermal Wind (Australia) plot saved to {path}")
                else:
                    print("Failed to generate Thermal Wind (Australia) plot.")
            elif cmd_name == "wind300":
                path = await wind300()
                if path:
                    print(f"300 hPa Wind (CONUS) plot saved to {path}")
                else:
                    print("Failed to generate 300 hPa Wind (CONUS) plot.")
            elif cmd_name == "wind500":
                path = await wind500()
                if path:
                    print(f"500 hPa Wind (CONUS) plot saved to {path}")
                else:
                    print("Failed to generate 500 hPa Wind (CONUS) plot.")
            elif cmd_name == "vort500":
                path = await vort500()
                if path:
                    print(f"500 hPa Vorticity (CONUS) plot saved to {path}")
                else:
                    print("Failed to generate 500 hPa Vorticity (CONUS) plot.")
            elif cmd_name == "rh700":
                path = await rh700()
                if path:
                    print(f"700 hPa Relative Humidity (CONUS) plot saved to {path}")
                else:
                    print("Failed to generate 700 hPa Relative Humidity (CONUS) plot.")
            elif cmd_name == "fronto700":
                path = await fronto700()
                if path:
                    print(f"700 hPa Frontogenesis (CONUS) plot saved to {path}")
                else:
                    print("Failed to generate 700 hPa Frontogenesis (CONUS) plot.")
            elif cmd_name == "wind850":
                path = await wind850()
                if path:
                    print(f"850 hPa Wind (CONUS) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Wind (CONUS) plot.")
            elif cmd_name == "dew850":
                path = await dew850()
                if path:
                    print(f"850 hPa Dewpoint (CONUS) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Dewpoint (CONUS) plot.")
            elif cmd_name == "madv850":
                path = await mAdv850()
                if path:
                    print(f"850 hPa Moisture Advection (CONUS) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Moisture Advection (CONUS) plot.")
            elif cmd_name == "tadv850":
                path = await tAdv850()
                if path:
                    print(f"850 hPa Temperature Advection (CONUS) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Temperature Advection (CONUS) plot.")
            elif cmd_name == "mslp_temp":
                path = await mslp_temp()
                if path:
                    print(f"MSLP with Temp Gradient (CONUS) plot saved to {path}")
                else:
                    print("Failed to generate MSLP with Temp Gradient (CONUS) plot.")
            elif cmd_name == "divcon300":
                path = await divcon300()
                if path:
                    print(f"300 hPa Divergence/Convergence (CONUS) plot saved to {path}")
                else:
                    print("Failed to generate 300 hPa Divergence/Convergence (CONUS) plot.")
            elif cmd_name == "eu_wind300":
                path = await eu_wind300()
                if path:
                    print(f"300 hPa Wind (Europe) plot saved to {path}")
                else:
                    print("Failed to generate 300 hPa Wind (Europe) plot.")
            elif cmd_name == "eu_wind500":
                path = await eu_wind500()
                if path:
                    print(f"500 hPa Wind (Europe) plot saved to {path}")
                else:
                    print("Failed to generate 500 hPa Wind (Europe) plot.")
            elif cmd_name == "eu_vort500":
                path = await eu_vort500()
                if path:
                    print(f"500 hPa Vorticity (Europe) plot saved to {path}")
                else:
                    print("Failed to generate 500 hPa Vorticity (Europe) plot.")
            elif cmd_name == "eu_rh700":
                path = await eu_rh700()
                if path:
                    print(f"700 hPa Relative Humidity (Europe) plot saved to {path}")
                else:
                    print("Failed to generate 700 hPa Relative Humidity (Europe) plot.")
            elif cmd_name == "eu_wind850":
                path = await eu_wind850()
                if path:
                    print(f"850 hPa Wind (Europe) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Wind (Europe) plot.")
            elif cmd_name == "eu_dew850":
                path = await eu_dew850()
                if path:
                    print(f"850 hPa Dewpoint (Europe) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Dewpoint (Europe) plot.")
            elif cmd_name == "eu_madv850":
                path = await eu_mAdv850()
                if path:
                    print(f"850 hPa Moisture Advection (Europe) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Moisture Advection (Europe) plot.")
            elif cmd_name == "eu_tadv850":
                path = await eu_tAdv850()
                if path:
                    print(f"850 hPa Temperature Advection (Europe) plot saved to {path}")
                else:
                    print("Failed to generate 850 hPa Temperature Advection (Europe) plot.")
            elif cmd_name == "eu_mslp_temp":
                path = await eu_mslp_temp()
                if path:
                    print(f"MSLP with Temp Gradient (Europe) plot saved to {path}")
                else:
                    print("Failed to generate MSLP with Temp Gradient (Europe) plot.")
            elif cmd_name == "eu_divcon300":
                path = await eu_divcon300()
                if path:
                    print(f"300 hPa Divergence/Convergence (Europe) plot saved to {path}")
                else:
                    print("Failed to generate 300 hPa Divergence/Convergence (Europe) plot.")
            else:
                print(f"Unknown command: {cmd_name}. Type 'help' for available commands.")
        except Exception as e:
            print(f"Error: {e}")
            logger.error(f"Error in command {cmd_name}: {e}")

def print_help():
    print("""
Available Commands:
  metar <icao> [hoursback] - Fetch METAR data for an airport (e.g., 'metar KMGE 6')
  taf <icao> - Fetch TAF data for an airport (e.g., 'taf KMGE')
  meteogram <icao> <hoursback> - Generate a meteogram for an airport (e.g., 'meteogram KMGE 12')
  windrose <longitude> <latitude> <start_date> <end_date> - Generate a wind rose (e.g., 'windrose -84.4281 33.6367 20230101 20231231')
  active_storms_atl - List active storms and invests in the North Atlantic and generate a map
  sat <region> <product_code> - Fetch satellite imagery (e.g., 'sat conus 14')
  radar [region] [overlay] - Fetch radar imagery (e.g., 'radar chase base')
  astro <location> [time] - Generate astronomical data (e.g., 'astro kmge 14:30' or 'astro 34.05/-118.25')
  alerts <state_abbr> - Fetch weather alerts for a state (e.g., 'alerts MT')
  forecast <location> - Fetch NWS forecast (e.g., 'forecast Cartersville, GA' or 'forecast 34.05/-118.25')
  worldtimes - Display current times in various time zones
  skewt <station> <time> - Generate Skew-T diagram for observed sounding (e.g., 'skewt FFC 12Z')
  skewt <station> <model> <forecast_hour> - Generate Skew-T diagram for forecast sounding (e.g., 'skewt VQQ gfs 6')
  rtma_rh <state> - Plot RTMA relative humidity for a state (e.g., 'rtma_rh CA')
  rtma_rh_24hr <state> - Plot 24-hour RTMA relative humidity comparison (e.g., 'rtma_rh_24hr CA')
  rtma_temp <state> - Plot RTMA surface temperature (e.g., 'rtma_temp CA')
  rtma_dry_gusty <state> - Plot RTMA dry and gusty areas (e.g., 'rtma_dry_gusty CA')
  rtma_rh_metar <state> - Plot RTMA relative humidity with METAR observations (e.g., 'rtma_rh_metar CA')
  rtma_low_rh_metar <state> - Plot RTMA low relative humidity with METAR observations (e.g., 'rtma_low_rh_metar CA')
  au_wind300 - 300 hPa wind map for Australia & New Zealand
  au_wind500 - 500 hPa wind map for Australia & New Zealand
  au_vort500 - 500 hPa relative vorticity map for Australia & New Zealand
  au_fronto700 - 700 hPa frontogenesis map for Australia & New Zealand
  au_rh700 - 700 hPa relative humidity map for Australia & New Zealand
  au_wind850 - 850 hPa wind map for Australia & New Zealand
  au_dew850 - 850 hPa dewpoint map for Australia & New Zealand
  au_mAdv850 - 850 hPa moisture advection map for Australia & New Zealand
  au_tAdv850 - 850 hPa temperature advection map for Australia & New Zealand
  au_mslp_temp - MSLP with temperature gradient map for Australia & New Zealand
  au_divcon300 - 300 hPa divergence/convergence map for Australia & New Zealand
  au_thermal_wind - Thermal wind map for Australia & New Zealand
  wind300 - 300 hPa wind map for CONUS
  wind500 - 500 hPa wind map for CONUS
  vort500 - 500 hPa vorticity map for CONUS
  rh700 - 700 hPa relative humidity map for CONUS
  fronto700 - 700 hPa frontogenesis map for CONUS
  wind850 - 850 hPa wind map for CONUS
  dew850 - 850 hPa dewpoint map for CONUS
  mAdv850 - 850 hPa moisture advection map for CONUS
  tAdv850 - 850 hPa temperature advection map for CONUS
  mslp_temp - MSLP with temperature gradient map for CONUS
  divcon300 - 300 hPa divergence/convergence map for CONUS
  eu_wind300 - 300 hPa wind map for Europe
  eu_wind500 - 500 hPa wind map for Europe
  eu_vort500 - 500 hPa vorticity map for Europe
  eu_rh700 - 700 hPa relative humidity map for Europe
  eu_wind850 - 850 hPa wind map for Europe
  eu_dew850 - 850 hPa dewpoint map for Europe
  eu_mAdv850 - 850 hPa moisture advection map for Europe
  eu_tAdv850 - 850 hPa temperature advection map for Europe
  eu_mslp_temp - MSLP with temperature gradient map for Europe
  eu_divcon300 - 300 hPa divergence/convergence map for Europe
  restart - Restart the application
  exit - Exit the application
  help - Show this help message
""")

if __name__ == "__main__":
    asyncio.run(main())
