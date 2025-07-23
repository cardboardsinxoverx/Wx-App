# skewt.py
import datetime
import os
import pytz
import requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units
from matplotlib.patches import Rectangle
from metpy.plots.wx_symbols import sky_cover
from metpy.plots import add_metpy_logo
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, LinearSegmentedColormap
from siphon.simplewebservice.wyoming import WyomingUpperAir
from matplotlib.collections import LineCollection
import asyncio
import logging
import pint
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = "output"

# Station coordinates (latitude, longitude)
STATIONS = {
    'FFC': (33.36, -84.57), 'OUN': (35.18, -97.44), 'HOU': (29.65, -95.28), 'OKC': (35.39, -97.60),
    'LAX': (33.94, -118.41), 'BOS': (42.36, -71.01), 'ORD': (41.98, -87.90), 'DEN': (39.86, -104.67),
    'DFW': (32.90, -97.04), 'IND': (39.72, -86.29), 'LAS': (36.08, -115.15), 'MIA': (25.79, -80.29),
    'PHX': (33.43, -112.01), 'SEA': (47.45, -122.31), 'MSP': (44.88, -93.22), 'ASH': (42.78, -71.51),
    'JFK': (40.64, -73.78), 'PDX': (45.59, -122.60), 'PHL': (39.87, -75.24), 'PIT': (40.49, -80.23),
    'SAN': (32.73, -117.19), 'SFO': (37.62, -122.37), 'STL': (38.75, -90.37), 'TPA': (27.98, -82.53),
    'TVC': (44.74, -85.58), 'IAD': (38.94, -77.46), 'GPT': (30.41, -89.07), 'VQQ': (30.22, -81.88),
    'TIH': (-31.94, 115.97), 'FNY': (55.75, 37.62)
}

# Mapping to NWS station IDs with working BUFKIT links
NWS_STATIONS = {
    'FFC': 'KATL', 'OUN': 'KOUN', 'HOU': 'KHOU', 'OKC': 'KOKC', 'LAX': 'KLAX', 'BOS': 'KBOS',
    'ORD': 'KORD', 'DEN': 'KDEN', 'DFW': 'KDFW', 'IND': 'KIND', 'LAS': 'KLAS', 'MIA': 'KMIA',
    'PHX': 'KPHX', 'SEA': 'KSEA', 'MSP': 'KMSP', 'ASH': 'KASH', 'JFK': 'KJFK', 'PDX': 'KPDX',
    'PHL': 'KPHL', 'PIT': 'KPIT', 'SAN': 'KSAN', 'SFO': 'KSFO', 'STL': 'KSTL', 'TPA': 'KTPA',
    'TVC': 'KTVC', 'IAD': 'KIAD', 'GPT': 'KGPT', 'VQQ': 'KCRG', 'TIH': 'XTIH', 'FNY': 'XFNY'
}

def parse_bufkit_text(bufkit_text, forecast_hour):
    lines = bufkit_text.splitlines()
    model_run = None
    for line in lines:
        if 'TIME =' in line:
            match = re.search(r'TIME =\s*(\d{6}/\d{4})', line)
            if match:
                time_str = match.group(1)
                try:
                    model_run = datetime.datetime.strptime(time_str, "%y%m%d/%H%M").replace(tzinfo=pytz.UTC)
                    break
                except ValueError:
                    continue
    if not model_run:
        raise ValueError("Could not find a valid model run time in BUFKIT file.")
    print(f"Model run time: {model_run}")
    profile_starts = []
    for i, line in enumerate(lines):
        if 'STIM =' in line:
            try:
                stim = int(line.split('=')[1].strip())
                profile_starts.append((stim, i))
            except (IndexError, ValueError):
                continue
    if not profile_starts:
        raise ValueError("No valid STIM entries found in BUFKIT file.")
    closest_stim, start_idx = min(profile_starts, key=lambda x: abs(x[0] - forecast_hour))
    valid_time = model_run + datetime.timedelta(hours=closest_stim)
    end_idx = len(lines)
    for stim, idx in profile_starts:
        if idx > start_idx:
            end_idx = idx
            break
    profile_lines = lines[start_idx:end_idx]
    data_start = None
    for i, line in enumerate(profile_lines):
        if line.strip().startswith('PRES'):
            data_start = i
            break
    if data_start is None:
        raise ValueError(f"No sounding data found for forecast hour {closest_stim}.")
    header = profile_lines[data_start].split()
    data_lines = [line.split() for line in profile_lines[data_start + 1:] if line.strip() and not line.startswith('%')]
    if not data_lines:
        raise ValueError(f"No data rows found for forecast hour {closest_stim}.")
    header_len = len(header)
    valid_data_lines = [line for line in data_lines if len(line) == header_len]
    if not valid_data_lines:
        raise ValueError(f"No valid data lines found for forecast hour {closest_stim} with {header_len} columns.")
    df = pd.DataFrame(valid_data_lines, columns=header)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df, valid_time

async def fetch_bufkit_data_async(station_code, model, forecast_hour, max_retries=10, retry_delay=2):
    logger.info("Starting BUFKIT data fetch process")
    nws_station = NWS_STATIONS.get(station_code)
    if not nws_station:
        error_msg = f"No NWS station mapping found for {station_code}. Supported stations: {', '.join(STATIONS.keys())}"
        logger.error(error_msg)
        return None, None
    model_dirs = {
        'gfs': 'gfs/gfs3', 'gfsm': 'gfsm/gfs3', 'nam': 'nam/nam', 'namm': 'namm/namm', 'rap': 'rap/rap'
    }
    model_dir = model_dirs.get(model.lower())
    if not model_dir:
        print(f"Model {model} not supported. Use 'gfs', 'gfsm', 'nam', 'namm', or 'rap'.")
        return None, None
    url = f"http://www.meteor.iastate.edu/~ckarsten/bufkit/data/{model_dir}_{nws_station.lower()}.buf"
    logger.info(f"Fetching BUFKIT file from {url}")
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 404:
                logger.error(f"BUFKIT file not found at {url}")
                print(f"No BUFKIT file found for {station_code} ({nws_station}) with model {model}.")
                return None, None
            elif response.status_code != 200:
                raise ValueError(f"Failed to fetch BUFKIT file: {response.status_code} - {response.reason}")
            bufkit_text = response.text
            df, valid_time = parse_bufkit_text(bufkit_text, forecast_hour)
            df.replace(-9999.00, np.nan, inplace=True)
            df = df.copy()
            required_cols = {
                'gfs': ['PRES', 'TMPC', 'DWPC', 'DRCT', 'SKNT'],
                'gfsm': ['PRES', 'TMPC', 'DWPC', 'DRCT', 'SKNT'],
                'nam': ['PRES', 'TMPC', 'DWPC', 'DRCT', 'SKNT'],
                'namm': ['PRES', 'TMPC', 'DWPC', 'DRCT', 'SKNT'],
                'rap': ['PRES', 'TMPC', 'DWPC', 'DRCT', 'SKNT']
            }[model.lower()]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"BUFKIT data missing columns: {missing_cols}")
            df = df[required_cols].copy()
            df = df.dropna(subset=required_cols)
            df = df.drop_duplicates(subset=['PRES'], keep='last')
            df.rename(columns={'PRES': 'pressure', 'TMPC': 'temperature', 'DWPC': 'dewpoint'}, inplace=True)
            wind_speed = df['SKNT'].values * units.knots
            wind_dir = df['DRCT'].values * units.degrees
            u, v = mpcalc.wind_components(wind_speed, wind_dir)
            df['u'] = u.to('m/s').magnitude
            df['v'] = v.to('m/s').magnitude
            pressure_array = df['pressure'].to_numpy() * units.hPa
            height_array = mpcalc.pressure_to_height_std(pressure_array)
            df['height'] = height_array.to('meters').magnitude
            logger.info(f"Successfully fetched and parsed BUFKIT data for {station_code} at forecast hour {forecast_hour} with model {model}")
            return df, valid_time
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            if attempt < max_retries - 1:
                print(f"Request timed out. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
            else:
                print(f"Failed to fetch BUFKIT data for {station_code} after {max_retries} attempts due to timeout.")
                return None, None
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Error fetching BUFKIT data: {str(e)}. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
            else:
                print(f"Failed to fetch BUFKIT data for {station_code} after {max_retries} attempts: {str(e)}")
                return None, None
    return None, None

async def fetch_sounding_data(time, station_code, data_type, max_retries=10, retry_delay=2):
    if station_code in ['TIH', 'FNY'] and data_type == "observed":
        print(f"Station {station_code} is not supported for observed soundings via University of Wyoming. Try a forecast model instead (e.g., 'skewt {station_code} gfs 0').")
        return None
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Fetching data for {station_code} at {time}")
            df = await asyncio.to_thread(WyomingUpperAir.request_data, time, station_code)
            if df is None or df.empty:
                raise ValueError("Received empty or None DataFrame from WyomingUpperAir")
            if data_type == "observed":
                df.rename(columns={'u_wind': 'u', 'v_wind': 'v'}, inplace=True)
            df = df[df['pressure'] >= 100]
            essential_cols = ['pressure', 'temperature', 'dewpoint', 'u', 'v']
            df = df.dropna(subset=essential_cols)
            df = df.sort_values(by='pressure', ascending=False)
            df = df.drop_duplicates(subset=['pressure'], keep='first')
            logger.info(f"Successfully fetched and cleaned data for {station_code}: {len(df)} rows")
            return df
        except ValueError as e:
            error_str = str(e)
            if "503" in error_str or "Server Error" in error_str:
                if attempt < max_retries - 1:
                    print(f"Server busy. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                    logger.warning(f"503 error on attempt {attempt + 1} for {station_code}: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Max retries ({max_retries}) reached for {station_code}: {e}")
                    print(f"Failed to fetch data for {station_code} after {max_retries} attempts due to server overload.")
                    return None
            else:
                logger.error(f"Value error fetching data for {station_code}: {e}")
                print(f"Error fetching data: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {station_code}: {e}")
            if attempt < max_retries - 1:
                print(f"Error fetching data. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
            else:
                print(f"Failed to fetch data for {station_code} after {max_retries} attempts.")
                return None
    return None

def calculate_temperature_advection(p, u, v, lat):
    try:
        f = mpcalc.coriolis_parameter(lat * units.degrees)
        R_d = 287 * units('J/(kg K)')
        p_pa = p.to('Pa')
        delta_p = np.diff(p_pa)
        delta_u = np.diff(u)
        delta_v = np.diff(v)
        p_avg = (p_pa[:-1] + p_pa[1:]) / 2
        dTdx_layers = - (f * p_avg / R_d) * (delta_v / delta_p)
        dTdy_layers = (f * p_avg / R_d) * (delta_u / delta_p)
        n = len(p)
        dTdx = np.zeros(n) * dTdx_layers.units
        dTdy = np.zeros(n) * dTdy_layers.units
        dTdx[0], dTdy[0] = dTdx_layers[0], dTdy_layers[0]
        dTdx[n-1], dTdy[n-1] = dTdx_layers[-1], dTdy_layers[-1]
        for k in range(1, n-1):
            dTdx[k] = (dTdx_layers[k-1] + dTdx_layers[k]) / 2
            dTdy[k] = (dTdy_layers[k-1] + dTdy_layers[k]) / 2
        advection = - (u * dTdx + v * dTdy)
        return (advection * 3600 * units('s/hour')).to('degC/hour')
    except Exception as e:
        logger.error(f"Error in temperature advection calculation: {e}")
        return np.full(len(p), np.nan) * units('degC/hour')

def calculate_total_totals(p, T, Td):
    try:
        idx_850 = np.argmin(np.abs(p - 850 * units.hPa))
        idx_500 = np.argmin(np.abs(p - 500 * units.hPa))
        T_850 = T[idx_850].to('degC')
        Td_850 = Td[idx_850].to('degC')
        T_500 = T[idx_500].to('degC')
        TT = (T_850.magnitude + Td_850.magnitude) - 2 * T_500.magnitude
        return TT * units.dimensionless
    except Exception as e:
        logger.error(f"Error in Total Totals calculation: {e}")
        return np.nan * units.dimensionless

def label_ccl(skew, p, T, Td):
    try:
        p_surface = p[0]
        Td_surface = Td[0]
        e_surface = mpcalc.saturation_vapor_pressure(Td_surface)
        w_surface = mpcalc.mixing_ratio(e_surface, p_surface)
        e_p = (w_surface * p) / (0.622 + w_surface)
        Td_p = mpcalc.dewpoint(e_p)
        diff = (T - Td_p).magnitude
        idx = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(idx) > 0:
            i = idx[0]
            p1, p2 = p[i], p[i+1]
            diff1, diff2 = diff[i], diff[i+1]
            p_CCL = p1 - (p1 - p2) * (diff1 / (diff1 - diff2))
            T_CCL = np.interp(p_CCL.magnitude, p.magnitude[::-1], T.magnitude[::-1]) * T.units
            skew.ax.scatter(T_CCL, p_CCL, color='purple', marker='o', s=50)
            skew.ax.text(T_CCL.magnitude + 2, p_CCL.magnitude, 'CCL', fontsize=10, color='purple')
        else:
            logger.warning("No CCL found in the sounding.")
    except Exception as e:
        logger.error(f"Error labeling CCL: {e}")

def label_mixing_ratios(skew):
    mixing_ratios = [1, 2, 4, 8, 16]
    label_pressure = 850 * units.hPa
    for mr in mixing_ratios:
        try:
            w = (mr / 1000.0) * units('kg/kg')
            e = (w * label_pressure) / (0.622 + w)
            Td = mpcalc.dewpoint(e)
            x_pos = Td.to('degC').magnitude + 5
            y_pos = label_pressure.magnitude
            skew.ax.text(
                x_pos, y_pos, f'{mr} g/kg',
                fontsize=13, color='#02a312',
                verticalalignment='center',
                horizontalalignment='left'
            )
        except Exception as e:
            logger.warning(f"Could not label mixing ratio {mr} g/kg: {e}")

def dry_adiabatic_descent(p_start, T_start, pressures):
    try:
        if p_start is None or T_start is None or pressures is None:
            raise ValueError("p_start, T_start, or pressures is None")
        if np.isnan(p_start.magnitude) or np.isnan(T_start.magnitude):
            raise ValueError("p_start or T_start contains NaN values")
        if np.any(np.isnan(pressures.magnitude)):
            raise ValueError("pressures contains NaN values")
        lapse_rate = 9.8 * units('K/km')
        z_start = mpcalc.pressure_to_height_std(p_start).to('km')
        z_levels = mpcalc.pressure_to_height_std(pressures).to('km')
        dz = z_levels - z_start
        T_start_K = T_start.to('K')
        delta_T = lapse_rate * dz
        T_profile_K = T_start_K - delta_T
        return T_profile_K.to('degC')
    except Exception as e:
        logger.error(f"Error in dry_adiabatic_descent: {e}")
        return np.full(len(pressures), np.nan) * units.degC

async def skewt(station_code, sounding_time=None, model=None, forecast_hour=None):
    """
    Generate an enhanced Skew-T diagram from observed or forecast sounding data.
    Args:
        station_code (str): Three-letter station code (e.g., 'FFC')
        sounding_time (str): '00Z' or '12Z' for observed soundings
        model (str): 'gfs', 'gfsm', 'nam', 'namm', or 'rap' for forecast soundings
        forecast_hour (int): Forecast hour for model data
    Returns:
        str: Path to saved plot or None if failed
    """
    logger.info(f"Generating Skew-T diagram for {station_code}, time={sounding_time}, model={model}, forecast_hour={forecast_hour}")
    utc_time = datetime.datetime.now(pytz.UTC)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        if sounding_time and not (model or forecast_hour):
            station_code = station_code.upper()
            sounding_time = sounding_time.upper()
            if sounding_time not in ['00Z', '12Z']:
                raise ValueError("Invalid time. Use '00Z' or '12Z'.")
            year, month, day = utc_time.year, utc_time.month, utc_time.day
            hour = 12 if sounding_time == "12Z" else 0
            if sounding_time == "00Z" and utc_time.hour >= 12:
                day += 1
            elif sounding_time == "12Z" and utc_time.hour < 12:
                day -= 1
            try:
                now = datetime.datetime(year, month, day, hour, 0, 0, tzinfo=pytz.UTC)
            except ValueError as e:
                logger.error(f"Invalid date: year={year}, month={month}, day={day}, hour={hour}")
                print("Error: The specified date is invalid (e.g., day out of range for the month).")
                return None
            data_type = "observed"
            df = await fetch_sounding_data(now, station_code, data_type)
            if df is None:
                return None
            try:
                station_lat = df['latitude'][0]
                station_lon = df['longitude'][0]
            except (KeyError, IndexError):
                logger.warning(f"Latitude/Longitude not found in DataFrame for {station_code}, using STATIONS dictionary.")
                station_lat, station_lon = STATIONS.get(station_code, (np.nan, np.nan))
            title = f"{station_code} - {now.strftime('%Y-%m-%d %HZ')}"
        elif model and forecast_hour is not None:
            station_code = station_code.upper()
            model = model.lower()
            forecast_hour = int(forecast_hour)
            if model not in ['gfs', 'gfsm', 'nam', 'namm', 'rap']:
                raise ValueError("Model not supported. Use 'gfs', 'gfsm', 'nam', 'namm', or 'rap'.")
            max_hours = {'gfs': 384, 'gfsm': 384, 'nam': 84, 'namm': 84, 'rap': 18}
            if forecast_hour > max_hours[model]:
                raise ValueError(f"Forecast hour exceeds maximum for {model} ({max_hours[model]} hours).")
            if station_code not in NWS_STATIONS:
                raise ValueError(f"Station {station_code} not found in NWS station mapping.")
            df, valid_time = await fetch_bufkit_data_async(station_code, model, forecast_hour)
            if df is None:
                return None
            station_lat, station_lon = STATIONS[station_code]
            title = f"{station_code} {model.upper()} Forecast {forecast_hour}-hr ({valid_time.strftime('%Y-%m-%d %HZ')})"
            data_type = "forecast"
        else:
            raise ValueError("Usage: skewt <station> <time> or skewt <station> <model> <forecast_hour>")
        df.drop_duplicates(subset=['pressure'], keep='first', inplace=True)
        essential_cols = ['pressure', 'height', 'temperature', 'dewpoint', 'u', 'v']
        df = df.dropna(subset=essential_cols)
        if df.empty:
            raise ValueError(f"No valid data for {station_code} after cleaning.")
        z = df['height'].values * units.m
        p = df['pressure'].values * units.hPa
        T = df['temperature'].values * units.degC
        Td = df['dewpoint'].values * units.degC
        u = df['u'].values * units('m/s')
        v = df['v'].values * units('m/s')
        sort_indices = np.argsort(-p.magnitude)
        z, p, T, Td, u, v = [arr[sort_indices] for arr in [z, p, T, Td, u, v]]
        Td = np.minimum(Td, T)
        if np.any(p.magnitude < 0):
            logger.warning("Negative pressure values detected")
        if np.any(p.magnitude < 100):
            logger.warning(f"Very low pressure values detected: min p = {np.min(p)}")
        T_kelvin = T.to('kelvin')
        Td_kelvin = Td.to('kelvin')
        for name, arr in [('pressure', p), ('temperature', T), ('dewpoint', Td), ('u', u), ('v', v)]:
            if np.any(np.isnan(arr.magnitude)):
                logger.warning(f"NaN values detected in {name} array")
        theta_e = mpcalc.equivalent_potential_temperature(p, T, Td)
        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
        p_truncated = p[p >= 100 * units.hPa]
        T_truncated = T[p >= 100 * units.hPa]
        Td_truncated = Td[p >= 100 * units.hPa]
        if data_type == "forecast":
            prof = mpcalc.parcel_profile(p_truncated, T[0], Td[0]).to('degC')
            sbcape, sbcin = mpcalc.surface_based_cape_cin(p, T, Td)
            ml_depth = 100 * units.hPa
            ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=ml_depth)
            ml_prof = mpcalc.parcel_profile(p_truncated, ml_t, ml_td).to('degC')
            mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, Td, depth=ml_depth)
            mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p, T, Td, depth=100 * units.hPa)
            mpl_height = np.interp(mu_p.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if mu_p is not None else np.nan * units.m
            mu_prof = mpcalc.parcel_profile(p_truncated, mu_t, mu_td).to('degC')
            mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=100 * units.hPa)
            valid_idx = np.where((p > 500 * units.hPa) & (~np.isnan(theta_e.magnitude)))[0]
            if len(valid_idx) > 0:
                min_theta_e_idx = valid_idx[np.argmin(theta_e[valid_idx].magnitude)]
                p_start = p[min_theta_e_idx]
                T_start = T[min_theta_e_idx]
                down_prof = dry_adiabatic_descent(p_start, T_start, p_truncated)
            else:
                down_prof = None
        else:
            prof = mpcalc.parcel_profile(p_truncated, T[0], Td[0]).to('degC')
            sbcape, sbcin = mpcalc.surface_based_cape_cin(p, T, Td)
            ml_depth = 100 * units.hPa
            ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=ml_depth)
            ml_prof = mpcalc.parcel_profile(p_truncated, ml_t, ml_td).to('degC')
            mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, Td, depth=ml_depth)
            mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p, T, Td, depth=300 * units.hPa)
            mpl_height = np.interp(mu_p.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if mu_p is not None else np.nan * units.m
            # skewt.py (continued)
        mu_prof = mpcalc.parcel_profile(p_truncated, mu_t, mu_td).to('degC')
        mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=100 * units.hPa)
        min_theta_e_idx = np.argmin(theta_e[p > 500 * units.hPa])
        p_start = p[min_theta_e_idx]
        T_start = T[min_theta_e_idx]
        down_prof = dry_adiabatic_descent(p_start, T_start, p_truncated).to('degC')

        wet_bulb = mpcalc.wet_bulb_temperature(p, T_kelvin, Td_kelvin)
        kindex = mpcalc.k_index(p, T_kelvin, Td_kelvin)
        total_totals = calculate_total_totals(p, T, Td)
        temperature_advection = calculate_temperature_advection(p, u, v, station_lat)
        lfc_pressure, lfc_temperature = mpcalc.lfc(p, T, Td)
        el_pressure, el_temperature = mpcalc.el(p, T, Td)

        idx = np.where(T.magnitude < 0)[0]
        p_freeze = None
        if len(idx) > 0 and idx[0] > 0:
            idx = idx[0]
            T1, T2 = T[idx-1], T[idx]
            p1, p2 = p[idx-1], p[idx]
            if T2 != T1 and not np.any(np.isnan([T1.magnitude, T2.magnitude, p1.magnitude, p2.magnitude])):
                p_freeze = p1 + (0 * units.degC - T1) * (p2 - p1) / (T2 - T1)
                if p_freeze < 100 * units.hPa or p_freeze > 1000 * units.hPa:
                    logger.warning(f"Freezing level pressure {p_freeze} out of bounds")
                    p_freeze = None

        p_CCL, T_CCL, ccl_height, Tc = None, None, None, None
        if len(p) > 1:
            try:
                e_surface = mpcalc.saturation_vapor_pressure(Td[0])
                w_surface = mpcalc.mixing_ratio(e_surface, p[0])
                e_p = (w_surface * p) / (0.622 + w_surface)
                Td_p = mpcalc.dewpoint(e_p)
                diff = (T - Td_p).magnitude
                idx = np.where(np.diff(np.sign(diff)) != 0)[0]
                if len(idx) > 0:
                    i = idx[0]
                    p1, p2 = p[i], p[i+1]
                    diff1, diff2 = diff[i], diff[i+1]
                    p_CCL = p1 - (p1 - p2) * (diff1 / (diff1 - diff2))
                    T_CCL = np.interp(p_CCL.magnitude, p.magnitude[::-1], T.magnitude[::-1]) * T.units
                    ccl_height = np.interp(p_CCL.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m
                    if p_CCL is not None and T_CCL is not None:
                        T_CCL_K = T_CCL.to('K')
                        p_surface = p[0]
                        Tc_K = T_CCL_K * (p_surface / p_CCL) ** 0.286
                        Tc = Tc_K.to('degC')
                    else:
                        Tc = np.nan * units.degC
                else:
                    p_CCL, T_CCL, ccl_height = None, None, np.nan * units.m
                    Tc = np.nan * units.degC
            except Exception as e:
                logger.error(f"Error calculating CCL/Tc: {e}")
                p_CCL, T_CCL, ccl_height, Tc = None, None, None, np.nan * units.degC
        else:
            p_CCL, T_CCL, ccl_height, Tc = None, None, np.nan * units.m, np.nan * units.degC

        max_T = np.max(T) if len(T) > 0 else np.nan * units.degC

        RM, LM, MW = mpcalc.bunkers_storm_motion(p, u, v, z)
        storm_u, storm_v = RM
        total_helicity1, _, _ = mpcalc.storm_relative_helicity(z, u, v, depth=1 * units.km, storm_u=storm_u, storm_v=storm_v)
        total_helicity3, _, _ = mpcalc.storm_relative_helicity(z, u, v, depth=3 * units.km, storm_u=storm_u, storm_v=storm_v)
        total_helicity6, _, _ = mpcalc.storm_relative_helicity(z, u, v, depth=6 * units.km, storm_u=storm_u, storm_v=storm_v)
        bshear1 = mpcalc.bulk_shear(p, u, v, depth=1 * units.km)
        bshear3 = mpcalc.bulk_shear(p, u, v, depth=3 * units.km)
        bshear6 = mpcalc.bulk_shear(p, u, v, depth=6 * units.km)
        bshear1_mag = np.sqrt(bshear1[0]**2 + bshear1[1]**2) if bshear1 is not None else np.nan * units('m/s')
        bshear3_mag = np.sqrt(bshear3[0]**2 + bshear3[1]**2) if bshear3 is not None else np.nan * units('m/s')
        bshear6_mag = np.sqrt(bshear6[0]**2 + bshear6[1]**2) if bshear6 is not None else np.nan * units('m/s')

        lcl_height = np.interp(lcl_pressure.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if lcl_pressure is not None else np.nan * units.m
        lfc_height = np.interp(lfc_pressure.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if lfc_pressure is not None and not np.isnan(lfc_pressure.magnitude) else np.nan * units.m
        el_height = np.interp(el_pressure.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if el_pressure is not None and not np.isnan(el_pressure.magnitude) else np.nan * units.m
        fl_height = np.interp(p_freeze.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if p_freeze is not None and not np.isnan(p_freeze.magnitude) else np.nan * units.m
        mpl_height = np.interp(mu_p.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if mu_p is not None and not np.isnan(mu_p.magnitude) else np.nan * units.m

        if isinstance(total_helicity1.magnitude, np.ndarray):
            total_helicity1 = total_helicity1[0] if total_helicity1.magnitude.size > 0 else np.nan * units('m**2/s**2')
        if isinstance(total_helicity3.magnitude, np.ndarray):
            total_helicity3 = total_helicity3[0] if total_helicity3.magnitude.size > 0 else np.nan * units('m**2/s**2')
        if isinstance(total_helicity6.magnitude, np.ndarray):
            total_helicity6 = total_helicity6[0] if total_helicity6.magnitude.size > 0 else np.nan * units('m**2/s**2')
        if isinstance(bshear1_mag.magnitude, np.ndarray):
            bshear1_mag = bshear1_mag[0] if bshear1_mag.magnitude.size > 0 else np.nan * units('m/s')
        if isinstance(bshear3_mag.magnitude, np.ndarray):
            bshear3_mag = bshear3_mag[0] if bshear3_mag.magnitude.size > 0 else np.nan * units('m/s')
        if isinstance(bshear6_mag.magnitude, np.ndarray):
            bshear6_mag = bshear6_mag[0] if bshear6_mag.magnitude.size > 0 else np.nan * units('m/s')

        sig_tor = mpcalc.significant_tornado(sbcape, lcl_height, total_helicity1, bshear6_mag).to_base_units()
        super_comp = mpcalc.supercell_composite(mucape, total_helicity3, bshear3_mag).to_base_units()

        if isinstance(sig_tor.magnitude, np.ndarray):
            sig_tor = sig_tor[0] if sig_tor.magnitude.size > 0 else np.nan * units.dimensionless
        if isinstance(super_comp.magnitude, np.ndarray):
            super_comp = super_comp[0] if super_comp.magnitude.size > 0 else np.nan * units.dimensionless

        RH = mpcalc.relative_humidity_from_dewpoint(T, Td) * 100
        e = mpcalc.saturation_vapor_pressure(Td)
        w = mpcalc.mixing_ratio(e, p).to('g/kg')
        z_km = z.to('km')

        layers = [
            (0, 0.5, '0 - 0.5 km'), (0, 1, '0 - 1 km'), (1, 3, '1 - 3 km'),
            (3, 6, '3 - 6 km'), (6, 9, '6 - 9 km')
        ]
        layer_data = []
        for lower, upper, label in layers:
            if lower == 0:
                idx = (z_km <= upper * units.km)
            else:
                idx = (z_km > lower * units.km) & (z_km <= upper * units.km)
            if np.any(idx):
                mean_RH = np.mean(RH[idx].magnitude)
                mean_w = np.mean(w[idx].magnitude)
                layer_data.append((label, mean_RH, mean_w))
            else:
                layer_data.append((label, np.nan, np.nan))

        p_mag = p.magnitude
        z_mag = z.magnitude
        T_mag = T.magnitude
        w_mag = w.magnitude

        if 700 >= p.min().magnitude and 700 <= p.max().magnitude:
            z_700 = np.interp(700, p_mag[::-1], z_mag[::-1]) * z.units
            T_700 = np.interp(700, p_mag[::-1], T_mag[::-1]) * T.units
            w_700 = np.interp(700, p_mag[::-1], w_mag[::-1]) * w.units
        else:
            z_700 = np.nan * z.units
            T_700 = np.nan * T.units
            w_700 = np.nan * w.units

        if 500 >= p.min().magnitude and 500 <= p.max().magnitude:
            z_500 = np.interp(500, p_mag[::-1], z_mag[::-1]) * z.units
            T_500 = np.interp(500, p_mag[::-1], T_mag[::-1]) * T.units
        else:
            z_500 = np.nan * z.units
            T_500 = np.nan * T.units

        if not np.isnan(z_500.magnitude) and not np.isnan(z_700.magnitude) and z_500 > z_700:
            dz_km = (z_500 - z_700).to('km').magnitude
            LR_700_500 = ((T_700 - T_500).magnitude / dz_km) * units('degC/km')
        else:
            LR_700_500 = np.nan * units('degC/km')

        if all(q is not None for q in [mucape, w_700, T_500, LR_700_500, bshear6_mag]) and not np.isnan([q.magnitude for q in [mucape, w_700, T_500, LR_700_500, bshear6_mag]]).any():
            ship = (mucape.magnitude / 1000) * (w_700.magnitude / 14) * (-T_500.magnitude / 30) * (LR_700_500.magnitude / 9) * (bshear6_mag.magnitude / 20) * units.dimensionless
            if isinstance(ship.magnitude, np.ndarray):
                ship = ship[0] if ship.magnitude.size > 0 else np.nan * units.dimensionless
        else:
            ship = np.nan * units.dimensionless

        PWAT = mpcalc.precipitable_water(p, Td).to('inch').magnitude
        WB_surface = wet_bulb[0].to('degC').magnitude
        idx_wb0 = np.where(wet_bulb.magnitude < 0)[0]
        if len(idx_wb0) > 0 and idx_wb0[0] > 0:
            i = idx_wb0[0]
            wb1, wb2 = wet_bulb[i-1], wet_bulb[i]
            p1, p2 = p[i-1], p[i]
            z1, z2 = z[i-1], z[i]
            if (wb2.magnitude - wb1.magnitude) != 0:
                p_wb0 = p1 + (0 - wb1.magnitude) * (p2 - p1) / (wb2.magnitude - wb1.magnitude)
                wb0_height = np.interp(p_wb0.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m
            else:
                wb0_height = np.nan * units.m
        else:
            wb0_height = np.nan * units.m

        surface_RH = mpcalc.relative_humidity_from_dewpoint(T[0], Td[0]) * 100 * units.percent
        surface_wet_bulb = wet_bulb[0].to('degC')

    except ValueError as e:
        logger.error(f"Value error in data prep: {e}")
        print(str(e))
        return None
    except Exception as e:
        logger.error(f"Unexpected error in data prep: {e}", exc_info=True)
        print("An unexpected error occurred while preparing the data.")
        return None

    # Plotting Section
    try:
        fig = plt.figure(figsize=(30, 15))
        fig.set_facecolor('lightsteelblue')
        skew = SkewT(fig, rotation=45, rect=[0.03, 0.15, 0.55, 0.8])

        x1 = np.linspace(-100, 40, 8)
        x2 = np.linspace(-90, 50, 8)
        y = [1050, 100]
        for i in range(0, 8):
            skew.shade_area(y=y, x1=x1[i], x2=x2[i], color='#ebbe2a', alpha=0.25, zorder=1)

        skew.plot(p, T, 'r', linewidth=2, label='Temperature')
        skew.plot(p, Td, 'g', linewidth=2, label='Dewpoint')
        skew.plot(p, wet_bulb.to('degC'), 'b', linestyle='--', linewidth=2, label='Wet Bulb')
        if prof is not None:
            skew.plot(p, prof, 'k', linewidth=2.5, label='SB Parcel')
        if ml_prof is not None:
            skew.plot(p, ml_prof, 'm--', linewidth=2, label='ML Parcel')
        if mu_prof is not None:
            skew.plot(p, mu_prof, 'y--', linewidth=2, label='MU Parcel')
        if down_prof is not None:
            skew.plot(p, down_prof, color='#a87308', linestyle='-.', linewidth=2.5, label='Downdraft')
        skew.plot_barbs(p[::2], u[::2], v[::2], color='#000000')
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 60)
        if prof is not None:
            skew.shade_cin(p, T, prof, Td)
            skew.shade_cape(p, T, prof)
        skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
        skew.ax.set_xlabel('Temperature (°C)', weight='bold')
        skew.ax.set_ylabel(f'Pressure ({p.units:~P})', weight='bold')
        skew.plot_dry_adiabats(linewidth=1.5, color='brown', label='Dry Adiabat')
        skew.plot_moist_adiabats(linewidth=1.5, color='purple', label='Moist Adiabat')
        skew.plot_mixing_lines(linewidth=1.5, color='#02a312', label='Mixing Ratio (g/kg)')
        label_mixing_ratios(skew)
        label_ccl(skew, p, T, Td)

        def plot_point(skew, pressure, temperature, marker, color, label, log_message):
            if (pressure is not None and temperature is not None and
                not np.isnan(pressure.magnitude) and not np.isnan(temperature.magnitude) and
                100 <= pressure.magnitude <= 1000 and -40 <= temperature.magnitude <= 60):
                try:
                    logger.debug(f"Plotting {label}: pressure={pressure}, temperature={temperature}")
                    skew.ax.scatter(temperature, pressure, marker='o', s=50, color=color)
                    skew.ax.text(temperature.magnitude + 2, pressure.magnitude, label,
                                 fontsize=10, color='k' if label in ['LFC', 'EL'] else color)
                    logger.info(log_message)
                except Exception as e:
                    logger.error(f"Error plotting {label}: {e}")
            else:
                logger.info(f"Skipping {label}: Invalid pressure={pressure}, temperature={temperature}")

        plot_point(skew, lcl_pressure, lcl_temperature, 'o', 'black', 'LCL', f"LCL plotted at {lcl_pressure}, {lcl_temperature}")
        plot_point(skew, lfc_pressure, lfc_temperature, 'o', 'black', 'LFC', f"LFC plotted at {lfc_pressure}, {lfc_temperature}")
        plot_point(skew, el_pressure, el_temperature, 'o', 'black', 'EL', f"EL plotted at {el_pressure}, {el_temperature}")
        plot_point(skew, p_freeze, 0 * units.degC if p_freeze is not None else None, 'o', 'blue', 'FL', f"Freezing Level plotted at {p_freeze}, 0°C")
        plot_point(skew, p_CCL, T_CCL, 'o', 'purple', 'CCL', f"CCL plotted at {p_CCL}, {T_CCL}")

        adv_ax = plt.axes([0.58, 0.05, 0.08, 0.9])
        temp_advection = temperature_advection.magnitude
        max_adv = 5
        norm = Normalize(vmin=-max_adv, vmax=max_adv)
        colors = plt.cm.coolwarm(norm(temp_advection))
        adv_ax.barh(p.magnitude, temp_advection, color=colors, height=8, align='center')
        adv_ax.set_ylim(1000, 100)
        adv_ax.set_xlim(-max_adv, max_adv)
        adv_ax.set_xlabel('Temp. Adv. (°C/hr)', weight='bold')
        adv_ax.set_ylabel('Pressure (hPa)', weight='bold')
        adv_ax.axvline(0, color='black', linewidth=0.5)
        adv_ax.set_facecolor('#fafad2')

        ax_hodo = plt.axes((0.61, 0.32, 0.4, 0.4))
        ax_hodo.set_facecolor('#fafad2')
        h = Hodograph(ax_hodo, component_range=60.)
        h.add_grid(increment=20, linestyle='--', linewidth=1)
        cmap = LinearSegmentedColormap.from_list('my_cmap', ['purple', 'blue', 'green', 'yellow', 'orange', 'red'])
        norm = Normalize(vmin=z.min().magnitude, vmax=z.max().magnitude)
        colored_line = h.plot_colormapped(u, v, c=z.magnitude, linewidth=6, cmap=cmap, label='0-12km WIND')
        cbar = plt.colorbar(colored_line, ax=h.ax, orientation='vertical', pad=0.01)
        cbar.set_label('Height (m)')
        ax_hodo.set_xlabel('U component (m/s)', weight='bold')
        ax_hodo.set_ylabel('V component (m/s)', weight='bold')
        ax_hodo.set_title(f'Hodograph {title}')

        idx_6km = np.argmin(np.abs(z.magnitude - 6000))
        h.ax.scatter(u[0], v[0], color='blue', marker='o', s=100, edgecolor='black', label='Surface Wind', zorder=3)
        h.ax.scatter(u[idx_6km], v[idx_6km], color='cyan', marker='x', s=100, label='6 km Wind', zorder=3)
        h.ax.text(u[idx_6km].magnitude + 2, v[idx_6km].magnitude + 2, '6 km', color='cyan', fontsize=10)
        h.ax.scatter(u[-1], v[-1], color='red', marker='o', s=100, edgecolor='black', label='12km Wind', zorder=3)

        for motion, label, offset in [(RM, 'RM', (0.5, -0.5)), (LM, 'LM', (0.5, -0.5)), (MW, 'MW', (0.5, -0.5))]:
            h.ax.text(motion[0].magnitude + offset[0], motion[1].magnitude + offset[1], label,
                      weight='bold', ha='left', fontsize=13, alpha=0.6)
        h.ax.arrow(0, 0, RM[0].magnitude - 0.3, RM[1].magnitude - 0.3, linewidth=2, color='red',
                   alpha=0.5, label='Bunkers RM Vector', length_includes_head=True, head_width=2)

        h.ax.fill([u[0].magnitude, u[idx_6km].magnitude, RM[0].magnitude],
                  [v[0].magnitude, v[idx_6km].magnitude, RM[1].magnitude],
                  color='blue', alpha=0.3, label='Inflow Shading')

        w_ax = 0.08
        h_ax = 2 * w_ax
        theta_e_ax = fig.add_axes([0.46, -0.06, w_ax, h_ax])
        theta_e_ax.plot(theta_e, p, color='purple', linewidth=1.5)
        theta_e_ax.set_xlabel('Theta-E (K)', fontsize=8)
        theta_e_ax.set_ylabel('Pressure (hPa)', fontsize=8)
        theta_e_ax.set_title('Theta-e(K)/Pressure(hPa)', fontsize=10, weight='bold')
        theta_e_ax.invert_yaxis()
        theta_e_ax.set_xlim(280, 360)
        theta_e_ax.set_ylim(1000, 100)
        theta_e_ax.tick_params(axis='both', labelsize=6)
        theta_e_ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        theta_e_ax.set_facecolor('#fafad2')

        sig_tor_ax = fig.add_axes([0.10, -0.06, w_ax, h_ax])
        sig_tor_ax.set_facecolor('#fafad2')
        sig_tor_ax.set_title('Significant Tornado Parameter', fontsize=10, weight='bold')
        if not np.isnan(sig_tor.magnitude):
            sig_tor_value = sig_tor.magnitude.item()
            sig_tor_ax.bar(0, sig_tor_value, color='blue', width=0.5)
            sig_tor_ax.text(0, sig_tor_value + 0.1, f'{sig_tor_value:.1f}', ha='center', va='bottom', fontsize=8)
        else:
            sig_tor_ax.text(0.5, 0.5, 'N/A', transform=sig_tor_ax.transAxes, fontsize=8, ha='center', va='center')
        sig_tor_ax.axhline(1, color='red', linestyle='--', linewidth=1)
        sig_tor_ax.set_xlim(-0.5, 0.5)
        sig_tor_ax.set_ylim(0, 10)
        sig_tor_ax.set_xticks([])
        sig_tor_ax.set_ylabel('SIG TOR', fontsize=8)
        sig_tor_ax.grid(True, axis='y', color='gray', linestyle='--', linewidth=0.5)

        u_sr = u - storm_u
        v_sr = v - storm_v
        sr_wind_speed = np.sqrt(u_sr**2 + v_sr**2).to('knots')
        z_km = z.to('km').magnitude

        layers = [(0, 2), (4, 6), (9, 11)]
        colors = ['red', 'blue', 'purple']
        labels = ['Lower (0-2 km)', 'Mid (4-6 km)', 'Upper (9-11 km)']

        mean_sr_wind = []
        for lower, upper in layers:
            mask = (z_km >= lower) & (z_km <= upper)
            if np.any(mask):
                mean_speed = np.mean(sr_wind_speed[mask].magnitude)
            else:
                mean_speed = np.nan
            mean_sr_wind.append(mean_speed)

        sr_wind_ax = fig.add_axes([0.22, -0.06, 0.08, 0.16])
        sr_wind_ax.set_facecolor('#fafad2')
        bar_width = 0.5
        for i, (mean_speed, color, label) in enumerate(zip(mean_sr_wind, colors, labels)):
            if not np.isnan(mean_speed):
                sr_wind_ax.barh(i, mean_speed, color=color, height=bar_width, label=label)
        sr_wind_ax.axvline(15, color='white', linestyle='--', linewidth=1, label='15 kts (Severe)')
        sr_wind_ax.axvline(50, color='purple', linestyle='--', linewidth=1, label='50 kts (Supercell)')
        sr_wind_ax.set_xlim(0, 70)
        sr_wind_ax.set_ylim(-0.5, 2.5)
        sr_wind_ax.set_yticks([0, 1, 2])
        sr_wind_ax.set_yticklabels(['Lower', 'Mid', 'Upper'])
        sr_wind_ax.set_xlabel('SR Wind (kts)', fontsize=8)
        sr_wind_ax.set_title('Storm Relative Wind\nvs Height', fontsize=10, weight='bold')
        sr_wind_ax.tick_params(axis='both', labelsize=6)
        sr_wind_ax.grid(True, color='gray', linestyle='--', linewidth='0.5')
        sr_wind_ax.legend(loc='upper right', fontsize=6)

        storm_slinky_ax = fig.add_axes([0.34, -0.06, w_ax, h_ax])
        x = [0]
        y = [0]
        w = 10
        for i in range(1, len(z)):
            if z[i].to('km').magnitude > 3:
                break
            dz = (z[i] - z[i-1]).to('m').magnitude
            dt = dz / w
            dx = u_sr[i-1].magnitude * dt
            dy = v_sr[i-1].magnitude * dt
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)

        storm_slinky_ax.plot(x, y, color='purple', linewidth=2, label='Storm Slinky')
        storm_slinky_ax.scatter(x, y, c=z[:len(x)].to('km').magnitude, cmap='brg', s=30)
        storm_slinky_ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        storm_slinky_ax.axvline(0, color='black', linewidth=2, zorder=0)
        storm_slinky_ax.axhline(0, color='black', linewidth=2, zorder=0)
        storm_slinky_ax.set_facecolor('#fafad2')
        storm_slinky_ax.set_xlabel('X Displacement (m)', fontsize=8)
        storm_slinky_ax.set_ylabel('Y Displacement (m)', fontsize=8)
        storm_slinky_ax.set_title('Storm Slinky\n(Trajectory starting at (0,0) at LFC)', fontsize=10, weight='bold')
        storm_slinky_ax.set_xlim(-5000, 5000)
        storm_slinky_ax.set_ylim(-5000, 5000)
        storm_slinky_ax.tick_params(axis='both', labelsize=6)
        norm = Normalize(vmin=0, vmax=3)
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='brg'), ax=storm_slinky_ax, orientation='vertical', pad=0.03, fraction=0.046)
        cbar.set_label('Height (km)', fontsize=8)

        storm_speed = np.sqrt(storm_u.magnitude**2 + storm_v.magnitude**2)
        max_line_length = 5000
        scale_factor = min(4000 / 20, max_line_length / storm_speed) if storm_speed > 0 else 1
        scaled_u = storm_u.magnitude * scale_factor
        scaled_v = storm_v.magnitude * scale_factor
        storm_slinky_ax.plot([0, scaled_u], [0, scaled_v], color='black', linewidth=2, label='Storm Motion (RM)')

        if lfc_height is not None and el_height is not None and not np.isnan([lfc_height.magnitude, el_height.magnitude]).any():
            try:
                height_diff = (el_height - lfc_height).to('km').magnitude
                if height_diff == 0:
                    logger.warning("LFC and EL heights are identical, cannot calculate tilt.")
                    angle_deg = 90.0
                else:
                    storm_speed = np.sqrt(storm_u.magnitude**2 + storm_v.magnitude**2)
                    if storm_speed > 0:
                        ascent_time = height_diff * 1000 / 10
                        horizontal_dist = storm_speed * ascent_time
                    else:
                        horizontal_dist = 0
                    angle_deg = np.arctan2(height_diff, horizontal_dist / 1000) * (180 / np.pi)
                    angle_deg = abs(angle_deg)
                storm_slinky_ax.text(4500, 4500, f'Updraft Tilt: {angle_deg:.1f}°', fontsize=8, color='black', ha='right', va='top')
                logger.info(f"Updraft tilt angle calculated: {angle_deg:.1f} degrees with height_diff={height_diff} km, horizontal_dist={horizontal_dist} m")
            except Exception as e:
                logger.error(f"Error calculating updraft tilt angle: {e}")
                angle_deg = 90.0

        storm_slinky_ax.text(4800, -4800, f'Speed: {storm_speed:.1f} m/s', fontsize=6, color='black', ha='right', va='bottom')
        storm_slinky_ax.legend(loc='upper left', fontsize=8, frameon=True)

        def determine_storm_hazard(sbcape, total_helicity1, lcl_height, bshear6_mag, mucape, PWAT, surface_RH):
            hazard = "No Significant Hazard"
            if any(np.isnan(q.magnitude) for q in [sbcape, total_helicity1, lcl_height, bshear6_mag, mucape] if q is not None):
                return hazard
            if PWAT is None or np.isnan(PWAT) or surface_RH is None or np.isnan(surface_RH.magnitude):
                return hazard
            if (sbcape.magnitude > 1000 and total_helicity1.magnitude > 150 and
                lcl_height.magnitude < 1000 and bshear6_mag.magnitude > 30):
                hazard = "Tornado Potential"
            elif (mucape.magnitude > 1500 and bshear6_mag.magnitude > 30 and PWAT > 1.5):
                hazard = "Hail Potential"
            elif (sbcape.magnitude > 1500 and bshear6_mag.magnitude > 25 and surface_RH.magnitude < 60):
                hazard = "Strong Wind Potential"
            return hazard

        storm_hazard = determine_storm_hazard(
            sbcape, total_helicity1, lcl_height, bshear6_mag, mucape, PWAT, surface_RH
        )
        logger.info(f"Determined storm hazard: {storm_hazard}")

        bbox_props = dict(facecolor='#fafad2', alpha=0.7, edgecolor='none', pad=3)
        indices = [
            ('SBCAPE', sbcape, 'red', 0.15, None), ('SBCIN', sbcin, 'purple', 0.13, None),
            ('MLCAPE', mlcape, 'red', 0.11, None), ('MLCIN', mlcin, 'purple', 0.09, None),
            ('MUCAPE', mucape, 'red', 0.07, None), ('MUCIN', mucin, 'purple', 0.05, None),
            ('TT-INDEX', total_totals, 'red', 0.03, None), ('K-INDEX', kindex, 'red', 0.01, None),
            ('SIG TORNADO', sig_tor, 'red', -0.01, None),
            ('0-1km SRH', total_helicity1, 'navy', 0.15, 0.88), ('0-1km SHEAR', bshear1_mag, 'blue', 0.13, 0.88),
            ('0-3km SRH', total_helicity3, 'navy', 0.11, 0.88), ('0-3km SHEAR', bshear3_mag, 'blue', 0.09, 0.88),
            ('0-6km SRH', total_helicity6, 'navy', 0.07, 0.88), ('0-6km SHEAR', bshear6_mag, 'blue', 0.05, 0.88),
            ('SUPERCELL COMP', super_comp, 'red', -0.01, 0.88),
            ('CCL', ccl_height, 'purple', -0.03, 0.88),
            ('LCL', lcl_height, 'black', -0.03, None), ('LFC', lfc_height, 'black', -0.05, None),
            ('EL', el_height, 'black', -0.07, None), ('MU Parcel Level', mpl_height, 'black', -0.09, None),
            ('FL', fl_height, 'blue', -0.11, None),
            ('Surface RH', surface_RH, 'green', -0.13, 0.88),
            ('Surface Wet Bulb', surface_wet_bulb, 'blue', -0.15, 0.88),
            ('PWAT', PWAT * units.inch, 'green', -0.17, 0.88),
            ('Convective Temp', Tc, 'purple', -0.19, 0.88),
            ('Max Temp', max_T, 'red', -0.21, 0.88),
            ('SHIP', ship, 'red', -0.23, 0.88),
        ]

        left_indices = [idx for idx in indices if idx[4] is None]
        right_indices = [idx for idx in indices if idx[4] == 0.88]
        max_rows = max(len(left_indices), len(right_indices))
        y_start = 0.15
        y_step = 0.02
        y_positions = [y_start - i * y_step for i in range(max_rows)]

        def format_value(label, value):
            if label == 'SHIP':
                return f'{value.magnitude:.1f}' if not np.isnan(value.magnitude) else 'N/A'
            if label == 'Surface RH':
                if isinstance(value, pint.Quantity):
                    magnitude = value.magnitude
                else:
                    magnitude = value
                if np.isscalar(magnitude):
                    return f'{magnitude:.0f}%' if not np.isnan(magnitude) else 'N/A'
                return 'N/A'
            elif isinstance(value, pint.Quantity):
                magnitude = value.magnitude
                if not np.isscalar(magnitude):
                    magnitude = magnitude[0] if len(magnitude) == 1 else np.nan
                if label in ['Surface Wet Bulb', 'Convective Temp', 'Max Temp']:
                    return f'{magnitude:.1f} °C' if not np.isnan(magnitude) else 'N/A'
                elif label == 'PWAT':
                    return f'{magnitude:.2f} {value.units}' if not np.isnan(magnitude) else 'N/A'
                elif value.dimensionality == {}:
                    return f'{magnitude:.0f}' if not np.isnan(magnitude) else 'N/A'
                else:
                    return f'{value:.0f~P}' if not np.isnan(magnitude) else 'N/A'
            else:
                return 'N/A' if np.isnan(value) else f'{value:.0f}'

        for i, (label, value, color, _, _) in enumerate(left_indices):
            y = y_positions[i]
            x_left = 0.71
            x_right = 0.80
            value_str = format_value(label, value)
            plt.figtext(x_left, y, f'{label}: ', weight='bold', fontsize=12, color='black', ha='left', bbox=bbox_props)
            plt.figtext(x_right, y, value_str, weight='bold', fontsize=12, color=color, ha='right', bbox=bbox_props)

        for i, (label, value, color, _, _) in enumerate(right_indices):
            y = y_positions[i]
            x_left = 0.88
            x_right = 0.96
            value_str = format_value(label, value)
            plt.figtext(x_left, y, f'{label}: ', weight='bold', fontsize=12, color='black', ha='left', bbox=bbox_props)
            plt.figtext(x_right, y, value_str, weight='bold', fontsize=12, color=color, ha='right', bbox=bbox_props)

        storm_hazard = determine_storm_hazard(
            sbcape, total_helicity1, lcl_height, bshear6_mag, mucape, PWAT, surface_RH
        )
        storm_hazard_text = f"Storm Hazard: {storm_hazard}"
        plt.figtext(
            0.54, 0.05,
            storm_hazard_text,
            ha='left', va='top',
            fontsize=12, fontweight='bold', color='red',
            bbox=dict(facecolor='#fafad2', alpha=0.7, edgecolor='none', pad=3)
        )

        plt.figtext(0.71, 0.23, f"Plot Created With MetPy (C) Evan J Lane 2024\nData Source: {'University of Wyoming' if data_type == 'observed' else f'NOAA ({model.upper()})'}\nImage Created: " +
                    utc_time.strftime('%H:%M UTC'), fontsize=16, fontweight='bold', bbox=bbox_props)
        add_metpy_logo(fig, 85, 85, size='small')

        skew.ax.legend(loc='upper left', fontsize=14, frameon=True, title='Skew-T Legend', title_fontsize=10)
        h.ax.legend(loc='upper left', fontsize=14, frameon=True, title='Hodograph Legend', title_fontsize=10)
        skew.ax.set_facecolor('#d9deda')
        plt.suptitle('UPPER AIR SOUNDING', fontsize=24, fontweight='bold', y=1.00)
        plt.figtext(0.7, 0.98, title, fontsize=20, fontweight='bold', ha='center')
        skew.ax.set_title(f'Skew-T Log-P Diagram - Latitude: {station_lat:.2f}, Longitude: {station_lon:.2f}')

        output_path = os.path.join(OUTPUT_DIR, f"skewt_{station_code}_{data_type}_{utc_time.strftime('%Y%m%d_%H%MZ')}.png")
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Skew-T diagram saved to {output_path}")
        print(f"Skew-T diagram saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Unexpected error in plotting: {e}", exc_info=True)
        print("An unexpected error occurred while generating the plot.")
        return None
