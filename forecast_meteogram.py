#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone FrostByte Forecast Meteogram Tool.

Pulls high-resolution gridded forecast data from the National Weather Service (NWS)
and generates a meteogram matching the FrostByte theme.
"""

import argparse
import asyncio
import os
import re
import logging
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

import metpy.calc as mpcalc
from metpy.units import units
import airportsdata

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

# --- THEME SETTINGS ---
plt.rcParams['font.family'] = 'DejaVu Sans'

FIG_BG_COLOR = '#333333'
AXES_BG_COLOR = '#2B2B2B'
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

# --- METEOROLOGICAL CALCULATIONS ---
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

# --- NWS DATA PARSING ---
def parse_iso_duration(duration_str):
    days, hours = 0, 0
    match_days = re.search(r'P(\d+)D', duration_str)
    match_hours = re.search(r'T(\d+)H', duration_str)
    if match_days: days = int(match_days.group(1))
    if match_hours: hours = int(match_hours.group(1))
    return max(1, (days * 24) + hours)

def get_var_df(grid_props, var_name):
    data = grid_props.get(var_name, {}).get('values', [])
    records = []
    for item in data:
        val = item['value']
        if val is None: val = np.nan
        try:
            start_str, dur_str = item['validTime'].split('/')
            start_time = datetime.fromisoformat(start_str).astimezone(timezone.utc)
            hours = parse_iso_duration(dur_str)
            
            # --- FIXED ACCUMULATION LOGIC ---
            # If the variable is an accumulation (QPF) and the block is longer than 1 hour,
            # divide the total value by the number of hours to get an hourly rate.
            if var_name == 'quantitativePrecipitation' and hours > 0 and not np.isnan(val):
                val = val / hours
            # --------------------------------
                
            for i in range(hours):
                records.append({'time': start_time + timedelta(hours=i), var_name: val})
        except Exception:
            continue
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.groupby('time').mean().reset_index()
    return df

def fetch_nws_forecast(lat, lon, hours_forward):
    headers = {'User-Agent': 'FrostbyteForecastTool/1.0'}
    
    logging.info(f"Fetching NWS grid data for {lat}, {lon}...")
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    response = requests.get(points_url, headers=headers, timeout=10)
    
    if response.status_code == 404:
        raise ValueError(f"NWS API returned 404. Coordinates {lat}, {lon} are likely outside the US. Did you forget a negative sign for longitude?")
    response.raise_for_status()
    
    points_data = response.json()
    
    grid_url = points_data['properties']['forecastGridData']
    logging.info("Downloading hourly gridded forecast variables...")
    grid_response = requests.get(grid_url, headers=headers, timeout=15)
    grid_response.raise_for_status()
    grid_props = grid_response.json()['properties']
    
    variables = [
        'temperature', 'dewpoint', 'relativeHumidity', 'windDirection', 
        'windSpeed', 'windGust', 'probabilityOfPrecipitation', 'skyCover', 'quantitativePrecipitation'
    ]
    
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    df_times = pd.DataFrame({'time': [now_utc + timedelta(hours=i) for i in range(hours_forward)]})
    master_df = df_times

    for var in variables:
        temp_df = get_var_df(grid_props, var)
        if not temp_df.empty:
            master_df = pd.merge(master_df, temp_df, on='time', how='left')
        else:
            master_df[var] = np.nan
            
    if 'windGust' in master_df.columns:
        master_df['windGust'] = master_df['windGust'].fillna(0)
    if 'quantitativePrecipitation' in master_df.columns:
        master_df['quantitativePrecipitation'] = master_df['quantitativePrecipitation'].fillna(0)

    master_df.ffill(inplace=True)
    
    # --- Data Conversions ---
    master_df['tempF'] = (master_df['temperature'] * 1.8) + 32
    master_df['dewF'] = (master_df['dewpoint'] * 1.8) + 32
    
    master_df['wind_speed'] = master_df['windSpeed'] / 1.852
    master_df['wind_gusts'] = master_df['windGust'] / 1.852
    
    master_df['precip_in'] = master_df['quantitativePrecipitation'] / 25.4
    
    master_df['heat_index'] = to_heat_index(master_df['tempF'].values, master_df['dewF'].values)
    master_df['wind_chill'] = to_wind_chill(master_df['tempF'].values, master_df['wind_speed'].values)
    master_df.loc[(master_df['wind_speed'] <= 5) | (master_df['tempF'] > 50), 'wind_chill'] = np.nan
    
    wet_bulb_values = []
    for index, row in master_df.iterrows():
        temp = row['temperature'] 
        dewpoint = row['dewpoint'] 
        if not np.isnan(temp) and not np.isnan(dewpoint):
            try:
                wet_bulb_temp = mpcalc.wet_bulb_temperature(
                    1013.25 * units.hPa, temp * units.degC, dewpoint * units.degC
                ).to('degF').m
                wet_bulb_values.append(wet_bulb_temp)
            except Exception:
                wet_bulb_values.append(np.nan)
        else:
            wet_bulb_values.append(np.nan)
    master_df['wet_bulb'] = wet_bulb_values
    
    return master_df

def generate_forecast_meteogram(df, lat, lon, hours_forward):
    utc_time = datetime.now(timezone.utc)
    
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
    avg_rh = df['relativeHumidity'].mean()
    max_gust = df['wind_gusts'].max()
    total_precip = df['precip_in'].sum()

    active_wind = df[df['wind_speed'] > 0].dropna(subset=['windDirection'])
    if not active_wind.empty:
        wind_dirs_rad = np.deg2rad(active_wind['windDirection'])
        avg_wind_dir = np.rad2deg(np.arctan2(np.mean(np.sin(wind_dirs_rad)), np.mean(np.cos(wind_dirs_rad)))) % 360
    else:
        avg_wind_dir = np.nan

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
        f'Forecast Meteogram for Lat: {lat}, Lon: {lon} - Next {hours_forward} Hours\n(Generated at: {utc_time.strftime("%Y-%m-%d %H:%M UTC")})',
        weight='bold', size='16', color='white', path_effects=TEXT_OUTLINE
    )

    # 0. Temperature Plots
    axs[0].plot(df['time'], df['tempF'], label='Temperature (°F)', linewidth=3, color='#FF6B6B')
    axs[0].plot(df['time'], df['dewF'], label='Dewpoint (°F)', linewidth=3, color='#6BCB77')
    axs[0].plot(df['time'], df['wet_bulb'], label='Wet Bulb (°F)', linewidth=3, linestyle='dotted', color='#4D96FF')
    axs[0].plot(df['time'], df['heat_index'], label='Heat Index (°F)', linestyle='--', color='#C780FA')
    axs[0].plot(df['time'], df['wind_chill'], label='Wind Chill (°F)', linestyle='--', color='cyan')
    axs[0].set_ylabel('Temperature (°F)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
    legend0 = axs[0].legend(loc='upper left', fontsize=10, frameon=True, title='Temperature / Dewpoint')
    style_legend(legend0)

    # 1. Wind Plot (Upgraded Gusts)
    valid_wind_data = df.dropna(subset=['windDirection', 'wind_speed'])
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
        scatter = axs[1].scatter(plot_wind['time'], plot_wind['windDirection'],
                                 c=plot_wind['wind_speed'], cmap='turbo', label='Wind Speed (knots)',
                                 s=150, zorder=2, # <-- Increased size to 150 and explicitly layered below gusts
                                 vmin=vmin if vmin <= vmax else 0, vmax=vmax if vmin <= vmax else 10)
        
        # Plot Gusts
        if not gust_data.empty:
            # Color map the X to match the actual gust speed
            axs[1].scatter(gust_data['time'], gust_data['windDirection'],
                           c=gust_data['wind_gusts'], cmap='turbo', marker='x', s=100, linewidth=2.5,
                           zorder=3, # <-- Explicitly layered on top of the sustained dot
                           vmin=vmin if vmin <= vmax else 0, vmax=vmax if vmin <= vmax else 10,
                           label='Wind Gusts (knots)')
                           
            # Add text annotation for the actual speed right next to the X
            for _, row in gust_data.iterrows():
                y_pos = row['windDirection']
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

    # 2. Probability of Precipitation Plot
    if not df['probabilityOfPrecipitation'].isnull().all():
        axs[2].fill_between(df['time'], 0, df['probabilityOfPrecipitation'], color='#4D96FF', alpha=0.4)
        axs[2].plot(df['time'], df['probabilityOfPrecipitation'], color='#4D96FF', linewidth=2, label='Precip Probability (%)')
        axs[2].set_ylabel('Precip Prob (%)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
        axs[2].set_ylim(0, 100)
        legend2 = axs[2].legend(loc='upper left', fontsize=10, frameon=True, title='PoP')
        style_legend(legend2)
    else:
        axs[2].text(0.5, 0.5, 'Precipitation Probability Unavailable', color='white', ha='center', va='center', transform=axs[2].transAxes)

    # 3. Sky Cover Plot 
    if not df['skyCover'].isnull().all():
        axs[3].fill_between(df['time'], 0, df['skyCover'], color='#8D9EFF', alpha=0.4)
        axs[3].plot(df['time'], df['skyCover'], color='#8D9EFF', linewidth=2, label='Sky Cover (%)')
        axs[3].set_ylabel('Sky Cover (%)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
        axs[3].set_ylim(0, 100)
        legend3 = axs[3].legend(loc='upper left', fontsize=10, frameon=True, title='Cloud Cover')
        style_legend(legend3)
        
    # 4. Precipitation Plot
    if not df['precip_in'].isnull().all():
        bar_width = timedelta(hours=0.8)
        axs[4].bar(df['time'], df['precip_in'], width=bar_width, color='#6BCB77', zorder=2, label='QPF (Rain/Liquid)')
        axs[4].set_ylabel('Precipitation (in)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
        axs[4].set_ylim(bottom=0)
        legend4 = axs[4].legend(loc='upper left', fontsize=10, frameon=True, title='Forecast Precipitation Amount')
        style_legend(legend4)

    # 5. Relative Humidity
    if not df['relativeHumidity'].isnull().all():
        rh_array = df['relativeHumidity'].values
        rh_lower = np.maximum(0, rh_array - 5)
        rh_upper = np.minimum(100, rh_array + 5)
        axs[5].fill_between(df['time'], rh_lower, rh_upper, color='#4D96FF', alpha=0.3)
        axs[5].plot(df['time'], rh_array, color='#4D96FF', label='Relative Humidity (%)', linewidth=2)
        axs[5].set_ylabel('Relative Humidity (%)', color='white', weight='bold', path_effects=TEXT_OUTLINE)
        legend5 = axs[5].legend(loc='upper left', fontsize=10, frameon=True, title='Relative Humidity')
        style_legend(legend5)
        axs[5].set_ylim(0, 100)

    # Summary Text Box
    if not df.empty:
        # Helper function to cleanly handle missing values (N/A) for HI/WC
        def fmt_t(val): return f"{val:.1f} °F" if pd.notna(val) else "N/A"
        
        precip_text = (
            f"--- Forecast Summary ({hours_forward} hrs) ---\n"
            f"Total Precipitation (QPF): {total_precip:.2f}\"\n"
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
        
        if not np.isnan(max_gust) and max_gust > 0:
            precip_text += f"  |  Max Gust: {max_gust:.1f} knots"
            
        precip_text += f"\nAverage RH:  {avg_rh:.1f}%\n"

        # Moved Y coordinate down to -0.45 to fully clear the date tick labels
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
    filename = f'fcst_meteogram_{lat}_{lon}_{utc_time.strftime("%Y%m%d_%H%M")}.png'
    out_path = os.path.join('outputs', filename)
    
    plt.savefig(out_path, format='png', dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    logging.info(f"✅ Generated {out_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Standalone FrostByte Forecast Meteogram Generator")
    parser.add_argument('location', type=str, help="ICAO or Lat/Lon separated by a slash (e.g., KATL or 34.16/-84.79)")
    parser.add_argument('--hoursforward', type=int, default=72, help="Number of hours to forecast forward (default: 72)")
    
    args = parser.parse_args()
    loc_input = args.location.strip()
    
    # 1. Check if the input is a Lat/Lon format
    if '/' in loc_input:
        try:
            lat_str, lon_str = loc_input.split('/')
            lat = round(float(lat_str), 4)
            lon = round(float(lon_str), 4)
        except ValueError:
            logging.error("Invalid location format. Please use 'lat/lon' format (e.g., 34.1652/-84.7954).")
            return
            
    # 2. If it's not a Lat/Lon, assume it's an ICAO code
    else:
        icao = loc_input.upper()
        if len(icao) == 3: 
            icao = 'K' + icao  # Assume US if only 3 letters are provided (e.g., "ATL" -> "KATL")
            
        try:
            airports = airportsdata.load('ICAO')
            if icao in airports:
                lat = round(airports[icao]['lat'], 4)
                lon = round(airports[icao]['lon'], 4)
                logging.info(f"Translated {icao} to Lat: {lat}, Lon: {lon}")
            else:
                logging.error(f"Could not find ICAO code: {icao}")
                return
        except ImportError:
            logging.error("The 'airportsdata' library is not installed. Cannot parse ICAO codes.")
            return

    try:
        df = fetch_nws_forecast(lat, lon, args.hoursforward)
        generate_forecast_meteogram(df, lat, lon, args.hoursforward)
    except Exception as e:
        logging.error(f"Error generating forecast meteogram: {e}", exc_info=True)

if __name__ == "__main__":
    main()