#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Stacked Wind Rose Generator.
Integrated with the Iowa Environmental Mesonet (IEM) ASOS API for historical data.
Refitted for FrostByte Dark Theme Aesthetics with precise legend color mapping,
knot conversions, calm wind filtering, and custom gridline path effects.
"""

import sys
import os
import io
import argparse
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import airportsdata
import requests

# --- APPLYING FROSTBYTE THEME GLOBALS ---
plt.rcParams['font.family'] = 'DejaVu Sans'

FIG_BG_COLOR = '#333333'
AXES_BG_COLOR = '#2B2B2B'
GRIDLINE_COLOR = '#3932A0'

TEXT_OUTLINE = [pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()]
WHITE_OUTLINE = [pe.withStroke(linewidth=3.0, foreground='white'), pe.Normal()]

plt.rcParams.update({
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': 'white',
    'figure.facecolor': FIG_BG_COLOR,
    'axes.facecolor': AXES_BG_COLOR
})
# --- END THEME SETTINGS ---

# Precise Color Extraction from User's Legend
WIND_COLORS = [
    '#41A344', # 1-3 kt (Green)
    '#B5ECA6', # 4-6 kt (Light Green)
    '#D0F2A9', # 7-9 kt (Pale Green)
    '#FDFB5E', # 10-12 kt (Yellow)
    '#FFB443', # 13-15 kt (Orange)
    '#FF4D4D', # 16-18 kt (Red)
    '#A34A4A', # 19-21 kt (Dark Red)
    '#B57474', # 22-24 kt (Brown)
    '#9A429A', # 25-27 kt (Purple)
    '#484848'  # 28+ kt (Dark Grey)
]

LABELS = [
    '1-3 kt', '4-6 kt', '7-9 kt', '10-12 kt', '13-15 kt',
    '16-18 kt', '19-21 kt', '22-24 kt', '25-27 kt', '28+ kt'
]

def parse_date(date_str):
    """Robustly parses date strings supporting both YYYY-MM-DD and YYYYMMDD formats."""
    date_str = date_str.strip()
    for fmt in ('%Y-%m-%d', '%Y%m%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date '{date_str}' does not match format YYYY-MM-DD or YYYYMMDD")

def fetch_iem_data(icao, dt_start, dt_end):
    """Fetches historical wind direction and speed data from IEM ASOS."""
    url = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
           f"station={icao}&data=drct&data=sknt&"
           f"year1={dt_start.year}&month1={dt_start.month}&day1={dt_start.day}&"
           f"year2={dt_end.year}&month2={dt_end.month}&day2={dt_end.day}&"
           f"tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no")
    
    print(f"Fetching data from IEM ASOS Database...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    df = pd.read_csv(io.StringIO(response.text), na_values=['M'])
    df = df.dropna(subset=['drct', 'sknt'])
    return df

def create_windrose(icao, start_date, end_date):
    try:
        icao = icao.upper()
        dt_start = parse_date(start_date)
        dt_end = parse_date(end_date)
        
        df = fetch_iem_data(icao, dt_start, dt_end)
        
        if df.empty:
            print("❌ No valid wind data found for this date range.")
            return

        # Identify and isolate calm winds (0 knots / 0 direction) to fix the massive North spike
        calm_mask = df['sknt'] < 0.5
        calm_pct = (calm_mask.sum() / len(df)) * 100
        
        # Filter dataframe to only plot actual moving air
        df_plot = df[~calm_mask]

        # Use ASOS knots natively
        # Shift bin edges so that integer knots (e.g., 1, 2, 3) fall cleanly into bins
        speed_bins = [0.5, 3.5, 6.5, 9.5, 12.5, 15.5, 18.5, 21.5, 24.5, 27.5, np.inf]
        speeds = df_plot['sknt'].values
        dirs = df_plot['drct'].values

        # Shift directions by 11.25 degrees so that North (0 deg) represents [-11.25, 11.25)
        shifted_dirs = (dirs + 11.25) % 360
        dir_edges = np.arange(0, 360 + 22.5, 22.5)

        # Calculate 2D Histogram counts
        H, _, _ = np.histogram2d(shifted_dirs, speeds, bins=[dir_edges, speed_bins])

        # Convert counts to frequency percentages relative to ALL records (including calm)
        total_counts = len(df)
        H_pct = (H / total_counts) * 100

        fig = plt.figure(figsize=(12, 12), facecolor=FIG_BG_COLOR)
        ax = fig.add_subplot(111, projection='polar')
        ax.set_facecolor(AXES_BG_COLOR)

        # Set 0 degrees (North) to top, and go clockwise
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # 16 standard compass points
        theta = np.deg2rad(np.arange(0, 360, 22.5))
        width = np.deg2rad(22.5)
        bottom = np.zeros(16)

        # Render Stacked Bars
        for i in range(len(speed_bins) - 1):
            radii = H_pct[:, i]
            ax.bar(theta, radii, width=width, bottom=bottom, color=WIND_COLORS[i],
                   edgecolor='none', label=LABELS[i], zorder=3)
            bottom += radii

        # Custom Grid Styling: Purple lines outlined in white
        ax.grid(False) # Turn off default grid to manually rebuild it
        
        # Manually apply path effects to the polar gridlines
        for line in ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines():
            line.set_color(GRIDLINE_COLOR)
            line.set_linewidth(1.5)
            line.set_linestyle(':')
            line.set_alpha(1.0)
            line.set_path_effects(WHITE_OUTLINE)
            line.set_visible(True)
            line.set_zorder(0)

        ax.tick_params(colors='white')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_path_effects(TEXT_OUTLINE)
            label.set_fontweight('bold')

        # Add Calm Winds annotation
        ax.text(0.5, -0.12, f"Calm Winds (< 1 kt): {calm_pct:.1f}%", 
                transform=ax.transAxes, ha='center', va='center', 
                color='white', weight='bold', fontsize=14, 
                bbox=dict(facecolor=AXES_BG_COLOR, edgecolor='white', boxstyle='round,pad=0.5'),
                path_effects=TEXT_OUTLINE)

        # Legend Theme overrides
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.05), facecolor=AXES_BG_COLOR, edgecolor='white')
        for text in legend.get_texts():
            text.set_color('white')
            text.set_path_effects(TEXT_OUTLINE)
            text.set_fontweight('bold')

        # Lookup full airport name for title
        try:
            airports = airportsdata.load('icao')
            airport = airports.get(icao)
            full_name = f"({airport['name']})" if airport else ""
        except Exception:
            full_name = ""

        clean_start = dt_start.strftime('%Y%m%d')
        clean_end = dt_end.strftime('%Y%m%d')

        title_str = f"Wind Rose for {icao} {full_name}\n{clean_start} to {clean_end}"
        ax.set_title(title_str, fontsize=18, color='white', pad=25, weight='bold', path_effects=TEXT_OUTLINE)

        # Dynamic Output Routing
        output_dir = "/media/evan/Main/frostbyte/frostbyte_project/output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Append exact generation time to filename to ensure uniqueness
        gen_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f'windrose_{icao}_{clean_start}_{clean_end}_{gen_time}.png'
        out_path = os.path.join(output_dir, out_name)
        
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        
        # Overwrite generic file locally for GUI backwards compatibility if needed
        plt.savefig("windrose.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"✅ Generated {out_path}")
        plt.close(fig)

    except Exception as e:
        print(f"❌ Error execution sequence failed: {e}")
        traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone FrostByte Wind Rose Generator")
    parser.add_argument("icao", type=str, help="ICAO Airport Code (e.g., YPPH)")
    parser.add_argument("start_date", type=str, help="Start Date (YYYY-MM-DD)")
    parser.add_argument("end_date", type=str, help="End Date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    create_windrose(args.icao, args.start_date, args.end_date)