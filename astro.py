#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Astronomical Data Generator.
Refitted for FrostByte Dark Theme Aesthetics.

This script fetches and plots detailed astronomical information for a given
location and time. It includes:
- Sun and Moon altitude/azimuth paths.
- Sunrise, sunset, and twilight times.
- Moon phase.
- Solar noon and optimal solar panel tilt.
- Top-down and side-on orbital plots for the inner, outer, and
  interstellar solar system, including planets, dwarf planets, and comets.
"""

import ephem
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe 
import numpy as np
import pytz
import sys
from timezonefinder import TimezoneFinder
import airportsdata
from datetime import datetime, timedelta
from io import BytesIO
import logging
import os
import re 
from matplotlib.colors import LinearSegmentedColormap 

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
# --- END THEME SETTINGS ---

def ephem_to_utc_datetime(ephem_date):
    y, m, d, h, mn, s = ephem_date.tuple()
    return datetime(y, m, d, h, mn, int(s), tzinfo=pytz.UTC)

def load_celestial_bodies():
    bodies = {}
    bodies['Pluto'] = ephem.Pluto() 
    ceres_line = "Ceres,e,10.593,80.322,73.168,2.7680,0.21396,0.0760,74.56,2451545.0,3.34,0.12,939,0"
    haumea_line = "Haumea,e,28.216,122.144,239.06,43.132,0.00319,0.1944,213.6,2451545.0,-0.4,0.0,1960,0"
    makemake_line = "Makemake,e,28.983,79.673,293.94,45.430,0.00342,0.1645,355.8,2451545.0,-0.3,0.0,1434,0"
    eris_line = "Eris,e,44.045,35.951,151.91,67.864,0.00176,0.4377,19.8,2451545.0,-1.12,0.0,2326,0"
    
    bodies['Ceres'] = ephem.readdb(ceres_line)
    bodies['Haumea'] = ephem.readdb(haumea_line)
    bodies['Makemake'] = ephem.readdb(makemake_line)
    bodies['Eris'] = ephem.readdb(eris_line)

    halley_line = "1P/Halley,e,162.26,58.42,111.33,17.834,0.01279,0.967,2451545.0,5.5,4.0,20,0"
    bodies['Halley'] = ephem.readdb(halley_line)
    
    lemmon_line = "C/2025 A6 (Lemmon),h,2025/11/08.4852,143.63,132.995,108.098,0.52918,0.99576,2025/11/08.4852,4.5,4.0"
    bodies['Lemmon'] = ephem.readdb(lemmon_line)
    return bodies

ORBITAL_PERIODS = {
    'Ceres': 4.6,
    'Pluto': 248.0,
    'Haumea': 283.0,
    'Makemake': 306.0,
    'Eris': 559.0,
    'Halley': 76.0,
    'Lemmon': 1400.0 
}

def plot_body_orbit_topdown(body, observer, ax, color, orbit_points=500):
    current_date = observer.date
    period_years = ORBITAL_PERIODS.get(body.name, 100.0)
    
    # Tightened overlap multiplier to 1.01 to close loops without messy crossing
    period_days = (period_years * 365.25) * 1.01 
    
    if period_years > 1000: 
        perihelion_jd = 2460987.9852
        perihelion_date = ephem.Date(perihelion_jd - 2415020.0)
        arc_days = 200 * 365.25 
        start_date = perihelion_date - arc_days
        end_date = perihelion_date + arc_days
    else:
        start_date = current_date - period_days
        end_date = current_date
        
    orbit_dates = [ephem.Date(d) for d in np.linspace(start_date, end_date, orbit_points)]
    
    orbit_segments = []
    current_segment = []
    current_linestyle = None
    
    for date in orbit_dates:
        obs_temp = observer.copy()
        obs_temp.date = date
        body.compute(obs_temp)
        
        theta = body.hlon
        r = body.sun_distance
        hlat = body.hlat 
            
        linestyle = '-' if hlat > 0 else '--'
        
        if current_linestyle and linestyle != current_linestyle:
            orbit_segments.append((current_segment, current_linestyle))
            current_segment = [current_segment[-1]] 
            current_linestyle = linestyle
        
        if not current_segment:
            current_linestyle = linestyle
            
        current_segment.append((theta, r))
    
    if current_segment:
        orbit_segments.append((current_segment, current_linestyle))
        
    for segment, linestyle in orbit_segments:
        if len(segment) < 2:
            continue
        thetas, rs = zip(*segment)
        ax.plot(thetas, rs, color=color, lw=1.5, linestyle=linestyle, zorder=2, alpha=0.8)
        
    observer.date = current_date

def plot_body_orbit_sideon(body, observer, ax, color, orbit_points=500):
    current_date = observer.date
    period_years = ORBITAL_PERIODS.get(body.name, 100.0)
    
    # Tightened overlap multiplier to 1.01
    period_days = (period_years * 365.25) * 1.01
    
    if period_years > 1000:
        perihelion_jd = 2460987.9852
        perihelion_date = ephem.Date(perihelion_jd - 2415020.0)
        arc_days = 200 * 365.25
        start_date = perihelion_date - arc_days
        end_date = perihelion_date + arc_days
    else:
        start_date = current_date - period_days
        end_date = current_date
        
    orbit_dates = [ephem.Date(d) for d in np.linspace(start_date, end_date, orbit_points)]
    orbit_points_xy = []
    
    for date in orbit_dates:
        obs_temp = observer.copy()
        obs_temp.date = date
        body.compute(obs_temp)
        
        r = body.sun_distance
        hlon = body.hlon
        hlat = body.hlat
        
        x = r * np.cos(hlon)
        y = r * np.sin(hlat)
        
        orbit_points_xy.append((x, y))
    
    if len(orbit_points_xy) > 1:
        xs, ys = zip(*orbit_points_xy)
        ax.plot(xs, ys, color=color, lw=1.5, zorder=2, alpha=0.8)
        
    observer.date = current_date

# --- Diurnal Coordinate Generator ---
def generate_path_segments(body, times, base_obs, is_southern, freeze_time=None):
    """
    Robust generator for celestial paths using Local Sidereal Time longitude shifting.
    This entirely freezes the object's orbital motion to produce a visually 
    flawless, 100% closed diurnal circle, eliminating spiral disconnects/overlaps.
    """
    segments = []
    current_segment = []
    current_linestyle = None
    
    obs_temp = base_obs.copy()
    base_lon = float(base_obs.lon)
    
    if freeze_time:
        obs_temp.date = ephem.Date(freeze_time)
        frozen_lst = float(obs_temp.sidereal_time())

    for t in times:
        if freeze_time:
            obs_temp.date = ephem.Date(t)
            obs_temp.lon = base_lon
            target_lst = float(obs_temp.sidereal_time())
            
            # Reset time to lock orbital motion, but adjust longitude to 
            # mimic Earth's sidereal rotation for the specific hour.
            obs_temp.date = ephem.Date(freeze_time)
            obs_temp.lon = target_lst - frozen_lst + base_lon
        else:
            obs_temp.date = ephem.Date(t)
            
        body.compute(obs_temp)
        az = np.degrees(body.az)
        if is_southern:
            az = (-az + 180) % 360
        else:
            az = az % 360
        alt = np.degrees(body.alt)
        ls = '-' if alt >= 0 else '--'

        if not current_segment:
            current_linestyle = ls
            current_segment.append((az, alt))
            continue

        prev_az, prev_alt = current_segment[-1]
        
        diff_az = az - prev_az
        is_wrap = abs(diff_az) > 180
        is_cross = (prev_alt * alt <= 0) and (prev_alt != alt)

        if is_wrap:
            segments.append((current_segment, current_linestyle))
            current_linestyle = ls
            current_segment = [(az, alt)]
        elif is_cross:
            frac = -prev_alt / (alt - prev_alt)
            interp_az = prev_az + frac * diff_az
            interp_alt = 0.0
            
            current_segment.append((interp_az, interp_alt))
            segments.append((current_segment, current_linestyle))
            
            current_linestyle = ls
            current_segment = [(interp_az, interp_alt), (az, alt)]
        else:
            current_segment.append((az, alt))
            
    if current_segment:
        segments.append((current_segment, current_linestyle))
        
    return segments

def get_diurnal_az_alt(body, target_time, base_obs, freeze_time, is_southern):
    """Retrieves locked-declination Az/Alt to ensure labels snap perfectly to the diurnal line."""
    obs_t = base_obs.copy()
    base_lon = float(base_obs.lon)
    
    obs_t.date = ephem.Date(freeze_time)
    frozen_lst = float(obs_t.sidereal_time())
    
    obs_t.date = ephem.Date(target_time)
    obs_t.lon = base_lon
    target_lst = float(obs_t.sidereal_time())
    
    obs_t.date = ephem.Date(freeze_time)
    obs_t.lon = target_lst - frozen_lst + base_lon
    
    body.compute(obs_t)
    az_deg = np.degrees(body.az)
    if is_southern:
        az_deg = (-az_deg + 180) % 360
    else:
        az_deg = az_deg % 360
    return az_deg, np.degrees(body.alt)

def generate_astro_plot(location: str, time_str: str = None):
    try:
        logging.info(f"🛰️ Generating astronomical data for {location}...")
        
        if '/' in location:
            lat, lon = map(float, location.split('/'))
            lat, lon = round(lat, 4), round(lon, 4)
            location_str = f"Lat: {lat}, Lon: {lon}"
            loc_file_slug = f"{lat}_{lon}".replace('-', 'M').replace('.', 'p')
        else:
            airports = airportsdata.load('icao')
            airport = airports.get(location.upper())
            if not airport:
                raise ValueError(f"Airport ICAO code '{location.upper()}' not found.")
            lat, lon = round(airport['lat'], 4), round(airport['lon'], 4)
            location_str = f"{location.upper()} ({airport.get('name', 'N/A')})"
            loc_file_slug = location.upper()
            
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=lon, lat=lat)
        if not timezone_str:
            raise ValueError("Could not determine timezone for the given coordinates.")
       
        timezone = pytz.timezone(timezone_str)
        now_local_base = datetime.now(timezone)
        
        if time_str:
            match = re.match(r'^([01]?[0-9]|2[0-3]):([0-5][0-9])$', time_str)
            if not match:
                raise ValueError("Time must be in 'HH:MM' format (e.g., '14:30').")
            try:
                parsed_time = datetime.strptime(time_str, '%H:%M').time()
                local_time = now_local_base.replace(hour=parsed_time.hour, minute=parsed_time.minute, second=0, microsecond=0)
            except ValueError:
                raise ValueError("Invalid time format. Use 'HH:MM'.")
        else:
            local_time = now_local_base
           
        now_utc = local_time.astimezone(pytz.utc)
        
        start_of_day_local = local_time.replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_day_utc = start_of_day_local.astimezone(pytz.utc)
        noon_local = local_time.replace(hour=12, minute=0, second=0, microsecond=0)
        noon_utc = noon_local.astimezone(pytz.utc)
        
        obs = ephem.Observer()
        obs.lat, obs.lon = str(lat), str(lon)
        obs.elevation = 0
       
        fig = plt.figure(figsize=(30, 36))
        fig.set_facecolor(FIG_BG_COLOR)
        
        gs = GridSpec(5, 3, 
                      height_ratios=[2.5, 2, 2, 0.3, 0.8],
                      width_ratios=[1, 1, 1],
                      hspace=0.4, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor(AXES_BG_COLOR)
        
        day_colors = ['#383838', AXES_BG_COLOR]
        for i in range(0, 9): 
            alt_start = i * 10
            alt_end = (i + 1) * 10
            color = day_colors[i % 2]
            ax1.axhspan(alt_start, alt_end, color=color, alpha=0.3, zorder=0)
        
        sun, moon = ephem.Sun(), ephem.Moon()
        obs.date = ephem.Date(noon_utc)
       
        obs.horizon = '0'
        sunrise_utc = obs.previous_rising(sun)
        sunset_utc = obs.next_setting(sun)
        sunrise = ephem_to_utc_datetime(sunrise_utc).astimezone(timezone)
        sunset = ephem_to_utc_datetime(sunset_utc).astimezone(timezone)
       
        obs.horizon = '-6'
        civil_twilight_begin_utc = obs.previous_rising(sun, use_center=True)
        civil_twilight_end_utc = obs.next_setting(sun, use_center=True)
        civil_twilight_begin = ephem_to_utc_datetime(civil_twilight_begin_utc).astimezone(timezone)
        civil_twilight_end = ephem_to_utc_datetime(civil_twilight_end_utc).astimezone(timezone)
       
        obs.horizon = '-12'
        nautical_twilight_begin_utc = obs.previous_rising(sun, use_center=True)
        nautical_twilight_end_utc = obs.next_setting(sun, use_center=True)
        nautical_twilight_begin = ephem_to_utc_datetime(nautical_twilight_begin_utc).astimezone(timezone)
        nautical_twilight_end = ephem_to_utc_datetime(nautical_twilight_end_utc).astimezone(timezone)
       
        obs.horizon = '-18'
        astronomical_twilight_begin_utc = obs.previous_rising(sun, use_center=True)
        astronomical_twilight_end_utc = obs.next_setting(sun, use_center=True)
        astronomical_twilight_begin = ephem_to_utc_datetime(astronomical_twilight_begin_utc).astimezone(timezone)
        astronomical_twilight_end = ephem_to_utc_datetime(astronomical_twilight_end_utc).astimezone(timezone)
       
        obs.horizon = '0'
        obs.date = ephem.Date(now_utc)
        moon.compute(obs)
        moon_phase = moon.phase
        m_next = ephem.next_new_moon(obs.date)
        m_prev = ephem.previous_new_moon(obs.date)
        
        moon_phase_text = f"Moon Phase: {moon_phase:.1f}% illuminated{' (Waning)' if (obs.date - m_prev) > ((m_next - m_prev) / 2) else ' (Waxing)'}"
        
        # --- FIXED MOON GAP & OVERLAP: DIURNAL CIRCLES ---
        # Returning duration down to exactly 24 hours (289 steps). 
        # Using the LST-Shifting generator to draw perfect loop tracks.
        times = [start_of_day_utc + timedelta(minutes=5 * i) for i in range(289)]
        is_southern = lat < 0

        sun_segments = generate_path_segments(sun, times, obs, is_southern, freeze_time=now_utc)
        for segment, linestyle in sun_segments:
            if len(segment) < 2: continue
            azimuths, altitudes = zip(*segment)
            ax1.plot(azimuths, altitudes, color='#FFD700', lw=3, linestyle=linestyle, path_effects=TEXT_OUTLINE, zorder=2)
            
        moon_segments = generate_path_segments(moon, times, obs, is_southern, freeze_time=now_utc)
        for segment, linestyle in moon_segments:
            if len(segment) < 2: continue
            azimuths, altitudes = zip(*segment)
            ax1.plot(azimuths, altitudes, color='white', lw=3, linestyle=linestyle, path_effects=TEXT_OUTLINE, zorder=2)

        for i in range(8):
            hour_to_plot = i * 3
            time_for_label_local = start_of_day_local + timedelta(hours=hour_to_plot)
            time_for_label_utc = time_for_label_local.astimezone(pytz.utc)
            
            # Pull Az/Alt from the frozen diurnal function so labels snap perfectly to the track
            az, alt = get_diurnal_az_alt(sun, time_for_label_utc, obs, now_utc, is_southern)
            label_str = time_for_label_local.strftime('%H:%M')
           
            ax1.text(az, alt + 1.5, label_str, fontsize=9, ha='center', va='bottom',
                     color='white', weight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', fc='#444444', ec='white', alpha=0.8),
                     zorder=5, path_effects=TEXT_OUTLINE) 

        obs.date = ephem.Date(now_utc)
        sun.compute(obs)
        moon.compute(obs)
        
        def get_plot_az(az_deg):
            if is_southern: return (-az_deg + 180) % 360
            else: return az_deg % 360
            
        current_sun_az_deg = np.degrees(sun.az)
        current_sun_az = get_plot_az(current_sun_az_deg)
        current_sun_alt = np.degrees(sun.alt)
        current_moon_az_deg = np.degrees(moon.az)
        current_moon_az = get_plot_az(current_moon_az_deg)
        current_moon_alt = np.degrees(moon.alt)
        
        ax1.axhspan(-90, -18, color='#000000', alpha=0.6, label='Night', zorder=1)
        ax1.axhspan(-18, -12, color='#0a0a2a', alpha=0.5, label='Astronomical Twilight', zorder=1)
        ax1.axhspan(-12, -6, color='#142b47', alpha=0.5, label='Nautical Twilight', zorder=1)
        ax1.axhspan(-6, 0, color='#234973', alpha=0.5, label='Civil Twilight', zorder=1)
       
        ax1.scatter(current_sun_az, current_sun_alt, color='#FFD700', edgecolors='white', s=150, label='Current Sun Position', zorder=10)
        ax1.scatter(current_moon_az, current_moon_alt, color='white', edgecolors='white', s=150, label='Current Moon Position', zorder=10)
        ax1.axhline(0, color='white', linestyle='--', lw=1.5, label='Horizon', zorder=3)
        
        ax1.plot([], [], color='#FFD700', lw=3, linestyle='-', label='Sun Path (above horizon)', path_effects=TEXT_OUTLINE)
        ax1.plot([], [], color='#FFD700', lw=3, linestyle='--', label='Sun Path (below horizon)', path_effects=TEXT_OUTLINE)
        ax1.plot([], [], color='white', lw=3, linestyle='-', label='Moon Path (above horizon)', path_effects=TEXT_OUTLINE)
        ax1.plot([], [], color='white', lw=3, linestyle='--', label='Moon Path (below horizon)', path_effects=TEXT_OUTLINE)
        
        obs.date = ephem.Date(start_of_day_utc)
        solar_noon_utc_ephem = obs.next_transit(sun)
        solar_noon_utc = ephem_to_utc_datetime(solar_noon_utc_ephem)
        solar_noon_local = solar_noon_utc.astimezone(timezone)
       
        obs.date = solar_noon_utc_ephem 
        sun.compute(obs)
        solar_noon_alt = np.degrees(sun.alt)
        optimal_angle_tilt_from_horizontal = 90.0 - solar_noon_alt
       
        time_format = '%Y-%m-%d %I:%M %p %Z'
       
        textstr = (f"Location: {location_str}\n"
                   f"Time: {local_time.strftime(time_format)}\n"
                   f"Sun Az/Alt: {current_sun_az_deg:.1f}° / {current_sun_alt:.1f}°\n"
                   f"Moon Az/Alt: {current_moon_az_deg:.1f}° / {current_moon_alt:.1f}°\n"
                   f"{moon_phase_text}\n\n"
                   f"Sunrise: {sunrise.strftime(time_format)}\n"
                   f"Sunset: {sunset.strftime(time_format)}\n"
                   f"Solar Noon: {solar_noon_local.strftime(time_format)}\n\n"
                   f"Civil Twilight Begin: {civil_twilight_begin.strftime(time_format)}\n"
                   f"Civil Twilight End: {civil_twilight_end.strftime(time_format)}\n"
                   f"Nautical Twilight Begin: {nautical_twilight_begin.strftime(time_format)}\n"
                   f"Nautical Twilight End: {nautical_twilight_end.strftime(time_format)}\n"
                   f"Astronomical Twilight Begin: {astronomical_twilight_begin.strftime(time_format)}\n"
                   f"Astronomical Twilight End: {astronomical_twilight_end.strftime(time_format)}\n\n"
                   f"Optimal Solar Panel Tilt (from horizontal): {optimal_angle_tilt_from_horizontal:.1f}°")
       
        ax1.set_title(f"Sun/Moon Position: {location_str} - {lat:.4f}, {lon:.4f}\n{local_time.strftime('%b %d %Y %H:%M')} Local Time ({timezone_str})",
                      fontsize=16, weight='bold', path_effects=TEXT_OUTLINE) 
        ax1.set_xlabel("Azimuth (degrees)", weight='bold', fontsize=12, path_effects=TEXT_OUTLINE) 
        ax1.set_ylabel("Altitude (degrees)", weight='bold', fontsize=12, path_effects=TEXT_OUTLINE) 
        ax1.set_xlim(0, 360)
        ax1.set_ylim(-90, 90)
        if is_southern:
            ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
            ax1.set_xticklabels(['S\n180°', 'SE\n135°', 'E\n90°', 'NE\n45°', 'N\n0°', 'NW\n315°', 'W\n270°', 'SW\n225°', 'S\n180°'], fontsize=10)
        else:
            ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
            ax1.set_xticklabels(['N\n0°', 'NE\n45°', 'E\n90°', 'SE\n135°', 'S\n180°', 'SW\n225°', 'W\n270°', 'NW\n315°', 'N\n360°'], fontsize=10)
        
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_path_effects(TEXT_OUTLINE)

        leg1 = ax1.legend(fontsize=10, loc='upper right', facecolor=AXES_BG_COLOR, edgecolor='white')
        for text in leg1.get_texts():
            text.set_color('white')
            text.set_path_effects(TEXT_OUTLINE)
            
        ax1.grid(True, color='#555555', linestyle=':', alpha=0.5, zorder=0)
        
        obs.date = ephem.Date(now_utc)
       
        planet_colors = {
            'Mercury': '#B0B0B0', 'Venus': '#EEDD82', 'Earth': '#6BCB77', 'Mars': '#FF6B6B',
            'Ceres': '#A0A0A0',
            'Jupiter': '#FFA500', 'Saturn': '#FFD700', 'Uranus': '#00FFFF', 'Neptune': '#4D96FF', 'Pluto': '#D3D3D3',
            'Haumea': '#D2B48C',
            'Makemake': '#A52A2A',
            'Eris': '#FA8072',
            'Moon': '#FFFFFF',
            'Halley': '#00FFFF',
            'Lemmon': '#FF00FF'
        }
        
        celestial_bodies = load_celestial_bodies() 

        cmap_gravity = LinearSegmentedColormap.from_list(
            'gravity_gradient', 
            [(0, (*plt.cm.colors.to_rgb(GRIDLINE_COLOR), 0.6)), 
             (1, (*plt.cm.colors.to_rgb(GRIDLINE_COLOR), 0.0))]
        )
        gravity_fade_distance = 5.0 

        sideon_inner_data = {}
        sideon_outer_data = {}
        sideon_probes_data = {}
        
        # --- Inner Solar System (TOP-DOWN) ---
        ax2 = fig.add_subplot(gs[1, 0], projection='polar') 
        ax2.set_facecolor(AXES_BG_COLOR)
        
        max_r_ax2 = 3.5
        r_grad_ax2 = np.linspace(0, max_r_ax2, 50)
        theta_grad_ax2 = np.linspace(0, 2 * np.pi, 100)
        r_mesh_ax2, theta_mesh_ax2 = np.meshgrid(r_grad_ax2, theta_grad_ax2)
        norm_data_ax2 = r_mesh_ax2 / gravity_fade_distance
        norm_data_ax2[norm_data_ax2 > 1.0] = 1.0 
        ax2.pcolormesh(theta_mesh_ax2, r_mesh_ax2, norm_data_ax2, cmap=cmap_gravity, shading='gouraud', zorder=1)
        ax2.set_ylim(0, max_r_ax2)
        
        orbit_r_ax2 = 2.0
        orbit_theta_start_ax2 = np.radians(240)
        orbit_theta_end_ax2 = np.radians(210)
        ax2.annotate(
            '', xy=(orbit_theta_end_ax2, orbit_r_ax2), 
            xytext=(orbit_theta_start_ax2, orbit_r_ax2),
            arrowprops=dict(arrowstyle='<-', color='lightgray', lw=2, mutation_scale=20, connectionstyle='arc3,rad=-0.3'),
            zorder=2
        )
        
        ax2.scatter(0, 0, color='#FFD700', s=300, edgecolor='white', label='Sun', zorder=4)
        inner_planets = {
            'Mercury': ephem.Mercury(), 'Venus': ephem.Venus(), 'Earth': 'Earth',
            'Moon': ephem.Moon(), 
            'Mars': ephem.Mars(), 'Ceres': celestial_bodies['Ceres'],
            'Halley': celestial_bodies['Halley'],
            'Lemmon': celestial_bodies['Lemmon']
        }
        
        sun.compute(obs)
        moon.compute(obs)
        earth_hlon_rad = (ephem.Ecliptic(sun).lon + np.pi) % (2 * np.pi)
        earth_r_au = 1.0 
        moon_geocentric_lon_rad = ephem.Ecliptic(moon).lon
        r_orbit_exaggerated_au = 0.08 
        x_earth = earth_r_au * np.cos(earth_hlon_rad)
        y_earth = earth_r_au * np.sin(earth_hlon_rad)
        x_moon_offset = r_orbit_exaggerated_au * np.cos(moon_geocentric_lon_rad)
        y_moon_offset = r_orbit_exaggerated_au * np.sin(moon_geocentric_lon_rad)
        x_moon_total = x_earth + x_moon_offset
        y_moon_total = y_earth + y_moon_offset
        moon_r_plot = np.sqrt(x_moon_total**2 + y_moon_total**2)
        moon_theta_plot = np.arctan2(y_moon_total, x_moon_total)

        for name, planet in inner_planets.items():
            s = 50 if name == 'Moon' else 150
            m = 'p' if 'Halley' in name or 'Lemmon' in name else 'o'
            
            if name == 'Earth':
                theta, r = earth_hlon_rad, earth_r_au
                sideon_inner_data[name] = (r * np.cos(theta), 0.0) 
            elif name == 'Moon':
                theta, r = moon_theta_plot, moon_r_plot
                moon.compute(obs) 
                sideon_inner_data[name] = (r * np.cos(theta), r * np.sin(moon.hlat))
            else:
                planet.compute(obs) 
                theta, r = planet.hlon, planet.sun_distance
                sideon_inner_data[name] = (r * np.cos(theta), r * np.sin(planet.hlat))
            
            if r <= max_r_ax2:
                ax2.scatter(theta, r, color=planet_colors[name], s=s, label=name, zorder=3, marker=m, edgecolor='white')

        theta_values = np.linspace(0, 2 * np.pi, 500)
        ax2.plot(theta_values, [2.7] * 500, '--', color='#C780FA', alpha=0.5, label='Asteroid Belt', lw=2, zorder=2)
        
        plot_body_orbit_topdown(celestial_bodies['Halley'], obs, ax2, color=planet_colors['Halley'], orbit_points=500)
        plot_body_orbit_topdown(celestial_bodies['Lemmon'], obs, ax2, color=planet_colors['Lemmon'], orbit_points=1000)
        plot_body_orbit_topdown(celestial_bodies['Ceres'], obs, ax2, color=planet_colors['Ceres'], orbit_points=500)
        
        ax2.set_yticks([0.5, 1, 2, 3])
        ax2.set_yticklabels(['0.5 AU', '1 AU', '2 AU', '3 AU'], fontsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.set_title("Inner Solar System (Top-Down)", fontsize=16, weight='bold', pad=20, path_effects=TEXT_OUTLINE)
        
        for label in ax2.get_xticklabels() + ax2.get_yticklabels(): 
            label.set_path_effects(TEXT_OUTLINE)

        # --- Outer Solar System (TOP-DOWN) ---
        ax3 = fig.add_subplot(gs[1, 1], projection='polar')
        ax3.set_facecolor(AXES_BG_COLOR)

        max_r_ax3 = 50
        r_grad_ax3 = np.linspace(0, max_r_ax3, 50)
        theta_grad_ax3 = np.linspace(0, 2 * np.pi, 100)
        r_mesh_ax3, theta_mesh_ax3 = np.meshgrid(r_grad_ax3, theta_grad_ax3)
        norm_data_ax3 = r_mesh_ax3 / gravity_fade_distance
        norm_data_ax3[norm_data_ax3 > 1.0] = 1.0 
        ax3.pcolormesh(theta_mesh_ax3, r_mesh_ax3, norm_data_ax3, cmap=cmap_gravity, shading='gouraud', zorder=1)
        ax3.set_ylim(0, max_r_ax3)
        
        orbit_r_ax3 = 25
        orbit_theta_start_ax3 = np.radians(240)
        orbit_theta_end_ax3 = np.radians(210)
        ax3.annotate(
            '', xy=(orbit_theta_end_ax3, orbit_r_ax3), 
            xytext=(orbit_theta_start_ax3, orbit_r_ax3),
            arrowprops=dict(arrowstyle='<-', color='lightgray', lw=2, mutation_scale=20, connectionstyle='arc3,rad=-0.3'),
            zorder=2
        )
        
        ax3.scatter(0, 0, color='#FFD700', s=300, edgecolor='white', label='Sun', zorder=4)
        outer_planets = {
            'Jupiter': ephem.Jupiter(), 'Saturn': ephem.Saturn(), 'Uranus': ephem.Uranus(),
            'Neptune': ephem.Neptune(), 'Pluto': celestial_bodies['Pluto'], 
            'Haumea': celestial_bodies['Haumea'],
            'Makemake': celestial_bodies['Makemake']
        }
        
        for name, planet in outer_planets.items():
            planet.compute(obs)
            theta, r = planet.hlon, planet.sun_distance
            sideon_outer_data[name] = (r * np.cos(theta), r * np.sin(planet.hlat))
            ax3.scatter(theta, r, color=planet_colors[name], s=150, label=name, zorder=3, edgecolor='white')
        
        halley = celestial_bodies['Halley']
        halley.compute(obs)
        theta_h, r_h = halley.hlon, halley.sun_distance
        sideon_outer_data['Halley'] = (r_h * np.cos(theta_h), r_h * np.sin(halley.hlat))
        ax3.scatter(theta_h, r_h, color=planet_colors['Halley'], s=150, label='1P/Halley', zorder=3, marker='p', edgecolor='white')
        plot_body_orbit_topdown(halley, obs, ax3, color=planet_colors['Halley'], orbit_points=500)

        lemmon = celestial_bodies['Lemmon']
        lemmon.compute(obs)
        theta_l, r_l = lemmon.hlon, lemmon.sun_distance
        sideon_outer_data['Lemmon'] = (r_l * np.cos(theta_l), r_l * np.sin(lemmon.hlat))
        ax3.scatter(theta_l, r_l, color=planet_colors['Lemmon'], s=150, label='C/2025 A6 (Lemmon)', zorder=3, marker='p', edgecolor='white')
        plot_body_orbit_topdown(lemmon, obs, ax3, color=planet_colors['Lemmon'], orbit_points=1000)
        
        plot_body_orbit_topdown(celestial_bodies['Pluto'], obs, ax3, color=planet_colors['Pluto'], orbit_points=1000)
        plot_body_orbit_topdown(celestial_bodies['Haumea'], obs, ax3, color=planet_colors['Haumea'], orbit_points=1000)
        plot_body_orbit_topdown(celestial_bodies['Makemake'], obs, ax3, color=planet_colors['Makemake'], orbit_points=1000)

        ax3.set_yticks([5, 10, 20, 30, 40, 50])
        ax3.set_yticklabels(['5 AU', '10 AU', '20 AU', '30 AU', '40 AU', '50 AU'], fontsize=14)
        ax3.tick_params(axis='x', labelsize=14)
        ax3.set_title("Outer Solar System (Top-Down)", fontsize=16, weight='bold', pad=20, path_effects=TEXT_OUTLINE)
        
        for label in ax3.get_xticklabels() + ax3.get_yticklabels(): 
            label.set_path_effects(TEXT_OUTLINE)

        # --- Interstellar Probes (TOP-DOWN) ---
        ax4 = fig.add_subplot(gs[1, 2], projection='polar')
        ax4.set_facecolor(AXES_BG_COLOR)

        max_r_ax4 = 200
        ax4.set_ylim(0, max_r_ax4)
        
        orbit_r_ax4 = 125
        orbit_theta_start_ax4 = np.radians(240)
        orbit_theta_end_ax4 = np.radians(210)
        ax4.annotate(
            '', xy=(orbit_theta_end_ax4, orbit_r_ax4), 
            xytext=(orbit_theta_start_ax4, orbit_r_ax4),
            arrowprops=dict(arrowstyle='<-', color='lightgray', lw=2, mutation_scale=20, connectionstyle='arc3,rad=-0.3'),
            zorder=2
        )
        
        ax4.scatter(0, 0, color='#FFD700', s=100, edgecolor='white', label='Sun', zorder=4)
        
        sideon_probes_data['Voyager 1'] = (167.0 * np.cos(np.radians(350)), 0) 
        sideon_probes_data['Voyager 2'] = (139.0 * np.cos(np.radians(320)), 0) 
        
        ax4.scatter(np.radians(350), 167, color='#FFFFFF', s=100, label='Voyager 1 (167 AU)', zorder=4, marker='^', edgecolor='black')
        ax4.scatter(np.radians(320), 139, color='#FFFFFF', s=100, label='Voyager 2 (139 AU)', zorder=4, marker='v', edgecolor='black')
        ax4.plot(theta_values, [50] * 500, '--', color='#00FFFF', alpha=0.5, label='Kuiper Belt (approx. 30-50 AU)', lw=2, zorder=2)

        eris = celestial_bodies['Eris']
        eris.compute(obs)
        theta_e, r_e = eris.hlon, eris.sun_distance
        sideon_probes_data['Eris'] = (r_e * np.cos(theta_e), r_e * np.sin(eris.hlat))
        ax4.scatter(theta_e, r_e, color=planet_colors['Eris'], s=100, label=f'Eris ({r_e:.1f} AU)', zorder=3, edgecolor='white')
        
        plot_body_orbit_topdown(eris, obs, ax4, color=planet_colors['Eris'], orbit_points=2000)

        ax4.set_yticks([50, 100, 150, 200])
        ax4.set_yticklabels(['50 AU', '100 AU', '150 AU', '200 AU'], fontsize=14)
        ax4.tick_params(axis='x', labelsize=14)
        ax4.set_title("Interstellar Probes (Top-Down)", fontsize=16, weight='bold', pad=20, path_effects=TEXT_OUTLINE)

        for label in ax4.get_xticklabels() + ax4.get_yticklabels(): 
            label.set_path_effects(TEXT_OUTLINE)

        # --- Side-On Plots (Row 3) ---
        
        # --- Inner Solar System (SIDE-ON) ---
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_facecolor(AXES_BG_COLOR)
        
        x_grad_ax5 = np.linspace(-max_r_ax2, max_r_ax2, 50)
        y_grad_ax5 = np.linspace(-max_r_ax2, max_r_ax2, 50)
        x_mesh_ax5, y_mesh_ax5 = np.meshgrid(x_grad_ax5, y_grad_ax5)
        r_mesh_ax5 = np.sqrt(x_mesh_ax5**2 + y_mesh_ax5**2)
        norm_data_ax5 = r_mesh_ax5 / gravity_fade_distance
        norm_data_ax5[norm_data_ax5 > 1.0] = 1.0
        ax5.pcolormesh(x_mesh_ax5, y_mesh_ax5, norm_data_ax5, cmap=cmap_gravity, shading='gouraud', zorder=1)
        
        ax5.axhline(0, color='white', linestyle='--', lw=1, zorder=2, label='Ecliptic Plane')
        ax5.scatter(0, 0, color='#FFD700', s=300, edgecolor='white', label='Sun', zorder=4)
        for name, (x, y) in sideon_inner_data.items():
            s = 50 if name == 'Moon' else 150 
            m = 'p' if 'Halley' in name or 'Lemmon' in name else 'o'
            if abs(x) <= max_r_ax2 and abs(y) <= max_r_ax2:
                ax5.scatter(x, y, color=planet_colors[name], s=s, label=name, zorder=3, marker=m, edgecolor='white')
        
        plot_body_orbit_sideon(celestial_bodies['Halley'], obs, ax5, color=planet_colors['Halley'], orbit_points=500)
        plot_body_orbit_sideon(celestial_bodies['Lemmon'], obs, ax5, color=planet_colors['Lemmon'], orbit_points=1000)
        plot_body_orbit_sideon(celestial_bodies['Ceres'], obs, ax5, color=planet_colors['Ceres'], orbit_points=500)
        
        ax5.set_xlabel("Distance from Sun (AU)", weight='bold', fontsize=12, path_effects=TEXT_OUTLINE)
        ax5.set_ylabel("Height from Ecliptic (AU)", weight='bold', fontsize=12, path_effects=TEXT_OUTLINE)
        ax5.set_title("Inner Solar System (Side-On)", fontsize=16, weight='bold', pad=20, path_effects=TEXT_OUTLINE)
        ax5.set_xlim(-max_r_ax2, max_r_ax2)
        ax5.set_ylim(-max_r_ax2, max_r_ax2)
        ax5.set_aspect('equal') 
        ax5.grid(True, color='#555555', linestyle=':', alpha=0.5, zorder=0)
        
        for label in ax5.get_xticklabels() + ax5.get_yticklabels(): 
            label.set_path_effects(TEXT_OUTLINE)

        # --- Outer Solar System (SIDE-ON) ---
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.set_facecolor(AXES_BG_COLOR)
        
        x_grad_ax6 = np.linspace(-max_r_ax3, max_r_ax3, 50)
        y_grad_ax6 = np.linspace(-max_r_ax3, max_r_ax3, 50)
        x_mesh_ax6, y_mesh_ax6 = np.meshgrid(x_grad_ax6, y_grad_ax6)
        r_mesh_ax6 = np.sqrt(x_mesh_ax6**2 + y_mesh_ax6**2)
        norm_data_ax6 = r_mesh_ax6 / gravity_fade_distance
        norm_data_ax6[norm_data_ax6 > 1.0] = 1.0
        ax6.pcolormesh(x_mesh_ax6, y_mesh_ax6, norm_data_ax6, cmap=cmap_gravity, shading='gouraud', zorder=1)

        ax6.axhline(0, color='white', linestyle='--', lw=1, zorder=2, label='Ecliptic Plane')
        ax6.scatter(0, 0, color='#FFD700', s=300, edgecolor='white', label='Sun', zorder=4)
        for name, (x, y) in sideon_outer_data.items():
            ax6.scatter(x, y, color=planet_colors[name], s=150, label=name, zorder=3, 
                        marker='p' if 'Halley' in name or 'Lemmon' in name else 'o', edgecolor='white')
        
        plot_body_orbit_sideon(celestial_bodies['Halley'], obs, ax6, color=planet_colors['Halley'], orbit_points=500)
        plot_body_orbit_sideon(celestial_bodies['Lemmon'], obs, ax6, color=planet_colors['Lemmon'], orbit_points=1000)
        plot_body_orbit_sideon(celestial_bodies['Pluto'], obs, ax6, color=planet_colors['Pluto'], orbit_points=1000)
        plot_body_orbit_sideon(celestial_bodies['Haumea'], obs, ax6, color=planet_colors['Haumea'], orbit_points=1000)
        plot_body_orbit_sideon(celestial_bodies['Makemake'], obs, ax6, color=planet_colors['Makemake'], orbit_points=1000)
        
        ax6.set_xlabel("Distance from Sun (AU)", weight='bold', fontsize=12, path_effects=TEXT_OUTLINE)
        ax6.set_ylabel("Height from Ecliptic (AU)", weight='bold', fontsize=12, path_effects=TEXT_OUTLINE)
        ax6.set_title("Outer Solar System (Side-On)", fontsize=16, weight='bold', pad=20, path_effects=TEXT_OUTLINE)
        ax6.set_xlim(-max_r_ax3, max_r_ax3)
        ax6.set_ylim(-max_r_ax3, max_r_ax3) 
        ax6.set_aspect('equal') 
        ax6.grid(True, color='#555555', linestyle=':', alpha=0.5, zorder=0)
        
        for label in ax6.get_xticklabels() + ax6.get_yticklabels(): 
            label.set_path_effects(TEXT_OUTLINE)

        # --- Interstellar Probes (SIDE-ON) ---
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.set_facecolor(AXES_BG_COLOR)
        
        ax7.axhline(0, color='white', linestyle='--', lw=1, zorder=2, label='Ecliptic Plane')
        ax7.scatter(0, 0, color='#FFD700', s=100, edgecolor='white', label='Sun', zorder=4)
        for name, (x, y) in sideon_probes_data.items():
            if 'Voyager' in name:
                m = '^' if '1' in name else 'v'
                c = '#FFFFFF'
                ec = 'black'
                s = 100
            else: 
                m = 'o'
                c = planet_colors[name]
                ec = 'white'
                s = 150
            ax7.scatter(x, y, color=c, s=s, label=name, zorder=3, marker=m, edgecolor=ec)
            
        plot_body_orbit_sideon(celestial_bodies['Eris'], obs, ax7, color=planet_colors['Eris'], orbit_points=2000)

        ax7.set_xlabel("Distance from Sun (AU)", weight='bold', fontsize=12, path_effects=TEXT_OUTLINE)
        ax7.set_ylabel("Height from Ecliptic (AU)", weight='bold', fontsize=12, path_effects=TEXT_OUTLINE)
        ax7.set_title("Interstellar Probes (Side-On)", fontsize=16, weight='bold', pad=20, path_effects=TEXT_OUTLINE)
        ax7.set_xlim(-max_r_ax4, max_r_ax4)
        ax7.set_ylim(-max_r_ax4, max_r_ax4) 
        ax7.set_aspect('equal') 
        ax7.grid(True, color='#555555', linestyle=':', alpha=0.5, zorder=0)
        
        for label in ax7.get_xticklabels() + ax7.get_yticklabels(): 
            label.set_path_effects(TEXT_OUTLINE)

        # --- Legends (Row 3) ---
        leg_ax2 = fig.add_subplot(gs[3, 0])
        leg_ax2.set_facecolor(AXES_BG_COLOR)
        handles, labels = ax2.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color=planet_colors['Ceres'], lw=1.5, linestyle='-'))
        labels.append('Ceres Orbit')
        handles.append(plt.Line2D([0], [0], color=planet_colors['Halley'], lw=1.5, linestyle='-'))
        labels.append('Halley Orbit')
        handles.append(plt.Line2D([0], [0], color=planet_colors['Lemmon'], lw=1.5, linestyle='-'))
        labels.append('Lemmon Orbit')
        leg2 = leg_ax2.legend(handles, labels, loc='upper center', frameon=False, ncol=3)
        for text in leg2.get_texts(): 
            text.set_color('white')
            text.set_path_effects(TEXT_OUTLINE)
        leg_ax2.axis('off')
        
        leg_ax3 = fig.add_subplot(gs[3, 1])
        leg_ax3.set_facecolor(AXES_BG_COLOR)
        handles, labels = ax3.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color=planet_colors['Pluto'], lw=1.5, linestyle='-'))
        labels.append('Pluto Orbit')
        handles.append(plt.Line2D([0], [0], color=planet_colors['Haumea'], lw=1.5, linestyle='-'))
        labels.append('Haumea Orbit')
        handles.append(plt.Line2D([0], [0], color=planet_colors['Makemake'], lw=1.5, linestyle='-'))
        labels.append('Makemake Orbit')
        handles.append(plt.Line2D([0], [0], color=planet_colors['Halley'], lw=1.5, linestyle='-'))
        labels.append('Halley Orbit')
        handles.append(plt.Line2D([0], [0], color=planet_colors['Lemmon'], lw=1.5, linestyle='-'))
        labels.append('Lemmon Orbit')
        leg3 = leg_ax3.legend(handles, labels, loc='upper center', frameon=False, ncol=3)
        for text in leg3.get_texts(): 
            text.set_color('white')
            text.set_path_effects(TEXT_OUTLINE)
        leg_ax3.axis('off')
        
        leg_ax4 = fig.add_subplot(gs[3, 2])
        leg_ax4.set_facecolor(AXES_BG_COLOR)
        handles, labels = ax4.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color=planet_colors['Eris'], lw=1.5, linestyle='-'))
        labels.append('Eris Orbit')
        leg4 = leg_ax4.legend(handles, labels, loc='upper center', frameon=False, ncol=3)
        for text in leg4.get_texts(): 
            text.set_color('white')
            text.set_path_effects(TEXT_OUTLINE)
        leg_ax4.axis('off')

        ax_text = fig.add_subplot(gs[4, :])
        ax_text.set_facecolor(AXES_BG_COLOR)
        
        ax_text.text(0.5, 0.95, textstr, fontsize=12, weight='bold', va='top', ha='center',
                     color='white', bbox=dict(boxstyle='round', facecolor=AXES_BG_COLOR, edgecolor='white', alpha=0.9),
                     path_effects=TEXT_OUTLINE) 
        ax_text.axis('off')
        
        plt.tight_layout()
        
        # --- EXPLICIT DUAL-SAVE LOGIC FOR DYNAMIC TITLING ---
        os.makedirs('outputs', exist_ok=True)
        time_clean = local_time.strftime('%H%M')
        dynamic_filename = f"outputs/astro_{loc_file_slug}_{time_clean}.png"
        static_filename = "astro_plot.png"
        
        # Save exact timestamped file
        plt.savefig(dynamic_filename, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
        
        # Overwrite generic static file for GUI compatibility
        plt.savefig(static_filename, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
        
        plt.close(fig)
        
        logging.info(f"✅ Generated {dynamic_filename}")
        print("\n" + "="*30)
        print("Astronomical Data")
        print("="*30)
        print(textstr)
        print("="*30)

    except (AttributeError, ValueError, TypeError) as e:
        logging.error(f"Error retrieving astronomy information: {e}", exc_info=True)
        print(f"Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")

def print_help():
    help_text = """
**Astro Script Help**
Provides sunrise, sunset, moon phase, twilight info, and solar system overview.

**Usage:**
Run the script and follow the prompts.

**Arguments:**
1. `location`: (Required) An ICAO airport code or a lat/lon pair.
   - *Examples:* `kmge`, `egll`, `34.0522/-118.2437`
2. `time`: (Optional) Time in 'HH:MM' format (24-hour local time).
   - *Example:* `14:30`
   - *Default:* Defaults to the current local time at the specified location.
"""
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        location_arg = sys.argv[1]
        time_arg = sys.argv[2] if len(sys.argv) > 2 else None
        generate_astro_plot(location_arg, time_arg)
        sys.exit(0)

    print("--- Standalone Astronomical Plot Generator ---")
    print("Type 'help' for instructions or 'quit' to exit.")
    
    try:
        while True:
            location = input("\nEnter location (ICAO or lat/lon): ").strip()

            if not location: continue
            if location.lower() == 'help':
                print_help()
                continue
            if location.lower() == 'quit': break
                
            time_str = input("Enter time (HH:MM, press Enter for now): ").strip()
            generate_astro_plot(location, time_str or None)

    except KeyboardInterrupt:
        print("\nExiting script.")
    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)
        print(f"A critical error occurred: {e}")