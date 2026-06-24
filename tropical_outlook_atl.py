import os
import logging
import aiohttp
import asyncio
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime, timedelta, timezone
import re
import requests
from io import BytesIO
import matplotlib.image as mpimg
from bs4 import BeautifulSoup
import zipfile
import shapefile  # Requires pyshp: pip install pyshp
from textwrap import wrap
from PIL import Image as PILImage
import math
import textwrap  
from matplotlib import patheffects
import xml.etree.ElementTree as ET
import numpy as np
from matplotlib.legend_handler import HandlerBase
import xarray as xr
import netCDF4 as nc

# Import Tropycal for active storm fetching
from tropycal import realtime as realtime_obj

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Icon directory
icon_dir = '/media/evan/Main/frostbyte/frostbyte_project/icons'

# Define geostationary projection for GOES-19 (Atlantic)
proj = ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0)
sat_height = 35786023.0
half_angle = 0.151872
x_max = sat_height * math.tan(half_angle)
y_max = x_max
y_min = -x_max
x_min = -x_max

# Define the map extent for filtering (Atlantic)
EXTENT_LON_MIN = -100
EXTENT_LON_MAX = -15
EXTENT_LAT_MIN = 5
EXTENT_LAT_MAX = 50

class ImageHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        img_path = orig_handle.get_label()
        
        fallback_artist = plt.Line2D([0], [0], marker='none', linestyle='none')
        
        if not img_path or not os.path.exists(img_path):
            return [fallback_artist]

        try:
            img = mpimg.imread(img_path)
            # Dropped zoom from 0.0267 to 0.018 to prevent vertical collision
            imagebox = OffsetImage(img, zoom=0.018)
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            ab = AnnotationBbox(imagebox, center,
                                xycoords=trans,
                                box_alignment=(0.5, 0.5),
                                frameon=False,
                                pad=0)
            return [ab]
        except Exception as e:
            logging.warning(f"Failed to render icon inside legend handler for path '{img_path}': {e}")
            return [fallback_artist]

def fetch_ocean_currents(extent):
    print("\n--- MARINE DATA FETCH (RTOFS Direct Download with Cache) ---")
    import time
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
        logging.info("Attempting to fetch RTOFS data via direct HTTPS...")
        today = datetime.now(timezone.utc)
        dates_to_try = [today, today - timedelta(days=1), today - timedelta(days=2)]
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 Frostbyte/1.0'})
        files_to_try = ["rtofs_glo_2ds_f000_prog.nc", "rtofs_glo_2ds_f024_prog.nc", "rtofs_glo_2ds.f000.nc", "rtofs_glo_2ds.f024.nc"]

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
                    print("Downloading (this takes a moment but only runs ONCE)...", end=" ", flush=True)
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
            mask = (lats_raw >= lat_min) & (lats_raw <= lat_max) & (lons_raw >= lon_min) & (lons_raw <= lon_max)
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

        sst_var = next((v for v in ['sst', 'SST', 'temperature', 'temp', 'sea_surface_temperature'] if v in ds.variables), None)
        u_var   = next((v for v in ['u_velocity', 'u', 'water_u', 'U', 'u_comp', 'velocity_u', 'surf_u'] if v in ds.variables), None)
        v_var   = next((v for v in ['v_velocity', 'v', 'water_v', 'V', 'v_comp', 'velocity_v', 'surf_v'] if v in ds.variables), None)
        sal_var = next((v for v in ['sss', 'SSS', 'salinity', 'salt', 'Salinity'] if v in ds.variables), None)

        def safe_slice(vname):
            if not vname: return None
            var_obj = ds.variables[vname]
            ndim = len(var_obj.shape)
            if ndim == 2: return var_obj[y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
            if ndim == 3: return var_obj[0, y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
            if ndim == 4: return var_obj[0, 0, y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]
            return None

        w_temp = safe_slice(sst_var)
        w_temp = np.asarray(w_temp, dtype=np.float32) if w_temp is not None else np.full(lats.shape, np.nan, dtype=np.float32)
        w_u = safe_slice(u_var)
        w_u = np.asarray(w_u, dtype=np.float32) if w_u is not None else np.full_like(w_temp, np.nan)
        w_v = safe_slice(v_var)
        w_v = np.asarray(w_v, dtype=np.float32) if w_v is not None else np.full_like(w_temp, np.nan)
        salinity = safe_slice(sal_var)
        salinity = np.asarray(salinity, dtype=np.float32) if salinity is not None else np.full_like(w_temp, np.nan)

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

async def fetch_satellite_image(session):
    current_utc = datetime.now(timezone.utc)
    minutes = current_utc.minute
    rounded_min = (minutes // 15) * 15
    fetch_time = current_utc.replace(minute=rounded_min, second=0, microsecond=0)

    for delta in range(0, 45, 15):
        try_time = fetch_time - timedelta(minutes=delta)
        year = try_time.strftime('%Y')
        julian_day = try_time.strftime('%j')
        hour = try_time.strftime('%H')
        minute = try_time.strftime('%M').zfill(2)
        timestamp = f"{year}{julian_day}{hour}{minute}"

        url = f"https://cdn.star.nesdis.noaa.gov/GOES19/ABI/FD/GEOCOLOR/{timestamp}_GOES19-ABI-FD-GEOCOLOR-1808x1808.jpg"
        logging.info(f"Trying direct fetch: {url}")
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    if response.headers.get('Content-Type', '').startswith('image/'):
                        img = PILImage.open(BytesIO(content)).convert('RGB')
                        return img
        except Exception as e:
            logging.warning(f"Error in direct fetch for {url}: {e}")

    base_url = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/FD/GEOCOLOR/"
    pattern = r'(\d{11}_GOES19-ABI-FD-GEOCOLOR-1808x1808\.jpg)'
    try:
        async with session.get(base_url) as response:
            if response.status == 200:
                html_content = await response.text()
                available_files = re.findall(pattern, html_content)
                if available_files:
                    available_times = []
                    for filename in available_files:
                        try:
                            timestamp_str = filename.split('_')[0]
                            ts = datetime.strptime(timestamp_str, '%Y%j%H%M').replace(tzinfo=timezone.utc)
                            if ts <= current_utc:
                                available_times.append((ts, filename))
                        except ValueError:
                            continue
                    if available_times:
                        available_times.sort(reverse=True)
                        latest_filename = available_times[0][1]
                        fallback_url = f"{base_url}{latest_filename}"
                        async with session.get(fallback_url) as resp:
                            if resp.status == 200:
                                content = await resp.read()
                                img = PILImage.open(BytesIO(content)).convert('RGB')
                                return img
    except Exception as e:
        logging.error(f"Directory fallback failed: {e}")

    rammb_url = "https://rammb-data.cira.colostate.edu/tc_realtime/products/general/atl/geocolor/latest.jpg"
    try:
        async with session.get(rammb_url) as response:
            if response.status == 200:
                content = await response.read()
                if response.headers.get('Content-Type', '').startswith('image/'):
                    img = PILImage.open(BytesIO(content)).convert('RGB')
                    return img
    except Exception as e:
        logging.error(f"RAMMB fallback failed: {e}")

    return None

async def fetch_shapefile_data(session):
    url = "https://www.nhc.noaa.gov/xgtwo/gtwo_shapefiles.zip"
    disturbances = []
    zero_percent_points = []
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return [], []
            zip_data = await response.read()

        with zipfile.ZipFile(BytesIO(zip_data)) as zf:
            namelist = zf.namelist()
            area_files = [f for f in namelist if f.startswith('gtwo_areas_') and f.endswith('.shp')]
            if not area_files:
                return [], []

            area_base = area_files[0].rsplit('.', 1)[0]
            shp_data = zf.read(area_base + '.shp')
            dbf_data = zf.read(area_base + '.dbf')
            shx_data = zf.read(area_base + '.shx')

            sf = shapefile.Reader(shp=BytesIO(shp_data), dbf=BytesIO(dbf_data), shx=BytesIO(shx_data))

            field_names = [field[0] for field in sf.fields[1:]]
            prob7_index = field_names.index('PROB7DAY') if 'PROB7DAY' in field_names else None

            for i in range(len(sf.shapes())):
                shape = sf.shapes()[i]
                record = sf.records()[i]

                if prob7_index is None:
                    continue

                chance_str = record[prob7_index]
                
                try:
                    if isinstance(chance_str, str):
                        chance_clean = chance_str.replace('%', '').strip()
                        chance = int(chance_clean) if chance_clean else 0
                    else:
                        chance = int(chance_str)
                except (ValueError, TypeError):
                    chance = 0

                if chance == 0:
                    continue

                coords = [(point[0], point[1]) for point in shape.points]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                center_lon = sum(lons) / len(lons)
                center_lat = sum(lats) / len(lats)

                if EXTENT_LON_MIN <= center_lon <= EXTENT_LON_MAX and EXTENT_LAT_MIN <= center_lat <= EXTENT_LAT_MAX:
                    disturbances.append({
                        'polygon': coords,
                        'formation_chance': chance,
                        'x_marker': None,
                    })

            point_files = [f for f in namelist if f.startswith('gtwo_points_') and f.endswith('.shp')]
            if point_files:
                point_base = point_files[0].rsplit('.', 1)[0]
                p_shp_data = zf.read(point_base + '.shp')
                p_dbf_data = zf.read(point_base + '.dbf')
                p_shx_data = zf.read(point_base + '.shx')

                p_sf = shapefile.Reader(shp=BytesIO(p_shp_data), dbf=BytesIO(p_dbf_data), shx=BytesIO(p_shx_data))
                p_field_names = [field[0] for field in p_sf.fields[1:]]
                p_prob7_index = p_field_names.index('PROB7DAY') if 'PROB7DAY' in p_field_names else None

                for j in range(len(p_sf.shapes())):
                    p_shape = p_sf.shapes()[j]
                    p_record = p_sf.records()[j]

                    if p_prob7_index is None:
                        continue

                    p_chance_str = p_record[p_prob7_index]
                    
                    try:
                        if isinstance(p_chance_str, str):
                            p_chance_clean = p_chance_str.replace('%', '').strip()
                            p_chance = int(p_chance_clean) if p_chance_clean else 0
                        else:
                            p_chance = int(p_chance_str)
                    except (ValueError, TypeError):
                        p_chance = 0
                        
                    x_lon, x_lat = p_shape.points[0]

                    if EXTENT_LON_MIN <= x_lon <= EXTENT_LON_MAX and EXTENT_LAT_MIN <= x_lat <= EXTENT_LAT_MAX:
                        if p_chance > 0:
                            matched = False
                            for dist in disturbances:
                                if dist['formation_chance'] == p_chance:
                                    dist['x_marker'] = (x_lon, x_lat)
                                    matched = True
                                    break
                        else:
                            zero_percent_points.append({
                                'lon': x_lon,
                                'lat': x_lat,
                                'chance': p_chance
                            })

        return disturbances, zero_percent_points
    except Exception as e:
        logging.error(f"An unexpected error occurred in fetch_shapefile_data: {e}")
        return [], []

def get_color_from_chance(chance):
    if chance < 40: return 'yellow'
    elif chance <= 60: return 'orange'
    else: return 'red'

def get_category_color(vmax_kt):
    if vmax_kt < 34: return '#00BFFF', '^' 
    elif vmax_kt < 64: return '#00FA9A', 'o' 
    elif vmax_kt < 83: return '#FFD700', 'o' 
    elif vmax_kt < 96: return '#FFA500', 'o' 
    elif vmax_kt < 113: return '#FF4500', 'o' 
    elif vmax_kt < 130: return '#DA70D6', 'o' 
    else: return '#FF00FF', 'o'

def is_within_extent(lon, lat):
    return (EXTENT_LON_MIN <= lon <= EXTENT_LON_MAX and EXTENT_LAT_MIN <= lat <= EXTENT_LAT_MAX)

async def seven_day_outlook_ATL(output_path):
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir and out_dir.strip() != "":
            os.makedirs(out_dir, exist_ok=True)
            logging.info(f"Output directory ensured: {out_dir}") 
    except Exception as e:
        logging.error(f"Error creating output directory: {e}")
        raise e

    # Fetch RTOFS data synchronously but in a separate thread
    extent_list = [EXTENT_LON_MIN, EXTENT_LON_MAX, EXTENT_LAT_MIN, EXTENT_LAT_MAX]
    lats, lons, w_temp, w_u, w_v, salinity = await asyncio.to_thread(fetch_ocean_currents, extent_list)

    async with aiohttp.ClientSession() as session:
        try:
            disturbances, zero_percent_points = await fetch_shapefile_data(session)
            img = await fetch_satellite_image(session)

            # --- EXPLICIT MARGIN CALCULATIONS ---
            fig_width = 16.0
            left_margin_in = 0.5
            right_margin_in = 0.5
            top_margin_in = 1.6  # Room for title, legend, and logo
            bottom_margin_in = 0.5

            ax_width_in = fig_width - left_margin_in - right_margin_in
            map_aspect = 1.8 # Approximation for the rectangular Atlantic extent
            ax_height_in = ax_width_in / map_aspect

            fig_height = ax_height_in + top_margin_in + bottom_margin_in

            fig = plt.figure(figsize=(fig_width, fig_height), facecolor='#333333')

            ax_x0 = left_margin_in / fig_width
            ax_y0 = bottom_margin_in / fig_height
            ax_w = ax_width_in / fig_width
            ax_h = ax_height_in / fig_height

            ax = fig.add_axes([ax_x0, ax_y0, ax_w, ax_h], projection=proj)
            ax.set_facecolor('#2B2B2B')
            ax.set_extent([EXTENT_LON_MIN, EXTENT_LON_MAX, EXTENT_LAT_MIN, EXTENT_LAT_MAX], crs=ccrs.PlateCarree())

            text_outline = [patheffects.Stroke(linewidth=2.5, foreground='black'), patheffects.Normal()]

            # --- START SST OVERLAY ---
            try:
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
                sst_cmap = mcolors.LinearSegmentedColormap.from_list('ga_temp_cmap', normalized_stops)
                
                sst_norm = mcolors.Normalize(vmin=-73.333, vmax=54.444)

                sst_url = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.2026.nc"
                with xr.open_dataset(sst_url) as ds:
                    sst = ds['sst'].isel(time=-1)
                    
                    sst = sst.sortby('lat').sortby('lon')
                    
                    lon_min_360 = 360 + EXTENT_LON_MIN - 5
                    lon_max_360 = 360 + EXTENT_LON_MAX + 5
                    
                    sst = sst.sel(lat=slice(EXTENT_LAT_MIN - 5, EXTENT_LAT_MAX + 5), 
                                  lon=slice(lon_min_360, lon_max_360))
                    
                    # User-modified levels kept intact
                    levels = np.arange(1, 35, 1)

                    sst_fill = ax.contourf(sst.lon, sst.lat, sst, 
                                           transform=ccrs.PlateCarree(),
                                           levels=levels, 
                                           cmap=sst_cmap, 
                                           norm=sst_norm,
                                           alpha=1.0, 
                                           zorder=0.5) 

                    sst_contours = ax.contour(sst.lon, sst.lat, sst, 
                                              transform=ccrs.PlateCarree(),
                                              levels=levels, 
                                              cmap=sst_cmap, 
                                              norm=sst_norm,
                                              linewidths=1.5, 
                                              alpha=0.9,
                                              zorder=2.5)
                    
                    clabels = ax.clabel(sst_contours, inline=True, fontsize=9, fmt='%1.0f°C')
                    for label in clabels:
                        label.set_path_effects(text_outline)
                        label.set_fontweight('bold')

            except Exception as e:
                logging.error(f"Error processing SST overlay: {e}")
            # --- END SST OVERLAY ---

            # --- START OCEAN CURRENTS ---
            if lats is not None and w_u is not None and w_temp is not None:
                try:
                    stride = 4
                    w_temp_c = w_temp - 273.15 # Convert to Celsius for Jet Colormap
                    strm = ax.streamplot(lons[::stride, ::stride], lats[::stride, ::stride], 
                                         w_u[::stride, ::stride], w_v[::stride, ::stride], 
                                         color=w_temp_c[::stride, ::stride], cmap='jet', 
                                         density=1.2, linewidth=1.0, arrowsize=1.2,
                                         transform=ccrs.PlateCarree(), zorder=1.8)
                    strm.lines.set_path_effects([patheffects.Stroke(linewidth=2.5, foreground='black'), patheffects.Normal()])
                except Exception as e:
                    logging.error(f"Marine Plotting Error: {e}")
            # --- END OCEAN CURRENTS ---

            if img is not None:
                ax.imshow(img, origin='upper', extent=(x_min, x_max, y_min, y_max), transform=proj, zorder=1, interpolation='bilinear', alpha=0.8)

            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='white', zorder=2)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='white', zorder=2)
            ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='white', zorder=2)

            line_outline = [patheffects.Stroke(linewidth=1.5, foreground='black'), patheffects.Normal()]

            gl = ax.gridlines(
                draw_labels=True, 
                alpha=0.6, 
                color='#3932A0', 
                linestyle='--',
                zorder=3,
                path_effects=line_outline
            )
            gl.top_labels = False
            gl.right_labels = False
            
            gl.xlabel_style = {'size': 10, 'color': 'white', 'path_effects': text_outline}
            gl.ylabel_style = {'size': 10, 'color': 'white', 'path_effects': text_outline}

            for dist in disturbances:
                color = get_color_from_chance(dist['formation_chance'])
                polygon_shape = Polygon(dist['polygon'], facecolor=color, alpha=0.4, hatch='////', edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree(), zorder=4)
                ax.add_patch(polygon_shape)

                lons = [c[0] for c in dist['polygon']]
                lats = [c[1] for c in dist['polygon']]
                marker_lon = dist['x_marker'][0] if dist.get('x_marker') else (sum(lons) / len(lons))
                marker_lat = dist['x_marker'][1] if dist.get('x_marker') else (sum(lats) / len(lats))

                if is_within_extent(marker_lon, marker_lat):
                    if dist.get('x_marker'):
                        arrow = FancyArrowPatch((marker_lon, marker_lat), (sum(lons)/len(lons), sum(lats)/len(lats)), connectionstyle="arc3,rad=0.3", arrowstyle="->,head_width=4.0,head_length=4.0", color=color, linewidth=8, transform=ccrs.PlateCarree(), zorder=4.5)
                        ax.add_patch(arrow)

                    ax.text(marker_lon, marker_lat, 'X', transform=ccrs.PlateCarree(), fontsize=18, fontweight='bold', ha='center', va='center', color=color, zorder=5, path_effects=text_outline)
                    ax.text(marker_lon + 1, marker_lat, f"{dist['formation_chance']}%", 
                            transform=ccrs.PlateCarree(), fontsize=14, ha='left', va='center', 
                            fontweight='bold', color='white', zorder=6, path_effects=text_outline)

            for point in zero_percent_points:
                if is_within_extent(point['lon'], point['lat']):
                    color = get_color_from_chance(point['chance'])
                    ax.text(point['lon'], point['lat'], 'X', transform=ccrs.PlateCarree(), fontsize=18, fontweight='bold', ha='center', va='center', color=color, zorder=5, path_effects=text_outline)
                    ax.text(point['lon'] + 1, point['lat'], f"{point['chance']}%", 
                            transform=ccrs.PlateCarree(), fontsize=14, ha='left', va='center', 
                            fontweight='bold', color='white', zorder=6, path_effects=text_outline)

            # --- CHECK FOR & PLOT ACTIVE STORMS ---
            rt = realtime_obj.Realtime()
            active_storms = rt.list_active_storms(basin='north_atlantic')
            
            for storm_id in active_storms:
                try:
                    storm = rt.get_storm(storm_id)
                    # Handle cases where the coordinates/winds might be lists or single values
                    lat = storm.lat[-1] if isinstance(storm.lat, (list, np.ndarray)) else storm.lat
                    lon = storm.lon[-1] if isinstance(storm.lon, (list, np.ndarray)) else storm.lon
                    vmax = storm.vmax[-1] if isinstance(storm.vmax, (list, np.ndarray)) else storm.vmax
                    
                    # Map vmax to the correct custom icon
                    if vmax < 34:
                        tc_type = 'tropical_depression'
                    elif vmax < 64:
                        tc_type = 'tropical_storm'
                    else:
                        tc_type = 'hurricane'

                    icon_path = os.path.join(icon_dir, f"{tc_type}.png")
                    plotted_icon = False

                    # Attempt to plot the custom image
                    if os.path.exists(icon_path):
                        try:
                            arr = mpimg.imread(icon_path)
                            zoom = 0.05  # Adjust this if the icons are too big/small
                            
                            # Safely project lat/lon to map coordinates for the AnnotationBbox
                            transform_result = proj.transform_points(ccrs.PlateCarree(), np.array([lon]), np.array([lat]))
                            projected_xy = transform_result[0, :2]

                            if not np.any(np.isnan(projected_xy)):
                                imagebox = OffsetImage(arr, zoom=zoom)
                                ab = AnnotationBbox(imagebox, projected_xy, xycoords='data', frameon=False, box_alignment=(0.5, 0.5), pad=0, zorder=10)
                                ax.add_artist(ab)
                                plotted_icon = True
                        except Exception as e:
                            logging.warning(f"Error plotting custom icon for {storm_id}: {e}")
                    
                    # Fallback to the standard dot if the icon is missing or fails to load
                    if not plotted_icon:
                        color, marker = get_category_color(vmax)
                        ax.scatter(lon, lat, color=color, marker=marker, s=250, edgecolors='black', linewidths=1.5, transform=ccrs.PlateCarree(), zorder=10)
                    
                    # Add the storm label below it
                    ax.text(lon, lat - 1.5, f"{storm.name}\n{vmax} kt", color='white', fontweight='bold', ha='center', path_effects=text_outline, transform=ccrs.PlateCarree(), zorder=11)
                except Exception as e:
                    logging.error(f"Could not plot active storm {storm_id}: {e}")

            # Silence text if storms are active OR disturbances are present
            if len(disturbances) == 0 and len(zero_percent_points) == 0 and len(active_storms) == 0:
                ax.text(0.5, 0.5, 'Tropical cyclone formation is not expected during the next 7 days.', transform=ax.transAxes, fontsize=16, ha='center', va='center', fontweight='bold', color='white', zorder=10)

            now = datetime.now()
            time_str = now.strftime("%I:%M %p EDT %a %b %d %Y")
            title_str = f'Seven-Day Graphical Tropical Weather Outlook\nNational Hurricane Center Miami, FL\nValid: {time_str}'
            
            # Place title dynamically in the top margin, aligned left with the axes
            title_y = ax_y0 + ax_h + (top_margin_in / 2.5) / fig_height 
            fig.text(ax_x0, title_y, title_str, fontsize=14, fontweight='bold', color='white', va='center', ha='left', path_effects=text_outline)

            td_icon_path = os.path.join(icon_dir, 'tropical_depression.png')
            ts_icon_path = os.path.join(icon_dir, 'tropical_storm.png')
            hu_icon_path = os.path.join(icon_dir, 'hurricane.png')

            td_handle = plt.Line2D([0], [0], marker='none', linestyle='none', label=td_icon_path)
            ts_handle = plt.Line2D([0], [0], marker='none', linestyle='none', label=ts_icon_path)
            hu_handle = plt.Line2D([0], [0], marker='none', linestyle='none', label=hu_icon_path)

            legend_labels = {td_handle: 'Tropical Depression', ts_handle: 'Tropical Storm', hu_handle: 'Hurricane'}
            handler_map = {td_handle: ImageHandler(), ts_handle: ImageHandler(), hu_handle: ImageHandler()}

            # --- LEGEND & LOGO ALIGNMENT ---
            legend_elements = [
                plt.Line2D([0], [0], marker='x', color='none', markeredgecolor='yellow', markerfacecolor='black', markersize=8, label='Formation chance <40%'),
                plt.Line2D([0], [0], marker='x', color='none', markeredgecolor='orange', markerfacecolor='black', markersize=8, label='Formation chance 40-60%'),
                plt.Line2D([0], [0], marker='x', color='none', markeredgecolor='red', markerfacecolor='black', markersize=8, label='Formation chance >60%'),
                td_handle, ts_handle, hu_handle
            ]

            # Scaled down fontsize and padding to match the logo's height proportionally
            leg = ax.legend(handles=legend_elements,
                      labels=[h.get_label() if h not in legend_labels else legend_labels[h] for h in legend_elements],
                      handler_map=handler_map, loc='upper right', bbox_to_anchor=(0.91, 1.14),
                      fontsize=7.5, facecolor='#2B2B2B', framealpha=0.8, edgecolor='black',
                      labelspacing=0.25, borderpad=0.4, handletextpad=0.4, handlelength=1.0)
            
            # Use a thinner outline for the smaller text
            legend_outline = [patheffects.Stroke(linewidth=1.2, foreground='black'), patheffects.Normal()]

            for text in leg.get_texts():
                text.set_color('white')
                text.set_fontweight('bold')
                text.set_path_effects(legend_outline)

            logo_path = os.path.join(icon_dir, 'boxlogo2.png')

            if os.path.exists(logo_path):
                logo_img = mpimg.imread(logo_path)
                imagebox = OffsetImage(logo_img, zoom=0.07) 
                
                # Locked to Y=1.14 to sit exactly flush with the top of the legend
                ab = AnnotationBbox(imagebox, (0.92, 1.14), xycoords='axes fraction', 
                                    box_alignment=(0, 1), frameon=False, pad=0, zorder=12)
                ax.add_artist(ab)

            out_dir, out_file = os.path.split(output_path)
            base_name, ext = os.path.splitext(out_file)
            if not ext:
                ext = ".png"
                
            timestamp = now.strftime("%Y_%m_%d_%H%MZ") 
            dynamic_filename = f"{base_name}_{timestamp}{ext}"
            final_output_path = os.path.join(out_dir, dynamic_filename)

            plt.savefig(final_output_path, dpi=150, facecolor=fig.get_facecolor(), pad_inches=0)
            plt.close()
            print(f"✅ Generated {final_output_path}")

        except Exception as e:
            logging.error(f"Error processing seven-day outlook: {e}")
            raise e

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True, stream=sys.stdout)
    
    output = sys.argv[1] if len(sys.argv) > 1 else "seven_day_outlook.png"
    
    try:
        asyncio.run(seven_day_outlook_ATL(output))
    except Exception as e:
        logging.error(f"Fatal script execution error: {e}")
        sys.exit(1)