#!/usr/bin/env python3
# sat.py — Global Satellite Plotter (GOES + EUMETSAT SEVIRI + Himawari)
#
# Requirements:
#   numpy, matplotlib, cartopy, xarray, metpy, siphon, s3fs (for Himawari)
#
from itertools import product
import os
import logging
from datetime import datetime as dt, timedelta, timezone
import shutil
import token
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize
import xarray as xr
from metpy.plots.ctables import registry
import cartopy.io.shapereader as shpreader
from siphon.catalog import TDSCatalog
from metpy.units import units
import matplotlib.patheffects as pe
import time 
import tempfile


# --- Try to import s3fs for Himawari, but don't fail if it's not there
try:
    import s3fs
    S3FS_AVAILABLE = True
except ImportError:
    S3FS_AVAILABLE = False
    logging.warning("s3fs library not found. Himawari S3 data will be unavailable.")
    logging.warning("Install it with: pip install s3fs")

# --- APPLYING FROSTBYTE THEME GLOBALS ---
plt.rcParams['font.family'] = 'DejaVu Sans'

FIG_BG_COLOR = '#333333'
AXES_BG_COLOR = '#2B2B2B'
GRIDLINE_COLOR = '#3932A0'
TEXT_OUTLINE = [pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()]

# Force Matplotlib global parameters for the dark theme
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

# --- Configuration ---
DYNAMIC_OUTPUT_DIR = 'output_sat' # New folder for archived plots

# --- Setup logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# --- Shape / annotation references for CONUS roads / cities
INTERSTATES_SHP = './shapefiles/tl_2023_us_primaryroads.shp'

# --- Pre-load Shapefiles (Optimization) ---
try:
    _road_reader = shpreader.Reader(INTERSTATES_SHP)
    GLOBAL_ROAD_GEOMS = list(_road_reader.geometries())
except Exception as e:
    logging.warning(f"Could not load interstate shapefile: {e}")
    GLOBAL_ROAD_GEOMS = []

# --- Expanded Cities List ---
cities = [
    {'name': 'New York',    'lat': 40.7128,  'lon': -74.0060},
    {'name': 'Chicago',     'lat': 41.8781,  'lon': -87.6298},
    {'name': 'Los Angeles', 'lat': 34.0522,  'lon': -118.2437},
    {'name': 'Houston',     'lat': 29.7604,  'lon': -95.3698},
    {'name': 'Phoenix',     'lat': 33.4484,  'lon': -112.0740},
    {'name': 'Atlanta',     'lat': 33.7490,  'lon': -84.3880},
    {'name': 'Denver',      'lat': 39.7392,  'lon': -104.9903},
    {'name': 'Seattle',     'lat': 47.6062,  'lon': -122.3321},
    {'name': 'Miami',       'lat': 25.7617,  'lon': -80.1918},
    {'name': 'Dallas',      'lat': 32.7767,  'lon': -96.7970},
    {'name': 'Boston',      'lat': 42.3601,  'lon': -71.0589},
    {'name': 'Philadelphia','lat': 39.9526,  'lon': -75.1652},
    {'name': 'San Diego',   'lat': 32.7157,  'lon': -117.1611},
    {'name': 'San Jose',    'lat': 37.3382,  'lon': -121.8863},
    {'name': 'San Antonio', 'lat': 29.4241,  'lon': -98.4936},
    {'name': 'Jacksonville','lat': 30.3322,  'lon': -81.6556},
    {'name': 'Indianapolis','lat': 39.7684,  'lon': -86.1580},
    {'name': 'Nashville',   'lat': 36.1627,  'lon': -86.7816},
    {'name': 'Charlotte',   'lat': 35.2271,  'lon': -80.8431},
    {'name': 'Detroit',     'lat': 42.3314,  'lon': -83.0458},
    {'name': 'Milwaukee',   'lat': 43.0389,  'lon': -87.9065},
    {'name': 'St. Louis',   'lat': 38.6270,  'lon': -90.1994},
    {'name': 'Salt Lake',   'lat': 40.7608,  'lon': -111.8910},
    {'name': 'Portland',    'lat': 45.5051,  'lon': -122.6750},
    {'name': 'DC',          'lat': 38.9072,  'lon': -77.0369},
]

# --- Custom Regional Zoom Definitions ---
# Bounds: [lon_min, lon_max, lat_min, lat_max]
CUSTOM_REGIONS = {
    'southeast': [-100, -70, 25, 40],
    'westcoast': [-130, -105, 30, 50],
    'northeast': [-85, -65, 35, 50],
    'gulfcoast': [-100, -80, 20, 35],
    
    # EUMETSAT 0-Degree Sub-Regions
    'europe': [-15, 45, 35, 65],
    'africa_zoom': [-20, 45, -15, 25],
    'capeverde': [-30, -15, 10, 25],
    
    # EUMETSAT IODC (41.5-Degree) Sub-Regions
    'middle_east': [30, 65, 10, 45],
    'madagascar': [40, 55, -27, -10]
}

import urllib.request
import tempfile
import os
import time
from datetime import timedelta, timezone
import xarray as xr
import logging
import shutil

try:
    import eumdac
except ImportError:
    logging.warning("eumdac library not found. Run 'pip install eumdac' to enable EUMETSAT plots.")

def fetch_seviri(date, channel, is_iodc=False, max_attempts=5, step_minutes=15):
    """
    Fetch SEVIRI Level1.5 data via EUMETSAT Data Store and Data Tailor API.
    Replaces the permanently deprecated THREDDS server.
    """
    consumer_key = os.environ.get('EUMETSAT_KEY', 'EO02DZF2n3HR7KIQFUzqrbymWEoa')
    consumer_secret = os.environ.get('EUMETSAT_SECRET', 'WBgtUJtHSTYVkwO2Nr7X0jfRT3sa')
    
    if consumer_key == 'YOUR_CONSUMER_KEY':
        raise RuntimeError("Missing EUMETSAT API keys. Register at https://eoportal.eumetsat.int/ for free keys.")

    logging.info("Authenticating with EUMETSAT Data Store API...")
    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials)
    datastore = eumdac.DataStore(token)
    
    # Route to the correct EUMETSAT Satellite Collection
    collection_id = 'EO:EUM:DAT:MSG:HRSEVIRI-IODC' if is_iodc else 'EO:EUM:DAT:MSG:HRSEVIRI'
    collection = datastore.get_collection(collection_id)
    
    start_time = date - timedelta(minutes=25)
    end_time = date + timedelta(minutes=5)
    
    logging.info(f"Searching catalog for {channel} between {start_time:%H:%M} and {end_time:%H:%M} UTC...")
    products = collection.search(dtstart=start_time, dtend=end_time)
    
    product = products.first()
    if not product:
         raise RuntimeError(f"No EUMETSAT SEVIRI products found near {date}")

    logging.info(f"Found product: {product}. Dispatching to Data Tailor for NetCDF conversion...")
    
    tailor = eumdac.DataTailor(token)

    # Clean remote workspace to prevent 20GB Quota crashes
    logging.info("Checking EUMETSAT Data Tailor workspace quota...")
    try:
        for old_job in tailor.customisations:
            if old_job.status in ["DONE", "FAILED", "KILLED", "INACTIVE"]:
                old_job.delete()
    except Exception as e:
        pass # Ignore minor API cleanup errors

    # Map FrostByte channel strings to internal band IDs
    dt_band_map = {
        'VIS006': 'channel_1',
        'WV_062': 'channel_5',
        'IR_108': 'channel_9'
    }
    band_id = dt_band_map.get(channel, 'channel_9')

    # THE FIX: Removed 'projection'. Regridding a full disk without an ROI/resolution causes backend crashes.
    chain = eumdac.tailor_models.Chain(
        product='HRSEVIRI',
        format='netcdf4',
        filter=eumdac.tailor_models.Filter(bands=[band_id])
    )
    
    job = tailor.new_customisation(product, chain)
    
    while job.status in ["QUEUED", "RUNNING"]:
        logging.info(f"Data Tailor converting... Status: {job.status} (Waiting 10s...)")
        time.sleep(10)
        
    if job.status != "DONE":
        # THE FIX: Corrected log extraction using '.logfile' instead of '.log'
        try:
            log_text = job.logfile
        except Exception as e:
            log_text = f"Could not retrieve log: {e}"
        raise RuntimeError(f"EUMETSAT Data Tailor failed. Status: {job.status}\n--- SERVER LOG ---\n{log_text}\n------------------")

    temp_dir = tempfile.gettempdir()
    local_filename = os.path.join(temp_dir, f"seviri_dt_{job._id}.nc")
    
    logging.info(f"Conversion complete. Downloading NetCDF payload to {local_filename}...")
    
    output_file = list(job.outputs)[0]
    with job.stream_output(output_file) as stream, open(local_filename, mode='wb') as fdst:
        shutil.copyfileobj(stream, fdst)
            
    try:
        job.delete()
    except Exception:
        pass

    logging.info("Parsing into Xarray.")
    ds = xr.open_dataset(local_filename, engine='netcdf4') 
    
    # Dynamically locate spatial variable based on standard or projected dimensions
    var = None
    for v_name in ds.data_vars:
        dims = ds[v_name].dims
        if ds[v_name].ndim >= 2 and (('y' in dims and 'x' in dims) or ('lat' in dims and 'lon' in dims) or ('latitude' in dims and 'longitude' in dims)):
            if v_name not in ['x', 'y', 'lat', 'lon', 'latitude', 'longitude']:
                var = ds[v_name]
                break
            
    if var is None:
        raise ValueError(f"Converted dataset missing spatial variables. Vars found: {list(ds.data_vars)}")
    
    if not hasattr(var, 'units'):
        if 'IR' in channel.upper() or 'WV' in channel.upper():
            var = var.assign_attrs(units='K')
        else:
            var = var.assign_attrs(units='W / (m^2 sr micron)')

    return ds, var, date

def plot_seviri(ds, var, output_path_preview, central_lon=0, custom_extent=None, output_path_archive=None, channel=None, timestamp_str=None, region=None): 
    proj = ccrs.Geostationary(satellite_height=35785831.0, central_longitude=central_lon, sweep_axis='y')
    
    fig = plt.figure(figsize=(15, 10))
    fig.set_facecolor(FIG_BG_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_facecolor(AXES_BG_COLOR)
    
    arr = var.values  
    units_str = str(var.units) if hasattr(var, 'units') else 'UNKNOWN'

    ext = 5568748.2757
    img_extent = (-ext, ext, -ext, ext)
    text_outline = TEXT_OUTLINE
    
    is_ir_channel = 'IR' in channel.upper()
    is_wv_channel = 'WV' in channel.upper()
    is_vis_channel = 'VIS' in channel.upper()
    
    if is_vis_channel:
        # --- VIS with CLAHE (professional local contrast) ---
        # This is the method used in many satellite viewers for best cloud/surface texture.
        # Requires: pip install scikit-image
        arr_safe = np.clip(arr, 0.0, 1.0)
        
        p_low = np.nanpercentile(arr_safe, 2)
        p_high = np.nanpercentile(arr_safe, 98)
        arr_stretched = (arr_safe - p_low) / (p_high - p_low + 1e-8)
        arr_stretched = np.clip(arr_stretched, 0.0, 1.0)
        
        # Light gamma before CLAHE often helps
        vis_gamma = 0.75
        arr_plot = arr_stretched ** vis_gamma
        
        # === Real CLAHE ===
        try:
            from skimage.exposure import equalize_adapthist
            arr_plot = equalize_adapthist(arr_plot, clip_limit=0.035, nbins=256)
        except ImportError:
            logging.warning("scikit-image not installed. Falling back to unsharp mask. "
                            "Run: pip install scikit-image for best results.")
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(arr_plot, sigma=1.0)
            arr_plot = arr_plot + 2.0 * (arr_plot - blurred)
            arr_plot = np.clip(arr_plot, 0.0, 1.0)
        
        logging.info(f"VIS CLAHE — raw min/max: {np.nanmin(arr_safe):.3f}/{np.nanmax(arr_safe):.3f} | "
                     f"gamma={vis_gamma} | clip_limit=0.035")
        
        cmap = 'gray'
        vmin = 0.0
        vmax = 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cbar_ticks = np.linspace(0.0, 1.0, 6)
        cbar_ticklabels = [f'{t:.1f}' for t in cbar_ticks]
        colorbar_label = "Reflectance"
    elif is_ir_channel or is_wv_channel:
        
        # --- THE FIX: PLANCK CALIBRATION ---
        # Data Tailor outputs Radiance (0 to ~150). FrostByte needs Celsius.
        if np.nanmax(arr) < 200:
            logging.info("Converting raw SEVIRI Radiance to Celsius via Planck function...")
            c1 = 1.19104e-5
            c2 = 1.43877
            if is_ir_channel:
                nu, a, b = 930.659, 0.9983, 0.627
            else:
                nu, a, b = 1598.566, 0.9991, 1.321
                
            arr_safe = np.where(arr <= 0.01, np.nan, arr)
            rad_term = (c1 * (nu**3)) / arr_safe
            tb_kelvin = ((c2 * nu) / np.log(rad_term + 1) - b) / a
            arr_plot = tb_kelvin - 273.15
        else:
            arr_plot = arr - 273.15 if units_str.lower().startswith('k') else arr
        # -----------------------------------
            
        if is_ir_channel:
            levels = np.array([-110, -75, -65, -55, -45, -28, -20, -15, 26, 56])
            C_WHITE = (1.0, 1.0, 1.0)
            C_BLACK = (0.0, 0.0, 0.0)
            C_RED = (1.0, 0.0, 0.0)
            C_ORANGE = (1.0, 0.5, 0.0)
            C_GREEN = (0.0, 1.0, 0.0)
            C_BLUE = (0.0, 0.0, 1.0)
            C_PURPLE = (0.5, 0.0, 0.5)
            C_LGRAY = (0.7, 0.7, 0.7)
            node_levels_stable = np.array([-110, -75, -65, -55, -45, -28, -20, -15, 25.999, 26, 56])
            node_colors_stable = [
                C_WHITE, C_BLACK, C_RED, C_ORANGE, C_GREEN, C_BLUE, C_PURPLE, C_WHITE, C_BLACK, C_LGRAY, C_BLACK
            ]
            vmin_data = node_levels_stable.min()
            vmax_data = node_levels_stable.max()
            span = vmax_data - vmin_data
            normalized_nodes = (node_levels_stable - vmin_data) / span
            normalized_nodes[0] = 0.0
            normalized_nodes[-1] = 1.0
            cmap_data = list(zip(normalized_nodes, node_colors_stable))
            cmap = LinearSegmentedColormap.from_list('custom_ir_cmap', cmap_data)
            norm = Normalize(vmin=levels.min(), vmax=levels.max())
            colorbar_label = "Temp (°C)"
            
            cbar_ticks = np.array([-110, -75, -65, -55, -45, -28, -15, 26, 56])
            cbar_ticklabels = [str(int(l)) for l in cbar_ticks]
        elif is_wv_channel:
            levels = np.array([0, -15.5, -30, -47, -75, -100, -109])
            C_RED = (1.0, 0.0, 0.0)
            C_YELLOW = (1.0, 1.0, 0.0)
            C_BLUE = (0.0, 0.0, 1.0)
            C_WHITE = (1.0, 1.0, 1.0)
            C_GREEN = (0.0, 0.5, 0.0)
            C_TEAL = (0.0, 1.0, 1.0)
            C_LB = (0.5, 0.5, 1.0)
            vmin_data = levels.min()
            vmax_data = levels.max()
            span = vmax_data - vmin_data
            normalized_nodes = (levels - vmin_data) / span
            nodes = np.flip(normalized_nodes)
            node_colors = [C_RED, C_YELLOW, C_BLUE, C_WHITE, C_GREEN, C_TEAL, C_LB]
            node_colors = node_colors[::-1]
            nodes[0] = 0.0
            nodes[-1] = 1.0
            cmap_data = list(zip(nodes, node_colors))
            cmap = LinearSegmentedColormap.from_list('custom_wv_cmap', cmap_data)
            norm = Normalize(vmin=vmin_data, vmax=vmax_data)
            colorbar_label = "Temp (°C)"
            cbar_ticks = np.sort(levels)
            cbar_ticklabels = [f'{l:.1f}' if abs(l + 15.5) < 1e-6 else str(int(l)) for l in cbar_ticks]
    else:
        arr_plot = arr - 273.15 if units_str.lower().startswith('k') else arr
        levels = np.linspace(-70, 10, 17)
        norm = BoundaryNorm(levels, 256, extend='both')
        cmap = 'magma_r'
        colorbar_label = "Temp (°C)" if units_str.lower().startswith('k') else units_str
        cbar_ticks = None
        cbar_ticklabels = None
    
    mesh = ax.imshow(arr_plot, origin='upper', extent=img_extent, 
                     transform=proj, cmap=cmap, norm=norm, interpolation='none')
    
    ax.coastlines(resolution='50m', color='black', linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1.2)
    ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.8)
    
    if custom_extent:
        ax.set_extent(custom_extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    gl = ax.gridlines(draw_labels=True,
                      color=GRIDLINE_COLOR, linestyle='--', alpha=0.4, zorder=10, linewidth=1.0,
                      xlocs=range(-180, 181, 10), ylocs=range(-90, 91, 10))
    gl.top_labels = False
    gl.right_labels = False
    
    gl.xlabel_style = {'color': 'white', 'weight': 'bold', 'path_effects': text_outline}
    gl.ylabel_style = {'color': 'white', 'weight': 'bold', 'path_effects': text_outline}

    # --- Dynamic Professional Titles ---
    sat_name = "Meteosat IODC (41.5°E)" if central_lon > 10 else "Meteosat 0°"
    
    region_display = region.replace('_', ' ').title()
    if region in ('eumetsat_fd', 'africa', 'iodc', 'iodc_fd'):
        region_display = "Full Disk"
        
    channel_display_map = {
        'VIS006': 'Visible (0.64µm)',
        'WV_062': 'Water Vapor (6.2µm)',
        'IR_108': 'Infrared (10.8µm)'
    }
    nice_channel = channel_display_map.get(channel, channel)
        
    title_top = f"{sat_name} {region_display} | {nice_channel}"
    title_bottom = timestamp_str if timestamp_str else ""
    
    ax.text(0.0, 1.05, title_top, transform=ax.transAxes, color='white', 
            path_effects=text_outline, weight='bold', fontsize=16, ha='left', va='bottom')
    ax.text(0.0, 1.005, title_bottom, transform=ax.transAxes, color='white', 
            path_effects=text_outline, weight='bold', fontsize=13, ha='left', va='bottom')
    
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.06, pos.width, 0.025])
    
    cbar = fig.colorbar(mesh, cax=cax, orientation='horizontal', extend='both')
    
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
    
    cbar.set_label(colorbar_label, color='white', path_effects=text_outline, weight='bold', fontsize=12)
    cbar.ax.tick_params(labelcolor='white', labelsize=10)
    for t in cbar.ax.get_xticklabels():
        t.set_path_effects(text_outline)
        t.set_fontweight('bold')
    
    from matplotlib.transforms import Bbox
    fw, fh = fig.get_size_inches()
    custom_bbox = Bbox.from_extents(
        (pos.x0 * fw) - 0.6, 
        ((pos.y0 - 0.06) * fh) - 0.5, 
        (pos.x1 * fw) + 0.3, 
        (pos.y1 * fh) + 0.8  
    )

    logging.info(f"Saving preview image to: {output_path_preview}")
    fig.savefig(output_path_preview, facecolor=fig.get_facecolor(), bbox_inches=custom_bbox, dpi=200)
    
    if output_path_archive:
        logging.info(f"Saving archive image to: {output_path_archive}")
        fig.savefig(output_path_archive, facecolor=fig.get_facecolor(), bbox_inches=custom_bbox, dpi=200)

    plt.close(fig)

# --- GOES/Himawari Utility Functions ---

def get_latest_goes_file(satellite_name, scan_area, product_name, channel):
    """
    Finds the latest available file for a GOES product from the Unidata THREDDS catalog.
    """
    base_url = "https://thredds.ucar.edu/thredds/catalog/satellite/"
    channel_catalog_url = f"{base_url}{satellite_name}/products/{product_name}/{scan_area}/Channel{channel}/catalog.xml"
    current_catalog_url = channel_catalog_url.replace('/catalog.xml', '/current/catalog.xml')
    logging.info(f"Checking current catalog: {current_catalog_url}")
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            date_cat = TDSCatalog(current_catalog_url)
            if not date_cat.datasets:
                raise ValueError("No datasets found in current catalog")
            datasets = sorted(list(date_cat.datasets.values()), key=lambda ds: ds.name)
            latest_file_ds = datasets[-1]
            return latest_file_ds.access_urls['OPeNDAP']
        except Exception as e:
            logging.warning(f"Failed to access catalog (Attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(5) 
            else:
                logging.error(f"Failed to access current catalog or find file: {e}")
                raise RuntimeError(f"Failed to find latest file in current catalog at {current_catalog_url}: {e}")

def get_latest_goes_fd_file(satellite_name, channel_str, max_attempts=18):
    if not S3FS_AVAILABLE:
        raise ImportError("s3fs required for GOES full disk data.")
    if 'east' in satellite_name.lower():
        bucket = 'noaa-goes19'
        sat_id = '19'
    else:
        bucket = 'noaa-goes18'
        sat_id = '18'
    
    fs = s3fs.S3FileSystem(anon=True)
    product = 'ABI-L2-CMIPF'
    mode = 'M6'
    prefix = f"OR_ABI-L2-CMIPF-{mode}C{channel_str}_G{sat_id}_"
    base_path = f"{bucket}/{product}"
    now_utc = dt.now(timezone.utc)
    
    recent_dirs = []
    for attempt in range(max_attempts):
        current_time = now_utc - timedelta(minutes=attempt * 10)
        year = current_time.year
        doy = current_time.timetuple().tm_yday
        hh = current_time.hour
        s3_dir = f"{base_path}/{year}/{doy:03d}/{hh:02d}/"
        recent_dirs.append(s3_dir)
    
    for s3_dir in recent_dirs:
        logging.info(f"Checking GOES FD S3 path: {s3_dir}")
        try:
            files = fs.glob(f"{s3_dir}{prefix}*.nc")
            if files:
                files.sort(reverse=True)
                latest_file = files[0]
                logging.info(f"Found GOES FD file: s3://{latest_file}")
                return f"s3://{latest_file}"
        except Exception as e:
            logging.warning(f"Error accessing GOES FD S3 path: {e}")
    
    raise RuntimeError(f"Failed to find GOES FD file after checking {max_attempts} paths.")

def get_latest_glm_file(satellite_name, region_type, max_attempts=18):
    """
    Finds the latest available GLM L2 LCFA file from NOAA AWS S3 bucket.
    """
    if "goes" not in satellite_name:
        logging.warning(f"GLM is only available for GOES satellites, not {satellite_name}")
        return None
    if not S3FS_AVAILABLE:
        logging.warning("s3fs library not found. GLM S3 data unavailable.")
        return None
    if 'east' in satellite_name.lower():
        bucket = 'noaa-goes19'
        sat_id = '19'
    else:
        bucket = 'noaa-goes18'
        sat_id = '18'

    fs = s3fs.S3FileSystem(anon=True)
    base_path = f"{bucket}/GLM-L2-LCFA"
    now_utc = dt.now(timezone.utc)
    prefix = f"OR_GLM-L2-LCFA_G{sat_id}_"

    recent_dirs = []
    for attempt in range(max_attempts):
        current_time = now_utc - timedelta(seconds=attempt * 20)
        year = current_time.year
        doy = current_time.timetuple().tm_yday
        hh = current_time.strftime('%H')
        s3_dir = f"{base_path}/{year}/{doy:03d}/{hh}/"
        recent_dirs.append(s3_dir)
    
    for s3_dir in recent_dirs:
        logging.info(f"Checking GLM S3 path: {s3_dir}")
        try:
            files = fs.glob(f"{s3_dir}{prefix}*.nc")
            if files:
                files.sort(reverse=True)
                latest_file = files[0]
                logging.info(f"Found GLM file: s3://{latest_file}")
                return f"s3://{latest_file}"
        except Exception as e:
            logging.warning(f"Error accessing GLM S3 path: {e}")

    logging.error(f"Failed to find GLM file after checking {max_attempts} paths.")
    return None

def get_latest_himawari_file(product_code, max_retries=6):
    """
    Finds the latest Himawari-9 file from the NOAA S3 bucket.
    Uses ISatSS L2 for albedo/temp, and L1b for Water Vapor.
    """
    if not S3FS_AVAILABLE:
        raise ImportError("The 's3fs' library is required for Himawari data. Please install it: pip install s3fs")

    if product_code == 9:
        # Water Vapor uses full L1b data (has all AHI bands including 6.9µm)
        folder = "AHI-L1b-FLDK"
        prefix = "AHI-H09-FLDK-"
    else:
        # Cloud Albedo (2) and Cloud Top Temp (14) use the simplified ISatSS product
        folder = "AHI-L2-FLDK-ISatSS"
        prefix = "AHI-ISATSS_"

    fs = s3fs.S3FileSystem(anon=True)
    base_path = f"noaa-himawari9/{folder}"
    now_utc = dt.now(timezone.utc)
    
    recent_paths = []
    for i in range(max_retries):
        current_time = now_utc - timedelta(minutes=i * 10)
        rounded_minute = (current_time.minute // 10) * 10
        current_time = current_time.replace(minute=rounded_minute, second=0, microsecond=0)
        s3_path = f"{base_path}/{current_time:%Y/%m/%d/%H%M}/"
        recent_paths.append(s3_path)
    
    for s3_path in recent_paths:
        logging.info(f"Checking Himawari S3 path: {s3_path}")
        try:
            if fs.exists(s3_path):
                files = fs.glob(f"{s3_path}{prefix}*.nc")
                if files:
                    logging.info(f"Found Himawari file: {files[0]}")
                    return f"s3://{files[0]}"
        except Exception as e:
            logging.error(f"Error accessing Himawari S3 bucket: {e}")
    
    raise RuntimeError(f"Could not find any Himawari data after checking back {max_retries*10} minutes.")

def plot_goes(ds, arr, var_name, units_str, region, output_path_preview, ds_glm=None, output_path_archive=None, timestamp_str=None): 
    text_outline = TEXT_OUTLINE
    
    # Bypass MetPy CF parser if the grid_mapping is missing
    try:
        if 'geostationary' in ds:
             proj = ds['geostationary'].metpy.cartopy_crs
        elif 'geospatial_lat_lon_ids' in ds: 
            proj = ds['geospatial_lat_lon_ids'].metpy.cartopy_crs
        else:
            proj = arr.metpy.cartopy_crs 
    except AttributeError:
        # Fallback for Himawari AWS files missing CF geostationary metadata
        logging.warning("CF metadata missing. Falling back to manual Himawari projection.")
        proj = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)
    
    fig = plt.figure(figsize=(15, 10))
    fig.set_facecolor(FIG_BG_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_facecolor(AXES_BG_COLOR)
    
    if 'x' in ds and 'y' in ds:
        x = ds['x']
        y = ds['y']
    elif 'longitude' in ds and 'latitude' in ds:
        x = ds['longitude']
        y = ds['latitude']
    else:
        raise ValueError("Could not find coordinate variables (x/y or longitude/latitude) in dataset.")
    
    grid_mapping = arr.attrs.get('grid_mapping')
    if grid_mapping:
        proj_var = ds[grid_mapping]
        if proj_var.grid_mapping_name == 'geostationary' and ds.x.units == 'rad':
            height = proj_var.perspective_point_height
            x = x * height
            y = y * height
    
    is_ir_channel = False
    is_wv_channel = False
    if 'temp' in var_name.lower() or 'brightness' in var_name.lower():
        if arr.metpy.units == units.kelvin:
            arr_plot = arr.metpy.convert_units('degC').metpy.magnitude
        else:
            arr_plot = arr.metpy.magnitude
        
        is_ir_channel = 'Ch 14' in var_name or 'Cloud Top Temperature' in var_name
        is_wv_channel = 'Ch 9' in var_name
        if is_ir_channel:
            levels = np.array([-110, -75, -65, -55, -45, -28, -20, -15, 26, 56])
            C_WHITE = (1.0, 1.0, 1.0)
            C_BLACK = (0.0, 0.0, 0.0)
            C_RED = (1.0, 0.0, 0.0)
            C_ORANGE = (1.0, 0.5, 0.0)
            C_GREEN = (0.0, 1.0, 0.0)
            C_BLUE = (0.0, 0.0, 1.0)
            C_PURPLE = (0.5, 0.0, 0.5)
            C_LGRAY = (0.7, 0.7, 0.7)
            node_levels_stable = np.array([-110, -75, -65, -55, -45, -28, -20, -15, 25.999, 26, 56])
            node_colors_stable = [
                C_WHITE, C_BLACK, C_RED, C_ORANGE, C_GREEN, C_BLUE, C_PURPLE, C_WHITE, C_BLACK, C_LGRAY, C_BLACK
            ]
            vmin_data = node_levels_stable.min()
            vmax_data = node_levels_stable.max()
            span = vmax_data - vmin_data
            normalized_nodes = (node_levels_stable - vmin_data) / span
            normalized_nodes[0] = 0.0
            normalized_nodes[-1] = 1.0
            cmap_data = list(zip(normalized_nodes, node_colors_stable))
            cmap = LinearSegmentedColormap.from_list('custom_ir_cmap', cmap_data)
            norm = Normalize(vmin=levels.min(), vmax=levels.max())
            colorbar_label = "Temp (°C)"
            
            # Removed -20 to prevent text collision
            cbar_ticks = np.array([-110, -75, -65, -55, -45, -28, -15, 26, 56])
            cbar_ticklabels = [str(int(l)) for l in cbar_ticks]
        elif is_wv_channel:
            levels = np.array([0, -15.5, -30, -47, -75, -100, -109])
            C_RED = (1.0, 0.0, 0.0)
            C_YELLOW = (1.0, 1.0, 0.0)
            C_BLUE = (0.0, 0.0, 1.0)
            C_WHITE = (1.0, 1.0, 1.0)
            C_GREEN = (0.0, 0.5, 0.0)
            C_TEAL = (0.0, 1.0, 1.0)
            C_LB = (0.5, 0.5, 1.0)
            vmin_data = levels.min()
            vmax_data = levels.max()
            span = vmax_data - vmin_data
            normalized_nodes = (levels - vmin_data) / span
            nodes = np.flip(normalized_nodes)
            node_colors = [C_RED, C_YELLOW, C_BLUE, C_WHITE, C_GREEN, C_TEAL, C_LB]
            node_colors = node_colors[::-1]
            nodes[0] = 0.0
            nodes[-1] = 1.0
            cmap_data = list(zip(nodes, node_colors))
            cmap = LinearSegmentedColormap.from_list('custom_wv_cmap', cmap_data)
            norm = Normalize(vmin=vmin_data, vmax=vmax_data)
            colorbar_label = "Temp (°C)"
            cbar_ticks = np.sort(levels)
            cbar_ticklabels = [f'{l:.1f}' if abs(l + 15.5) < 1e-6 else str(int(l)) for l in cbar_ticks]
        else: 
            levels = np.linspace(-90, 20, 16)
            norm = BoundaryNorm(levels, 256, extend='both')
            cmap = 'turbo_r'
            colorbar_label = "Temp (°C)"
            cbar_ticks = None
            cbar_ticklabels = None
    elif 'albedo' in var_name.lower() or 'reflectance' in var_name.lower():
        arr_plot = arr.metpy.magnitude
        cmap = 'gray'
        vmin = 0.0
        vmax = 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cbar_ticks = np.linspace(0.0, 1.0, 6)
        cbar_ticklabels = [f'{t:.1f}' for t in cbar_ticks]
        colorbar_label = "Reflectance/Albedo"
    else:
        arr_plot = arr.metpy.magnitude
        cmap = 'viridis'
        levels = np.linspace(np.nanmin(arr_plot), np.nanmax(arr_plot), 20)
        colorbar_label = units_str
        norm = BoundaryNorm(levels, 256, extend="both")
        cbar_ticks = None
        cbar_ticklabels = None
    
    # Optimization: Striding to speed up rendering
    skip = 3 if region in ("west_fulldisk", "goes_east_fd", "himawari") else 1
    
    mesh = ax.pcolormesh(x[::skip], y[::skip], arr_plot[::skip, ::skip], 
                         cmap=cmap, norm=norm, transform=proj, rasterized=True)
    
    # Optimization: Dynamic coastlines
    res = '110m' if skip > 1 else '50m'
    ax.coastlines(resolution=res, color='black', linewidth=1.2)
    ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.8)
    
    # Optimization: Global Shapefile Cache
    if GLOBAL_ROAD_GEOMS:
        ax.add_geometries(GLOBAL_ROAD_GEOMS, ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=1.0, zorder=10)
        ax.add_geometries(GLOBAL_ROAD_GEOMS, ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=0.5, zorder=11)
    
    # Handle geographic extents
    current_extent_lonlat = None
    if region in CUSTOM_REGIONS:
        logging.info(f"Applying custom extent for region: {region}")
        current_extent_lonlat = CUSTOM_REGIONS[region]
        ax.set_extent(current_extent_lonlat, crs=ccrs.PlateCarree())
    elif region in ('conus', 'east_meso1', 'east_meso2', 'west_meso1', 'west_meso2'):
        logging.info(f"Setting extent to data (sector) bounds for region: {region}")
        ax.set_extent([x.min(), x.max(), y.min(), y.max()], crs=proj)
    elif region in ('west_fulldisk', 'goes_east_fd', 'himawari'):
        logging.info(f"Setting extent to global (full disk) for region: {region}")
        ax.set_global()
    
    gl = ax.gridlines(draw_labels=True,
                      color=GRIDLINE_COLOR, linestyle='--', alpha=0.4, zorder=10, linewidth=1.0,
                      xlocs=range(-180, 181, 10), ylocs=range(-90, 91, 10))
    gl.top_labels = False
    gl.right_labels = False
    
    # Gridline Path Effects Restored
    gl.xlabel_style = {'color': 'white', 'weight': 'bold', 'path_effects': text_outline}
    gl.ylabel_style = {'color': 'white', 'weight': 'bold', 'path_effects': text_outline}
    
    if region in ("conus", "west_fulldisk", "east_meso1", "east_meso2", "west_meso1", "west_meso2", "southeast", "westcoast", "northeast", "gulfcoast", "goes_east_fd"):
        logging.info("Adding filtered CONUS features with text effects...")
        
        if current_extent_lonlat is None:
             lon_min, lon_max, lat_min, lat_max = -130, -60, 20, 50 
             if region == 'west_fulldisk' or region == 'goes_east_fd':
                 lon_min, lon_max, lat_min, lat_max = -170, -30, -20, 60
             elif region in ('east_meso1', 'east_meso2', 'west_meso1', 'west_meso2'):
                 min_lon, min_lat = ccrs.PlateCarree().transform_point(x.min(), y.min(), proj)
                 max_lon, max_lat = ccrs.PlateCarree().transform_point(x.max(), y.max(), proj)
                 lon_min, lon_max, lat_min, lat_max = min_lon, max_lon, min_lat, max_lat
        else:
             lon_min, lon_max, lat_min, lat_max = current_extent_lonlat
             
        # DYNAMIC CITY FILTER LOGIC
        is_full_disk = region in ("west_fulldisk", "goes_east_fd", "himawari")
        if is_full_disk:
            major_hubs = ['New York', 'Los Angeles', 'Miami', 'Seattle', 'Chicago', 'Houston']
            display_cities = [c for c in cities if c['name'] in major_hubs]
        else:
            display_cities = cities
        
        for city in display_cities:
            city_lon = city['lon']
            city_lat = city['lat']
            
            if lon_min <= city_lon <= lon_max and lat_min <= city_lat <= lat_max:
                ax.plot(city_lon, city_lat, 'o', color='#6BCB77', markersize=5, markeredgecolor='black',
                        transform=ccrs.PlateCarree(), zorder=10)
                ax.text(city_lon + 0.5, city_lat - 0.5, city['name'],
                        transform=ccrs.PlateCarree(), color='white', fontsize=10, weight='bold',
                        ha='left', va='top', zorder=10, path_effects=text_outline)
            
    # Split Title Logic (Two Rows)
    title_top = f"{region.upper()} - {var_name}"
    title_bottom = timestamp_str if timestamp_str else ""

    if ds_glm is not None:
        logging.info("Overlaying GLM flash data...")
        try:
            lons = ds_glm.variables['flash_lon'][:]
            lats = ds_glm.variables['flash_lat'][:]
            ax.scatter(lons, lats,
                       color='magenta', marker='+', s=10, linewidths=0.5, 
                       transform=ccrs.PlateCarree(), zorder=10)
            if title_bottom:
                title_bottom += " | GLM Flashes"
            else:
                title_bottom = "GLM Flashes"
        except Exception as e:
            logging.warning(f"Failed to plot GLM data: {e}")

    ax.text(0.0, 1.05, title_top, transform=ax.transAxes, color='white', 
            path_effects=text_outline, weight='bold', fontsize=16, ha='left', va='bottom')
    ax.text(0.0, 1.005, title_bottom, transform=ax.transAxes, color='white', 
            path_effects=text_outline, weight='bold', fontsize=13, ha='left', va='bottom')

    # --- Exact Width Colorbar Logic ---
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.06, pos.width, 0.025])
    
    cbar = fig.colorbar(mesh, cax=cax, orientation='horizontal', extend='both')
    
    if (is_ir_channel or is_wv_channel or 'albedo' in var_name.lower()) and cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
    
    cbar.set_label(colorbar_label, color='white', path_effects=text_outline, weight='bold', fontsize=12)
    cbar.ax.tick_params(labelcolor='white', labelsize=10)
    for t in cbar.ax.get_xticklabels():
        t.set_path_effects(text_outline)
        t.set_fontweight('bold')
    
    # Widened Bounding Box (Accommodates 2-row title and Gridline text)
    from matplotlib.transforms import Bbox
    fw, fh = fig.get_size_inches()
    custom_bbox = Bbox.from_extents(
        (pos.x0 * fw) - 0.6, 
        ((pos.y0 - 0.06) * fh) - 0.5, 
        (pos.x1 * fw) + 0.3, 
        (pos.y1 * fh) + 0.8  # Raised the roof to 0.8
    )
    
    logging.info(f"Saving preview image to: {output_path_preview}")
    fig.savefig(output_path_preview, facecolor=fig.get_facecolor(), bbox_inches=custom_bbox, dpi=200)
    
    if output_path_archive:
        logging.info(f"Saving archive image to: {output_path_archive}")
        fig.savefig(output_path_archive, facecolor=fig.get_facecolor(), bbox_inches=custom_bbox, dpi=200)

    plt.close(fig)

# --- Main Satellite Plotter Function ---

def sat(region: str, product_code: int, output_path: str = "sat_preview.jpg", overlay_glm: bool = False):
    logging.info("=== SAT.PY START ===")
    logging.info(f"Region: {region}, Product: {product_code}, Output: {output_path}, GLM Overlay: {overlay_glm}")
    region = region.lower()
    output_path_preview = output_path 

    valid_regions = [
        "conus","west_fulldisk","east_meso1","east_meso2","west_meso1","west_meso2","himawari",
        "europe","africa","eatlantic","iodc", "goes_east_fd", "eumetsat_fd", "iodc_fd"
    ] + list(CUSTOM_REGIONS.keys())
    
    if region not in valid_regions:
        raise ValueError(f"Invalid region: {region}")

    # Product names for logging/printing
    product_codes = {
        "conus":      {2:"Red Visible (0.64µm)", 14:"Clean LW IR Window (10.3µm)", 9:"Mid-level WV (6.9µm)"},
        "west_fulldisk": {2:"Red Visible (0.64µm)", 14:"Clean LW IR Window (10.3µm)", 9:"Mid-level WV (6.9µm)"},
        "east_meso1": {2:"Red Visible (0.64µm)", 14:"Clean LW IR Window (10.3µm)", 9:"Mid-level WV (6.9µm)"},
        "east_meso2": {2:"Red Visible (0.64µm)", 14:"Clean LW IR Window (10.3µm)", 9:"Mid-level WV (6.9µm)"},
        "west_meso1": {2:"Red Visible (0.64µm)", 14:"Clean LW IR Window (10.3µm)", 9:"Mid-level WV (6.9µm)"},
        "west_meso2": {2:"Red Visible (0.64µm)", 14:"Clean LW IR Window (10.3µm)", 9:"Mid-level WV (6.9µm)"},
        "himawari":   {2:"Cloud Albedo", 9:"Water Vapor (6.9µm)", 14:"Cloud Top Temperature"},
        "europe":     {2:"Red Visible", 9:"Water Vapor", 14:"Infrared"},
        "africa":     {2:"Red Visible", 9:"Water Vapor", 14:"Infrared"},
        "eatlantic":  {2:"Red Visible", 9:"Water Vapor", 14:"Infrared"},
        "iodc":       {2:"Red Visible", 9:"Water Vapor", 14:"Infrared"},
        "southeast":  {2:"Red Visible (0.64µm)", 14:"Clean LW IR Window (10.3µm)", 9:"Mid-level WV (6.9µm)"},
        "westcoast":  {2:"Red Visible (0.64µm)", 14:"Clean LW IR Window (10.3µm)", 9:"Mid-level WV (6.9µm)"},
        "northeast":  {2:"Red Visible (0.64µm)", 14:"Clean LW IR Window (10.3µm)", 9:"Mid-level WV (6.9µm)"},
        "gulfcoast":  {2:"Red Visible (0.64µm)", 14:"Clean LW IR Window (10.3µm)", 9:"Mid-level WV (6.9µm)"},
        "capeverde":  {2:"Red Visible", 9:"Water Vapor", 14:"Infrared"},
        "goes_east_fd": {2:"Red Visible (0.64µm)", 14:"Clean LW IR Window (10.3µm)", 9:"Mid-level WV (6.9µm)"},
        "eumetsat_fd": {2:"Red Visible", 9:"Water Vapor", 14:"Infrared"},
        "iodc_fd":    {2:"Red Visible", 9:"Water Vapor", 14:"Infrared"},
        "middle_east": {2:"Red Visible", 9:"Water Vapor", 14:"Infrared"},
        "madagascar":  {2:"Red Visible", 9:"Water Vapor", 14:"Infrared"},
        "africa_zoom": {2:"Red Visible", 9:"Water Vapor", 14:"Infrared"}
    }

    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    dynamic_filename = f"sat_{region}_{product_code}_{timestamp}.jpg"
    output_path_archive = os.path.join(DYNAMIC_OUTPUT_DIR, dynamic_filename)
    
    os.makedirs(DYNAMIC_OUTPUT_DIR, exist_ok=True)
    
    # Automatically grab all explicitly named regions plus any custom EUMETSAT boxes
    eumetsat_regions = ["europe", "africa", "eatlantic", "iodc", "eumetsat_fd", "iodc_fd", "africa_zoom", "middle_east", "madagascar", "capeverde"]
    
    if region in eumetsat_regions:
        logging.info(f"Fetching EUMETSAT SEVIRI data for region: {region}")
        channel_map = { 2: "VIS006", 9: "WV_062", 14:"IR_108" }
        channel = channel_map.get(product_code)
        if channel is None:
            raise ValueError(f"No channel defined for product code {product_code} in {region}")
            
        now_utc = dt.now(timezone.utc)
        
        # Enforce a 65-minute offset to bypass EUMETSAT NRT Embargo wall
        safe_time = now_utc - timedelta(minutes=65)
        minute = (safe_time.minute // 15) * 15
        date = safe_time.replace(minute=minute, second=0, microsecond=0)
        
        # Route to IODC (41.5E) or Standard (0E)
        iodc_regions = ['iodc', 'iodc_fd', 'madagascar', 'middle_east']
        is_iodc = region in iodc_regions
        central_lon = 41.5 if is_iodc else 0.0
        
        ds, var, valid_date = fetch_seviri(date, channel, is_iodc=is_iodc)
        timestamp_str = valid_date.strftime("%Y-%m-%d %H:%M UTC")
        
        plot_seviri(ds, var, output_path_preview, central_lon=central_lon, 
                    custom_extent=CUSTOM_REGIONS.get(region),
                    output_path_archive=output_path_archive, channel=channel,
                    timestamp_str=timestamp_str, region=region)
                    
        print(f"Success: {region.upper()} saved to {output_path_preview} (Archive: {dynamic_filename})")
        logging.info("=== SAT.PY END ===")
        return

    ds_glm = None
    if region in ("conus","west_fulldisk","east_meso1","east_meso2","west_meso1","west_meso2", "goes_east_fd", "himawari") or (region in CUSTOM_REGIONS and region not in eumetsat_regions):
        
        if region == "himawari":
            logging.info(f"Fetching Himawari-9 S3 data for region: {region}")
            if not S3FS_AVAILABLE:
                raise ImportError("s3fs library not found. Run 'pip install s3fs' to enable Himawari plots.")
            
            latest_file_url = get_latest_himawari_file(product_code)
            
            fs = s3fs.S3FileSystem(anon=True)
            try:
                with fs.open(latest_file_url, 'rb') as f:
                    ds = xr.open_dataset(f, engine="h5netcdf")
                    ds = ds.metpy.parse_cf()

                    if product_code == 2:
                        var = ds['CldAlbedo']
                        var_name = "Cloud Albedo"
                        units_str = "dimensionless"
                        arr = var.metpy.quantify().load()

                    elif product_code == 14:
                        var = ds['CldTopTemp']
                        var_name = "Cloud Top Temperature"
                        units_str = "K" 
                        arr = var.metpy.quantify().load()

                    elif product_code == 9:
                        # === Himawari Water Vapor (Band 9 - 6.9 µm) from L1b ===
                        logging.info("Converting Himawari L1b radiance to Brightness Temperature (Band 9 WV)...")
                        
                        # Himawari AHI Band 9 (Mid-level Water Vapor ~6.9µm)
                        c1 = 1.191042e-5
                        c2 = 1.4387752
                        nu = 1596.0807   # Central wavenumber for AHI Band 9

                        if 'Rad' in ds:
                            rad = ds['Rad']
                        else:
                            # fallback to first data variable if 'Rad' not found
                            rad = list(ds.data_vars.values())[0]

                        rad_safe = np.where(rad <= 0, np.nan, rad)
                        rad_term = (c1 * (nu ** 3)) / rad_safe
                        tb_kelvin = (c2 * nu) / np.log(rad_term + 1)
                        
                        var_name = "Water Vapor (6.9µm)"
                        units_str = "K"
                        arr = (tb_kelvin - 273.15).load()   # °C for plotting

                    else:
                        raise ValueError(f"Unsupported product_code {product_code} for Himawari")

            except Exception as e:
                raise RuntimeError(f"Failed to open/process Himawari S3 file {latest_file_url}. Error: {e}")

        else:
            logging.info(f"Fetching GOES-16/18 data for region: {region}")
            satellite_name = "goes/east" 
            if region in ("west_fulldisk", "mesosector2", "westcoast"):
                satellite_name = "goes/west"
            
            goes_channels = {
                2: {"channel": "02", "product": "CloudAndMoistureImagery"},
                9: {"channel": "09", "product": "CloudAndMoistureImagery"},
                14: {"channel": "14", "product": "CloudAndMoistureImagery"},
            }
            
            product_info = goes_channels.get(product_code)
            if product_info is None:
                raise ValueError(f"No GOES channel defined for product code {product_code} in {region}")

            glm_region = None
            if region in ("west_fulldisk", "goes_east_fd"):
                scan_area = "FullDisk"
                glm_region = "FullDisk"
            elif region in CUSTOM_REGIONS or region == "conus":
                scan_area = "CONUS"
                glm_region = "Conus"
            elif region in ("east_meso1", "west_meso1"):
                scan_area = "Mesoscale-1"
                glm_region = "Conus"
            elif region in ("east_meso2", "west_meso2"):
                scan_area = "Mesoscale-2"
                glm_region = "Conus"
                
            goes_product_name = product_info['product']
            channel_str = product_info['channel']

            if scan_area == "FullDisk":
                latest_file_url = get_latest_goes_fd_file(satellite_name, channel_str)
            else:
                latest_file_url = get_latest_goes_file(
                    satellite_name, scan_area, goes_product_name, channel_str
                )
            
            if overlay_glm and glm_region is not None:
                glm_url = get_latest_glm_file(satellite_name, glm_region)
                if glm_url:
                    try:
                        logging.info(f"Opening GLM: {glm_url}")
                        if glm_url.startswith('s3://'):
                            fs = s3fs.S3FileSystem(anon=True)
                            with fs.open(glm_url[5:], 'rb') as f:
                                ds_glm = xr.open_dataset(f, engine='h5netcdf')
                                ds_glm = ds_glm.load()
                        else:
                            ds_glm = xr.open_dataset(glm_url)
                            ds_glm = ds_glm.load()
                    except Exception as e:
                        print(f"[ERROR] Failed to open GLM dataset: {e}")
                        logging.warning(f"Failed to open GLM dataset: {e}")
                        ds_glm = None
                else:
                    ds_glm = None

            if scan_area == "FullDisk":
                fs = s3fs.S3FileSystem(anon=True)
                logging.info(f"Opening GOES S3: {latest_file_url}")
                with fs.open(latest_file_url, 'rb') as f:
                    ds = xr.open_dataset(f, engine='h5netcdf')
                    ds = ds.metpy.parse_cf()
                    if 'CMI' in ds:
                        var = ds['CMI']
                    elif 'Sectorized_CMI' in ds:
                        logging.warning(f"Scan area is FullDisk, but only 'Sectorized_CMI' was found.")
                        var = ds['Sectorized_CMI']
                    else:
                        raise ValueError("Could not find 'CMI' data variable in Full Disk dataset.")
                    channel_num = int(channel_str)
                    arr = var.metpy.quantify().load()
                    if channel_num <= 6:
                        units_str = "dimensionless"
                        var_name = f"Ch {channel_num} Reflectance"
                    else:
                        units_str = "K"
                        var_name = f"Ch {channel_num} Brightness Temperature"
            else:
                logging.info(f"Opening GOES OPeNDAP: {latest_file_url}")
                ds = xr.open_dataset(latest_file_url)
                ds = ds.metpy.parse_cf()

                if scan_area == "FullDisk":
                    if 'CMI' in ds:
                        var = ds['CMI']
                    elif 'Sectorized_CMI' in ds:
                         logging.warning(f"Scan area is FullDisk, but only 'Sectorized_CMI' was found.")
                         var = ds['Sectorized_CMI']
                    else:
                        raise ValueError("Could not find 'CMI' data variable in Full Disk dataset.")
                else: 
                    if 'Sectorized_CMI' in ds:
                        var = ds['Sectorized_CMI']
                    elif 'CMI' in ds:
                        logging.warning(f"Scan area is {scan_area}, but only 'CMI' was found.")
                        var = ds['CMI']
                    else:
                        raise ValueError(f"Could not find 'Sectorized_CMI' data variable in {scan_area} dataset.")
                
                channel_num = int(product_info['channel'])
                arr = var.metpy.quantify()
                if channel_num <= 6:
                    units_str = "dimensionless"
                    var_name = f"Ch {channel_num} Reflectance"
                else:
                    units_str = "K"
                    var_name = f"Ch {channel_num} Brightness Temperature"

        timestamp_str = "Unknown Time"
        if 'time_coverage_start' in ds.attrs:
            ts_str_raw = ds.attrs['time_coverage_start']
            timestamp_str = ts_str_raw.replace("T", " ").split('.')[0] + " UTC"
        elif 'time' in ds:
            try:
                time_val = ds.time.values[0] if ds.time.ndim > 0 else ds.time.values
                dt_obj = (time_val - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                dt_obj = dt.fromtimestamp(dt_obj, timezone.utc)
                timestamp_str = dt_obj.strftime("%Y-%m-%d %H:%M UTC")
            except Exception as e:
                logging.warning(f"Could not parse time from 'time' coordinate: {e}")

        plot_goes(ds, arr, var_name, units_str, region, output_path_preview, 
                  ds_glm=ds_glm, output_path_archive=output_path_archive,
                  timestamp_str=timestamp_str) 

        logging.info("=== SAT.PY END ===")
        
        # Fallback dictionary catch for the print statement
        pc_dict = product_codes.get(region, {})
        product_desc = pc_dict.get(product_code, f"Product {product_code}")

        print(f"Success: {product_desc} for {region.upper()} saved to {output_path_preview} (Archive: {dynamic_filename}) (GLM: {'On' if ds_glm else 'Off'})")
        return
        
    logging.info("=== SAT.PY END ===")
    print(f"Logic branch missed for {region} (output: {output_path_preview})")
    return


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python sat.py <region> <product_code> <output_file> [overlay_glm]")
        print(f"\nExample IR for GOES West Full Disk: python sat.py west_fulldisk 14 fd_ir.jpg")
        print(f"Example IR for GOES East Full Disk: python sat.py goes_east_fd 14 ge_ir.jpg")
        print(f"Example IR for Himawari Full Disk: python sat.py himawari 14 fd_ir.jpg")
        print(f"Supported Custom Regions: {', '.join(CUSTOM_REGIONS.keys())}")
        sys.exit(1)
    
    region       = sys.argv[1]
    product_code = int(sys.argv[2])
    output_path  = sys.argv[3]
    overlay_glm  = sys.argv[4].lower() == 'true' if len(sys.argv) == 5 else False
    
    try:
        result = sat(region, product_code, output_path, overlay_glm)
        if result: 
            print(result)
        sys.exit(0)
    except RuntimeError as e:
        if "Failed to fetch SEVIRI data" in str(e):
             logging.critical(f"EUMETSAT SERVER FAILURE: {e}")
             print(f"EUMETSAT SERVER FAILURE: {e}")
             sys.exit(1)
        elif "Failed to find latest file" in str(e):
             logging.critical(f"THREDDS CATALOG FAILURE: {e}")
             print(f"THREDDS CATALOG FAILURE: {e}")
             sys.exit(1)
        elif "Failed to find any Himawari data" in str(e):
             logging.critical(f"HIMAWARI S3 FAILURE: {e}")
             print(f"HIMAWARI S3 FAILURE: {e}")
             sys.exit(1)
        else:
            logging.critical(f"SCRIPT FAILED: {e}", exc_info=True) 
            print(f"SCRIPT FAILED: {e}")
            sys.exit(1)
    except ImportError as e:
        logging.critical(f"SCRIPT FAILED (Missing Library): {e}", exc_info=True) 
        print(f"SCRIPT FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"SCRIPT FAILED: {e}", exc_info=True) 
        print(f"SCRIPT FAILED: {e}")
        sys.exit(1)