# au_weather_maps.py
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from siphon.catalog import TDSCatalog
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import ndimage
import io
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import logging
from metpy.units import units
import metpy.calc as mpcalc
import os
import zipfile
import requests
from cartopy.io import shapereader

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Constants
EARTH_RADIUS = 6371000  # meters
OMEGA = 7.292e-5  # Earth's angular velocity (rad/s)
REGION = "Aus/Nz"  # Define the region
OUTPUT_DIR = "output"  # Output directory for saving plots

# List of major cities in Australia and New Zealand
cities = [
    {'name': 'Sydney', 'lat': -33.8688, 'lon': 151.2093},
    {'name': 'Melbourne', 'lat': -37.8136, 'lon': 144.9631},
    {'name': 'Brisbane', 'lat': -27.4698, 'lon': 153.0251},
    {'name': 'Perth', 'lat': -31.9505, 'lon': 115.8605},
    {'name': 'Adelaide', 'lat': -34.9285, 'lon': 138.6007},
    {'name': 'Canberra', 'lat': -35.2809, 'lon': 149.1300},
    {'name': 'Hobart', 'lat': -42.8821, 'lon': 147.3272},
    {'name': 'Darwin', 'lat': -12.4634, 'lon': 130.8456},
    {'name': 'Gold Coast', 'lat': -28.0167, 'lon': 153.4000},
    {'name': 'Newcastle', 'lat': -32.9283, 'lon': 151.7817},
    {'name': 'Wellington', 'lat': -41.2865, 'lon': 174.7762},
    {'name': 'Auckland', 'lat': -36.8485, 'lon': 174.7633},
    {'name': 'Christchurch', 'lat': -43.5321, 'lon': 172.6362},
    {'name': 'Dunedin', 'lat': -45.8788, 'lon': 170.5028},
    {'name': 'Hamilton', 'lat': -37.7870, 'lon': 175.2793},
    {'name': 'Geraldton', 'lat': -28.7790, 'lon': 114.6146},
    {'name': 'Albany', 'lat': -35.0031, 'lon': 117.8659},
]

# Download and load highway shapefile
def download_shapefile(url, output_dir, filename):
    """Downloads and extracts a shapefile from a URL to a specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

    shapefile_path = os.path.join(output_dir, filename)
    if os.path.exists(shapefile_path):
        logging.info(f"Shapefile {filename} already exists in {output_dir}.")
        return True

    zip_path = os.path.join(output_dir, 'temp.zip')
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Downloaded zip file from {url} to {zip_path}.")
    except requests.RequestException as e:
        logging.error(f"Failed to download zip file: {e}")
        return False

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        logging.info(f"Extracted shapefile to {output_dir}.")
        os.remove(zip_path)
        return True
    except zipfile.BadZipFile as e:
        logging.error(f"Failed to extract zip file: {e}")
        return False

australia_shapefile_url = 'https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_roads.zip'
output_dir = os.path.join(OUTPUT_DIR, 'shapefiles')  # Store shapefiles in output/shapefiles
filename = 'ne_10m_roads.shp'

if download_shapefile(australia_shapefile_url, output_dir, filename):
    try:
        highways = cfeature.ShapelyFeature(
            shapereader.Reader(os.path.join(output_dir, filename)).geometries(),
            ccrs.PlateCarree(),
            edgecolor='purple',
            facecolor='none',
            linewidth=0.7
        )
    except Exception as e:
        logging.warning(f"Failed to load highways shapefile: {e}")
        highways = None
else:
    logging.warning("Failed to download or extract shapefile. Highways will not be added.")
    highways = None

# Utility Functions (unchanged from original)
def compute_wind_speed(u, v):
    wind_speed_ms = np.sqrt(u**2 + v**2)
    return wind_speed_ms * 1.94384

def compute_vorticity(u, v, lat, lon):
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
    return zeta_r * 1e5

def compute_advection(phi, u, v, lat, lon):
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
    T_C = T - 273.15
    rh = np.clip(rh, 1e-10, 100)
    ln_rh = np.log(rh / 100.0)
    a = 17.67
    b = 243.5
    gamma = (a * T_C) / (b + T_C)
    dewpoint_C = (b * (ln_rh + gamma)) / (a - ln_rh - gamma)
    return dewpoint_C

def compute_divergence(u, v, lat, lon):
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

def add_cities(ax):
    for city in cities:
        ax.plot(city['lon'], city['lat'], 'o', color='white', markeredgecolor='black', markersize=4, transform=ccrs.PlateCarree())
        ax.text(city['lon'] + 0.5, city['lat'] + 0.5, city['name'], color='black', fontsize=6, transform=ccrs.PlateCarree(),
                ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

def add_highway_labels(ax):
    if highways is not None:
        shapefile_path = os.path.join(output_dir, filename)
        try:
            reader = shapereader.Reader(shapefile_path)
            for geom, record in zip(reader.geometries(), reader.records()):
                label = record.attributes.get('name', '')
                if label and (label.startswith('M') or label.startswith('A')) and record.attributes.get('scalerank', 999) <= 2:
                    midpoint = geom.centroid
                    ax.text(midpoint.x, midpoint.y, label, color='blue', fontsize=5, transform=ccrs.PlateCarree(),
                            ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        except Exception as e:
            logging.warning(f"Failed to add highway labels: {e}")

def plot_background(ax):
    ax.set_extent([112, 179, -47, -10], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='-', linewidths=2.5, edgecolor='#750b7a')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle='--', linewidths=2, edgecolor='#750b7a')
    ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.7)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.7)
    ax.add_feature(cfeature.LAND, facecolor='lightgreen', alpha=0.7)
    if highways is not None:
        ax.add_feature(highways)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

def get_time_dimension(ds, run_date):
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

def get_gfs_data_for_level(level):
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    logging.debug(f"Selected GFS run time for level {level} Pa: {run_date}")

    catalog_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml'
    try:
        cat = TDSCatalog(catalog_url)
        latest_dataset = list(cat.datasets.values())[0]
        ncss = latest_dataset.subset()
        query = ncss.query()
        query.accept('netcdf4')
        query.time(run_date)
        query.variables('Geopotential_height_isobaric',
                        'u-component_of_wind_isobaric',
                        'v-component_of_wind_isobaric',
                        'Relative_humidity_isobaric',
                        'Temperature_isobaric')
        query.vertical_level([level])
        query.lonlat_box(north=-10, south=-47, east=179, west=112)
        data = ncss.get_data(query)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
        ds = ds.metpy.parse_cf()
        logging.debug(f"Available variables for level {level}: {list(ds.variables)}")
        return ds, run_date
    except Exception as e:
        logging.error(f"Error fetching THREDDS data for level {level}: {e}")
        return None, None

def get_gfs_surface_data():
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    logging.debug(f"Selected GFS run time for surface data: {run_date}")

    bounds = {"north": -10, "south": -47, "east": 179, "west": 112}
    catalog_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml'
    try:
        cat = TDSCatalog(catalog_url)
        latest_dataset = list(cat.datasets.values())[0]
        ncss = latest_dataset.subset()
        query = ncss.query()
        query.accept('netcdf4')
        query.time(run_date)
        query.variables(
            'Temperature_surface',
            'Pressure_surface',
            'Geopotential_height_surface',
            'u-component_of_wind_height_above_ground',
            'v-component_of_wind_height_above_ground'
        )
        query.lonlat_box(north=bounds['north'], south=bounds['south'], east=bounds['east'], west=bounds['west'])
        query.vertical_level(10)
        data = ncss.get_data(query)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
        ds = ds.metpy.parse_cf()
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
        return ds, run_date
    except Exception as e:
        logging.error(f"Error in get_gfs_surface_data: {e}")
        return None, None

def generate_map(ds, run_date, level, variable, cmap, title, cb_label, levels=None):
    try:
        time_dims = get_time_dimension(ds, run_date)
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
        if u_wind.ndim != 2 or v_wind.ndim != 2:
            raise ValueError(f"Wind components are not 2D: u={u_wind.shape}, v={v_wind.shape}")
        if variable == 'wind_speed':
            data = compute_wind_speed(u_wind, v_wind)
        elif variable == 'vorticity':
            data = compute_vorticity(u_wind, v_wind, lat, lon)
        elif variable == 'relative_humidity':
            data = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            if data.ndim != 2:
                raise ValueError(f"Relative humidity is not 2D: {data.shape}")
        elif variable == 'temp_advection':
            temp = ds['Temperature_isobaric'].sel(isobaric=level, method='nearest').squeeze().metpy.convert_units('degC').metpy.dequantify().values.copy()
            if temp.ndim != 2:
                raise ValueError(f"Temperature is not 2D: {temp.shape}")
            data = compute_advection(temp, u_wind, v_wind, lat, lon) * 3600
        elif variable == 'moisture_advection':
            rh = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            if rh.ndim != 2:
                raise ValueError(f"Relative humidity is not 2D: {data.shape}")
            data = compute_advection(rh, u_wind, v_wind, lat, lon) * 1e4
        elif variable == 'dewpoint':
            temp = ds['Temperature_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            rh = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            if temp.ndim != 2 or rh.ndim != 2:
                raise ValueError(f"Temperature or RH is not 2D: temp={temp.shape}, rh={rh.shape}")
            data = compute_dewpoint(temp, rh)
        elif variable == 'divergence':
            data = compute_divergence(u_wind, v_wind, lat, lon) * 1e5
        elif variable == 'frontogenesis':
            if level != 70000:
                raise ValueError("Frontogenesis is only computed at 700 hPa.")
            temp = ds['Temperature_isobaric'].sel(isobaric=level, method='nearest').squeeze().metpy.convert_units('degC').metpy.dequantify().values.copy()
            if temp.ndim != 2:
                raise ValueError(f"Temperature is not 2D: {temp.shape}")
            data = frontogenesis_700hPa(temp, u_wind, v_wind, lat, lon)
            if data is None:
                raise ValueError("Failed to compute frontogenesis at 700 hPa.")
        else:
            raise ValueError(f"Unsupported variable type: {variable}")
        if np.isnan(data).all():
            raise ValueError(f"Computed data for {variable} contains only NaN values")
        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': crs})
        fig.patch.set_facecolor('lightsteelblue')
        if levels is None:
            cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs)
        else:
            cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs, levels=levels)
        c = ax.contour(lon_2d, lat_2d, heights_smooth, colors='black', linewidths=2, transform=crs)
        ax.clabel(c, fontsize=8, inline=1, fmt='%i')
        u_wind_knots = u_wind * 1.94384
        v_wind_knots = v_wind * 1.94384
        ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind_knots[::5, ::5], v_wind_knots[::5, ::5],
                 transform=crs, length=6)
        ax.set_title(f"{title} {run_date.strftime('%d %B %Y %H:%MZ')}", fontsize=16)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03, extend='both')
        cb.set_label(cb_label, size='large')
        plot_background(ax)
        add_cities(ax)
        add_highway_labels(ax)
        # Skip logo addition due to missing files
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        output_path = os.path.join(OUTPUT_DIR, f"au_{variable}_{level//100}_{run_date.strftime('%Y%m%d_%H%MZ')}.png")
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.close(fig)
        return output_path
    except ValueError as e:
        logging.error(f"Data extraction error in generate_map: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred in generate_map: {e}")
        return None

def generate_mslp_temp_map():
    try:
        ds, run_date = get_gfs_surface_data()
        if ds is None:
            raise ValueError("Failed to retrieve surface data.")
        time_dims = get_time_dimension(ds, run_date)
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
        mslp_grad_x, mslp_grad_y = np.gradient(mslp_smooth, lon[1] - lon[0], lat[1] - lat[0])
        mslp_lap = np.gradient(mslp_grad_x, axis=1) + np.gradient(mslp_grad_y, axis=0)
        curvature = mslp_grad_x * np.gradient(mslp_grad_y, axis=0) - mslp_grad_y * np.gradient(mslp_grad_x, axis=1)
        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': crs})
        fig.patch.set_facecolor('lightsteelblue')
        plot_background(ax)
        add_cities(ax)
        add_highway_labels(ax)
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
        main_title = f"Australia & New Zealand: MSLP with Temperature Gradient (°C) and Features"
        ax.set_title(main_title, fontsize=16)
        fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=12, y=1.02)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
        cb.set_label('Temperature (°C)', size='large')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        output_path = os.path.join(OUTPUT_DIR, f"au_mslp_temp_{run_date.strftime('%Y%m%d_%H%MZ')}.png")
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.close(fig)
        return output_path
    except Exception as e:
        logging.error(f"Error in generate_mslp_temp_map: {e}")
        return None

def generate_thermal_wind_map(level_lower=85000, level_upper=50000):
    try:
        ds, run_date = get_gfs_data_for_level([level_lower, level_upper])
        if ds is None:
            return None
        time_dims = get_time_dimension(ds, run_date)
        ds = ds.isel(**time_dims)
        lon = ds['longitude'].values
        lat = ds['latitude'].values
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        height_lower = ds['Geopotential_height_isobaric'].sel(isobaric=level_lower).squeeze().values
        height_upper = ds['Geopotential_height_isobaric'].sel(isobaric=level_upper).squeeze().values
        thickness = height_upper - height_lower
        height_upper_smooth = ndimage.gaussian_filter(height_upper, sigma=3, order=0)
        u_wind_upper = ds['u-component_of_wind_isobaric'].sel(isobaric=level_upper).squeeze().values
        v_wind_upper = ds['v-component_of_wind_isobaric'].sel(isobaric=level_upper).squeeze().values
        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': crs})
        fig.patch.set_facecolor('lightsteelblue')
        thickness_levels = np.arange(np.floor(np.min(thickness)/30)*30,
                                    np.ceil(np.max(thickness)/30)*30 + 30, 30)
        c_thickness = ax.contour(lon_2d, lat_2d, thickness, levels=thickness_levels,
                                colors='red', linestyles='dashed', linewidths=1.5, transform=crs)
        ax.clabel(c_thickness, fontsize=8, inline=True, fmt='%i')
        height_levels = np.arange(np.floor(np.min(height_upper_smooth)/60)*60,
                                 np.ceil(np.max(height_upper_smooth)/60)*60 + 60, 60)
        c_heights = ax.contour(lon_2d, lat_2d, height_upper_smooth, levels=height_levels,
                              colors='black', linewidths=2, transform=crs)
        ax.clabel(c_heights, fontsize=8, inline=True, fmt='%i')
        u_wind_knots = u_wind_upper * 1.94384
        v_wind_knots = v_wind_upper * 1.94384
        ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind_knots[::5, ::5], v_wind_knots[::5, ::5],
                 transform=crs, length=6)
        title = f"{int(level_upper/100)} hPa Heights, {int(level_lower/100)}-{int(level_upper/100)} hPa Thickness, and {int(level_upper/100)} hPa Wind (Australia & New Zealand)"
        ax.set_title(f"{title} {run_date.strftime('%d %B %Y %H:%MZ')}", fontsize=16)
        plot_background(ax)
        add_cities(ax)
        add_highway_labels(ax)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        output_path = os.path.join(OUTPUT_DIR, f"au_thermal_wind_{run_date.strftime('%Y%m%d_%H%MZ')}.png")
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.close(fig)
        return output_path
    except Exception as e:
        logging.error(f"Error in generate_thermal_wind_map: {e}")
        return None

async def au_wind300():
    print('Generating 300 hPa wind map for Australia & New Zealand...')
    ds, run_date = get_gfs_data_for_level(30000)
    if ds is None:
        print('Failed to retrieve data for the 300 hPa wind map.')
        return
    output_path = generate_map(
        ds, run_date, 30000, 'wind_speed', 'cool', '300-hPa Wind Speeds and Heights (Australia & New Zealand)', 'Wind Speed (knots)'
    )
    if output_path is None:
        print('Failed to generate the 300 hPa wind map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")

async def au_wind500():
    print('Generating 500 hPa wind map for Australia & New Zealand...')
    ds, run_date = get_gfs_data_for_level(50000)
    if ds is None:
        print('Failed to retrieve data for the 500 hPa wind map.')
        return
    output_path = generate_map(
        ds, run_date, 50000, 'wind_speed', 'YlOrBr', '500-hPa Wind Speeds and Heights (Australia & New Zealand)', 'Wind Speed (knots)'
    )
    if output_path is None:
        print('Failed to generate the 500 hPa wind map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")

async def au_vort500():
    print('Generating 500 hPa relative vorticity map for Australia & New Zealand...')
    ds, run_date = get_gfs_data_for_level(50000)
    if ds is None:
        print('Failed to retrieve data for the 500 hPa relative vorticity map.')
        return
    output_path = generate_map(
        ds, run_date, 50000, 'vorticity', 'seismic', '500-hPa Relative Vorticity and Heights (Australia & New Zealand)',
        r'Relative Vorticity ($10^{-5}$ s$^{-1}$)', levels=np.linspace(-15, 15, 31)
    )
    if output_path is None:
        print('Failed to generate the 500 hPa relative vorticity map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")

async def au_fronto700():
    print('Generating 700 hPa frontogenesis map for Australia & New Zealand...')
    ds, run_date = get_gfs_data_for_level(70000)
    if ds is None:
        print('Failed to retrieve data for the 700 hPa frontogenesis map.')
        return
    output_path = generate_map(
        ds, run_date, 70000, 'frontogenesis', 'RdBu_r', '700-hPa Frontogenesis and Heights (Australia & New Zealand)',
        'Frontogenesis (K/100km/3hr)', levels=np.linspace(-10, 10, 41)
    )
    if output_path is None:
        print('Failed to generate the 700 hPa frontogenesis map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")

async def au_rh700():
    print('Generating 700 hPa relative humidity map for Australia & New Zealand...')
    ds, run_date = get_gfs_data_for_level(70000)
    if ds is None:
        print('Failed to retrieve data for the 700 hPa relative humidity map.')
        return
    output_path = generate_map(
        ds, run_date, 70000, 'relative_humidity', 'BuGn', '700-hPa Relative Humidity and Heights (Australia & New Zealand)', 'Relative Humidity (%)'
    )
    if output_path is None:
        print('Failed to generate the 700 hPa relative humidity map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")

async def au_wind850():
    print('Generating 850 hPa wind map for Australia & New Zealand...')
    ds, run_date = get_gfs_data_for_level(85000)
    if ds is None:
        print('Failed to retrieve data for the 850 hPa wind map.')
        return
    output_path = generate_map(
        ds, run_date, 85000, 'wind_speed', 'YlOrBr', '850-hPa Wind Speeds and Heights (Australia & New Zealand)', 'Wind Speed (knots)'
    )
    if output_path is None:
        print('Failed to generate the 850 hPa wind map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")

async def au_dew850():
    print('Generating 850 hPa dewpoint map for Australia & New Zealand...')
    ds, run_date = get_gfs_data_for_level(85000)
    if ds is None:
        print('Failed to retrieve data for the 850 hPa dewpoint map.')
        return
    output_path = generate_map(
        ds, run_date, 85000, 'dewpoint', 'BuGn', '850-hPa Dewpoint (°C) (Australia & New Zealand)', 'Dewpoint (°C)',
        levels=np.linspace(-20, 20, 41)
    )
    if output_path is None:
        print('Failed to generate the 850 hPa dewpoint map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")

async def au_mAdv850():
    print('Generating 850 hPa moisture advection map for Australia & New Zealand...')
    ds, run_date = get_gfs_data_for_level(85000)
    if ds is None:
        print('Failed to retrieve data for the 850 hPa moisture advection map.')
        return
    output_path = generate_map(
        ds, run_date, 85000, 'moisture_advection', 'PRGn', '850-hPa Moisture Advection (Australia & New Zealand)', 'Moisture Advection (%/hour)'
    )
    if output_path is None:
        print('Failed to generate the 850 hPa moisture advection map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")

async def au_tAdv850():
    print('Generating 850 hPa temperature advection map for Australia & New Zealand...')
    ds, run_date = get_gfs_data_for_level(85000)
    if ds is None:
        print('Failed to retrieve data for the 850 hPa temperature advection map.')
        return
    output_path = generate_map(
        ds, run_date, 85000, 'temp_advection', 'coolwarm', '850-hPa Temperature Advection (Australia & New Zealand)', 'Temperature Advection (K/hour)',
        levels=np.linspace(-20, 20, 41)
    )
    if output_path is None:
        print('Failed to generate the 850 hPa temperature advection map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")

async def au_mslp_temp():
    print('Generating MSLP with temperature gradient map for Australia & New Zealand...')
    output_path = generate_mslp_temp_map()
    if output_path is None:
        print('Failed to generate the MSLP with temperature gradient map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")

async def au_divcon300():
    print('Generating 300 hPa divergence map for Australia & New Zealand...')
    ds, run_date = get_gfs_data_for_level(30000)
    if ds is None:
        print('Failed to retrieve data for the 300 hPa divergence/convergence map.')
        return
    output_path = generate_map(
        ds, run_date, 30000, 'divergence', 'RdBu_r',
        '300-hPa Divergence/Convergence (Australia & New Zealand)', r'Divergence ($10^{-5}$ s$^{-1}$)'
    )
    if output_path is None:
        print('Failed to generate the 300 hPa divergence/convergence map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")

async def au_thermal_wind():
    print('Generating thermal wind map for Australia & New Zealand...')
    output_path = generate_thermal_wind_map()
    if output_path is None:
        print('Failed to generate the thermal wind map due to missing or invalid data.')
    else:
        print(f"Map saved to {output_path}")
