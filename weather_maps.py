# weather_maps.py
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import ndimage
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import logging
from metpy.units import units
import metpy.calc as mpcalc
import os
from data_fetcher import DataFetcher
import asyncio
from weather_calculations import calc_mslp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EARTH_RADIUS = 6371000  # meters
OMEGA = 7.292e-5  # Earth's angular velocity (rad/s)
REGION = "CONUS"
OUTPUT_DIR = "output"

# Helper Functions
def compute_wind_speed(u, v):
    """Calculate wind speed from u and v components."""
    wind_speed_ms = np.sqrt(u**2 + v**2)
    return wind_speed_ms * 1.94384  # Convert to knots

def compute_vorticity(u, v, lat, lon):
    """Calculate absolute vorticity."""
    try:
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
    except Exception as e:
        logger.error(f"Error computing vorticity: {e}")
        return None

def compute_advection(phi, u, v, lat, lon):
    """Calculate advection of a scalar field."""
    try:
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
    except Exception as e:
        logger.error(f"Error computing advection: {e}")
        return None

def compute_dewpoint(T, rh):
    """Calculate dewpoint temperature from temperature and relative humidity."""
    try:
        T_C = T - 273.15
        rh = np.clip(rh, 1e-10, 100)
        ln_rh = np.log(rh / 100)
        numerator = 243.5 * ln_rh + (17.67 * T_C) / (243.5 + T_C)
        denominator = 17.67 - ln_rh - (17.67 * T_C) / (243.5 + T_C)
        return numerator / denominator
    except Exception as e:
        logger.error(f"Error computing dewpoint: {e}")
        return None

def compute_divergence(u, v, lat, lon):
    """Calculate divergence."""
    try:
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
        return divergence * 1e5
    except Exception as e:
        logger.error(f"Error computing divergence: {e}")
        return None

def frontogenesis_700hPa(T, u, v, lat, lon):
    """Calculate frontogenesis at 700 hPa."""
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
        logger.error(f"Error computing frontogenesis: {e}")
        return None

def plot_background(ax):
    """Set up map background for CONUS."""
    ax.set_extent([-125, -65, 25, 50], crs=ccrs.PlateCarree())
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

def get_time_dimension(ds, run_date):
    """Determine the appropriate time dimension for the dataset."""
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
                logger.debug(f"Found time dimension: {dim}")
                selected_dims[dim] = 0
                break
        if not selected_dims and 'time' in ds.dims:
            selected_dims['time'] = 0
            logger.debug(f"Added time dimension: time")
        if not selected_dims:
            logger.error(f"No time dimension found. Available dimensions: {ds.dims}")
            raise ValueError("No valid time dimension found in dataset")
        return selected_dims
    except Exception as e:
        logger.error(f"Error in get_time_dimension: {e}")
        raise

async def generate_map(ds, run_date, level, variable, cmap, title, cb_label, levels=None):
    """Generate a weather map for a specific variable and level."""
    try:
        time_dims = get_time_dimension(ds, run_date)
        ds = ds.isel(**time_dims)
        lon = ds.get('longitude').values
        lat = ds.get('latitude').values
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
            temp = ds['Temperature_isobaric'].sel(isobaric=level, method='nearest').squeeze().metpy.convert_units('degC').metpy.dequantify().values.copy()
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

        if data is None or np.isnan(data).all():
            raise ValueError(f"Computed data for {variable} is None or contains only NaN values")

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
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        output_path = os.path.join(OUTPUT_DIR, f"conus_{variable}_{level//100}_{run_date.strftime('%Y%m%d_%H%MZ')}.png")
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Map saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_map: {e}")
        return None

async def generate_mslp_temp_map():
    """Generate MSLP with temperature gradient map."""
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs_surface(
            run_date, {"north": 50, "south": 25, "east": -65, "west": -125}
        )
        if ds is None:
            raise ValueError("Failed to retrieve surface data.")
        time_dims = get_time_dimension(ds, run_date)
        ds = ds.isel(**time_dims)
        lon = ds.get('longitude').values
        lat = ds.get('latitude').values
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        temp_surface = ds["t2m"].squeeze().metpy.convert_units("degC").metpy.dequantify()
        surface_pressure = ds["sp"].squeeze().metpy.dequantify()
        elevation = ds["orog"].squeeze().metpy.dequantify()
        u_wind = ds["u10"].squeeze().metpy.dequantify()
        v_wind = ds["v10"].squeeze().metpy.dequantify()
        mslp = await calc_mslp(surface_pressure, elevation, temp_surface)
        mslp_smooth = ndimage.gaussian_filter(mslp, sigma=3, order=0)
        temp_grad_x, temp_grad_y = np.gradient(temp_surface, lon[1] - lon[0], lat[1] - lat[0])
        temp_grad_mag = np.sqrt(temp_grad_x**2 + temp_grad_y**2)
        frontogenesis = compute_advection(temp_grad_mag, u_wind, v_wind, lat, lon)
        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': crs})
        fig.patch.set_facecolor('lightsteelblue')
        plot_background(ax)
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
        ax.set_title(f"CONUS: MSLP with Temperature Gradient (°C) and Features", fontsize=16)
        fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=12, y=1.02)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
        cb.set_label('Temperature (°C)', size='large')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        output_path = os.path.join(OUTPUT_DIR, f"conus_mslp_temp_{run_date.strftime('%Y%m%d_%H%MZ')}.png")
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"MSLP map saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_mslp_temp_map: {e}")
        return None

async def generate_thermal_wind_map(level_lower=85000, level_upper=50000):
    """Generate thermal wind map for 850-500 hPa."""
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs(
            run_date, [level_lower, level_upper],
            ['Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric'],
            {"north": 50, "south": 25, "east": -65, "west": -125}
        )
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
        ax.set_title(f"{int(level_upper/100)} hPa Heights, {int(level_lower/100)}-{int(level_upper/100)} hPa Thickness, and {int(level_upper/100)} hPa Wind (CONUS)", fontsize=16)
        plot_background(ax)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        output_path = os.path.join(OUTPUT_DIR, f"conus_thermal_wind_{run_date.strftime('%Y%m%d_%H%MZ')}.png")
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Thermal wind map saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_thermal_wind_map: {e}")
        return None

async def wind300():
    """Generate 300 hPa wind map."""
    print('Generating 300 hPa wind map for CONUS...')
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs(
            run_date, 30000,
            ['Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'Relative_humidity_isobaric', 'Temperature_isobaric'],
            {"north": 50, "south": 25, "east": -65, "west": -125}
        )
        if ds is None:
            print('Failed to retrieve data for the 300 hPa wind map.')
            return
        output_path = await generate_map(
            ds, run_date, 30000, 'wind_speed', 'cool',
            '300-hPa Wind Speeds and Heights (CONUS)', 'Wind Speed (knots)'
        )
        if output_path:
            print(f"Map saved to {output_path}")
        else:
            print('Failed to generate the 300 hPa wind map due to missing or invalid data.')
    except Exception as e:
        logger.error(f"Error in wind300: {e}")
        print('Failed to generate the 300 hPa wind map.')

async def wind500():
    """Generate 500 hPa wind map."""
    print('Generating 500 hPa wind map for CONUS...')
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta5(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs(
            run_date, 50000,
            ['Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'Relative_humidity_isobaric', 'Temperature_isobaric'],
            {"north": 50, "south": 25, "east": -65, "west": -125}
        )
        if ds is None:
            print('Failed to retrieve data for the 500 hPa wind map.')
            return
        output_path = await generate_map(
            ds, run_date, 50000, 'wind_speed', 'YlOrBr',
            '500-hPa Wind Speeds and Heights (CONUS)', 'Wind Speed (knots)'
        )
        if output_path:
            print(f"Map saved to {output_path}")
        else:
            print('Failed to generate the 500 hPa wind map due to missing or invalid data.')
    except Exception as e:
        logger.error(f"Error in wind500: {e}")
        print('Failed to generate the 500 hPa wind map.')

async def vort500():
    """Generate 500 hPa relative vorticity map."""
    print('Generating 500 hPa relative vorticity map for CONUS...')
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs(
            run_date, 50000,
            ['Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'Relative_humidity_isobaric', 'Temperature_isobaric'],
            {"north": 50, "south": 25, "east": -65, "west": -125}
        )
        if ds is None:
            print('Failed to retrieve data for the 500 hPa relative vorticity map.')
            return
        output_path = await generate_map(
            ds, run_date, 50000, 'vorticity', 'seismic',
            '500-hPa Relative Vorticity and Heights (CONUS)',
            r'Relative Vorticity ($10^{-5}$ s$^{-1}$)', levels=np.linspace(-15, 15, 31)
        )
        if output_path:
            print(f"Map saved to {output_path}")
        else:
            print('Failed to generate the 500 hPa relative vorticity map due to missing or invalid data.')
    except Exception as e:
        logger.error(f"Error in vort500: {e}")
        print('Failed to generate the 500 hPa relative vorticity map.')

async def fronto700():
    """Generate 700 hPa frontogenesis map."""
    print('Generating 700 hPa frontogenesis map for CONUS...')
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs(
            run_date, 70000,
            ['Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'Relative_humidity_isobaric', 'Temperature_isobaric'],
            {"north": 50, "south": 25, "east": -65, "west": -125}
        )
        if ds is None:
            print('Failed to retrieve data for the 700 hPa frontogenesis map.')
            return
        output_path = await generate_map(
            ds, run_date, 70000, 'frontogenesis', 'RdBu_r',
            '700-hPa Frontogenesis and Heights (CONUS)',
            'Frontogenesis (K/100km/3hr)', levels=np.linspace(-10, 10, 41)
        )
        if output_path:
            print(f"Map saved to {output_path}")
        else:
            print('Failed to generate the 700 hPa frontogenesis map due to missing or invalid data.')
    except Exception as e:
        logger.error(f"Error in fronto700: {e}")
        print('Failed to generate the 700 hPa frontogenesis map.')

async def rh700():
    """Generate 700 hPa relative humidity map."""
    print('Generating 700 hPa relative humidity map for CONUS...')
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs(
            run_date, 70000,
            ['Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'Relative_humidity_isobaric', 'Temperature_isobaric'],
            {"north": 50, "south": 25, "east": -65, "west": -125}
        )
        if ds is None:
            print('Failed to retrieve data for the 700 hPa relative humidity map.')
            return
        output_path = await generate_map(
            ds, run_date, 70000, 'relative_humidity', 'BuGn',
            '700-hPa Relative Humidity and Heights (CONUS)', 'Relative Humidity (%)'
        )
        if output_path:
            print(f"Map saved to {output_path}")
        else:
            print('Failed to generate the 700 hPa relative humidity map due to missing or invalid data.')
    except Exception as e:
        logger.error(f"Error in rh700: {e}")
        print('Failed to generate the 700 hPa relative humidity map.')

async def wind850():
    """Generate 850 hPa wind map."""
    print('Generating 850 hPa wind map for CONUS...')
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs(
            run_date, 85000,
            ['Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'Relative_humidity_isobaric', 'Temperature_isobaric'],
            {"north": 50, "south": 25, "east": -65, "west": -125}
        )
        if ds is None:
            print('Failed to retrieve data for the 850 hPa wind map.')
            return
        output_path = await generate_map(
            ds, run_date, 85000, 'wind_speed', 'YlOrBr',
            '850-hPa Wind Speeds and Heights (CONUS)', 'Wind Speed (knots)'
        )
        if output_path:
            print(f"Map saved to {output_path}")
        else:
            print('Failed to generate the 850 hPa wind map due to missing or invalid data.')
    except Exception as e:
        logger.error(f"Error in wind850: {e}")
        print('Failed to generate the 850 hPa wind map.')

async def dew850():
    """Generate 850 hPa dewpoint map."""
    print('Generating 850 hPa dewpoint map for CONUS...')
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs(
            run_date, 85000,
            ['Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'Relative_humidity_isobaric', 'Temperature_isobaric'],
            {"north": 50, "south": 25, "east": -65, "west": -125}
        )
        if ds is None:
            print('Failed to retrieve data for the 850 hPa dewpoint map.')
            return
        output_path = await generate_map(
            ds, run_date, 85000, 'dewpoint', 'BuGn',
            '850-hPa Dewpoint (°C) (CONUS)', 'Dewpoint (°C)',
            levels=np.linspace(-20, 20, 41)
        )
        if output_path:
            print(f"Map saved to {output_path}")
        else:
            print('Failed to generate the 850 hPa dewpoint map due to missing or invalid data.')
    except Exception as e:
        logger.error(f"Error in dew850: {e}")
        print('Failed to generate the 850 hPa dewpoint map.')

async def mAdv850():
    """Generate 850 hPa moisture advection map."""
    print('Generating 850 hPa moisture advection map for CONUS...')
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs(
            run_date, 85000,
            ['Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'Relative_humidity_isobaric', 'Temperature_isobaric'],
            {"north": 50, "south": 25, "east": -65, "west": -125}
        )
        if ds is None:
            print('Failed to retrieve data for the 850 hPa moisture advection map.')
            return
        output_path = await generate_map(
            ds, run_date, 85000, 'moisture_advection', 'PRGn',
            '850-hPa Moisture Advection (CONUS)', 'Moisture Advection (%/hour)'
        )
        if output_path:
            print(f"Map saved to {output_path}")
        else:
            print('Failed to generate the 850 hPa moisture advection map due to missing or invalid data.')
    except Exception as e:
        logger.error(f"Error in mAdv850: {e}")
        print('Failed to generate the 850 hPa moisture advection map.')

async def tAdv850():
    """Generate 850 hPa temperature advection map."""
    print('Generating 850 hPa temperature advection map for CONUS...')
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs(
            run_date, 85000,
            ['Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'Relative_humidity_isobaric', 'Temperature_isobaric'],
            {"north": 50, "south": 25, "east": -65, "west": -125}
        )
        if ds is None:
            print('Failed to retrieve data for the 850 hPa temperature advection map.')
            return
        output_path = await generate_map(
            ds, run_date, 85000, 'temp_advection', 'coolwarm',
            '850-hPa Temperature Advection (CONUS)', 'Temperature Advection (K/hour)',
            levels=np.linspace(-20, 20, 41)
        )
        if output_path:
            print(f"Map saved to {output_path}")
        else:
            print('Failed to generate the 850 hPa temperature advection map due to missing or invalid data.')
    except Exception as e:
        logger.error(f"Error in tAdv850: {e}")
        print('Failed to generate the 850 hPa temperature advection map.')

async def mslp_temp():
    """Generate MSLP with temperature gradient map."""
    print('Generating MSLP with temperature gradient map for CONUS...')
    output_path = await generate_mslp_temp_map()
    if output_path:
        print(f"Map saved to {output_path}")
    else:
        print('Failed to generate the MSLP with temperature gradient map due to missing or invalid data.')

async def divcon300():
    """Generate 300 hPa divergence map."""
    print('Generating 300 hPa divergence map for CONUS...')
    fetcher = DataFetcher()
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 2.5:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    try:
        ds, run_date = await fetcher.fetch_gfs(
            run_date, 30000,
            ['Geopotential_height_isobaric', 'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'Relative_humidity_isobaric', 'Temperature_isobaric'],
            {"north": 50, "south": 25, "east": -65, "west": -125}
        )
        if ds is None:
            print('Failed to retrieve data for the 300 hPa divergence/convergence map.')
            return
        output_path = await generate_map(
            ds, run_date, 30000, 'divergence', 'RdBu_r',
            '300-hPa Divergence/Convergence (CONUS)', r'Divergence ($10^{-5}$ s$^{-1}$)'
        )
        if output_path:
            print(f"Map saved to {output_path}")
        else:
            print('Failed to generate the 300 hPa divergence/convergence map due to missing or invalid data.')
    except Exception as e:
        logger.error(f"Error in divcon300: {e}")
        print('Failed to generate the 300 hPa divergence/convergence map.')

async def thermal_wind():
    """Generate thermal wind map."""
    print('Generating thermal wind map for CONUS...')
    output_path = await generate_thermal_wind_map()
    if output_path:
        print(f"Map saved to {output_path}")
    else:
        print('Failed to generate the thermal wind map due to missing or invalid data.')
