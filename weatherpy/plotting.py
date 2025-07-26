import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES, StationPlot
from matplotlib.patheffects import withStroke
from datetime import datetime, timedelta
import numpy as np
from dateutil import tz
import matplotlib.gridspec as gridspec
import pytz
import logging

from . import colormaps, geometry, settings, thermodynamics, unit_conversion, utilities

logger = logging.getLogger(__name__)

async def plot_relative_humidity(
    western_bound=None, eastern_bound=None, southern_bound=None, northern_bound=None,
    fig_x_length=None, fig_y_length=None, signature_x_position=None, signature_y_position=None,
    color_table_shrink=1, title_fontsize=12, subplot_title_fontsize=10, signature_fontsize=10,
    colorbar_fontsize=8, show_rivers=True, reference_system='States & Counties',
    show_state_borders=False, show_county_borders=False, show_gacc_borders=False,
    show_psa_borders=False, show_cwa_borders=False, show_nws_firewx_zones=False,
    show_nws_public_zones=False, state_border_linewidth=2, county_border_linewidth=1,
    gacc_border_linewidth=2, psa_border_linewidth=1, cwa_border_linewidth=1,
    nws_firewx_zones_linewidth=0.5, nws_public_zones_linewidth=0.5, state_border_linestyle='-',
    county_border_linestyle='-', gacc_border_linestyle='-', psa_border_linestyle='-',
    cwa_border_linestyle='-', nws_firewx_zones_linestyle='-', nws_public_zones_linestyle='-',
    psa_color='black', gacc_color='black', cwa_color='black', fwz_color='black',
    pz_color='black', show_sample_points=True, sample_point_fontsize=8, alpha=0.5,
    data=None, time=None, state='us', gacc_region=None, colorbar_pad=0.02
):
    """
    Plot RTMA relative humidity data with optional METAR overlays.

    Args:
        (Same as original docstring, generalized to remove specific branding)
    Returns:
        str: Path to saved plot or None if failed.
    """
    mapcrs = ccrs.PlateCarree()
    datacrs = ccrs.PlateCarree()
    contourf = np.arange(0, 101, 1)
    labels = contourf[::4]

    if state.lower() in ['us', 'usa']:
        contours = [0, 10, 20, 30, 40, 60, 80, 100]
        linewidths = 1
    else:
        contours = np.arange(0, 110, 10)
        linewidths = 0.5

    # Configure reference system
    show_state_borders, show_county_borders, show_gacc_borders, show_psa_borders, \
    show_cwa_borders, show_nws_firewx_zones, show_nws_public_zones, \
    county_border_linewidth, psa_border_linewidth, nws_public_zones_linewidth, \
    nws_firewx_zones_linewidth, cwa_border_linewidth = settings.configure_reference_system(
        reference_system, state, show_state_borders, show_county_borders, show_gacc_borders,
        show_psa_borders, show_cwa_borders, show_nws_firewx_zones, show_nws_public_zones,
        county_border_linewidth, psa_border_linewidth, nws_public_zones_linewidth,
        nws_firewx_zones_linewidth, cwa_border_linewidth
    )

    # Configure plot dimensions
    western_bound, eastern_bound, southern_bound, northern_bound, fig_x_length, fig_y_length, \
    signature_x_position, signature_y_position, title_fontsize, subplot_title_fontsize, \
    signature_fontsize, sample_point_fontsize, colorbar_fontsize, color_table_shrink, \
    mapcrs, datacrs, aspect, tick = settings.configure_plot_dimensions(
        state, gacc_region, western_bound, eastern_bound, southern_bound, northern_bound,
        fig_x_length, fig_y_length, signature_x_position, signature_y_position, title_fontsize,
        subplot_title_fontsize, signature_fontsize, sample_point_fontsize, colorbar_fontsize,
        color_table_shrink, mapcrs, datacrs
    )

    local_time, utc_time = utilities.plot_creation_time()

    # Load shapefiles
    shapefiles = geometry.load_shapefiles(psa_color, gacc_color, cwa_color, fwz_color, pz_color)

    cmap = colormaps.relative_humidity_colormap()

    # Fetch data
    try:
        if data is None and time is None:
            from .data_access import NOMADS_OPENDAP_Downloads
            ds, rtma_time = NOMADS_OPENDAP_Downloads.RTMA_CONUS.get_RTMA_dataset(utc_time)
            temp = ds['tmp2m'] - 273.15
            dwpt = ds['dpt2m'] - 273.15
            lat = ds['lat']
            lon = ds['lon']
            rtma_data = thermodynamics.relative_humidity_from_temperature_and_dewpoint_celsius(temp, dwpt)
        elif data is not None and time is not None:
            ds = data
            rtma_time = time
            temp = ds['tmp2m'] - 273.15
            dwpt = ds['dpt2m'] - 273.15
            lat = ds['lat']
            lon = ds['lon']
            rtma_data = thermodynamics.relative_humidity_from_temperature_and_dewpoint_celsius(temp, dwpt)
        else:
            raise ValueError("Both data and time must be None or both provided.")
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

    # Time conversion
    rtma_time = rtma_time.replace(tzinfo=tz.tzutc()).astimezone(tz.tzlocal())
    rtma_time_utc = rtma_time.astimezone(tz.tzutc())

    # Create figure
    fig = plt.figure(figsize=(fig_x_length, fig_y_length))
    fig.set_facecolor('lightsteelblue')  # Match provided files
    ax = fig.add_subplot(1, 1, 1, projection=mapcrs)
    ax.set_extent((western_bound, eastern_bound, southern_bound, northern_bound), crs=ccrs.PlateCarree())

    # Add map features
    geometry.add_map_features(ax, show_rivers, show_state_borders, show_county_borders,
                             show_gacc_borders, show_psa_borders, show_cwa_borders,
                             show_nws_firewx_zones, show_nws_public_zones, shapefiles,
                             state_border_linewidth, county_border_linewidth, gacc_border_linewidth,
                             psa_border_linewidth, cwa_border_linewidth, nws_firewx_zones_linewidth,
                             nws_public_zones_linewidth, state_border_linestyle, county_border_linestyle,
                             gacc_border_linestyle, psa_border_linestyle, cwa_border_linestyle,
                             nws_firewx_zones_linestyle, nws_public_zones_linestyle)

    # Plot data
    cs = ax.contourf(lon, lat, rtma_data[0, :, :], transform=datacrs, levels=contourf, cmap=cmap, alpha=alpha)
    cbar = fig.colorbar(cs, shrink=color_table_shrink, pad=colorbar_pad, location='bottom', aspect=30, ticks=labels)
    cbar.set_label(label="Relative Humidity (%)", size=colorbar_fontsize, fontweight='bold')

    # Plot sample points
    if show_sample_points:
        decimate = settings.get_nomads_decimation(western_bound, eastern_bound, southern_bound, northern_bound)
        plot_lon, plot_lat = np.meshgrid(lon[::decimate], lat[::decimate])
        stn = StationPlot(ax, plot_lon.flatten(), plot_lat.flatten(), transform=datacrs, fontsize=sample_point_fontsize, zorder=7, clip_on=True)
        stn.plot_parameter('C', rtma_data[0, ::decimate, ::decimate].to_numpy().flatten(), color='blue', path_effects=[withStroke(linewidth=1, foreground='black')])

    # Add titles
    plt.title("RTMA Relative Humidity (%)", fontsize=title_fontsize, fontweight='bold', loc='left')
    plt.title(f"Analysis Valid: {rtma_time.strftime('%m/%d/%Y %H:00 Local')} ({rtma_time_utc.strftime('%H:00 UTC')})",
              fontsize=subplot_title_fontsize, fontweight='bold', loc='right')
    ax.text(signature_x_position, signature_y_position,
            f"Reference System: {reference_system}\nData Source: Public Weather Data\nImage Created: {local_time.strftime('%m/%d/%Y %H:%M Local')} ({utc_time.strftime('%H:%M UTC')})",
            transform=ax.transAxes, fontsize=signature_fontsize, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1), zorder=10)

    # Save plot
    path = utilities.check_file_paths(state, gacc_region, 'RTMA_RH', reference_system)
    plt.savefig(path, format='png', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"RTMA Relative Humidity plot saved to {path}")
    return path

# Placeholder for other plotting functions
async def plot_24_hour_relative_humidity_comparison(state, data, data_24, time, time_24):
    """Generate 24-hour RTMA relative humidity comparison plot."""
    # Implement similar to plot_relative_humidity, using data and data_24
    pass

async def plot_temperature(state, data, time):
    """Generate RTMA surface temperature plot."""
    pass

async def plot_dry_and_gusty_areas(state, data, time):
    """Generate RTMA dry and gusty areas plot."""
    pass

async def plot_relative_humidity_with_metar_obs(state, data):
    """Generate RTMA relative humidity with METAR observations plot."""
    pass

async def plot_low_relative_humidity_with_metar_obs(state, data):
    """Generate RTMA low relative humidity with METAR observations plot."""
    pass
