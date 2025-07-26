import cartopy.crs as ccrs
import numpy as np
import logging

logger = logging.getLogger(__name__)

def configure_reference_system(reference_system, state, show_state_borders, show_county_borders,
                              show_gacc_borders, show_psa_borders, show_cwa_borders,
                              show_nws_firewx_zones, show_nws_public_zones, county_border_linewidth,
                              psa_border_linewidth, nws_public_zones_linewidth, nws_firewx_zones_linewidth,
                              cwa_border_linewidth):
    """Configure map reference system settings."""
    if reference_system.lower() != 'custom':
        show_state_borders = False
        show_county_borders = False
        show_gacc_borders = False
        show_psa_borders = False
        show_cwa_borders = False
        show_nws_firewx_zones = False
        show_nws_public_zones = False
        if reference_system == 'States Only':
            show_state_borders = True
        elif reference_system == 'States & Counties':
            show_state_borders = True
            show_county_borders = True
            if state.lower() in ['us', 'usa']:
                county_border_linewidth = 0.25
        elif reference_system == 'GACC Only':
            show_gacc_borders = True
        elif reference_system == 'GACC & PSA':
            show_gacc_borders = True
            show_psa_borders = True
            if state.lower() in ['us', 'usa']:
                psa_border_linewidth = 0.25
        elif reference_system == 'CWA Only':
            show_cwa_borders = True
        elif reference_system == 'NWS CWAs & NWS Public Zones':
            show_cwa_borders = True
            show_nws_public_zones = True
            if state.lower() in ['us', 'usa']:
                nws_public_zones_linewidth = 0.25
        elif reference_system == 'NWS CWAs & NWS Fire Weather Zones':
            show_cwa_borders = True
            show_nws_firewx_zones = True
            if state.lower() in ['us', 'usa']:
                nws_firewx_zones_linewidth = 0.25
        elif reference_system == 'NWS CWAs & Counties':
            show_cwa_borders = True
            show_county_borders = True
            if state.lower() in ['us', 'usa']:
                county_border_linewidth = 0.25
        elif reference_system == 'GACC & PSA & NWS Fire Weather Zones':
            show_gacc_borders = True
            show_psa_borders = True
            show_nws_firewx_zones = True
            nws_firewx_zones_linewidth = 0.25
            if state.lower() in ['us', 'usa']:
                psa_border_linewidth = 0.5
        elif reference_system == 'GACC & PSA & NWS Public Zones':
            show_gacc_borders = True
            show_psa_borders = True
            show_nws_public_zones = True
            nws_public_zones_linewidth = 0.25
            if state.lower() in ['us', 'usa']:
                psa_border_linewidth = 0.5
        elif reference_system == 'GACC & PSA & NWS CWA':
            show_gacc_borders = True
            show_psa_borders = True
            show_cwa_borders = True
            cwa_border_linewidth = 0.25
            if state.lower() in ['us', 'usa']:
                psa_border_linewidth = 0.5
        elif reference_system == 'GACC & PSA & Counties':
            show_gacc_borders = True
            show_psa_borders = True
            show_county_borders = True
            county_border_linewidth = 0.25
        elif reference_system == 'GACC & Counties':
            show_gacc_borders = True
            show_county_borders = True
            if state.lower() in ['us', 'usa']:
                county_border_linewidth = 0.25
    logger.debug(f"Reference system configured: {reference_system}")
    return (show_state_borders, show_county_borders, show_gacc_borders, show_psa_borders,
            show_cwa_borders, show_nws_firewx_zones, show_nws_public_zones,
            county_border_linewidth, psa_border_linewidth, nws_public_zones_linewidth,
            nws_firewx_zones_linewidth, cwa_border_linewidth)

def configure_plot_dimensions(state, gacc_region, western_bound, eastern_bound, southern_bound,
                             northern_bound, fig_x_length, fig_y_length, signature_x_position,
                             signature_y_position, title_fontsize, subplot_title_fontsize,
                             signature_fontsize, sample_point_fontsize, colorbar_fontsize,
                             color_table_shrink, mapcrs, datacrs):
    """Configure plot dimensions and coordinates."""
    if state and not gacc_region:
        if state.lower() == 'ca':
            title_fontsize += 2
            subplot_title_fontsize += 2
        return ('output', -125, -65, 25, 50, 20, 12, 0.95, 0.05, title_fontsize,
                subplot_title_fontsize, signature_fontsize, sample_point_fontsize,
                colorbar_fontsize, color_table_shrink, 8, mapcrs, datacrs, 0.5, 30, 8)
    elif gacc_region:
        return ('output', -125, -65, 25, 50, 20, 12, 0.95, 0.05, title_fontsize,
                subplot_title_fontsize, signature_fontsize, sample_point_fontsize,
                colorbar_fontsize, color_table_shrink, 8, mapcrs, datacrs, 0.5, 30, 8)
    else:
        return ('output', western_bound, eastern_bound, southern_bound, northern_bound,
                fig_x_length or 20, fig_y_length or 12, signature_x_position, signature_y_position,
                title_fontsize, subplot_title_fontsize, signature_fontsize,
                sample_point_fontsize, colorbar_fontsize, color_table_shrink, 8,
                mapcrs, datacrs, 0.5, 30, 8)

def get_nomads_decimation(western_bound, eastern_bound, southern_bound, northern_bound):
    """Return decimation factor for sampling data points."""
    return 10

def get_metar_mask(state, gacc_region):
    """Return METAR mask configuration."""
    return None
