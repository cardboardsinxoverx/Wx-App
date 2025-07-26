import cartopy.feature as cfeature
import logging
import os
import requests
import zipfile
import io
from .utilities import OUTPUT_DIR  # Import from utilities

logger = logging.getLogger(__name__)

def download_shapefile(shapefile_name, output_dir):
    """
    Download and extract shapefile from Natural Earth if not already present.

    Args:
        shapefile_name (str): Name of the shapefile (e.g., 'ne_10m_admin_1_states_provinces').
        output_dir (str): Directory to store shapefiles.
    """
    shapefile_dir = os.path.join(output_dir, 'shapefiles')
    os.makedirs(shapefile_dir, exist_ok=True)
    shapefile_path = os.path.join(shapefile_dir, f"{shapefile_name}.shp")

    if os.path.exists(shapefile_path):
        logger.info(f"Shapefile {shapefile_name}.shp already exists in {shapefile_dir}")
        return shapefile_path

    try:
        url = f"https://naturalearth.s3.amazonaws.com/10m_cultural/{shapefile_name}.zip"
        logger.debug(f"Downloading shapefile from {url}")
        response = requests.get(url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(shapefile_dir)
        logger.info(f"Shapefile {shapefile_name} downloaded and extracted to {shapefile_dir}")
        return shapefile_path
    except Exception as e:
        logger.error(f"Failed to download shapefile {shapefile_name}: {e}")
        raise

def load_shapefiles(psa_color, gacc_color, cwa_color, fwz_color, pz_color):
    """Load shapefiles for map features."""
    shapefiles = {}
    try:
        shapefile_dir = os.path.join(OUTPUT_DIR, 'shapefiles')
        shapefiles['PSAs'] = cfeature.ShapelyFeature(
            cfeature.Reader(download_shapefile('ne_10m_admin_1_states_provinces', OUTPUT_DIR)).geometries(),
            cfeature.CRS(), edgecolor=psa_color, facecolor='none'
        )
        shapefiles['GACC'] = cfeature.ShapelyFeature(
            cfeature.Reader(download_shapefile('ne_10m_admin_0_countries', OUTPUT_DIR)).geometries(),
            cfeature.CRS(), edgecolor=gacc_color, facecolor='none'
        )
        shapefiles['CWAs'] = cfeature.ShapelyFeature(
            cfeature.Reader(download_shapefile('ne_10m_admin_1_states_provinces', OUTPUT_DIR)).geometries(),
            cfeature.CRS(), edgecolor=cwa_color, facecolor='none'
        )
        shapefiles['FWZs'] = cfeature.ShapelyFeature(
            cfeature.Reader(download_shapefile('ne_10m_admin_1_states_provinces', OUTPUT_DIR)).geometries(),
            cfeature.CRS(), edgecolor=fwz_color, facecolor='none'
        )
        shapefiles['PZs'] = cfeature.ShapelyFeature(
            cfeature.Reader(download_shapefile('ne_10m_admin_1_states_provinces', OUTPUT_DIR)).geometries(),
            cfeature.CRS(), edgecolor=pz_color, facecolor='none'
        )
        logger.debug("Shapefiles loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load shapefiles: {e}. Using default features.")
    return shapefiles

def add_map_features(ax, show_rivers, show_state_borders, show_county_borders, show_gacc_borders,
                     show_psa_borders, show_cwa_borders, show_nws_firewx_zones, show_nws_public_zones,
                     shapefiles, state_border_linewidth, county_border_linewidth, gacc_border_linewidth,
                     psa_border_linewidth, cwa_border_linewidth, nws_firewx_zones_linewidth,
                     nws_public_zones_linewidth, state_border_linestyle, county_border_linestyle,
                     gacc_border_linestyle, psa_border_linestyle, cwa_border_linestyle,
                     nws_firewx_zones_linestyle, nws_public_zones_linestyle):
    """Add map features to the plot."""
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=9)
    ax.add_feature(cfeature.LAND, facecolor='lightgreen', alpha=0.7, zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.7, zorder=3)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.7, zorder=3)
    if show_rivers:
        ax.add_feature(cfeature.RIVERS, color='lightblue', zorder=3)
    if show_state_borders:
        ax.add_feature(cfeature.STATES, linewidth=state_border_linewidth, linestyle=state_border_linestyle, edgecolor='black', zorder=6)
    if show_county_borders:
        ax.add_feature(cfeature.USCOUNTIES, linewidth=county_border_linewidth, linestyle=county_border_linestyle, zorder=5)
    if show_gacc_borders and 'GACC' in shapefiles:
        ax.add_feature(shapefiles['GACC'], linewidth=gacc_border_linewidth, linestyle=gacc_border_linestyle, zorder=6)
    if show_psa_borders and 'PSAs' in shapefiles:
        ax.add_feature(shapefiles['PSAs'], linewidth=psa_border_linewidth, linestyle=psa_border_linestyle, zorder=5)
    if show_cwa_borders and 'CWAs' in shapefiles:
        ax.add_feature(shapefiles['CWAs'], linewidth=cwa_border_linewidth, linestyle=cwa_border_linestyle, zorder=5)
    if show_nws_firewx_zones and 'FWZs' in shapefiles:
        ax.add_feature(shapefiles['FWZs'], linewidth=nws_firewx_zones_linewidth, linestyle=nws_firewx_zones_linestyle, zorder=5)
    if show_nws_public_zones and 'PZs' in shapefiles:
        ax.add_feature(shapefiles['PZs'], linewidth=nws_public_zones_linewidth, linestyle=nws_public_zones_linestyle, zorder=5)
