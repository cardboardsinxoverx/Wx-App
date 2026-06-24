import os
import logging
import aiohttp
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime, timedelta
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
import textwrap  # Ensure full module for wrap function options
from matplotlib import patheffects
import xml.etree.ElementTree as ET
import numpy as np
from matplotlib.legend_handler import HandlerBase

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Icon directory
icon_dir = '/media/desoxyn/Main/wx/bot/icons'

# Define geostationary projection for GOES-18 (Eastern Pacific)
proj = ccrs.Geostationary(central_longitude=-137.0, satellite_height=35786023.0)
sat_height = 35786023.0
half_angle = 0.151872
x_max = sat_height * math.tan(half_angle)
y_max = x_max
y_min = -x_max
x_min = -x_max

# Define the map extent for filtering (Eastern Pacific)
EXTENT_LON_MIN = -140
EXTENT_LON_MAX = -90
EXTENT_LAT_MIN = 0
EXTENT_LAT_MAX = 30

class ImageHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Read the image from the path provided in the handler map
        img_path = orig_handle.get_label()
        img = mpimg.imread(img_path)

        # Create an OffsetImage
        imagebox = OffsetImage(img, zoom=0.0267) # Reduced to 1/3 of original (0.08 / 3)

        # Create an AnnotationBbox to place the image, centered in the legend box
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        ab = AnnotationBbox(imagebox, center,
                            xycoords=trans,
                            box_alignment=(0.5, 0.5),
                            frameon=False,
                            pad=0)

        # We return a list of the artists to be drawn
        return [ab]

async def fetch_satellite_image(session):
    """
    Fetch the latest GOES-18 ABI FD GEOCOLOR image for Eastern Pacific basin.
    Prioritizes direct URL construction based on current UTC time.
    Falls back to directory listing if needed.
    Returns PIL Image or None.
    """
    current_utc = datetime.utcnow()
    # GOES-18 GEOCOLOR cadence: every 15 minutes (00, 15, 30, 45 past the hour)
    # Round down to the last full 15-min interval
    minutes = current_utc.minute
    rounded_min = (minutes // 15) * 15
    fetch_time = current_utc.replace(minute=rounded_min, second=0, microsecond=0)

    # Try up to 3 previous intervals if the exact one fails (e.g., processing lag)
    for delta in range(0, 45, 15):  # 0, 15, 30 min back
        try_time = fetch_time - timedelta(minutes=delta)
        year = try_time.strftime('%Y')
        julian_day = try_time.strftime('%j')  # 001-366
        hour = try_time.strftime('%H')
        minute = try_time.strftime('%M').zfill(2)
        timestamp = f"{year}{julian_day}{hour}{minute}"

        # Construct direct URL (NOAA STAR - basin-wide)
        url = f"https://cdn.star.nesdis.noaa.gov/GOES18/ABI/FD/GEOCOLOR/{timestamp}_GOES18-ABI-FD-GEOCOLOR-1808x1808.jpg"

        logging.info(f"Trying direct fetch: {url}")
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    if response.headers.get('Content-Type', '').startswith('image/'):
                        img = PILImage.open(BytesIO(content)).convert('RGB')
                        logging.info(f"Successfully fetched satellite image via direct URL: {url}")
                        return img
                    else:
                        logging.warning(f"Non-image content from {url}: {response.headers.get('Content-Type')}")
                else:
                    logging.warning(f"Direct fetch failed for {url} (status: {response.status})")
        except Exception as e:
            logging.warning(f"Error in direct fetch for {url}: {e}")

    # Fallback: Directory listing on NOAA STAR
    base_url = "https://cdn.star.nesdis.noaa.gov/GOES18/ABI/FD/GEOCOLOR/"
    pattern = r'(\d{11}_GOES18-ABI-FD-GEOCOLOR-1808x1808\.jpg)'
    try:
        async with session.get(base_url) as response:
            if response.status == 200:
                html_content = await response.text()
                available_files = re.findall(pattern, html_content)
                if available_files:
                    # Parse timestamps and pick latest valid
                    available_times = []
                    for filename in available_files:
                        try:
                            timestamp_str = filename.split('_')[0]
                            ts = datetime.strptime(timestamp_str, '%Y%j%H%M')
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
                                logging.info(f"Successfully fetched via directory fallback: {fallback_url}")
                                return img
    except Exception as e:
        logging.error(f"Directory fallback failed: {e}")

    # Final fallback: RAMMB basin-wide geocolor for EPAC
    rammb_url = "https://rammb-data.cira.colostate.edu/tc_realtime/products/general/epac/geocolor/latest.jpg"
    try:
        async with session.get(rammb_url) as response:
            if response.status == 200:
                content = await response.read()
                if response.headers.get('Content-Type', '').startswith('image/'):
                    img = PILImage.open(BytesIO(content)).convert('RGB')
                    logging.info(f"Successfully fetched RAMMB fallback: {rammb_url}")
                    return img
    except Exception as e:
        logging.error(f"RAMMB fallback failed: {e}")

    logging.warning("All satellite fetch attempts failed; using no background image.")
    return None

async def fetch_active_tcs(session, basin):
    """
    Fetch active tropical cyclones from NHC's JSON status feed.
    Filters for specified basin (epac or atl) and returns current positions/types.
    """
    if basin == 'epac':
        json_url = 'https://www.nhc.noaa.gov/CurrentStorms.json'
    else:
        json_url = 'https://www.nhc.noaa.gov/CurrentStorms.json'  # Same JSON covers both; filter below

    tcs = []
    try:
        async with session.get(json_url) as resp:
            if resp.status != 200:
                logging.warning(f"Failed to fetch JSON: {resp.status}")
                return tcs
            json_data = await resp.json()
            logging.info(f"JSON activeStorms length: {len(json_data.get('activeStorms', []))}")

        active_storms = json_data.get('activeStorms', [])
        for storm in active_storms:
            bin_num = storm.get('binNumber', '')
            if basin == 'epac' and not bin_num.startswith('EP'):
                continue
            if basin != 'epac' and not bin_num.startswith('AL'):
                continue

            name = storm.get('name', '') or ''
            classification = storm.get('classification', '')
            maxwind = int(storm.get('intensity', 0) or 0)  # knots

            # Determine type from classification (primary), fallback to maxwind
            if classification == 'TD':
                tc_type = 'tropical_depression'
            elif classification == 'TS':
                tc_type = 'tropical_storm'
            elif classification in ['HU', 'TY']:  # TY for typhoon equiv, but rare in EPAC
                tc_type = 'hurricane'
            else:
                # Fallback to maxwind
                if maxwind < 34:
                    tc_type = 'tropical_depression'
                elif maxwind < 64:
                    tc_type = 'tropical_storm'
                else:
                    tc_type = 'hurricane'

            lat_str = storm.get('latitude', '')
            lon_str = storm.get('longitude', '')
            # Parse lat/lon (e.g., "14.2N", "123.2W" -> 14.2, -123.2)
            try:
                # Better parsing for lat/lon strings like "14.2°N" or "123.2°W"
                import re
                lat_match = re.search(r'(\d+\.?\d*)[°N S]', lat_str.upper())
                lon_match = re.search(r'(\d+\.?\d*)[°W E]', lon_str.upper())
                if lat_match:
                    lat = float(lat_match.group(1))
                    if 'S' in lat_str.upper():
                        lat = -lat
                else:
                    lat = 0
                if lon_match:
                    lon = float(lon_match.group(1))
                    if 'W' in lon_str.upper():
                        lon = -lon
                else:
                    lon = 0
            except ValueError:
                logging.warning(f"Invalid lat/lon for {name}: {lat_str}, {lon_str}")
                continue

            tcs.append({
                'lon': lon,
                'lat': lat,
                'name': name,
                'type': tc_type,
                'status': classification,
                'maxwind': maxwind
            })
            logging.info(f"Fetched active TC: {name} ({tc_type}) at ({lon}, {lat}) with {maxwind} kt")

    except Exception as e:
        logging.error(f"Error fetching active TCs from JSON: {e}")

    logging.info(f"Total active TCs fetched: {len(tcs)}")
    return tcs

async def fetch_shapefile_data(session):
    """
    Fetch and parse the official NHC shapefile for graphical tropical weather outlook.
    Filters for Eastern Pacific basin based on location.
    """
    url = "https://www.nhc.noaa.gov/xgtwo/gtwo_shapefiles.zip"
    disturbances = []
    zero_percent_points = []
    try:
        async with session.get(url) as response:
            if response.status != 200:
                logging.error(f"Failed to fetch shapefile data. HTTP Status: {response.status}")
                return [], []
            zip_data = await response.read()

        with zipfile.ZipFile(BytesIO(zip_data)) as zf:
            namelist = zf.namelist()
            logging.info(f"Files in shapefile zip: {namelist}")

            # Find areas shapefile (assuming combined; filter by location later)
            area_files = [f for f in namelist if f.startswith('gtwo_areas_') and f.endswith('.shp')]
            if not area_files:
                logging.error("No areas shapefile found.")
                return [], []

            area_base = area_files[0].rsplit('.', 1)[0]  # Remove .shp; take first if multiple
            shp_data = zf.read(area_base + '.shp')
            dbf_data = zf.read(area_base + '.dbf')
            shx_data = zf.read(area_base + '.shx')
            prj_data = zf.read(area_base + '.prj') if area_base + '.prj' in namelist else None

            # Read shapefile
            sf = shapefile.Reader(shp=BytesIO(shp_data), dbf=BytesIO(dbf_data), shx=BytesIO(shx_data))
            if prj_data:
                sf = shapefile.Reader(shp=BytesIO(shp_data), dbf=BytesIO(dbf_data), shx=BytesIO(shx_data), prj=BytesIO(prj_data))

            field_names = [field[0] for field in sf.fields[1:]]  # Skip deletion field
            prob7_index = field_names.index('PROB7DAY') if 'PROB7DAY' in field_names else None

            for i in range(len(sf.shapes())):
                shape = sf.shapes()[i]
                record = sf.records()[i]

                if prob7_index is None:
                    continue

                chance_str = record[prob7_index]
                if isinstance(chance_str, str) and '%' in chance_str:
                    chance = int(chance_str.replace('%', ''))
                else:
                    chance = int(chance_str)

                if chance == 0:
                    continue

                # Extract polygon coordinates (assume simple polygon, outer ring)
                coords = [(point[0], point[1]) for point in shape.points]

                # Compute center to filter for EPAC extent
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                center_lon = sum(lons) / len(lons)
                center_lat = sum(lats) / len(lats)

                # Filter for Eastern Pacific
                if EXTENT_LON_MIN <= center_lon <= EXTENT_LON_MAX and EXTENT_LAT_MIN <= center_lat <= EXTENT_LAT_MAX:
                    disturbances.append({
                        'polygon': coords,
                        'formation_chance': chance,
                        'x_marker': None,
                    })
                    logging.info(f"Found EPAC disturbance polygon with {chance}% chance at center ({center_lon}, {center_lat})")
                else:
                    logging.info(f"Skipped non-EPAC disturbance with {chance}% at ({center_lon}, {center_lat})")

            # Find points shapefile for X markers
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
                    if isinstance(p_chance_str, str) and '%' in p_chance_str:
                        p_chance = int(p_chance_str.replace('%', ''))
                    else:
                        p_chance = int(p_chance_str)

                    x_lon, x_lat = p_shape.points[0]  # Single point

                    # Filter for EPAC extent
                    if EXTENT_LON_MIN <= x_lon <= EXTENT_LON_MAX and EXTENT_LAT_MIN <= x_lat <= EXTENT_LAT_MAX:
                        if p_chance > 0:
                            # Match to disturbance by chance only (remove bbox check to allow external positions)
                            matched = False
                            for dist in disturbances:
                                if dist['formation_chance'] == p_chance:
                                    dist['x_marker'] = (x_lon, x_lat)
                                    logging.info(f"Associated X marker ({x_lon}, {x_lat}) with {p_chance}% EPAC disturbance")
                                    matched = True
                                    break  # Assign to first matching disturbance
                            if not matched:
                                logging.warning(f"Could not match {p_chance}% point ({x_lon}, {x_lat}) to a EPAC disturbance polygon")
                        else:
                            # 0% point: collect separately
                            zero_percent_points.append({
                                'lon': x_lon,
                                'lat': x_lat,
                                'chance': p_chance
                            })
                            logging.info(f"Found 0% X marker at EPAC ({x_lon}, {x_lat})")
                    else:
                        logging.info(f"Skipped non-EPAC point ({x_lon}, {x_lat}) with {p_chance}%")

        logging.info(f"Parsed {len(disturbances)} EPAC disturbances and {len(zero_percent_points)} 0% points from shapefile.")
        return disturbances, zero_percent_points

    except Exception as e:
        logging.error(f"An unexpected error occurred in fetch_shapefile_data: {e}")
        return [], []

def get_color_from_chance(chance):
    """
    Return color based on formation chance.
    <40%: yellow
    40-60%: orange
    >60%: red
    """
    if chance < 40:
        return 'yellow'
    elif chance <= 60:
        return 'orange'
    else:
        return 'red'

def get_tc_color(tc_type):
    """
    Return color for active TC symbol based on type.
    """
    if tc_type == 'tropical_depression':
        return 'blue'
    elif tc_type == 'tropical_storm':
        return 'green'
    else:  # hurricane
        return 'red'

def is_within_extent(lon, lat):
    """
    Check if a point is within the map extent.
    """
    return (EXTENT_LON_MIN <= lon <= EXTENT_LON_MAX and
            EXTENT_LAT_MIN <= lat <= EXTENT_LAT_MAX)

async def seven_day_outlook_EPAC(output_path):
    """
    Generates a Seven-Day Graphical Tropical Weather Outlook similar to NHC product for Eastern Pacific.
    Saves the image to the provided output_path.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logging.info(f"Output directory ensured: {os.path.dirname(output_path)}")
    except Exception as e:
        logging.error(f"Error creating output directory: {e}")
        raise e

    async with aiohttp.ClientSession() as session:
        try:
            logging.info("Fetching latest NHC outlook data...")
            disturbances, zero_percent_points = await fetch_shapefile_data(session)
            active_tcs = await fetch_active_tcs(session, basin='epac')

            # Fetch satellite image using robust method
            img = await fetch_satellite_image(session)

            # Create figure with lightsteelblue background
            fig = plt.figure(figsize=(16, 12), facecolor='lightsteelblue')
            # Use Geostationary projection for curved satellite background
            ax = fig.add_subplot(1, 1, 1, projection=proj)
            ax.set_facecolor('lightsteelblue')

            # Set extent to Eastern Pacific basin
            ax.set_extent([EXTENT_LON_MIN, EXTENT_LON_MAX, EXTENT_LAT_MIN, EXTENT_LAT_MAX], crs=ccrs.PlateCarree())

            # Add satellite background if available (overlay on top of lightsteelblue with alpha)
            if img is not None:
                ax.imshow(img, origin='upper', extent=(x_min, x_max, y_min, y_max), transform=proj, zorder=1, interpolation='bilinear', alpha=0.7)
                logging.info("Satellite imagery overlaid successfully.")
            else:
                logging.warning("No satellite imagery available; using plain map background.")

            # Add map features - only outlines, no fill to preserve satellite imagery
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
            ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=2)
            ax.add_feature(cfeature.LAKES, edgecolor='blue', facecolor='none', linewidth=0.5, zorder=2)
            gl = ax.gridlines(draw_labels=True, alpha=0.5, zorder=3)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}

            # Plot active tropical cyclones icons or fallback symbols
            for tc in active_tcs:
                logging.info(f"Processing TC {tc['name'] or 'Unnamed'} at ({tc['lon']}, {tc['lat']}), within extent: {is_within_extent(tc['lon'], tc['lat'])}")
                if is_within_extent(tc['lon'], tc['lat']):
                    icon_path = os.path.join(icon_dir, f"{tc['type']}.png")
                    logging.info(f"Icon path: {icon_path}, exists: {os.path.exists(icon_path)}")
                    plotted_icon = False
                    if os.path.exists(icon_path):
                        try:
                            arr = mpimg.imread(icon_path)
                            zoom = 0.05  # Reduced to half of original (0.1 / 2)

                            # Transform lon/lat to projected coordinates for AnnotationBbox
                            transform_result = proj.transform_points(ccrs.PlateCarree(), np.array([tc['lon']]), np.array([tc['lat']]))
                            projected_xy = transform_result[0, :2]

                            if np.any(np.isnan(projected_xy)):
                                raise ValueError(f"Invalid projected coordinates (likely outside satellite disk view): {projected_xy}")

                            imagebox = OffsetImage(arr, zoom=zoom)
                            ab = AnnotationBbox(imagebox, projected_xy,
                                                xycoords='data',  # Use 'data' since xy is now in projected coords
                                                frameon=False,
                                                box_alignment=(0.5, 0.5),
                                                pad=0,
                                                zorder=10)
                            ax.add_artist(ab)
                            logging.info(f"Successfully plotted {tc['type']} icon for {tc['name'] or 'Unnamed'} at ({tc['lon']}, {tc['lat']}) with projected xy: {projected_xy}")
                            plotted_icon = True
                        except Exception as e:
                            logging.warning(f"Error plotting icon for {tc['name'] or 'Unnamed'}: {e}")
                    if not plotted_icon:
                        # Fallback to circle symbol - always plot this if no icon succeeded
                        color = get_tc_color(tc['type'])
                        ax.scatter(tc['lon'], tc['lat'], marker='o', s=300, facecolor=color, edgecolor='black', linewidth=2,
                                   transform=ccrs.PlateCarree(), zorder=10)
                        logging.info(f"Plotted fallback {tc['type']} symbol for {tc['name'] or 'Unnamed'} at ({tc['lon']}, {tc['lat']})")

                    # Add name label below the icon/symbol
                    name = tc['name'] if tc['name'] else tc['type'].replace('_', ' ').title()
                    ax.text(tc['lon'], tc['lat'] - 0.6, name,
                            transform=ccrs.PlateCarree(),
                            ha='center', va='top',
                            fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                            zorder=10)
                else:
                    logging.warning(f"TC {tc['name'] or 'Unnamed'} outside extent, skipping plot")

            num_disturbances = len(disturbances)
            if num_disturbances > 0:
                # Plot disturbances (only Eastern Pacific invests and storms are fetched by default)
                for dist in disturbances:
                    color = get_color_from_chance(dist['formation_chance'])

                    # Always add the precise polygon for the disturbance area (will be clipped if outside)
                    polygon_shape = Polygon(dist['polygon'],
                                      facecolor=color, alpha=0.4, hatch='////',
                                      edgecolor='black', linewidth=0.5,
                                      transform=ccrs.PlateCarree(), zorder=4)
                    ax.add_patch(polygon_shape)

                    # Compute center of polygon
                    lons = [c[0] for c in dist['polygon']]
                    lats = [c[1] for c in dist['polygon']]
                    center_lon = sum(lons) / len(lons)
                    center_lat = sum(lats) / len(lats)

                    logging.info(f"Plotted disturbance polygon with {dist['formation_chance']}% at center ({center_lon}, {center_lat})")

                    # Determine position for X and label
                    if dist.get('x_marker'):
                        marker_lon, marker_lat = dist['x_marker']
                        logging.info(f"Using x_marker ({marker_lon}, {marker_lat}) for {dist['formation_chance']}%")
                    else:
                        # Fallback to center (should not happen with improved matching)
                        marker_lon = center_lon
                        marker_lat = center_lat
                        logging.info(f"No x_marker, using center ({marker_lon}, {marker_lat}) for {dist['formation_chance']}%")

                    # Only add 'X' marker, line, and percentage label if marker within extent
                    if is_within_extent(marker_lon, marker_lat):
                        # Add curved arrow line from X to center of shaded area only if marker differs from center
                        if dist.get('x_marker'):
                            arrow = FancyArrowPatch((marker_lon, marker_lat), (center_lon, center_lat),
                                                    connectionstyle="arc3,rad=0.3",  # Adjust rad for curvature (positive for one direction, negative for other)
                                                    arrowstyle="->,head_width=4.0,head_length=4.0", color=color, linewidth=8,
                                                    transform=ccrs.PlateCarree(), zorder=4.5)
                            ax.add_patch(arrow)
                            logging.info(f"Plotted arrow from marker to center for {dist['formation_chance']}%")

                        # Add 'X' marker
                        ax.text(marker_lon, marker_lat, 'X', transform=ccrs.PlateCarree(), fontsize=18,
                                fontfamily='sans-serif', fontweight='bold', ha='center', va='center', color=color,
                                zorder=5, path_effects=[patheffects.Stroke(linewidth=2, foreground='black'),
                                                        patheffects.Normal()])
                        logging.info(f"Plotted X marker at ({marker_lon}, {marker_lat}) for {dist['formation_chance']}%")

                        # Always add percentage label next to the position (for >0%)
                        ax.text(marker_lon + 1, marker_lat, f"{dist['formation_chance']}%",
                                transform=ccrs.PlateCarree(), fontsize=14, ha='left', va='center',
                                fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9), zorder=6)
                        logging.info(f"Plotted {dist['formation_chance']}% label at ({marker_lon + 1}, {marker_lat})")
                    else:
                        logging.warning(f"Marker position ({marker_lon}, {marker_lat}) outside extent for {dist['formation_chance']}% disturbance")

            # Plot 0% X markers (yellow, with 0% label, no polygon) only if within extent
            for point in zero_percent_points:
                if is_within_extent(point['lon'], point['lat']):
                    color = get_color_from_chance(point['chance'])  # Will be yellow for 0%
                    # Add 'X' marker
                    ax.text(point['lon'], point['lat'], 'X', transform=ccrs.PlateCarree(), fontsize=18,
                            fontfamily='sans-serif', fontweight='bold', ha='center', va='center', color=color,
                            zorder=5, path_effects=[patheffects.Stroke(linewidth=2, foreground='black'),
                                                    patheffects.Normal()])
                    # Add 0% label next to the position
                    ax.text(point['lon'] + 1, point['lat'], f"{point['chance']}%",
                            transform=ccrs.PlateCarree(), fontsize=14, ha='left', va='center',
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9), zorder=6)
                    logging.info(f"Plotted 0% X marker at ({point['lon']}, {point['lat']})")
                else:
                    logging.info(f"Skipped 0% X marker outside extent at ({point['lon']}, {point['lat']})")

            if num_disturbances == 0 and len(zero_percent_points) == 0 and len(active_tcs) == 0:
                # Add a note when no disturbances (centered)
                ax.text(0.5, 0.5, 'Tropical cyclone formation is not expected during the next 7 days.',
                        transform=ax.transAxes, fontsize=16, ha='center', va='center',
                        fontweight='bold', color='darkblue', zorder=10)

            # Title with adjusted padding
            now = datetime.now()
            time_str = now.strftime("%I:%M %p EDT %a %b %d %Y")
            ax.set_title(f'Seven-Day Graphical Tropical Weather Outlook\n'
                         f'National Hurricane Center Miami, FL\nValid: {time_str}',
                         fontsize=14, loc='left', pad=30)

            # --- UPDATED LEGEND LOGIC ---

            # Create placeholder plot objects for our icons.
            # The 'label' will temporarily hold the file path for the handler.
            td_icon_path = os.path.join(icon_dir, 'tropical_depression.png')
            ts_icon_path = os.path.join(icon_dir, 'tropical_storm.png')
            hu_icon_path = os.path.join(icon_dir, 'hurricane.png')

            # We use a dummy plot object (Line2D with nothing visible)
            # The handler_map will use this object as a key.
            td_handle = plt.Line2D([0], [0], marker='none', linestyle='none', label=td_icon_path)
            ts_handle = plt.Line2D([0], [0], marker='none', linestyle='none', label=ts_icon_path)
            hu_handle = plt.Line2D([0], [0], marker='none', linestyle='none', label=hu_icon_path)

            # Define the text labels that will actually appear in the legend
            legend_labels = {
                td_handle: 'Tropical Depression',
                ts_handle: 'Tropical Storm',
                hu_handle: 'Hurricane'
            }

            # Define the handler map to connect our placeholder objects to the ImageHandler
            handler_map = {
                td_handle: ImageHandler(),
                ts_handle: ImageHandler(),
                hu_handle: ImageHandler()
            }

            # Build the list of elements for the legend
            legend_elements = [
                plt.Line2D([0], [0], marker='x', color='none', markeredgecolor='yellow', markerfacecolor='black',
                           markersize=12, label='Formation chance <40%'),
                plt.Line2D([0], [0], marker='x', color='none', markeredgecolor='orange', markerfacecolor='black',
                           markersize=12, label='Formation chance 40-60%'),
                plt.Line2D([0], [0], marker='x', color='none', markeredgecolor='red', markerfacecolor='black',
                           markersize=12, label='Formation chance >60%'),
                td_handle,
                ts_handle,
                hu_handle
            ]

            # Create the legend, passing in the handler_map and the new labels
            ax.legend(handles=legend_elements,
                      labels=[h.get_label() if h not in legend_labels else legend_labels[h] for h in legend_elements],
                      handler_map=handler_map,
                      loc='lower left',
                      fontsize=10,
                      facecolor='white',
                      framealpha=0.8)

            # Add icons in top right corner, adjusted y-position higher for better alignment with title
            photo_path = '/media/desoxyn/Main/wx/bot/photo.jpg'
            logo_path = '/media/desoxyn/Main/wx/bot/boxlogo2.png'

            if os.path.exists(photo_path):
                photo_img = mpimg.imread(photo_path)
                photo_ax = fig.add_axes([0.78, 0.88, 0.08, 0.08], zorder=12)
                photo_ax.imshow(photo_img, aspect='equal', interpolation='bilinear')
                photo_ax.axis('off')
            else:
                logging.warning(f"Photo icon not found: {photo_path}")

            if os.path.exists(logo_path):
                logo_img = mpimg.imread(logo_path)
                logo_ax = fig.add_axes([0.87, 0.88, 0.08, 0.08], zorder=12)
                logo_ax.imshow(logo_img, aspect='equal', interpolation='bilinear')
                logo_ax.axis('off')
            else:
                logging.warning(f"Logo icon not found: {logo_path}")

            # Save image
            plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='lightsteelblue')
            plt.close()

            num_active = len(active_tcs)
            num_disturbances = len(disturbances)
            summary_text = f"Generated from official NHC data. {num_disturbances} potential disturbances and {num_active} active systems tracked." if num_disturbances > 0 or num_active > 0 else "Generated from official NHC data. No tropical cyclone formation expected in the Eastern Pacific basin over the next 7 days."
            logging.info(f"Seven-Day Outlook map saved to {output_path}. Summary: {summary_text}")

        except Exception as e:
            logging.error(f"Error processing seven-day outlook: {e}")
            raise e
