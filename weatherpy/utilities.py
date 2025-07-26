import os
import logging
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

# Output directory constant
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_creation_time():
    """Return local and UTC time for plot creation."""
    local_time = datetime.now(pytz.tzlocal())
    utc_time = local_time.astimezone(pytz.UTC)
    return local_time, utc_time

def check_file_paths(state, gacc_region, plot_type, reference_system):
    """Return file path for saving plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"rtma_{plot_type.lower()}_{state or gacc_region}_{reference_system.replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%MZ')}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    logger.debug(f"Generated file path: {path}")
    return path
