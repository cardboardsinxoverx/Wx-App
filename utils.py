# utils.py
from datetime import datetime
import pytz
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def parse_date(date_str):
    """
    Parse a date string into a datetime object in UTC.

    Args:
        date_str (str): Date string in formats like 'YYYY-MM-DD', 'YYYYMMDD', 'MM/DD/YYYY HH:MM'

    Returns:
        datetime: UTC datetime object

    Raises:
        ValueError: If the date string format is invalid
    """
    try:
        # Supported date formats
        formats = (
            "%Y-%m-%d", "%Y%m%d", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M", "%Y%m%d%H%M"
        )
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(tzinfo=pytz.UTC)
            except ValueError:
                continue
        logger.error(f"Invalid date format: {date_str}")
        raise ValueError(f"Invalid date format: {date_str}")
    except Exception as e:
        logger.error(f"Error parsing date {date_str}: {e}")
        raise ValueError(f"Error parsing date {date_str}: {e}")
