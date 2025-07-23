# utils.py
from datetime import datetime
import pytz

def parse_date(date_str):
    """
    Parse a date string into a datetime object in UTC.

    Args:
        date_str (str): Date string in formats like 'YYYY-MM-DD', 'YYYYMMDD', 'MM/DD/YYYY HH:MM'

    Returns:
        datetime: UTC datetime object
    """
    try:
        # Try common date formats
        for fmt in (
            "%Y-%m-%d", "%Y%m%d", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M", "%Y%m%d%H%M"
        ):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(tzinfo=pytz.UTC)
            except ValueError:
                continue
        raise ValueError(f"Invalid date format: {date_str}")
    except Exception as e:
        raise ValueError(f"Error parsing date {date_str}: {e}")
