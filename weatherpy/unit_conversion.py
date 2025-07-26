import logging

logger = logging.getLogger(__name__)

def meters_per_second_to_mph(speed):
    """Convert wind speed from meters per second to miles per hour."""
    try:
        converted = speed * 2.23694
        logger.debug(f"Converted wind speed: {speed} m/s to {converted} mph")
        return converted
    except Exception as e:
        logger.error(f"Error converting wind speed: {e}")
        raise
