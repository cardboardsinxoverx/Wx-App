import numpy as np
import logging
from metpy.calc import relative_humidity_from_dewpoint

logger = logging.getLogger(__name__)

def relative_humidity_from_temperature_and_dewpoint_celsius(temp, dwpt):
    """
    Calculate relative humidity from temperature and dewpoint in Celsius.

    Args:
        temp: Temperature in Celsius.
        dwpt: Dewpoint in Celsius.

    Returns:
        array-like: Relative humidity in percent.
    """
    try:
        temp_c = np.array(temp) * units.degC
        dwpt_c = np.array(dwpt) * units.degC
        rh = relative_humidity_from_dewpoint(temp_c, dwpt_c) * 100
        logger.debug("Relative humidity calculated successfully")
        return rh
    except Exception as e:
        logger.error(f"Error calculating relative humidity: {e}")
        raise
