# weather_calculations.py
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def calc_mslp(surface_pressure, elevation, temperature):
    """
    Calculate Mean Sea Level Pressure (MSLP) from surface pressure, elevation, and temperature.

    Args:
        surface_pressure (array-like): Surface pressure in Pa
        elevation (array-like): Elevation in meters
        temperature (array-like): Temperature in Celsius or Kelvin

    Returns:
        array-like: MSLP in hPa

    Raises:
        ValueError: If inputs are invalid or calculation fails
    """
    try:
        # Validate inputs
        if any(x is None for x in [surface_pressure, elevation, temperature]):
            raise ValueError("Input arrays cannot be None")

        # Convert inputs to appropriate units
        surface_pressure = np.array(surface_pressure) * units.Pa
        elevation = np.array(elevation) * units.meter
        temp_kelvin = (np.array(temperature) * units.degC).to('kelvin') if np.max(temperature) < 100 else np.array(temperature) * units.kelvin

        # Constants
        g = 9.80665  # Gravity (m/s^2)
        Rd = 287.05  # Gas constant for dry air (J/(kg·K))

        # Calculate MSLP
        mslp = surface_pressure * np.exp((g * elevation) / (Rd * temp_kelvin))
        mslp = mslp / 100  # Convert to hPa
        mslp = np.where((mslp >= 850) & (mslp <= 1100), mslp, np.nan)  # Filter unrealistic values

        logger.info("MSLP calculation completed successfully")
        return mslp
    except Exception as e:
        logger.error(f"Error calculating MSLP: {e}")
        raise ValueError(f"Error calculating MSLP: {e}")
