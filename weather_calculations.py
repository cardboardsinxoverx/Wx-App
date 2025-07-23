# weather_calculations.py
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

def calc_mslp(surface_pressure, elevation, temperature):
    """
    Calculate Mean Sea Level Pressure (MSLP) from surface pressure, elevation, and temperature.

    Args:
        surface_pressure (array-like): Surface pressure in Pa
        elevation (array-like): Elevation in meters
        temperature (array-like): Temperature in Celsius or Kelvin

    Returns:
        array-like: MSLP in hPa
    """
    try:
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

        return mslp
    except Exception as e:
        raise ValueError(f"Error calculating MSLP: {e}")
