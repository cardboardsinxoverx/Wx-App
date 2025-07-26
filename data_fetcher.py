import aiohttp
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import pytz
import re

logger = logging.getLogger(__name__)

class DataFetcher:
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def fetch_observed_sounding(self, time: datetime, station_code: str) -> pd.DataFrame:
        """
        Fetch observed sounding data from University of Wyoming.
        Args:
            time (datetime): UTC time of the sounding (00Z or 12Z).
            station_code (str): Uppercase station code (e.g., 'FFC').
        Returns:
            pd.DataFrame: DataFrame with sounding data or None if failed.
        """
        try:
            # Construct URL for University of Wyoming sounding data
            base_url = "http://weather.uwyo.edu/cgi-bin/sounding"
            time_str = time.strftime("%Y%m%d%H")
            station_code = station_code.upper()
            url = f"{base_url}?region=naconf&TYPE=TEXT%3ALIST&YEAR={time.year}&MONTH={time.strftime('%m')}&FROM={time.strftime('%d%H')}&TO={time.strftime('%d%H')}&STNM={station_code}"
            logger.info(f"Fetching observed sounding for {station_code} at {time_str} from {url}")

            async with self.session.get(url, timeout=30) as response:
                response.raise_for_status()
                text = await response.text()

            # Parse the HTML response
            lines = text.splitlines()
            data_start = None
            for i, line in enumerate(lines):
                if line.strip().startswith("PRES"):
                    data_start = i
                    break
            if data_start is None:
                logger.error(f"No sounding data found for {station_code} at {time_str}")
                return None

            header = lines[data_start].strip().split()
            data_lines = [line.strip().split() for line in lines[data_start + 1:] if line.strip() and not line.startswith("</PRE>")]
            if not data_lines:
                logger.error(f"No data rows found for {station_code} at {time_str}")
                return None

            df = pd.DataFrame(data_lines, columns=header)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=['PRES', 'HGHT', 'TEMP', 'DWPT', 'DRCT', 'SKNT'])

            # Convert wind direction and speed to u, v components
            wind_speed = df['SKNT'].values * units.knots
            wind_dir = df['DRCT'].values * units.degrees
            df['u'], df['v'] = mpcalc.wind_components(wind_speed, wind_dir)
            df.rename(columns={'PRES': 'pressure', 'HGHT': 'height', 'TEMP': 'temperature', 'DWPT': 'dewpoint'}, inplace=True)

            # Add latitude and longitude from skewt.py STATIONS
            from skewt import STATIONS
            lat, lon = STATIONS.get(station_code, (np.nan, np.nan))
            df['latitude'] = lat
            df['longitude'] = lon

            logger.info(f"Successfully fetched observed sounding for {station_code} at {time_str}")
            return df
        except Exception as e:
            logger.error(f"Error fetching observed sounding for {station_code} at {time_str}: {e}")
            return None

    async def fetch_bufkit(self, station_code: str, model: str, forecast_hour: int) -> tuple:
        """
        Fetch forecast sounding data from BUFKIT source.
        Args:
            station_code (str): Uppercase station code (e.g., 'FFC').
            model (str): Model name ('gfs', 'gfsm', 'nam', 'namm', 'rap').
            forecast_hour (int): Forecast hour.
        Returns:
            tuple: (pd.DataFrame, valid_time) or (None, None) if failed.
        """
        try:
            from skewt import NWS_STATIONS, parse_bufkit_text
            nws_station = NWS_STATIONS.get(station_code.upper(), station_code.upper())
            base_url = f"https://mtarchive.geol.iastate.edu/{datetime.now(pytz.UTC).strftime('%Y/%m/%d')}/bufkit/{model}/{nws_station.lower()}_{model}.buf"
            logger.info(f"Fetching BUFKIT data for {station_code} ({nws_station}) model {model} hour {forecast_hour} from {base_url}")

            async with self.session.get(base_url, timeout=30) as response:
                response.raise_for_status()
                bufkit_text = await response.text()

            df, valid_time = parse_bufkit_text(bufkit_text, forecast_hour)
            logger.info(f"Successfully fetched BUFKIT data for {station_code} model {model} hour {forecast_hour}")
            return df, valid_time
        except Exception as e:
            logger.error(f"Error fetching BUFKIT data for {station_code} model {model} hour {forecast_hour}: {e}")
            return None, None

    async def fetch_gfs(self, time: datetime, forecast_hour: int, variables: list, bounds: dict) -> dict:
        """
        Fetch GFS data (placeholder).
        Args:
            time (datetime): Model run time.
            forecast_hour (int): Forecast hour.
            variables (list): List of variable names.
            bounds (dict): Geographic bounds {'west', 'east', 'south', 'north'}.
        Returns:
            dict: Data dictionary or empty dict if failed.
        """
        try:
            # Placeholder: Implement GFS data fetching using NOMADS OpenDAP
            logger.info(f"Fetching GFS data for {time} hour {forecast_hour} variables {variables}")
            # Example URL: https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{time.strftime('%Y%m%d')}/gfs_0p25_{time.strftime('%H')}z
            return {}  # Replace with actual implementation
        except Exception as e:
            logger.error(f"Error fetching GFS data: {e}")
            return {}
