import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Toplevel
import asyncio
import logging
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import aiohttp
from wx_box.weather_app import (
    metar_command, taf_command, meteogram_command, windrose_command,
    active_storms_atl_command, sat_command, radar_command, astro_command,
    alerts_command, forecast_command, worldtimes_command, skewt,
    au_wind300, au_wind500, au_vort500, au_fronto700, au_rh700, au_wind850,
    au_dew850, au_mAdv850, au_tAdv850, au_mslp_temp, au_divcon300, au_thermal_wind,
    wind300, wind500, vort500, rh700, fronto700, wind850, dew850, mAdv850,
    tAdv850, mslp_temp, divcon300,
    eu_wind300, eu_wind500, eu_vort500, eu_rh700, eu_wind850, eu_dew850,
    eu_mAdv850, eu_tAdv850, eu_mslp_temp, eu_divcon300, get_metar, get_taf,
    fetch_jtwc_storms, fetch_nhc_invests
)
from wx_box.weatherpy.plotting import (
    plot_relative_humidity, plot_24_hour_relative_humidity_comparison,
    plot_temperature, plot_dry_and_gusty_areas,
    plot_relative_humidity_with_metar_obs, plot_low_relative_humidity_with_metar_obs
)
from wx_box.weatherpy.data_access import NOMADS_OPENDAP_Downloads, UCAR_THREDDS_SERVER_OPENDAP_Downloads
from wx_box.weatherpy.settings import get_metar_mask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.see(tk.END)

class WeatherAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Weather Application")
        self.root.geometry("1000x800")

        # Output directory
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Output directory selector
        self.output_dir_label = ttk.Label(self.main_frame, text=f"Output Directory: {self.output_dir}")
        self.output_dir_label.pack(anchor="w")
        ttk.Button(self.main_frame, text="Change Output Directory", command=self.change_output_dir).pack(anchor="w", pady=5)

        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Ready", foreground="blue")
        self.status_label.pack(anchor="w", pady=5)

        # Create notebook (tabs) for commands
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True)

        # Create tabs
        self.create_metar_tab()
        self.create_taf_tab()
        self.create_meteogram_tab()
        self.create_windrose_tab()
        self.create_active_storms_tab()
        self.create_sat_tab()
        self.create_radar_tab()
        self.create_astro_tab()
        self.create_alerts_tab()
        self.create_forecast_tab()
        self.create_worldtimes_tab()
        self.create_skewt_tab()
        self.create_weather_maps_tab()

        # Output text area
        self.output_text = tk.Text(self.main_frame, height=10, width=80)
        self.output_text.pack(pady=10, fill="x")

        # Image display area
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.pack(pady=10)

        # Redirect logging to text area
        self.log_handler = TextHandler(self.output_text)
        logger.addHandler(self.log_handler)

    def change_output_dir(self):
        new_dir = filedialog.askdirectory(title="Select Output Directory")
        if new_dir:
            self.output_dir = new_dir
            self.output_dir_label.config(text=f"Output Directory: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
            from wx_box.weatherpy.utilities import OUTPUT_DIR
            globals()['OUTPUT_DIR'] = self.output_dir

    def set_status(self, message, color="blue"):
        self.status_label.config(text=message, foreground=color)
        self.root.update()

    def create_popup_text(self, title, text):
        popup = Toplevel(self.root)
        popup.title(title)
        popup.geometry("600x400")
        text_widget = tk.Text(popup, wrap="word")
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, text)
        text_widget.config(state="normal")  # Allow copy-paste
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=5)

    def create_popup_image(self, title, image_path):
        popup = Toplevel(self.root)
        popup.title(title)
        popup.geometry("800x600")
        try:
            img = Image.open(image_path)
            img.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(img)
            image_label = ttk.Label(popup, image=photo)
            image_label.image = photo  # Keep reference
            image_label.pack(pady=10)
            ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=5)
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying image: {e}")
            popup.destroy()

    def create_metar_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="METAR")
        ttk.Label(tab, text="ICAO Code:").grid(row=0, column=0, padx=5, pady=5)
        self.metar_icao = ttk.Entry(tab)
        self.metar_icao.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(tab, text="Hours Back:").grid(row=1, column=0, padx=5, pady=5)
        self.metar_hours = ttk.Entry(tab)
        self.metar_hours.grid(row=1, column=1, padx=5, pady=5)
        self.metar_hours.insert(0, "0")
        ttk.Button(tab, text="Fetch METAR", command=self.run_metar).grid(row=2, column=0, columnspan=2, pady=10)

    def create_taf_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="TAF")
        ttk.Label(tab, text="ICAO Code:").grid(row=0, column=0, padx=5, pady=5)
        self.taf_icao = ttk.Entry(tab)
        self.taf_icao.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Fetch TAF", command=self.run_taf).grid(row=1, column=0, columnspan=2, pady=10)

    def create_meteogram_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Meteogram")
        ttk.Label(tab, text="ICAO Code:").grid(row=0, column=0, padx=5, pady=5)
        self.meteogram_icao = ttk.Entry(tab)
        self.meteogram_icao.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(tab, text="Hours Back:").grid(row=1, column=0, padx=5, pady=5)
        self.meteogram_hours = ttk.Entry(tab)
        self.meteogram_hours.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Generate Meteogram", command=self.run_meteogram).grid(row=2, column=0, columnspan=2, pady=10)

    def create_windrose_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Wind Rose")
        ttk.Label(tab, text="Longitude:").grid(row=0, column=0, padx=5, pady=5)
        self.windrose_lon = ttk.Entry(tab)
        self.windrose_lon.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(tab, text="Latitude:").grid(row=1, column=0, padx=5, pady=5)
        self.windrose_lat = ttk.Entry(tab)
        self.windrose_lat.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(tab, text="Start Date (YYYYMMDD):").grid(row=2, column=0, padx=5, pady=5)
        self.windrose_start = ttk.Entry(tab)
        self.windrose_start.grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(tab, text="End Date (YYYYMMDD):").grid(row=3, column=0, padx=5, pady=5)
        self.windrose_end = ttk.Entry(tab)
        self.windrose_end.grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Generate Wind Rose", command=self.run_windrose).grid(row=4, column=0, columnspan=2, pady=10)

    def create_active_storms_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Active Storms ATL")
        ttk.Button(tab, text="Show Active Storms", command=self.run_active_storms).grid(row=0, column=0, pady=10)

    def create_sat_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Satellite")
        ttk.Label(tab, text="Region:").grid(row=0, column=0, padx=5, pady=5)
        self.sat_region = ttk.Combobox(tab, values=["conus", "chase"], state="readonly")
        self.sat_region.grid(row=0, column=1, padx=5, pady=5)
        self.sat_region.set("conus")
        ttk.Label(tab, text="Product Code:").grid(row=1, column=0, padx=5, pady=5)
        self.sat_product = ttk.Combobox(tab, values=[str(i).zfill(2) for i in range(1, 17)], state="readonly")
        self.sat_product.grid(row=1, column=1, padx=5, pady=5)
        self.sat_product.set("01")
        ttk.Button(tab, text="Fetch Satellite Image", command=self.run_sat).grid(row=2, column=0, columnspan=2, pady=10)

    def create_radar_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Radar")
        ttk.Label(tab, text="Region:").grid(row=0, column=0, padx=5, pady=5)
        self.radar_region = ttk.Combobox(tab, values=["conus", "chase"], state="readonly")
        self.radar_region.grid(row=0, column=1, padx=5, pady=5)
        self.radar_region.set("chase")
        ttk.Label(tab, text="Overlay:").grid(row=1, column=0, padx=5, pady=5)
        self.radar_overlay = ttk.Combobox(tab, values=["base", "tops", "vil"], state="readonly")
        self.radar_overlay.grid(row=1, column=1, padx=5, pady=5)
        self.radar_overlay.set("base")
        ttk.Button(tab, text="Fetch Radar Image", command=self.run_radar).grid(row=2, column=0, columnspan=2, pady=10)

    def create_astro_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Astronomical Data")
        ttk.Label(tab, text="Location (city, ZIP, ICAO, or lat/lon):").grid(row=0, column=0, padx=5, pady=5)
        self.astro_location = ttk.Entry(tab)
        self.astro_location.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(tab, text="Time (HH:MM, optional):").grid(row=1, column=0, padx=5, pady=5)
        self.astro_time = ttk.Entry(tab)
        self.astro_time.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Generate Astronomical Data", command=self.run_astro).grid(row=2, column=0, columnspan=2, pady=10)

    def create_alerts_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Alerts")
        ttk.Label(tab, text="State Abbreviation (e.g., MT):").grid(row=0, column=0, padx=5, pady=5)
        self.alerts_state = ttk.Entry(tab)
        self.alerts_state.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Fetch Alerts", command=self.run_alerts).grid(row=1, column=0, columnspan=2, pady=10)

    def create_forecast_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Forecast")
        ttk.Label(tab, text="Location (city, ZIP, or lat/lon):").grid(row=0, column=0, padx=5, pady=5)
        self.forecast_location = ttk.Entry(tab)
        self.forecast_location.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Fetch Forecast", command=self.run_forecast).grid(row=1, column=0, columnspan=2, pady=10)

    def create_worldtimes_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="World Times")
        ttk.Button(tab, text="Show World Times", command=self.run_worldtimes).grid(row=0, column=0, pady=10)

    def create_skewt_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Skew-T")
        ttk.Label(tab, text="Station Code:").grid(row=0, column=0, padx=5, pady=5)
        self.skewt_station = ttk.Entry(tab)
        self.skewt_station.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(tab, text="Time (e.g., 12Z) or Model (e.g., gfs):").grid(row=1, column=0, padx=5, pady=5)
        self.skewt_time_model = ttk.Entry(tab)
        self.skewt_time_model.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(tab, text="Forecast Hour (if model, e.g., 6):").grid(row=2, column=0, padx=5, pady=5)
        self.skewt_forecast_hour = ttk.Entry(tab)
        self.skewt_forecast_hour.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Generate Skew-T", command=self.run_skewt).grid(row=3, column=0, columnspan=2, pady=10)

    def create_weather_maps_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Weather Maps")
        ttk.Label(tab, text="State (for RTMA plots):").grid(row=0, column=0, padx=5, pady=5)
        self.rtma_state = ttk.Entry(tab)
        self.rtma_state.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(tab, text="Select Map:").grid(row=1, column=0, padx=5, pady=5)
        map_options = [
            "Australia: 300 hPa Wind", "Australia: 500 hPa Wind", "Australia: 500 hPa Vorticity",
            "Australia: 700 hPa Frontogenesis", "Australia: 700 hPa Relative Humidity",
            "Australia: 850 hPa Wind", "Australia: 850 hPa Dewpoint", "Australia: 850 hPa Moisture Advection",
            "Australia: 850 hPa Temperature Advection", "Australia: MSLP with Temp Gradient",
            "Australia: 300 hPa Divergence/Convergence", "Australia: Thermal Wind",
            "CONUS: 300 hPa Wind", "CONUS: 500 hPa Wind", "CONUS: 500 hPa Vorticity",
            "CONUS: 700 hPa Relative Humidity", "CONUS: 700 hPa Frontogenesis",
            "CONUS: 850 hPa Wind", "CONUS: 850 hPa Dewpoint", "CONUS: 850 hPa Moisture Advection",
            "CONUS: 850 hPa Temperature Advection", "CONUS: MSLP with Temp Gradient",
            "CONUS: 300 hPa Divergence/Convergence", "CONUS: RTMA Relative Humidity",
            "CONUS: RTMA 24-Hour RH Comparison", "CONUS: RTMA Temperature",
            "CONUS: RTMA Dry and Gusty Areas", "CONUS: RTMA RH with METAR",
            "CONUS: RTMA Low RH with METAR",
            "Europe: 300 hPa Wind", "Europe: 500 hPa Wind", "Europe: 500 hPa Vorticity",
            "Europe: 700 hPa Relative Humidity", "Europe: 850 hPa Wind", "Europe: 850 hPa Dewpoint",
            "Europe: 850 hPa Moisture Advection", "Europe: 850 hPa Temperature Advection",
            "Europe: MSLP with Temp Gradient", "Europe: 300 hPa Divergence/Convergence"
        ]
        self.map_selection = ttk.Combobox(tab, values=map_options, state="readonly")
        self.map_selection.grid(row=1, column=1, padx=5, pady=5)
        self.map_selection.set(map_options[0])
        ttk.Button(tab, text="Generate Map", command=self.run_weather_map).grid(row=2, column=0, columnspan=2, pady=10)

    def run_metar(self):
        icao = self.metar_icao.get().strip().upper()
        hours = self.metar_hours.get().strip()
        try:
            hours = int(hours) if hours else 0
            if not icao:
                messagebox.showerror("Error", "Please enter an ICAO code.")
                return
            self.set_status("Fetching METAR...", "blue")
            self.output_text.delete(1.0, tk.END)
            output = []
            with RedirectText(self.output_text):
                metar_command(icao, hours)
                if hours > 0:
                    output.append(f"METARs for {icao} (Last {hours} Hours):")
                    metars = get_metar(icao, hours)
                    for i, metar in enumerate(metars):
                        output.append(f"Observation {i+1}: {metar}")
                else:
                    metars = get_metar(icao, 0)
                    output.append(f"METAR for {icao}: {metars[0]}")
            self.create_popup_text("METAR Output", "\n".join(output))
            self.set_status("Ready", "blue")
        except ValueError:
            messagebox.showerror("Error", "Hours back must be a number.")
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Error: Hours back must be a number.\n")
            self.set_status("Error", "red")
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching METAR: {e}")
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Error fetching METAR: {e}\n")
            self.set_status("Error", "red")

    def run_taf(self):
        icao = self.taf_icao.get().strip().upper()
        if not icao:
            messagebox.showerror("Error", "Please enter an ICAO code.")
            return
        self.set_status("Fetching TAF...", "blue")
        self.output_text.delete(1.0, tk.END)
        try:
            output = []
            with RedirectText(self.output_text):
                taf_command(icao)
                tafs = get_taf(icao)
                output.append(f"TAF for {icao}:\n{tafs[0]}")
            self.create_popup_text("TAF Output", "\n".join(output))
            self.set_status("Ready", "blue")
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching TAF: {e}")
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Error fetching TAF: {e}\n")
            self.set_status("Error", "red")

    def run_meteogram(self):
        icao = self.meteogram_icao.get().strip().upper()
        hours = self.meteogram_hours.get().strip()
        try:
            hours = int(hours)
            if not icao:
                messagebox.showerror("Error", "Please enter an ICAO code.")
                return
            if hours <= 0:
                messagebox.showerror("Error", "Hours back must be a positive number.")
                return
            self.set_status("Generating Meteogram...", "blue")
            self.output_text.delete(1.0, tk.END)
            with RedirectText(self.output_text):
                loop = asyncio.get_event_loop()
                output_path = loop.run_until_complete(meteogram_command(icao, hours))
            if output_path and os.path.exists(output_path):
                self.create_popup_image("Meteogram", output_path)
            else:
                messagebox.showerror("Error", "Meteogram image not found.")
            self.set_status("Ready", "blue")
        except ValueError:
            messagebox.showerror("Error", "Hours back must be a number.")
            self.set_status("Error", "red")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating meteogram: {e}")
            self.set_status("Error", "red")

    def run_windrose(self):
        lon = self.windrose_lon.get().strip()
        lat = self.windrose_lat.get().strip()
        start_date = self.windrose_start.get().strip()
        end_date = self.windrose_end.get().strip()
        try:
            lon = float(lon)
            lat = float(lat)
            if not (start_date.isdigit() and len(start_date) == 8 and end_date.isdigit() and len(end_date) == 8):
                messagebox.showerror("Error", "Dates must be in YYYYMMDD format.")
                return
            self.set_status("Generating Wind Rose...", "blue")
            self.output_text.delete(1.0, tk.END)
            with RedirectText(self.output_text):
                loop = asyncio.get_event_loop()
                output_path = loop.run_until_complete(windrose_command(lon, lat, start_date, end_date))
            if output_path and os.path.exists(output_path):
                self.create_popup_image("Wind Rose", output_path)
            else:
                messagebox.showerror("Error", "Wind rose image not found.")
            self.set_status("Ready", "blue")
        except ValueError:
            messagebox.showerror("Error", "Longitude and latitude must be numbers.")
            self.set_status("Error", "red")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating wind rose: {e}")
            self.set_status("Error", "red")

    async def run_active_storms(self):
        self.set_status("Fetching Active Storms...", "blue")
        self.output_text.delete(1.0, tk.END)
        try:
            output = []
            with RedirectText(self.output_text):
                output_path = await active_storms_atl_command()
                output.append("Active Storms and Invests in North Atlantic:")
                async with aiohttp.ClientSession() as session:
                    storms = await fetch_jtwc_storms(session) + await fetch_nhc_invests(session)
                for storm in storms:
                    output.append(f"Storm ID: {storm['id']}, Name: {storm['name']}, Lat: {storm['lat']}, Lon: {storm['lon']}, Vmax: {storm['vmax']}, MSLP: {storm['mslp']}")
            self.create_popup_text("Active Storms", "\n".join(output))
            if output_path and os.path.exists(output_path):
                self.create_popup_image("Active Storms Map", output_path)
            else:
                messagebox.showerror("Error", "No storm map image found.")
            self.set_status("Ready", "blue")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating active storms map: {e}")
            self.set_status("Error", "red")

    def run_sat(self):
        region = self.sat_region.get()
        product_code = self.sat_product.get()
        self.set_status("Fetching Satellite Image...", "blue")
        self.output_text.delete(1.0, tk.END)
        try:
            with RedirectText(self.output_text):
                loop = asyncio.get_event_loop()
                output_path = loop.run_until_complete(sat_command(region, product_code))
            if output_path and os.path.exists(output_path):
                self.create_popup_image("Satellite Image", output_path)
            else:
                messagebox.showerror("Error", "Satellite image not found.")
            self.set_status("Ready", "blue")
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching satellite image: {e}")
            self.set_status("Error", "red")

    def run_radar(self):
        region = self.radar_region.get()
        overlay = self.radar_overlay.get()
        self.set_status("Fetching Radar Image...", "blue")
        self.output_text.delete(1.0, tk.END)
        try:
            with RedirectText(self.output_text):
                output_path = radar_command(region, overlay)
            if output_path and os.path.exists(output_path):
                self.create_popup_image("Radar Image", output_path)
            else:
                messagebox.showerror("Error", "Radar image not found.")
            self.set_status("Ready", "blue")
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching radar image: {e}")
            self.set_status("Error", "red")

    def run_astro(self):
        location = self.astro_location.get().strip()
        time_str = self.astro_time.get().strip() or None
        if not location:
            messagebox.showerror("Error", "Please enter a location.")
            return
        self.set_status("Generating Astronomical Data...", "blue")
        self.output_text.delete(1.0, tk.END)
        try:
            output = []
            with RedirectText(self.output_text):
                loop = asyncio.get_event_loop()
                output_path = loop.run_until_complete(astro_command(location, time_str))
                # Simulate astro output for popup
                from wx_box.weather_app import parse_time
                lat, lon = None, None
                location_data = {
                    "KATL": (33.6367, -84.4281), "KJFK": (40.6398, -73.7789), "KBIX": (30.4103, -88.9261),
                    "KVQQ": (30.2264, -81.8878), "KVPC": (34.1561, -84.7983), "KRMG": (34.4956, -85.2214),
                    "KMGE": (33.9131, -84.5197), "KGPT": (30.4075, -89.0753), "KPIT": (40.4915, -80.2329),
                    "KSGJ": (34.25, -84.95), "KPEZ": (40.3073, -75.6192), "KNSE": (30.72247, -87.02390),
                    "KTPA": (27.9755, -82.5332), "30184": (34.1561, -84.7983), "30303": (33.7525, -84.3922)
                }
                if '/' in location:
                    lat, lon = map(float, location.split('/'))
                elif location.upper() in location_data or location in location_data:
                    lat, lon = location_data.get(location.upper(), location_data.get(location))
                else:
                    geolocator = Nominatim(user_agent="weather_app")
                    location_obj = geolocator.geocode(location)
                    if location_obj:
                        lat, lon = location_obj.latitude, location_obj.longitude
                observer = ephem.Observer()
                observer.lat = str(lat)
                observer.lon = str(lon)
                observer.date = parse_time(time_str)
                sun = ephem.Sun()
                moon = ephem.Moon()
                sun.compute(observer)
                moon.compute(observer)
                sun_alt = float(sun.alt) * 180 / math.pi
                sun_az = float(sun.az) * 180 / math.pi
                moon_alt = float(moon.alt) * 180 / math.pi
                moon_az = float(moon.az) * 180 / math.pi
                moon_phase = moon.phase
                tf = TimezoneFinder()
                timezone_str = tf.timezone_at(lat=lat, lng=lon) or 'UTC'
                local_tz = pytz.timezone(timezone_str)
                local_time = observer.date.datetime().replace(tzinfo=pytz.UTC).astimezone(local_tz)
                output.append(f"Astronomical Data for {location} at {local_time.strftime('%Y-%m-%d %H:%M %Z')} (Lat: {lat}, Lon: {lon}):")
                output.append(f"Sun Altitude: {sun_alt:.2f}°")
                output.append(f"Sun Azimuth: {sun_az:.2f}°")
                output.append(f"Moon Altitude: {moon_alt:.2f}°")
                output.append(f"Moon Azimuth: {moon_az:.2f}°")
                output.append(f"Moon Phase: {moon_phase:.2f}%")
                planets = ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']
                planet_data = []
                for planet_name in planets:
                    planet = getattr(ephem, planet_name)()
                    planet.compute(observer)
                    planet_alt = float(planet.alt) * 180 / math.pi
                    planet_az = float(planet.az) * 180 / math.pi
                    planet_data.append({'name': planet_name, 'alt': planet_alt, 'az': planet_az})
                output.append("\nPlanetary Positions:")
                for planet in planet_data:
                    output.append(f"{planet['name']}: Altitude {planet['alt']:.2f}°, Azimuth {planet['az']:.2f}°")
            self.create_popup_text("Astronomical Data", "\n".join(output))
            if output_path and os.path.exists(output_path):
                self.create_popup_image("Astronomical Plot", output_path)
            else:
                messagebox.showerror("Error", "Astronomical plot not found.")
            self.set_status("Ready", "blue")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating astronomical data: {e}")
            self.set_status("Error", "red")

    async def run_alerts(self):
        state_abbr = self.alerts_state.get().strip().upper()
        if not state_abbr:
            messagebox.showerror("Error", "Please enter a state abbreviation.")
            return
        self.set_status("Fetching Alerts...", "blue")
        self.output_text.delete(1.0, tk.END)
        try:
            output = []
            with RedirectText(self.output_text):
                success = await alerts_command(state_abbr)
                if success:
                    alerts_url = f"https://api.weather.gov/alerts/active?area={state_abbr}"
                    output.append(f"Weather Alerts for {state_abbr}:")
                    async with aiohttp.ClientSession() as session:
                        async with session.get(alerts_url, headers={'User-Agent': 'WeatherApp/1.0'}) as response:
                            response.raise_for_status()
                            alerts_data = await response.json()
                            filtered_alerts = [
                                alert for alert in alerts_data.get('features', [])
                                if alert.get('properties') and alert['properties'].get('event') and alert['properties'].get('severity')
                            ]
                            if filtered_alerts:
                                for alert in filtered_alerts:
                                    properties = alert['properties']
                                    headline = properties.get('headline', 'No Headline')
                                    event = properties.get('event', 'Unknown Event')
                                    severity = properties.get('severity', 'Unknown Severity')
                                    description = properties.get('description', 'No description available.')
                                    area_desc = "".join(properties.get('areaDesc', '')).split(";")
                                    area_desc = [area.strip() for area in area_desc if area.strip()]
                                    area_desc = ", ".join(area_desc)
                                    output.append(f"\n--- {headline} ---")
                                    output.append(f"Event: {event}")
                                    output.append(f"Severity: {severity}")
                                    output.append(f"Area: {area_desc}")
                                    output.append(f"Description:\n{description}")
                            else:
                                output.append(f"No weather alerts found for {state_abbr}.")
            self.create_popup_text("Weather Alerts", "\n".join(output))
            self.set_status("Ready", "blue")
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching alerts: {e}")
            self.set_status("Error", "red")

    async def run_forecast(self):
        location = self.forecast_location.get().strip()
        if not location:
            messagebox.showerror("Error", "Please enter a location.")
            return
        self.set_status("Fetching Forecast...", "blue")
        self.output_text.delete(1.0, tk.END)
        try:
            output = []
            with RedirectText(self.output_text):
                success = await forecast_command(location)
                if success:
                    lat, lon = None, None
                    location_data = {
                        "KATL": (33.6367, -84.4281), "KJFK": (40.6398, -73.7789), "KBIX": (30.4103, -88.9261),
                        "KVQQ": (30.2264, -81.8878), "KVPC": (34.1561, -84.7983), "KRMG": (34.4956, -85.2214),
                        "KMGE": (33.9131, -84.5197), "KGPT": (30.4075, -89.0753), "KPIT": (40.4915, -80.2329),
                        "KSGJ": (34.25, -84.95), "KPEZ": (40.3073, -75.6192), "KNSE": (30.72247, -87.02390),
                        "KTPA": (27.9755, -82.5332), "30184": (34.1561, -84.7983), "30303": (33.7525, -84.3922)
                    }
                    if '/' in location:
                        lat, lon = map(float, location.split('/'))
                    elif location.upper() in location_data or location in location_data:
                        lat, lon = location_data.get(location.upper(), location_data.get(location))
                    elif location.isdigit() and len(location) == 5:
                        async with aiohttp.ClientSession() as session:
                            geonames_url = f"http://api.geonames.org/postalCodeSearchJSON?postalcode={location}&country=US&username=freeuser&maxRows=1"
                            async with session.get(geonames_url, headers={'User-Agent': 'WeatherApp/1.0'}) as response:
                                data = await response.json()
                                if data.get('postalCodes'):
                                    lat = data['postalCodes'][0]['lat']
                                    lon = data['postalCodes'][0]['lng']
                    else:
                        async with aiohttp.ClientSession() as session:
                            geocode_url = f"https://api.weather.gov/points/{location.replace(' ', '+')}"
                            async with session.get(geocode_url, headers={'User-Agent': 'WeatherApp/1.0'}) as response:
                                data = await response.json()
                                lat = data['properties']['relativeLocation']['geometry']['coordinates'][1]
                                lon = data['properties']['relativeLocation']['geometry']['coordinates'][0]
                    async with aiohttp.ClientSession() as session:
                        nws_zone_url = f"https://api.weather.gov/points/{lat},{lon}"
                        async with session.get(nws_zone_url, headers={'User-Agent': 'WeatherApp/1.0'}) as response:
                            nws_data = await response.json()
                            zone_id = nws_data['properties']['forecastZone'].split('/')[-1]
                            state_code = zone_id[:2].lower()
                            zone_code = zone_id.lower()
                    forecast_url = f"https://tgftp.nws.noaa.gov/data/forecasts/zone/{state_code}/{zone_code}.txt"
                    response = requests.get(forecast_url)
                    forecast_text = response.text
                    output.append(f"Weather Forecast for {location.title()} (Zone: {zone_id}):")
                    output.append(f"```\n{forecast_text.strip()}\n```")
            self.create_popup_text("Weather Forecast", "\n".join(output))
            self.set_status("Ready", "blue")
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching forecast: {e}")
            self.set_status("Error", "red")

    def run_worldtimes(self):
        self.set_status("Fetching World Times...", "blue")
        self.output_text.delete(1.0, tk.END)
        try:
            output = []
            with RedirectText(self.output_text):
                success = worldtimes_command()
                if success:
                    utc_now = pytz.utc.localize(datetime.utcnow())
                    us_timezones = {
                        "Hawaii": "Pacific/Honolulu",
                        "Alaska": "America/Anchorage",
                        "Pacific": "America/Los_Angeles",
                        "Mountain": "America/Denver",
                        "Central": "America/Chicago",
                        "Eastern": "America/New_York"
                    }
                    international_timezones = {
                        "London": "Europe/London",
                        "Berlin": "Europe/Berlin",
                        "Tokyo": "Asia/Tokyo",
                        "Sydney": "Australia/Sydney",
                        "Tehran (Iran)": "Asia/Tehran",
                        "Jerusalem (Israel)": "Asia/Jerusalem",
                        "Moscow": "Europe/Moscow",
                        "Beijing": "Asia/Shanghai"
                    }
                    output.append("Time Zones:")
                    output.append("\nUS Time Zones:")
                    for region, timezone_str in us_timezones.items():
                        timezone = pytz.timezone(timezone_str)
                        local_time = utc_now.astimezone(timezone)
                        output.append(f"{region} (US): {local_time.strftime('%H:%M:%S')}")
                    output.append("\nInternational Time Zones:")
                    for city, timezone_str in international_timezones.items():
                        timezone = pytz.timezone(timezone_str)
                        local_time = utc_now.astimezone(timezone)
                        output.append(f"{city}: {local_time.strftime('%H:%M:%S')}")
            self.create_popup_text("World Times", "\n".join(output))
            self.set_status("Ready", "blue")
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching world times: {e}")
            self.set_status("Error", "red")

    def run_skewt(self):
        station = self.skewt_station.get().strip()
        time_model = self.skewt_time_model.get().strip()
        forecast_hour = self.skewt_forecast_hour.get().strip()
        if not station or not time_model:
            messagebox.showerror("Error", "Please enter station code and time/model.")
            return
        self.set_status("Generating Skew-T...", "blue")
        self.output_text.delete(1.0, tk.END)
        try:
            with RedirectText(self.output_text):
                loop = asyncio.get_event_loop()
                if forecast_hour:
                    output_path = loop.run_until_complete(skewt(station, model=time_model, forecast_hour=int(forecast_hour)))
                else:
                    output_path = loop.run_until_complete(skewt(station, sounding_time=time_model))
            if output_path and os.path.exists(output_path):
                self.create_popup_image("Skew-T Diagram", output_path)
            else:
                messagebox.showerror("Error", "Skew-T image not found.")
            self.set_status("Ready", "blue")
        except ValueError:
            messagebox.showerror("Error", "Forecast hour must be a number.")
            self.set_status("Error", "red")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating Skew-T: {e}")
            self.set_status("Error", "red")

    def run_weather_map(self):
        map_name = self.map_selection.get()
        state = self.rtma_state.get().strip().upper()
        self.set_status(f"Generating {map_name}...", "blue")
        self.output_text.delete(1.0, tk.END)
        try:
            with RedirectText(self.output_text):
                loop = asyncio.get_event_loop()
                map_functions = {
                    "Australia: 300 hPa Wind": au_wind300,
                    "Australia: 500 hPa Wind": au_wind500,
                    "Australia: 500 hPa Vorticity": au_vort500,
                    "Australia: 700 hPa Frontogenesis": au_fronto700,
                    "Australia: 700 hPa Relative Humidity": au_rh700,
                    "Australia: 850 hPa Wind": au_wind850,
                    "Australia: 850 hPa Dewpoint": au_dew850,
                    "Australia: 850 hPa Moisture Advection": au_mAdv850,
                    "Australia: 850 hPa Temperature Advection": au_tAdv850,
                    "Australia: MSLP with Temp Gradient": au_mslp_temp,
                    "Australia: 300 hPa Divergence/Convergence": au_divcon300,
                    "Australia: Thermal Wind": au_thermal_wind,
                    "CONUS: 300 hPa Wind": wind300,
                    "CONUS: 500 hPa Wind": wind500,
                    "CONUS: 500 hPa Vorticity": vort500,
                    "CONUS: 700 hPa Relative Humidity": rh700,
                    "CONUS: 700 hPa Frontogenesis": fronto700,
                    "CONUS: 850 hPa Wind": wind850,
                    "CONUS: 850 hPa Dewpoint": dew850,
                    "CONUS: 850 hPa Moisture Advection": mAdv850,
                    "CONUS: 850 hPa Temperature Advection": tAdv850,
                    "CONUS: MSLP with Temp Gradient": mslp_temp,
                    "CONUS: 300 hPa Divergence/Convergence": divcon300,
                    "CONUS: RTMA Relative Humidity": lambda: self.run_rtma_rh(),
                    "CONUS: RTMA 24-Hour RH Comparison": lambda: self.run_rtma_rh_24hr(),
                    "CONUS: RTMA Temperature": lambda: self.run_rtma_temp(),
                    "CONUS: RTMA Dry and Gusty Areas": lambda: self.run_rtma_dry_gusty(),
                    "CONUS: RTMA RH with METAR": lambda: self.run_rtma_rh_metar(),
                    "CONUS: RTMA Low RH with METAR": lambda: self.run_rtma_low_rh_metar(),
                    "Europe: 300 hPa Wind": eu_wind300,
                    "Europe: 500 hPa Wind": eu_wind500,
                    "Europe: 500 hPa Vorticity": eu_vort500,
                    "Europe: 700 hPa Relative Humidity": eu_rh700,
                    "Europe: 850 hPa Wind": eu_wind850,
                    "Europe: 850 hPa Dewpoint": eu_dew850,
                    "Europe: 850 hPa Moisture Advection": eu_mAdv850,
                    "Europe: 850 hPa Temperature Advection": eu_tAdv850,
                    "Europe: MSLP with Temp Gradient": eu_mslp_temp,
                    "Europe: 300 hPa Divergence/Convergence": eu_divcon300
                }
                map_func = map_functions.get(map_name)
                if map_func:
                    output_path = loop.run_until_complete(map_func())
                    if output_path and os.path.exists(output_path):
                        self.create_popup_image(map_name, output_path)
                    else:
                        messagebox.showerror("Error", f"{map_name} image not found.")
                else:
                    messagebox.showerror("Error", f"Invalid map selection: {map_name}")
            self.set_status("Ready", "blue")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating {map_name}: {e}")
            self.set_status("Error", "red")

    def run_rtma_rh(self):
        state = self.rtma_state.get().strip().upper()
        if not state:
            messagebox.showerror("Error", "Please enter a state abbreviation for RTMA plots.")
            return None
        try:
            loop = asyncio.get_event_loop()
            utc_time = datetime.now(pytz.UTC)
            ds, rtma_time = NOMADS_OPENDAP_Downloads.RTMA_CONUS.get_RTMA_dataset(utc_time)
            path = loop.run_until_complete(plot_relative_humidity(state=state, data=ds, time=rtma_time))
            return path
        except Exception as e:
            logger.error(f"Error generating RTMA Relative Humidity: {e}")
            return None

    def run_rtma_rh_24hr(self):
        state = self.rtma_state.get().strip().upper()
        if not state:
            messagebox.showerror("Error", "Please enter a state abbreviation for RTMA plots.")
            return None
        try:
            loop = asyncio.get_event_loop()
            utc_time = datetime.now(pytz.UTC)
            ds, ds_24, rtma_time, rtma_time_24 = NOMADS_OPENDAP_Downloads.RTMA_CONUS.get_RTMA_24_hour_comparison_datasets(utc_time)
            path = loop.run_until_complete(plot_24_hour_relative_humidity_comparison(state=state, data=ds, data_24=ds_24, time=rtma_time, time_24=rtma_time_24))
            return path
        except Exception as e:
            logger.error(f"Error generating RTMA 24-Hour RH Comparison: {e}")
            return None

    def run_rtma_temp(self):
        state = self.rtma_state.get().strip().upper()
        if not state:
            messagebox.showerror("Error", "Please enter a state abbreviation for RTMA plots.")
            return None
        try:
            loop = asyncio.get_event_loop()
            utc_time = datetime.now(pytz.UTC)
            ds, rtma_time = NOMADS_OPENDAP_Downloads.RTMA_CONUS.get_RTMA_dataset(utc_time)
            path = loop.run_until_complete(plot_temperature(state=state, data=ds, time=rtma_time))
            return path
        except Exception as e:
            logger.error(f"Error generating RTMA Temperature: {e}")
            return None

    def run_rtma_dry_gusty(self):
        state = self.rtma_state.get().strip().upper()
        if not state:
            messagebox.showerror("Error", "Please enter a state abbreviation for RTMA plots.")
            return None
        try:
            loop = asyncio.get_event_loop()
            utc_time = datetime.now(pytz.UTC)
            ds, rtma_time = NOMADS_OPENDAP_Downloads.RTMA_CONUS.get_RTMA_dataset(utc_time)
            path = loop.run_until_complete(plot_dry_and_gusty_areas(state=state, data=ds, time=rtma_time))
            return path
        except Exception as e:
            logger.error(f"Error generating RTMA Dry and Gusty Areas: {e}")
            return None

    def run_rtma_rh_metar(self):
        state = self.rtma_state.get().strip().upper()
        if not state:
            messagebox.showerror("Error", "Please enter a state abbreviation for RTMA plots.")
            return None
        try:
            loop = asyncio.get_event_loop()
            utc_time = datetime.now(pytz.UTC)
            mask = get_metar_mask(state, None)
            data = loop.run_until_complete(UCAR_THREDDS_SERVER_OPENDAP_Downloads.METARs.RTMA_Relative_Humidity_Synced_With_METAR(utc_time, mask))
            path = loop.run_until_complete(plot_relative_humidity_with_metar_obs(state=state, data=data))
            return path
        except Exception as e:
            logger.error(f"Error generating RTMA RH with METAR: {e}")
            return None

    def run_rtma_low_rh_metar(self):
        state = self.rtma_state.get().strip().upper()
        if not state:
            messagebox.showerror("Error", "Please enter a state abbreviation for RTMA plots.")
            return None
        try:
            loop = asyncio.get_event_loop()
            utc_time = datetime.now(pytz.UTC)
            mask = get_metar_mask(state, None)
            data = loop.run_until_complete(UCAR_THREDDS_SERVER_OPENDAP_Downloads.METARs.RTMA_Relative_Humidity_Synced_With_METAR(utc_time, mask))
            path = loop.run_until_complete(plot_low_relative_humidity_with_metar_obs(state=state, data=data))
            return path
        except Exception as e:
            logger.error(f"Error generating RTMA Low RH with METAR: {e}")
            return None

def main():
    root = tk.Tk()
    app = WeatherAppGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
