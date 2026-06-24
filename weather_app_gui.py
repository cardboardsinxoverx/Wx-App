import customtkinter as ctk
import subprocess
import threading
import sys
import os
import re
import asyncio
import time
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from numpy.random import choice
from skewt import generate_skewt_plot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Configuration ---
SCRIPT_FILES = {
    "ga_wx": "state maps/ga_wx.py",
    "al_wx": "state maps/al_wx.py",
    "fl_wx": "state maps/fl_wx.py",
    "la_wx": "state maps/la_wx.py",
    "ms_wx": "state maps/ms_wx.py",
    "mt_wx": "state maps/mt_wx.py",
    "tx_wx": "state maps/tx_wx.py",
    "nc_wx": "state maps/nc_wx.py", 
    "ok_wx": "state maps/ok_wx.py", 
    "mn_wx": "state maps/mn_wx.py",
    "sc_wx": "state maps/sc_wx.py",
    "ca_wx": "state maps/ca_wx.py",
    "tn_wx": "state maps/tn_wx.py",
    "va_wx": "state maps/va_wx.py",
    "wv_wx": "state maps/wv_wx.py",
    "ga_ndfd_fcst": "state maps/ga_wx_fcst.py",
    "awa_wx": "awa_wx.py",
    "atl_maps": "atl_weather_maps.py",
    "conus_maps": "weather_maps.py",
    "eu_maps": "eu_weather_maps.py",
    "au_maps": "au_weather_maps.py",
    "trop_atl": "tropical_atl.py",
    "trop_outlook": "tropical_outlook_atl.py",
    "skewt": "skewt.py",
    "windrose": "windrose.py",
    "cross_section": "cross_section.py",
    "sat": "sat.py",
    "main_cli": "frostbyte.py",
    "fcst_meteogram": "forecast_meteogram.py"  # <-- Added Forecast Meteogram Script
}

# --- Core Script Execution Logic ---
def run_script_in_thread(command, status_label, progress_bar, log_textbox, preview_label=None, output_image_name=None):
    def _execute():
        try:
            progress_bar.set(0.0)
            log_textbox.delete("1.0", "end")
            log_textbox.insert("end", "FrostByte Terminal > Initializing...\n")
            if preview_label:
                preview_label.configure(image=None, text="Map preview will appear here")
                preview_label.pil_image = None

            time.sleep(0.3)
            log_textbox.insert("end", "Fetching data...\n")
            for i in range(41):
                progress_bar.set(i / 100)
                time.sleep(0.01)
            log_textbox.insert("end", "Data loaded.\n")

            log_textbox.insert("end", "Processing...\n")
            for i in range(40, 81):
                progress_bar.set(i / 100)
                time.sleep(0.02)
            log_textbox.insert("end", "Processing complete.\n")

            log_textbox.insert("end", f"Executing: {' '.join(command)}\n")
            log_textbox.insert("end", "="*60 + "\n")
            status_label.configure(text="Running script...")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )

            stdout_lines = []
            for line in process.stdout:
                line = line.strip()
                if line:
                    stdout_lines.append(line)
                    log_textbox.insert("end", line + "\n")
                    log_textbox.see("end")
                if len(stdout_lines) % 5 == 0:
                    progress_bar.set(min(0.95, 0.8 + (len(stdout_lines) * 0.005)))
            
            stdout, stderr = process.communicate()
            if stderr:
                log_textbox.insert("end", f"[STDERR]\n{stderr}\n")
                log_textbox.see("end")

            returncode = process.returncode
            progress_bar.set(1.0)

            if returncode == 0:
                status_label.configure(text="Completed successfully.")
                log_textbox.insert("end", "\nSuccess.\n")
                
                if preview_label and output_image_name:
                    time.sleep(0.5) 
                    
                    final_output_image_name = output_image_name
                    
                    if output_image_name == 'cross_section.png':
                        try:
                            final_output_match = re.search(r'Cross-section plot saved to: (.*\.png)', log_textbox.get("1.0", "end"))
                            if final_output_match:
                                final_output_image_name = final_output_match.group(1).strip()
                        except: pass
                    elif output_image_name == 'DYNAMIC_CONUS':
                        try:
                            log_text = log_textbox.get("1.0", "end")
                            
                            patterns = [
                                r'✅ Generated (.*?\.png)',                    
                                r'(wind_speed|mslp_temp|.*mb_.*\d{8}_\d{4}Z\.png)',  
                                r'(?:Saved|Generated|saved to:).*?([^\s]+\.png)',       
                            ]
                            
                            final_output_image_name = "DYNAMIC_CONUS"
                            for pattern in patterns:
                                match = re.search(pattern, log_text, re.IGNORECASE)
                                if match:
                                    final_output_image_name = match.group(1).strip()
                                    break
                        except Exception as e:
                            print(f"Preview regex error: {e}")
                            final_output_image_name = "DYNAMIC_CONUS"
                            
                    if os.path.exists(final_output_image_name):
                        pil_image = Image.open(final_output_image_name)
                        img_width, img_height = pil_image.size
                        
                        scale_h = 400 / img_height
                        new_width_h = int(img_width * scale_h)
                        new_height_h = 400

                        scale_w = 850 / img_width
                        new_width_w = 850
                        new_height_w = int(img_height * scale_w)

                        if new_width_h > 850:
                            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(new_width_w, new_height_w))
                        else:
                            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(new_width_h, new_height_h))

                        preview_label.configure(image=ctk_image, text="")
                        preview_label.pil_image = pil_image 
                        preview_label.ctk_image = ctk_image
                    else:
                        log_textbox.insert("end", f"\nPreview failed: {final_output_image_name} not found.\n")
                        full_text_output = "\n".join(stdout_lines)
                        preview_label.configure(image=None, text=full_text_output, anchor="nw", justify="left")
                        preview_label.pil_image = None 
                
                elif stdout_lines:
                    full_text_output = "\n".join(stdout_lines)
                    preview_label.configure(image=None, text=full_text_output, anchor="nw", justify="left") 
                    preview_label.pil_image = None
            else:
                status_label.configure(text="Failed.")
                error_msg = stderr.strip() if stderr else "Unknown error."
                log_textbox.insert("end", f"\nError: {error_msg[:300]}...\n")
                messagebox.showerror("Error", error_msg)

        except FileNotFoundError:
            progress_bar.set(0.0)
            status_label.configure(text="Script not found.")
            log_textbox.insert("end", f"\nError: Script not found: {command[1]}\n")
            messagebox.showerror("Error", f"Script not found: {command[1]}")
        except Exception as e:
            progress_bar.set(0.0)
            status_label.configure(text="Unexpected error.")
            log_textbox.insert("end", f"\nError: {str(e)}\n")
            messagebox.showerror("Error", str(e))

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()

# --- Main Application GUI ---
class WeatherLauncherApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FrostByte")
        self.geometry("900x900")
        
        # --- FrostByte Unified GUI Theming ---
        ctk.set_appearance_mode("dark")
        self.configure(fg_color="#333333")

        self.tab_view = ctk.CTkTabview(
            self, width=880, height=200, 
            fg_color="#2B2B2B",
            segmented_button_fg_color="#1A1A1A",
            segmented_button_unselected_color="#1A1A1A",
            segmented_button_selected_color="#3932A0",
            segmented_button_selected_hover_color="#4D96FF"
        )
        self.tab_view.pack(padx=10, pady=10, fill="x")
        self.tab_view.add("State Maps")
        self.tab_view.add("Regional Maps")
        self.tab_view.add("Specialty Plots")
        self.tab_view.add("Utilities")

        self.preview_frame = ctk.CTkFrame(self, height=400, fg_color="#2B2B2B", border_width=2, border_color="#3932A0")
        self.preview_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.preview_frame.grid_propagate(False)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Map preview will appear here", text_color="grey")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        self.progress = ctk.CTkProgressBar(self, mode="determinate", height=16, progress_color="#00FFFF", fg_color="#1A1A1A")
        self.progress.pack(padx=10, pady=5, fill="x")
        self.progress.set(0)

        self.status_label = ctk.CTkLabel(self, text="Ready", anchor="w", font=ctk.CTkFont(size=12))
        self.status_label.pack(padx=10, pady=(0, 5), fill="x")

        self.log_textbox = ctk.CTkTextbox(
            self,
            height=150,
            fg_color="#121212",
            text_color="#00FFFF",
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word",
            border_width=2,
            border_color="#3932A0"
        )
        self.log_textbox.pack(padx=10, pady=(0, 10), fill="both")
        self.log_textbox.insert("end", "FrostByte Terminal > System ready.\n")

        self.setup_state_maps_tab()
        self.setup_regional_maps_tab()
        self.setup_specialty_plots_tab()
        self.setup_utilities_tab()

    def run_simple_script(self, script_key, output_image_name=None):
        script_name = SCRIPT_FILES.get(script_key)
        if script_name and os.path.exists(script_name):
            command = [sys.executable, script_name]
            run_script_in_thread(command, self.status_label, self.progress, self.log_textbox, self.preview_label, output_image_name)
        else:
            messagebox.showerror("Error", f"Script not found: {script_name}")

    def validate_icao(self, icao):
        return bool(re.match(r'^[A-Z]{3,4}$', icao.upper()))
        
    def validate_lat_lon(self, loc_str):
        return bool(re.match(r'^-?\d+(\.\d+)?, *-?\d+(\.\d+)?$', loc_str.strip()))

    def validate_cross_section_loc(self, loc_str):
        loc_str = loc_str.strip()
        # Check if it matches the standard lat,lon format
        if re.match(r'^-?\d+(\.\d+)?, *-?\d+(\.\d+)?$', loc_str):
            return True
        # Check if it is a valid ICAO code
        if self.validate_icao(loc_str):
            return True
        return False
    
    def setup_state_maps_tab(self):
        tab = self.tab_view.tab("State Maps")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure((0, 1, 2), weight=1) 

        frame = ctk.CTkFrame(tab, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(frame, text="States", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(fill="x", padx=10, pady=5)
        
        self.combo_states = ctk.CTkComboBox(
            inner,
            values=[
                "Alabama (AL)", "California (CA)", "Florida (FL)", "Georgia (GA)",
                "Louisiana (LA)", "Minnesota (MN)", "Mississippi (MS)", "Montana (MT)",
                "North Carolina (NC)", "Oklahoma (OK)", "South Carolina (SC)",
                "Tennessee (TN)", "Texas (TX)", "Virginia (VA)", "West Virginia (WV)"
            ],
            state="readonly"
        )
        self.combo_states.pack(fill="x", pady=5)
        self.combo_states.set("Georgia (GA)")

        ctk.CTkButton(inner, text="Generate Map", command=self.on_generate_state_map).pack(fill="x", pady=5)

        frame_ndfd = ctk.CTkFrame(tab, fg_color="transparent")
        frame_ndfd.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(frame_ndfd, text="NDFD Forecasts", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        inner_ndfd = ctk.CTkFrame(frame_ndfd, fg_color="transparent")
        inner_ndfd.pack(fill="x", padx=10, pady=5)
        
        self.combo_ndfd = ctk.CTkComboBox(
            inner_ndfd,
            values=["Georgia (GA)"],
            state="readonly"
        )
        self.combo_ndfd.pack(fill="x", pady=5)
        self.combo_ndfd.set("Georgia (GA)")

        ctk.CTkButton(inner_ndfd, text="Generate NDFD Map", command=self.on_generate_ndfd_map).pack(fill="x", pady=5)

    def on_generate_state_map(self):
        state = self.combo_states.get().split()[-1].strip("()").lower()
        self.run_simple_script(f"{state}_wx", f"{state}_detailed_weather.png")

    def on_generate_ndfd_map(self):
        state_str = self.combo_ndfd.get()
        if "Georgia (GA)" in state_str:
            script_key = "ga_ndfd_fcst"
            preview_image = "ga_forecast_highs.png"
            self.run_simple_script(script_key, preview_image)
        else:
            messagebox.showerror("Error", "Selected NDFD state not configured.")

    def setup_regional_maps_tab(self):
        tab = self.tab_view.tab("Regional Maps")
        tab.grid_rowconfigure((0,1,2,3,4,5), weight=1)
        tab.grid_columnconfigure(0, weight=1)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="AWA", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", command=lambda: self.run_simple_script("awa_wx", "awa_weather.png")).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="ATL", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.atl_map_type = ctk.CTkComboBox(f, values=[
            "Surface Analysis (MSLP)", "300 mb Wind Speed", "300 mb Divergence/Convergence",
            "500 mb Wind Speed", "500 mb Relative Vorticity", "700 mb Relative Humidity",
            "700 mb Frontogenesis", "850 mb Wind Speed", "850 mb Dewpoint",
            "850 mb Moisture Advection", "850 mb Temp Advection"
        ], state="readonly", width=250)
        self.atl_map_type.pack(side="left", padx=5)
        self.atl_map_type.set("Surface Analysis (MSLP)")
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_atl).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="CONUS", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.conus_map_type = ctk.CTkComboBox(f, values=[
            "Surface Analysis (MSLP)", "300 mb Wind Speed", "300 mb Divergence/Convergence",
            "500 mb Wind Speed", "500 mb Relative Vorticity", "500 mb Water Vapor & Vorticity", 
            "700 mb Relative Humidity", "700 mb Frontogenesis", "850 mb Wind Speed",
            "850 mb Dewpoint", "850 mb Moisture Advection", "850 mb Temp Advection"
        ], state="readonly", width=250)
        self.conus_map_type.pack(side="left", padx=5)
        self.conus_map_type.set("Surface Analysis (MSLP)")
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_conus).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="EU", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", command=lambda: self.run_simple_script("eu_maps", "eu_weather.png")).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="AU", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        
        # Domain Selector (Added "AU" to the front of the list)
        self.au_domain = ctk.CTkComboBox(f, values=["AU", "SY", "ME", "BN", "AD", "PH", "DN", "R"], state="readonly", width=65)
        self.au_domain.pack(side="left", padx=5)
        self.au_domain.set("AU") # Optional: set it as the default

        # Map Type Selector
        self.au_map_type = ctk.CTkComboBox(f, values=[
            "Surface Analysis (MSLP)", "300 mb Wind Speed", "300 mb Divergence/Convergence",
            "500 mb Wind Speed", "500 mb Relative Vorticity", "500 mb Water Vapor & Vorticity", 
            "700 mb Relative Humidity", "700 mb Frontogenesis", "850 mb Wind Speed",
            "850 mb Dewpoint", "850 mb Moisture Advection", "850 mb Temp Advection"
        ], state="readonly", width=250)
        self.au_map_type.pack(side="left", padx=5)
        self.au_map_type.set("Surface Analysis (MSLP)")
        
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_au).pack(side="right", padx=5)

        # --- Satellite Section ---
        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Satellite", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)

        # Base product dictionaries to keep code clean
        goes_prods = {"2":"Red Visible (0.64µm)", "14":"Clean LW IR Window (10.3µm)", "9":"Mid-level WV (6.9µm)"}
        eum_prods = {"2":"Red Visible", "9":"Water Vapor", "14":"Infrared"}
        him_prods = {"2":"Cloud Albedo", "9":"Water Vapor (6.9µm)", "14":"Cloud Top Temperature"}

        # Master mapping of all terminal regions to their respective products
        self.sat_product_codes = {
            # GOES Full Disk & Mesoscales
            "conus": goes_prods, "goes_east_fd": goes_prods, "west_fulldisk": goes_prods,
            "east_meso1": goes_prods, "east_meso2": goes_prods, "west_meso1": goes_prods, "west_meso2": goes_prods,
            
            # GOES Custom North America
            "midwest": goes_prods, "southwest": goes_prods, "northwest": goes_prods, "florida": goes_prods,
            "texas": goes_prods, "great_lakes": goes_prods, "rocky_mountains": goes_prods, "caribbean": goes_prods,
            "central_america": goes_prods, "southeast": goes_prods, "westcoast": goes_prods, "northeast": goes_prods,
            "gulfcoast": goes_prods,
            
            # Himawari
            "himawari": him_prods,
            
            # EUMETSAT 0° (Europe & Africa)
            "europe": eum_prods, "uk_ireland": eum_prods, "scandinavia": eum_prods, "mediterranean": eum_prods,
            "iberian_peninsula": eum_prods, "north_sea": eum_prods, "africa": eum_prods, "west_africa": eum_prods, 
            "sahel": eum_prods, "central_africa": eum_prods, "southern_africa": eum_prods, "north_africa": eum_prods,
            "capeverde": eum_prods, "africa_zoom": eum_prods, "eatlantic": eum_prods, "eumetsat_fd": eum_prods,
            
            # EUMETSAT IODC (41.5°E) - Middle East & East Africa
            "middle_east": eum_prods, "arabian_peninsula": eum_prods, "persian_gulf": eum_prods, "iran": eum_prods,
            "horn_of_africa": eum_prods, "east_africa_iodc": eum_prods, "madagascar": eum_prods, 
            "iodc": eum_prods, "iodc_fd": eum_prods
        }

        # Dynamically populate the combobox with every key from the dictionary
        all_regions = list(self.sat_product_codes.keys())

        # --- Native ttk Import & Bulletproof Dark Styling ---
        from tkinter import ttk
        
        style = ttk.Style()
        
        # Force the 'clam' theme to prevent Linux from overriding our dark colors
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        style.configure("Dark.TCombobox", 
                        fieldbackground="#2B2B2B",  # Dark grey box
                        background="#333333",       # Dark grey dropdown arrow button
                        foreground="white",         # White text
                        bordercolor="#333333",
                        arrowcolor="white")
        
        # Read-only boxes require explicit state mapping to keep the dark background
        style.map("Dark.TCombobox",
                  fieldbackground=[('readonly', '#2B2B2B')],
                  background=[('readonly', '#333333')],
                  foreground=[('readonly', 'white')],
                  selectbackground=[('readonly', '#3932A0')],
                  selectforeground=[('readonly', 'white')])
        
        # 2. Style the actual popup dropdown list
        self.option_add('*TCombobox*Listbox.background', '#1A1A1A')       
        self.option_add('*TCombobox*Listbox.foreground', 'white')         
        self.option_add('*TCombobox*Listbox.selectBackground', '#3932A0') 
        self.option_add('*TCombobox*Listbox.selectForeground', 'white')

        # The Dropdowns
        self.sat_region = ttk.Combobox(f, values=all_regions, state="readonly", style="Dark.TCombobox", width=16)
        self.sat_region.pack(side="left", padx=5)
        self.sat_region.bind("<<ComboboxSelected>>", lambda e: self.update_sat_products(self.sat_region.get()))

        self.sat_product = ttk.Combobox(f, state="readonly", width=35, style="Dark.TCombobox") 
        self.sat_product.pack(side="left", padx=5)
        
        self.glm_switch = ctk.CTkSwitch(f, text="GLM", width=60)
        self.glm_switch.pack(side="left", padx=5)
        
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_sat).pack(side="right", padx=5)

        # Initialize the default state
        self.sat_region.set("conus")
        self.update_sat_products("conus")
        
        # ... rest of method ...

    def update_sat_products(self, selected_region):
        products = self.sat_product_codes.get(selected_region, {"2":"Visible", "9":"Water Vapor", "14":"Infrared"})
        display_values = [f"{code} - {desc}" for code, desc in products.items()]
        self.sat_product.configure(values=display_values)
        
        ir_default = next((val for val in display_values if val.startswith("14")), display_values[-1])
        self.sat_product.set(ir_default)

    def on_run_atl(self):
        map_selection = self.atl_map_type.get()
        cmd_map = {
            "Surface Analysis (MSLP)": "mslp_temp", "300 mb Wind Speed": "wind300",
            "300 mb Divergence/Convergence": "divcon300", "500 mb Wind Speed": "wind500",
            "500 mb Relative Vorticity": "vort500", "700 mb Relative Humidity": "rh700",
            "700 mb Frontogenesis": "fronto700", "850 mb Wind Speed": "wind850",
            "850 mb Dewpoint": "dew850", "850 mb Moisture Advection": "mAdv850",
            "850 mb Temp Advection": "tAdv850"
        }
        subcommand = cmd_map.get(map_selection, "mslp_temp")
        script_name = SCRIPT_FILES.get("atl_maps")
        if script_name and os.path.exists(script_name):
            command = [sys.executable, script_name, subcommand]
            run_script_in_thread(command, self.status_label, self.progress, self.log_textbox, self.preview_label, output_image_name="DYNAMIC_CONUS")
        else:
            messagebox.showerror("Error", f"Script not found: {script_name}")

    def on_run_conus(self):
        map_selection = self.conus_map_type.get()
        cmd_map = {
            "Surface Analysis (MSLP)": "mslp", "300 mb Wind Speed": "wind300",
            "300 mb Divergence/Convergence": "divcon300", "500 mb Wind Speed": "wind500",
            "500 mb Relative Vorticity": "vort500", "500 mb Water Vapor & Vorticity": "500wv",
            "700 mb Relative Humidity": "rh700", "700 mb Frontogenesis": "fronto700",
            "850 mb Wind Speed": "wind850", "850 mb Dewpoint": "dew850",
            "850 mb Moisture Advection": "mAdv850", "850 mb Temp Advection": "tAdv850"
        }
        subcommand = cmd_map.get(map_selection, "mslp")
        script_name = SCRIPT_FILES.get("conus_maps")
        if script_name and os.path.exists(script_name):
            command = [sys.executable, script_name, subcommand]
            run_script_in_thread(command, self.status_label, self.progress, self.log_textbox, self.preview_label, output_image_name="DYNAMIC_CONUS")
        else:
            messagebox.showerror("Error", f"Script not found: {script_name}")
    def on_run_au(self):
        map_selection = self.au_map_type.get()
        domain = self.au_domain.get()
        
        cmd_map = {
            "Surface Analysis (MSLP)": "mslp", "300 mb Wind Speed": "wind300",
            "300 mb Divergence/Convergence": "divcon300", "500 mb Wind Speed": "wind500",
            "500 mb Relative Vorticity": "vort500", "500 mb Water Vapor & Vorticity": "500wv",
            "700 mb Relative Humidity": "rh700", "700 mb Frontogenesis": "fronto700",
            "850 mb Wind Speed": "wind850", "850 mb Dewpoint": "dew850",
            "850 mb Moisture Advection": "mAdv850", "850 mb Temp Advection": "tAdv850"
        }
        
        subcommand = cmd_map.get(map_selection, "mslp")
        script_name = SCRIPT_FILES.get("au_maps")
        
        if script_name and os.path.exists(script_name):
            # Passes both the map type and the target domain to the CLI
            command = [sys.executable, script_name, subcommand, "--domain", domain]
            run_script_in_thread(command, self.status_label, self.progress, self.log_textbox, self.preview_label, output_image_name="DYNAMIC_CONUS")
        else:
            messagebox.showerror("Error", f"Script not found: {script_name}")

    def on_run_sat(self):
        region = self.sat_region.get()
        
        # FIXED: Removed the '1' so it matches your new single dropdown
        product_raw = self.sat_product.get().split(" - ")[0]
        
        output_file = "sat_preview.jpg"
        cmd = [sys.executable, SCRIPT_FILES["sat"], region, product_raw, output_file]
        
        if self.glm_switch.get():
            cmd.append("true")
            
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, output_file)

    def setup_specialty_plots_tab(self):
        tab = self.tab_view.tab("Specialty Plots")
        tab.grid_columnconfigure((0,1), weight=1)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Skew-T", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        
        self.skewt_type = ctk.CTkComboBox(f, values=["Observed", "Forecast"], state="readonly", width=100, command=self.toggle_skewt_fields)
        self.skewt_type.pack(side="left", padx=5)
        self.skewt_type.set("Observed")

        self.skewt_icao = ctk.CTkEntry(f, placeholder_text="ICAO", width=60)
        self.skewt_icao.pack(side="left", padx=5)

        self.skewt_time = ctk.CTkComboBox(f, values=["00Z", "06Z", "12Z", "18Z"], width=70)
        self.skewt_time.pack(side="left", padx=5)
        self.skewt_time.set("12Z")

        # --- HRRR NOW ADDED TO DROPDOWN ---
        self.skewt_model = ctk.CTkComboBox(f, values=["GFS", "HRRR", "NAM", "NAMM", "RAP"], state="readonly", width=80)
        self.skewt_model.pack(side="left", padx=5)
        self.skewt_model.set("NAM")
        self.skewt_model.configure(state="disabled")

        self.skewt_fhr = ctk.CTkEntry(f, placeholder_text="FHR", width=50)
        self.skewt_fhr.pack(side="left", padx=5)
        self.skewt_fhr.configure(state="disabled")

        ctk.CTkButton(f, text="Plot", width=50, command=self.on_run_skewt).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Windrose", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.windrose_icao = ctk.CTkEntry(f, placeholder_text="ICAO"); self.windrose_icao.pack(side="left", padx=5)
        self.windrose_start = ctk.CTkEntry(f, placeholder_text="Start YYYY-MM-DD", width=120); self.windrose_start.pack(side="left", padx=5)
        self.windrose_end = ctk.CTkEntry(f, placeholder_text="End YYYY-MM-DD", width=120); self.windrose_end.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_windrose).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Cross-Sect", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        # ICAO inputs or lat/lon coordinates for start and end points, plus steps
        self.cross_start = ctk.CTkEntry(f, placeholder_text="Start ICAO or lat,lon"); self.cross_start.pack(side="left", padx=5, fill="x", expand=True)
        self.cross_end = ctk.CTkEntry(f, placeholder_text="End ICAO or lat,lon"); self.cross_end.pack(side="left", padx=5, fill="x", expand=True)
        self.cross_steps = ctk.CTkEntry(f, placeholder_text="Steps", width=60); self.cross_steps.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_cross_section).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Storm Tracker", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.trop_storm_id = ctk.CTkEntry(f, placeholder_text="Storm ID (e.g. AL08)", width=150)
        self.trop_storm_id.pack(side="left", padx=5)
        self.trop_action = ctk.CTkComboBox(f, values=["Spaghetti Tracks", "Marine Key Messages"], state="readonly", width=160)
        self.trop_action.pack(side="left", padx=5); self.trop_action.set("Spaghetti Tracks")
        self.trop_surge = ctk.CTkSwitch(f, text="Surge", width=60); self.trop_surge.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_tropical).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="7-Day Outlook", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(f, text="Atlantic Basin Tropical Genesis", text_color="grey").pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_7day_outlook).pack(side="right", padx=5)

        # --- Column 1 Widgets ---
        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Meteogram", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.meteogram_icao = ctk.CTkEntry(f, placeholder_text="ICAO"); self.meteogram_icao.pack(side="left", padx=5, fill="x", expand=True)
        self.meteogram_hours = ctk.CTkEntry(f, placeholder_text="Hours (0=latest)", width=120); self.meteogram_hours.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_meteogram).pack(side="right", padx=5)

        # --- NEW: Forecast Meteogram ---
        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Fcst Meteo", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.fcst_meteo_loc = ctk.CTkEntry(f, placeholder_text="ICAO or lat/lon (e.g. KATL or 34.1/-84.8)"); self.fcst_meteo_loc.pack(side="left", padx=5, fill="x", expand=True)
        self.fcst_meteo_hours = ctk.CTkEntry(f, placeholder_text="Hours (def: 72)", width=120); self.fcst_meteo_hours.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_fcst_meteogram).pack(side="right", padx=5)

        # Astro (Shifted down to row 2)
        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Astro", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.astro_loc = ctk.CTkEntry(f, placeholder_text="ICAO or Lat/Lon (e.g., 34.1/-84.8)"); self.astro_loc.pack(side="left", padx=5, fill="x", expand=True)
        self.astro_time = ctk.CTkEntry(f, placeholder_text="Time HH:MM (opt)", width=120); self.astro_time.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_astro).pack(side="right", padx=5)

    def toggle_skewt_fields(self, choice):
        if choice == "Observed":
            self.skewt_time.configure(state="normal")  # <-- Change this from "readonly"
            self.skewt_model.configure(state="disabled")
            self.skewt_fhr.configure(state="disabled")
            self.skewt_fhr.delete(0, "end")
        else:
            self.skewt_time.configure(state="disabled")
            self.skewt_model.configure(state="readonly")
            self.skewt_fhr.configure(state="normal")

    def on_run_skewt(self):
        obs_type = self.skewt_type.get()
        icao = self.skewt_icao.get().strip()

        if not icao or not self.validate_icao(icao):
            messagebox.showerror("Error", "Valid ICAO required.")
            return

        # Prepare arguments based on the selected type
        if obs_type == "Observed":
            time_val = self.skewt_time.get()
            args = [icao.upper().lstrip('K'), time_val]
        else:
            model = self.skewt_model.get().lower()
            fhr = self.skewt_fhr.get().strip()
            if not fhr or not fhr.isdigit():
                messagebox.showerror("Error", "Forecast Hour (FHR) must be a valid number.")
                return
            args = [icao.upper().lstrip('K'), model, fhr]

        self.log_textbox.insert("end", f"Generating {obs_type} Skew-T for {icao.upper()}...\n")
        self.progress.set(0.3)

        # 1. Background task to do the heavy fetching and math
        def fetch_and_plot():
            try:
                fig = asyncio.run(generate_skewt_plot(args))
                # 2. Hand the figure BACK to the main thread to build the GUI safely
                self.after(0, build_gui, fig)
            except Exception as e:
                self.after(0, handle_error, e)

        def build_gui(fig):
            self.progress.set(1.0)
            if fig is None:
                self.log_textbox.insert("end", "Error: Failed to fetch data or generate plot.\n")
                return

            win = ctk.CTkToplevel(self)
            win.title(f"Skew-T for {icao.upper()} ({obs_type})")
            win.resizable(True, True)                    # Allow free resizing

            # Start with a good initial size (you can change this)
            win.geometry("1600x850")

            # Try to maximize on startup (optional)
            try:
                win.attributes('-zoomed', True)
            except Exception:
                pass

            # Main container
            main_frame = ctk.CTkFrame(win, fg_color="#2B2B2B")
            main_frame.pack(fill="both", expand=True, padx=5, pady=5)

            # Matplotlib canvas - this will scale when the window resizes
            canvas = FigureCanvasTkAgg(fig, main_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side="top", fill="both", expand=True)

            # Toolbar stays at the bottom
            toolbar_frame = ctk.CTkFrame(win, height=40, fg_color="#2B2B2B")
            toolbar_frame.pack(side="bottom", fill="x")

            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.config(background="#2B2B2B")
            for button in toolbar.winfo_children():
                button.config(background="#2B2B2B")
            toolbar.update()

            self.log_textbox.insert("end", "Skew-T plot displayed (fully resizable).\n")

        def handle_error(e):
            self.progress.set(0.0)
            self.log_textbox.insert("end", f"Skew-T Error: {e}\n")

        # 4. Start the background thread
        threading.Thread(target=fetch_and_plot, daemon=True).start()

        def plot():
            try:
                self.log_textbox.insert("end", f"Generating {obs_type} Skew-T for {icao.upper()}...\n")
                self.progress.set(0.3)
                
                # Trigger the async process
                fig = asyncio.run(generate_skewt_plot(args))
                self.progress.set(1.0)
                
                if fig is None:
                    self.log_textbox.insert("end", "Error: Failed to fetch data or generate plot.\n")
                    return
                    
                win = ctk.CTkToplevel(self)
                win.title(f"Skew-T for {icao.upper()} ({obs_type})")
                win.geometry("1400x900") # Larger fallback geometry
                
                # Natively maximize the window for X11/Cinnamon
                try:
                    win.attributes('-zoomed', True)
                except Exception:
                    pass 
                
                # --- INTERACTIVE TOOLBAR SETUP ---
                # Create a frame for the toolbar so it sits nicely at the bottom
                toolbar_frame = ctk.CTkFrame(win, height=40, fg_color="#2B2B2B")
                toolbar_frame.pack(side="bottom", fill="x")
                
                canvas = FigureCanvasTkAgg(fig, win)
                canvas.draw()
                
                # Embed the interactive Matplotlib toolbar
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.config(background="#2B2B2B")
                for button in toolbar.winfo_children():
                    button.config(background="#2B2B2B")
                toolbar.update()
                
                # Pack the canvas into the remaining space
                canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
                
                self.log_textbox.insert("end", "Skew-T plot displayed.\n")
                
            except Exception as e:
                self.progress.set(0.0)
                self.log_textbox.insert("end", f"Skew-T Error: {e}\n")

    def on_run_windrose(self):
        icao = self.windrose_icao.get().strip()
        start = self.windrose_start.get().strip()
        end = self.windrose_end.get().strip()
        if not icao or not start or not end:
            messagebox.showerror("Error", "ICAO, start, and end dates required.")
            return
        cmd = [sys.executable, SCRIPT_FILES["windrose"], icao, start, end]
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, "DYNAMIC_CONUS")

    def on_run_cross_section(self):
        start_loc = self.cross_start.get().strip()
        end_loc = self.cross_end.get().strip()
        steps = self.cross_steps.get().strip()
        
        # Use the updated validation method
        if not self.validate_cross_section_loc(start_loc):
            messagebox.showerror("Error", "Invalid format for Start (requires lat,lon or ICAO).")
            return
        if not self.validate_cross_section_loc(end_loc):
            messagebox.showerror("Error", "Invalid format for End (requires lat,lon or ICAO).")
            return
            
        cmd = [sys.executable, SCRIPT_FILES["cross_section"], start_loc, end_loc]
        if steps:
            if not steps.isdigit():
                messagebox.showerror("Error", "Steps must be a number.")
                return
            cmd.extend(["--steps", steps])
            
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, "cross_section.png")

    def on_run_tropical(self):
        action = self.trop_action.get()
        storm_id = self.trop_storm_id.get().strip().upper()
        show_surge = "true" if self.trop_surge.get() else "false"
        if not storm_id:
            messagebox.showerror("Error", "A valid Storm ID (e.g., AL082025) is required for storm tracking.")
            return
        if action == "Spaghetti Tracks":
            script_name = SCRIPT_FILES.get("trop_atl")
            command = [sys.executable, script_name, "spaghetti_atl", storm_id, show_surge]
            run_script_in_thread(command, self.status_label, self.progress, self.log_textbox, self.preview_label, output_image_name="DYNAMIC_CONUS")
        elif action == "Marine Key Messages":
            script_name = SCRIPT_FILES.get("trop_atl")
            command = [sys.executable, script_name, "marine_key_messages_atl", storm_id]
            run_script_in_thread(command, self.status_label, self.progress, self.log_textbox, self.preview_label, output_image_name=f"outputs/marine_{storm_id}.png")

    def on_run_7day_outlook(self):
        output_name = "seven_day_outlook.png"
        script_name = SCRIPT_FILES.get("trop_outlook")
        if script_name and os.path.exists(script_name):
            command = [sys.executable, script_name, output_name]
            run_script_in_thread(command, self.status_label, self.progress, self.log_textbox, self.preview_label, output_image_name=output_name)
        else:
            messagebox.showerror("Error", f"Script not found: {script_name}")

    def on_run_meteogram(self):
        icao = self.meteogram_icao.get().strip()
        hours = self.meteogram_hours.get().strip() or "0"
        if not icao or not hours.isdigit():
            messagebox.showerror("Error", "Valid ICAO and hours required.")
            return
        cmd = [sys.executable, SCRIPT_FILES["main_cli"], "meteogram", icao, hours]
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, "DYNAMIC_CONUS")

    def on_run_fcst_meteogram(self):
        loc = self.fcst_meteo_loc.get().strip()
        hours = self.fcst_meteo_hours.get().strip()
        
        # Relaxed validation to allow ICAO (no slash required)
        if not loc:
            messagebox.showerror("Error", "Valid ICAO or lat/lon required (e.g., KATL or 34.1/-84.8).")
            return
            
        cmd = [sys.executable, SCRIPT_FILES["fcst_meteogram"], loc]
        if hours.isdigit():
            cmd.extend(["--hoursforward", hours])
            
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, "DYNAMIC_CONUS")

    def on_run_astro(self):
        loc = self.astro_loc.get().strip()
        time_str = self.astro_time.get().strip()
        
        if not loc:
            messagebox.showerror("Error", "Location required.")
            return

        # 1. Format the location string gracefully for astro.py
        parsed_loc = loc
        # If user inputs lat/lon like "34.12, -84.85", convert it to "34.12/-84.85"
        parts = re.split(r'[,\s]+', loc.strip())
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                parsed_loc = f"{lat}/{lon}"
            except ValueError:
                pass # Keeps original loc string intact if it's an ICAO

        script_name = SCRIPT_FILES.get("astro", "astro.py")
        cmd = [sys.executable, script_name, parsed_loc]

        # 2. Append the time strictly as the second positional argument
        if time_str:
            if not re.match(r'^([01]?[0-9]|2[0-3]):([0-5][0-9])$', time_str):
                messagebox.showerror("Error", "Time must be HH:MM.")
                return
            cmd.append(time_str)

        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, "astro_plot.png")

    def setup_utilities_tab(self):
        tab = self.tab_view.tab("Utilities")
        tab.grid_columnconfigure((0,1), weight=1)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="METAR", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.metar_icao = ctk.CTkEntry(f, placeholder_text="ICAO"); self.metar_icao.pack(side="left", fill="x", expand=True, padx=5)
        self.metar_hours = ctk.CTkEntry(f, placeholder_text="Hours", width=80); self.metar_hours.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_metar).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="TAF", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.taf_icao = ctk.CTkEntry(f, placeholder_text="ICAO"); self.taf_icao.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_taf).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="AIRMET/SIGMET", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.mets_icao = ctk.CTkEntry(f, placeholder_text="ICAO"); self.mets_icao.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_mets).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Radar", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.radar_region = ctk.CTkComboBox(f, values=["chase","ne","se","sw","nw","pr"], state="readonly"); self.radar_region.pack(side="left", padx=5); self.radar_region.set("chase")
        self.radar_overlay = ctk.CTkComboBox(f, values=["base","totals"], state="readonly"); self.radar_overlay.pack(side="left", padx=5); self.radar_overlay.set("base")
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_radar).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="NWS Alerts", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.alerts_state = ctk.CTkEntry(f, placeholder_text="State (GA)"); self.alerts_state.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_alerts).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="NWS Forecast", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.forecast_loc = ctk.CTkEntry(f, placeholder_text="City, ZIP, ICAO, lat/lon"); self.forecast_loc.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_forecast).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="World Times", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkButton(f, text="Show", width=60, command=self.on_run_utc).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab, fg_color="transparent"); f.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Convert", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.convert_val = ctk.CTkEntry(f, placeholder_text="Value", width=80); self.convert_val.pack(side="left", padx=5)
        self.convert_from = ctk.CTkEntry(f, placeholder_text="From", width=80); self.convert_from.pack(side="left", padx=5)
        self.convert_to = ctk.CTkEntry(f, placeholder_text="To", width=80); self.convert_to.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_convert).pack(side="right", padx=5)

    def on_run_metar(self):
        icao = self.metar_icao.get().strip()
        hours = self.metar_hours.get().strip()
        if not icao or not self.validate_icao(icao):
            messagebox.showerror("Error", "Valid ICAO required.")
            return
        cmd = [sys.executable, SCRIPT_FILES["main_cli"], "metar", icao]
        if hours and re.match(r'^\d+$', hours):
            cmd.extend(["--hoursback", hours])
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label)

    def on_run_taf(self):
        icao = self.taf_icao.get().strip()
        if not icao or not self.validate_icao(icao):
            messagebox.showerror("Error", "Valid ICAO required.")
            return
        cmd = [sys.executable, SCRIPT_FILES["main_cli"], "taf", icao]
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label)

    def on_run_mets(self):
        icao = self.mets_icao.get().strip()
        if not icao or not self.validate_icao(icao):
            messagebox.showerror("Error", "Valid ICAO required.")
            return
        cmd = [sys.executable, SCRIPT_FILES["main_cli"], "mets", icao]
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label)

    def on_run_radar(self):
        region = self.radar_region.get()
        overlay = self.radar_overlay.get()
        cmd = [sys.executable, SCRIPT_FILES["main_cli"], "radar", region, overlay]
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, f"radar_{region}_{overlay}.gif")

    def on_run_alerts(self):
        state = self.alerts_state.get().strip()
        if not state or len(state) != 2:
            messagebox.showerror("Error", "Valid 2-letter state required.")
            return
        cmd = [sys.executable, SCRIPT_FILES["main_cli"], "alerts", state.upper()]
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label)

    def on_run_forecast(self):
        loc = self.forecast_loc.get().strip()
        if not loc:
            messagebox.showerror("Error", "Location required.")
            return
        cmd = [sys.executable, SCRIPT_FILES["main_cli"], "forecast", loc]
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label)

    def on_run_utc(self):
        cmd = [sys.executable, SCRIPT_FILES["main_cli"], "utc"]
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label)

    def on_run_convert(self):
        val = self.convert_val.get().strip()
        from_unit = self.convert_from.get().strip()
        to_unit = self.convert_to.get().strip()
        if not val or not from_unit or not to_unit:
            messagebox.showerror("Error", "Value, from, and to units required.")
            return
        cmd = [sys.executable, SCRIPT_FILES["main_cli"], "convert", val, from_unit, to_unit]
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label)

if __name__ == "__main__":
    missing = [f for k, f in SCRIPT_FILES.items() if not os.path.exists(f)]
    if missing:
        print("Warning: Missing scripts, GUI may fail for these:")
        for m in missing:
            print(f" - {m}")
    app = WeatherLauncherApp()
    app.mainloop()
