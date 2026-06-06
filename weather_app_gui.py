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
from skewt import generate_skewt_plot

# --- Configuration ---
SCRIPT_FILES = {
    # State Maps (Moved to subfolder)
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
    # --- ADDED NEW STATES ---
    "sc_wx": "state maps/sc_wx.py",
    "ca_wx": "state maps/ca_wx.py",
    "tn_wx": "state maps/tn_wx.py",
    "va_wx": "state maps/va_wx.py",
    "wv_wx": "state maps/wv_wx.py",
    "ga_ndfd_fcst": "state maps/ga_wx_fcst.py",
    
    # Other Maps (Assume they are in the root)
    "awa_wx": "awa_wx.py",
    "atl_maps": "atl_weather_maps.py",
    "conus_maps": "weather_maps.py",
    "eu_maps": "eu_weather_maps.py",
    "au_maps": "au_weather_maps.py",
    
    # Specialty Plots (Assume in root)
    "skewt": "skewt.py",
    "windrose": "windrose.py",
    "cross_section": "cross_section.py",
    
    # Satellite
    "sat": "sat.py",
    
    # Main CLI
    "main_cli": "frostbyte.py"
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
                            final_output_match = re.findall(r'(?:Map Saved:|Map Saved to:|Generated Successfully at:)\s*(.*\.png)', log_text)
                            if final_output_match:
                                final_output_image_name = final_output_match[-1].strip()
                        except: pass
                            
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
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.tab_view = ctk.CTkTabview(self, width=880, height=200)
        self.tab_view.pack(padx=10, pady=10, fill="x")
        self.tab_view.add("State Maps")
        self.tab_view.add("Regional Maps")
        self.tab_view.add("Specialty Plots")
        self.tab_view.add("Utilities")

        self.preview_frame = ctk.CTkFrame(self, height=400)
        self.preview_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.preview_frame.grid_propagate(False)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Map preview will appear here", text_color="grey")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        self.progress = ctk.CTkProgressBar(self, mode="determinate", height=16, progress_color="#00ff00")
        self.progress.pack(padx=10, pady=5, fill="x")
        self.progress.set(0)

        self.status_label = ctk.CTkLabel(self, text="Ready", anchor="w", font=ctk.CTkFont(size=12))
        self.status_label.pack(padx=10, pady=(0, 5), fill="x")

        self.log_textbox = ctk.CTkTextbox(
            self,
            height=150,
            fg_color="#0d1117",
            text_color="#c9d1d9",
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
            border_width=2,
            border_color="#30363d"
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

    def setup_state_maps_tab(self):
        tab = self.tab_view.tab("State Maps")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure((0, 1, 2), weight=1) 

        frame = ctk.CTkFrame(tab)
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(frame, text="States", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        inner = ctk.CTkFrame(frame)
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

        frame_ndfd = ctk.CTkFrame(tab)
        frame_ndfd.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(frame_ndfd, text="NDFD Forecasts", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        inner_ndfd = ctk.CTkFrame(frame_ndfd)
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

        f = ctk.CTkFrame(tab); f.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="AWA", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", command=lambda: self.run_simple_script("awa_wx", "awa_weather.png")).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="ATL", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", command=lambda: self.run_simple_script("atl_maps", "atl_weather.png")).pack(side="right", padx=5)

        # --- UPDATED CONUS UI (ALL UPPER AIR MAPS ADDED) ---
        f = ctk.CTkFrame(tab); f.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="CONUS", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        
        self.conus_map_type = ctk.CTkComboBox(f, values=[
            "Surface Analysis (MSLP)",
            "300 mb Wind Speed",
            "300 mb Divergence/Convergence",
            "500 mb Wind Speed",
            "500 mb Relative Vorticity",
            "700 mb Relative Humidity",
            "700 mb Frontogenesis",
            "850 mb Wind Speed",
            "850 mb Dewpoint",
            "850 mb Moisture Advection",
            "850 mb Temp Advection"
        ], state="readonly", width=250)
        self.conus_map_type.pack(side="left", padx=5)
        self.conus_map_type.set("Surface Analysis (MSLP)")
        
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_conus).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="EU", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", command=lambda: self.run_simple_script("eu_maps", "eu_weather.png")).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="AU", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", command=lambda: self.run_simple_script("au_maps", "au_weather.png")).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Satellite", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        
        self.sat_region = ctk.CTkComboBox(f, values=[
            "conus", "west_fulldisk", "mesosector", "mesosector2", "goes_east_fd", "eumetsat_fd",
            "europe", "africa", "eatlantic", "iodc", 
            "southeast", "westcoast", "northeast", "gulfcoast", "capeverde"
        ], state="readonly"); self.sat_region.pack(side="left", padx=5); self.sat_region.set("conus")
        
        self.sat_product = ctk.CTkComboBox(f, values=["2", "9", "14"], state="readonly"); self.sat_product.pack(side="left", padx=5); self.sat_product.set("14")
        self.glm_switch = ctk.CTkSwitch(f, text="GLM Overlay"); self.glm_switch.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_sat).pack(side="right", padx=5)

    def on_run_conus(self):
        map_selection = self.conus_map_type.get()
        cmd_map = {
            "Surface Analysis (MSLP)": "mslp",
            "300 mb Wind Speed": "wind300",
            "300 mb Divergence/Convergence": "divcon300",
            "500 mb Wind Speed": "wind500",
            "500 mb Relative Vorticity": "vort500",
            "700 mb Relative Humidity": "rh700",
            "700 mb Frontogenesis": "fronto700",
            "850 mb Wind Speed": "wind850",
            "850 mb Dewpoint": "dew850",
            "850 mb Moisture Advection": "mAdv850",
            "850 mb Temp Advection": "tAdv850"
        }
        subcommand = cmd_map.get(map_selection, "mslp")
        script_name = SCRIPT_FILES.get("conus_maps")
        
        if script_name and os.path.exists(script_name):
            command = [sys.executable, script_name, subcommand]
            run_script_in_thread(command, self.status_label, self.progress, self.log_textbox, self.preview_label, output_image_name="DYNAMIC_CONUS")
        else:
            messagebox.showerror("Error", f"Script not found: {script_name}")

    def on_run_sat(self):
        region = self.sat_region.get()
        product = self.sat_product.get()
        output_file = "sat_preview.jpg"
        cmd = [sys.executable, SCRIPT_FILES["sat"], region, product, output_file]
        if self.glm_switch.get():
            cmd.append("true")
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, output_file)

    def setup_specialty_plots_tab(self):
        tab = self.tab_view.tab("Specialty Plots")
        tab.grid_columnconfigure((0,1), weight=1)

        f = ctk.CTkFrame(tab); f.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Skew-T", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.skewt_icao = ctk.CTkEntry(f, placeholder_text="ICAO"); self.skewt_icao.pack(side="left", padx=5, fill="x", expand=True)
        self.skewt_time = ctk.CTkComboBox(f, values=["00", "12"], state="readonly", width=60); self.skewt_time.pack(side="left", padx=5); self.skewt_time.set("00")
        self.skewt_fhr = ctk.CTkEntry(f, placeholder_text="FHR", width=80); self.skewt_fhr.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Plot", width=660, command=self.on_run_skewt).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Windrose", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.windrose_icao = ctk.CTkEntry(f, placeholder_text="ICAO"); self.windrose_icao.pack(side="left", padx=5)
        self.windrose_start = ctk.CTkEntry(f, placeholder_text="Start YYYY-MM-DD", width=120); self.windrose_start.pack(side="left", padx=5)
        self.windrose_end = ctk.CTkEntry(f, placeholder_text="End YYYY-MM-DD", width=120); self.windrose_end.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_windrose).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Cross-Section", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.cross_start = ctk.CTkEntry(f, placeholder_text="Start lat,lon"); self.cross_start.pack(side="left", padx=5, fill="x", expand=True)
        self.cross_end = ctk.CTkEntry(f, placeholder_text="End lat,lon"); self.cross_end.pack(side="left", padx=5, fill="x", expand=True)
        self.cross_steps = ctk.CTkEntry(f, placeholder_text="Steps (opt)", width=80); self.cross_steps.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_cross_section).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Meteogram", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.meteogram_icao = ctk.CTkEntry(f, placeholder_text="ICAO"); self.meteogram_icao.pack(side="left", padx=5, fill="x", expand=True)
        self.meteogram_hours = ctk.CTkEntry(f, placeholder_text="Hours (0=latest)", width=120); self.meteogram_hours.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_meteogram).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Astro", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.astro_loc = ctk.CTkEntry(f, placeholder_text="City/State"); self.astro_loc.pack(side="left", padx=5, fill="x", expand=True)
        self.astro_time = ctk.CTkEntry(f, placeholder_text="Time HH:MM (opt)", width=120); self.astro_time.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Generate", width=60, command=self.on_run_astro).pack(side="right", padx=5)

    def on_run_skewt(self):
        icao = self.skewt_icao.get().strip()
        time_model = self.skewt_time.get()
        fhr = self.skewt_fhr.get().strip()
        if not icao or not time_model or not self.validate_icao(icao):
            messagebox.showerror("Error", "Valid ICAO and Time required.")
            return
        
        args = [icao.upper().lstrip('K'), time_model + 'Z']
        
        if fhr and re.match(r'^\d+$', fhr):
            args.append(fhr)

        def plot():
            try:
                self.log_textbox.insert("end", f"Generating Skew-T for {icao}...\n")
                self.progress.set(0.3)
                fig = asyncio.run(generate_skewt_plot(args))
                self.progress.set(1.0)
                if fig is None:
                    messagebox.showerror("Error", "Failed to generate Skew-T plot.")
                    return
                win = ctk.CTkToplevel(self)
                win.title(f"Skew-T for {icao.upper()}")
                win.geometry("1200x800")
                canvas = FigureCanvasTkAgg(fig, win)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                self.log_textbox.insert("end", "Skew-T plot displayed.\n")
            except Exception as e:
                self.progress.set(0.0)
                self.log_textbox.insert("end", f"Skew-T Error: {e}\n")
                messagebox.showerror("Skew-T Error", str(e))
        threading.Thread(target=plot, daemon=True).start()

    def on_run_windrose(self):
        icao = self.windrose_icao.get().strip()
        start = self.windrose_start.get().strip()
        end = self.windrose_end.get().strip()
        if not icao or not start or not end:
            messagebox.showerror("Error", "ICAO, start, and end dates required.")
            return
        cmd = [sys.executable, SCRIPT_FILES["windrose"], icao, start, end]
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, "windrose.png")

    def on_run_cross_section(self):
        start_loc = self.cross_start.get().strip()
        end_loc = self.cross_end.get().strip()
        steps = self.cross_steps.get().strip()
        
        if not self.validate_lat_lon(start_loc):
            messagebox.showerror("Error", "Invalid format for Start (lat,lon).")
            return
            
        if not self.validate_lat_lon(end_loc):
            messagebox.showerror("Error", "Invalid format for End (lat,lon).")
            return
            
        cmd = [sys.executable, SCRIPT_FILES["cross_section"], start_loc, end_loc]
        
        if steps:
            if not steps.isdigit():
                messagebox.showerror("Error", "Steps must be a number.")
                return
            cmd.extend(["--steps", steps])
            
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, "cross_section.png")

    def on_run_meteogram(self):
        icao = self.meteogram_icao.get().strip()
        hours = self.meteogram_hours.get().strip() or "0"
        if not icao or not hours.isdigit():
            messagebox.showerror("Error", "Valid ICAO and hours required.")
            return
        cmd = [sys.executable, SCRIPT_FILES["main_cli"], "meteogram", icao, hours]
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, f"meteogram_{icao.upper()}.png")

    def on_run_astro(self):
        loc = self.astro_loc.get().strip()
        time_str = self.astro_time.get().strip()
        if not loc:
            messagebox.showerror("Error", "Location required.")
            return
        cmd = [sys.executable, "astro.py", loc]
        if time_str:
            if not re.match(r'^\d{2}:\d{2}$', time_str):
                messagebox.showerror("Error", "Time must be HH:MM.")
                return
            cmd.append(time_str)
        run_script_in_thread(cmd, self.status_label, self.progress, self.log_textbox, self.preview_label, "astro_plot.png")

    def setup_utilities_tab(self):
        tab = self.tab_view.tab("Utilities")
        tab.grid_columnconfigure((0,1), weight=1)

        f = ctk.CTkFrame(tab); f.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="METAR", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.metar_icao = ctk.CTkEntry(f, placeholder_text="ICAO"); self.metar_icao.pack(side="left", fill="x", expand=True, padx=5)
        self.metar_hours = ctk.CTkEntry(f, placeholder_text="Hours", width=80); self.metar_hours.pack(side="left", padx=5)
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_metar).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="TAF", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.taf_icao = ctk.CTkEntry(f, placeholder_text="ICAO"); self.taf_icao.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_taf).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="AIRMET/SIGMET", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.mets_icao = ctk.CTkEntry(f, placeholder_text="ICAO"); self.mets_icao.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_mets).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="Radar", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.radar_region = ctk.CTkComboBox(f, values=["chase","ne","se","sw","nw","pr"], state="readonly"); self.radar_region.pack(side="left", padx=5); self.radar_region.set("chase")
        self.radar_overlay = ctk.CTkComboBox(f, values=["base","totals"], state="readonly"); self.radar_overlay.pack(side="left", padx=5); self.radar_overlay.set("base")
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_radar).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="NWS Alerts", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.alerts_state = ctk.CTkEntry(f, placeholder_text="State (GA)"); self.alerts_state.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_alerts).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="NWS Forecast", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.forecast_loc = ctk.CTkEntry(f, placeholder_text="City, ZIP, ICAO, lat/lon"); self.forecast_loc.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f, text="Run", width=60, command=self.on_run_forecast).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text="World Times", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkButton(f, text="Show", width=60, command=self.on_run_utc).pack(side="right", padx=5)

        f = ctk.CTkFrame(tab); f.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
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
