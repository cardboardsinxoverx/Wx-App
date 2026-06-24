import matplotlib
# Force Agg backend to prevent windows popping up
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import requests
import io
import os
import textwrap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import qrcode
from astral import LocationInfo
from astral.sun import sun
from scipy.interpolate import make_interp_spline

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

# AUTOMATIC WINDOWS PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(SCRIPT_DIR, "icons")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
LOGO_PATH = os.path.join(SCRIPT_DIR, "boxlogo2.png")

# JACKSONVILLE COORDINATES
LAT = "30.4941"
LON = "-81.6879"
STATION_ID = "KJAX"
CITY_NAME = "JACKSONVILLE"

FIG_BG_COLOR = '#121212'
PANEL_BG_COLOR = '#1e1e1e' 
TODAY_BG_COLOR = '#222222'
ACCENT_GREEN = '#7FFF00'
ACCENT_RED = '#FF4500'
ACCENT_BLUE = '#00BFFF'

WIND_DIRS = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
}

# Ensure output exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_font(size):
    # Windows standard font fallbacks
    options = ["arialbd.ttf", "seguiemj.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf"]
    for font_name in options:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()

def create_silver_gloss_header():
    height, width = 256, 1
    stops = [(0.0, (0.2,0.2,0.2)), (0.3, (0.7,0.7,0.7)), (0.5, (1.0,1.0,1.0)), (0.7, (0.7,0.7,0.7)), (1.0, (0.2,0.2,0.2))]
    gradient = np.zeros((height, width, 3))
    y_points = np.linspace(0, 1, height)
    for i in range(len(stops) - 1):
        pos1, c1 = stops[i]
        pos2, c2 = stops[i+1]
        mask = (y_points >= pos1) & (y_points <= pos2)
        local_t = (y_points[mask] - pos1) / (pos2 - pos1)
        for ch in range(3):
            gradient[mask, 0, ch] = c1[ch] * (1 - local_t) + c2[ch] * local_t
    return np.transpose(gradient, (1, 0, 2))

def create_glass_panel():
    h, w = 150, 400
    arr = np.zeros((h, w, 4))
    arr[:, :, 0:3] = 1.0 
    arr[:, :, 3] = 0.12  
    return arr

def create_frosted_overlay():
    h, w = 150, 400
    arr = np.zeros((h, w, 4))
    arr[:, :, 0:3] = 0.9 
    arr[:, :, 3] = 0.3    
    return arr

def create_gradient_text_image(text, size, gradient_type='hot'):
    font_size = int(size * 3) 
    font = get_font(font_size)
    stroke_w = 4 
    
    dummy = Image.new("RGBA", (1,1))
    draw_dummy = ImageDraw.Draw(dummy)
    
    bbox = draw_dummy.textbbox((0,0), text, font=font, stroke_width=stroke_w, align='center')
    w = int((bbox[2]-bbox[0]) + 60)
    h = int((bbox[3]-bbox[1]) + 60)
    
    if w <= 0 or h <= 0: return None
    
    txt_x = (w - (bbox[2]-bbox[0])) / 2 - bbox[0]
    txt_y = (h - (bbox[3]-bbox[1])) / 2 - bbox[1]

    shadow = Image.new("RGBA", (w, h), (0,0,0,0))
    draw_shadow = ImageDraw.Draw(shadow)
    draw_shadow.text((txt_x + 5, txt_y + 5), text, fill="black", font=font, stroke_width=stroke_w, stroke_fill="black", align='center')
    shadow = shadow.filter(ImageFilter.GaussianBlur(3))

    outline = Image.new("RGBA", (w, h), (0,0,0,0))
    draw_outline = ImageDraw.Draw(outline)
    draw_outline.text((txt_x, txt_y), text, fill="black", font=font, stroke_width=stroke_w, stroke_fill="black", align='center')

    grad = Image.new("RGBA", (w, h))
    draw_grad = ImageDraw.Draw(grad)
    
    if gradient_type == 'hot': c_top, c_bot = (255, 69, 0), (255, 215, 0)
    elif gradient_type == 'cold': c_top, c_bot = (0, 0, 139), (0, 255, 255)
    elif gradient_type == 'green': c_top, c_bot = (127, 255, 0), (0, 100, 0)
    elif gradient_type == 'white': c_top, c_bot = (255, 255, 255), (180, 180, 180)
    else: c_top, c_bot = (80, 80, 80), (255, 255, 255)

    for y in range(h):
        t = y / h
        r = int(c_top[0] * (1-t) + c_bot[0] * t)
        g = int(c_top[1] * (1-t) + c_bot[1] * t)
        b = int(c_top[2] * (1-t) + c_bot[2] * t)
        draw_grad.line([(0, y), (w, y)], fill=(r,g,b), width=1)

    mask = Image.new("L", (w, h), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.text((txt_x, txt_y), text, fill=255, font=font, align='center')
    
    grad_text = Image.new("RGBA", (w, h), (0,0,0,0))
    grad_text.paste(grad, (0,0), mask=mask)
    
    combined = Image.alpha_composite(shadow, outline)
    final = Image.alpha_composite(combined, grad_text)
    
    return np.array(final) / 255.0

# ... [Reusing logic from your original script for icons logic] ...
# To save space, I am including the critical fetch/plot logic below 
# assuming the user has the ICON folder.

def load_icon_file(filename):
    # Try multiple extensions
    candidates = [filename]
    if not filename.endswith(".png"): candidates.append(filename + ".png")
    
    for f in candidates:
        path = os.path.join(ICON_DIR, f)
        if os.path.exists(path):
            try:
                pil_img = Image.open(path).convert("RGBA")
                return np.array(pil_img) / 255.0
            except: pass
    return None

def get_current_metar():
    try:
        # Changed to dynamic station ID
        r = requests.get(f"https://api.weather.gov/stations/{STATION_ID}/observations/latest", headers={"User-Agent": "(jax_fcst)"}, timeout=12)
        if r.status_code != 200: return {"temp": None, "condition": "N/A"} 
        d = r.json()["properties"]
        t_c = d["temperature"]["value"]
        if t_c is not None: t_f = round(t_c*1.8+32)
        else: t_f = None
        
        return {
            "temp": t_f,
            "condition": d["textDescription"] or "N/A",
            "wind_spd": round(d["windSpeed"]["value"]*1.94384) if d["windSpeed"]["value"] else 0,
            "wind_dir": d["windDirection"]["value"],
            "humidity": d.get("relativeHumidity", {}).get("value"),
            "barometer": round(d["barometricPressure"]["value"]/3386.39, 2) if d.get("barometricPressure", {}).get("value") else None
        }
    except: return {"temp": None, "condition": "Data Unavailable"}

def get_forecast_data():
    try:
        headers = {"User-Agent": "(jax_fcst)"}
        # 1. Get Grid Points
        r = requests.get(f"https://api.weather.gov/points/{LAT},{LON}", headers=headers, timeout=10)
        grid_url = r.json()["properties"]["forecast"]
        hourly_url = r.json()["properties"]["forecastHourly"]
        
        # 2. Get 7-Day
        r_daily = requests.get(grid_url, headers=headers, timeout=10)
        daily_periods = r_daily.json()["properties"]["periods"]
        
        # 3. Get Hourly (for charts)
        r_hourly = requests.get(hourly_url, headers=headers, timeout=10)
        hourly_periods = r_hourly.json()["properties"]["periods"]
        
        # Process Hourly
        hourly_map = {}
        for p in hourly_periods:
            dt_obj = datetime.fromisoformat(p['startTime'])
            d_key = dt_obj.date()
            if d_key not in hourly_map: hourly_map[d_key] = {"hours": [], "temps": [], "precip": []}
            hourly_map[d_key]["hours"].append(dt_obj.hour)
            hourly_map[d_key]["temps"].append(p['temperature'])
            hourly_map[d_key]["precip"].append(p['probabilityOfPrecipitation']['value'] or 0)
            
        return daily_periods, hourly_map
    except Exception as e:
        print(f"Error fetching forecast: {e}")
        return None, None

def draw_full_graphic():
    print("Fetching Jacksonville data...")
    periods, hourly_data = get_forecast_data()
    live_now = get_current_metar()
    
    if not periods:
        print("Failed to fetch forecast.")
        return

    print("Generating Graphic...")
    fig = plt.figure(figsize=(16, 9), facecolor=FIG_BG_COLOR, dpi=100)
    
    # --- HEADER ---
    ax_header = fig.add_axes([0.0, 0.85, 1.0, 0.15])
    ax_header.axis('off')
    ax_header.imshow(create_silver_gloss_header(), aspect='auto', extent=[0, 1, 0, 1])
    
    # Logo
    if os.path.exists(LOGO_PATH):
        logo_img = plt.imread(LOGO_PATH)
        ax_logo = fig.add_axes([0.015, 0.86, 0.08, 0.12], zorder=10)
        ax_logo.axis('off')
        ax_logo.imshow(logo_img)
        
    # Title
    t_img = create_gradient_text_image(f"{CITY_NAME}, FL", 26, 'green')
    ax_header.add_artist(AnnotationBbox(OffsetImage(t_img, zoom=0.5), (0.28, 0.65), frameon=False))
    
    sub_img = create_gradient_text_image("7-DAY FORECAST", 18, 'green')
    ax_header.add_artist(AnnotationBbox(OffsetImage(sub_img, zoom=0.45), (0.28, 0.30), frameon=False))

    # QR Code
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(f"https://forecast.weather.gov/MapClick.php?lat={LAT}&lon={LON}")
    qr.make(fit=True)
    ax_qr = fig.add_axes([0.56, 0.86, 0.08, 0.13], zorder=10)
    ax_qr.axis('off')
    ax_qr.imshow(np.array(qr.make_image(fill_color="white", back_color="transparent").convert("RGBA")) / 255.0)

    # Current Conditions Panel
    ax_glass = fig.add_axes([0.66, 0.86, 0.33, 0.13], zorder=10)
    ax_glass.axis('off')
    ax_glass.imshow(create_glass_panel(), aspect='auto', extent=[0,1,0,1])
    
    curr_temp_img = create_gradient_text_image(f"{live_now['temp']}F", 20, 'green')
    ax_glass.add_artist(AnnotationBbox(OffsetImage(curr_temp_img, zoom=0.5), (0.2, 0.5), frameon=False))
    
    ax_glass.text(0.5, 0.7, f"Wind: {live_now['wind_spd']} kt", color='white', fontweight='bold')
    ax_glass.text(0.5, 0.4, f"Baro: {live_now['barometer']} inHg", color='white', fontweight='bold')
    ax_glass.text(0.5, 0.1, f"Cond: {live_now['condition']}", color='cyan', fontsize=8, fontweight='bold')

    # --- PANELS ---
    # Simplified logic to grab Day/Night pairs from the flat NWS list
    days_processed = 0
    idx = 0
    w = 1.0/7
    
    while days_processed < 7 and idx < len(periods) - 1:
        p1 = periods[idx]
        p2 = periods[idx+1]
        
        # Check if p1 is day or night
        if p1['isDaytime']:
            day_data = p1
            night_data = p2
            idx += 2
        else:
            # It's night already (e.g., run late at night)
            day_data = None 
            night_data = p1
            idx += 1
            
        ax_panel = fig.add_axes([days_processed*w, 0.05, w, 0.78])
        ax_panel.set_xlim(0,1); ax_panel.set_ylim(0,1); ax_panel.axis('off')
        
        bg_col = TODAY_BG_COLOR if days_processed == 0 else PANEL_BG_COLOR
        ax_panel.add_patch(mpatches.Rectangle((0,0),1,1, color=bg_col))
        
        # Day Label
        if days_processed == 0: lbl = "TODAY"
        elif day_data: lbl = day_data['name'].split()[0].upper()[:3]
        else: lbl = "TDY"
            
        l_img = create_gradient_text_image(lbl, 16, 'green')
        ax_panel.add_artist(AnnotationBbox(OffsetImage(l_img, zoom=0.5), (0.5, 0.95), frameon=False))
        
        # Data
        if day_data:
            # High Temp
            h_img = create_gradient_text_image(f"{day_data['temperature']}", 22, 'hot')
            ax_panel.add_artist(AnnotationBbox(OffsetImage(h_img, zoom=0.5), (0.5, 0.8), frameon=False))
            # Icon (Simple search)
            icn = day_data['shortForecast'].split()[0].lower() # simple match
            if "rain" in day_data['shortForecast'].lower(): icn = "rain"
            if "sunny" in day_data['shortForecast'].lower(): icn = "sunny"
            if "cloud" in day_data['shortForecast'].lower(): icn = "cloudy"
            
            img_arr = load_icon_file(f"{icn}@2x.png")
            if img_arr is None: img_arr = load_icon_file("na@2x.png")
            
            if img_arr is not None:
                ax_panel.add_artist(AnnotationBbox(OffsetImage(img_arr, zoom=0.25), (0.5, 0.6), frameon=False))

        if night_data:
            l_img = create_gradient_text_image(f"{night_data['temperature']}", 16, 'cold')
            ax_panel.add_artist(AnnotationBbox(OffsetImage(l_img, zoom=0.5), (0.5, 0.45), frameon=False))
            
        # Divider
        if days_processed > 0:
            ax_panel.plot([0,0],[0.05,0.95], color='#444444', lw=1)
            
        days_processed += 1

    path = os.path.join(OUTPUT_DIR, f"jax_7day_{datetime.now():%Y%m%d}.png")
    plt.savefig(path, facecolor=FIG_BG_COLOR)
    print(f"Saved: {path}")
    plt.close()

if __name__ == "__main__":
    draw_full_graphic()