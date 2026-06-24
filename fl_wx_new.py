import matplotlib
# Force Agg backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import numpy as np
import requests
import io
import os
import textwrap
import math
import re
import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import qrcode
from astral import LocationInfo
from astral.sun import sun
from scipy.interpolate import make_interp_spline

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

FIG_BG_COLOR = '#121212'
PANEL_BG_COLOR = '#1e1e1e' 
TODAY_BG_COLOR = '#222222'
ACCENT_CYAN = '#00E5FF'
ACCENT_GREEN = '#7FFF00'
ACCENT_RED = '#FF4500'
ACCENT_BLUE = '#00BFFF'

# USER PATHS (Matches your Rome script)
LOGO_PATH = "/home/desoxyn/frostbyte/frostbyte_project/boxlogo2.png"
OUTPUT_DIR = "/home/desoxyn/frostbyte/frostbyte_project/output"
ICON_DIR = "/home/desoxyn/frostbyte/frostbyte_project/icons/new icons"

# JACKSONVILLE SETTINGS
LAT = "30.4941"
LON = "-81.6879"
STATION_ID = "KJAX" 
CITY_NAME = "JACKSONVILLE"

WIND_DIRS = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
}

# ==========================================
# PART 1: GRAPHICS ENGINES
# ==========================================

def get_font(size):
    options = ["DejaVuSans-Bold.ttf", "arialbd.ttf", "FreeSansBold.ttf", "LiberationSans-Bold.ttf"]
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
    noise = np.random.normal(0, 0.05, (h, w))
    arr[:, :, 3] = np.clip(arr[:, :, 3] + noise, 0.05, 0.25)
    return arr

def create_frosted_overlay():
    h, w = 150, 400
    arr = np.zeros((h, w, 4))
    arr[:, :, 0:3] = 0.9 
    arr[:, :, 3] = 0.3    
    noise = np.random.normal(0, 0.1, (h, w))
    arr[:, :, 3] = np.clip(arr[:, :, 3] + noise, 0.1, 0.5)
    return arr

def create_gradient_text_image(text, size, gradient_type='hot'):
    font_size = int(size * 3) 
    font = get_font(font_size)
    stroke_w = 8 
    
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
    draw_shadow.text((txt_x + 10, txt_y + 10), text, fill="black", font=font, stroke_width=stroke_w, stroke_fill="black", align='center')
    shadow = shadow.filter(ImageFilter.GaussianBlur(5))

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

# ==========================================
# PART 2: ICONS & GRAPHICS
# ==========================================

def create_base_fig(size=(1,1)):
    fig, ax = plt.subplots(figsize=size, dpi=100)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    return fig, ax

def render_fig_to_numpy(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)

def create_pop_badge(pop_val, holographic=False):
    fig, ax = create_base_fig()
    path_data = [
        (Path.MOVETO, [0.50, 0.98]),
        (Path.CURVE4, [0.42, 0.75, 0.12, 0.55, 0.12, 0.35]),
        (Path.CURVE4, [0.12, 0.05, 0.88, 0.05, 0.88, 0.35]),
        (Path.CURVE4, [0.88, 0.55, 0.58, 0.75, 0.50, 0.98]),
    ]
    verts = []; codes = []
    for cmd, coords in path_data:
        points = list(zip(coords[0::2], coords[1::2]))
        verts.extend(points); codes.extend([cmd] * len(points))
    path = Path(verts, codes)
    
    patch = patches.PathPatch(path, facecolor='#00BFFF', edgecolor='black', lw=4, zorder=1)
    ax.add_patch(patch)

    if holographic and pop_val > 60:
        for i in range(5):
            s = 0.8 - (i*0.1)
            shim = mpatches.Circle((0.5, 0.35), s/2, color=['#ff00ff', '#00ffff', '#ffff00'][i%3], alpha=0.15, zorder=1+i*0.1)
            patch.set_clip_path(path, transform=patch.get_transform()) 
            ax.add_patch(shim)
        ax.add_patch(mpatches.Rectangle((0.48, 0.4), 0.04, 0.3, color='white', alpha=0.6, angle=20, zorder=2))

    ax.text(0.5, 0.28, f"{int(pop_val)}%", color='white', fontsize=20, fontweight='bold', 
            ha='center', va='center', zorder=3, 
            path_effects=[pe.withStroke(linewidth=4, foreground='black')])
            
    return render_fig_to_numpy(fig)

def create_uv_badge(uv_index):
    if uv_index < 0.5: return None
    uv_idx = int(round(uv_index))
    colors = {2: "#00ff00", 5: "#ffff00", 7: "#ff8000", 10: "#ff0000", 100: "#990099"}
    col = "#990099"
    for lim, c in sorted(colors.items()):
        if uv_idx <= lim: col = c; break
    fig, ax = create_base_fig(size=(0.8,0.8))
    ax.add_patch(mpatches.Circle((0.5,0.5), 0.4, color=col, ec='white', lw=2))
    ax.text(0.5,0.5, str(uv_idx), color='black', fontsize=40, fontweight='bold', ha='center', va='center')
    ax.text(0.5,0.8, "UV", color='black', fontsize=12, fontweight='bold', ha='center', va='center')
    return render_fig_to_numpy(fig)

def create_3d_wind_arrow(direction_deg):
    fig, ax = plt.subplots(figsize=(1,1), dpi=120)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.add_patch(mpatches.Rectangle((0.45, 0.1), 0.1, 0.6, color='white', ec='#333', lw=1))
    ax.add_patch(mpatches.Polygon([[0.5,0.9],[0.25,0.6],[0.75,0.6]], color='white', ec='#333', lw=1))
    ax.add_patch(mpatches.Rectangle((0.47, 0.12), 0.04, 0.55, color='#ffffff88'))
    img = render_fig_to_numpy(fig)
    pil = Image.fromarray((img*255).astype('uint8'))
    rotated = pil.rotate(-direction_deg, resample=Image.BICUBIC, expand=False)
    return np.array(rotated)/255.0

def create_moon_icon(phase):
    fig, ax = create_base_fig()
    ax.add_patch(mpatches.Circle((0.5, 0.5), 0.45, color='#333333'))
    if phase == "full": ax.add_patch(mpatches.Circle((0.5, 0.5), 0.45, color='#FFFFDD'))
    elif phase == "first_quarter": ax.add_patch(mpatches.Wedge((0.5, 0.5), 0.45, 270, 90, color='#FFFFDD'))
    elif phase == "third_quarter": ax.add_patch(mpatches.Wedge((0.5, 0.5), 0.45, 90, 270, color='#FFFFDD'))
    elif phase == "waxing_crescent": 
        ax.add_patch(mpatches.Wedge((0.5, 0.5), 0.45, 270, 90, color='#FFFFDD'))
        ax.add_patch(mpatches.Ellipse((0.5, 0.5), 0.60, 0.90, color='#333333'))
    elif phase == "waning_crescent":
        ax.add_patch(mpatches.Wedge((0.5, 0.5), 0.45, 90, 270, color='#FFFFDD'))
        ax.add_patch(mpatches.Ellipse((0.5, 0.5), 0.60, 0.90, color='#333333'))
    elif phase == "waxing_gibbous":
        ax.add_patch(mpatches.Wedge((0.5, 0.5), 0.45, 270, 90, color='#FFFFDD'))
        ax.add_patch(mpatches.Ellipse((0.5, 0.5), 0.60, 0.90, color='#FFFFDD'))
    elif phase == "waning_gibbous":
        ax.add_patch(mpatches.Wedge((0.5, 0.5), 0.45, 90, 270, color='#FFFFDD'))
        ax.add_patch(mpatches.Ellipse((0.5, 0.5), 0.60, 0.90, color='#FFFFDD'))
    return render_fig_to_numpy(fig)

def remove_background(img):
    try:
        img = img.convert("RGBA")
        datas = img.getdata()
        bg_color = datas[0]
        new_data = []
        tolerance = 40 
        for item in datas:
            diff = sum([abs(item[i] - bg_color[i]) for i in range(3)])
            if diff < tolerance: new_data.append((255, 255, 255, 0)) 
            else: new_data.append(item)
        img.putdata(new_data)
        return img
    except: return img

def load_icon_file(filename):
    candidates = [filename]
    if not filename.endswith(".png") and not filename.endswith(".jpg"):
        candidates.append(filename + ".png"); candidates.append(filename + ".jpg")
    base = os.path.splitext(filename)[0]
    candidates.append(base + ".jpg"); candidates.append(base + ".png")

    for f in candidates:
        path = os.path.join(ICON_DIR, f)
        if os.path.exists(path):
            try:
                pil_img = Image.open(path)
                if f.endswith(".jpg") or "prain" in f or "mrain" in f:
                    pil_img = remove_background(pil_img)
                return np.array(pil_img.convert("RGBA")) / 255.0
            except: pass
    return None

def create_split_icon(img_left, img_right):
    if img_left is None or img_right is None: return img_left if img_left is not None else img_right
    rows, cols, ch = img_left.shape
    def place_icon(img, corner):
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        new_size = (int(cols * 0.55), int(rows * 0.55))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
        layer = Image.new("RGBA", (cols, rows), (0,0,0,0))
        if corner == 'top_left': layer.paste(pil_img, (2, 2))
        else: layer.paste(pil_img, (cols - new_size[0] - 2, rows - new_size[1] - 2))
        return np.array(layer) / 255.0
    left_layer = place_icon(img_left, 'top_left')
    right_layer = place_icon(img_right, 'bottom_right')
    combined = left_layer + right_layer
    y, x = np.ogrid[:rows, :cols]
    center_diag = (cols + rows) / 2.0
    dist_from_diag = np.abs((x + y) - center_diag)
    is_outline = dist_from_diag < 5.0; is_line = dist_from_diag < 3.0
    combined[is_outline, 0:3] = 0.0; combined[is_outline, 3] = 1.0
    combined[is_line, 0:3] = 1.0; combined[is_line, 3] = 1.0
    return combined

def _get_single_icon_name(text, is_day=True, pop=0):
    t = text.lower()
    if not t or "unknown" in t or t == "up" or " n/a" in t: return "na@2x.png"
    if "rain" in t and "snow" in t and "sleet" not in t and "freezing" not in t: return "rainandsnow@2x.png" if is_day else "rainandsnown@2x.png"
    if "sleet" in t and "snow" in t: return "sleetsnow@2x.png" if is_day else "sleetsnown@2x.png"
    if "freezing" in t and "drizzle" in t: return "fdrizzle@2x.png" if is_day else "fdrizzlen@2x.png"
    if "wintry" in t or "mix" in t or ("rain" in t and "sleet" in t): return "wintrymix@2x.png" if is_day else "wintrymixn@2x.png"
    if "snow" in t and "shower" in t:
        if pop < 25: return "pcloudys@2x.png" if is_day else "pcloudysn@2x.png"
        elif pop < 55: return "mcloudys@2x.png" if is_day else "mcloudysn@2x.png"
        else: return "snowshowers@2x.png" if is_day else "snowshowersn@2x.png"
    if "flurries" in t:
        if pop < 25: return "pcloudysf@2x.png" if is_day else "pcloudysfn@2x.png"
        elif pop < 55: return "mcloudysf@2x.png" if is_day else "mcloudysfn@2x.png"
        else: return "flurries@2x.png" if is_day else "flurriesn@2x.png"
    if "snow" in t:
        if pop < 25: return "pcloudys@2x.png" if is_day else "pcloudysn@2x.png"
        elif pop < 55: return "mcloudys@2x.png" if is_day else "mcloudysn@2x.png"
        else: return "snow@2x.png" if is_day else "snown@2x.png"
    if "sleet" in t or "ice" in t: return "sleet@2x.png"
    if "freezing rain" in t: return "freezingrain@2x.png"
    if "thunder" in t or "storm" in t:
        if pop < 25: return "pcloudyt@2x.png" if is_day else "pcloudytn@2x.png" 
        elif pop < 55: return "mcloudyt@2x.png" if is_day else "mcloudytn@2x.png" 
        else: return "tstorm@2x.png" if is_day else "tstormn@2x.png" 
    if "shower" in t:
        if pop < 25: return "pcloudyr@2x.png" if is_day else "pcloudyrn@2x.png"
        elif pop < 55: return "mcloudyr@2x.png" if is_day else "mcloudyrn@2x.png"
        else: return "showers@2x.png" if is_day else "showersn@2x.png"
    if "rain" in t or "drizzle" in t:
        if pop < 25: return "prain@2x.png" if is_day else "prainn@2x.png"
        elif pop < 75: return "mrain@2x.png" if is_day else "mrainn@2x.png"
        else: return "rain@2x.png" if is_day else "rainn@2x.png"
    if "overcast" in t or "cloudy" in t:
        if "partly" in t: return "pcloudy@2x.png" if is_day else "pcloudyn@2x.png"
        if "mostly" in t: return "mcloudy@2x.png" if is_day else "mcloudyn@2x.png"
        return "cloudy@2x.png"
    if "fog" in t or "mist" in t: return "fog@2x.png"
    if "sunny" in t or "clear" in t or "fair" in t:
        if "partly" in t: return "pcloudy@2x.png" if is_day else "pcloudyn@2x.png"
        if "mostly" in t: return "fair@2x.png" if is_day else "clearn@2x.png"
        return "sunny@2x.png" if is_day else "clearn@2x.png"
    return "na@2x.png"

def get_label_for_precip(t, pop):
    t = t.lower()
    if "rain" in t and "snow" in t: return "RAIN & SNOW"
    if "sleet" in t and "snow" in t: return "SLEET & SNOW"
    if "freezing" in t and "drizzle" in t: return "FRZ. DRIZZLE"
    if "snow" in t and "shower" in t:
        if pop < 25: return "ISO. SNOW SHWRS"
        elif pop < 55: return "SCT. SNOW SHWRS"
        elif pop < 75: return "NUM. SNOW SHWRS"
        else: return "SNOW SHOWERS"
    if "flurries" in t:
        if pop < 25: return "ISO. FLURRIES"
        elif pop < 55: return "SCT. FLURRIES"
        else: return "FLURRIES"
    if "snow" in t:
        if pop < 25: return "SLGT CHC SNOW"
        elif pop < 55: return "CHC SNOW"
        elif pop < 75: return "LIKELY SNOW"
        else: return "SNOW"
    if "thunder" in t or "storm" in t:
        if pop < 25: return "ISO. STORMS"
        elif pop < 55: return "SCT. STORMS"
        elif pop < 75: return "NUM. STORMS"
        else: return "STORMS"
    if "shower" in t:
        if pop < 25: return "ISO. SHOWERS"
        elif pop < 55: return "SCT. SHOWERS"
        elif pop < 75: return "NUM. SHOWERS"
        else: return "SHOWERS"
    if "rain" in t or "drizzle" in t:
        if pop < 25: return "SLGT CHC RAIN"
        elif pop < 55: return "CHC RAIN"
        elif pop < 75: return "LIKELY RAIN"
        else: return "RAIN"
    return None

def get_icon_by_text(text, is_day=True, pop=0):
    t_clean = text.lower().replace("chance ", "").replace("slight ", "")
    split_keywords = [" then ", " becoming ", " until ", " likely ", " and "]
    found_split = None
    if "rain" in t_clean and "snow" in t_clean: found_split = None 
    else:
        for kw in split_keywords:
            if kw in t_clean: found_split = kw; break
    if found_split:
        parts = t_clean.split(found_split)
        if len(parts) >= 2:
            name_1 = _get_single_icon_name(parts[0], is_day, pop)
            name_2 = _get_single_icon_name(parts[1], is_day, pop)
            if name_1 and name_2 and name_1 != name_2 and "na@2x" not in name_1 and "na@2x" not in name_2:
                img_1 = load_icon_file(name_1)
                img_2 = load_icon_file(name_2)
                if img_1 is not None and img_2 is not None:
                    def clean_lbl(s, p):
                        precip_lbl = get_label_for_precip(s, p)
                        if precip_lbl: return precip_lbl
                        if "partly" in s: return "P. SUNNY" if is_day else "P. CLOUDY"
                        if "mostly" in s and "cloud" in s: return "M. CLOUDY"
                        if "mostly" in s and "sun" in s: return "M. SUNNY"
                        if "fog" in s: return "FOG"
                        if "cloud" in s or "overcast" in s: return "CLOUDY"
                        if "clear" in s or "sunny" in s: return "SUNNY" if is_day else "CLEAR"
                        return s.upper().split()[0]
                    lbl = f"{clean_lbl(parts[0], pop)} -> {clean_lbl(parts[1], pop)}"
                    return create_split_icon(img_1, img_2), lbl
    fname = _get_single_icon_name(t_clean, is_day, pop)
    custom_label = get_label_for_precip(t_clean, pop)
    if fname and "na@2x" in fname:
        if "unknown" in t_clean or "up" in t_clean: custom_label = "UNKNOWN PRECIP"
        else: custom_label = "N/A"
    img = load_icon_file(fname)
    if img is None: img = load_icon_file("na@2x.png") 
    if img is None: img = np.zeros((100, 100, 4))
    return img, custom_label

def process_logo(path):
    if not os.path.exists(path): return None
    try:
        img = Image.open(path).convert("RGBA")
        stroke_size = 6; shadow_blur = 12; shadow_offset = (8, 8); padding = 40; shadow_color = (34, 139, 34, 255) 
        w, h = img.size; new_w, new_h = w + padding*2, h + padding*2
        original_layer = Image.new("RGBA", (new_w, new_h), (0,0,0,0)); original_layer.paste(img, (padding, padding))
        alpha = original_layer.split()[-1]; grown_shape = alpha.filter(ImageFilter.MaxFilter(stroke_size * 2 + 1))
        outline_layer = Image.new("RGBA", (new_w, new_h), (0,0,0,255)); outline_layer.putalpha(grown_shape)
        shadow_base = Image.new("RGBA", (new_w, new_h), shadow_color); shadow_base.putalpha(grown_shape)
        shadow_layer = shadow_base.filter(ImageFilter.GaussianBlur(shadow_blur))
        final = Image.new("RGBA", (new_w, new_h), (0,0,0,0))
        final.paste(shadow_layer, shadow_offset, shadow_layer); final.paste(outline_layer, (0,0), outline_layer); final.paste(original_layer, (0,0), original_layer)
        return np.array(final) / 255.0
    except: return plt.imread(path)

# ==========================================
# PART 3: ASTRONOMY & DATA
# ==========================================

def get_accurate_moon_phase(date_obj):
    new_moon_ref = datetime(2025, 1, 29); days = (date_obj.replace(hour=0,minute=0,second=0,microsecond=0) - new_moon_ref.replace(hour=0,minute=0,second=0,microsecond=0)).total_seconds() / 86400; phase = days % 29.53059
    if phase < 1.5: return "New Moon"
    elif phase < 5.5: return "Waxing Crescent"
    elif phase < 9.5: return "First Quarter"
    elif phase < 13.5: return "Waxing Gibbous"
    elif phase < 16.5: return "Full Moon"
    elif phase < 20.5: return "Waning Gibbous"
    elif phase < 24.5: return "Last Quarter"
    else: return "Waning Crescent"

def get_sun_times(lat_str, lon_str, date_obj):
    try:
        loc = LocationInfo(CITY_NAME, "Florida", "US/Eastern", float(LAT), float(LON)); s = sun(loc.observer, date=date_obj.date(), tzinfo=loc.timezone)
        return f"{s['sunrise'].strftime('%I:%M %p').lstrip('0')} ↑ {s['sunset'].strftime('%I:%M %p').lstrip('0')} ↓"
    except: return "Sun Data N/A"

def get_current_metar():
    try:
        r = requests.get(f"https://api.weather.gov/stations/{STATION_ID}/observations/latest", headers={"User-Agent": "(jax_fcst)"}, timeout=12)
        if r.status_code != 200: return {"temp": None, "condition": "N/A"} 
        d = r.json()["properties"]
        t_c, d_c, w_ms, w_gust_ms = d["temperature"]["value"], d["dewpoint"]["value"], d["windSpeed"]["value"], d.get("windGust", {}).get("value")
        p_pa, v_m, chill_c = d.get("barometricPressure", {}).get("value"), d.get("visibility", {}).get("value"), d.get("windChill", {}).get("value")
        raw_cond = d["textDescription"]
        if raw_cond is None: final_cond = "N/A"
        elif raw_cond == "Unknown Precipitation": final_cond = "Unknown Precip"
        else: final_cond = raw_cond
        return {"temp": round(t_c*1.8+32) if t_c is not None else None, "dew": round(d_c*1.8+32) if d_c is not None else None, "wind_dir": d["windDirection"]["value"], "wind_spd": round(w_ms*1.94384) if w_ms else 0, "wind_gust": round(w_gust_ms*1.94384) if w_gust_ms else None, "condition": final_cond, "humidity": d.get("relativeHumidity", {}).get("value"), "barometer": round(p_pa/3386.39, 2) if p_pa else None, "visibility": round(v_m/1609.34, 1) if v_m else None, "wind_chill": round(chill_c*1.8+32) if chill_c is not None else None}
    except: return {"temp": None, "condition": "Data Unavailable"}

def get_extra_open_meteo_data():
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&current=us_aqi,uv_index,alder_pollen,birch_pollen,grass_pollen,mugwort_pollen,olive_pollen,ragweed_pollen&timezone=auto"
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return {}
        d = r.json().get('current', {})
        pollens = {k: v for k,v in d.items() if 'pollen' in k and v is not None}
        pollen_text = "Low Pollen"
        if pollens:
            max_p = max(pollens, key=pollens.get); val = pollens[max_p]
            if val > 10: pollen_text = f"High: {max_p.replace('_pollen','').capitalize()}"
            elif val > 5: pollen_text = f"Mod: {max_p.replace('_pollen','').capitalize()}"
        return {"aqi": d.get("us_aqi"), "uv": d.get("uv_index"), "pollen": pollen_text}
    except: return {}

def get_hourly_forecast():
    try:
        headers = {"User-Agent": "(jax_fcst)"}
        # Step 1: Get the Grid Points for JAX (Dynamic)
        pt_r = requests.get(f"https://api.weather.gov/points/{LAT},{LON}", headers=headers, timeout=10)
        pt_r.raise_for_status()
        hourly_url = pt_r.json()["properties"]["forecastHourly"]
        
        # Step 2: Fetch Hourly Data
        r = requests.get(hourly_url, headers=headers, timeout=15)
        if r.status_code != 200: return {}
        
        hourly_map = {}
        for p in r.json()['properties']['periods']:
            dt = datetime.fromisoformat(p['startTime'].replace("Z", "+00:00")); date_key = dt.date()
            if date_key not in hourly_map: 
                hourly_map[date_key] = {"hours": [], "temps": [], "precip": [], "dew": [], "app_temp": []} 
            hourly_map[date_key]["hours"].append(dt.hour); hourly_map[date_key]["temps"].append(p['temperature'])
            precip_val = p.get('probabilityOfPrecipitation', {}).get('value', 0) or 0
            hourly_map[date_key]["precip"].append(precip_val)
            dew = p.get('dewpoint', {}).get('value', 0)
            if dew: hourly_map[date_key]["dew"].append(round(dew*1.8+32))
            else: hourly_map[date_key]["dew"].append(None)
            t = p['temperature']
            hourly_map[date_key]["app_temp"].append(t)
        return hourly_map
    except Exception as e: 
        print(f"Hourly fetch failed: {e}")
        return {}

def get_weather_data():
    try:
        headers = {"User-Agent": "(jax_fcst)"}
        r = requests.get(f"https://api.weather.gov/points/{LAT},{LON}", headers=headers, timeout=10); r.raise_for_status()
        forecast_url = r.json()["properties"]["forecast"]
        r = requests.get(forecast_url, headers=headers, timeout=10); periods = r.json()["properties"]["periods"]
        daily = {}; current_day_name = datetime.now().strftime("%A") 
        for p in periods:
            name = p["name"]; is_day = p["isDaytime"]
            is_today_match = (name == current_day_name or name == current_day_name + " Night" or name in ["Today", "Tonight", "This Afternoon", "Overnight", "Rest of Tonight", "This Morning"])
            day_key = "Today" if is_today_match else name.replace(" Night", "")
            if day_key not in daily: daily[day_key] = {"day": None, "night": None}
            pop_val = p["probabilityOfPrecipitation"]["value"] or 0
            icon, custom_lbl = get_icon_by_text(p["shortForecast"], is_day=is_day, pop=pop_val)
            final_text = custom_lbl if custom_lbl else p["shortForecast"]
            data_packet = {
                "temp": p["temperature"], "text": final_text, "icon_img": icon, 
                "wind_spd": p["windSpeed"], "wind_dir": p["windDirection"], "pop": pop_val
            }
            if is_day: daily[day_key]["day"] = data_packet
            else: daily[day_key]["night"] = data_packet
        ordered_days = []
        na_icon = load_icon_file("na@2x.png")
        for i in range(7):
            key = "Today" if i == 0 else (datetime.now() + timedelta(days=i)).strftime("%A")
            if key not in daily:
                for d_key in daily:
                    if key in d_key: key = d_key; break
            day_record = daily.get(key, {"day": None, "night": None})
            if day_record["day"] is None: day_record["day"] = {"temp": "--", "text": "FCST PENDING", "icon_img": na_icon, "wind_spd": "", "wind_dir": "", "pop": 0}
            if day_record["night"] is None: day_record["night"] = {"temp": "--", "text": "FCST PENDING", "icon_img": na_icon, "wind_spd": "", "wind_dir": "", "pop": 0}
            try: astro_text = get_sun_times(LAT, LON, datetime.now() + timedelta(days=i))
            except: astro_text = "Sun Data N/A"
            ordered_days.append({"day_label": "TODAY" if i == 0 else key[:3].upper(), "astro_text": astro_text, "day_data": day_record["day"], "night_data": day_record["night"]})
        return {}, ordered_days
    except Exception as e: print(f"Error: {e}"); return None, []

def draw_hourly_curve(ax, hours, temps):
    if len(temps) < 3: return
    # Position: SHIFTED HIGHER (0.20 start) with taller height (0.15)
    chart_ax = ax.inset_axes([0.05, 0.20, 0.9, 0.15], transform=ax.transAxes); chart_ax.set_facecolor('#00000000') 
    chart_ax.grid(True, color='#444444', linestyle='--', linewidth=0.5, alpha=0.5)
    line_effects = [pe.withStroke(linewidth=3, foreground='white')]; text_effects = [pe.withStroke(linewidth=2.5, foreground='white')]
    for spine_name, spine in chart_ax.spines.items():
        if spine_name in ['top', 'right']: spine.set_visible(False)
        else: spine.set_visible(True); spine.set_linewidth(1.0); spine.set_edgecolor('black'); spine.set_path_effects(line_effects)
    chart_ax.tick_params(axis='both', colors='black', labelsize=6, width=1.0, length=2, pad=1)
    for label in chart_ax.get_xticklabels(): label.set_fontweight('bold'); label.set_path_effects(text_effects)
    for label in chart_ax.get_yticklabels(): label.set_fontweight('bold'); label.set_path_effects(text_effects)
    for tick in chart_ax.xaxis.get_major_ticks() + chart_ax.yaxis.get_major_ticks(): tick.tick1line.set_path_effects(line_effects); tick.tick2line.set_path_effects(line_effects)
    y_min, y_max = min(temps) - 2, max(temps) + 2; chart_ax.set_ylim(y_min, y_max); chart_ax.set_xlim(min(hours), max(hours))
    chart_ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True)); chart_ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    try: x_new = np.linspace(min(hours), max(hours), 100); spl = make_interp_spline(hours, temps, k=3); y_new = spl(x_new)
    except: x_new, y_new = hours, temps
    chart_ax.plot(x_new, y_new, color=ACCENT_GREEN, linewidth=2, alpha=1.0); chart_ax.plot(x_new, y_new, color=ACCENT_GREEN, linewidth=6, alpha=0.3)
    h_idx, l_idx = np.argmax(temps), np.argmin(temps)
    chart_ax.plot(hours[h_idx], temps[h_idx], 'o', color=ACCENT_RED, markersize=4, zorder=10, markeredgecolor='white', markeredgewidth=0.5)
    chart_ax.plot(hours[l_idx], temps[l_idx], 'o', color=ACCENT_BLUE, markersize=4, zorder=10, markeredgecolor='white', markeredgewidth=0.5)

def add_reflection(ax, icon_img, x, y):
    if icon_img is None: return
    try:
        refl = np.flipud(icon_img); rows = refl.shape[0]; alpha_mask = np.linspace(0.3, 0.0, rows).reshape(-1, 1)
        if refl.shape[2] == 4: refl = refl.copy(); refl[:, :, 3] = refl[:, :, 3] * alpha_mask[:, 0]; ax.add_artist(AnnotationBbox(OffsetImage(refl, zoom=0.28), (x, y - 0.08), frameon=False, zorder=5))
    except: pass

def draw_precip_bars(ax, precip_data, hours):
    # Position: TALLER (Height 0.15), ALIGNED X-AXIS, STYLED
    bar_ax = ax.inset_axes([0.05, 0.02, 0.9, 0.15], transform=ax.transAxes)
    bar_ax.set_facecolor('#00000000')
    
    if not precip_data: precip_data = [0] * len(hours)
    
    if len(hours) != len(precip_data):
        x = range(len(precip_data)); bar_ax.set_xlim(0, len(precip_data))
    else:
        x = hours; bar_ax.set_xlim(min(hours), max(hours))
        
    vals = [min(p, 100) for p in precip_data]
    
    # Bold Axis Lines (White with stroke)
    line_effects = [pe.withStroke(linewidth=3, foreground='white')]
    text_effects = [pe.withStroke(linewidth=2.5, foreground='white')]

    for spine_name, spine in bar_ax.spines.items():
        if spine_name in ['top', 'right']: 
            spine.set_visible(False)
        else: 
            spine.set_visible(True); spine.set_linewidth(1.5); spine.set_edgecolor('white') 
            spine.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])
            
    # Tick styling - Visible Y ticks and X ticks (RESTORED & STYLED)
    bar_ax.tick_params(axis='x', colors='black', width=1.5, length=3, labelsize=5, labelbottom=True, pad=1)
    bar_ax.tick_params(axis='y', colors='black', width=1.5, length=3, labelsize=5, labelleft=True, pad=1)
    
    for label in bar_ax.get_xticklabels() + bar_ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_path_effects(text_effects)

    bar_ax.set_yticks([25, 50, 75])
    bar_ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    bar_ax.grid(axis='y', linestyle=':', alpha=0.3, color='white')

    bar_ax.bar(x, vals, color='#00BFFF', alpha=0.7, width=0.8)
    bar_ax.set_ylim(0, 100)

# ==========================================
# STATIC PARTICLE LOGIC
# ==========================================

def add_particles(ax, text, count=30):
    t = text.lower()
    # Z-ORDER CHANGED: 6 -> 1 (Sit behind text)
    if "blowing snow" in t:
        for _ in range(count):
            x = np.random.uniform(0, 1); y = np.random.uniform(0.2, 0.8)
            ax.plot([x, x+0.1], [y, y-0.01], color='white', lw=1, alpha=0.5, zorder=1)
    elif "snow" in t or "flurries" in t or "sleet" in t:
        c = 'white' if "snow" in t else '#aaffff' 
        ax.scatter(np.random.rand(count), np.random.rand(count)*0.7+0.2, s=15, c=c, marker='*', edgecolors='none', zorder=1, alpha=0.7)
    elif "thunder" in t or "storm" in t:
        for _ in range(count):
            x = np.random.uniform(0.1, 0.9); y_top = np.random.uniform(0.3, 0.8)
            length = np.random.uniform(0.05, 0.15); y_bot = y_top - length
            ax.plot([x, x - 0.02], [y_top, y_bot], color='#00E5FF', lw=1.5, alpha=np.random.uniform(0.4, 0.8), zorder=1)
        for _ in range(3):
            bx = np.random.uniform(0.2, 0.8); by = np.random.uniform(0.5, 0.8)
            bolt_x = [bx, bx-0.02, bx+0.01, bx-0.01]; bolt_y = [by, by-0.05, by-0.10, by-0.15]
            # Lightning slightly higher (2) but still behind text
            ax.plot(bolt_x, bolt_y, color='#FFD700', lw=2, zorder=2, alpha=0.9, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    elif "rain" in t or "shower" in t or "drizzle" in t:
        col = '#00E5FF'
        if "freezing" in t: col = '#E0FFFF' 
        for _ in range(count):
            x = np.random.uniform(0.1, 0.9); y_top = np.random.uniform(0.3, 0.8)
            length = np.random.uniform(0.05, 0.15); y_bot = y_top - length
            ax.plot([x, x - 0.02], [y_top, y_bot], color=col, lw=1.5, alpha=np.random.uniform(0.4, 0.8), zorder=1)

def add_today_glow(ax):
    glow = mpatches.Circle((0.5, 0.5), 0.48, color='#00ffea', alpha=0.15, transform=ax.transAxes, zorder=-1)
    ax.add_patch(glow)
    glow2 = mpatches.Circle((0.5, 0.5), 0.42, color='#00ffea', alpha=0.25, transform=ax.transAxes, zorder=-1)
    ax.add_patch(glow2)

# ==========================================
# MAIN RENDERER
# ==========================================

def draw_full_graphic():
    print("Fetching Jacksonville Data...")
    curr, data = get_weather_data()
    if not data: 
        print("Failed to fetch forecast.")
        return

    # Dynamic hourly fetch
    hourly_data = get_hourly_forecast()
    extras = get_extra_open_meteo_data()
    live_now = get_current_metar()
    
    # Generate Moon phases
    moon_phases = [get_accurate_moon_phase(datetime.now() + timedelta(days=i)) for i in range(7)]
    
    print("Generating Graphic...")
    fig = plt.figure(figsize=(16, 9), facecolor=FIG_BG_COLOR, dpi=130)
    
    # --- HEADER ---
    ax_header = fig.add_axes([0.0, 0.85, 1.0, 0.15]); ax_header.axis('off'); ax_header.imshow(create_silver_gloss_header(), aspect='auto', extent=[0, 1, 0, 1])
    if os.path.exists(LOGO_PATH):
        # SHRINK LOGO
        ax_logo = fig.add_axes([0.015, 0.86, 0.08, 0.12], zorder=10); ax_logo.axis('off')
        logo_data = process_logo(LOGO_PATH); 
        if logo_data is not None: ax_logo.imshow(logo_data)
        
    center_x = 0.28
    # Title changed to JACKSONVILLE
    title_img = create_gradient_text_image(f"{CITY_NAME}, FL", 26, 'green'); ax_header.add_artist(AnnotationBbox(OffsetImage(title_img, zoom=0.5), (center_x, 0.65), frameon=False))
    sub_img = create_gradient_text_image("7-DAY FORECAST", 18, 'green'); ax_header.add_artist(AnnotationBbox(OffsetImage(sub_img, zoom=0.45), (center_x, 0.30), frameon=False))
    qr = qrcode.QRCode(box_size=4, border=1); qr.add_data(f"https://forecast.weather.gov/MapClick.php?lat={LAT}&lon={LON}"); qr.make(fit=True)
    ax_qr = fig.add_axes([0.56, 0.86, 0.08, 0.13], zorder=10); ax_qr.axis('off'); ax_qr.imshow(np.array(qr.make_image(fill_color="white", back_color="#00000000").convert("RGBA")) / 255.0)
    
    # --- GLASS PANEL ---
    ax_glass = fig.add_axes([0.66, 0.86, 0.33, 0.13], zorder=10); ax_glass.axis('off'); ax_glass.imshow(create_glass_panel(), aspect='auto', extent=[0, 1, 0, 1])
    
    # Frosted Overlay
    if live_now['temp'] and live_now['temp'] <= 32: ax_glass.imshow(create_frosted_overlay(), aspect='auto', extent=[0, 1, 0, 1], zorder=11)
    
    # Particles (Rain/Snow in Header)
    if live_now['condition']: add_particles(ax_glass, live_now['condition'], count=25)

    # HEAD WEATHER ICON (CENTERED: 0.48)
    now = datetime.now()
    try:
        loc = LocationInfo(CITY_NAME, "Florida", "US/Eastern", float(LAT), float(LON)); s = sun(loc.observer, date=now.date(), tzinfo=loc.timezone)
        is_day_now = s['sunrise'].replace(tzinfo=None) < now < s['sunset'].replace(tzinfo=None)
    except: is_day_now = 6 <= now.hour < 20
        
    head_icon_name = _get_single_icon_name(live_now['condition'], is_day=is_day_now)
    head_icon = load_icon_file(head_icon_name)
    if head_icon is not None:
        pil_icon = Image.fromarray((head_icon * 255).astype(np.uint8))
        target_height = 75
        dynamic_zoom = target_height / pil_icon.height
        ax_glass.add_artist(AnnotationBbox(OffsetImage(head_icon, zoom=dynamic_zoom), (0.48, 0.5), frameon=False, zorder=12))

    # TEMP & SKY (LEFT ALIGNED)
    t_str = f"{live_now['temp']}°F" if live_now['temp'] else "--°F"; t_img = create_gradient_text_image(t_str, 18, 'green'); ax_glass.add_artist(AnnotationBbox(OffsetImage(t_img, zoom=0.45), (0.15, 0.65), frameon=False, zorder=12))
    c_img = create_gradient_text_image(live_now['condition'][:20], 14, 'white'); ax_glass.add_artist(AnnotationBbox(OffsetImage(c_img, zoom=0.35), (0.15, 0.30), frameon=False, zorder=12))
    
    # TWO-COLUMN STATS LAYOUT (FAR RIGHT)
    col1 = []
    if live_now.get('wind_spd') is not None:
        d = round(live_now['wind_dir'] or 0); wdir = list(WIND_DIRS.keys())[int((d + 11.25) % 360 // 22.5)]
        col1.append(f"Wind: {wdir} {live_now['wind_spd']}")
    if live_now.get('wind_chill'): col1.append(f"Chill: {live_now['wind_chill']}°F")
    if live_now.get('barometer'): col1.append(f"Baro: {live_now['barometer']}")

    col2 = []
    if live_now.get('humidity'): col2.append(f"Hum: {int(round(live_now['humidity']))}%")
    if extras.get('aqi'): col2.append(f"AQI: {extras['aqi']}")
    if extras.get('pollen'): col2.append(f"Pol: {extras['pollen'].split(':')[-1].strip()}") # Shorten
    col2.append(f"Upd: {datetime.now().strftime('%I:%M %p')}")

    stat_effects = [pe.withStroke(linewidth=2, foreground='black'), pe.SimpleLineShadow(offset=(1.0, -1.0), shadow_color='black', alpha=0.5), pe.Normal()]
    
    y = 0.80
    for txt in col1:
        ax_glass.text(0.65, y, txt, color=ACCENT_GREEN, fontsize=7, fontweight='bold', path_effects=stat_effects, zorder=12, ha='left')
        y -= 0.18
        
    y = 0.80
    for txt in col2:
        c = ACCENT_GREEN
        if "AQI" in txt:
            val = extras.get('aqi', 0)
            if val > 50: c = '#FFFF00'
            if val > 100: c = '#FF7E00'
        if "Pol" in txt: c = '#FF7E00'
        
        ax_glass.text(0.82, y, txt, color=c, fontsize=7, fontweight='bold', path_effects=stat_effects, zorder=12, ha='left')
        y -= 0.18

    # --- PANELS ---
    w = 1.0 / 7; phase_map = {"New Moon":"new", "Waxing Crescent":"waxing_crescent", "First Quarter":"first_quarter", "Waxing Gibbous":"waxing_gibbous", "Full Moon":"full", "Waning Gibbous":"waning_gibbous", "Last Quarter":"third_quarter", "Waning Crescent":"waning_crescent"}
    
    def get_dynamic_zoom(img_arr, target_height_px):
        if img_arr is None: return 0.1
        current_h = img_arr.shape[0]
        return target_height_px / current_h

    for i, day in enumerate(data):
        ax = fig.add_axes([i*w, 0.05, w, 0.78]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        bg = TODAY_BG_COLOR if i == 0 else PANEL_BG_COLOR; ax.add_patch(mpatches.Rectangle((0,0),1,1, color=bg, zorder=0))
        
        if i == 0: add_today_glow(ax)

        lbl = datetime.now().strftime("%a").upper() if i == 0 else day['day_label']
        day_img = create_gradient_text_image(lbl, 16, 'green'); ax.add_artist(AnnotationBbox(OffsetImage(day_img, zoom=0.5), (0.5, 0.97), frameon=False, zorder=10))
        astro_img = create_gradient_text_image(day['astro_text'], 9, 'white'); ax.add_artist(AnnotationBbox(OffsetImage(astro_img, zoom=0.35), (0.5, 0.92), frameon=False, zorder=10))
        
        d_dat = day['day_data']
        y_day_center = 0.75 
        
        full_text = (d_dat['text'] if d_dat else "") + (day['night_data']['text'] if day['night_data'] else "")
        add_particles(ax, full_text)

        if d_dat:
            h_img = create_gradient_text_image(f"H:{d_dat['temp']}°F", 20, 'hot'); ax.add_artist(AnnotationBbox(OffsetImage(h_img, zoom=0.5), (0.5, 0.86), frameon=False, zorder=10))
            
            if d_dat['icon_img'] is not None: 
                zoom_val = get_dynamic_zoom(d_dat['icon_img'], 70)
                ax.add_artist(AnnotationBbox(OffsetImage(d_dat['icon_img'], zoom=zoom_val), (0.5, y_day_center), frameon=False, zorder=10))
                add_reflection(ax, d_dat['icon_img'], 0.5, y_day_center)

            if d_dat['pop'] >= 10: 
                pop = create_pop_badge(d_dat['pop'], holographic=True)
                ax.add_artist(AnnotationBbox(OffsetImage(pop, zoom=0.50), (0.78, y_day_center - 0.03), frameon=False, zorder=10))
            
            desc_str = "\n".join(textwrap.wrap((d_dat['text']).upper(), 16)[:2])
            desc_img = create_gradient_text_image(desc_str, 11, 'white')
            ax.add_artist(AnnotationBbox(OffsetImage(desc_img, zoom=0.35), (0.5, 0.65), frameon=False, zorder=10))
            
            wspd = str(d_dat['wind_spd']).replace(" mph", "")
            if wspd and wspd != "0":
                 wind_val = 0
                 try: wind_val = WIND_DIRS[d_dat['wind_dir']]
                 except: pass
                 wind_arrow = create_3d_wind_arrow(wind_val)
                 ax.add_artist(AnnotationBbox(OffsetImage(wind_arrow, zoom=0.25), (0.35, 0.58), frameon=False, zorder=10))
                 wind_txt = f"{wspd}"; wind_img = create_gradient_text_image(wind_txt, 10, 'white')
                 ax.add_artist(AnnotationBbox(OffsetImage(wind_img, zoom=0.35), (0.60, 0.58), frameon=False, zorder=10))
        else: ax.text(0.5, 0.70, "--", color='#555555', fontsize=30, ha='center', fontweight='bold', zorder=10)
        
        n_dat = day['night_data']
        
        if n_dat:
            l_img = create_gradient_text_image(f"L:{n_dat['temp']}°F", 16, 'cold')
            ax.add_artist(AnnotationBbox(OffsetImage(l_img, zoom=0.5), (0.5, 0.58), frameon=False, zorder=10))
            
            if n_dat['icon_img'] is not None:
                zoom_val_n = get_dynamic_zoom(n_dat['icon_img'], 70)
                ax.add_artist(AnnotationBbox(OffsetImage(n_dat['icon_img'], zoom=zoom_val_n), (0.5, 0.50), frameon=False, zorder=10))
                add_reflection(ax, n_dat['icon_img'], 0.5, 0.50)

            m_img = create_moon_icon(phase_map.get(moon_phases[i], "new"))
            ax.add_artist(AnnotationBbox(OffsetImage(m_img, zoom=0.12), (0.85, 0.58), frameon=False, zorder=10))
            
            desc_str_n = "\n".join(textwrap.wrap((n_dat['text']).upper(), 16)[:2])
            desc_img_n = create_gradient_text_image(desc_str_n, 11, 'white')
            ax.add_artist(AnnotationBbox(OffsetImage(desc_img_n, zoom=0.35), (0.5, 0.42), frameon=False, zorder=10))

        check_date = (datetime.now() + timedelta(days=i)).date()
        if check_date in hourly_data: 
            draw_hourly_curve(ax, hourly_data[check_date]["hours"], hourly_data[check_date]["temps"])
            draw_precip_bars(ax, hourly_data[check_date]["precip"], hourly_data[check_date]["hours"])
            app_temps = hourly_data[check_date]["app_temp"]; dews = [d for d in hourly_data[check_date]["dew"] if d is not None]
            if app_temps and dews:
                feels = max(app_temps); dew_pt = int(sum(dews)/len(dews))
                ax.text(0.35, 0.37, f"Feels: {feels}°F", color='#2E64FE', fontsize=7, ha='center', fontweight='bold', path_effects=[pe.withStroke(linewidth=1, foreground='black')], zorder=10)
                ax.text(0.65, 0.37, f"Dew: {dew_pt}°F", color='#00FF00', fontsize=7, ha='center', fontweight='bold', path_effects=[pe.withStroke(linewidth=1, foreground='black')], zorder=10)

        if i == 0:
            if extras.get('uv') and extras.get('uv') > 2:
                 uv_badge = create_uv_badge(extras['uv']); 
                 if uv_badge is not None: 
                    ax.add_artist(AnnotationBbox(OffsetImage(uv_badge, zoom=0.25), (0.15, 0.88), frameon=False, zorder=10))
        if i > 0: ax.plot([0,0],[0.05,0.95], color='#444444', lw=1, zorder=2)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"jax_7day_MIRROR_{datetime.now():%Y-%m-%d_%H%M%S}.png")
    plt.savefig(path, dpi=130, facecolor=FIG_BG_COLOR, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

if __name__ == "__main__":
    draw_full_graphic()