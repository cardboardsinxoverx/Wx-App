#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
import os
import logging
import xarray as xr
import warnings
from pathlib import Path
import importlib.util

warnings.filterwarnings("ignore")

try:
    from herbie import Herbie
    HERBIE_AVAILABLE = True
except ImportError:
    HERBIE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MAP_EXTENT = [-86.8, -80.7, 30.2, 35.8]

# Load Helper
crtm_helper = None
try:
    if os.path.exists("crtm_helper.py"):
        spec = importlib.util.spec_from_file_location("crtm_helper", "crtm_helper.py")
        crtm_helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crtm_helper)
        logging.info("Loaded crtm_helper module.")
except Exception as e:
    logging.warning(f"Helper load fail: {e}")

def manual_herbie_download(H, search_str):
    try:
        save_path = Path(H.save_dir) / H.model / H.date.strftime('%Y%m%d')
        if not save_path.exists(): save_path.mkdir(parents=True, exist_ok=True)
            
        files = H.download(search_str, verbose=False)
        if isinstance(files, list):
            if len(files) > 0: file_path = files[0]
            else: return None
        else:
            file_path = files
            
        if file_path.exists() and file_path.stat().st_size > 0:
            return file_path
        return None
    except:
        return None

def open_grib_safe(path, filter_keys=None):
    try:
        kwargs = {}
        if filter_keys: kwargs['filter_by_keys'] = filter_keys
        return xr.open_dataset(path, engine='cfgrib', backend_kwargs=kwargs)
    except:
        return None

def fetch_model_data(target_time_utc):
    if not HERBIE_AVAILABLE: return None
    if target_time_utc.tzinfo is None: target_time_utc = target_time_utc.replace(tzinfo=timezone.utc)
    
    now_utc = datetime.now(timezone.utc)
    latest_hrrr = now_utc - timedelta(hours=1)
    latest_hrrr = latest_hrrr.replace(minute=0, second=0, microsecond=0)
    fxx = int((target_time_utc - latest_hrrr).total_seconds() / 3600)
    if fxx < 0: fxx = 0
    
    run_str = latest_hrrr.strftime("%Y-%m-%d %H:00")
    logging.info(f"Checking HRRR {run_str} F{fxx:02d}")
    
    try:
        data_root = Path(os.getcwd()) / "weather_data"
        
        # 1. SURFACE (Reflectivity, Skin Temp, Surface Pressure)
        H_sfc = Herbie(run_str, model='hrrr', product='sfc', fxx=fxx, save_dir=str(data_root))
        logging.info("Downloading Surface Data...")
        
        path_refl = manual_herbie_download(H_sfc, ":(REFC|refc):entire atmosphere")
        path_ts = manual_herbie_download(H_sfc, ":TMP:surface")
        path_ps = manual_herbie_download(H_sfc, ":PRES:surface")
        
        if not path_refl: return None
        ds_refl = open_grib_safe(path_refl)
        refl = ds_refl['refc'] if 'refc' in ds_refl else ds_refl['REFC']
        
        out = {'refl': refl, 'source': 'HRRR'}

        # Prepare Surface Arrays
        ts_arr = None
        ps_arr = None
        if path_ts:
            ds_ts = open_grib_safe(path_ts)
            if ds_ts: ts_arr = (ds_ts['t'] if 't' in ds_ts else ds_ts['TMP']).values.flatten()
        if path_ps:
            ds_ps = open_grib_safe(path_ps)
            # CRITICAL FIX: Convert hPa (Surface Pressure) to Pascals
            if ds_ps: ps_arr = (ds_ps['sp'] if 'sp' in ds_ps else ds_ps['PRES']).values.flatten() * 100.0

        # 2. Profiles (PRS)
        if crtm_helper and crtm_helper.is_available():
            H_prs = Herbie(run_str, model='hrrr', product='prs', fxx=fxx, save_dir=str(data_root))
            logging.info("Downloading 3D Profiles from PRS product...")
            
            path_t = manual_herbie_download(H_prs, ":TMP:[0-9]+ mb")
            path_q = manual_herbie_download(H_prs, ":SPFH:[0-9]+ mb")
            
            if path_t and path_q:
                try:
                    logging.info("Opening Profiles...")
                    filter_kv = {'typeOfLevel': 'isobaricInhPa'}
                    ds_tprof = open_grib_safe(path_t, filter_keys=filter_kv)
                    ds_qprof = open_grib_safe(path_q, filter_keys=filter_kv)
                    
                    t_var = 't' if 't' in ds_tprof else 'TMP'
                    q_var = 'q' if 'q' in ds_qprof else 'SPFH'
                    
                    t_da = ds_tprof[t_var]
                    q_da = ds_qprof[q_var]

                    levels = np.intersect1d(t_da.isobaricInhPa, q_da.isobaricInhPa)
                    levels = np.sort(levels) 
                    
                    t_da = t_da.sel(isobaricInhPa=levels)
                    q_da = q_da.sel(isobaricInhPa=levels)
                    
                    t_arr = t_da.squeeze().values
                    q_arr = q_da.squeeze().values
                    
                    # Sanitize Q
                    q_arr = np.maximum(q_arr, 1e-9)
                    
                    # Pressure (Pa)
                    p_levs = levels * 100.0
                    p_arr = np.broadcast_to(p_levs[:, None, None], t_arr.shape)
                    
                    # Add Ceiling
                    top_p = np.full((1, p_arr.shape[1], p_arr.shape[2]), 5.0) 
                    top_t = t_arr[[0], :, :] 
                    top_q = np.full_like(top_t, 1e-9)
                    
                    p_final = np.concatenate((top_p, p_arr), axis=0)
                    t_final = np.concatenate((top_t, t_arr), axis=0)
                    q_final = np.concatenate((top_q, q_arr), axis=0)

                    # Reshape to (Pixels, Levels)
                    nlev = p_final.shape[0]
                    npix = p_final.shape[1] * p_final.shape[2]
                    
                    p_flat = p_final.reshape(nlev, npix).T
                    t_flat = t_final.reshape(nlev, npix).T
                    q_flat = q_final.reshape(nlev, npix).T
                    
                    profiles = {
                        'p': p_flat, 't': t_flat, 'q': q_flat,
                        'ts': ts_arr, 'ps': ps_arr
                    }
                    
                    logging.info(f"Data Ready: {npix} pixels, {nlev} levels. Shape: {profiles['p'].shape}")
                    
                    bt = crtm_helper.compute_bt_for_channel(profiles, 12)
                    
                    if bt is not None:
                        logging.info("CRTM Success!")
                        bt_grid = bt.reshape(t_arr.shape[1], t_arr.shape[2])
                        if np.nanmax(bt_grid) > 200: bt_grid -= 273.15
                        out['tb'] = xr.DataArray(bt_grid, coords=refl.coords, dims=refl.dims)
                        
                except Exception as e:
                    logging.error(f"CRTM processing error: {e}")

        return out

    except Exception as e:
        logging.error(f"Herbie/Processing fail: {e}")
        return None

def plot_map(data, fname):
    logging.info(f"Plotting {fname}...")
    fig = plt.figure(figsize=(10, 10), facecolor='#333333')
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white')
    ax.add_feature(cfeature.STATES, edgecolor='white')
    
    if 'tb' in data:
        tb = data['tb']
        vmin = np.nanmin(tb)
        vmax = np.nanmax(tb)
        
        if np.isnan(vmin) or vmax - vmin < 1.0: vmin, vmax = -80, 50
        else: vmin, vmax = vmin - 1, vmax + 1
            
        levels = np.linspace(vmin, vmax, 100)
        
        ax.contourf(tb.longitude, tb.latitude, tb, cmap='gray_r', levels=levels, transform=ccrs.PlateCarree())
        ax.text(0.02, 0.95, "Simulated GOES-16 IR (CRTM)", transform=ax.transAxes, color='yellow', fontsize=16, fontweight='bold')
    
    if 'refl' in data:
        rf = data['refl']
        rf = rf.where(rf > 15)
        ax.contourf(rf.longitude, rf.latitude, rf, cmap='jet', alpha=0.6, levels=np.arange(15, 75, 5), transform=ccrs.PlateCarree())

    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    logging.info("Done.")

if __name__ == "__main__":
    dt = datetime.now(timezone.utc)
    data = fetch_model_data(dt)
    if data: plot_map(data, "ga_forecast_test.png")
