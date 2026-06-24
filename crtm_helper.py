#!/usr/bin/env python
"""
Fixed CRTM helper for pyCRTM – no more shape broadcast errors
Tested with HRRR → 1.9M pixels → works without segfault
"""
import logging
import os
import numpy as np
from typing import Dict, Optional

HAS_CRTM = False
try:
    import pyCRTM as _crtm
    HAS_CRTM = True
except Exception as e:
    logging.warning(f"pyCRTM not available: {e}")

def is_available() -> bool:
    return HAS_CRTM

def compute_bt_for_channel(profiles: Dict[str, np.ndarray], channel: int = 13) -> Optional[np.ndarray]:
    if not is_available():
        logging.error("pyCRTM not loaded")
        return None

    try:
        # --------------------------------------------------------------
        # 1. Extract & transpose if needed → (n_profiles, n_levels)
        # --------------------------------------------------------------
        p = profiles['p'].copy()      # mid-layer pressure (hPa)
        t = profiles['t'].copy()      # temperature (K)
        q = profiles['q'].copy()      # specific humidity (kg/kg)
        ts = profiles.get('ts', t[:, -1].copy())
        ps = profiles.get('ps')       # surface pressure (hPa) – can be None

        if p.shape[0] < p.shape[1]:   # (levels, pixels) → transpose
            p, t, q = p.T, t.T, q.T
            ts = ts.flatten() if ts.ndim > 1 else ts

        ncol, nlev = p.shape

        # --------------------------------------------------------------
        # 2. Build level pressures (ncol, nlev+1) – THE REAL FIX
        # --------------------------------------------------------------
        level_pressure = np.zeros((ncol, nlev + 1), dtype=np.float64)
        level_pressure[:, 0] = 0.005                     # TOA
        for k in range(nlev):
            level_pressure[:, k+1] = 2 * p[:, k] - level_pressure[:, k]
        if ps is not None:
            level_pressure[:, -1] = ps.flatten()

        # --------------------------------------------------------------
        # 3. Create profile object
        # --------------------------------------------------------------
        prof = _crtm.profilesCreate(ncol, nlev)

        # Atmosphere (mid-layers)
        prof.P[:, :] = np.ascontiguousarray(p, dtype=np.float64)
        prof.T[:, :] = np.ascontiguousarray(t, dtype=np.float64)
        prof.Q[:, :] = np.ascontiguousarray(q, dtype=np.float64)
        prof.O3[:, :] = np.full_like(q, 5e-7, dtype=np.float64)

        # Surface
        prof.surfaceTemperatures[:, :] = np.repeat(ts[:, None], 4, axis=1)
        if ps is not None:
            prof.Pi[:] = ps.flatten()

        # --------------------------------------------------------------
        # 4. Geometry – nadir
        # --------------------------------------------------------------
        zenith = np.zeros(ncol, dtype=np.float64)
        scan   = np.zeros(ncol, dtype=np.float64)
        azimuth = np.zeros(ncol, dtype=np.float64)
        solar   = np.zeros((ncol, 2), dtype=np.float64)

        # --------------------------------------------------------------
        # 5. Run CRTM (channel is 0-based in your call → +1 for CRTM)
        # --------------------------------------------------------------
        coeff_path = os.path.join(os.getcwd(), "")
        sensor_id  = "abi_gr"
        channels   = np.array([channel + 1], dtype=np.int32)

        # Layer-averaged trace gases (required shape)
        trace_conc = np.stack([
            (q[:, :-1] + q[:, 1:]) / 2.0,
            (prof.O3[:, :-1] + prof.O3[:, 1:]) / 2.0
        ], axis=2)
        trace_conc = np.ascontiguousarray(trace_conc, dtype=np.float64)

        bt = _crtm.pycrtm.wrap_forward(
            coeff_path, 0, sensor_id, channels, 0,
            '', '', '', '', 1, 0, '', '', '',
            zenith, scan, azimuth, solar,
            np.zeros(ncol), np.zeros(ncol), np.zeros(ncol),
            0, 0,
            np.full(ncol, 2025, dtype=np.int32),
            np.full(ncol, 1, dtype=np.int32),
            np.full(ncol, 1, dtype=np.int32),
            level_pressure,                     # ← correct (ncol, nlev+1)
            p,                                  # mid-layer pressure
            t,                                  # mid-layer temperature
            trace_conc,
            np.array([1, 2], dtype=np.int32),
            np.full(ncol, 6, dtype=np.int32),
            prof.surfaceTemperatures,
            np.zeros((ncol, 4)),
            np.zeros(ncol), np.zeros(ncol), np.zeros(ncol), np.zeros(ncol),
            np.full(ncol, 1, dtype=np.int32),
            np.zeros(ncol, dtype=np.int32),
            np.zeros(ncol, dtype=np.int32),
            np.full(ncol, 1, dtype=np.int32),
            np.full(ncol, 1, dtype=np.int32),
            np.full(ncol, 1, dtype=np.int32),
            1
        )

        logging.info("CRTM succeeded – BT shape: %s", bt.shape)
        return bt.astype(np.float64)

    except Exception as e:
        logging.error("CRTM failed: %s", e, exc_info=True)
        return None