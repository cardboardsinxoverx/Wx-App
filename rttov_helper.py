"""
RTTOV/CRTM helper wrapper

This module provides a small, safe wrapper around radiative transfer libraries
(RTTOV or CRTM) to compute model-equivalent brightness temperatures (TB) for
satellite channels. It intentionally keeps a small, well-documented API and
falls back gracefully when the native libraries are not installed.

Usage (high-level):

1. Install native library + Python wrapper (see README notes below).
2. Build per-column profiles: arrays of pressure (Pa), temperature (K),
   specific humidity (kg/kg) for each model level, plus surface temperature
   and surface pressure for each profile.
3. Call `compute_bt_for_channel(profiles, channel_id)` which returns a 2D
   brightness-temperature array with shape (ny, nx) in Kelvin.

Notes:
- This file does not implement the full fetch of 3D model fields.
- It provides a single place to add bindings for `pyrttov` or `pycrtm` and
  maps the high-level data format to what those libraries expect.

If you want me to integrate this into `ga_wx_fcst.py` I can, but first we
need to install the native libs or confirm use of conda packages.

"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional
import numpy as np

HAS_RTTOV = False
HAS_CRTM = False

try:
    import pyrttov as rttov  # type: ignore
    HAS_RTTOV = True
    logging.info("pyRTTOV detected and will be used for TB computations.")
except Exception:
    try:
        import crtm  # type: ignore
        HAS_CRTM = True
        logging.info("pyCRTM detected and will be used for TB computations.")
    except Exception:
        logging.info("No RTTOV/CRTM bindings detected. RTTOV helper in fallback mode.")


def compute_bt_for_channel(profiles: Dict[str, np.ndarray], channel: int, options: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
    """
    Compute brightness temperature for a single ABI-like channel.

    profiles: dictionary containing at least the following keys:
      - 'p' : pressure array, shape (nl, ny, nx) in Pa
      - 't' : temperature array, shape (nl, ny, nx) in K
      - 'q' : specific humidity array, shape (nl, ny, nx) in kg/kg
      - 'ts': surface/skin temperature, shape (ny, nx) in K
      - 'ps': surface pressure, shape (ny, nx) in Pa

    channel: numeric channel id (implementation-specific; RTTOV/CRTM use indices or strings)
    options: optional runtime options (e.g., emissivity, aerosols)

    Returns TB in Kelvin with shape (ny, nx), or None if not available.
    """
    if not (HAS_RTTOV or HAS_CRTM):
        logging.warning("RTTOV/CRTM not available; compute_bt_for_channel returning None.")
        return None

    # Minimal validation
    required = ('p', 't', 'q', 'ts', 'ps')
    for k in required:
        if k not in profiles:
            logging.error(f"Missing profile key: {k}")
            return None

    p = profiles['p']
    t = profiles['t']
    q = profiles['q']
    ts = profiles['ts']
    ps = profiles['ps']

    # Expect shapes: p,t,q -> (nl, ny, nx); ts,ps -> (ny, nx)
    if p.ndim != 3 or t.ndim != 3 or q.ndim != 3:
        logging.error("Profile arrays must be 3-D (nl, ny, nx)")
        return None

    nl, ny, nx = p.shape

    # Example approach: call RTTOV in column-batched mode.
    # Implementation here depends on the python wrapper's API and the
    # coefficient files available on the host system. For now, provide
    # a clear placeholder where actual library calls should be made.

    # Placeholder: return None to indicate not computed.
    logging.info("RTTOV helper placeholder invoked; no native call executed.")
    return None


def is_available() -> bool:
    """Return True if either RTTOV or CRTM bindings are available."""
    return HAS_RTTOV or HAS_CRTM


__all__ = ["compute_bt_for_channel", "is_available"]
