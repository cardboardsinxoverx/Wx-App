import numpy as np
import logging
import crtm_helper

logging.basicConfig(level=logging.INFO)

def test():
    print("Checking availability...")
    if not crtm_helper.is_available():
        print("CRTM not found.")
        return

    print("Building dummy profile...")
    # 5 levels -> 4 layers
    nlev = 5
    ncol = 1
    
    profiles = {
        'p': np.array([[1000, 800, 600, 400, 200]], dtype=np.float64), # hPa
        't': np.array([[300, 290, 280, 260, 240]], dtype=np.float64),  # K
        'q': np.array([[0.015, 0.01, 0.005, 0.001, 0.0]], dtype=np.float64), # kg/kg
        'ts': np.array([300.0], dtype=np.float64),
        'ps': np.array([1013.0], dtype=np.float64)
    }

    # AIRS Channel 399 is roughly 11um (Window)
    channel = 399 
    
    print(f"Calling CRTM for AIRS channel {channel}...")
    try:
        bt = crtm_helper.compute_bt_for_channel(profiles, channel)
        if bt is not None:
            print(f"SUCCESS! BT = {bt[0][0]:.2f} K")
        else:
            print("FAILURE: Returned None.")
    except Exception as e:
        print(f"CRASH: {e}")

if __name__ == "__main__":
    test()