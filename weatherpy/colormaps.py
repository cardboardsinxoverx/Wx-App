import matplotlib.colors as mcolors

def relative_humidity_colormap():
    """Return colormap for relative humidity."""
    colors = ['purple', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    return mcolors.LinearSegmentedColormap.from_list('rh_cmap', colors, N=100)

def relative_humidity_change_colormap():
    """Return colormap for relative humidity change."""
    colors = ['blue', 'white', 'red']
    return mcolors.LinearSegmentedColormap.from_list('rh_change_cmap', colors, N=100)

def low_relative_humidity_colormap():
    """Return colormap for low relative humidity."""
    colors = ['red', 'orange', 'yellow']
    return mcolors.LinearSegmentedColormap.from_list('low_rh_cmap', colors, N=50)

def wind_speed_colormap():
    """Return colormap for wind speed."""
    colors = ['cyan', 'green', 'yellow', 'orange', 'red']
    return mcolors.LinearSegmentedColormap.from_list('wind_cmap', colors, N=100)

def red_flag_warning_criteria_colormap():
    """Return colormap for red flag warning criteria."""
    colors = ['white', 'red']
    return mcolors.LinearSegmentedColormap.from_list('red_flag_cmap', colors, N=2)
