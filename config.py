"""
config.py
---------
Global configuration constants and settings for the EL-TOP endoscope simulation.
"""

# Application metadata
APP_NAME = "EL-TOP — Endoscope Simulation System"
APP_SHORT_NAME = "EndoSim Pro"
ORG_NAME = "EndoSim"

# Window settings
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 820

# Navigation
PAN_STEP = 20  # pixels per keypress
ZOOM_STEP = 0.1  # zoom increment per keypress
ZOOM_MIN = 0.5
ZOOM_MAX = 3.0

# Processing defaults
DEFAULT_BRIGHTNESS = 100  # percentage
DEFAULT_EDGE_THRESHOLD = 100
EDGE_THRESHOLD_MIN = 1
EDGE_THRESHOLD_MAX = 255

# Video capture
DEFAULT_FPS = 30.0

# Feature detection refresh rate
FEATURE_REFRESH_RATE_MS = 500

# Screenshots directory
SCREENSHOTS_DIR = "assets/screenshots"
