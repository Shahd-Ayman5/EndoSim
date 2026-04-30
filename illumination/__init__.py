"""
illumination module
-------------------
Illumination effects: brightness control and spotlight vignetting.
"""

from .brightness_control import apply_brightness, apply_spotlight

__all__ = ['apply_brightness', 'apply_spotlight']
