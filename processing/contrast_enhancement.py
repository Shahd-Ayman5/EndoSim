"""
processing/contrast_enhancement.py
----------------------------------
Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalisation).
"""

import cv2
import numpy as np


def enhance_contrast(frame: np.ndarray, 
                    clip_limit: float = 2.0,
                    tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
