"""
utils/helpers.py
----------------
Miscellaneous helper functions used across the application.
"""

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import os
import datetime


# ------------------------------------------------------------------ #
# Frame → Qt conversion
# ------------------------------------------------------------------ #

def ndarray_to_qpixmap(frame: np.ndarray,
                        target_w: int,
                        target_h: int) -> QPixmap:
    """
    Convert an OpenCV BGR NumPy array to a QPixmap scaled to fit
    (target_w × target_h) while preserving aspect ratio.
    """
    if frame is None or frame.size == 0:
        return QPixmap()

    # OpenCV is BGR; Qt expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(target_w, target_h,
                         Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)


# ------------------------------------------------------------------ #
# Frame saving
# ------------------------------------------------------------------ #

def save_frame(frame: np.ndarray,
               directory: str = "assets/screenshots") -> str:
    """
    Save *frame* as a timestamped JPEG in *directory*.
    Returns the full path of the saved file, or an empty string on failure.
    """
    os.makedirs(directory, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(directory, f"capture_{ts}.jpg")
    ok   = cv2.imwrite(path, frame)
    return path if ok else ""


# ------------------------------------------------------------------ #
# Feature-text formatting
# ------------------------------------------------------------------ #

def format_color_features(features: dict) -> str:
    """Format color feature dict into a human-readable string."""
    lines = [
        "── Color Features ──",
        f"  Red   mean={features['mean_R']:.1f}  σ={features['std_R']:.1f}",
        f"  Green mean={features['mean_G']:.1f}  σ={features['std_G']:.1f}",
        f"  Blue  mean={features['mean_B']:.1f}  σ={features['std_B']:.1f}",
    ]
    return "\n".join(lines)


def format_edge_count(count: int) -> str:
    """Format edge-pixel count into a human-readable string."""
    return f"── Shape Feature ──\n  Edge pixels: {count:,}"


