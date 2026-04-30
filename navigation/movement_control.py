import cv2
import numpy as np


def pan_crop(frame: np.ndarray,
             offset_x: int,
             offset_y: int,
             view_w: int,
             view_h: int) -> np.ndarray:
   
    h, w = frame.shape[:2]
    # Clamp offsets
    ox = max(0, min(offset_x, w - view_w))
    oy = max(0, min(offset_y, h - view_h))
    # If the frame is smaller than the desired view, return the full frame
    if w <= view_w or h <= view_h:
        return frame.copy()
    cropped = frame[oy:oy + view_h, ox:ox + view_w]
    return cv2.resize(cropped, (view_w, view_h))
