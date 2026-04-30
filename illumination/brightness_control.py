
import cv2
import numpy as np


def apply_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
   
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_spotlight(frame: np.ndarray, strength: float = 0.6) -> np.ndarray:
   
    h, w = frame.shape[:2]
    # Create a radial gradient mask
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    # Normalise distance from centre to [0, 1]
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    # Smooth vignette: 1 at centre → (1-strength) at corners
    mask = 1.0 - strength * np.clip(dist, 0, 1)
    mask = mask[:, :, np.newaxis]  # broadcast over channels
    return np.clip(frame.astype(np.float32) * mask, 0, 255).astype(np.uint8)
