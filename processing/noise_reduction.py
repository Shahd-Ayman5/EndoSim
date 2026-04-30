import cv2
import numpy as np


def reduce_noise(frame: np.ndarray, ksize: int = 5) -> np.ndarray:
    
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)
