"""
processing/feature_extraction.py
--------------------------------
Feature extraction: edge detection and color analysis.
"""

import cv2
import numpy as np


def detect_edges(frame: np.ndarray,
                 low_thresh: int = 50,
                 high_thresh: int = 150) -> np.ndarray:
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def extract_color_features(frame: np.ndarray) -> dict:
    
    b, g, r = cv2.split(frame)
    return {
        "mean_R": float(np.mean(r)),
        "mean_G": float(np.mean(g)),
        "mean_B": float(np.mean(b)),
        "std_R":  float(np.std(r)),
        "std_G":  float(np.std(g)),
        "std_B":  float(np.std(b)),
    }


def count_edges(frame: np.ndarray,
                low_thresh: int = 50,
                high_thresh: int = 150) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return int(np.count_nonzero(edges))


def extract_color_histogram(frame: np.ndarray, bins: int = 256) -> dict:
    
    b, g, r = cv2.split(frame)
    
    # Calculate histograms
    hist_range = (0, 256)
    hist_b = cv2.calcHist([b], [0], None, [bins], hist_range)
    hist_g = cv2.calcHist([g], [0], None, [bins], hist_range)
    hist_r = cv2.calcHist([r], [0], None, [bins], hist_range)
    
    # Normalize histograms
    hist_b = hist_b.flatten() / hist_b.sum()
    hist_g = hist_g.flatten() / hist_g.sum()
    hist_r = hist_r.flatten() / hist_r.sum()
    
    # Calculate histogram statistics
    def histogram_stats(hist):
        """Calculate mean, dominant bin, and entropy from histogram."""
        bins_array = np.arange(len(hist))
        mean_intensity = float(np.sum(bins_array * hist))
        dominant_bin = int(np.argmax(hist))
        # Shannon entropy
        hist_nonzero = hist[hist > 0]
        entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero))) if len(hist_nonzero) > 0 else 0.0
        return mean_intensity, dominant_bin, entropy
    
    mean_b, dom_b, ent_b = histogram_stats(hist_b)
    mean_g, dom_g, ent_g = histogram_stats(hist_g)
    mean_r, dom_r, ent_r = histogram_stats(hist_r)
    
    return {
        "hist_B": hist_b.tolist(),
        "hist_G": hist_g.tolist(),
        "hist_R": hist_r.tolist(),
        "mean_intensity_B": mean_b,
        "mean_intensity_G": mean_g,
        "mean_intensity_R": mean_r,
        "dominant_bin_B": dom_b,
        "dominant_bin_G": dom_g,
        "dominant_bin_R": dom_r,
        "entropy_B": ent_b,
        "entropy_G": ent_g,
        "entropy_R": ent_r,
    }


def extract_histogram_peaks(frame: np.ndarray, bins: int = 256, num_peaks: int = 3) -> dict:
   
    b, g, r = cv2.split(frame)
    
    hist_range = (0, 256)
    hist_b = cv2.calcHist([b], [0], None, [bins], hist_range).flatten()
    hist_g = cv2.calcHist([g], [0], None, [bins], hist_range).flatten()
    hist_r = cv2.calcHist([r], [0], None, [bins], hist_range).flatten()
    
    def get_top_peaks(hist, num_peaks):
        """Get top N peak positions and their frequencies."""
        top_indices = np.argsort(hist)[-num_peaks:][::-1]
        return [(int(idx), float(hist[idx])) for idx in top_indices]
    
    return {
        "peaks_B": get_top_peaks(hist_b, num_peaks),
        "peaks_G": get_top_peaks(hist_g, num_peaks),
        "peaks_R": get_top_peaks(hist_r, num_peaks),
    }


def detect_colors(frame: np.ndarray, num_colors: int = 3) -> np.ndarray:
   
    # Reshape image to list of pixels
    h, w = frame.shape[:2]
    pixels = frame.reshape((-1, 3)).astype(np.float32)
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8 bit
    centers = np.uint8(centers)
    
    # Create segmented image
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape((h, w, 3))
    
    return segmented
