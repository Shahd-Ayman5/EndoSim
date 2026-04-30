"""
processing module
-----------------
Image processing pipeline components: noise reduction, contrast enhancement, feature extraction.
"""

from .noise_reduction import reduce_noise
from .contrast_enhancement import enhance_contrast
from .feature_extraction import (
    detect_edges,
    extract_color_features,
    count_edges,
    extract_color_histogram,
    extract_histogram_peaks,
    detect_colors,
)

__all__ = [
    'reduce_noise',
    'enhance_contrast',
    'detect_edges',
    'extract_color_features',
    'count_edges',
    'extract_color_histogram',
    'extract_histogram_peaks',
    'detect_colors',
]
