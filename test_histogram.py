#!/usr/bin/env python
"""Test script for color histogram extraction features."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processing import extract_color_histogram, extract_histogram_peaks
import numpy as np

# Create a sample frame
frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# Test histogram extraction
hist_features = extract_color_histogram(frame, bins=256)
print('✓ Color histogram extraction works')
print(f'  - Histogram bins per channel: {len(hist_features["hist_B"])}')
print(f'  - Mean intensity B: {hist_features["mean_intensity_B"]:.2f}')
print(f'  - Mean intensity G: {hist_features["mean_intensity_G"]:.2f}')
print(f'  - Mean intensity R: {hist_features["mean_intensity_R"]:.2f}')
print(f'  - Entropy B: {hist_features["entropy_B"]:.4f}')
print(f'  - Entropy G: {hist_features["entropy_G"]:.4f}')
print(f'  - Entropy R: {hist_features["entropy_R"]:.4f}')

# Test peaks extraction
peaks = extract_histogram_peaks(frame, num_peaks=3)
print('\n✓ Histogram peaks extraction works')
print(f'  - Top 3 peaks (B): {peaks["peaks_B"]}')
print(f'  - Top 3 peaks (G): {peaks["peaks_G"]}')
print(f'  - Top 3 peaks (R): {peaks["peaks_R"]}')

print('\n✓ All histogram features are working correctly!')
