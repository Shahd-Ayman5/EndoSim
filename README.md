# Intelligent Endoscopic Assistance System

## Overview
A simulation-based endoscope system built with **Python**, **OpenCV**, and **PyQt5** that models the core components of a real endoscope. The system integrates real-time imaging, illumination control, navigation, image processing, and feature extraction capabilities.

---

## System Components

### 1. Illumination System
- **LED light source simulation** with adjustable intensity
- **Brightness slider** (0–200%) for real-time intensity control
- **Spotlight vignette effect** simulating realistic LED ring illumination
- Brightness adjustments applied in real-time to the video stream

### 2. Imaging System
- **Video input:** OpenCV-compatible formats (MP4, AVI, MKV, MOV, WMV)
- **Background processing:** QThread-based frame capture with no GUI blocking
- **Real-time display:** Dual feed system (primary + edge overlay)

### 3. Insertion Tube & Navigation
- **Keyboard control:** W/A/S/D or arrow keys for pan navigation
- **Movement:** Pan/crop within 75% viewport
- **Reset:** R key to return to origin
- Real-time position display showing current offset (X, Y)

### 4. Control System
- Light intensity adjustment via slider
- Pan navigation offset display
- Frame capture with timestamped storage
- Real-time parameter updates

### 5. Display System
- **PyQt5 GUI** with real-time video rendering
- **Primary feed:** Fully processed frame (brightness, contrast, vignette)
- **Secondary feed:** Canny edge-detection overlay
- Feature analysis panel with live statistics

---

## Image Processing Features

### Preprocessing
- **Noise reduction:** Gaussian blur (5 × 5 kernel)
- **Contrast enhancement:** CLAHE on LAB L-channel

### Feature Extraction
- **Color features:** Mean & standard deviation of R, G, B channels
- **Shape features:** Canny edge detection with edge-pixel count analysis

---

## Technologies Used
- **Python 3.10+**
- **OpenCV** – Image processing & video handling
- **PyQt5** – GUI framework
- **NumPy** – Numerical computing

---

## Controls
| Key | Action |
|-----|--------|
| **W / ↑** | Pan up |
| **S / ↓** | Pan down |
| **A / ←** | Pan left |
| **D / →** | Pan right |
| **R** | Reset pan |

---

## GUI Features
- **Brightness slider:** Adjust LED intensity (0–200%)
- **Spotlight vignette:** Toggle LED ring effect
- **Noise reduction:** Enable/disable Gaussian blur
- **Contrast enhancement:** Toggle CLAHE processing
- **Capture Frame:** Save timestamped JPEG to `assets/screenshots/`
- **Feature panel:** Real-time color & shape analysis

---
