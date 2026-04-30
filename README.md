# EndoSim Pro — Endoscope Simulation System

A simulation-based endoscope system built with **Python**, **OpenCV**, and **PyQt5**.  
The application mimics a real-time endoscope workstation: live video processing, adjustable
LED illumination, pan/crop navigation, edge-detection overlay, colour-feature extraction,
and one-click frame capture.

---

## Project Structure

```
endoscope_sim/
├── main.py                   Entry point
├── requirements.txt
├── ui/
│   ├── main.ui               Qt Designer XML layout (reference)
│   ├── main_window.py        MainWindow — GUI + all wiring
│   └── __init__.py
├── core/
│   ├── capture_thread.py     CaptureThread — background video reading (QThread)
│   ├── video_processor.py    VideoProcessor — OpenCV pipeline (stateless)
│   └── __init__.py
├── utils/
│   ├── helpers.py            Qt/NumPy converters, save_frame
│   └── __init__.py
└── assets/
    └── screenshots/          Captured frames saved here (auto-created)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (uses `str | int` union types in type hints).

---

## Usage

```bash
# 1. Launch the GUI with no video
python main.py

# 2. Load a real endoscopic video from the command line
python main.py /path/to/video.mp4

```

Inside the GUI you can click **📂 Load Video…** to browse for a file.

---

## Subsystems

### 1 · Imaging System
- Reads any OpenCV-supported video file (MP4, AVI, MKV, MOV, WMV).
- `CaptureThread` runs in a background `QThread` and emits frames via `frame_ready` signal.
- The video loops automatically when it reaches the end.

### 2 · Illumination System
- **Brightness slider** (0 – 200 %) scales pixel intensities in real time.
- **Spotlight vignette** checkbox applies a radial gradient mask (bright centre, dark edges) simulating LED ring illumination.

### 3 · Navigation
- **W / ↑ S / ↓ A / ← D / →** keys pan inside the frame by cropping a 75 % viewport.
- **R** resets the pan to (0, 0).
- Current pan offset is displayed in the control panel.

### 4 · Processing Pipeline
- **Noise reduction** — Gaussian blur (5 × 5 kernel).
- **Contrast enhancement** — CLAHE applied to the L channel in LAB colour space.
- Individual checkboxes enable/disable each stage independently.

### 5 · Display
- **Primary feed** — fully processed, brightness-adjusted, optionally vignetted.
- **Secondary feed** — Canny edge-detection overlay on the processed frame.

### 6 · Feature Extraction (refreshed every 500 ms)
- **Colour features** — mean & standard deviation of R, G, B channels.
- **Shape feature** — total Canny edge-pixel count (proxy for scene complexity).

### 7 · Capture
- Click **📷 Capture Frame** to save the current processed frame as a timestamped JPEG under `assets/screenshots/`.

---

## Architecture Notes

| Component | Responsibility | Thread |
|-----------|---------------|--------|
| `CaptureThread` | Read frames from disk | Background QThread |
| `VideoProcessor` | Stateless OpenCV transforms | GUI thread (fast) |
| `MainWindow._on_frame` | Receive frame, process, display | GUI thread (via signal) |
| `QTimer` (500 ms) | Refresh feature-analysis panel | GUI thread |

Processing is kept intentionally lightweight so the GUI thread is never blocked for more than one frame interval.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| W / ↑ | Pan up |
| S / ↓ | Pan down |
| A / ← | Pan left |
| D / → | Pan right |
| R | Reset pan |

---

## License
MIT — free to use and modify.
