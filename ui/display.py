"""
ui/display.py
-------------
Main window and display logic for the EL-TOP endoscope simulation GUI.
"""

import sys
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QFileDialog, QSizePolicy, QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
import numpy as np
import cv2

import config
from imaging import CaptureThread, frame_to_qpixmap, save_frame
from illumination import apply_brightness, apply_spotlight
from navigation import pan_crop
from processing import reduce_noise, enhance_contrast, detect_edges, extract_color_features, count_edges, detect_colors, extract_histogram_peaks

STYLESHEET = """
QMainWindow {
    background-color: #070b1a;
}
QWidget {
    color: #d7ecff;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
    background: transparent;
}
QWidget#appRoot {
    background-color: qradialgradient(
        cx: 0.5, cy: 0.15, radius: 1.2,
        fx: 0.5, fy: 0.15,
        stop: 0 #15254a,
        stop: 0.45 #0b1531,
        stop: 1 #060b18
    );
}
QWidget#rightPanel {
    background-color: rgba(7, 11, 27, 180);
    border: 1px solid rgba(86, 149, 255, 70);
    border-radius: 16px;
}
QLabel#panelTitle {
    color: #79d1ff;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 8px 0 2px 0;
}
QGroupBox {
    background-color: rgba(6, 12, 27, 185);
    border: 1px solid rgba(99, 166, 255, 65);
    border-radius: 12px;
    margin-top: 16px;
    padding: 12px 10px 10px 10px;
    color: #c9e7ff;
    font-size: 12px;
    font-weight: 600;
}
QGroupBox#cardMagenta {
    border: 1px solid rgba(226, 107, 255, 190);
}
QGroupBox#cardCyan {
    border: 1px solid rgba(82, 214, 255, 210);
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 2px 10px;
    color: #8dd8ff;
}
QLabel#videoLabel {
    background-color: rgba(2, 8, 20, 220);
    border: 1px solid rgba(80, 144, 227, 90);
    border-radius: 10px;
    color: #6d7d96;
    font-size: 14px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: rgba(48, 65, 98, 180);
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #54d9ff;
    border: 1px solid #9be9ff;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #2da3db;
    border-radius: 3px;
}
QPushButton {
    background-color: rgba(18, 34, 63, 220);
    border: 1px solid rgba(89, 156, 248, 130);
    border-radius: 8px;
    color: #d2ebff;
    font-size: 13px;
    padding: 8px 12px;
    min-height: 34px;
}
QPushButton:hover {
    background-color: rgba(34, 54, 92, 235);
    border-color: rgba(129, 217, 255, 230);
    color: #ffffff;
}
QPushButton:pressed {
    background-color: rgba(14, 26, 46, 245);
}
QPushButton#btnPause[paused="true"] {
    border-color: #5bf08d;
    color: #8affad;
}
QCheckBox {
    spacing: 6px;
    font-size: 13px;
    color: #cbe6ff;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid rgba(94, 142, 214, 160);
    border-radius: 3px;
    background: rgba(13, 25, 44, 220);
}
QCheckBox::indicator:checked {
    background: #2aa5d9;
    border-color: #86e5ff;
}
QStatusBar {
    background: rgba(3, 8, 19, 245);
    color: #6484a8;
    font-size: 12px;
    border-top: 1px solid rgba(73, 128, 198, 90);
    padding: 3px 8px;
}
"""


class MainWindow(QMainWindow):
    """Main application window for the EL-TOP endoscope simulation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.setMinimumSize(config.WINDOW_MIN_WIDTH, config.WINDOW_MIN_HEIGHT)

        # ── state ────────────────────────────────────────────────────────
        self._current_raw = None
        self._current_processed = None
        self._pan_x = self._pan_y = 0
        self._frame_w = self._frame_h = 0
        self._zoom = 1.0
        self._paused = False
        self._frozen_frame = None
        self._spotlight_on = False
        self._noise_on = False
        self._contrast_on = False
        self._overlay_on = False
        self._edge_detection_on = False
        self._color_detection_on = False
        self._histogram_peaks_on = False
        self._edge_threshold = config.DEFAULT_EDGE_THRESHOLD

        # ── capture thread ───────────────────────────────────────────────
        self._thread = CaptureThread(source="")
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.error.connect(self._on_capture_error)
        self._thread.video_ended.connect(self._on_video_ended)

        # ── feature refresh timer ────────────────────────────────────────
        self._feature_timer = QTimer(self)
        self._feature_timer.timeout.connect(self._refresh_features)
        self._feature_timer.start(config.FEATURE_REFRESH_RATE_MS)

        self._build_ui()
        self.setStyleSheet(STYLESHEET)
        self.statusBar().showMessage("Ready — load a video file.")
        self.setFocusPolicy(Qt.StrongFocus)

    # ================================================================== #
    # UI Construction
    # ================================================================== #

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("appRoot")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        root.addLayout(self._build_left_panel(), stretch=3)

        right = QWidget()
        right.setObjectName("rightPanel")
        right.setLayout(self._build_right_panel())
        right.setFixedWidth(290)
        root.addWidget(right)

    def _build_left_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)

        self._lbl_video = QLabel("No video loaded\n\nLoad a video file to begin")
        self._lbl_video.setObjectName("videoLabel")
        self._lbl_video.setAlignment(Qt.AlignCenter)
        self._lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        box1 = QGroupBox("PRIMARY FEED  [LIVE]")
        box1.setObjectName("cardMagenta")
        b1 = QVBoxLayout(box1)
        b1.setContentsMargins(4, 4, 4, 4)
        b1.addWidget(self._lbl_video)
        layout.addWidget(box1, stretch=1)

        return layout

    def _build_right_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel(config.APP_SHORT_NAME)
        title.setObjectName("panelTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        illum_group = self._build_illumination_group()
        zoom_group = self._build_zoom_group()
        processing_group = self._build_processing_group()

        illum_group.setMaximumHeight(120)
        zoom_group.setMaximumHeight(140)
        processing_group.setMaximumHeight(250)

        layout.addWidget(illum_group)
        layout.addWidget(zoom_group)
        layout.addWidget(processing_group)
        layout.addWidget(self._build_actions_group())
        layout.addWidget(self._build_nav_group())
        layout.addStretch()

        return layout

    def _build_illumination_group(self):
        box = QGroupBox("Illumination")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Brightness"))
        self._lbl_brightness = QLabel("100%")
        self._lbl_brightness.setStyleSheet("color:#7ed9ff; font-weight:bold;")
        row.addStretch();
        row.addWidget(self._lbl_brightness)
        v.addLayout(row)

        self._slider_brightness = QSlider(Qt.Horizontal)
        self._slider_brightness.setRange(0, 200)
        self._slider_brightness.setValue(config.DEFAULT_BRIGHTNESS)
        self._slider_brightness.valueChanged.connect(self._on_brightness_changed)
        v.addWidget(self._slider_brightness)

        self._chk_spotlight = QCheckBox("Spotlight vignette")
        self._chk_spotlight.stateChanged.connect(
            lambda s: setattr(self, "_spotlight_on", bool(s)))
        v.addWidget(self._chk_spotlight)
        return box

    def _build_zoom_group(self):
        box = QGroupBox("Zoom  ( +  /  - )")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)

        row = QHBoxLayout()
        zoom_label = QLabel("Zoom level")
        zoom_label.setFocusPolicy(Qt.NoFocus)
        row.addWidget(zoom_label)
        self._lbl_zoom = QLabel("1.0×")
        self._lbl_zoom.setStyleSheet("color:#7ed9ff; font-weight:bold;")
        self._lbl_zoom.setFocusPolicy(Qt.NoFocus)
        row.addStretch();
        row.addWidget(self._lbl_zoom)
        v.addLayout(row)

        self._slider_zoom = QSlider(Qt.Horizontal)
        self._slider_zoom.setRange(100, 300)
        self._slider_zoom.setValue(100)
        self._slider_zoom.setTickInterval(50)
        self._slider_zoom.setFocusPolicy(Qt.NoFocus)
        self._slider_zoom.valueChanged.connect(self._on_zoom_changed)
        v.addWidget(self._slider_zoom)

        return box

    def _build_processing_group(self):
        box = QGroupBox("Processing Pipeline")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)

        self._chk_noise = QCheckBox("Noise reduction")
        self._chk_noise.setChecked(False)
        self._chk_noise.stateChanged.connect(
            lambda s: self._on_processing_option_changed("_noise_on", "Noise reduction", s)
        )
        v.addWidget(self._chk_noise)

        self._chk_contrast = QCheckBox("Contrast enhance (CLAHE)")
        self._chk_contrast.setChecked(False)
        self._chk_contrast.stateChanged.connect(
            lambda s: self._on_processing_option_changed("_contrast_on", "Contrast enhance", s)
        )
        v.addWidget(self._chk_contrast)

        self._chk_overlay = QCheckBox("Feature overlay on frame")
        self._chk_overlay.setChecked(False)
        self._chk_overlay.stateChanged.connect(
            lambda s: self._on_processing_option_changed("_overlay_on", "Feature overlay", s)
        )
        v.addWidget(self._chk_overlay)

        self._chk_edge = QCheckBox("Apply edge detection")
        self._chk_edge.setChecked(False)
        self._chk_edge.stateChanged.connect(self._on_edge_detection_changed)
        v.addWidget(self._chk_edge)

        self._chk_color = QCheckBox("Apply color detection")
        self._chk_color.setChecked(False)
        self._chk_color.stateChanged.connect(self._on_color_detection_changed)
        v.addWidget(self._chk_color)

        self._chk_histogram = QCheckBox("Show histogram peaks")
        self._chk_histogram.setChecked(False)
        self._chk_histogram.stateChanged.connect(self._on_histogram_peaks_changed)
        v.addWidget(self._chk_histogram)

        row_threshold = QHBoxLayout()
        row_threshold.addWidget(QLabel("Threshold"))
        self._lbl_edge_threshold = QLabel(str(self._edge_threshold))
        self._lbl_edge_threshold.setStyleSheet("color:#7ed9ff; font-weight:bold;")
        row_threshold.addStretch();
        row_threshold.addWidget(self._lbl_edge_threshold)
        v.addLayout(row_threshold)

        self._slider_edge_threshold = QSlider(Qt.Horizontal)
        self._slider_edge_threshold.setRange(config.EDGE_THRESHOLD_MIN, config.EDGE_THRESHOLD_MAX)
        self._slider_edge_threshold.setValue(self._edge_threshold)
        self._slider_edge_threshold.valueChanged.connect(self._on_edge_threshold_changed)
        v.addWidget(self._slider_edge_threshold)
        return box

    def _build_actions_group(self):
        box = QGroupBox("Actions")
        v = QVBoxLayout(box)
        v.setSpacing(14)

        for label, slot in [
            ("Load Video", self._on_load_video),
            ("Capture Frame", self._on_capture),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            v.addWidget(btn)

        self._btn_pause = QPushButton("Pause")
        self._btn_pause.setObjectName("btnPause")
        self._btn_pause.setProperty("paused", False)
        self._btn_pause.clicked.connect(self._on_toggle_pause)
        v.addWidget(self._btn_pause)
        return box

    def _build_nav_group(self):
        box = QGroupBox("Navigation  (W A S D / Arrows)")
        v = QVBoxLayout(box)

        hint = QLabel(
            "W / ↑   pan up\n"
            "S / ↓   pan down\n"
            "A / ←   pan left\n"
            "D / →   pan right\n"
            "+  /  =  zoom in\n"
            "  −       zoom out\n"
            "R         reset all"
        )
        hint.setStyleSheet("color: #94acc9; font-size: 12px;")
        v.addWidget(hint)

        self._lbl_pan = QLabel("Pan: (0, 0)   Zoom: 1.0×")
        self._lbl_pan.setStyleSheet("color:#7ed9ff; font-size:12px; font-weight:bold;")
        v.addWidget(self._lbl_pan)
        return box

    # ================================================================== #
    # Feature overlay — draw text + edge count onto the frame itself
    # ================================================================== #

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Burn feature text into a copy of frame and return it."""
        out = frame.copy()
        c = extract_color_features(out)
        e = count_edges(out)

        lines = [
            f"R:{c['mean_R']:.0f} G:{c['mean_G']:.0f} B:{c['mean_B']:.0f}",
            f"Edge px: {e:,}",
            f"Zoom: {self._zoom:.1f}x  Pan:({self._pan_x},{self._pan_y})",
        ]

        # Add histogram peaks if enabled
        if self._histogram_peaks_on:
            peaks = extract_histogram_peaks(out, num_peaks=3)
            # Format peaks for display
            peaks_b = ', '.join([str(p[0]) for p in peaks['peaks_B']])
            peaks_g = ', '.join([str(p[0]) for p in peaks['peaks_G']])
            peaks_r = ', '.join([str(p[0]) for p in peaks['peaks_R']])
            lines.extend([
                f"Hist B peaks: {peaks_b}",
                f"Hist G peaks: {peaks_g}",
                f"Hist R peaks: {peaks_r}",
            ])

        h, w = out.shape[:2]
        banner_h = 22 * len(lines) + 12
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thick = 1
        y0 = h - banner_h + 18
        for i, line in enumerate(lines):
            y = y0 + i * 22
            cv2.putText(out, line, (8, y), font, scale, (200, 230, 200), thick, cv2.LINE_AA)

        return out

    def _apply_edge_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw detected edges on top of the original frame."""
        high_thresh = self._edge_threshold
        low_thresh = max(0, self._edge_threshold // 3)

        edge_bgr = detect_edges(frame, low_thresh=low_thresh, high_thresh=high_thresh)
        edge_gray = cv2.cvtColor(edge_bgr, cv2.COLOR_BGR2GRAY)

        out = frame.copy()
        edge_mask = edge_gray > 0

        out[edge_mask] = (0.45 * out[edge_mask] + 0.55 * np.array([255, 255, 0])).astype(np.uint8)
        return out

    def _apply_color_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Apply color segmentation overlay showing dominant colors."""
        segmented = detect_colors(frame, num_colors=3)
        blended = cv2.addWeighted(frame, 0.6, segmented, 0.4, 0)
        return blended

    # ================================================================== #
    # Slots — Video frames
    # ================================================================== #

    @pyqtSlot(np.ndarray)
    def _on_frame(self, frame: np.ndarray):
        """Called for every new frame from the capture thread."""
        if self._paused:
            return

        self._current_raw = frame.copy()
        self._frame_h, self._frame_w = frame.shape[:2]
        self._render(frame)

    def _render(self, frame: np.ndarray):
        """Process and display one frame."""
        self._frame_h, self._frame_w = frame.shape[:2]

        zoom = max(0.1, self._zoom)
        view_w = max(int(self._frame_w / zoom), 80)
        view_h = max(int(self._frame_h / zoom), 60)

        display = pan_crop(frame, self._pan_x, self._pan_y, view_w, view_h)

        brightness = self._slider_brightness.value() / 100.0
        out = display.copy()
        if self._noise_on:
            out = reduce_noise(out)
        if self._contrast_on:
            out = enhance_contrast(out)
        out = apply_brightness(out, brightness)
        if self._spotlight_on:
            out = apply_spotlight(out)
        processed = out

        output_frame = self._apply_edge_overlay(processed) if self._edge_detection_on else processed
        if self._color_detection_on:
            output_frame = self._apply_color_overlay(output_frame if self._edge_detection_on else processed)
        self._current_processed = output_frame

        display_frame = self._draw_overlay(output_frame) if self._overlay_on else output_frame

        lw, lh = self._lbl_video.width(), self._lbl_video.height()
        self._lbl_video.setPixmap(frame_to_qpixmap(display_frame, lw, lh))

    @pyqtSlot()
    def _on_video_ended(self):
        self.statusBar().showMessage("Video looped.")

    @pyqtSlot(str)
    def _on_capture_error(self, msg: str):
        self.statusBar().showMessage(f"Error: {msg}")

    # ================================================================== #
    # Slots — Controls
    # ================================================================== #

    def _on_brightness_changed(self, value: int):
        self._lbl_brightness.setText(f"{value}%")
        if self._paused and self._frozen_frame is not None:
            self._render(self._frozen_frame)

    def _on_zoom_changed(self, value: int):
        self._zoom = value / 100.0
        self._lbl_zoom.setText(f"{self._zoom:.1f}×")
        self._update_nav_label()
        if self._paused and self._frozen_frame is not None:
            self._render(self._frozen_frame)

    def _on_processing_option_changed(self, attr_name: str, label: str, state: int):
        setattr(self, attr_name, bool(state))
        self.statusBar().showMessage(f"{label}: {'ON' if bool(state) else 'OFF'}")
        if self._paused and self._frozen_frame is not None:
            self._render(self._frozen_frame)

    def _on_toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._frozen_frame = self._current_raw.copy() \
                if self._current_raw is not None else None
            self._btn_pause.setText("Resume")
            self._btn_pause.setProperty("paused", True)
            self.statusBar().showMessage("PAUSED — adjust controls; frame updates in real time.")
        else:
            self._frozen_frame = None
            self._btn_pause.setText("Pause")
            self._btn_pause.setProperty("paused", False)
            self.statusBar().showMessage("Resumed.")
        self._btn_pause.style().unpolish(self._btn_pause)
        self._btn_pause.style().polish(self._btn_pause)

    def _on_edge_detection_changed(self, state: int):
        self._edge_detection_on = bool(state)
        self.statusBar().showMessage(f"Edge detection: {'ON' if self._edge_detection_on else 'OFF'}")
        if self._paused and self._frozen_frame is not None:
            self._render(self._frozen_frame)

    def _on_color_detection_changed(self, state: int):
        self._color_detection_on = bool(state)
        self.statusBar().showMessage(f"Color detection: {'ON' if self._color_detection_on else 'OFF'}")
        if self._paused and self._frozen_frame is not None:
            self._render(self._frozen_frame)

    def _on_histogram_peaks_changed(self, state: int):
        self._histogram_peaks_on = bool(state)
        self.statusBar().showMessage(f"Histogram peaks: {'ON' if self._histogram_peaks_on else 'OFF'}")
        if self._paused and self._frozen_frame is not None:
            self._render(self._frozen_frame)

    def _on_edge_threshold_changed(self, value: int):
        self._edge_threshold = value
        self._lbl_edge_threshold.setText(str(self._edge_threshold))
        if self._paused and self._frozen_frame is not None:
            self._render(self._frozen_frame)

    def _on_load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)")
        if path:
            self._start_video(path)

    def _on_capture(self):
        if self._paused and self._frozen_frame is not None:
            frame = self._frozen_frame.copy()
        else:
            frame = self._current_processed if self._current_processed is not None else self._current_raw
        
        if frame is None:
            self.statusBar().showMessage("Nothing to capture — load a video first.")
            return
        
        save_frame_data = self._draw_overlay(frame) if self._overlay_on else frame
        _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = save_frame(save_frame_data, os.path.join(_ROOT, config.SCREENSHOTS_DIR))
        self.statusBar().showMessage(f"Saved → {path}" if path else "Capture failed.")

    def _refresh_features(self):
        """Feature refresh placeholder."""
        pass

    # ================================================================== #
    # Keyboard navigation + zoom
    # ================================================================== #

    def keyPressEvent(self, event):
        k = event.key()
        changed = True
        if k in (Qt.Key_W, Qt.Key_Up):
            self._pan_y = max(0, self._pan_y - config.PAN_STEP)
        elif k in (Qt.Key_S, Qt.Key_Down):
            self._pan_y += config.PAN_STEP
        elif k in (Qt.Key_A, Qt.Key_Left):
            self._pan_x = max(0, self._pan_x - config.PAN_STEP)
        elif k in (Qt.Key_D, Qt.Key_Right):
            self._pan_x += config.PAN_STEP
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            new_val = min(300, self._slider_zoom.value() + int(config.ZOOM_STEP * 100))
            self._slider_zoom.setValue(new_val)
        elif k == Qt.Key_Minus:
            new_val = max(100, self._slider_zoom.value() - int(config.ZOOM_STEP * 100))
            self._slider_zoom.setValue(new_val)
        elif k == Qt.Key_R:
            self._pan_x = self._pan_y = 0
            self._slider_zoom.setValue(100)
        elif k == Qt.Key_Space:
            self._on_toggle_pause()
            changed = False
        else:
            super().keyPressEvent(event)
            return

        if changed:
            self._update_nav_label()
            if self._paused and self._frozen_frame is not None:
                self._render(self._frozen_frame)

    def _update_nav_label(self):
        self._lbl_pan.setText(f"Pan: ({self._pan_x}, {self._pan_y})   Zoom: {self._zoom:.1f}×")

    def _start_video(self, path: str):
        self._paused = False
        self._frozen_frame = None
        self._btn_pause.setText("Pause")
        self._btn_pause.setProperty("paused", False)

        if self._thread.isRunning():
            self._thread.stop_capture()
        self._thread = CaptureThread(source=path)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.error.connect(self._on_capture_error)
        self._thread.video_ended.connect(self._on_video_ended)
        self._thread.start_capture()
        self.statusBar().showMessage(f"Playing: {os.path.basename(path)}")
        self._pan_x = self._pan_y = 0
        self._slider_zoom.setValue(100)
        self._update_nav_label()

    def closeEvent(self, event):
        self._feature_timer.stop()
        if self._thread.isRunning():
            self._thread.stop_capture()
        event.accept()
