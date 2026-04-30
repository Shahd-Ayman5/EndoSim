"""
imaging/camera.py
-----------------
Video capture thread for reading frames from video files or webcam.
"""

from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import cv2
import numpy as np


class CaptureThread(QThread):
    

    frame_ready = pyqtSignal(np.ndarray)
    video_ended = pyqtSignal()
    error       = pyqtSignal(str)

    def __init__(self, source: str | int, fps: float = 30.0, parent=None):
       
        super().__init__(parent)
        self.source   = source          # file path or camera index
        self.fps      = fps
        self._running = False
        self._cap     = None

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def start_capture(self):
        """Open the video source and start the thread."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self.error.emit(f"Cannot open video source: {self.source}")
            return
        # Use the actual FPS from the file if available
        file_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if file_fps and file_fps > 0:
            self.fps = file_fps
        self._running = True
        self.start()          # QThread.start() → calls run()

    def stop_capture(self):
        """Signal the thread to stop and wait for it to finish."""
        self._running = False
        self.wait(2000)       # wait up to 2 s
        if self._cap:
            self._cap.release()
            self._cap = None

    def set_source(self, source: str | int):
        """Switch to a different video source (stop → change → restart)."""
        was_running = self._running
        if was_running:
            self.stop_capture()
        self.source = source
        if was_running:
            self.start_capture()

    # ------------------------------------------------------------------ #
    # Thread body
    # ------------------------------------------------------------------ #

    def run(self):
        """Main loop: read frames at the correct interval, emit each one."""
        interval_ms = int(1000 / max(self.fps, 1))

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                # Loop the video rather than stopping
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    self.video_ended.emit()
                    break
                self.video_ended.emit()   # notify GUI that a loop occurred

            self.frame_ready.emit(frame)
            # Sleep for one frame interval (keeps CPU usage sane)
            self.msleep(interval_ms)
