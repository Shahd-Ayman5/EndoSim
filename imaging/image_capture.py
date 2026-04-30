

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


def frame_to_qpixmap(frame: np.ndarray,
                     target_w: int,
                     target_h: int) -> QPixmap:
   
    if frame is None or frame.size == 0:
        return QPixmap()

    # OpenCV is BGR; Qt expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(target_w, target_h,
                         Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
