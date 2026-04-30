
import cv2
import numpy as np
import os
import datetime


def save_frame(frame: np.ndarray,
               directory: str = "assets/screenshots") -> str:
    
    os.makedirs(directory, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(directory, f"capture_{ts}.jpg")
    ok   = cv2.imwrite(path, frame)
    return path if ok else ""


def get_video_properties(video_path: str) -> dict:
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'fps': 0,
            'frame_count': 0,
            'width': 0,
            'height': 0,
        }
    
    props = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return props
