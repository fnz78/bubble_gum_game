"""
face_tracker.py - MediaPipe Face Landmarker (new Tasks API, mediapipe 0.10+)
Auto-downloads the face_landmarker.task model (~25 MB) on first run.
"""

import os
import urllib.request
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import sys
import tempfile

# ── Model ──────────────────────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))

# Use temporary directory for Streamlit Cloud
if hasattr(sys, "_MEIPASS") or os.environ.get("STREAMLIT_CLOUD"):
    _MODEL_DIR = os.path.join(tempfile.gettempdir(), ".bubble_gum_game")
    os.makedirs(_MODEL_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")
else:
    # If running as packaged executable, store in user home directory
    if hasattr(sys, "_MEIPASS"):
        _MODEL_DIR = os.path.join(os.path.expanduser("~"), ".bubble_gum_game")
        os.makedirs(_MODEL_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")
    else:
        MODEL_PATH  = os.path.join(_THIS_DIR, "face_landmarker.task")

MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
# ── Landmark indices (same 468-point Face Mesh topology) ──────────────────────
UPPER_LIP    = 13
LOWER_LIP    = 14
LEFT_CORNER  = 61
RIGHT_CORNER = 291
FACE_LEFT    = 234
FACE_RIGHT   = 454


# ── Compat wrappers (mimic old mp.solutions result objects) ───────────────────

class _LandmarkList:
    """Wraps a list of NormalizedLandmark so code can do landmarks.landmark[i]."""
    def __init__(self, landmarks):
        self.landmark = landmarks          # list of NormalizedLandmark (x,y,z)


class _Result:
    """Wraps FaceLandmarkerResult so code can do results.multi_face_landmarks."""
    def __init__(self, face_lists):
        # None when empty to match old API truthiness check
        self.multi_face_landmarks = face_lists if face_lists else None


# ── FaceTracker ───────────────────────────────────────────────────────────────

class FaceTracker:
    """Tracks face landmarks using the MediaPipe Tasks FaceLandmarker."""

    def __init__(self, max_faces: int = 2):
        # Download model once
        if not os.path.exists(MODEL_PATH):
            print(f"[FaceTracker] Downloading face landmarker model (~25 MB) ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[FaceTracker] Model saved to", MODEL_PATH)

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp = 0

    def process(self, frame: np.ndarray) -> _Result:
        """Process a BGR frame and return a compat result object."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Monotonically increasing timestamp in milliseconds
        current_ts = int(time.time() * 1000)
        if current_ts <= self._last_timestamp:
            current_ts = self._last_timestamp + 1
        self._last_timestamp = current_ts

        detection = self._landmarker.detect_for_video(mp_image, current_ts)

        if not detection.face_landmarks:
            return _Result([])

        wrapped = [_LandmarkList(face_lms) for face_lms in detection.face_landmarks]
        return _Result(wrapped)

    def get_mouth_metrics(self, landmarks, frame_w: int, frame_h: int) -> dict:
        """Extract mouth openness and landmark positions."""
        def lm(idx):
            l = landmarks.landmark[idx]
            return np.array([l.x * frame_w, l.y * frame_h])

        lip_top      = lm(UPPER_LIP)
        lip_bot      = lm(LOWER_LIP)
        left_corner  = lm(LEFT_CORNER)
        right_corner = lm(RIGHT_CORNER)

        lip_gap     = np.linalg.norm(lip_bot - lip_top)
        mouth_width = np.linalg.norm(right_corner - left_corner)
        center      = (lip_top + lip_bot) / 2
        ratio       = lip_gap / (mouth_width + 1e-6)

        return {
            "openness_ratio": float(ratio),
            "lip_gap_px":     float(lip_gap),
            "mouth_width_px": float(mouth_width),
            "lip_top_pt":     lip_top.astype(int),
            "lip_bot_pt":     lip_bot.astype(int),
            "left_pt":        left_corner.astype(int),
            "right_pt":       right_corner.astype(int),
            "center_pt":      center.astype(int),
        }

    def get_face_center_x(self, landmarks, frame_w: int, frame_h: int) -> float:
        """Horizontal center of face for left/right player assignment."""
        l = landmarks.landmark[FACE_LEFT]
        r = landmarks.landmark[FACE_RIGHT]
        return ((l.x + r.x) / 2) * frame_w

    def draw_mouth_landmarks(self, frame: np.ndarray, metrics: dict,
                              color=(0, 255, 120), thickness=2):
        """Draw mouth landmark points and cross lines."""
        pts = {
            "lip_top_pt": metrics["lip_top_pt"],
            "lip_bot_pt": metrics["lip_bot_pt"],
            "left_pt":    metrics["left_pt"],
            "right_pt":   metrics["right_pt"],
        }
        for pt in pts.values():
            cv2.circle(frame, tuple(pt), 4, color, -1)
        cv2.line(frame, tuple(pts["lip_top_pt"]), tuple(pts["lip_bot_pt"]),
                 color, thickness)
        cv2.line(frame, tuple(pts["left_pt"]), tuple(pts["right_pt"]),
                 color, thickness)

    def close(self):
        self._landmarker.close()
