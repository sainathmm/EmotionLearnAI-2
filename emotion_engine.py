"""
emotion_engine.py — Confidence-Aware & Temporal Emotion Inference Engine
========================================================================
Handles ONNX emotion model loading, confidence thresholding, uncertainty
detection, and probability-based temporal smoothing across video frames.

Features:
  - Robust ONNX model loading via ONNXRuntime with OpenCV DNN fallback.
  - Numerically stable Softmax probabilities across FERPlus 8 emotion classes.
  - Configurable confidence thresholding based on model confidence estimation.
  - Per-session rolling history for exponential probability temporal fusion.
  - Emotion stability metric (0.0 to 1.0) measuring consistency across recent frames.
  - Graceful decay on missing/corrupt faces preventing indefinitely stale predictions.
  - Safe error handling ensuring ONNX inference never crashes camera capture loops.
"""

import os
import time
import math
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
from vision_pipeline import (
    preprocess_emotion_face,
    align_face,
    detect_faces_and_landmarks,
    decode_base64_frame
)

# ── Paths & Classes ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, 'emotion_model.onnx')

# Standard FERPlus 8-class ordering
ONNX_LABELS = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'contempt']

# Simplified pedagogical emotion labels (mapping contempt -> disgust)
PEDAGOGICAL_EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
LABEL_MAP = {'contempt': 'disgust'}

# Configurable Thresholds
CONFIDENCE_THRESHOLD = float(os.getenv('EMOTION_CONFIDENCE_THRESHOLD', '0.35'))
HISTORY_WINDOW_SIZE = 8         # Number of frames in rolling temporal buffer (calibrated for real-time interaction)
HISTORY_MAX_AGE_SECS = 3.0      # Max age of frames to consider in temporal fusion (prevents long hysteresis)


def get_confidence_threshold() -> float:
    """Returns the current emotion confidence threshold."""
    global CONFIDENCE_THRESHOLD
    return CONFIDENCE_THRESHOLD


def set_confidence_threshold(val: float) -> None:
    """Sets a new emotion confidence threshold (clamped between 0.05 and 0.95)."""
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = float(np.clip(val, 0.05, 0.95))


# ── Model Loader Singleton ───────────────────────────────────────────────────

class EmotionModelRunner:
    """Encapsulates ONNX inference with ONNXRuntime or OpenCV DNN fallback."""

    def __init__(self, model_path: str = ONNX_MODEL_PATH):
        self.model_path = model_path
        self.ort_session = None
        self.cv2_net = None
        self.backend = None
        self.input_name = 'Input3'
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            print(f"[EMOTION ERROR] Model file not found at {self.model_path}")
            return

        # Attempt ONNXRuntime first
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self.ort_session = ort.InferenceSession(
                self.model_path, opts, providers=['CPUExecutionProvider']
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            self.backend = 'onnxruntime'
            print(f"[EMOTION] Loaded model via ONNXRuntime (input: {self.input_name})")
            return
        except Exception as e:
            print(f"[EMOTION] ONNXRuntime load failed: {e}. Trying cv2.dnn...")

        # Fallback to OpenCV DNN
        try:
            import cv2
            self.cv2_net = cv2.dnn.readNetFromONNX(self.model_path)
            self.backend = 'opencv_dnn'
            print("[EMOTION] Loaded model via OpenCV DNN")
        except Exception as e:
            print(f"[EMOTION ERROR] Failed to load model with OpenCV DNN: {e}")

    def predict(self, blob: np.ndarray) -> np.ndarray:
        """
        Runs inference on blob of shape (1, 1, 64, 64) and returns 8 raw logits.
        Guarantees finite, non-crashing output under any input state.
        """
        # Validate input blob shape & dtype
        if not isinstance(blob, np.ndarray) or blob.shape != (1, 1, 64, 64):
            if isinstance(blob, np.ndarray) and blob.size == 64 * 64:
                blob = blob.reshape(1, 1, 64, 64).astype(np.float32)
            else:
                return np.zeros(8, dtype=np.float32)

        if blob.dtype != np.float32:
            blob = blob.astype(np.float32)

        try:
            if self.ort_session is not None:
                outputs = self.ort_session.run(None, {self.input_name: blob})
                raw_logits = outputs[0][0]
            elif self.cv2_net is not None:
                self.cv2_net.setInput(blob)
                out = self.cv2_net.forward()
                raw_logits = out[0]
            else:
                raw_logits = np.zeros(8, dtype=np.float32)
        except Exception as e:
            print(f"[EMOTION ERROR] Inference execution exception: {e}")
            raw_logits = np.zeros(8, dtype=np.float32)

        # Output shape and finite validation
        raw_logits = np.asarray(raw_logits, dtype=np.float32).flatten()
        if len(raw_logits) != 8 or not np.all(np.isfinite(raw_logits)):
            raw_logits = np.zeros(8, dtype=np.float32)

        return raw_logits


# Global singleton runner
_MODEL_RUNNER = EmotionModelRunner()


# ── Softmax Probability Helper ───────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Computes numerically stable softmax probabilities."""
    shifted = logits - np.max(logits)
    e_x = np.exp(shifted)
    sum_e = np.sum(e_x)
    if sum_e <= 0 or not np.isfinite(sum_e):
        return np.full_like(logits, 1.0 / len(logits))
    return e_x / sum_e


# ── Per-Session Temporal Emotion Tracker ─────────────────────────────────────

class TemporalEmotionTracker:
    """
    Maintains a rolling history of recent frame predictions per user session,
    calculates probability-weighted temporal smoothing, and derives emotion stability.
    Gracefully decays when no face is present to avoid freezing stale emotions.
    """

    def __init__(self, window_size: int = HISTORY_WINDOW_SIZE):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.last_stable_emotion: str = 'neutral'
        self.last_stable_confidence: float = 0.65
        self.last_face_time: float = 0.0

    def notify_no_face(self) -> None:
        """Called when no face is detected in the current frame to decay stale state."""
        now = time.time()
        # If no face has been seen for > 1.5 seconds, decay historical predictions
        if self.last_face_time > 0 and (now - self.last_face_time > 1.5):
            self.history.clear()
            self.last_stable_emotion = 'neutral'
            self.last_stable_confidence = 0.0

    def add_prediction(
        self,
        raw_emotion: str,
        confidence: float,
        probs: Dict[str, float],
        is_uncertain: bool
    ) -> Tuple[str, float, float, bool]:
        """
        Adds a new frame prediction and computes temporally smoothed probability distribution.

        Returns:
            Tuple: (smoothed_emotion, smoothed_confidence, stability_score, is_uncertain)
        """
        now = time.time()
        self.last_face_time = now

        self.history.append({
            'time': now,
            'emotion': raw_emotion,
            'confidence': confidence,
            'probs': probs,
            'is_uncertain': is_uncertain
        })

        # Filter out stale frames beyond max age window
        cutoff = now - HISTORY_MAX_AGE_SECS
        active_entries = [e for e in self.history if e['time'] >= cutoff]
        if not active_entries:
            return raw_emotion, confidence, 0.5, is_uncertain

        # Probability smoothing with exponential recency weighting
        weighted_probs = {e: 0.0 for e in PEDAGOGICAL_EMOTIONS}
        total_weight = 0.0

        for entry in active_entries:
            age = max(0.0, now - entry['time'])
            # Exponential recency decay (half-life approx 0.85s)
            recency = math.exp(-0.80 * age)
            weight = recency * (0.50 if entry['is_uncertain'] else 1.0)

            for emo, p in entry['probs'].items():
                if emo in weighted_probs:
                    weighted_probs[emo] += p * weight
            total_weight += weight

        if total_weight > 0:
            for emo in weighted_probs:
                weighted_probs[emo] /= total_weight
        else:
            weighted_probs = probs.copy()

        # Determine smoothed dominant emotion
        smoothed_emotion = max(weighted_probs, key=weighted_probs.get)
        smoothed_conf = float(weighted_probs[smoothed_emotion])

        # Emotion Stability Metric (0.0 to 1.0)
        # 1. Frequency of dominant emotion in active window
        count_dominant = sum(1 for e in active_entries if e['emotion'] == smoothed_emotion and not e['is_uncertain'])
        fraction_dominant = count_dominant / len(active_entries)

        # 2. Probability margin between 1st and 2nd class
        sorted_probs = sorted(weighted_probs.values(), reverse=True)
        margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else sorted_probs[0]

        stability = float(np.clip(0.60 * fraction_dominant + 0.40 * min(1.0, margin * 2.0), 0.0, 1.0))

        # Effective uncertainty calculation
        effective_uncertain = is_uncertain or (smoothed_conf < CONFIDENCE_THRESHOLD)

        if not effective_uncertain and stability >= 0.45:
            self.last_stable_emotion = smoothed_emotion
            self.last_stable_confidence = smoothed_conf

        return smoothed_emotion, round(smoothed_conf, 3), round(stability, 3), effective_uncertain

    def clear(self):
        """Clears the temporal tracker history and resets stable baseline."""
        self.history.clear()
        self.last_stable_emotion = 'neutral'
        self.last_stable_confidence = 0.65
        self.last_face_time = 0.0


# Global session trackers dictionary
_session_trackers: Dict[str, TemporalEmotionTracker] = {}


def get_session_tracker(session_id: str) -> TemporalEmotionTracker:
    """Retrieves or creates a TemporalEmotionTracker for the given session_id."""
    if session_id not in _session_trackers:
        _session_trackers[session_id] = TemporalEmotionTracker()
    return _session_trackers[session_id]


def clear_session_tracker(session_id: str):
    """Removes the session tracker on session logout or reset."""
    _session_trackers.pop(session_id, None)


# ── Public Emotion Inference API ─────────────────────────────────────────────

def infer_emotion(
    frame_b64: Optional[str] = None,
    img: Optional[np.ndarray] = None,
    session_id: str = 'default'
) -> Dict[str, Any]:
    """
    Complete emotion analysis pipeline on a frame:
      1. Decode base64 frame if needed.
      2. Detect faces, landmarks, and evaluate multi-face/no-face status.
      3. Perform anatomical face alignment.
      4. Apply standardized preprocessing (matching ONNX Emotion-FERPlus input).
      5. Run ONNX model inference and compute full probability distribution.
      6. Calculate confidence, stability, and temporal smoothed emotion.

    Returns:
        dict:
            face_detected       (bool)
            face_count          (int)
            face_status         (str)   — 'single_face', 'multiple_faces', 'no_face'
            emotion             (str)   — Temporally smoothed dominant emotion
            raw_emotion         (str)   — Single-frame dominant emotion
            confidence          (float) — Smoothed model confidence score (0.0 - 1.0)
            raw_confidence      (float) — Raw single-frame confidence score
            stability           (float) — Emotion stability score (0.0 - 1.0)
            stability_label     (str)   — 'stable' (>=0.60), 'moderate' (0.35-0.59), 'unstable' (<0.35)
            is_uncertain        (bool)  — True if confidence < threshold
            all                 (dict)  — Full probability distribution over pedagogical emotions
            raw_all             (dict)  — Raw 8-class probability distribution
            alignment           (dict)  — Face alignment transform info
            primary_face        (dict)  — Primary face landmarks and bounding box
    """
    if img is None and frame_b64:
        img = decode_base64_frame(frame_b64)

    tracker = get_session_tracker(session_id)

    # 1. Handle missing/corrupt image
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        tracker.notify_no_face()
        return {
            'face_detected': False,
            'face_count': 0,
            'face_status': 'no_face',
            'emotion': 'neutral',
            'raw_emotion': 'no_face',
            'confidence': 0.0,
            'raw_confidence': 0.0,
            'stability': 0.0,
            'stability_label': 'unstable',
            'is_uncertain': True,
            'all': {e: round(1.0 / len(PEDAGOGICAL_EMOTIONS), 3) for e in PEDAGOGICAL_EMOTIONS},
            'raw_all': {l: 0.125 for l in ONNX_LABELS},
            'alignment': {'aligned': False, 'angle_deg': 0.0},
            'primary_face': None
        }

    # 2. Detect faces & landmarks
    det = detect_faces_and_landmarks(img)
    face_status = det['face_status']

    if not det['face_detected'] or det['primary_face'] is None:
        tracker.notify_no_face()
        return {
            'face_detected': False,
            'face_count': 0,
            'face_status': 'no_face',
            'emotion': 'neutral',
            'raw_emotion': 'no_face',
            'confidence': 0.0,
            'raw_confidence': 0.0,
            'stability': 0.0,
            'stability_label': 'unstable',
            'is_uncertain': True,
            'all': {e: round(1.0 / len(PEDAGOGICAL_EMOTIONS), 3) for e in PEDAGOGICAL_EMOTIONS},
            'raw_all': {l: 0.125 for l in ONNX_LABELS},
            'alignment': {'aligned': False, 'angle_deg': 0.0},
            'primary_face': None
        }

    # 3. Face Alignment
    aligned_face, align_meta = align_face(img, det['primary_face'])

    # 4. Standardized ONNX Preprocessing
    blob = preprocess_emotion_face(aligned_face, target_size=(64, 64), enhance_lighting=True)

    # 5. Run ONNX Model Inference
    raw_logits = _MODEL_RUNNER.predict(blob).astype(np.float64)

    # Softmax probabilities directly on model logits
    raw_probs = _softmax(raw_logits)

    # Build raw 8-class mapping
    raw_dist = {ONNX_LABELS[i]: float(raw_probs[i]) for i in range(len(ONNX_LABELS))}

    # Map to pedagogical emotions
    ped_dist = {e: 0.0 for e in PEDAGOGICAL_EMOTIONS}
    for label, prob in raw_dist.items():
        mapped_target = LABEL_MAP.get(label, label)
        if mapped_target in ped_dist:
            ped_dist[mapped_target] += prob
        else:
            ped_dist['neutral'] += prob

    # Normalize mapped distribution
    total_p = sum(ped_dist.values())
    if total_p > 0:
        for e in ped_dist:
            ped_dist[e] /= total_p

    # Raw single-frame dominant emotion & confidence
    raw_dominant = max(ped_dist, key=ped_dist.get)
    raw_conf = float(ped_dist[raw_dominant])
    is_uncertain = (raw_conf < CONFIDENCE_THRESHOLD)

    # 6. Temporal Fusion & Stability Calculation
    smoothed_emo, smoothed_conf, stability, final_uncertain = tracker.add_prediction(
        raw_dominant, raw_conf, ped_dist, is_uncertain
    )

    stability_label = 'stable' if stability >= 0.60 else ('moderate' if stability >= 0.35 else 'unstable')

    return {
        'face_detected': True,
        'face_count': int(det['face_count']),
        'face_status': face_status,
        'emotion': smoothed_emo,
        'raw_emotion': raw_dominant,
        'confidence': round(smoothed_conf, 3),
        'raw_confidence': round(raw_conf, 3),
        'stability': round(stability, 3),
        'stability_label': stability_label,
        'is_uncertain': bool(final_uncertain),
        'all': {k: round(float(v), 3) for k, v in ped_dist.items()},
        'raw_all': {k: round(float(v), 3) for k, v in raw_dist.items()},
        'alignment': align_meta,
        'primary_face': det['primary_face']
    }
