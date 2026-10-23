"""
attention.py — Real-Time Landmark-Based Attention & 3D Head Pose Module
=======================================================================
Calculates:
  - Eye Aspect Ratio (EAR) for left and right eyes.
  - State-transition blink detection (open -> closed -> open) and rolling blink rate.
  - Real 3D Head Pose estimation via OpenCV solvePnP (Yaw, Pitch, Roll, Direction).
  - Multi-component composite attention score (0-100).
  - User calibration offsets for resting posture and eye openness baseline.

Ensures all returned values use native Python datatypes for clean JSON serialization.
"""

import cv2
import numpy as np
import base64
import time
import math
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
from vision_pipeline import detect_faces_and_landmarks, decode_base64_frame

# ── Generic 3D Facial Model Reference Points (mm) ────────────────────────────
# Used by solvePnP to compute 3D rotation and translation vectors
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye outer corner
    (225.0, 170.0, -135.0),      # Right eye outer corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0),     # Right mouth corner
], dtype=np.float64)

# MediaPipe landmark indices for 6 key points
# 1: Nose tip, 152: Chin, 33: Left eye outer, 263: Right eye outer, 61: Left mouth, 291: Right mouth
FACE_MODEL_INDICES = [1, 152, 33, 263, 61, 291]

# MediaPipe landmark indices for Eye Aspect Ratio (EAR)
# Left eye: [p1(outer), p2(top_outer), p3(top_inner), p4(inner), p5(bottom_inner), p6(bottom_outer)]
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# Right eye: [p1(inner), p2(top_inner), p3(top_outer), p4(outer), p5(bottom_outer), p6(bottom_inner)]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Configurable Thresholds & Weights
EAR_CLOSED_THRESHOLD = 0.20      # EAR below this is considered eye closure
BLINK_MIN_DURATION = 0.08        # Minimum seconds for a valid blink
BLINK_MAX_DURATION = 0.48        # Maximum seconds for a blink (longer = prolonged closure)
BLINK_WINDOW_SECS = 30.0         # Rolling window for blinks per minute

# Attention Weighting (sum = 100%)
WEIGHT_HEAD_POSE = 0.35
WEIGHT_EYE_STATE = 0.30
WEIGHT_FACE_PRESENCE = 0.20
WEIGHT_BLINK_BEHAVIOR = 0.15


# ── Per-Session State Tracking ───────────────────────────────────────────────

class SessionAttentionState:
    """Tracks temporal attention, blink state machines, and calibration per session."""

    def __init__(self):
        self.blink_history: deque = deque()          # deque of timestamps of valid blinks
        self.eye_closed_start: Optional[float] = None
        self.last_eye_open: bool = True
        self.total_blinks: int = 0
        # Calibration baseline offsets
        self.calibrated: bool = False
        self.baseline_yaw: float = 0.0
        self.baseline_pitch: float = 0.0
        self.baseline_ear: float = 0.28
        # EMA smoothed attention score to prevent frame-to-frame jumps
        self.ema_attention: float = 70.0
        self.ema_alpha: float = 0.20  # smoothing factor (lower = smoother)

    def update_blink(self, is_eyes_open: bool, now: float) -> int:
        """
        State machine to detect valid blinks: open -> closed -> open with valid duration.
        Returns total lifetime blinks detected.
        """
        if self.last_eye_open and not is_eyes_open:
            # Transitioned from open to closed
            self.eye_closed_start = now
        elif not self.last_eye_open and is_eyes_open:
            # Transitioned from closed to open
            if self.eye_closed_start is not None:
                duration = now - self.eye_closed_start
                if BLINK_MIN_DURATION <= duration <= BLINK_MAX_DURATION:
                    self.blink_history.append(now)
                    self.total_blinks += 1
                self.eye_closed_start = None

        self.last_eye_open = is_eyes_open

        # Prune old blink history
        cutoff = now - BLINK_WINDOW_SECS
        while self.blink_history and self.blink_history[0] < cutoff:
            self.blink_history.popleft()

        return self.total_blinks

    def get_blink_rate(self, now: float) -> float:
        """Calculates estimated blinks per minute over rolling window."""
        cutoff = now - BLINK_WINDOW_SECS
        recent_blinks = [t for t in self.blink_history if t >= cutoff]
        if not recent_blinks:
            return 0.0
        elapsed = min(BLINK_WINDOW_SECS, max(now - recent_blinks[0], 5.0))
        rate = (len(recent_blinks) / elapsed) * 60.0
        return round(float(rate), 1)

    def set_calibration(self, baseline_yaw: float, baseline_pitch: float, baseline_ear: float):
        """Sets calibrated baseline parameters for this session."""
        self.calibrated = True
        self.baseline_yaw = float(baseline_yaw)
        self.baseline_pitch = float(baseline_pitch)
        self.baseline_ear = max(0.20, float(baseline_ear))

    def reset(self):
        self.blink_history.clear()
        self.eye_closed_start = None
        self.last_eye_open = True
        self.total_blinks = 0
        self.ema_attention = 70.0


# Global session attention dictionary
_attention_sessions: Dict[str, SessionAttentionState] = {}


def get_attention_session(session_id: str) -> SessionAttentionState:
    if session_id not in _attention_sessions:
        _attention_sessions[session_id] = SessionAttentionState()
    return _attention_sessions[session_id]


def clear_session(session_id: str):
    _attention_sessions.pop(session_id, None)


# ── Mathematical Helper Functions ───────────────────────────────────────────

def _calculate_ear_from_points(pts: np.ndarray) -> float:
    """
    Calculates Eye Aspect Ratio (EAR) from 6 2D points:
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    if pts is None or len(pts) < 6:
        return 0.28
    p1, p2, p3, p4, p5, p6 = pts[:6]

    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)

    if h < 1e-6:
        return 0.28
    ear = float((v1 + v2) / (2.0 * h))
    return round(ear, 3)


def _estimate_head_pose_pnp(
    landmarks_2d: Optional[np.ndarray],
    primary_face: Dict[str, Any],
    frame_w: int,
    frame_h: int
) -> Tuple[float, float, float, str, bool]:
    """
    Estimates 3D head pose (Yaw, Pitch, Roll) using OpenCV solvePnP.

    Returns:
        Tuple of (yaw_deg, pitch_deg, roll_deg, direction_label, looking_forward_bool)
    """
    # Approximate camera intrinsic matrix
    focal_length = float(frame_w)
    center = (frame_w / 2.0, frame_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    if landmarks_2d is not None and len(landmarks_2d) >= 468:
        try:
            image_points = landmarks_2d[FACE_MODEL_INDICES, :2].astype(np.float64)
            success, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                rot_mat, _ = cv2.Rodrigues(rvec)
                # Extract Euler angles in degrees using RQDecomp3D
                angles, _, _, _, _, _ = cv2.RQDecomp3D(rot_mat)
                pitch = float(angles[0])
                yaw = float(angles[1])
                roll = float(angles[2])

                # Determine direction label
                direction = 'forward'
                abs_yaw = abs(yaw)
                abs_pitch = abs(pitch)

                looking_forward = bool(abs_yaw <= 18.0 and abs_pitch <= 16.0)

                if not looking_forward:
                    if abs_yaw >= abs_pitch:
                        direction = 'left' if yaw < 0 else 'right'
                    else:
                        direction = 'up' if pitch < 0 else 'down'

                return (
                    round(yaw, 1),
                    round(pitch, 1),
                    round(roll, 1),
                    direction,
                    looking_forward
                )
        except Exception as e:
            print(f"[ATTENTION] solvePnP landmark error: {e}")

    # Fallback to geometric bounding-box center offset
    fx, fy, fw, fh = primary_face['bbox']
    cx = fx + fw / 2.0
    cy = fy + fh / 2.0
    norm_x = (cx - frame_w / 2.0) / (frame_w / 2.0)
    norm_y = (cy - frame_h / 2.0) / (frame_h / 2.0)

    # Approximate yaw/pitch in degrees from offset
    yaw = float(norm_x * 35.0)
    pitch = float(norm_y * 30.0)
    roll = 0.0

    looking_forward = bool(abs(norm_x) <= 0.28 and abs(norm_y) <= 0.28)
    if looking_forward:
        direction = 'forward'
    elif abs(norm_x) >= abs(norm_y):
        direction = 'left' if norm_x < 0 else 'right'
    else:
        direction = 'up' if norm_y < 0 else 'down'

    return round(yaw, 1), round(pitch, 1), round(roll, 1), direction, looking_forward


# ── Public Attention Analysis API ───────────────────────────────────────────

def detect_attention(
    frame_b64: Optional[str] = None,
    img: Optional[np.ndarray] = None,
    session_id: str = 'default',
    primary_face: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Comprehensive real-time attention analysis:
      - 3D Head Pose estimation (Yaw, Pitch, Roll, Direction)
      - Eye Aspect Ratio (EAR) & Eye state (open/closed)
      - State-transition blink tracking & blink rate
      - Multi-factor composite attention score (0-100)

    Returns:
        dict:
            attention_score (int)   — 0-100 composite score
            looking_away    (bool)  — True if head turned or eyes prolonged closed
            head_direction  (str)   — 'forward', 'left', 'right', 'up', 'down', 'none'
            head_pose       (dict)  — {yaw, pitch, roll, direction, looking_forward}
            eye_state       (str)   — 'open', 'closed', 'unknown'
            ear_left        (float) — Left Eye Aspect Ratio
            ear_right       (float) — Right Eye Aspect Ratio
            ear_avg         (float) — Mean Eye Aspect Ratio
            blink_rate      (float) — Blinks per minute (0-60)
            total_blinks    (int)   — Total valid blinks in session
            face_detected   (bool)  — True if face present
            is_calibrated   (bool)  — True if user baseline is active
    """
    if img is None and frame_b64:
        img = decode_base64_frame(frame_b64)

    state = get_attention_session(session_id)
    now = time.time()

    # 1. No image / empty frame
    if img is None or img.size == 0:
        state.update_blink(False, now)
        blink_rate = state.get_blink_rate(now)
        return {
            'attention_score': 15,
            'looking_away': True,
            'head_direction': 'none',
            'head_pose': {
                'yaw': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
                'direction': 'none',
                'looking_forward': False
            },
            'eye_state': 'unknown',
            'ear_left': 0.0,
            'ear_right': 0.0,
            'ear_avg': 0.0,
            'blink_rate': blink_rate,
            'total_blinks': int(state.total_blinks),
            'face_detected': False,
            'is_calibrated': bool(state.calibrated)
        }

    h, w = img.shape[:2]

    # 2. Extract landmarks if not already passed
    if primary_face is None:
        det = detect_faces_and_landmarks(img)
        primary_face = det['primary_face']

    # 3. Handle no face detected
    if primary_face is None:
        state.update_blink(False, now)
        blink_rate = state.get_blink_rate(now)
        return {
            'attention_score': 15,
            'looking_away': True,
            'head_direction': 'none',
            'head_pose': {
                'yaw': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
                'direction': 'none',
                'looking_forward': False
            },
            'eye_state': 'unknown',
            'ear_left': 0.0,
            'ear_right': 0.0,
            'ear_avg': 0.0,
            'blink_rate': blink_rate,
            'total_blinks': int(state.total_blinks),
            'face_detected': False,
            'is_calibrated': bool(state.calibrated)
        }

    landmarks_2d = primary_face.get('landmarks_2d')

    # 4. 3D Head Pose Estimation
    raw_yaw, raw_pitch, roll, raw_direction, raw_forward = _estimate_head_pose_pnp(
        landmarks_2d, primary_face, w, h
    )

    # Apply calibration offsets
    eff_yaw = raw_yaw - state.baseline_yaw if state.calibrated else raw_yaw
    eff_pitch = raw_pitch - state.baseline_pitch if state.calibrated else raw_pitch
    abs_yaw = abs(eff_yaw)
    abs_pitch = abs(eff_pitch)

    looking_forward = bool(abs_yaw <= 18.0 and abs_pitch <= 16.0)
    if looking_forward:
        direction = 'forward'
    elif abs_yaw >= abs_pitch:
        direction = 'left' if eff_yaw < 0 else 'right'
    else:
        direction = 'up' if eff_pitch < 0 else 'down'

    # 5. Eye Aspect Ratio & Eye State
    if landmarks_2d is not None and len(landmarks_2d) >= 468:
        pts_left = landmarks_2d[LEFT_EYE_INDICES, :2]
        pts_right = landmarks_2d[RIGHT_EYE_INDICES, :2]
        ear_left = _calculate_ear_from_points(pts_left)
        ear_right = _calculate_ear_from_points(pts_right)
        ear_avg = round((ear_left + ear_right) / 2.0, 3)
    else:
        # Fallback estimation
        ear_left = 0.27
        ear_right = 0.27
        ear_avg = 0.27

    # Eye closure detection (calibrated baseline if available)
    thresh = (state.baseline_ear * 0.70) if state.calibrated else EAR_CLOSED_THRESHOLD
    is_open = bool(ear_avg >= thresh)
    eye_state = 'open' if is_open else 'closed'

    # 6. Blink Tracking
    state.update_blink(is_open, now)
    blink_rate = state.get_blink_rate(now)

    # 7. Composite Attention Score Calculation (0-100)
    # ── Head Pose Score (0-100) ──
    pose_error = math.sqrt((abs_yaw / 25.0) ** 2 + (abs_pitch / 20.0) ** 2)
    head_score = float(max(0.0, min(100.0, 100.0 * (1.0 - pose_error * 0.85))))

    # ── Eye State Score (0-100) ──
    # Use a gentler formula: EAR 0.20 (closed) → 0, EAR 0.28 (normal) → 100
    # Avoids over-penalizing slightly tired eyes
    if is_open:
        ear_ratio = min(1.0, (ear_avg - EAR_CLOSED_THRESHOLD) / max(0.08, state.baseline_ear - EAR_CLOSED_THRESHOLD))
        eye_score = float(max(0.0, min(100.0, ear_ratio * 100.0)))
    else:
        eye_score = 10.0

    # ── Face Presence & Framing Score (0-100) ──
    fw, fh = primary_face['bbox'][2], primary_face['bbox'][3]
    face_ratio = (fw * fh) / (w * h)
    face_score = 100.0 if (0.05 <= face_ratio <= 0.50) else 75.0

    # ── Blink Behavior Score (0-100) ──
    # Normal resting blink rate is 12-25 blinks/min. High rate (>35) indicates fatigue/stress.
    if 10.0 <= blink_rate <= 26.0:
        blink_score = 100.0
    elif blink_rate < 10.0:
        blink_score = 80.0
    elif blink_rate <= 38.0:
        blink_score = 70.0
    else:
        blink_score = 45.0   # Excessive blinking

    # Weighted Composite Score
    raw_composite = (
        WEIGHT_HEAD_POSE * head_score +
        WEIGHT_EYE_STATE * eye_score +
        WEIGHT_FACE_PRESENCE * face_score +
        WEIGHT_BLINK_BEHAVIOR * blink_score
    )
    raw_attention = float(max(0.0, min(100.0, raw_composite)))

    # Apply EMA smoothing to prevent frame-to-frame jumps
    state.ema_attention = state.ema_alpha * raw_attention + (1.0 - state.ema_alpha) * state.ema_attention
    final_attention = int(round(state.ema_attention))

    # Looking away: only based on head pose and eye state, NOT on score threshold
    # This prevents circular dependency where low score causes looking_away=True
    looking_away = bool(not looking_forward or eye_state == 'closed')

    return {
        'attention_score': final_attention,
        'looking_away': looking_away,
        'head_direction': direction,
        'head_pose': {
            'yaw': round(eff_yaw, 1),
            'pitch': round(eff_pitch, 1),
            'roll': round(roll, 1),
            'direction': direction,
            'looking_forward': looking_forward
        },
        'eye_state': eye_state,
        'ear_left': ear_left,
        'ear_right': ear_right,
        'ear_avg': ear_avg,
        'blink_rate': blink_rate,
        'total_blinks': int(state.total_blinks),
        'face_detected': True,
        'is_calibrated': bool(state.calibrated)
    }


def calibrate_session(session_id: str, baseline_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calibrates user baseline from multiple calibration sample frames.
    """
    state = get_attention_session(session_id)
    if not baseline_samples:
        return {'status': 'failed', 'message': 'No calibration samples provided.'}

    valid_yaws = [s['yaw'] for s in baseline_samples if 'yaw' in s]
    valid_pitches = [s['pitch'] for s in baseline_samples if 'pitch' in s]
    valid_ears = [s['ear'] for s in baseline_samples if 'ear' in s and s['ear'] > 0.15]

    mean_yaw = float(np.mean(valid_yaws)) if valid_yaws else 0.0
    mean_pitch = float(np.mean(valid_pitches)) if valid_pitches else 0.0
    mean_ear = float(np.mean(valid_ears)) if valid_ears else 0.28

    state.set_calibration(mean_yaw, mean_pitch, mean_ear)
    return {
        'status': 'calibrated',
        'baseline_yaw': round(mean_yaw, 1),
        'baseline_pitch': round(mean_pitch, 1),
        'baseline_ear': round(mean_ear, 3)
    }
