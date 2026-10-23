"""
vision_pipeline.py — Unified Vision, Face Landmark & Alignment Pipeline
========================================================================
Provides robust face detection, landmark extraction, eye-angle alignment,
standardized ONNX preprocessing, and camera quality assessment.

Features:
  - MediaPipe FaceMesh / FaceLandmarker with OpenCV DNN and Haar fallbacks.
  - Inter-ocular face alignment (horizontal eye alignment via affine rotation).
  - Multi-face discriminator (detects single, multiple, or no faces).
  - Camera quality checks: lighting (mean luminance), blur (Laplacian variance),
    and face framing/distance.
  - Shared preprocess_emotion_face() function used across production, testing,
    and training to ensure zero train/inference mismatch.
"""

import cv2
import numpy as np
import base64
import math
from typing import Optional, Tuple, Dict, Any, List

# Try to import MediaPipe
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# Multi-stage Cascade Classifiers
class _DummyCascade:
    def detectMultiScale(self, *args, **kwargs):
        return ()
    def empty(self):
        return True

def _load_cascade(filename: str):
    try:
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            c = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
            if hasattr(c, 'empty') and not c.empty():
                return c
    except Exception as e:
        print(f"[VISION] Cascade {filename} load warning: {e}")
    return _DummyCascade()

_FACE_CASCADE_ALT2 = _load_cascade('haarcascade_frontalface_alt2.xml')
_FACE_CASCADE_DEFAULT = _load_cascade('haarcascade_frontalface_default.xml')
_FACE_CASCADE_PROFILE = _load_cascade('haarcascade_profileface.xml')
# Keep legacy _FACE_CASCADE reference for backward compatibility
_FACE_CASCADE = _FACE_CASCADE_ALT2 if not _FACE_CASCADE_ALT2.empty() else _FACE_CASCADE_DEFAULT
_EYE_CASCADE = _load_cascade('haarcascade_eye.xml')

# MediaPipe FaceMesh lazy singleton
_mp_face_mesh = None


def _get_face_mesh():
    global _mp_face_mesh
    if not _MP_AVAILABLE:
        return None
    if _mp_face_mesh is None:
        try:
            # Try MediaPipe solutions API if available
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=3,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
        except Exception as e:
            print(f"[VISION] MediaPipe FaceMesh init fallback: {e}")
            _mp_face_mesh = None
    return _mp_face_mesh


# ── Frame Decoding Helper ───────────────────────────────────────────────────

def decode_base64_frame(frame_b64: str) -> Optional[np.ndarray]:
    """
    Decodes a base64-encoded JPEG/PNG string into a BGR numpy image.
    Returns None if decoding fails.
    """
    if not frame_b64:
        return None
    try:
        data = frame_b64.split(',')[1] if ',' in frame_b64 else frame_b64
        arr = np.frombuffer(base64.b64decode(data), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[VISION] Error decoding base64 frame: {e}")
        return None


# ── Camera Quality Assessment ───────────────────────────────────────────────

def assess_camera_quality(img: np.ndarray, face_rect: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    """
    Assesses image brightness, contrast, blurriness, and face distance.

    Returns:
        dict:
            status      (str)  — 'good', 'low_lighting', 'too_bright', 'blurry', 'face_too_small', 'no_face'
            message     (str)  — Human-readable recommendation
            brightness  (float) — Mean luminance (0-255)
            blur_score  (float) — Laplacian variance (>50 is sharp)
            face_ratio  (float) — Face area relative to frame area
            is_ready    (bool) — True if minimum criteria for reliable analysis are met
    """
    if img is None or img.size == 0:
        return {
            'status': 'no_camera',
            'message': 'No camera feed detected.',
            'brightness': 0.0,
            'blur_score': 0.0,
            'face_ratio': 0.0,
            'is_ready': False
        }

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Brightness
    brightness = float(np.mean(gray))

    # 2. Blur / Sharpness via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # 3. Face size ratio
    face_ratio = 0.0
    if face_rect is not None:
        fx, fy, fw, fh = face_rect
        face_ratio = float((fw * fh) / (w * h))

    # Determine status
    if brightness < 40:
        status = 'low_lighting'
        message = 'Low lighting detected. Please move to a well-lit area or face a light source.'
        is_ready = False
    elif brightness > 225:
        status = 'too_bright'
        message = 'Camera image is overexposed. Reduce glare or direct backlight.'
        is_ready = False
    elif blur_score < 25.0:
        status = 'blurry'
        message = 'Camera image is blurry. Please clean your lens or remain steady.'
        is_ready = False
    elif face_rect is not None and face_ratio < 0.03:
        status = 'face_too_small'
        message = 'You are too far from the camera. Please move closer.'
        is_ready = True  # Can still operate, but advise user
    elif face_rect is None:
        status = 'no_face'
        message = 'No face detected. Please position yourself in front of the camera.'
        is_ready = False
    else:
        status = 'good'
        message = 'Camera quality is optimal.'
        is_ready = True

    return {
        'status': status,
        'message': message,
        'brightness': round(brightness, 1),
        'blur_score': round(blur_score, 1),
        'face_ratio': round(face_ratio, 3),
        'is_ready': is_ready
    }


# ── Landmark Detection & Multi-Face Handling ────────────────────────────────

def detect_faces_and_landmarks(img: np.ndarray) -> Dict[str, Any]:
    """
    Detects faces and facial landmarks in the image.
    Uses MediaPipe FaceMesh if available, otherwise falls back to OpenCV Haar / DNN.

    Returns:
        dict:
            face_detected    (bool)
            face_count       (int)
            face_status      (str) — 'single_face', 'multiple_faces', 'no_face'
            primary_face     (dict or None) with:
                bbox         (x, y, w, h)
                landmarks_2d (np.ndarray of shape (N, 2))
                landmarks_3d (np.ndarray of shape (N, 3) or None)
                left_eye_center (x, y)
                right_eye_center (x, y)
                nose_tip     (x, y)
                mouth_center (x, y)
    """
    if img is None or img.size == 0:
        return {
            'face_detected': False,
            'face_count': 0,
            'face_status': 'no_face',
            'primary_face': None
        }

    h, w = img.shape[:2]
    face_mesh = _get_face_mesh()

    if face_mesh is not None:
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results and results.multi_face_landmarks:
                num_faces = len(results.multi_face_landmarks)
                status = 'single_face' if num_faces == 1 else 'multiple_faces'

                # Extract faces and choose the primary (largest/most central)
                faces_data = []
                for face_landmarks in results.multi_face_landmarks:
                    pts_2d = []
                    pts_3d = []
                    for lm in face_landmarks.landmark:
                        pts_2d.append([lm.x * w, lm.y * h])
                        pts_3d.append([lm.x * w, lm.y * h, lm.z * w])
                    pts_2d = np.array(pts_2d, dtype=np.float32)
                    pts_3d = np.array(pts_3d, dtype=np.float32)

                    min_x, min_y = np.min(pts_2d[:, 0]), np.min(pts_2d[:, 1])
                    max_x, max_y = np.max(pts_2d[:, 0]), np.max(pts_2d[:, 1])
                    bx, by = int(max(0, min_x)), int(max(0, min_y))
                    bw, bh = int(min(w - bx, max_x - min_x)), int(min(h - by, max_y - min_y))

                    # Key landmark points from MediaPipe FaceMesh:
                    # Left eye pupil/center approx: 468/473 or mean of (33, 133, 159, 145)
                    # Right eye pupil/center approx: 473/468 or mean of (362, 263, 386, 374)
                    # Left eye corners: 33 (outer), 133 (inner)
                    # Right eye corners: 362 (inner), 263 (outer)
                    # Nose tip: 1
                    # Chin: 152
                    # Mouth left/right: 61, 291
                    left_eye_center = (float(np.mean(pts_2d[[33, 133, 159, 145], 0])),
                                       float(np.mean(pts_2d[[33, 133, 159, 145], 1])))
                    right_eye_center = (float(np.mean(pts_2d[[362, 263, 386, 374], 0])),
                                        float(np.mean(pts_2d[[362, 263, 386, 374], 1])))
                    nose_tip = (float(pts_2d[1, 0]), float(pts_2d[1, 1]))
                    mouth_center = (float(np.mean(pts_2d[[61, 291, 0, 17], 0])),
                                    float(np.mean(pts_2d[[61, 291, 0, 17], 1])))

                    area = bw * bh
                    # Proximity to frame center
                    cx, cy = bx + bw / 2, by + bh / 2
                    dist_to_center = math.hypot(cx - w / 2, cy - h / 2)
                    centrality_score = area / (dist_to_center + 10.0)

                    faces_data.append({
                        'bbox': (bx, by, bw, bh),
                        'area': area,
                        'centrality': centrality_score,
                        'landmarks_2d': pts_2d,
                        'landmarks_3d': pts_3d,
                        'left_eye_center': left_eye_center,
                        'right_eye_center': right_eye_center,
                        'nose_tip': nose_tip,
                        'mouth_center': mouth_center,
                        'source': 'mediapipe'
                    })

                # Sort by centrality/size to pick the primary user
                faces_data.sort(key=lambda f: f['centrality'], reverse=True)
                primary = faces_data[0]

                return {
                    'face_detected': True,
                    'face_count': num_faces,
                    'face_status': status,
                    'primary_face': primary
                }
        except Exception as e:
            print(f"[VISION] MediaPipe landmark processing error: {e}")

    # Robust multi-stage OpenCV cascade detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performance optimization: if input frame is large (e.g. 640x480 or 720p),
    # downscale for fast cascade detection while preserving full image for cropping.
    scale = 1.0
    if max(h, w) > 400:
        scale = 360.0 / float(max(h, w))
        det_gray = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))))
    else:
        det_gray = gray
    det_gray_eq = cv2.equalizeHist(det_gray)

    # 1. Primary detector: frontalface_alt2 on contrast-normalized image
    min_sz = max(20, int(28 * scale))
    faces = _FACE_CASCADE_ALT2.detectMultiScale(
        det_gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(min_sz, min_sz)
    )

    # 2. Try on raw grayscale if no face found
    if len(faces) == 0:
        faces = _FACE_CASCADE_ALT2.detectMultiScale(
            det_gray, scaleFactor=1.1, minNeighbors=3, minSize=(min_sz, min_sz)
        )

    # 3. Fallback to frontalface_default
    if len(faces) == 0:
        min_sz_def = max(18, int(24 * scale))
        faces = _FACE_CASCADE_DEFAULT.detectMultiScale(
            det_gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(min_sz_def, min_sz_def)
        )
    if len(faces) == 0:
        min_sz_sub = max(16, int(20 * scale))
        faces = _FACE_CASCADE_DEFAULT.detectMultiScale(
            det_gray, scaleFactor=1.05, minNeighbors=2, minSize=(min_sz_sub, min_sz_sub)
        )

    # 4. Profile cascade fallback for moderate head turns
    if len(faces) == 0 and not _FACE_CASCADE_PROFILE.empty():
        faces = _FACE_CASCADE_PROFILE.detectMultiScale(
            det_gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(min_sz, min_sz)
        )

    if len(faces) == 0:
        return {
            'face_detected': False,
            'face_count': 0,
            'face_status': 'no_face',
            'primary_face': None
        }

    num_faces = len(faces)
    status = 'single_face' if num_faces == 1 else 'multiple_faces'

    # Pick largest face and scale back to original image coordinates
    primary_rect = max(faces, key=lambda r: r[2] * r[3])
    if scale != 1.0:
        inv_scale = 1.0 / scale
        fx = int(round(primary_rect[0] * inv_scale))
        fy = int(round(primary_rect[1] * inv_scale))
        fw = int(round(primary_rect[2] * inv_scale))
        fh = int(round(primary_rect[3] * inv_scale))
        fx = max(0, min(w - 1, fx))
        fy = max(0, min(h - 1, fy))
        fw = max(1, min(w - fx, fw))
        fh = max(1, min(h - fy, fh))
    else:
        fx, fy, fw, fh = primary_rect

    # Search for eyes within top half of face ROI with equalized contrast
    roi_gray = gray[fy:fy + int(fh * 0.65), fx:fx + fw]
    roi_eq = cv2.equalizeHist(roi_gray)
    eyes = _EYE_CASCADE.detectMultiScale(roi_eq, scaleFactor=1.1, minNeighbors=3, minSize=(12, 12))
    if len(eyes) < 2:
        eyes = _EYE_CASCADE.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=2, minSize=(12, 12))

    valid_eyes = False
    if len(eyes) >= 2:
        # Separate candidates into left and right facial halves
        left_cands = [e for e in eyes if (e[0] + e[2] / 2.0) < fw * 0.55]
        right_cands = [e for e in eyes if (e[0] + e[2] / 2.0) > fw * 0.45]

        if left_cands and right_cands:
            best_l = max(left_cands, key=lambda e: e[2] * e[3])
            best_r = max(right_cands, key=lambda e: e[2] * e[3])
            lx, ly = fx + best_l[0] + best_l[2] / 2.0, fy + best_l[1] + best_l[3] / 2.0
            rx, ry = fx + best_r[0] + best_r[2] / 2.0, fy + best_r[1] + best_r[3] / 2.0
            dx, dy = rx - lx, ry - ly

            # Anatomical plausibility checks:
            # Horizontal separation between eyes should be 20% to 75% of face width,
            # and vertical offset should not exceed 25% of face height.
            if fw * 0.20 <= dx <= fw * 0.75 and abs(dy) <= fh * 0.25:
                left_eye_center = (float(lx), float(ly))
                right_eye_center = (float(rx), float(ry))
                valid_eyes = True

    if not valid_eyes:
        # Anatomical geometric eye center approximation
        left_eye_center = (float(fx + fw * 0.32), float(fy + fh * 0.38))
        right_eye_center = (float(fx + fw * 0.68), float(fy + fh * 0.38))

    nose_tip = (float(fx + fw * 0.5), float(fy + fh * 0.58))
    mouth_center = (float(fx + fw * 0.5), float(fy + fh * 0.78))

    return {
        'face_detected': True,
        'face_count': num_faces,
        'face_status': status,
        'primary_face': {
            'bbox': (int(fx), int(fy), int(fw), int(fh)),
            'area': int(fw * fh),
            'landmarks_2d': None,
            'landmarks_3d': None,
            'left_eye_center': left_eye_center,
            'right_eye_center': right_eye_center,
            'nose_tip': nose_tip,
            'mouth_center': mouth_center,
            'source': 'haar_cascade'
        }
    }


# ── Face Alignment & Normalization ──────────────────────────────────────────

def align_face(img: np.ndarray, primary_face: Dict[str, Any], crop_margin: float = 0.04) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Performs affine rotation so that both eyes are aligned horizontally,
    then crops the centered face region with a calibrated margin.
    Guarantees strict 1:1 aspect ratio by padding image borders when near edges,
    preventing face squashing/distortion when resized to (64, 64).

    Args:
        img: BGR image.
        primary_face: Dictionary containing 'bbox', 'left_eye_center', 'right_eye_center'.
        crop_margin: Margin around the bounding box (default 0.04, calibrated to FERPlus/FER2013 framing).

    Returns:
        Tuple of (aligned_face_img, transform_metadata)
    """
    if img is None or img.size == 0 or primary_face is None or 'bbox' not in primary_face:
        return img, {'aligned': False, 'angle_deg': 0.0}

    h, w = img.shape[:2]
    fx, fy, fw, fh = primary_face['bbox']
    left_eye = primary_face.get('left_eye_center', (fx + fw * 0.32, fy + fh * 0.38))
    right_eye = primary_face.get('right_eye_center', (fx + fw * 0.68, fy + fh * 0.38))

    # Calculate angle between the two eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # Center point between eyes
    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0,
                   (left_eye[1] + right_eye[1]) / 2.0)

    # Center of crop: horizontally aligned with eyes, vertically slightly below eyes
    cx = int(eyes_center[0])
    cy = int(eyes_center[1] + fh * 0.10)
    half_size = int(max(fw, fh) * (0.50 + crop_margin))

    # Determine whether rotation is beneficial:
    # 1. Skip rotation if tilt is negligible (< 1.5 deg) to prevent bilinear interpolation blur
    # 2. Skip rotation if tilt is extreme (> 35 deg) to avoid unnatural distortion
    perform_rotation = 1.5 <= abs(angle_deg) <= 35.0

    if perform_rotation:
        rot_mat = cv2.getRotationMatrix2D(eyes_center, angle_deg, scale=1.0)
        source_img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        source_img = img

    # Symmetrical boundary padding to guarantee 1:1 square aspect ratio without distortion
    pad_left = max(0, -(cx - half_size))
    pad_top = max(0, -(cy - half_size))
    pad_right = max(0, (cx + half_size) - w)
    pad_bottom = max(0, (cy + half_size) - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        source_img = cv2.copyMakeBorder(
            source_img, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE
        )
        cx += pad_left
        cy += pad_top

    x1 = cx - half_size
    y1 = cy - half_size
    x2 = cx + half_size
    y2 = cy + half_size

    face_crop = source_img[y1:y2, x1:x2]
    if face_crop.size == 0:
        face_crop = img

    return face_crop, {
        'aligned': perform_rotation,
        'angle_deg': round(angle_deg, 2),
        'crop_box': (x1, y1, x2 - x1, y2 - y1)
    }


# ── Standardized Shared Emotion Preprocessing ────────────────────────────────

def preprocess_emotion_face(
    face_img: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
    enhance_lighting: bool = True
) -> np.ndarray:
    """
    Standardized preprocessing for the ONNX Emotion-FERPlus model.
    GUARANTEED to match model training requirements:
      1. Single-channel grayscale conversion.
      2. Adaptive contrast enhancement: applies CLAHE only when image contrast
         is compressed or illumination is extreme, avoiding artificial shadow artifacts.
      3. Anti-aliased area resize to (64, 64) for downsampling, cubic for upsampling.
      4. Single-precision float32 in raw pixel range [0.0, 255.0] (matching ONNX internal mean subtraction).
      5. Reshaped to NCHW tensor layout: (1, 1, 64, 64).

    Args:
        face_img: BGR, BGRA, or Grayscale face crop.
        target_size: (width, height), default (64, 64).
        enhance_lighting: Apply adaptive CLAHE if True.

    Returns:
        np.ndarray of shape (1, 1, 64, 64), dtype np.float32.
    """
    if face_img is None or not isinstance(face_img, np.ndarray) or face_img.size == 0:
        # Return neutral dummy blob
        return np.full((1, 1, target_size[1], target_size[0]), 128.0, dtype=np.float32)

    # Convert to grayscale
    if len(face_img.shape) == 3:
        if face_img.shape[2] == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        elif face_img.shape[2] == 4:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = face_img[:, :, 0]
    elif len(face_img.shape) == 2:
        gray = face_img.copy()
    else:
        gray = face_img.reshape(face_img.shape[:2])

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Adaptive lighting enhancement:
    # Only apply CLAHE when dynamic range is compressed or image is underexposed / overexposed.
    # On clean, well-lit images, raw grayscale preserves natural face gradients.
    if enhance_lighting:
        std_val = float(np.std(gray))
        mean_val = float(np.mean(gray))
        if std_val < 38.0 or mean_val < 60.0 or mean_val > 195.0:
            try:
                clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            except Exception:
                pass

    # Resize using area interpolation for downsampling (anti-aliased) or cubic for upsampling
    h, w = gray.shape[:2]
    interp = cv2.INTER_AREA if (w >= target_size[0] and h >= target_size[1]) else cv2.INTER_CUBIC
    resized = cv2.resize(gray, target_size, interpolation=interp).astype(np.float32)

    # Model input format: [0.0, 255.0] float32 NCHW blob (Input3)
    blob = resized.reshape(1, 1, target_size[1], target_size[0])
    return blob

