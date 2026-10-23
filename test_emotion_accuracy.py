"""
test_emotion_accuracy.py — Comprehensive Unit & Integration Tests for Emotion Detection
=======================================================================================
Verifies accuracy, stability, thresholding, temporal smoothing, and edge-case handling.
"""

import sys
import io
import time
import base64
import cv2
import numpy as np

# Ensure Windows console encoding compatibility
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from vision_pipeline import (
    preprocess_emotion_face,
    align_face,
    detect_faces_and_landmarks,
    decode_base64_frame,
    assess_camera_quality
)
from emotion_engine import (
    _MODEL_RUNNER,
    _softmax,
    infer_emotion,
    get_session_tracker,
    clear_session_tracker,
    TemporalEmotionTracker,
    ONNX_LABELS,
    PEDAGOGICAL_EMOTIONS,
    CONFIDENCE_THRESHOLD,
    set_confidence_threshold,
    get_confidence_threshold
)

PASS = "[PASS]"
FAIL = "[FAIL]"
test_results = {"passed": 0, "failed": 0}


def check(name, condition, msg=""):
    if condition:
        print(f"  {PASS} {name}")
        test_results["passed"] += 1
    else:
        print(f"  {FAIL} {name} : {msg}")
        test_results["failed"] += 1


def test_preprocessing():
    print("\n--- 1. Image Preprocessing & Format Tests ---")
    # Test BGR image
    bgr = np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8)
    blob_bgr = preprocess_emotion_face(bgr)
    check("BGR input produces (1, 1, 64, 64) float32",
          blob_bgr.shape == (1, 1, 64, 64) and blob_bgr.dtype == np.float32)

    # Test BGRA (4-channel) image
    bgra = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    blob_bgra = preprocess_emotion_face(bgra)
    check("BGRA (4-channel) input produces (1, 1, 64, 64)",
          blob_bgra.shape == (1, 1, 64, 64))

    # Test 2D Grayscale image
    gray = np.random.randint(0, 255, (80, 80), dtype=np.uint8)
    blob_gray = preprocess_emotion_face(gray)
    check("2D Grayscale input produces (1, 1, 64, 64)",
          blob_gray.shape == (1, 1, 64, 64))

    # Test empty / None handling
    blob_empty = preprocess_emotion_face(None)
    check("None input returns neutral fallback blob without crashing",
          blob_empty.shape == (1, 1, 64, 64) and np.isclose(blob_empty.mean(), 128.0))

    blob_zero = preprocess_emotion_face(np.empty((0, 0, 3), dtype=np.uint8))
    check("Empty 0-size array returns safe fallback blob",
          blob_zero.shape == (1, 1, 64, 64))


def test_model_runner_resilience():
    print("\n--- 2. Model Inference & Stability Tests ---")
    # Valid blob inference
    valid_blob = np.random.uniform(0.0, 255.0, (1, 1, 64, 64)).astype(np.float32)
    logits = _MODEL_RUNNER.predict(valid_blob)
    check("Valid blob produces 8 raw logits",
          len(logits) == 8 and np.all(np.isfinite(logits)))

    # Softmax sums to 1.0
    probs = _softmax(logits)
    check("Softmax yields valid probability distribution summing to 1.0",
          np.isclose(float(np.sum(probs)), 1.0, atol=1e-5) and np.all(probs >= 0.0))

    # Test NaN / Inf input resilience
    bad_blob = np.full((1, 1, 64, 64), np.nan, dtype=np.float32)
    nan_logits = _MODEL_RUNNER.predict(bad_blob)
    check("NaN input is handled gracefully returning finite logits",
          np.all(np.isfinite(nan_logits)))

    # Test wrong shape resilience
    wrong_shape = np.zeros((10, 10), dtype=np.float32)
    wrong_logits = _MODEL_RUNNER.predict(wrong_shape)
    check("Malformed input shape does not crash model runner",
          len(wrong_logits) == 8 and np.all(np.isfinite(wrong_logits)))


def test_confidence_and_thresholding():
    print("\n--- 3. Confidence Thresholding Tests ---")
    orig_th = get_confidence_threshold()
    set_confidence_threshold(0.40)
    check("Configurable confidence threshold setter/getter works",
          abs(get_confidence_threshold() - 0.40) < 1e-4)

    tracker = TemporalEmotionTracker(window_size=5)
    # Low confidence distribution (each class 0.125 < 0.40)
    low_dist = {e: 0.125 for e in PEDAGOGICAL_EMOTIONS}
    _, conf, _, is_unc = tracker.add_prediction("neutral", 0.125, low_dist, is_uncertain=True)
    check("Low confidence output flags is_uncertain=True", is_unc is True)

    # High confidence distribution on clean tracker
    tracker_high = TemporalEmotionTracker(window_size=5)
    high_dist = {'happy': 0.85, 'neutral': 0.05, 'sad': 0.02, 'angry': 0.02, 'fear': 0.02, 'surprise': 0.02, 'disgust': 0.02}
    _, conf_high, _, is_unc_high = tracker_high.add_prediction("happy", 0.85, high_dist, is_uncertain=False)
    check("High confidence output flags is_uncertain=False", is_unc_high is False and conf_high >= 0.40)

    # Restore threshold
    set_confidence_threshold(orig_th)


def test_temporal_smoothing_and_decay():
    print("\n--- 4. Temporal Smoothing & Face Jump / Stale State Decay ---")
    tracker = TemporalEmotionTracker(window_size=6)
    happy_dist = {'happy': 0.80, 'neutral': 0.10, 'sad': 0.02, 'angry': 0.02, 'fear': 0.02, 'surprise': 0.02, 'disgust': 0.02}

    # Feed consistent happy predictions with stable bbox
    for _ in range(5):
        tracker.add_prediction("happy", 0.80, happy_dist, is_uncertain=False, bbox=(100, 100, 80, 80))

    emo, conf, stab, _ = tracker.add_prediction("happy", 0.80, happy_dist, is_uncertain=False, bbox=(102, 101, 80, 80))
    check("Consistent predictions produce high stability score (>=0.60)",
          stab >= 0.60, f"Got stability {stab}")
    check("Smoothed emotion matches dominant stream", emo == "happy")

    # Simulate face displacement/switch (center moves from (142, 141) to (440, 440))
    tracker.add_prediction("sad", 0.70, {'sad': 0.70, 'neutral': 0.30}, is_uncertain=False, bbox=(400, 400, 80, 80))
    check("Face jump / displacement resets history buffer to prevent ghosting",
          len(tracker.history) == 1)

    # Simulate face disappearance
    tracker.last_face_time = time.time() - 2.5
    tracker.notify_no_face()
    check("Decay on missing face resets stale history to prevent indefinite display",
          len(tracker.history) == 0 and tracker.last_stable_confidence == 0.0)


def test_infer_emotion_contracts():
    print("\n--- 5. End-to-End API Contracts & Edge Cases ---")
    clear_session_tracker("test_contract")

    # 1. Invalid / empty frame
    res_none = infer_emotion(frame_b64="", session_id="test_contract")
    check("Empty frame returns face_detected=False", res_none['face_detected'] is False)
    check("Empty frame returns confidence=0.0", res_none['confidence'] == 0.0)
    check("Empty frame returns is_uncertain=True", res_none['is_uncertain'] is True)
    check("Empty frame returns emotion='neutral'", res_none['emotion'] == 'neutral')
    check("Empty frame returns predicted_expression='No Face Detected'", res_none['predicted_expression'] == 'No Face Detected')
    check("Empty frame returns stability_label='unstable'", res_none['stability_label'] == 'unstable')

    # 2. Corrupt base64 string
    res_corrupt = infer_emotion(frame_b64="data:image/jpeg;base64,not_real_data!!!", session_id="test_contract")
    check("Corrupt base64 does not crash and returns safe failure dict",
          res_corrupt['face_detected'] is False)

    # 3. Real face image (test_face.jpg)
    real_img = cv2.imread("test_face.jpg")
    if real_img is not None:
        res_face = infer_emotion(img=real_img, session_id="test_contract")
        check("Real face image detects face", res_face['face_detected'] is True)
        check("Real face returns non-zero confidence", res_face['confidence'] > 0.0)
        check("Real face returns predicted_expression", 'predicted_expression' in res_face and len(res_face['predicted_expression']) > 0)
        check("Real face returns all 7 pedagogical emotions in distribution",
              len(res_face['all']) == 7 and all(e in res_face['all'] for e in PEDAGOGICAL_EMOTIONS))
        check("Real face returns all 8 raw ONNX emotions in raw_all",
              len(res_face['raw_all']) == 8 and all(e in res_face['raw_all'] for e in ONNX_LABELS))
        check("Real face returns primary_face landmarks & bbox",
              res_face['primary_face'] is not None and 'bbox' in res_face['primary_face'])


def main():
    print("=" * 65)
    print("   EmotionLearnAI — Emotion Pipeline Verification Suite")
    print("=" * 65)

    test_preprocessing()
    test_model_runner_resilience()
    test_confidence_and_thresholding()
    test_temporal_smoothing_and_decay()
    test_infer_emotion_contracts()

    print("\n" + "=" * 65)
    total = test_results["passed"] + test_results["failed"]
    print(f"  SUMMARY: {test_results['passed']}/{total} passed | {test_results['failed']} failed")
    print("=" * 65)

    if test_results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
