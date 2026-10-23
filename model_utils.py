"""
model_utils.py — Multimodal Learning Behavior Prediction Utilities
===================================================================
Provides predict_next_correct() and metadata helpers for the Flask app.
Supports multimodal telemetry inputs (facial valence, confidence, stability,
attention score, 3D head pose, blink rate, recent accuracy, response time, difficulty).

Design principles:
  - Model is loaded once and cached in memory (_MODEL_CACHE).
  - Robust fallback dictionary returned if model is unavailable.
  - Dynamic adaptation to both legacy 3-feature models and multimodal 10-feature models.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'performance_model.pkl')

# ── Feature constants ─────────────────────────────────────────────────────────
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
DIFFICULTIES = ['easy', 'medium', 'hard']

_MODEL_CACHE: Optional[Dict[str, Any]] = None


def _load_model() -> Optional[Dict[str, Any]]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        _MODEL_CACHE = joblib.load(MODEL_PATH)
        acc_pct = round(_MODEL_CACHE.get('accuracy', 0) * 100, 1)
        model_type = _MODEL_CACHE.get('model_type', 'ML Model')
        print(f"[ML] Model loaded: {model_type} | Test Accuracy: {acc_pct}%")
        return _MODEL_CACHE
    except Exception as e:
        print(f"[ML ERROR] Failed to load model bundle: {e}")
        return None


def _safe_encode(encoder, value: str, known_values: list, fallback: str) -> int:
    if value not in known_values:
        value = fallback
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        return 0


def _fallback_result() -> Dict[str, Any]:
    return {
        'available': False,
        'prediction': None,
        'will_correct': None,
        'confidence': None,
        'proba_correct': None,
        'proba_wrong': None,
        'suggested_difficulty': None,
    }


def model_available() -> bool:
    return _load_model() is not None


def predict_next_correct(
    emotion: str = 'neutral',
    time_taken: float = 15.0,
    difficulty: str = 'medium',
    emotion_confidence: float = 0.75,
    emotion_stability: float = 0.75,
    attention_score: float = 75.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    blink_rate: float = 16.0,
    recent_accuracy: float = 0.70
) -> Dict[str, Any]:
    """
    Predicts student correctness probability on the upcoming question.
    """
    bundle = _load_model()
    if bundle is None:
        return _fallback_result()

    try:
        clf = bundle['model']
        emo_enc = bundle['emotion_enc']
        diff_enc = bundle['diff_enc']
        features_list = bundle.get('features', ['emotion_enc', 'time_taken', 'difficulty_enc'])

        emo_idx = _safe_encode(emo_enc, emotion, EMOTIONS, 'neutral')
        diff_idx = _safe_encode(diff_enc, difficulty, DIFFICULTIES, 'medium')
        time_val = float(np.clip(time_taken, 0, 120))

        if len(features_list) == 3:
            # Legacy 3-feature input vector
            X = np.array([[emo_idx, time_val, diff_idx]], dtype=float)
        else:
            # Multimodal 10-feature input vector
            X = np.array([[
                emo_idx,
                float(np.clip(emotion_confidence, 0.0, 1.0)),
                float(np.clip(emotion_stability, 0.0, 1.0)),
                float(np.clip(attention_score, 0.0, 100.0)),
                float(np.clip(yaw, -45.0, 45.0)),
                float(np.clip(pitch, -45.0, 45.0)),
                float(np.clip(blink_rate, 0.0, 60.0)),
                float(np.clip(recent_accuracy, 0.0, 1.0)),
                time_val,
                diff_idx
            ]], dtype=float)

        prediction = int(clf.predict(X)[0])

        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(X)[0]
            classes = list(clf.classes_)
            proba_wrong = round(float(proba[classes.index(0)]), 3) if 0 in classes else 0.5
            proba_correct = round(float(proba[classes.index(1)]), 3) if 1 in classes else 0.5
            confidence = round(float(max(proba)), 3)
        else:
            proba_correct = 1.0 if prediction == 1 else 0.0
            proba_wrong = 1.0 - proba_correct
            confidence = 1.0

        if confidence < 0.58:
            suggested = difficulty  # Keep steady if uncertain
        elif prediction == 0:
            suggested = 'easy' if difficulty != 'easy' else 'easy'
        else:
            suggested = 'hard' if difficulty == 'medium' else ('medium' if difficulty == 'easy' else 'hard')

        return {
            'available': True,
            'prediction': prediction,
            'will_correct': bool(prediction == 1),
            'confidence': confidence,
            'proba_correct': proba_correct,
            'proba_wrong': proba_wrong,
            'suggested_difficulty': suggested,
        }

    except Exception as e:
        print(f"[ML ERROR] predict_next_correct error: {e}")
        return _fallback_result()


def get_model_info() -> Dict[str, Any]:
    bundle = _load_model()
    if bundle is None:
        return {
            'loaded': False,
            'model_type': None,
            'accuracy': None,
            'path': MODEL_PATH,
            'exists': os.path.exists(MODEL_PATH),
        }
    return {
        'loaded': True,
        'model_type': bundle.get('model_type'),
        'accuracy': round(bundle.get('accuracy', 0) * 100, 1),
        'f1_score': round(bundle.get('f1_score', bundle.get('accuracy', 0)) * 100, 1),
        'features': bundle.get('features'),
        'emotions': bundle.get('emotions'),
        'difficulties': bundle.get('difficulties'),
        'path': MODEL_PATH,
        'exists': True,
    }


def reload_model() -> bool:
    global _MODEL_CACHE
    _MODEL_CACHE = None
    return _load_model() is not None
