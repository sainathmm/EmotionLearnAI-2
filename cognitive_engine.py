"""
cognitive_engine.py — Explainable Multimodal Cognitive-State Engine
====================================================================
Synthesizes multimodal biometric and pedagogical telemetry:
  - Facial Emotion (valence, confidence, stability, uncertainty)
  - Attention & Gaze (attention score, 3D head pose, EAR eye state, blink rate)
  - Quiz Performance (recent accuracy, session accuracy, streaks, response times)
  - Subject Context (weak topic status, current difficulty)

Produces 8 well-defined cognitive states with explicit, human-readable
explainability reasons:
  - FLOW, ENGAGED, DISTRACTED, OVERLOADED, STRUGGLING, BORED, ANXIOUS, UNCERTAIN
"""

from typing import Dict, Any, List, Optional, Tuple

COGNITIVE_STATE_METADATA = {
    'flow': {
        'name': 'Flow',
        'desc': 'Optimal Learning Zone — Deep focus and high mastery.',
        'color': '#00C851',
        'emoji': '🌟'
    },
    'engaged': {
        'name': 'Engaged',
        'desc': 'Actively Processing & Thinking — Steady cognitive effort.',
        'color': '#0099CC',
        'emoji': '🎯'
    },
    'distracted': {
        'name': 'Distracted',
        'desc': 'Attention Lapsed — Looking away or visual focus lost.',
        'color': '#FFBB33',
        'emoji': '⚠️'
    },
    'overloaded': {
        'name': 'Overloaded',
        'desc': 'Cognitively Overwhelmed — High task difficulty or fatigue.',
        'color': '#FF4444',
        'emoji': '⚡'
    },
    'struggling': {
        'name': 'Struggling',
        'desc': 'Conceptual Obstacle — Needs scaffolding, hints, or review.',
        'color': '#E040FB',
        'emoji': '🧩'
    },
    'bored': {
        'name': 'Bored',
        'desc': 'Under-Challenged — Material may be too easy.',
        'color': '#FF8800',
        'emoji': '💡'
    },
    'anxious': {
        'name': 'Anxious',
        'desc': 'Performance Tension — Elevated stress or uncertainty.',
        'color': '#AA00FF',
        'emoji': '🌀'
    },
    'uncertain': {
        'name': 'Stabilizing',
        'desc': 'Gathering Multimodal Signals — Retaining stable baseline.',
        'color': '#78909C',
        'emoji': '🔄'
    }
}

# Expected response times per difficulty for normalized tempo evaluation (seconds)
EXPECTED_RESPONSE_TIMES = {
    'easy': 12.0,
    'medium': 20.0,
    'hard': 32.0
}


def evaluate_cognitive_state(
    emotion_data: Dict[str, Any],
    attention_data: Dict[str, Any],
    performance_data: Optional[Dict[str, Any]] = None,
    previous_state: Optional[str] = None
) -> Dict[str, Any]:
    """
    Multimodal rule + heuristic cognitive-state deduction with explicit explainability.

    Args:
        emotion_data: dict from emotion_engine.infer_emotion()
        attention_data: dict from attention.detect_attention()
        performance_data: optional dict with:
            recent_accuracy (float 0.0 - 1.0)
            session_accuracy (float 0.0 - 1.0)
            consecutive_wrong (int)
            consecutive_correct (int)
            last_response_time (float)
            total_questions (int)
            difficulty (str: 'easy', 'medium', 'hard')
            is_weak_topic (bool)
        previous_state: last stable cognitive state

    Returns:
        dict:
            state           (str)   — e.g. 'flow', 'engaged', 'distracted', etc.
            name            (str)   — Display name
            confidence      (float) — Confidence in deduction (0.0 - 1.0)
            reasons         (list)  — List of explanation strings
            color           (str)   — UI color hex
            emoji           (str)   — UI emoji symbol
            desc            (str)   — Detailed description
    """
    perf = performance_data or {}
    recent_acc = float(perf.get('recent_accuracy', 0.65))
    session_acc = float(perf.get('session_accuracy', 0.65))
    cons_wrong = int(perf.get('consecutive_wrong', 0))
    cons_correct = int(perf.get('consecutive_correct', 0))
    time_taken = float(perf.get('last_response_time', 0.0))
    total_q = int(perf.get('total_questions', 0))
    diff = perf.get('difficulty', 'medium')
    is_weak = bool(perf.get('is_weak_topic', False))

    exp_time = EXPECTED_RESPONSE_TIMES.get(diff, 20.0)

    # Telemetry signals
    face_detected = bool(emotion_data.get('face_detected', False))
    face_status = emotion_data.get('face_status', 'single_face')
    emotion = emotion_data.get('emotion', 'neutral')
    emo_conf = float(emotion_data.get('confidence', 0.60))
    emo_stability = float(emotion_data.get('stability', 0.60))
    is_emo_uncertain = bool(emotion_data.get('is_uncertain', False))

    att_score = int(attention_data.get('attention_score', 65))
    looking_away = bool(attention_data.get('looking_away', False))
    eye_state = attention_data.get('eye_state', 'open')
    blink_rate = float(attention_data.get('blink_rate', 16.0))
    head_pose = attention_data.get('head_pose', {})
    head_forward = bool(head_pose.get('looking_forward', True))

    reasons: List[str] = []
    state = 'engaged'
    confidence = 0.75

    # ── 1. No Face or Multiple Faces Detected ────────────────────────────────
    if not face_detected or face_status == 'no_face':
        state = 'distracted'
        confidence = 0.85
        reasons.append('No student face detected in camera viewport.')
        if total_q > 0:
            reasons.append(f'Quiz activity: {total_q} questions answered.')
        meta = COGNITIVE_STATE_METADATA[state]
        return {
            'state': state,
            'name': meta['name'],
            'confidence': confidence,
            'reasons': reasons,
            'color': meta['color'],
            'emoji': meta['emoji'],
            'desc': meta['desc']
        }

    if face_status == 'multiple_faces':
        reasons.append('Multiple faces in camera view — tracking primary learner.')

    # ── 2. Severe Inattention / Distraction ──────────────────────────────────
    if att_score < 40 or looking_away or not head_forward or eye_state == 'closed':
        state = 'distracted'
        confidence = 0.88
        if looking_away or not head_forward:
            reasons.append(f'Head turned away from screen (Direction: {head_pose.get("direction", "aside")}).')
        if eye_state == 'closed':
            reasons.append('Prolonged eye closure / lack of visual focus.')
        reasons.append(f'Low composite attention score ({att_score}/100).')
        meta = COGNITIVE_STATE_METADATA[state]
        return {
            'state': state,
            'name': meta['name'],
            'confidence': confidence,
            'reasons': reasons,
            'color': meta['color'],
            'emoji': meta['emoji'],
            'desc': meta['desc']
        }

    # Student optional feedback
    feedback = str(perf.get('student_feedback') or perf.get('feedback') or '').strip().lower()

    # ── 3. High Stress / Anxiety ─────────────────────────────────────────────
    # Requires repeated mistakes (cons_wrong >= 2) or explicit feedback, and confirmed confident expression
    has_fear_cue = (emotion == 'fear' and not is_emo_uncertain and emo_conf >= 0.50)
    if feedback in ['anxious', 'stressed'] or ((blink_rate > 38.0 or has_fear_cue) and cons_wrong >= 2):
        state = 'anxious'
        confidence = 0.82
        if feedback in ['anxious', 'stressed']:
            reasons.append(f'Student reported feeling {feedback}.')
        if blink_rate > 38.0:
            reasons.append(f'Elevated blink frequency ({blink_rate} blinks/min indicates tension).')
        if has_fear_cue:
            reasons.append(f'Tense facial expression detected (Confidence: {int(emo_conf*100)}%).')
        if cons_wrong >= 2:
            reasons.append(f'Repeated missed questions ({cons_wrong} in a row on {diff.title()}).')
        meta = COGNITIVE_STATE_METADATA[state]
        return {
            'state': state,
            'name': meta['name'],
            'confidence': confidence,
            'reasons': reasons,
            'color': meta['color'],
            'emoji': meta['emoji'],
            'desc': meta['desc']
        }

    # ── 4. Overloaded / Strained Cognitive State ─────────────────────────────
    # High response time + repeated mistakes OR explicit feedback (never on single mistake or slow response alone)
    is_time_over = (time_taken > exp_time * 1.8) if time_taken > 0 else False
    has_strain_cue = (emotion in ['angry', 'sad', 'fear'] and not is_emo_uncertain and emo_conf >= 0.55)
    if feedback in ['overloaded', 'overwhelmed'] or ((cons_wrong >= 2 or (recent_acc < 0.35 and total_q >= 3)) and (is_time_over or has_strain_cue)):
        state = 'overloaded'
        confidence = 0.84
        if feedback in ['overloaded', 'overwhelmed']:
            reasons.append(f'Student indicated high cognitive load ({feedback}).')
        else:
            reasons.append(f'Repeated mistakes: {cons_wrong} consecutive wrong (accuracy: {int(recent_acc*100)}%).')
            if is_time_over:
                reasons.append(f'Response time ({time_taken:.1f}s) significantly exceeded baseline ({exp_time:.0f}s).')
            if has_strain_cue:
                reasons.append(f'Strained facial expression detected ({emotion.title()}).')
        meta = COGNITIVE_STATE_METADATA[state]
        return {
            'state': state,
            'name': meta['name'],
            'confidence': confidence,
            'reasons': reasons,
            'color': meta['color'],
            'emoji': meta['emoji'],
            'desc': meta['desc']
        }

    # ── 5. Conceptual Struggle (Needs Guidance) ──────────────────────────────
    if feedback in ['confused', 'struggling', 'need_help'] or ((cons_wrong >= 2 or recent_acc < 0.45) and att_score >= 55):
        state = 'struggling'
        confidence = 0.85
        if feedback in ['confused', 'struggling', 'need_help']:
            reasons.append(f'Student requested help/scaffolding ({feedback}).')
        else:
            reasons.append(f'Attentive ({att_score}/100) but encountering conceptual hurdle ({cons_wrong} wrong on {diff.title()}).')
            reasons.append(f'Recent accuracy: {int(recent_acc*100)}%.')
            if is_weak:
                reasons.append('Practicing a flagged weak topic.')
        meta = COGNITIVE_STATE_METADATA[state]
        return {
            'state': state,
            'name': meta['name'],
            'confidence': confidence,
            'reasons': reasons,
            'color': meta['color'],
            'emoji': meta['emoji'],
            'desc': meta['desc']
        }

    # ── 6. Under-Challenged / Boredom ────────────────────────────────────────
    # High accuracy and fast answering on easy questions (never triggered by neutral face or single quick answer)
    is_fast = (0 < time_taken < exp_time * 0.4) if time_taken > 0 else False
    has_bored_face = (emotion == 'sad' and not is_emo_uncertain and emo_conf >= 0.60 and att_score < 60)
    if feedback in ['bored', 'too_easy'] or (recent_acc >= 0.85 and diff == 'easy' and total_q >= 4 and is_fast) or (has_bored_face and recent_acc >= 0.85):
        state = 'bored'
        confidence = 0.80
        if feedback in ['bored', 'too_easy']:
            reasons.append(f'Student feedback: content is {feedback}.')
        else:
            reasons.append(f'Consistently high mastery ({int(recent_acc*100)}%) on {diff.title()} questions with rapid solving.')
            reasons.append('Material is mastered; student is ready for higher difficulty.')
        meta = COGNITIVE_STATE_METADATA[state]
        return {
            'state': state,
            'name': meta['name'],
            'confidence': confidence,
            'reasons': reasons,
            'color': meta['color'],
            'emoji': meta['emoji'],
            'desc': meta['desc']
        }

    # ── 7. Flow State (Optimal Learning Zone) ────────────────────────────────
    if att_score >= 75 and (recent_acc >= 0.75 or cons_correct >= 2) and emotion in ['happy', 'neutral', 'surprise'] and emo_stability >= 0.45:
        state = 'flow'
        confidence = 0.88
        reasons.append(f'High sustained attention ({att_score}/100) and forward gaze.')
        reasons.append(f'Strong performance: {int(recent_acc*100)}% recent accuracy.')
        reasons.append(f'Positive/calm facial valence ({emotion.title()}, Stability: {int(emo_stability*100)}%).')
        meta = COGNITIVE_STATE_METADATA[state]
        return {
            'state': state,
            'name': meta['name'],
            'confidence': confidence,
            'reasons': reasons,
            'color': meta['color'],
            'emoji': meta['emoji'],
            'desc': meta['desc']
        }

    # ── 8. Active Engagement (Default Positive State) ────────────────────────
    if att_score >= 50:
        state = 'engaged'
        confidence = 0.80
        reasons.append(f'Steady attention ({att_score}/100) facing display.')
        if total_q > 0:
            reasons.append(f'Overall session accuracy: {int(session_acc*100)}% across {total_q} questions.')
        reasons.append(f'Facial valence: {emotion.title()} (Confidence: {int(emo_conf*100)}%).')
        meta = COGNITIVE_STATE_METADATA[state]
        return {
            'state': state,
            'name': meta['name'],
            'confidence': confidence,
            'reasons': reasons,
            'color': meta['color'],
            'emoji': meta['emoji'],
            'desc': meta['desc']
        }

    # ── 9. Uncertain / Stabilizing Baseline ──────────────────────────────────
    if is_emo_uncertain and previous_state:
        state = previous_state
        confidence = 0.60
        reasons.append('Retaining previous cognitive state while stabilizing sensory telemetry.')
    else:
        state = 'engaged'
        confidence = 0.70
        reasons.append('Active session initialized.')

    meta = COGNITIVE_STATE_METADATA[state]
    return {
        'state': state,
        'name': meta['name'],
        'confidence': confidence,
        'reasons': reasons,
        'color': meta['color'],
        'emoji': meta['emoji'],
        'desc': meta['desc']
    }
