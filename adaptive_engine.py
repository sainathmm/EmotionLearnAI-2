"""
adaptive_engine.py — Emotion-Aware Adaptive Learning & Difficulty Engine
========================================================================
Continuously evaluates student learning performance and selects the appropriate
next learning action.

Signals evaluated:
  1. Quiz accuracy (recent window and cumulative).
  2. Recent sequence of correct and incorrect answers.
  3. Response time compared with the student's historical baseline.
  4. Topic-wise mastery and learning history.
  5. Available emotion and attention signals (with confidence and reliability checks).
  6. Revision history and spaced repetition retention due flags.

Decisions:
  - increase_difficulty: student shows sustained mastery and reasonable pacing.
  - maintain_difficulty: baseline is still being established, mixed performance,
    or long response time with correct answers (building fluency without demotion).
  - reduce_difficulty: repeated struggle at hard/medium level requiring scaffolding.
  - guided_practice: foundation review with worked examples or hints.
  - explain_answer: concept explanation for incorrect answers.
  - hint_or_example: scaffolded assistance when struggling with complex questions.
  - recommend_revision: retention decline or weak topic due for spaced repetition.
  - suggest_break: repeated errors combined with reliable fatigue/stress indicators.

Guiding Principles:
  - Never change difficulty based on a single facial expression or one slow answer.
  - Never infer emotional state with absolute certainty.
  - Never punish students, reduce grades, or block learning due to emotion readings.
  - Track difficulty separately for each topic.
"""

from typing import Dict, Any, List, Optional, Tuple
import time

# ── Difficulty Progression ───────────────────────────────────────────────────
DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']

DIFFICULTY_RANK: Dict[str, int] = {
    'easy': 1,
    'medium': 2,
    'hard': 3,
}

RANK_TO_DIFFICULTY: Dict[int, str] = {
    1: 'easy',
    2: 'medium',
    3: 'hard',
}

DIFFICULTY_ALIASES: Dict[str, str] = {
    'beginner': 'easy',
    'easy': 'easy',
    'medium': 'medium',
    'intermediate': 'medium',
    'hard': 'hard',
    'expert': 'hard',
    'advanced': 'hard',
}

def normalize_difficulty(diff: Optional[str]) -> str:
    """Normalizes any difficulty label or alias into 'easy', 'medium', or 'hard'."""
    if not diff:
        return 'easy'
    cleaned = str(diff).strip().lower()
    return DIFFICULTY_ALIASES.get(cleaned, 'easy')


# ── Configurable Decision Thresholds ─────────────────────────────────────────
MIN_ATTEMPTS_BEFORE_CHANGE = 3       # Minimum attempts on a topic before level shift
HIGH_ACCURACY_THRESHOLD = 0.80      # >= 80% (e.g., at least 4 of 5 correct)
LOW_ACCURACY_THRESHOLD = 0.40       # <= 40% (e.g., 3 or more of 5 wrong)
SLOW_PACING_RATIO = 2.0             # Response time > 2.0x baseline indicates deliberate thinking
EMOTION_MIN_CONFIDENCE = 0.60       # Below this, emotion readings are deemed unreliable
EMOTION_MIN_STABILITY = 0.50        # Stability required to consider emotion steady
LOW_ATTENTION_THRESHOLD = 50        # Composite attention score below this is disengaged


# ── Topic-Specific State Store (in-memory + DB backed) ───────────────────────
class TopicDifficultyTracker:
    """
    Maintains per-topic performance metrics and difficulty state for a user or session.
    """

    def __init__(self, topic: str, initial_difficulty: str = 'easy'):
        self.topic: str = topic
        self.current_difficulty: str = normalize_difficulty(initial_difficulty)
        self.attempts_history: List[Dict[str, Any]] = []
        self.baseline_response_time: float = 12.0
        self.total_attempts: int = 0
        self.total_correct: int = 0
        self.consecutive_correct: int = 0
        self.consecutive_wrong: int = 0
        self.last_change_time: float = 0.0

    def record_attempt(self, correct: bool, time_taken: float, qid: Optional[str] = None):
        """Records a single question attempt and updates moving statistics."""
        self.total_attempts += 1
        is_c = 1 if correct else 0
        if correct:
            self.total_correct += 1
            self.consecutive_correct += 1
            self.consecutive_wrong = 0
        else:
            self.consecutive_wrong += 1
            self.consecutive_correct = 0

        # Update baseline response time using exponential moving average
        safe_time = max(0.5, min(float(time_taken), 300.0))
        if self.total_attempts == 1:
            self.baseline_response_time = safe_time
        else:
            # 80% history weight, 20% recent weight
            self.baseline_response_time = round(0.80 * self.baseline_response_time + 0.20 * safe_time, 2)

        self.attempts_history.append({
            'correct': is_c,
            'time_taken': safe_time,
            'timestamp': time.time(),
            'qid': qid,
        })
        if len(self.attempts_history) > 50:
            self.attempts_history.pop(0)

    def get_recent_accuracy(self, window: int = 5) -> float:
        """Returns accuracy over the last `window` attempts."""
        if not self.attempts_history:
            return 0.0
        recent = self.attempts_history[-window:]
        return sum(a['correct'] for a in recent) / len(recent)


# ── Central Learning Decision Function ───────────────────────────────────────

def evaluate_learning_decision(
    topic: str,
    current_difficulty: str = 'easy',
    total_topic_attempts: int = 0,
    recent_accuracy: float = 0.0,
    consecutive_correct: int = 0,
    consecutive_wrong: int = 0,
    current_response_time: float = 10.0,
    baseline_response_time: Optional[float] = None,
    topic_mastery: Optional[float] = None,
    emotion_data: Optional[Dict[str, Any]] = None,
    attention_data: Optional[Dict[str, Any]] = None,
    revision_due: bool = False,
    recent_attempts: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluates learning performance signals and selects an appropriate next learning action.

    Deterministic, highly testable decision function implementing:
      Priority 1: Correctness and demonstrated understanding
      Priority 2: Repeated mistakes and topic mastery
      Priority 3: Response-time trends relative to student baseline
      Priority 4: Revision and retention history
      Priority 5: Reliable supporting emotion/attention signals (never primary)

    Returns:
        Dict adhering to the standard adaptive decision schema:
        {
            "action": str,
            "topic": str,
            "current_difficulty": str,
            "recommended_difficulty": str,
            "reason": str,
            "support_type": str,
            "confidence": str,
            "signals_used": Dict[str, Any]
        }
    """
    current_diff = normalize_difficulty(current_difficulty)
    curr_rank = DIFFICULTY_RANK[current_diff]
    baseline_time = baseline_response_time or 12.0
    safe_resp_time = max(0.5, float(current_response_time))

    # Evaluate attempt counts from recent_attempts if provided
    if recent_attempts is not None and len(recent_attempts) > 0:
        total_topic_attempts = max(total_topic_attempts, len(recent_attempts))
        window_slice = recent_attempts[-5:]
        recent_accuracy = sum(1 for a in window_slice if a.get('correct')) / len(window_slice)

    # ── Reliability & Confidence Checks for Emotion / Attention ──
    reliable_emotion = False
    emotion_label = 'neutral'
    emotion_conf = 0.0

    if emotion_data and isinstance(emotion_data, dict):
        emotion_label = emotion_data.get('emotion', 'neutral')
        emotion_conf = float(emotion_data.get('confidence', 0.0))
        emotion_stab = float(emotion_data.get('stability', 1.0))
        is_uncertain = bool(emotion_data.get('is_uncertain', False))

        if emotion_conf >= EMOTION_MIN_CONFIDENCE and emotion_stab >= EMOTION_MIN_STABILITY and not is_uncertain:
            reliable_emotion = True

    att_score = 75
    is_disengaged = False
    if attention_data and isinstance(attention_data, dict):
        att_score = int(attention_data.get('attention_score', 75))
        is_disengaged = att_score < LOW_ATTENTION_THRESHOLD or bool(attention_data.get('looking_away', False))

    signals_summary = {
        'total_attempts': total_topic_attempts,
        'recent_accuracy': round(recent_accuracy, 2),
        'consecutive_correct': consecutive_correct,
        'consecutive_wrong': consecutive_wrong,
        'current_response_time': round(safe_resp_time, 1),
        'baseline_response_time': round(baseline_time, 1),
        'reliable_emotion': reliable_emotion,
        'emotion_label': emotion_label if reliable_emotion else 'unreliable/ignored',
        'attention_score': att_score,
        'revision_due': revision_due
    }

    # ── Rule 1: Minimum Attempts / Insufficient Data Baseline ────────────────
    if total_topic_attempts < MIN_ATTEMPTS_BEFORE_CHANGE:
        return {
            "action": "maintain_difficulty",
            "topic": topic,
            "current_difficulty": current_diff,
            "recommended_difficulty": current_diff,
            "reason": (
                f"Establishing baseline performance for {topic} "
                f"({total_topic_attempts}/{MIN_ATTEMPTS_BEFORE_CHANGE} attempts recorded)."
            ),
            "support_type": "none",
            "confidence": "low",
            "signals_used": signals_summary
        }

    # ── Rule 2: Repeated Mistakes + Reliable Stress / Fatigue -> Offer Break ─
    stress_emotions = {'angry', 'fear', 'sad'}
    is_stress_indicated = reliable_emotion and (emotion_label in stress_emotions)
    if consecutive_wrong >= 2 and (is_stress_indicated or is_disengaged):
        return {
            "action": "suggest_break",
            "topic": topic,
            "current_difficulty": current_diff,
            "recommended_difficulty": current_diff,
            "reason": (
                f"You've had a challenging few questions on {topic}. "
                "Would you like a short break or a worked example?"
            ),
            "support_type": "rest_break",
            "confidence": "high" if reliable_emotion else "medium",
            "signals_used": signals_summary
        }

    # ── Rule 3: Declining Retention / Due Spaced Repetition Review ───────────
    if revision_due and recent_accuracy < 0.60:
        return {
            "action": "recommend_revision",
            "topic": topic,
            "current_difficulty": current_diff,
            "recommended_difficulty": current_diff,
            "reason": (
                f"Your recent {topic} answers show some difficulty and review is scheduled. "
                "Review the concept summary, then try another question."
            ),
            "support_type": "concept_review",
            "confidence": "high",
            "signals_used": signals_summary
        }

    # ── Rule 4: High Accuracy with Long Response Time -> Maintain Level ──────
    # Important Requirement: Long response time with high accuracy MUST NOT lower difficulty!
    # The student is answering correctly; taking time shows thorough deliberation.
    if recent_accuracy >= HIGH_ACCURACY_THRESHOLD and safe_resp_time > (baseline_time * SLOW_PACING_RATIO):
        return {
            "action": "maintain_difficulty",
            "topic": topic,
            "current_difficulty": current_diff,
            "recommended_difficulty": current_diff,
            "reason": (
                f"Solid accuracy on {topic}! You are taking time to think questions through thoroughly. "
                f"Continuing at {current_diff.title()} to build fluency before advancing."
            ),
            "support_type": "none",
            "confidence": "high",
            "signals_used": signals_summary
        }

    # ── Rule 5: Sustained High Performance -> Increase Difficulty ───────────
    if recent_accuracy >= HIGH_ACCURACY_THRESHOLD or consecutive_correct >= 4:
        if curr_rank < 3:
            next_rank = curr_rank + 1
            next_diff = RANK_TO_DIFFICULTY[next_rank]
            return {
                "action": "increase_difficulty",
                "topic": topic,
                "current_difficulty": current_diff,
                "recommended_difficulty": next_diff,
                "reason": (
                    f"Great progress! You answered {int(recent_accuracy * 100)}% of your recent questions "
                    f"correctly on {topic}. Let's try a more challenging {next_diff.title()} question."
                ),
                "support_type": "none",
                "confidence": "high",
                "signals_used": signals_summary
            }
        else:
            return {
                "action": "maintain_difficulty",
                "topic": topic,
                "current_difficulty": current_diff,
                "recommended_difficulty": current_diff,
                "reason": (
                    f"Mastery demonstrated! You have reached and maintained {current_diff.title()} level "
                    f"with {int(recent_accuracy * 100)}% accuracy on {topic}."
                ),
                "support_type": "none",
                "confidence": "high",
                "signals_used": signals_summary
            }

    # ── Rule 6: Low Accuracy / Repeated Errors -> Reduce or Guided Practice ──
    if recent_accuracy <= LOW_ACCURACY_THRESHOLD or consecutive_wrong >= 2:
        if curr_rank > 1:
            next_rank = curr_rank - 1
            next_diff = RANK_TO_DIFFICULTY[next_rank]
            return {
                "action": "reduce_difficulty",
                "topic": topic,
                "current_difficulty": current_diff,
                "recommended_difficulty": next_diff,
                "reason": (
                    f"Let's strengthen this concept before moving on. Here is a worked example "
                    f"and a foundational {next_diff.title()} question to build confidence."
                ),
                "support_type": "worked_example",
                "confidence": "high",
                "signals_used": signals_summary
            }
        else:
            return {
                "action": "guided_practice",
                "topic": topic,
                "current_difficulty": current_diff,
                "recommended_difficulty": current_diff,
                "reason": (
                    f"Let's strengthen fundamental concepts in {topic}. "
                    "Here is a guided explanation and hint to help you succeed."
                ),
                "support_type": "hint",
                "confidence": "high",
                "signals_used": signals_summary
            }

    # ── Rule 7: Mixed / Steady Performance -> Maintain Difficulty ───────────
    return {
        "action": "maintain_difficulty",
        "topic": topic,
        "current_difficulty": current_diff,
        "recommended_difficulty": current_diff,
        "reason": (
            f"Performance is steady on {topic} ({int(recent_accuracy * 100)}% accuracy). "
            f"Continuing practice at {current_diff.title()} level to solidify understanding."
        ),
        "support_type": "none",
        "confidence": "medium",
        "signals_used": signals_summary
    }


# ── Multi-Topic State Manager ────────────────────────────────────────────────

class StudentAdaptiveManager:
    """
    Manages topic-specific trackers for a student to ensure topic isolation.
    """

    def __init__(self, user_id: Optional[int] = None):
        self.user_id = user_id
        self.trackers: Dict[str, TopicDifficultyTracker] = {}

    def get_tracker(self, topic: str, default_diff: str = 'easy') -> TopicDifficultyTracker:
        if topic not in self.trackers:
            self.trackers[topic] = TopicDifficultyTracker(topic, default_diff)
        return self.trackers[topic]

    def record_attempt(self, topic: str, correct: bool, time_taken: float, qid: Optional[str] = None):
        tracker = self.get_tracker(topic)
        tracker.record_attempt(correct, time_taken, qid)

    def evaluate_next_step(
        self,
        topic: str,
        emotion_data: Optional[Dict[str, Any]] = None,
        attention_data: Optional[Dict[str, Any]] = None,
        revision_due: bool = False
    ) -> Dict[str, Any]:
        tracker = self.get_tracker(topic)
        last_time = tracker.attempts_history[-1]['time_taken'] if tracker.attempts_history else 10.0

        decision = evaluate_learning_decision(
            topic=topic,
            current_difficulty=tracker.current_difficulty,
            total_topic_attempts=tracker.total_attempts,
            recent_accuracy=tracker.get_recent_accuracy(5),
            consecutive_correct=tracker.consecutive_correct,
            consecutive_wrong=tracker.consecutive_wrong,
            current_response_time=last_time,
            baseline_response_time=tracker.baseline_response_time,
            emotion_data=emotion_data,
            attention_data=attention_data,
            revision_due=revision_due,
            recent_attempts=tracker.attempts_history
        )

        # Update difficulty if changed
        if decision['recommended_difficulty'] != tracker.current_difficulty:
            tracker.current_difficulty = decision['recommended_difficulty']
            tracker.last_change_time = time.time()

        return decision
