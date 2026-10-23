"""
personalization.py — Personalized Learning, Weak Topic Detection & Adaptation Hysteresis
========================================================================================
Analyses user_attempts to:
  - Identify subjects where student is struggling (accuracy < threshold).
  - Prioritize questions from weak topics and spaced repetition due items.
  - Implement Adaptation Hysteresis & Cooldown to prevent rapid difficulty oscillation.
  - Record explicit, explainable adaptation reasoning.
"""

import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import sqlite3
from db import DB_PATH, get_conn, adapt_sql

# Accuracy threshold below which a subject is considered "weak"
WEAK_THRESHOLD = 0.60   # 60%

# Minimum attempts required before a subject is evaluated as weak
MIN_ATTEMPTS = 3

# Adaptation Hysteresis Constants
CONSECUTIVE_SIGNALS_REQUIRED = 3     # Must see 3 consistent signals before changing difficulty
DIFFICULTY_COOLDOWN_SECS = 15.0       # Minimum seconds between difficulty transitions
DIFFICULTY_COOLDOWN_QUESTIONS = 2     # Minimum questions answered between difficulty changes


# ── Per-Session Adaptation Hysteresis Tracker ────────────────────────────────

class AdaptationHysteresisTracker:
    """
    Prevents difficulty oscillation (Easy <-> Medium <-> Hard) by requiring
    sustained confirming evidence and enforcing a minimum cooldown window.
    """

    def __init__(self, initial_difficulty: str = 'easy'):
        self.current_difficulty: str = initial_difficulty
        self.signal_history: deque = deque(maxlen=6)
        self.last_change_time: float = 0.0
        self.questions_since_change: int = 0
        self.last_adaptation_reason: str = "Initial starting difficulty."

    def evaluate_adaptation(
        self,
        cognitive_state: str,
        recent_accuracy: float,
        session_accuracy: float,
        consecutive_wrong: int,
        consecutive_correct: int,
        time_taken: float,
        now: Optional[float] = None
    ) -> Tuple[str, bool, str]:
        """
        Determines whether difficulty should increase, decrease, or remain stable.

        Returns:
            Tuple of: (new_difficulty: str, did_change: bool, reason: str)
        """
        if now is None:
            now = time.time()

        # Check candidate direction from cognitive state and performance
        candidate_shift = 0  # -1 = decrease, 0 = keep, +1 = increase
        candidate_reason = ""

        # High performance in Flow / Engaged / Bored -> Candidate for increase
        if cognitive_state in ['flow', 'bored'] and (recent_accuracy >= 0.75 or consecutive_correct >= 2):
            if self.current_difficulty == 'easy':
                candidate_shift = +1
                candidate_reason = f"Consistent mastery in {cognitive_state.upper()} state ({int(recent_accuracy*100)}% accuracy)."
            elif self.current_difficulty == 'medium' and recent_accuracy >= 0.80:
                candidate_shift = +1
                candidate_reason = f"Excellent sustained performance ({consecutive_correct} consecutive correct)."
        # Struggling / Overloaded / Consecutive wrong -> Candidate for decrease
        elif cognitive_state in ['overloaded', 'struggling'] or consecutive_wrong >= 2 or (recent_accuracy <= 0.35 and self.questions_since_change >= 2):
            if self.current_difficulty in ['hard', 'medium']:
                candidate_shift = -1
                candidate_reason = f"Scaffolding needed: {cognitive_state.upper()} state ({consecutive_wrong} wrong answers)."

        # Record candidate signal in rolling history
        self.signal_history.append(candidate_shift)

        # Evaluate cooldown
        time_since_change = now - self.last_change_time
        in_cooldown = (
            self.last_change_time > 0 and
            (time_since_change < DIFFICULTY_COOLDOWN_SECS or self.questions_since_change < DIFFICULTY_COOLDOWN_QUESTIONS)
        )

        # Check for consecutive confirming signals
        confirming_increases = sum(1 for s in self.signal_history if s > 0)
        confirming_decreases = sum(1 for s in self.signal_history if s < 0)

        target_diff = self.current_difficulty
        did_change = False
        reason = f"Maintaining {self.current_difficulty.title()} difficulty."

        if candidate_shift > 0 and confirming_increases >= CONSECUTIVE_SIGNALS_REQUIRED and not in_cooldown:
            if self.current_difficulty == 'easy':
                target_diff = 'medium'
            elif self.current_difficulty == 'medium':
                target_diff = 'hard'

            if target_diff != self.current_difficulty:
                did_change = True
                self.current_difficulty = target_diff
                self.last_change_time = now
                self.questions_since_change = 0
                self.signal_history.clear()
                reason = f"Difficulty increased to {target_diff.title()}: {candidate_reason}"
                self.last_adaptation_reason = reason

        elif candidate_shift < 0 and confirming_decreases >= CONSECUTIVE_SIGNALS_REQUIRED and not in_cooldown:
            if self.current_difficulty == 'hard':
                target_diff = 'medium'
            elif self.current_difficulty == 'medium':
                target_diff = 'easy'

            if target_diff != self.current_difficulty:
                did_change = True
                self.current_difficulty = target_diff
                self.last_change_time = now
                self.questions_since_change = 0
                self.signal_history.clear()
                reason = f"Difficulty reduced to {target_diff.title()}: {candidate_reason}"
                self.last_adaptation_reason = reason
        elif in_cooldown and candidate_shift != 0:
            reason = f"Holding at {self.current_difficulty.title()} (Stabilization cooldown active: {int(DIFFICULTY_COOLDOWN_SECS - time_since_change)}s remaining)."
        else:
            reason = self.last_adaptation_reason

        return self.current_difficulty, did_change, reason

    def record_question_answered(self):
        """Increments question counter since last difficulty transition."""
        self.questions_since_change += 1


# Global dictionary of session hysteresis trackers
_session_hysteresis: Dict[str, AdaptationHysteresisTracker] = {}


def get_adaptation_tracker(session_id: str, default_diff: str = 'easy') -> AdaptationHysteresisTracker:
    if session_id not in _session_hysteresis:
        _session_hysteresis[session_id] = AdaptationHysteresisTracker(default_diff)
    return _session_hysteresis[session_id]


def clear_adaptation_tracker(session_id: str):
    _session_hysteresis.pop(session_id, None)


# ── Weak Topic Analysis ──────────────────────────────────────────────────────

def get_subject_accuracy(user_id: int) -> dict:
    """
    Calculate per-subject accuracy for a user from user_attempts.
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(adapt_sql("""
            SELECT subject,
                   COUNT(*)        AS total,
                   SUM(correct)    AS correct_count
            FROM   user_attempts
            WHERE  user_id = ?
            GROUP  BY subject
        """), (user_id,))
        rows = cur.fetchall()
        conn.close()

        result = {}
        for row in rows:
            subj = row['subject'] or 'Unknown'
            total = int(row['total'])
            correct = int(row['correct_count'] or 0)
            acc = round(correct / total, 3) if total > 0 else 0.0
            result[subj] = {
                'total': total,
                'correct': correct,
                'accuracy': acc,
                'is_weak': total >= MIN_ATTEMPTS and acc < WEAK_THRESHOLD,
            }
        return result

    except Exception as e:
        print(f"[PERSONALIZATION ERROR] get_subject_accuracy: {e}")
        return {}


def get_weak_topics(user_id: int) -> list:
    """
    Return a list of subjects where the user's accuracy is below WEAK_THRESHOLD
    and they have answered at least MIN_ATTEMPTS questions.
    """
    subject_data = get_subject_accuracy(user_id)

    weak = []
    for subj, data in subject_data.items():
        if data['is_weak']:
            weak.append({
                'subject': subj,
                'accuracy': round(data['accuracy'] * 100, 1),
                'total': data['total'],
                'correct': data['correct'],
                'recommendation': (
                    f"Practice {subj} — your current accuracy is {round(data['accuracy']*100,1)}%. "
                    f"Master the fundamentals first."
                ),
            })

    # Worst accuracy first
    weak.sort(key=lambda x: x['accuracy'])
    return weak


def get_weak_subject_names(user_id: int) -> list:
    return [w['subject'] for w in get_weak_topics(user_id)]


def prioritize_question_pool(
    pool: list,
    user_id: Optional[int],
    current_subject: str,
    cognitive_state: Optional[str] = None
) -> list:
    """
    Re-orders question pool based on weak topic prioritization and cognitive load.
    """
    if not pool:
        return pool

    if not user_id:
        return pool

    weak_subjects = get_weak_subject_names(user_id)

    if not weak_subjects or current_subject in weak_subjects:
        return pool

    # Separate weak-subject questions from others
    weak_qs = [q for q in pool if q.get('subject') in weak_subjects]
    other_qs = [q for q in pool if q.get('subject') not in weak_subjects]

    return weak_qs + other_qs


def get_personalized_recommendation(user_id: int) -> str:
    """
    Build a single human-readable recommendation string based on weak topics.
    """
    weak = get_weak_topics(user_id)
    if not weak:
        return "Great work! Keep practising to maintain your high performance."

    worst = weak[0]
    if len(weak) == 1:
        return (
            f"Focus on {worst['subject']} — your accuracy is {worst['accuracy']}%. "
            "Try easier foundational questions to build confidence."
        )

    names = ', '.join(w['subject'] for w in weak[:3])
    return (
        f"Recommended review areas: {names}. "
        f"Start with {worst['subject']} ({worst['accuracy']}% accuracy)."
    )


# ── Adaptive Engine Re-exports & Per-Student Topic Management ─────────────────
from adaptive_engine import (
    evaluate_learning_decision,
    normalize_difficulty,
    StudentAdaptiveManager,
    TopicDifficultyTracker,
    DIFFICULTY_LEVELS,
    DIFFICULTY_RANK,
)

# Per-user adaptive managers for in-memory session acceleration
_user_adaptive_managers: Dict[Any, StudentAdaptiveManager] = {}

def get_student_adaptive_manager(user_id: Optional[int] = None) -> StudentAdaptiveManager:
    key = user_id if user_id is not None else 'anonymous'
    if key not in _user_adaptive_managers:
        _user_adaptive_managers[key] = StudentAdaptiveManager(user_id)
    return _user_adaptive_managers[key]


# ── Database Persistence for Topic-Specific Difficulty & Decisions ───────────

def init_adaptive_tables():
    """Ensure user_topic_difficulty and adaptive_decisions tables exist."""
    try:
        from db import USE_POSTGRES
        conn = get_conn()
        cur = conn.cursor()
        pk = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS user_topic_difficulty (
                id {pk},
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                current_difficulty TEXT DEFAULT 'easy',
                total_attempts INTEGER DEFAULT 0,
                total_correct INTEGER DEFAULT 0,
                consecutive_correct INTEGER DEFAULT 0,
                consecutive_wrong INTEGER DEFAULT 0,
                avg_response_time REAL DEFAULT 0.0,
                last_updated TEXT,
                UNIQUE(user_id, topic)
            )
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS adaptive_decisions (
                id {pk},
                user_id INTEGER,
                session_id TEXT,
                topic TEXT NOT NULL,
                previous_difficulty TEXT,
                recommended_difficulty TEXT,
                action TEXT NOT NULL,
                reason TEXT,
                support_type TEXT,
                confidence TEXT,
                signals_json TEXT,
                followed INTEGER DEFAULT NULL,
                subsequent_correct INTEGER DEFAULT NULL,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[PERSONALIZATION ERROR] init_adaptive_tables: {e}")


def get_user_topic_difficulty(user_id: Optional[int], topic: str, default_diff: str = 'easy') -> Dict[str, Any]:
    """Fetch persistent topic difficulty record for user, or return default."""
    if not user_id:
        return {
            'topic': topic,
            'current_difficulty': normalize_difficulty(default_diff),
            'total_attempts': 0,
            'total_correct': 0,
            'consecutive_correct': 0,
            'consecutive_wrong': 0,
            'avg_response_time': 12.0
        }

    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(adapt_sql("""
            SELECT current_difficulty, total_attempts, total_correct,
                   consecutive_correct, consecutive_wrong, avg_response_time
            FROM user_topic_difficulty
            WHERE user_id = ? AND topic = ?
        """), (user_id, topic))
        row = cur.fetchone()
        conn.close()

        if row:
            return {
                'topic': topic,
                'current_difficulty': normalize_difficulty(row['current_difficulty']),
                'total_attempts': int(row['total_attempts']),
                'total_correct': int(row['total_correct']),
                'consecutive_correct': int(row['consecutive_correct']),
                'consecutive_wrong': int(row['consecutive_wrong']),
                'avg_response_time': float(row['avg_response_time'] or 12.0)
            }
    except Exception as e:
        print(f"[PERSONALIZATION ERROR] get_user_topic_difficulty: {e}")

    return {
        'topic': topic,
        'current_difficulty': normalize_difficulty(default_diff),
        'total_attempts': 0,
        'total_correct': 0,
        'consecutive_correct': 0,
        'consecutive_wrong': 0,
        'avg_response_time': 12.0
    }


def update_user_topic_difficulty(
    user_id: Optional[int],
    topic: str,
    new_difficulty: str,
    is_correct: bool,
    time_taken: float
) -> None:
    """Updates or inserts the student's persistent topic difficulty state."""
    if not user_id:
        return

    safe_diff = normalize_difficulty(new_difficulty)
    safe_time = max(0.5, min(float(time_taken), 300.0))
    now_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(adapt_sql("""
            SELECT id, total_attempts, total_correct, consecutive_correct,
                   consecutive_wrong, avg_response_time
            FROM user_topic_difficulty
            WHERE user_id = ? AND topic = ?
        """), (user_id, topic))
        existing = cur.fetchone()

        if existing:
            tot_att = int(existing['total_attempts']) + 1
            tot_cor = int(existing['total_correct']) + (1 if is_correct else 0)
            c_cor = (int(existing['consecutive_correct']) + 1) if is_correct else 0
            c_wrg = (int(existing['consecutive_wrong']) + 1) if not is_correct else 0
            old_avg = float(existing['avg_response_time'] or safe_time)
            new_avg = round(0.80 * old_avg + 0.20 * safe_time, 2)

            cur.execute(adapt_sql("""
                UPDATE user_topic_difficulty
                SET current_difficulty = ?,
                    total_attempts = ?,
                    total_correct = ?,
                    consecutive_correct = ?,
                    consecutive_wrong = ?,
                    avg_response_time = ?,
                    last_updated = ?
                WHERE user_id = ? AND topic = ?
            """), (safe_diff, tot_att, tot_cor, c_cor, c_wrg, new_avg, now_iso, user_id, topic))
        else:
            cur.execute(adapt_sql("""
                INSERT INTO user_topic_difficulty
                (user_id, topic, current_difficulty, total_attempts, total_correct,
                 consecutive_correct, consecutive_wrong, avg_response_time, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """), (user_id, topic, safe_diff, 1, 1 if is_correct else 0,
                   1 if is_correct else 0, 0 if is_correct else 1, safe_time, now_iso))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[PERSONALIZATION ERROR] update_user_topic_difficulty: {e}")


def record_adaptive_decision(
    user_id: Optional[int],
    session_id: str,
    topic: str,
    previous_diff: str,
    recommended_diff: str,
    action: str,
    reason: str,
    support_type: str = 'none',
    confidence: str = 'high',
    signals: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """Records an adaptive decision into the database for auditability and research."""
    import json
    now_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    signals_str = json.dumps(signals) if signals else "{}"

    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(adapt_sql("""
            INSERT INTO adaptive_decisions
            (user_id, session_id, topic, previous_difficulty, recommended_difficulty,
             action, reason, support_type, confidence, signals_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """), (user_id, session_id, topic, previous_diff, recommended_diff,
               action, reason, support_type, confidence, signals_str, now_iso))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[PERSONALIZATION ERROR] record_adaptive_decision: {e}")
    return None


def get_adaptive_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Returns recent adaptive decisions for the user."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(adapt_sql("""
            SELECT id, topic, previous_difficulty, recommended_difficulty,
                   action, reason, support_type, confidence, timestamp
            FROM adaptive_decisions
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """), (user_id, limit))
        rows = cur.fetchall()
        conn.close()

        results = []
        for r in rows:
            results.append({
                'id': r['id'],
                'topic': r['topic'],
                'previous_difficulty': r['previous_difficulty'],
                'recommended_difficulty': r['recommended_difficulty'],
                'action': r['action'],
                'reason': r['reason'],
                'support_type': r['support_type'],
                'confidence': r['confidence'],
                'timestamp': r['timestamp']
            })
        return results
    except Exception as e:
        print(f"[PERSONALIZATION ERROR] get_adaptive_history: {e}")
        return []


def get_adaptive_summary(user_id: int) -> Dict[str, Any]:
    """
    Returns an aggregated breakdown of difficulty performance and adaptation outcomes
    for display in student analytics.
    """
    try:
        conn = get_conn()
        cur = conn.cursor()

        # Topic difficulty breakdown
        cur.execute(adapt_sql("""
            SELECT topic, current_difficulty, total_attempts, total_correct, avg_response_time
            FROM user_topic_difficulty
            WHERE user_id = ?
        """), (user_id,))
        topic_rows = cur.fetchall()

        topic_status = []
        for r in topic_rows:
            tot = int(r['total_attempts'] or 0)
            cor = int(r['total_correct'] or 0)
            acc = round((cor / tot) * 100.0, 1) if tot > 0 else 0.0
            topic_status.append({
                'topic': r['topic'],
                'difficulty': r['current_difficulty'],
                'attempts': tot,
                'correct': cor,
                'accuracy': acc,
                'avg_time': round(float(r['avg_response_time'] or 0.0), 1)
            })

        # Decision counts
        cur.execute(adapt_sql("""
            SELECT action, COUNT(*) as cnt
            FROM adaptive_decisions
            WHERE user_id = ?
            GROUP BY action
        """), (user_id,))
        decision_counts = {r['action']: int(r['cnt']) for r in cur.fetchall()}

        # Performance by difficulty level from user_attempts
        cur.execute(adapt_sql("""
            SELECT difficulty, COUNT(*) as total, SUM(correct) as correct
            FROM user_attempts
            WHERE user_id = ?
            GROUP BY difficulty
        """), (user_id,))
        by_diff = {}
        for r in cur.fetchall():
            d = normalize_difficulty(r['difficulty'])
            tot = int(r['total'])
            c = int(r['correct'] or 0)
            acc = round((c / tot) * 100.0, 1) if tot > 0 else 0.0
            by_diff[d] = {'total': tot, 'correct': c, 'accuracy': acc}

        conn.close()

        return {
            'topic_status': topic_status,
            'decision_counts': decision_counts,
            'accuracy_by_difficulty': by_diff,
            'total_decisions': sum(decision_counts.values())
        }
    except Exception as e:
        print(f"[PERSONALIZATION ERROR] get_adaptive_summary: {e}")
        return {
            'topic_status': [],
            'decision_counts': {},
            'accuracy_by_difficulty': {},
            'total_decisions': 0
        }

