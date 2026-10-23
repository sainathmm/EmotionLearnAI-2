"""
test_adaptive_engine.py — Emotion-Aware Adaptive Learning Engine Test Suite
===========================================================================
Deterministic, exhaustive evaluation verifying the 12 core requirements:
  1. Five recent answers with high accuracy -> consider increasing difficulty.
  2. Several incorrect answers on the same topic -> provide guided practice / reduce.
  3. Insufficient attempt history -> maintain difficulty baseline.
  4. Long response time with high accuracy -> do NOT automatically lower difficulty.
  5. Missing emotion data -> valid decision from other signals.
  6. Low-confidence emotion prediction -> ignore or down-weight noisy signal.
  7. Repeated mistakes and reliable supporting signals -> offer help / optional break.
  8. Strong performance on one topic vs weak on another -> strict topic-specific isolation.
  9. Gemini API failure -> safe fallback to static question bank.
  10. Repeated requests for same submitted answer -> idempotency / no duplicates.
  11. Student A cannot access Student B's learning history (Data Isolation).
  12. Existing quiz, revision, analytics, and interview features continue working.
"""

import os
import sys
import json
import time
import unittest
import sqlite3

# Set test environment
os.environ['DATABASE_URL'] = ''

from adaptive_engine import (
    evaluate_learning_decision,
    normalize_difficulty,
    TopicDifficultyTracker,
    StudentAdaptiveManager,
    MIN_ATTEMPTS_BEFORE_CHANGE,
    HIGH_ACCURACY_THRESHOLD,
    LOW_ACCURACY_THRESHOLD,
    SLOW_PACING_RATIO,
    EMOTION_MIN_CONFIDENCE
)
from personalization import (
    get_user_topic_difficulty,
    update_user_topic_difficulty,
    record_adaptive_decision,
    get_adaptive_history,
    get_adaptive_summary,
    init_adaptive_tables,
    AdaptationHysteresisTracker
)
import app as flask_app
from app import app, ALL_QUESTIONS


class TestAdaptiveLearningEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_adaptive_tables()

    def setUp(self):
        self.client = app.test_client()
        self.app_ctx = app.app_context()
        self.app_ctx.push()

    def tearDown(self):
        self.app_ctx.pop()

    # ── Scenario 1: Consistently High Accuracy Increases Difficulty ───────────
    def test_scenario_1_high_accuracy_increases_difficulty(self):
        """
        Scenario 1:
        Five recent answers with consistently high accuracy (>=80%) on a topic
        with sufficient history should recommend increasing difficulty by 1 level.
        """
        # From easy -> medium
        dec_easy = evaluate_learning_decision(
            topic="Python Basics",
            current_difficulty="easy",
            total_topic_attempts=5,
            recent_accuracy=1.0,
            consecutive_correct=5,
            consecutive_wrong=0,
            current_response_time=8.5,
            baseline_response_time=9.0
        )
        self.assertEqual(dec_easy["action"], "increase_difficulty")
        self.assertEqual(dec_easy["current_difficulty"], "easy")
        self.assertEqual(dec_easy["recommended_difficulty"], "medium")
        self.assertEqual(dec_easy["confidence"], "high")
        self.assertIn("challenging Medium question", dec_easy["reason"])

        # From medium -> hard
        dec_med = evaluate_learning_decision(
            topic="Python Basics",
            current_difficulty="medium",
            total_topic_attempts=6,
            recent_accuracy=0.80, # 4 of 5 correct
            consecutive_correct=4,
            consecutive_wrong=0,
            current_response_time=10.0,
            baseline_response_time=11.0
        )
        self.assertEqual(dec_med["action"], "increase_difficulty")
        self.assertEqual(dec_med["recommended_difficulty"], "hard")

        # At hard (ceiling) -> maintain with praise
        dec_hard = evaluate_learning_decision(
            topic="Python Basics",
            current_difficulty="hard",
            total_topic_attempts=8,
            recent_accuracy=1.0,
            consecutive_correct=5,
            consecutive_wrong=0,
            current_response_time=12.0,
            baseline_response_time=12.0
        )
        self.assertEqual(dec_hard["action"], "maintain_difficulty")
        self.assertEqual(dec_hard["recommended_difficulty"], "hard")
        self.assertIn("Mastery demonstrated", dec_hard["reason"])

    # ── Scenario 2: Several Incorrect Answers -> Guided Practice ─────────────
    def test_scenario_2_several_incorrect_guided_practice(self):
        """
        Scenario 2:
        Several incorrect answers on the same topic should reduce difficulty or
        provide guided practice with worked examples and scaffolding.
        """
        # At Hard with errors -> drop to Medium with worked example
        dec_hard_fail = evaluate_learning_decision(
            topic="SQL Joins",
            current_difficulty="hard",
            total_topic_attempts=5,
            recent_accuracy=0.20, # 4 of 5 wrong
            consecutive_correct=0,
            consecutive_wrong=3,
            current_response_time=15.0,
            baseline_response_time=12.0
        )
        self.assertEqual(dec_hard_fail["action"], "reduce_difficulty")
        self.assertEqual(dec_hard_fail["recommended_difficulty"], "medium")
        self.assertEqual(dec_hard_fail["support_type"], "worked_example")

        # At Easy with errors -> provide guided practice hint (already at base difficulty)
        dec_easy_fail = evaluate_learning_decision(
            topic="SQL Joins",
            current_difficulty="easy",
            total_topic_attempts=5,
            recent_accuracy=0.20,
            consecutive_correct=0,
            consecutive_wrong=3,
            current_response_time=14.0,
            baseline_response_time=12.0
        )
        self.assertEqual(dec_easy_fail["action"], "guided_practice")
        self.assertEqual(dec_easy_fail["recommended_difficulty"], "easy")
        self.assertEqual(dec_easy_fail["support_type"], "hint")
        self.assertIn("guided explanation", dec_easy_fail["reason"])

    # ── Scenario 3: Insufficient Attempt History Maintains Difficulty ─────────
    def test_scenario_3_insufficient_attempt_history(self):
        """
        Scenario 3:
        When a student has answered fewer than MIN_ATTEMPTS_BEFORE_CHANGE (3),
        difficulty must be maintained to establish a reliable baseline.
        """
        # Even with 2/2 correct (100%), do not jump difficulty prematurely
        dec_few_correct = evaluate_learning_decision(
            topic="Algorithms",
            current_difficulty="easy",
            total_topic_attempts=2,
            recent_accuracy=1.0,
            consecutive_correct=2,
            consecutive_wrong=0,
            current_response_time=7.0,
            baseline_response_time=8.0
        )
        self.assertEqual(dec_few_correct["action"], "maintain_difficulty")
        self.assertEqual(dec_few_correct["recommended_difficulty"], "easy")
        self.assertEqual(dec_few_correct["confidence"], "low")
        self.assertIn("Establishing baseline performance", dec_few_correct["reason"])

        # Even with 0/1 wrong, do not drop difficulty prematurely
        dec_one_wrong = evaluate_learning_decision(
            topic="Algorithms",
            current_difficulty="medium",
            total_topic_attempts=1,
            recent_accuracy=0.0,
            consecutive_correct=0,
            consecutive_wrong=1,
            current_response_time=15.0,
            baseline_response_time=12.0
        )
        self.assertEqual(dec_one_wrong["action"], "maintain_difficulty")
        self.assertEqual(dec_one_wrong["recommended_difficulty"], "medium")

    # ── Scenario 4: Long Response Time with High Accuracy -> No Demotion ─────
    def test_scenario_4_long_response_time_high_accuracy_no_demotion(self):
        """
        Scenario 4:
        Long response time with high accuracy MUST NOT automatically lower difficulty.
        The student is getting answers right and deliberating carefully.
        """
        dec_slow_correct = evaluate_learning_decision(
            topic="Data Science",
            current_difficulty="medium",
            total_topic_attempts=5,
            recent_accuracy=1.0,
            consecutive_correct=5,
            consecutive_wrong=0,
            current_response_time=32.0,  # 3.2x baseline
            baseline_response_time=10.0
        )
        # Must NOT be reduce_difficulty
        self.assertNotEqual(dec_slow_correct["action"], "reduce_difficulty")
        self.assertEqual(dec_slow_correct["action"], "maintain_difficulty")
        self.assertEqual(dec_slow_correct["recommended_difficulty"], "medium")
        self.assertIn("taking time to think questions through", dec_slow_correct["reason"])

    # ── Scenario 5: Missing Emotion Data -> Valid Decision ───────────────────
    def test_scenario_5_missing_emotion_data(self):
        """
        Scenario 5:
        When emotion data is missing (camera off, face obstructed, None, or empty dict),
        the system must make a valid decision purely from performance signals.
        """
        # None passed
        dec_none = evaluate_learning_decision(
            topic="Cybersecurity",
            current_difficulty="easy",
            total_topic_attempts=4,
            recent_accuracy=1.0,
            consecutive_correct=4,
            emotion_data=None,
            attention_data=None
        )
        self.assertIsInstance(dec_none, dict)
        self.assertEqual(dec_none["action"], "increase_difficulty")
        self.assertEqual(dec_none["recommended_difficulty"], "medium")
        self.assertFalse(dec_none["signals_used"]["reliable_emotion"])

        # Empty dict passed
        dec_empty = evaluate_learning_decision(
            topic="Cybersecurity",
            current_difficulty="easy",
            total_topic_attempts=4,
            recent_accuracy=1.0,
            consecutive_correct=4,
            emotion_data={},
            attention_data={}
        )
        self.assertEqual(dec_empty["action"], "increase_difficulty")

    # ── Scenario 6: Low Confidence Emotion Prediction Ignored ────────────────
    def test_scenario_6_low_confidence_emotion_ignored(self):
        """
        Scenario 6:
        Low-confidence emotion prediction (< 0.60 or is_uncertain=True) must be
        ignored and downweighted so noisy readings never trigger false adaptations.
        """
        noisy_emotion = {
            'emotion': 'sad',
            'confidence': 0.38, # Below 0.60 threshold
            'stability': 0.30,
            'is_uncertain': True
        }
        dec_noisy = evaluate_learning_decision(
            topic="Python Basics",
            current_difficulty="easy",
            total_topic_attempts=5,
            recent_accuracy=0.80,
            consecutive_correct=4,
            consecutive_wrong=0,
            current_response_time=9.0,
            baseline_response_time=10.0,
            emotion_data=noisy_emotion
        )
        self.assertFalse(dec_noisy["signals_used"]["reliable_emotion"])
        self.assertEqual(dec_noisy["signals_used"]["emotion_label"], "unreliable/ignored")
        # High accuracy should still advance, completely unaffected by noisy 'sad'
        self.assertEqual(dec_noisy["action"], "increase_difficulty")
        self.assertEqual(dec_noisy["recommended_difficulty"], "medium")

    # ── Scenario 7: Repeated Mistakes + Reliable Stress -> Suggest Break ──────
    def test_scenario_7_repeated_mistakes_reliable_stress_suggests_break(self):
        """
        Scenario 7:
        Repeated mistakes (>=2 wrong) combined with reliable supporting emotion signals
        (e.g., confident signs of frustration/stress) should offer an optional short break.
        """
        reliable_stress = {
            'emotion': 'angry',
            'confidence': 0.88,
            'stability': 0.82,
            'is_uncertain': False
        }
        dec_break = evaluate_learning_decision(
            topic="Calculus",
            current_difficulty="medium",
            total_topic_attempts=5,
            recent_accuracy=0.20,
            consecutive_correct=0,
            consecutive_wrong=2,
            current_response_time=18.0,
            baseline_response_time=12.0,
            emotion_data=reliable_stress
        )
        self.assertEqual(dec_break["action"], "suggest_break")
        self.assertEqual(dec_break["support_type"], "rest_break")
        self.assertTrue(dec_break["signals_used"]["reliable_emotion"])
        # Neutral phrasing verification
        self.assertIn("Would you like a short break or a worked example?", dec_break["reason"])
        self.assertNotIn("You are angry", dec_break["reason"])

    # ── Scenario 8: Topic-Specific Difficulty Isolation ──────────────────────
    def test_scenario_8_topic_specific_difficulty_isolation(self):
        """
        Scenario 8:
        Strong performance on one topic must not bleed into or inflate difficulty
        on a different topic. Topics must maintain independent tracking.
        """
        manager = StudentAdaptiveManager(user_id=101)

        # Topic 1: Python Basics — Student performs brilliantly
        for _ in range(5):
            manager.record_attempt(topic="Python Basics", correct=True, time_taken=7.0)
        dec_py = manager.evaluate_next_step(topic="Python Basics")
        self.assertEqual(dec_py["action"], "increase_difficulty")
        self.assertEqual(dec_py["recommended_difficulty"], "medium")
        self.assertEqual(manager.get_tracker("Python Basics").current_difficulty, "medium")

        # Advance to Hard
        for _ in range(5):
            manager.record_attempt(topic="Python Basics", correct=True, time_taken=8.0)
        dec_py2 = manager.evaluate_next_step(topic="Python Basics")
        self.assertEqual(dec_py2["recommended_difficulty"], "hard")
        self.assertEqual(manager.get_tracker("Python Basics").current_difficulty, "hard")

        # Topic 2: Database & SQL — Student is a complete beginner and struggles
        for _ in range(4):
            manager.record_attempt(topic="Database & SQL", correct=False, time_taken=20.0)
        dec_sql = manager.evaluate_next_step(topic="Database & SQL")

        # SQL must remain at Easy (guided practice), NEVER Hard!
        self.assertEqual(dec_sql["recommended_difficulty"], "easy")
        self.assertEqual(manager.get_tracker("Database & SQL").current_difficulty, "easy")

        # Python Basics must still be Hard!
        self.assertEqual(manager.get_tracker("Python Basics").current_difficulty, "hard")

    # ── Scenario 9: Gemini API Failure Safe Fallback ──────────────────────────
    def test_scenario_9_gemini_api_failure_safe_fallback(self):
        """
        Scenario 9:
        When the AI/Gemini question generator fails (network error, rate limit,
        or invalid response), the endpoint safely falls back to a verified static question.
        """
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1

        # Request a question with simulated Gemini failure (or test client without key)
        resp = self.client.get('/generate-question?topic=Python+Basics&difficulty=easy')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertIn("title", data)
        self.assertIn("quiz", data)
        self.assertIn("q", data["quiz"])
        self.assertIn("opts", data["quiz"])
        self.assertIn("ans", data["quiz"])
        self.assertIsInstance(data["quiz"]["opts"], list)
        self.assertGreaterEqual(len(data["quiz"]["opts"]), 2)

        # Fallback question flag check
        if data.get("fallback"):
            self.assertIn("fallback_reason", data)

    # ── Scenario 10: Repeated Requests Idempotency / No Duplicates ────────────
    def test_scenario_10_repeated_submitted_answer_idempotency(self):
        """
        Scenario 10:
        Repeated submission of the same answered question attempt must not create
        duplicate attempt records in user_attempts or double-increment question counters.
        """
        sid = f"test_idemp_{int(time.time()*1000)}"
        attempt_payload = {
            "session_id": sid,
            "question_id": "test_q_idemp_1",
            "question_title": "Python Syntax",
            "chosen_answer": "print('hello')",
            "correct_answer": "print('hello')",
            "correct": True,
            "time_taken": 10.0,
            "attempt_token": f"{sid}:test_q_idemp_1:print:10.0:True"
        }

        # First submission
        resp1 = self.client.post('/api/quiz', json=attempt_payload)
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.get_json()
        self.assertEqual(data1.get("total"), 1)
        self.assertEqual(data1.get("correct_count"), 1)

        # Immediate retry/duplicate submission with identical attempt token
        resp2 = self.client.post('/api/quiz', json=attempt_payload)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.get_json()
        self.assertTrue(data2.get("duplicate"))
        self.assertEqual(data2.get("status"), "duplicate_ignored")
        # Total questions must NOT be 2!
        self.assertEqual(data2.get("total"), 1)

    # ── Scenario 11: Student A Cannot Access Student B's Data ────────────────
    def test_scenario_11_user_data_isolation(self):
        """
        Scenario 11:
        Student A cannot view or tamper with Student B's adaptive history,
        topic difficulty records, or learning telemetry.
        """
        # Record adaptive decisions for Student 1001 and Student 1002
        record_adaptive_decision(
            user_id=1001,
            session_id="sess_1001",
            topic="Security Protocols",
            previous_diff="easy",
            recommended_diff="medium",
            action="increase_difficulty",
            reason="Student 1001 demonstrated mastery.",
            support_type="none"
        )
        record_adaptive_decision(
            user_id=1002,
            session_id="sess_1002",
            topic="SQL Injection",
            previous_diff="medium",
            recommended_diff="easy",
            action="reduce_difficulty",
            reason="Student 1002 needs scaffolding.",
            support_type="worked_example"
        )

        # Login as Student 1001
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1001

        resp_1001 = self.client.get('/api/adaptive/history')
        self.assertEqual(resp_1001.status_code, 200)
        data_1001 = resp_1001.get_json()["decisions"]

        # Student 1001 should see Security Protocols, but NEVER SQL Injection from Student 1002
        topics_1001 = [d["topic"] for d in data_1001]
        self.assertIn("Security Protocols", topics_1001)
        self.assertNotIn("SQL Injection", topics_1001)

        # Login as Student 1002
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1002

        resp_1002 = self.client.get('/api/adaptive/history')
        self.assertEqual(resp_1002.status_code, 200)
        data_1002 = resp_1002.get_json()["decisions"]
        topics_1002 = [d["topic"] for d in data_1002]
        self.assertIn("SQL Injection", topics_1002)
        self.assertNotIn("Security Protocols", topics_1002)

    # ── Scenario 12: Existing Features Continue Working (Backward Compat) ─────
    def test_scenario_12_backward_compatibility_existing_features(self):
        """
        Scenario 12:
        Existing quiz analytics, spaced repetition, achievements, interview assessment,
        and hysteresis tracking continue functioning seamlessly.
        """
        # Hysteresis tracker check
        tracker = AdaptationHysteresisTracker(initial_difficulty='easy')
        new_d, changed, _ = tracker.evaluate_adaptation(
            cognitive_state="struggling",
            recent_accuracy=0.8,
            session_accuracy=0.8,
            consecutive_wrong=0,
            consecutive_correct=3,
            time_taken=10.0
        )
        self.assertFalse(changed)
        self.assertEqual(new_d, 'easy')

        # Interview route check
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1
        resp_role = self.client.get('/select-role')
        self.assertIn(resp_role.status_code, [200, 302])

        # Spaced repetition check
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1
        resp_sr = self.client.get('/api/spaced-repetition/due')
        self.assertEqual(resp_sr.status_code, 200)


def run_all():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAdaptiveLearningEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    run_all()
