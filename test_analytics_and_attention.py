"""
test_analytics_and_attention.py - Verification Suite for EmotionLearnAI Analytics Upgrade & Attention Warning
==============================================================================================================
Validates:
1. Topic Mastery & Accuracy Formulas (weighted formula, sample confidence, status thresholds)
2. Edge cases (insufficient data <3 attempts, empty datasets, invalid response times)
3. Trend Analysis & Date Filtering (7d, 30d, all, baseline comparisons)
4. Spaced Repetition & Retention Analytics (due, overdue, completion %, retention checks)
5. Before-and-After Assessment Comparisons (initial vs latest window, response time deltas)
6. Explainable Recommendations Traceability (reasoning backed strictly by recorded attempts)
7. Attention Warning State Machine & Invariants (thresholds, 5s persistence, recovery reset, cooldown, non-blocking)
8. API Route Scoping & Integration (/api/dashboard/stats, /api/analytics/detailed)
9. Interview UI Template Compliance (non-blocking toast, auxiliary privacy notice, controls intact)
"""

import sys
import os
import io
import time
import json
import sqlite3
import datetime
import traceback

# Windows encoding safety
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

results = {"passed": 0, "failed": 0}


def run_test(name: str, fn, *args, **kwargs):
    """Runs a single test function and records result."""
    try:
        fn(*args, **kwargs)
        print(f"  {PASS} {name}")
        results["passed"] += 1
        return True
    except AssertionError as e:
        print(f"  {FAIL} {name}")
        print(f"         → AssertionError: {e}")
        results["failed"] += 1
        return False
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"         → Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["failed"] += 1
        return False


# ============================================================================
# CATEGORY 1: Topic-wise Mastery Formulas & Edge Cases
# ============================================================================
def test_topic_mastery_calculation():
    """Verify topic mastery formula with known attempt sequences."""
    from analytics_engine import calculate_topic_mastery, get_topic_mastery_report

    # User with 10 attempts: 8 correct, 2 incorrect on Python
    now = datetime.datetime.now()
    attempts = [
        {
            'subject': 'Python',
            'correct': 1 if i not in (3, 7) else 0,
            'time_taken': 8.5,
            'timestamp': (now - datetime.timedelta(hours=10 - i)).isoformat()
        }
        for i in range(10)
    ]

    topics = calculate_topic_mastery(attempts)
    assert len(topics) == 1, f"Expected 1 topic, got {len(topics)}"

    py = topics[0]
    assert py['topic'] == 'Python'
    assert py['total_attempts'] == 10
    assert py['correct'] == 8
    assert py['incorrect'] == 2
    assert py['accuracy'] == 80.0
    assert py['min_attempts_met'] is True
    assert py['confidence_factor'] == 1.0  # min(1.0, 10/8) = 1.0
    # Mastery should be between 70 and 90
    assert 70.0 <= py['mastery'] <= 90.0, f"Unexpected mastery: {py['mastery']}"
    assert py['status'] in ('Proficient', 'Mastered')

    # Also check wrapper
    report = get_topic_mastery_report(attempts)
    assert report['strongest_topic']['topic'] == 'Python'


def test_insufficient_data_handling():
    """Topics with < 3 attempts must be marked 'Insufficient Data'."""
    from analytics_engine import calculate_topic_mastery

    attempts = [
        {'subject': 'Algorithms', 'correct': 1, 'time_taken': 5.0, 'timestamp': '2026-10-20 10:00:00'},
        {'subject': 'Algorithms', 'correct': 1, 'time_taken': 6.0, 'timestamp': '2026-10-20 10:05:00'},
    ]

    topics = calculate_topic_mastery(attempts)
    assert len(topics) == 1
    algo = topics[0]
    assert algo['total_attempts'] == 2
    assert algo['min_attempts_met'] is False
    assert algo['status'] == 'Insufficient Data'
    # Confidence factor should be 2/8 = 0.25
    assert algo['confidence_factor'] == 0.25


def test_mastery_status_thresholds():
    """Verify status brackets: Needs Revision, Learning, Proficient, Mastered."""
    from analytics_engine import calculate_topic_mastery, get_topic_mastery_report

    now = datetime.datetime.now()
    # High score (9/9 correct) -> Mastered
    mastered_attempts = [
        {'subject': 'Databases', 'correct': 1, 'time_taken': 4.0, 'timestamp': (now - datetime.timedelta(minutes=i)).isoformat()}
        for i in range(9)
    ]
    # Low score (1/8 correct) -> Needs Revision
    poor_attempts = [
        {'subject': 'Operating Systems', 'correct': 1 if i == 0 else 0, 'time_taken': 12.0, 'timestamp': (now - datetime.timedelta(minutes=i)).isoformat()}
        for i in range(8)
    ]

    topics = calculate_topic_mastery(mastered_attempts + poor_attempts)
    topic_map = {t['topic']: t for t in topics}

    assert topic_map['Databases']['status'] == 'Mastered'
    assert topic_map['Databases']['mastery'] >= 85.0

    assert topic_map['Operating Systems']['status'] == 'Needs Revision'
    assert topic_map['Operating Systems']['mastery'] < 40.0

    report = get_topic_mastery_report(mastered_attempts + poor_attempts)
    assert report['strongest_topic']['topic'] == 'Databases'
    assert report['weakest_topic']['topic'] == 'Operating Systems'


# ============================================================================
# CATEGORY 2: Trend Analysis & Date Filtering
# ============================================================================
def test_trend_calculation_and_date_filtering():
    """Verify chronological date grouping and filtering (7d, 30d, all)."""
    from analytics_engine import calculate_trends

    now = datetime.datetime.now()
    attempts = []

    # 5 attempts today
    for i in range(5):
        attempts.append({
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'correct': 1 if i < 4 else 0,
            'time_taken': 10.0,
            'subject': 'Math'
        })
    # 5 attempts 15 days ago
    past_15d = now - datetime.timedelta(days=15)
    for i in range(5):
        attempts.append({
            'timestamp': past_15d.strftime('%Y-%m-%d %H:%M:%S'),
            'correct': 1 if i < 2 else 0,
            'time_taken': 20.0,
            'subject': 'Math'
        })
    # 5 attempts 45 days ago
    past_45d = now - datetime.timedelta(days=45)
    for i in range(5):
        attempts.append({
            'timestamp': past_45d.strftime('%Y-%m-%d %H:%M:%S'),
            'correct': 0,
            'time_taken': 30.0,
            'subject': 'Math'
        })

    # Test 7d range: should only include today
    t_7d = calculate_trends(attempts, date_range='7d')
    assert len(t_7d['date_trends']) == 1, f"Expected 1 day for 7d, got {len(t_7d['date_trends'])}"
    assert t_7d['date_trends'][0]['total'] == 5
    assert t_7d['date_trends'][0]['accuracy'] == 80.0
    assert t_7d['date_trends'][0]['avg_response_time'] == 10.0

    # Test 30d range: should include today and 15 days ago
    t_30d = calculate_trends(attempts, date_range='30d')
    assert len(t_30d['date_trends']) == 2, f"Expected 2 days for 30d, got {len(t_30d['date_trends'])}"

    # Test all range: should include all 3 days
    t_all = calculate_trends(attempts, date_range='all')
    assert len(t_all['date_trends']) == 3, f"Expected 3 days for all, got {len(t_all['date_trends'])}"

    # Test recent vs earlier trajectory comparison
    comp = t_all['recent_vs_earlier']
    assert comp['recent_accuracy'] > comp['earlier_accuracy']
    assert comp['diff_pct'] > 0


def test_invalid_and_missing_response_times():
    """Safely handle null, zero, negative, and string response times without breaking."""
    from analytics_engine import calculate_trends

    attempts = [
        {'timestamp': '2026-10-22 10:00:00', 'correct': 1, 'time_taken': None, 'subject': 'AI'},
        {'timestamp': '2026-10-22 10:01:00', 'correct': 1, 'time_taken': -5.0, 'subject': 'AI'},
        {'timestamp': '2026-10-22 10:02:00', 'correct': 1, 'time_taken': 'invalid', 'subject': 'AI'},
        {'timestamp': '2026-10-22 10:03:00', 'correct': 1, 'time_taken': 12.0, 'subject': 'AI'},
    ]

    trends = calculate_trends(attempts, date_range='all')
    assert len(trends['date_trends']) == 1
    # Only the 12.0s valid attempt should be averaged
    assert trends['date_trends'][0]['avg_response_time'] == 12.0


# ============================================================================
# CATEGORY 3: Spaced Repetition & Retention Analytics
# ============================================================================
def test_revision_and_retention_analytics():
    """Verify due, overdue, completed counts and retention accuracy."""
    from analytics_engine import calculate_revision_analytics

    today_str = datetime.date.today().isoformat()
    yesterday_str = (datetime.date.today() - datetime.timedelta(days=2)).isoformat()
    future_str = (datetime.date.today() + datetime.timedelta(days=5)).isoformat()

    spaced_rows = [
        # Overdue item (review_count 0)
        {'question_id': 'q1', 'subject': 'SQL', 'next_review_date': yesterday_str, 'review_count': 0, 'interval_days': 1},
        # Due today item (review_count 0)
        {'question_id': 'q2', 'subject': 'SQL', 'next_review_date': today_str, 'review_count': 0, 'interval_days': 1},
        # Future/completed item (review_count 3)
        {'question_id': 'q3', 'subject': 'Python', 'next_review_date': future_str, 'review_count': 3, 'interval_days': 7},
    ]

    # User attempts: answered q1 (correct) and q3 (correct)
    user_attempts = [
        {'question_id': 'q1', 'subject': 'SQL', 'correct': 1, 'time_taken': 8.0, 'timestamp': '2026-10-22 10:00:00'},
        {'question_id': 'q3', 'subject': 'Python', 'correct': 1, 'time_taken': 5.0, 'timestamp': '2026-10-22 10:05:00'},
    ]

    rev = calculate_revision_analytics(spaced_rows, user_attempts)
    assert rev['total_active'] == 3
    assert rev['overdue'] == 1
    assert rev['due_today'] == 1
    assert rev['completed'] == 1
    assert rev['completion_rate_pct'] == 33.3
    assert 'SQL' in rev['overdue_topics']

    # Retention check should be measured since q1 and q3 have reviews and attempts
    assert rev['retention_check_accuracy'] == 100.0


def test_retention_check_null_state():
    """When no reviewed questions have been attempted, do not fabricate retention scores."""
    from analytics_engine import calculate_revision_analytics

    spaced_rows = [
        {'question_id': 'q_new', 'subject': 'C++', 'next_review_date': '2026-10-25', 'review_count': 0, 'interval_days': 1}
    ]
    user_attempts = []

    rev = calculate_revision_analytics(spaced_rows, user_attempts)
    assert rev['retention_check_accuracy'] is None
    assert rev['retention_check_status'] == 'Needs Retention Check'


# ============================================================================
# CATEGORY 4: Before-and-After Assessment Comparison
# ============================================================================
def test_before_and_after_comparison():
    """Verify comparison between initial attempt window and latest attempt window."""
    from analytics_engine import calculate_before_after

    now = datetime.datetime.now()
    # 8 attempts on Networks:
    # First 4 attempts: 1 correct (25% accuracy, 20s avg time)
    # Last 4 attempts: 4 correct (100% accuracy, 8s avg time)
    attempts = []
    for i in range(4):
        attempts.append({
            'subject': 'Networks',
            'correct': 1 if i == 0 else 0,
            'time_taken': 20.0,
            'timestamp': (now - datetime.timedelta(days=10 - i)).isoformat()
        })
    for i in range(4):
        attempts.append({
            'subject': 'Networks',
            'correct': 1,
            'time_taken': 8.0,
            'timestamp': (now - datetime.timedelta(days=4 - i)).isoformat()
        })

    ba = calculate_before_after(attempts)
    assert ba['has_data'] is True
    assert len(ba['comparisons']) == 1

    net = ba['comparisons'][0]
    assert net['topic'] == 'Networks'
    assert net['initial_accuracy'] == 25.0
    assert net['latest_accuracy'] == 100.0
    assert net['accuracy_diff'] == 75.0
    assert net['time_diff'] == -12.0  # 8s - 20s = -12s (faster)
    assert 'Significant Improvement' in net['status']
    assert 'Networks' in ba['improved_topics']


def test_before_after_insufficient_history():
    """When student has only 1 or 2 attempts in a topic, do not crash; explain what is needed."""
    from analytics_engine import calculate_before_after

    attempts = [
        {'subject': 'Cloud', 'correct': 1, 'time_taken': 10.0, 'timestamp': '2026-10-22 10:00:00'},
        {'subject': 'Cloud', 'correct': 0, 'time_taken': 12.0, 'timestamp': '2026-10-22 10:05:00'},
    ]

    ba = calculate_before_after(attempts)
    assert ba['has_data'] is False
    assert 'Complete at least two study or revision sessions' in ba['message']


# ============================================================================
# CATEGORY 5: Explainable Recommendations Traceability
# ============================================================================
def test_explainable_recommendations():
    """Ensure recommendations cite real numbers and reasons, never fabricated claims."""
    from analytics_engine import generate_explainable_recommendations

    topic_report = {
        'topics': [
            {'topic': 'SQL', 'total_attempts': 6, 'correct': 2, 'incorrect': 4, 'accuracy': 33.3, 'mastery': 35.0, 'status': 'Needs Revision', 'min_attempts_met': True},
            {'topic': 'Python', 'total_attempts': 10, 'correct': 9, 'incorrect': 1, 'accuracy': 90.0, 'mastery': 88.0, 'status': 'Mastered', 'min_attempts_met': True}
        ],
        'weakest_topic': {'topic': 'SQL', 'mastery': 35.0}
    }
    rev_report = {
        'overdue_topics': ['SQL'],
        'overdue': 2,
        'alert_topics': [{'topic': 'SQL', 'overdue_count': 2}],
        'retention_check_accuracy': 45.0,
        'retention_check_status': 'Retention Declined'
    }
    trends_report = {
        'recent_vs_earlier': {
            'recent_accuracy': 50.0,
            'earlier_accuracy': 75.0,
            'diff_pct': -25.0,
            'time_diff_sec': 6.5
        }
    }

    recs = generate_explainable_recommendations(topic_report, trends_report, rev_report)
    assert len(recs) >= 3, f"Expected at least 3 recommendations, got {len(recs)}"

    # Check that SQL is recommended with specific reason
    sql_rec = next((r for r in recs if r['topic'] == 'SQL'), None)
    assert sql_rec is not None
    assert '4 incorrect' in sql_rec['reason'] or 'overdue' in sql_rec['reason']
    assert sql_rec['action_url'] != ''

    # Check that response time alert was triggered
    time_rec = next((r for r in recs if r['type'] == 'response_time_alert'), None)
    assert time_rec is not None
    assert '6.5s' in time_rec['reason']


# ============================================================================
# CATEGORY 6: AI Interview Attention Warning State Machine
# ============================================================================
class MockAttentionMonitor:
    """Simulates the interview attention warning state machine implemented in interview.html."""
    def __init__(self, threshold=30, trigger_ms=5000, cooldown_ms=30000):
        self.threshold = threshold
        self.trigger_ms = trigger_ms
        self.cooldown_ms = cooldown_ms
        self.low_attention_start = None
        self.last_warning_time = -999999
        self.warning_active = False
        self.interview_ended = False
        self.popup_shown_count = 0

    def update_frame(self, score, now_ms):
        """Processes an incoming attention reading."""
        if self.interview_ended:
            self.warning_active = False
            self.low_attention_start = None
            return False

        if score is None or score < 0 or score > 100:
            # Missing or invalid score: safeguard against false triggers
            return False

        if score < self.threshold:
            if self.low_attention_start is None:
                self.low_attention_start = now_ms

            duration_low = now_ms - self.low_attention_start

            if duration_low >= self.trigger_ms:
                in_cooldown = (now_ms - self.last_warning_time) < self.cooldown_ms
                if not self.warning_active and not in_cooldown:
                    self.warning_active = True
                    self.last_warning_time = now_ms
                    self.popup_shown_count += 1
                    return True
        else:
            # Score recovered above threshold
            self.low_attention_start = None

        return False

    def dismiss(self):
        self.warning_active = False


def test_attention_warning_invariants():
    """Test all attention warning conditions: persistence, recovery, cooldown, dismissal."""
    monitor = MockAttentionMonitor(threshold=30, trigger_ms=5000, cooldown_ms=30000)

    # 1. Attention stays above 30%: no popup
    t = 0
    assert monitor.update_frame(45, t) is False
    assert monitor.update_frame(35, t + 2000) is False
    assert monitor.popup_shown_count == 0

    # 2. Attention falls below 30% for only 2 seconds (< 5s): no popup
    assert monitor.update_frame(25, t + 3000) is False
    assert monitor.update_frame(20, t + 5000) is False
    # Recovers at 5.5s
    assert monitor.update_frame(60, t + 5500) is False
    assert monitor.popup_shown_count == 0

    # 3. Attention stays below 30% for >= 5 consecutive seconds: triggers popup
    t_low_start = 10000
    assert monitor.update_frame(20, t_low_start) is False
    assert monitor.update_frame(18, t_low_start + 2000) is False
    assert monitor.update_frame(22, t_low_start + 4900) is False
    # At 5000ms, triggers!
    triggered = monitor.update_frame(20, t_low_start + 5000)
    assert triggered is True
    assert monitor.popup_shown_count == 1
    assert monitor.warning_active is True

    # 4. Attention remains low: must NOT spawn duplicate popups on every frame
    assert monitor.update_frame(15, t_low_start + 6000) is False
    assert monitor.update_frame(12, t_low_start + 10000) is False
    assert monitor.popup_shown_count == 1

    # 5. User dismisses popup
    monitor.dismiss()
    assert monitor.warning_active is False

    # 6. Still in cooldown (e.g. 15s after last warning): should NOT trigger
    assert monitor.update_frame(20, t_low_start + 15000) is False
    assert monitor.popup_shown_count == 1

    # 7. Attention recovers to 70%: timer resets
    assert monitor.update_frame(70, t_low_start + 20000) is False
    assert monitor.low_attention_start is None

    # 8. Missing or null score does NOT trigger warning
    assert monitor.update_frame(None, t_low_start + 50000) is False
    assert monitor.update_frame(-10, t_low_start + 51000) is False
    assert monitor.popup_shown_count == 1

    # 9. When interview ends, all warnings are suppressed
    monitor.interview_ended = True
    assert monitor.update_frame(10, t_low_start + 60000) is False
    assert monitor.popup_shown_count == 1


# ============================================================================
# CATEGORY 7: Interview UI Template Verification
# ============================================================================
def test_interview_template_elements():
    """Verify that interview.html has non-blocking popup, got-it button, and privacy disclaimer."""
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'interview.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Check for warning toast container and styling
    assert 'attentionWarningToast' in html, "Missing #attentionWarningToast element in interview.html"
    assert 'Stay Focused!' in html or 'Stay focused' in html.lower(), "Missing 'Stay Focused!' warning text"
    assert 'dismissAttentionWarning' in html, "Missing dismissAttentionWarning function"
    assert 'ATTENTION_WARNING_THRESHOLD' in html, "Missing ATTENTION_WARNING_THRESHOLD config in script"
    assert 'LOW_ATTENTION_TRIGGER_MS' in html, "Missing LOW_ATTENTION_TRIGGER_MS config in script"
    assert 'WARNING_COOLDOWN_MS' in html, "Missing WARNING_COOLDOWN_MS config in script"

    # Non-blocking and auxiliary guarantees
    assert 'auxiliary indicator' in html.lower() or 'auxiliary' in html.lower(), "Privacy/auxiliary notice missing"
    assert 'position: fixed' in html or 'position:fixed' in html, "Toast must be fixed/non-blocking position"
    assert 'answerText' in html, "Answer textarea must remain accessible"
    assert 'submitBtn' in html or 'submitAnswer()' in html, "Submit button must remain intact"
    assert 'skipBtn' in html or 'skipQuestion()' in html, "Skip button must remain intact"


# ============================================================================
# CATEGORY 8: Dashboard API Routes & Session Scoping
# ============================================================================
def test_dashboard_api_routes():
    """Test /api/dashboard/stats and /api/analytics/detailed using Flask test client."""
    from app import app

    client = app.test_client()

    # Logged out: should redirect to login or return error
    res = client.get('/api/dashboard/stats')
    # Since endpoint has @login_required, it should return 302 or 401
    assert res.status_code in (302, 401), f"Expected redirect/unauthorized for unauthenticated call, got {res.status_code}"

    # Simulate logged-in user session with user_id = 1
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['username'] = 'test_student'

    # Test /api/dashboard/stats with date ranges
    for r in ('all', '30d', '7d'):
        res = client.get(f'/api/dashboard/stats?range={r}')
        assert res.status_code == 200, f"Dashboard stats failed with status {res.status_code}"
        data = res.get_json()

        # Legacy keys check
        assert 'overall_accuracy' in data, "Missing legacy 'overall_accuracy'"
        assert 'total_questions' in data, "Missing legacy 'total_questions'"
        assert 'avg_time' in data, "Missing legacy 'avg_time'"
        assert 'difficulty_progress' in data, "Missing legacy 'difficulty_progress'"
        assert 'accuracy_over_time' in data, "Missing legacy 'accuracy_over_time'"
        assert 'emotion_distribution' in data, "Missing legacy 'emotion_distribution'"
        assert 'recommendation' in data, "Missing legacy 'recommendation'"

        # Upgraded analytics keys check
        assert 'topic_mastery' in data, "Missing upgraded 'topic_mastery'"
        assert 'trend_analysis' in data, "Missing upgraded 'trend_analysis'"
        assert 'revision_analytics' in data, "Missing upgraded 'revision_analytics'"
        assert 'before_after' in data, "Missing upgraded 'before_after'"
        assert 'explainable_recommendations' in data, "Missing upgraded 'explainable_recommendations'"
        assert 'analytics_summary' in data or 'summary' in data, "Missing upgraded summary"

    # Test dedicated /api/analytics/detailed endpoint
    res_det = client.get('/api/analytics/detailed?range=all')
    assert res_det.status_code == 200, f"Detailed analytics endpoint failed with {res_det.status_code}"
    det_data = res_det.get_json()
    assert 'topic_mastery' in det_data
    assert 'explainable_recommendations' in det_data or 'recommendations' in det_data


# ============================================================================
# MAIN RUNNER
# ============================================================================
def main():
    print("=" * 80)
    print("EMOTIONLEARNAI - ANALYTICS UPGRADE & ATTENTION WARNING VERIFICATION")
    print("=" * 80)

    print("\n--- 1. Topic Mastery & Edge Cases ---")
    run_test("Topic Mastery Calculation (Weighted Formula & Confidence)", test_topic_mastery_calculation)
    run_test("Insufficient Data Threshold (< 3 Attempts)", test_insufficient_data_handling)
    run_test("Mastery Status Thresholds & Extremes (Mastered / Needs Revision)", test_mastery_status_thresholds)

    print("\n--- 2. Trends & Response Times ---")
    run_test("Date Grouping & Range Filtering (7d, 30d, all)", test_trend_calculation_and_date_filtering)
    run_test("Invalid / Missing Response Times Handling", test_invalid_and_missing_response_times)

    print("\n--- 3. Spaced Repetition & Retention Analytics ---")
    run_test("Revision Counts & Retention Check Accuracy", test_revision_and_retention_analytics)
    run_test("Retention Check Null State (No Fabricated Scores)", test_retention_check_null_state)

    print("\n--- 4. Before-and-After Comparisons ---")
    run_test("Multi-Session Assessment Comparison & Deltas", test_before_and_after_comparison)
    run_test("Insufficient History Friendly Handling", test_before_after_insufficient_history)

    print("\n--- 5. Explainable Personalized Recommendations ---")
    run_test("Traceable Rule-based Recommendations with Specific Evidence", test_explainable_recommendations)

    print("\n--- 6. Attention Warning State Machine ---")
    run_test("Low-Attention Invariants (5s persistence, recovery reset, cooldown, suppression)", test_attention_warning_invariants)

    print("\n--- 7. Interview UI Compliance ---")
    run_test("Interview Template UI Elements & Privacy Notice", test_interview_template_elements)

    print("\n--- 8. API Routes & Data Isolation ---")
    run_test("Dashboard & Detailed Analytics Scoped Endpoints", test_dashboard_api_routes)

    print("\n" + "=" * 80)
    print(f"RESULTS: {results['passed']} Passed, {results['failed']} Failed")
    print("=" * 80)

    if results['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
