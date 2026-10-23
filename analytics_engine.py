"""
analytics_engine.py — Advanced Learning Analytics & Recommendation Engine
========================================================================
Implements topic-wise mastery calculations, temporal performance trends,
spaced-repetition retention analytics, before-and-after comparisons, and
transparent, explainable learning recommendations.
"""

import math
import urllib.parse
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union


# ── Mastery Formula Constants ────────────────────────────────────────────────
MIN_ATTEMPTS_FOR_MASTERY = 3   # Minimum attempts before assigning mastery status
FULL_VOLUME_ATTEMPTS = 8       # Attempts threshold where volume confidence reaches 1.0


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely parse a float without raising exceptions on null, empty, or string values."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def calculate_topic_mastery(attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes topic-wise mastery metrics from stored question attempts.

    Mastery Formula:
      - Total attempts: N
      - Overall Accuracy: A = (correct / N) * 100
      - Recent Accuracy (last 5 attempts): A_recent
      - Volume confidence: V = min(1.0, N / FULL_VOLUME_ATTEMPTS)
      - Weighted Mastery = round((0.5 * A + 0.5 * A_recent) * (0.7 + 0.3 * V), 1)

    Status Thresholds:
      - N < MIN_ATTEMPTS_FOR_MASTERY -> 'Insufficient Data'
      - Mastery >= 85%                -> 'Mastered'
      - Mastery >= 65%                -> 'Proficient'
      - Mastery >= 40%                -> 'Learning'
      - Otherwise                     -> 'Needs Revision'
    """
    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for a in attempts:
        topic = a.get('subject') or 'General'
        if topic not in by_topic:
            by_topic[topic] = []
        by_topic[topic].append(a)

    results: List[Dict[str, Any]] = []

    for topic, topic_attempts in by_topic.items():
        sorted_attempts = sorted(
            topic_attempts,
            key=lambda x: str(x.get('timestamp') or x.get('id') or '')
        )

        n = len(sorted_attempts)
        correct_count = sum(1 for x in sorted_attempts if int(x.get('correct') or 0) == 1)
        incorrect_count = n - correct_count
        overall_accuracy = round((correct_count / n) * 100.0, 1) if n > 0 else 0.0

        valid_times = [
            _safe_float(x.get('time_taken')) for x in sorted_attempts
            if _safe_float(x.get('time_taken')) is not None and 0.5 <= _safe_float(x.get('time_taken')) <= 300.0
        ]
        avg_time = round(sum(valid_times) / len(valid_times), 1) if valid_times else 0.0

        recent_slice = sorted_attempts[-5:]
        recent_c = sum(1 for x in recent_slice if int(x.get('correct') or 0) == 1)
        recent_acc = round((recent_c / len(recent_slice)) * 100.0, 1)

        baseline_slice = sorted_attempts[:max(1, min(5, n // 2))]
        baseline_c = sum(1 for x in baseline_slice if int(x.get('correct') or 0) == 1)
        baseline_acc = round((baseline_c / len(baseline_slice)) * 100.0, 1)

        volume_factor = min(1.0, n / float(FULL_VOLUME_ATTEMPTS))

        if n < MIN_ATTEMPTS_FOR_MASTERY:
            mastery = round(overall_accuracy * 0.5 * volume_factor, 1)
            status = 'Insufficient Data'
            status_class = 'muted'
            status_desc = f'Needs {MIN_ATTEMPTS_FOR_MASTERY - n} more attempt(s) for baseline'
        else:
            raw_mastery = (0.50 * overall_accuracy + 0.50 * recent_acc) * (0.70 + 0.30 * volume_factor)
            mastery = round(min(100.0, max(0.0, raw_mastery)), 1)

            if mastery >= 85.0:
                status = 'Mastered'
                status_class = 'green'
                status_desc = 'Consistently high performance across question sets'
            elif mastery >= 65.0:
                status = 'Proficient'
                status_class = 'blue'
                status_desc = 'Solid grasp of core concepts with occasional errors'
            elif mastery >= 40.0:
                status = 'Learning'
                status_class = 'yellow'
                status_desc = 'Active progress; requires targeted practice'
            else:
                status = 'Needs Revision'
                status_class = 'red'
                status_desc = 'Multiple incorrect answers; revision strongly advised'

        results.append({
            'topic': topic,
            'total_attempts': n,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'correct_count': correct_count,
            'incorrect_count': incorrect_count,
            'accuracy': overall_accuracy,
            'recent_accuracy': recent_acc,
            'baseline_accuracy': baseline_acc,
            'mastery': mastery,
            'status': status,
            'status_class': status_class,
            'status_desc': status_desc,
            'avg_response_time': avg_time,
            'is_sufficient': n >= MIN_ATTEMPTS_FOR_MASTERY,
            'min_attempts_met': n >= MIN_ATTEMPTS_FOR_MASTERY,
            'confidence_factor': round(volume_factor, 2)
        })

    results.sort(key=lambda t: t['mastery'], reverse=True)
    return results


def get_topic_mastery_report(attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Returns mastery list alongside strongest and weakest topic pointers."""
    topics = calculate_topic_mastery(attempts)
    sufficient = [t for t in topics if t['is_sufficient']]
    strongest = max(sufficient, key=lambda x: x['mastery']) if sufficient else (topics[0] if topics else None)
    weakest = min(sufficient, key=lambda x: x['mastery']) if sufficient else (topics[-1] if topics else None)
    return {
        'topics': topics,
        'strongest_topic': strongest,
        'weakest_topic': weakest
    }


def calculate_trends(
    attempts: List[Dict[str, Any]],
    date_range: str = 'all'
) -> Dict[str, Any]:
    """
    Computes accuracy over time, average response time over time,
    and recent vs earlier baseline comparisons.

    Date range options: '7d', '30d', 'all'.
    """
    if not attempts:
        return {
            'date_range': date_range,
            'range': date_range,
            'dates': [],
            'accuracy_series': [],
            'time_series': [],
            'attempts_series': [],
            'date_trends': [],
            'recent_vs_earlier': {
                'recent_accuracy': 0.0,
                'earlier_accuracy': 0.0,
                'diff_pct': 0.0,
                'recent_avg_time': 0.0,
                'earlier_avg_time': 0.0,
                'time_diff_sec': 0.0
            },
            'recent_accuracy': 0.0,
            'earlier_accuracy': 0.0,
            'accuracy_change': 0.0,
            'recent_time': 0.0,
            'earlier_time': 0.0,
            'time_change': 0.0
        }

    now = datetime.now()
    if date_range == '7d':
        cutoff = now - timedelta(days=7)
    elif date_range == '30d':
        cutoff = now - timedelta(days=30)
    else:
        cutoff = None

    filtered = []
    for a in attempts:
        ts_str = a.get('timestamp') or ''
        if ts_str:
            try:
                clean_ts = ts_str.replace('Z', '').split('.')[0]
                dt = datetime.fromisoformat(clean_ts)
                if cutoff and dt < cutoff:
                    continue
            except Exception:
                pass
        filtered.append(a)

    if not filtered:
        filtered = attempts

    # Ensure chronological sorting
    filtered.sort(key=lambda x: str(x.get('timestamp') or ''))

    date_groups: Dict[str, List[Dict[str, Any]]] = {}
    for a in filtered:
        ts_str = str(a.get('timestamp') or '')
        d_key = ts_str[:10] if len(ts_str) >= 10 else 'Undated'
        if d_key not in date_groups:
            date_groups[d_key] = []
        date_groups[d_key].append(a)

    sorted_dates = sorted(date_groups.keys())
    dates: List[str] = []
    accuracy_series: List[float] = []
    time_series: List[float] = []
    attempts_series: List[int] = []
    date_trends: List[Dict[str, Any]] = []

    for d in sorted_dates:
        d_attempts = date_groups[d]
        n = len(d_attempts)
        c = sum(1 for x in d_attempts if int(x.get('correct') or 0) == 1)
        acc = round((c / n) * 100.0, 1)

        valid_times = [
            _safe_float(x.get('time_taken')) for x in d_attempts
            if _safe_float(x.get('time_taken')) is not None and 0.5 <= _safe_float(x.get('time_taken')) <= 300.0
        ]
        avg_t = round(sum(valid_times) / len(valid_times), 1) if valid_times else 0.0

        dates.append(d)
        accuracy_series.append(acc)
        time_series.append(avg_t)
        attempts_series.append(n)
        date_trends.append({
            'date': d,
            'accuracy': acc,
            'avg_response_time': avg_t,
            'total': n,
            'correct': c
        })

    # Recent vs Earlier comparison
    total_len = len(filtered)
    split_size = max(2, total_len // 3)
    earlier_slice = filtered[:split_size]
    recent_slice = filtered[-split_size:]

    earlier_c = sum(1 for x in earlier_slice if int(x.get('correct') or 0) == 1)
    earlier_acc = round((earlier_c / len(earlier_slice)) * 100.0, 1) if earlier_slice else 0.0

    recent_c = sum(1 for x in recent_slice if int(x.get('correct') or 0) == 1)
    recent_acc = round((recent_c / len(recent_slice)) * 100.0, 1) if recent_slice else 0.0

    earlier_times = [_safe_float(x.get('time_taken')) for x in earlier_slice if _safe_float(x.get('time_taken')) and _safe_float(x.get('time_taken')) > 0]
    earlier_avg_t = round(sum(earlier_times) / len(earlier_times), 1) if earlier_times else 0.0

    recent_times = [_safe_float(x.get('time_taken')) for x in recent_slice if _safe_float(x.get('time_taken')) and _safe_float(x.get('time_taken')) > 0]
    recent_avg_t = round(sum(recent_times) / len(recent_times), 1) if recent_times else 0.0

    acc_diff = round(recent_acc - earlier_acc, 1)
    time_diff = round(recent_avg_t - earlier_avg_t, 1)

    recent_vs_earlier = {
        'recent_accuracy': recent_acc,
        'earlier_accuracy': earlier_acc,
        'diff_pct': acc_diff,
        'recent_avg_time': recent_avg_t,
        'earlier_avg_time': earlier_avg_t,
        'time_diff_sec': time_diff
    }

    return {
        'date_range': date_range,
        'range': date_range,
        'dates': dates,
        'accuracy_series': accuracy_series,
        'time_series': time_series,
        'attempts_series': attempts_series,
        'date_trends': date_trends,
        'recent_vs_earlier': recent_vs_earlier,
        'recent_accuracy': recent_acc,
        'earlier_accuracy': earlier_acc,
        'accuracy_change': acc_diff,
        'recent_time': recent_avg_t,
        'earlier_time': earlier_avg_t,
        'time_change': time_diff
    }


def calculate_revision_analytics(
    arg1: Any,
    arg2: Any = None,
    arg3: Any = None
) -> Dict[str, Any]:
    """
    Flexible wrapper supporting:
      calculate_revision_analytics(user_id, spaced_rep_rows, attempts)
      OR
      calculate_revision_analytics(spaced_rep_rows, attempts, user_id=None)
    """
    if isinstance(arg1, (int, type(None))) and isinstance(arg2, list):
        user_id = arg1
        spaced_rep_rows = arg2
        attempts = arg3 or []
    else:
        spaced_rep_rows = arg1 or []
        attempts = arg2 or []
        user_id = arg3

    today_str = date.today().isoformat()
    total_scheduled = len(spaced_rep_rows)
    due_today_count = 0
    overdue_count = 0
    completed_count = 0
    overdue_topics: Dict[str, int] = {}

    for row in spaced_rep_rows:
        nrd = str(row.get('next_review_date') or '')
        rc = int(row.get('review_count') or 0)
        subj = str(row.get('subject') or 'General')

        if rc > 0:
            completed_count += 1

        if nrd:
            if nrd < today_str:
                overdue_count += 1
                overdue_topics[subj] = overdue_topics.get(subj, 0) + 1
            elif nrd == today_str:
                due_today_count += 1

    completion_rate = round((completed_count / total_scheduled) * 100.0, 1) if total_scheduled > 0 else 0.0

    scheduled_qids = {r.get('question_id') for r in spaced_rep_rows if r.get('question_id')}
    review_attempts = [a for a in attempts if a.get('question_id') in scheduled_qids]

    if review_attempts:
        correct_reviews = sum(1 for a in review_attempts if int(a.get('correct') or 0) == 1)
        retention_acc = round((correct_reviews / len(review_attempts)) * 100.0, 1)
        retention_status = 'Measured Retention'
    else:
        retention_acc = None
        retention_status = 'Needs Retention Check'

    alert_topics = []
    for t, cnt in overdue_topics.items():
        alert_topics.append({
            'topic': t,
            'overdue_count': cnt,
            'reason': f'{cnt} spaced-repetition card(s) past scheduled review date'
        })

    return {
        'total_scheduled': total_scheduled,
        'total_active': total_scheduled,
        'due_today': due_today_count,
        'overdue': overdue_count,
        'completed': completed_count,
        'completion_rate': completion_rate,
        'completion_rate_pct': completion_rate,
        'retention_check_accuracy': retention_acc,
        'retention_check_status': retention_status,
        'alert_topics': alert_topics,
        'overdue_topics': list(overdue_topics.keys())
    }


def calculate_before_after(
    attempts: List[Dict[str, Any]],
    skill_assessments: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Compares student accuracy and response times between initial and latest
    assessment/practice windows for each topic with sufficient attempts.
    """
    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for a in attempts:
        topic = a.get('subject') or 'General'
        if topic not in by_topic:
            by_topic[topic] = []
        by_topic[topic].append(a)

    comparisons: List[Dict[str, Any]] = []

    for topic, topic_attempts in by_topic.items():
        sorted_attempts = sorted(
            topic_attempts,
            key=lambda x: str(x.get('timestamp') or x.get('id') or '')
        )

        n = len(sorted_attempts)
        if n < 4:
            continue

        window_size = min(5, max(2, n // 2))
        initial_set = sorted_attempts[:window_size]
        latest_set = sorted_attempts[-window_size:]

        init_c = sum(1 for x in initial_set if int(x.get('correct') or 0) == 1)
        init_acc = round((init_c / len(initial_set)) * 100.0, 1)

        late_c = sum(1 for x in latest_set if int(x.get('correct') or 0) == 1)
        late_acc = round((late_c / len(latest_set)) * 100.0, 1)

        point_change = round(late_acc - init_acc, 1)

        init_times = [_safe_float(x.get('time_taken')) for x in initial_set if _safe_float(x.get('time_taken')) and _safe_float(x.get('time_taken')) > 0]
        init_avg_t = round(sum(init_times) / len(init_times), 1) if init_times else 0.0

        late_times = [_safe_float(x.get('time_taken')) for x in latest_set if _safe_float(x.get('time_taken')) and _safe_float(x.get('time_taken')) > 0]
        late_avg_t = round(sum(late_times) / len(late_times), 1) if late_times else 0.0

        time_change = round(late_avg_t - init_avg_t, 1)

        init_date = str(initial_set[0].get('timestamp') or '')[:10]
        late_date = str(latest_set[-1].get('timestamp') or '')[:10]

        if point_change >= 10.0:
            status = 'Significant Improvement'
            status_class = 'green'
        elif point_change >= 3.0:
            status = 'Moderate Improvement'
            status_class = 'blue'
        elif point_change <= -10.0:
            status = 'Needs Attention'
            status_class = 'red'
        else:
            status = 'Steady / Maintained'
            status_class = 'muted'

        comparisons.append({
            'topic': topic,
            'initial_date': init_date or 'Initial',
            'latest_date': late_date or 'Latest',
            'initial_accuracy': init_acc,
            'latest_accuracy': late_acc,
            'point_change': point_change,
            'accuracy_diff': point_change,
            'initial_time': init_avg_t,
            'initial_avg_time': init_avg_t,
            'latest_time': late_avg_t,
            'latest_avg_time': late_avg_t,
            'time_change': time_change,
            'time_diff': time_change,
            'initial_count': len(initial_set),
            'latest_count': len(latest_set),
            'status': status,
            'status_class': status_class,
            'sample_size': n
        })

    comparisons.sort(key=lambda x: x['point_change'], reverse=True)
    has_data = len(comparisons) > 0
    message = "Comparison generated." if has_data else "Complete at least two study or revision sessions per topic to unlock before-and-after improvement analysis."
    improved_topics = [c['topic'] for c in comparisons if c['point_change'] >= 3.0]
    declined_topics = [c['topic'] for c in comparisons if c['point_change'] <= -3.0]

    return {
        'has_data': has_data,
        'comparisons': comparisons,
        'improved_topics': improved_topics,
        'declined_topics': declined_topics,
        'message': message
    }


def generate_explainable_recommendations(
    topic_mastery: Any,
    trends: Optional[Dict[str, Any]] = None,
    revision_data: Optional[Dict[str, Any]] = None,
    attempts: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Produces transparent, data-traceable recommendations.
    Every recommendation references the concrete performance metrics behind it.
    """
    recs: List[Dict[str, Any]] = []

    # Handle topic_mastery whether list or dict
    if isinstance(topic_mastery, dict) and 'topics' in topic_mastery:
        topics_list = topic_mastery['topics']
    elif isinstance(topic_mastery, list):
        topics_list = topic_mastery
    else:
        topics_list = []

    rev_dict = revision_data or {}
    trends_dict = trends or {}

    # 1. Overdue Spaced-Repetition Cards
    for alert in rev_dict.get('alert_topics', []):
        topic = alert.get('topic', 'General')
        cnt = alert.get('overdue_count', 1)
        recs.append({
            'type': 'revision_overdue',
            'badge': '⏰ Overdue Review',
            'badge_class': 'red',
            'topic': topic,
            'title': f'Complete pending review for {topic}',
            'recommendation': f'Revise {topic} because you have {cnt} overdue revision item(s).',
            'reason': f'You have {cnt} spaced-repetition question(s) overdue for review. Prompt revision solidifies memory retention.',
            'action_label': f'Review {topic}',
            'action_param': topic,
            'action_url': f"/?subject={urllib.parse.quote_plus(topic)}"
        })

    # 2. Weak Topics (Needs Revision / Multiple Errors)
    weak_topics = [t for t in topics_list if t.get('status') == 'Needs Revision' and t.get('total_attempts', 0) >= MIN_ATTEMPTS_FOR_MASTERY]
    for wt in weak_topics[:2]:
        topic = wt['topic']
        inc = wt.get('incorrect_count', wt.get('incorrect', 0))
        tot = wt.get('total_attempts', 0)
        acc = wt.get('accuracy', 0.0)
        recs.append({
            'type': 'accuracy_alert',
            'badge': '🎯 Weak Concept',
            'badge_class': 'red',
            'topic': topic,
            'title': f'Targeted revision needed in {topic}',
            'recommendation': f'Practice {topic} to improve your accuracy above {acc}%.',
            'reason': f'Your overall accuracy in {topic} is {acc}% with {inc} incorrect answers out of {tot} attempted questions.',
            'action_label': f'Practice {topic}',
            'action_param': topic,
            'action_url': f"/?subject={urllib.parse.quote_plus(topic)}"
        })

    # 3. Declining Performance Alert
    recent_vs_earlier = trends_dict.get('recent_vs_earlier', {})
    if recent_vs_earlier and recent_vs_earlier.get('diff_pct', 0) <= -15.0:
        r_acc = recent_vs_earlier.get('recent_accuracy', 0)
        b_acc = recent_vs_earlier.get('earlier_accuracy', 0)
        recs.append({
            'type': 'accuracy_alert',
            'badge': '📉 Accuracy Drop',
            'badge_class': 'yellow',
            'topic': 'Core Topics',
            'title': 'Recent accuracy decline detected',
            'recommendation': 'Review recent topics before moving to advanced concepts.',
            'reason': f'Your recent accuracy ({r_acc}%) has dropped below your earlier baseline ({b_acc}%). A brief review is recommended.',
            'action_label': 'Review Baseline',
            'action_param': 'General',
            'action_url': '/?subject=General'
        })

    # 4. Response Time Outlier
    if recent_vs_earlier and recent_vs_earlier.get('time_diff_sec', 0) > 5.0:
        diff_s = recent_vs_earlier.get('time_diff_sec', 0)
        recs.append({
            'type': 'response_time_alert',
            'badge': '⏱️ Response Pacing',
            'badge_class': 'blue',
            'topic': 'Time Management',
            'title': 'Response time is increasing',
            'recommendation': 'Focus on building speed through timed practice drills.',
            'reason': f'Your average response time has increased by {diff_s}s compared with earlier sessions.',
            'action_label': 'Speed Practice',
            'action_param': 'General',
            'action_url': '/?subject=General'
        })

    # 5. Mastered Topic Challenge
    mastered_topics = [t for t in topics_list if t.get('status') == 'Mastered']
    for mt in mastered_topics[:1]:
        topic = mt['topic']
        recs.append({
            'type': 'mastery_milestone',
            'badge': '🏆 Topic Mastered',
            'badge_class': 'green',
            'topic': topic,
            'title': f'Advance to Hard challenges in {topic}',
            'recommendation': f'You have mastered {topic}. Try hard questions to maintain fluency.',
            'reason': f'You have achieved {mt.get("mastery", 85)}% mastery with {mt.get("correct_count", mt.get("correct", 0))} correct answers.',
            'action_label': f'Challenge {topic}',
            'action_param': topic,
            'action_url': f"/?subject={urllib.parse.quote_plus(topic)}"
        })

    if not recs:
        recs.append({
            'type': 'general',
            'badge': '🚀 Getting Started',
            'badge_class': 'blue',
            'topic': 'Python Basics',
            'title': 'Start your foundational practice sessions',
            'recommendation': 'Answer 3-5 questions to establish your baseline mastery score.',
            'reason': 'Answer at least 3-5 questions in any topic to establish your baseline mastery and unlock personalized telemetry.',
            'action_label': 'Start Practice',
            'action_param': 'Python Basics',
            'action_url': '/?subject=Python'
        })

    return recs


def get_student_analytics_report(
    user_id: Optional[int],
    attempts: List[Dict[str, Any]],
    spaced_rep_rows: List[Dict[str, Any]],
    skill_assessments: Optional[List[Dict[str, Any]]] = None,
    date_range: str = 'all'
) -> Dict[str, Any]:
    """
    Compiles the full analytics report for a student.
    Handles empty attempts, zero values, and missing timestamps safely.
    """
    topic_mastery = calculate_topic_mastery(attempts)
    mastery_report = get_topic_mastery_report(attempts)
    trends = calculate_trends(attempts, date_range=date_range)
    revision_data = calculate_revision_analytics(user_id, spaced_rep_rows, attempts)
    before_after = calculate_before_after(attempts, skill_assessments)
    recommendations = generate_explainable_recommendations(topic_mastery, trends, revision_data, attempts)

    total_q = len(attempts)
    total_c = sum(1 for a in attempts if int(a.get('correct') or 0) == 1)
    overall_acc = round((total_c / total_q) * 100.0, 1) if total_q > 0 else 0.0

    valid_masteries = [t['mastery'] for t in topic_mastery if t['is_sufficient']]
    avg_mastery = round(sum(valid_masteries) / len(valid_masteries), 1) if valid_masteries else 0.0

    strongest = mastery_report.get('strongest_topic')
    weakest = mastery_report.get('weakest_topic')

    times = [_safe_float(a.get('time_taken')) for a in attempts if _safe_float(a.get('time_taken')) and _safe_float(a.get('time_taken')) > 0]
    overall_avg_time = round(sum(times) / len(times), 1) if times else 0.0

    return {
        'summary': {
            'total_questions': total_q,
            'total_correct': total_c,
            'overall_accuracy': overall_acc,
            'average_mastery': avg_mastery,
            'overall_avg_time': overall_avg_time,
            'strongest_topic': strongest['topic'] if strongest else 'None yet',
            'strongest_mastery': strongest['mastery'] if strongest else 0.0,
            'weakest_topic': weakest['topic'] if weakest else 'None yet',
            'weakest_mastery': weakest['mastery'] if weakest else 0.0,
            'revision_completion_rate': revision_data['completion_rate'],
            'overdue_count': revision_data['overdue'],
            'has_sufficient_data': len(valid_masteries) > 0,
            'recent_improvement_pct': trends['recent_vs_earlier']['diff_pct']
        },
        'topic_mastery': mastery_report,
        'trends': trends,
        'revision': revision_data,
        'before_after': before_after,
        'recommendations': recommendations
    }
