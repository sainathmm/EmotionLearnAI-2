"""
ai_explainer.py — AI-Powered Wrong Answer Explanation
======================================================
Uses Google Gemini (google.genai SDK) to generate a concise, student-friendly
explanation when a learner answers a question incorrectly.

Falls back to a static template explanation if Gemini is unavailable or
the API call fails, so the feature never breaks the quiz flow.

Usage in app.py:
    from ai_explainer import generate_explanation
    text = generate_explanation(question_title, question_body, correct_option)
"""

import os
import re

# GEMINI_API_KEY is loaded from .env by app.py before this module is imported.
# We read it here so ai_explainer can be used standalone too.
_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

# Lazy-initialise the Gemini client so import never raises even if the
# google-genai package is missing.
_client = None
_GEMINI_OK = False

try:
    import google.genai as genai
    if _GEMINI_KEY:
        _client   = genai.Client(api_key=_GEMINI_KEY)
        _GEMINI_OK = True
except ImportError:
    pass  # Gemini not installed — fallback will be used


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explanation(
    question_title: str,
    question_body:  str,
    correct_answer: str,
    wrong_answer:   str = "",
    subject:        str = "",
) -> dict:
    """
    Generate a plain-English explanation for why the correct answer is right.

    Args:
        question_title : Short title of the question (e.g. "What is Python?").
        question_body  : The concept explanation shown to the student.
        correct_answer : Text of the correct option.
        wrong_answer   : Text of the option the student chose (optional).
        subject        : Subject name for context (optional).

    Returns:
        dict:
            explanation  (str)  — 2-4 sentence explanation
            tip          (str)  — one-line memory tip
            source       (str)  — 'gemini' | 'fallback'
    """
    if _GEMINI_OK and _client is not None:
        result = _gemini_explanation(
            question_title, question_body, correct_answer, wrong_answer, subject
        )
        if result:
            return result

    # Fallback: build a static explanation from the question content
    return _static_explanation(
        question_title, question_body, correct_answer, wrong_answer
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gemini_explanation(title, body, correct, wrong, subject) -> dict | None:
    """Call Gemini API and parse the response. Returns None on any failure."""
    wrong_clause = f' The student chose "{wrong}" instead.' if wrong else ''
    subj_clause  = f' This is a {subject} question.' if subject else ''

    prompt = (
        f'A student answered a quiz question incorrectly.{subj_clause}\n\n'
        f'Question: "{title}"\n'
        f'Concept: {body}\n'
        f'Correct answer: "{correct}"{wrong_clause}\n\n'
        'In 2-3 simple sentences, explain WHY the correct answer is right '
        'and why the student might have been confused. '
        'Then give ONE short memory tip (max 15 words) to help them remember.\n\n'
        'Reply in this exact format (no markdown, no extra text):\n'
        'EXPLANATION: <your explanation here>\n'
        'TIP: <your memory tip here>'
    )

    for model_name in ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-2.0-flash']:
        try:
            response = _client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = response.text.strip()

            # Parse structured response
            exp_match = re.search(r'EXPLANATION:\s*(.+?)(?=TIP:|$)', text, re.DOTALL)
            tip_match = re.search(r'TIP:\s*(.+)', text, re.DOTALL)

            explanation = exp_match.group(1).strip() if exp_match else text[:300]
            tip         = tip_match.group(1).strip()  if tip_match else ''

            return {
                'explanation': explanation,
                'tip':         tip,
                'source':      'gemini',
            }
        except Exception as e:
            # Try next model or fallback
            continue

    return None


def _static_explanation(title, body, correct, wrong) -> dict:
    """
    Build a template-based explanation without any API call.
    Always succeeds.
    """
    wrong_clause = (
        f' "{wrong}" is incorrect because it does not match the definition.'
        if wrong else ''
    )
    explanation = (
        f'The correct answer is "{correct}".{wrong_clause} '
        f'{body[:200].rstrip(".")}. '
        'Re-read the concept above and try the question again to reinforce your understanding.'
    )
    tip = f'Remember: {correct} — review the concept once more to lock it in.'

    return {
        'explanation': explanation,
        'tip':         tip,
        'source':      'fallback',
    }
