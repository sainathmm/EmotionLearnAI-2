# -*- coding: utf-8 -*-
"""
EmotionLearnAI - Intelligent Personalized Learning Framework
============================================================
PHASE 0: Emotion Smoothing, SQLite Tracking, Dashboard, Interventions, Recommendations
PHASE 1: User Auth, Persistent Profiles, XP & Levels, Daily Streaks
PHASE 2: Per-User Dashboard, Leaderboard, Achievement Badges, Profile Page
PHASE 3 (PRODUCTION UPGRADE):
  - PostgreSQL support with SQLite fallback  (db.py)
  - Gemini AI Question Generator            (generate_question)
  - Spaced Repetition System                (schedule_review / get_due_reviews)
  - Per-Question Analytics                  (update_question_stats)
  - Rate Limiting via Flask-Limiter
  - Environment Variables via python-dotenv
"""

# ── Standard Library ───────────────────────────────────────────────────────────
import cv2, numpy as np, base64, time, random, sqlite3, os, secrets, re, json, math
from datetime import datetime, date, timedelta
from collections import deque, Counter
from functools import wraps
from typing import Any, Optional, Dict, List, Tuple

# ── Phase 3: Load .env file before anything else ──────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Phase 3: Database abstraction (PostgreSQL + SQLite fallback) ───────────────
from db import get_conn, adapt_sql, USE_POSTGRES, DB_PATH, db_info

# ── New Multimodal AI & Vision Modules ───────────────────────────────────────
from vision_pipeline import (assess_camera_quality, detect_faces_and_landmarks,
                             align_face, preprocess_emotion_face, decode_base64_frame)
from emotion_engine import infer_emotion, clear_session_tracker
from attention       import (detect_attention, calibrate_session,
                             clear_session as clear_attention_session)
from cognitive_engine import evaluate_cognitive_state, COGNITIVE_STATE_METADATA
from personalization import (get_weak_topics, get_subject_accuracy,
                              get_personalized_recommendation, prioritize_question_pool,
                              get_adaptation_tracker, clear_adaptation_tracker,
                              evaluate_learning_decision, get_user_topic_difficulty,
                              update_user_topic_difficulty, record_adaptive_decision,
                              get_adaptive_history, get_adaptive_summary,
                              init_adaptive_tables, normalize_difficulty)
from ai_explainer    import generate_explanation
from model_utils     import (predict_next_correct, model_available,
                              get_model_info, reload_model)
from analytics_engine import get_student_analytics_report

# -- Phase 3: Google Gemini AI (new SDK: google.genai) -------------------------
try:
    import google.genai as genai
    from google.genai import types as genai_types
    _GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
    if _GEMINI_KEY:
        _GEMINI_CLIENT = genai.Client(api_key=_GEMINI_KEY)
        GEMINI_AVAILABLE = True
    else:
        _GEMINI_CLIENT  = None
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False
    print("[INFO] google-generativeai not installed — AI questions disabled.")

# ── Werkzeug (already installed with Flask) ───────────────────────────────────
from werkzeug.security import generate_password_hash, check_password_hash

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import (Flask, request, jsonify, render_template,
                   session, redirect, url_for, flash)
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# SECRET_KEY — loaded from .env (production) or random (development)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))

# ── Phase 3: Rate Limiter ─────────────────────────────────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["1000 per day", "300 per hour"],
        storage_uri=os.getenv("REDIS_URL", "memory://"),
    )
except ImportError:
    class _DummyLimiter:
        def limit(self, *a, **kw):
            def decorator(f): return f
            return decorator
        def exempt(self, f): return f
    limiter = _DummyLimiter()
    print("[INFO] flask-limiter not installed — rate limiting disabled.")

# =============================================================================
# CONSTANTS — Emotion / Multimodal Cognitive States
# =============================================================================

EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']

COGNITIVE_STATES = {
    'flow':       {'desc': 'Optimal Learning Zone — Deep focus and high mastery.',       'color': '#00C851', 'emoji': '🌟'},
    'engaged':    {'desc': 'Actively Processing & Thinking — Steady cognitive effort.',  'color': '#0099CC', 'emoji': '🎯'},
    'distracted': {'desc': 'Attention Lapsed — Looking away or visual focus lost.',      'color': '#FFBB33', 'emoji': '⚠️'},
    'overloaded': {'desc': 'Cognitively Overwhelmed — High task difficulty or fatigue.', 'color': '#FF4444', 'emoji': '⚡'},
    'struggling': {'desc': 'Conceptual Obstacle — Needs scaffolding, hints, or review.', 'color': '#E040FB', 'emoji': '🧩'},
    'bored':      {'desc': 'Under-Challenged — Material may be too easy.',              'color': '#FF8800', 'emoji': '💡'},
    'anxious':    {'desc': 'Performance Tension — Elevated stress or uncertainty.',      'color': '#AA00FF', 'emoji': '🌀'},
    'uncertain':  {'desc': 'Gathering Multimodal Signals — Retaining stable baseline.',  'color': '#78909C', 'emoji': '🔄'},
}

EMOTION_TO_COGNITIVE = {
    'happy':    'flow',
    'neutral':  'engaged',
    'surprise': 'engaged',
    'sad':      'bored',
    'angry':    'overloaded',
    'fear':     'anxious',
    'disgust':  'distracted',
}

DIFF_RULE = {
    'flow':       'increase',
    'engaged':    'maintain',
    'bored':      'increase',
    'overloaded': 'decrease',
    'anxious':    'decrease',
    'distracted': 'maintain',
}

# =============================================================================
# PHASE 1 — XP & LEVEL CONSTANTS
# =============================================================================

XP_PER_CORRECT    = 10   # XP for a correct answer
XP_PER_ATTEMPT    = 2    # XP just for attempting (even if wrong)
XP_STREAK_BONUS   = 25   # Bonus XP every 5 consecutive correct answers

# (min_xp, level_number, level_name, display_emoji)
LEVEL_THRESHOLDS = [
    (0,    1, 'Beginner', '🌱'),
    (100,  2, 'Explorer', '🔍'),
    (300,  3, 'Learner',  '📚'),
    (600,  4, 'Thinker',  '💡'),
    (1000, 5, 'Scholar',  '🎓'),
    (1500, 6, 'Expert',   '🏆'),
    (2500, 7, 'Master',   '⭐'),
]

# =============================================================================
# PHASE 2 — ACHIEVEMENT BADGE CONSTANTS
# =============================================================================

# Each badge: (badge_id, emoji, name, description)
ALL_BADGES = [
    ('first_step',    '🌟', 'First Step',    'Answer your first question'),
    ('on_fire',       '🔥', 'On Fire',       'Achieve a 5-day login streak'),
    ('perfect_score', '💯', 'Perfect Score', '100% accuracy over 5+ questions in a session'),
    ('scholar',       '🎓', 'Scholar',       'Reach Level 5'),
    ('speed_demon',   '⚡', 'Speed Demon',   'Answer correctly in under 5 seconds'),
    ('brain_power',   '🧠', 'Brain Power',   'Answer 50 questions total'),
    ('top_learner',   '🏆', 'Top Learner',   'Reach #1 on the leaderboard'),
    ('streak_master', '🔑', 'Streak Master', 'Achieve a 10-day login streak'),
    ('century',       '💎', 'Century',       'Answer 100 questions total'),
    ('expert_level',  '👑', 'Expert',        'Reach Level 6'),
]

# =============================================================================
# QUESTION BANK (unchanged)
# =============================================================================

ALL_QUESTIONS = {
    "Python Basics": [
        {"id": "p1",  "diff": "easy",   "title": "What is Python?",
         "body": "Python is a high-level, easy-to-read programming language used in AI, web, and data science.",
         "code": "print('Hello, World!')",
         "quiz": {"q": "Python is a:", "opts": ["Compiled", "Interpreted", "Assembly"], "ans": 1}},
        {"id": "p2",  "diff": "easy",   "title": "Variables & Data Types",
         "body": "Variables store data. Python supports int, float, str, bool types.",
         "code": "name = 'Alice'\nage = 20\nheight = 5.6\nprint(name, age, height)",
         "quiz": {"q": "Which data type stores text?", "opts": ["int", "float", "str"], "ans": 2}},
        {"id": "p3",  "diff": "easy",   "title": "Input & Output",
         "body": "Use print() to display output and input() to take user input.",
         "code": "name = input('Enter your name: ')\nprint('Hello,', name)",
         "quiz": {"q": "Which function takes user input?", "opts": ["print()", "input()", "scan()"], "ans": 1}},
        {"id": "p4",  "diff": "medium", "title": "Functions",
         "body": "Functions are reusable blocks of code defined with 'def' keyword.",
         "code": "def add(a, b):\n    return a + b\nprint(add(3, 4))  # 7",
         "quiz": {"q": "Keyword to define a function?", "opts": ["func", "def", "lambda"], "ans": 1}},
        {"id": "p5",  "diff": "medium", "title": "Lists & Loops",
         "body": "Lists hold multiple items. For loops iterate over sequences.",
         "code": "nums = [1,2,3,4,5]\nfor n in nums:\n    print(n * 2)",
         "quiz": {"q": "First element index in a list?", "opts": ["1", "0", "-1"], "ans": 1}},
        {"id": "p6",  "diff": "medium", "title": "If-Else Conditions",
         "body": "Conditional statements control the flow of a program.",
         "code": "age = 18\nif age >= 18:\n    print('Adult')\nelse:\n    print('Minor')",
         "quiz": {"q": "Which keyword starts a condition?", "opts": ["when", "if", "check"], "ans": 1}},
        {"id": "p7",  "diff": "medium", "title": "Dictionaries",
         "body": "Dictionaries store key-value pairs. Keys must be unique.",
         "code": "student = {'name':'Alice','age':20}\nprint(student['name'])  # Alice",
         "quiz": {"q": "How to access dict value?", "opts": ["dict.get", "dict[key]", "dict->key"], "ans": 1}},
        {"id": "p8",  "diff": "hard",   "title": "OOP & Classes",
         "body": "Classes are blueprints for objects. __init__ is the constructor.",
         "code": "class Dog:\n    def __init__(self, name):\n        self.name = name\n    def bark(self):\n        return f'{self.name} says Woof!'",
         "quiz": {"q": "__init__ is a:", "opts": ["Destructor", "Constructor", "Static method"], "ans": 1}},
        {"id": "p9",  "diff": "hard",   "title": "Exception Handling",
         "body": "Use try-except to handle errors gracefully in Python.",
         "code": "try:\n    x = int(input('Enter number: '))\nexcept ValueError:\n    print('Invalid input!')",
         "quiz": {"q": "Which block catches errors?", "opts": ["try", "except", "finally"], "ans": 1}},
        {"id": "p10", "diff": "hard",   "title": "File Handling",
         "body": "Python can read and write files using open() function.",
         "code": "with open('file.txt','w') as f:\n    f.write('Hello!')\nwith open('file.txt','r') as f:\n    print(f.read())",
         "quiz": {"q": "Which mode opens file for writing?", "opts": ["r", "w", "a"], "ans": 1}},
    ],
    "Data Science": [
        {"id": "d1", "diff": "easy",   "title": "What is Data Science?",
         "body": "Data Science extracts insights from data using statistics, programming, and domain knowledge.",
         "code": "import pandas as pd\ndf = pd.DataFrame({'A':[1,2,3]})\nprint(df)",
         "quiz": {"q": "Data Science combines:", "opts": ["Only math", "Stats+Code+Domain", "Only coding"], "ans": 1}},
        {"id": "d2", "diff": "easy",   "title": "Types of Data",
         "body": "Data can be structured (tables), unstructured (text/images), or semi-structured (JSON).",
         "code": "data = {'Name':['Alice','Bob'],'Score':[90,85]}\nimport pandas as pd\nprint(pd.DataFrame(data))",
         "quiz": {"q": "CSV files are:", "opts": ["Unstructured", "Structured", "Semi-structured"], "ans": 1}},
        {"id": "d3", "diff": "medium", "title": "NumPy Arrays",
         "body": "NumPy provides fast N-dimensional arrays for numerical computing.",
         "code": "import numpy as np\narr = np.array([1,2,3,4,5])\nprint(arr.mean(), arr.std())",
         "quiz": {"q": "NumPy is used for:", "opts": ["Web dev", "Numerical computing", "Databases"], "ans": 1}},
        {"id": "d4", "diff": "medium", "title": "Pandas DataFrames",
         "body": "Pandas DataFrame is a 2D table structure for data manipulation.",
         "code": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())\nprint(df.describe())",
         "quiz": {"q": "df.head() shows:", "opts": ["Last 5 rows", "First 5 rows", "All rows"], "ans": 1}},
        {"id": "d5", "diff": "hard",   "title": "ML Pipeline",
         "body": "ML Pipeline: Data -> Preprocess -> Train Model -> Evaluate -> Deploy.",
         "code": "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)\nprint(model.score(X_test, y_test))",
         "quiz": {"q": "Overfitting means:", "opts": ["Too simple", "Memorizes training data", "Perfect model"], "ans": 1}},
    ],
    "Machine Learning": [
        {"id": "ml1", "diff": "easy",   "title": "What is Machine Learning?",
         "body": "ML is a subset of AI where systems learn from data to make predictions.",
         "code": "# ML workflow\n# 1. Collect Data  2. Train  3. Evaluate  4. Predict",
         "quiz": {"q": "ML learns from:", "opts": ["Rules", "Data", "Humans"], "ans": 1}},
        {"id": "ml2", "diff": "easy",   "title": "Types of ML",
         "body": "Supervised (labeled), Unsupervised (unlabeled), Reinforcement (reward-based).",
         "code": "# Supervised: predict price\n# Unsupervised: group customers\n# Reinforcement: game bot",
         "quiz": {"q": "Labeled data is used in:", "opts": ["Unsupervised", "Supervised", "Reinforcement"], "ans": 1}},
        {"id": "ml3", "diff": "medium", "title": "Linear Regression",
         "body": "Linear Regression predicts a continuous value by fitting a line to data.",
         "code": "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)",
         "quiz": {"q": "Linear Regression predicts:", "opts": ["Categories", "Continuous values", "Clusters"], "ans": 1}},
        {"id": "ml4", "diff": "medium", "title": "Classification",
         "body": "Classification predicts a category. Example: spam or not spam.",
         "code": "from sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier()\nmodel.fit(X_train, y_train)",
         "quiz": {"q": "Classification output is:", "opts": ["A number", "A category", "A cluster"], "ans": 1}},
        {"id": "ml5", "diff": "hard",   "title": "Neural Networks",
         "body": "Neural Networks are layers of neurons that learn complex patterns.",
         "code": "from tensorflow import keras\nmodel = keras.Sequential([\n    keras.layers.Dense(64, activation='relu'),\n    keras.layers.Dense(1, activation='sigmoid')\n])",
         "quiz": {"q": "Activation function adds:", "opts": ["Linearity", "Non-linearity", "Noise"], "ans": 1}},
    ],
    "Artificial Intelligence": [
        {"id": "ai1", "diff": "easy",   "title": "What is AI?",
         "body": "AI simulates human intelligence in machines for reasoning, learning, problem-solving.",
         "code": "# AI Applications:\n# Face Recognition, Voice Assistants\n# Self-driving Cars, Recommendations",
         "quiz": {"q": "AI simulates:", "opts": ["Human intelligence", "Machine speed", "Database queries"], "ans": 0}},
        {"id": "ai2", "diff": "easy",   "title": "AI vs ML vs DL",
         "body": "AI is the broad concept. ML is a subset. Deep Learning uses neural networks.",
         "code": "# AI > ML > Deep Learning\n# AI: intelligent behavior\n# ML: learns from data\n# DL: deep neural networks",
         "quiz": {"q": "Deep Learning is a subset of:", "opts": ["AI only", "ML", "Statistics"], "ans": 1}},
        {"id": "ai3", "diff": "medium", "title": "Natural Language Processing",
         "body": "NLP enables machines to understand human language. Used in chatbots, translation.",
         "code": "from transformers import pipeline\nsentiment = pipeline('sentiment-analysis')\nresult = sentiment('I love AI!')",
         "quiz": {"q": "NLP is used in:", "opts": ["Image recognition", "Chatbots", "Self-driving cars"], "ans": 1}},
        {"id": "ai4", "diff": "hard",   "title": "Reinforcement Learning",
         "body": "RL trains agents by rewarding good actions and penalizing bad ones.",
         "code": "# RL: Agent + Environment + State + Action + Reward",
         "quiz": {"q": "RL uses which signal?", "opts": ["Labels", "Reward", "Clusters"], "ans": 1}},
    ],
    "Web Development": [
        {"id": "w1", "diff": "easy",   "title": "What is HTML?",
         "body": "HTML is the structure of web pages using tags.",
         "code": "<h1>Hello World</h1>\n<p>Welcome to my page!</p>",
         "quiz": {"q": "HTML stands for:", "opts": ["HyperText Markup Language", "High Text Machine Language", "HyperTool Markup Logic"], "ans": 0}},
        {"id": "w2", "diff": "easy",   "title": "CSS Basics",
         "body": "CSS styles HTML elements with colors, fonts, and layouts.",
         "code": "body { background: #0f0f1a; color: white; }\nh1 { color: #00C851; }",
         "quiz": {"q": "CSS is used for:", "opts": ["Logic", "Styling", "Database"], "ans": 1}},
        {"id": "w3", "diff": "medium", "title": "Flask Web Framework",
         "body": "Flask is a lightweight Python web framework for building web apps and APIs.",
         "code": "from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef home():\n    return 'Hello!'",
         "quiz": {"q": "Flask is a:", "opts": ["Database", "Web framework", "ML library"], "ans": 1}},
        {"id": "w4", "diff": "hard",   "title": "REST API",
         "body": "REST API uses HTTP methods: GET, POST, PUT, DELETE.",
         "code": "@app.route('/api/data', methods=['GET','POST'])\ndef data():\n    return jsonify({'status':'ok'})",
         "quiz": {"q": "GET method is used to:", "opts": ["Delete data", "Read data", "Update data"], "ans": 1}},
    ],
    "Database & SQL": [
        {"id": "db1", "diff": "easy",   "title": "What is a Database?",
         "body": "A database is an organized collection of structured data. SQL manages relational databases.",
         "code": "CREATE TABLE students (\n    id INT PRIMARY KEY,\n    name VARCHAR(50),\n    grade FLOAT\n);",
         "quiz": {"q": "SQL stands for:", "opts": ["Structured Query Language", "Simple Query Logic", "System Query Layer"], "ans": 0}},
        {"id": "db2", "diff": "easy",   "title": "SELECT Statement",
         "body": "SELECT retrieves data from a database table.",
         "code": "SELECT * FROM students;\nSELECT name, grade FROM students\nWHERE grade > 80;",
         "quiz": {"q": "SELECT * means:", "opts": ["Select one column", "Select all columns", "Delete all"], "ans": 1}},
        {"id": "db3", "diff": "medium", "title": "INSERT & UPDATE",
         "body": "INSERT adds new rows. UPDATE modifies existing rows.",
         "code": "INSERT INTO students VALUES (1, 'Alice', 92.5);\nUPDATE students SET grade=95 WHERE name='Alice';",
         "quiz": {"q": "INSERT is used to:", "opts": ["Delete rows", "Add new rows", "Update rows"], "ans": 1}},
        {"id": "db4", "diff": "hard",   "title": "JOIN Operations",
         "body": "JOIN combines rows from multiple tables based on a related column.",
         "code": "SELECT s.name, c.course\nFROM students s\nINNER JOIN courses c ON s.id = c.student_id;",
         "quiz": {"q": "JOIN combines:", "opts": ["One table", "Multiple tables", "Columns only"], "ans": 1}},
    ],
    "Cybersecurity": [
        {"id": "cs1", "diff": "easy",   "title": "What is Cybersecurity?",
         "body": "Cybersecurity protects systems, networks, and data from digital attacks.",
         "code": "# Key areas:\n# Network Security, Encryption\n# Ethical Hacking, Data Protection",
         "quiz": {"q": "Cybersecurity protects:", "opts": ["Only hardware", "Systems and data", "Only software"], "ans": 1}},
        {"id": "cs2", "diff": "easy",   "title": "Encryption Basics",
         "body": "Encryption converts data into unreadable format. Only authorized parties can decrypt.",
         "code": "from cryptography.fernet import Fernet\nkey = Fernet.generate_key()\nf = Fernet(key)\ntoken = f.encrypt(b'Secret')",
         "quiz": {"q": "Encryption makes data:", "opts": ["Faster", "Unreadable", "Smaller"], "ans": 1}},
        {"id": "cs3", "diff": "medium", "title": "Hashing Passwords",
         "body": "Never store plain passwords. Use hashing (SHA256, bcrypt).",
         "code": "import hashlib\nhashed = hashlib.sha256('mypassword'.encode()).hexdigest()\nprint(hashed)",
         "quiz": {"q": "Hashing is:", "opts": ["Reversible", "One-way", "Encryption"], "ans": 1}},
        {"id": "cs4", "diff": "hard",   "title": "SQL Injection",
         "body": "SQL Injection inserts malicious SQL into queries. Use parameterized queries to prevent.",
         "code": "# SAFE - parameterized query:\ncursor.execute('SELECT * FROM users WHERE name=?', (name,))",
         "quiz": {"q": "SQL Injection prevented by:", "opts": ["Long passwords", "Parameterized queries", "Firewalls"], "ans": 1}},
    ],
}

# ── In-memory session store ────────────────────────────────────────────────────
sessions = {}

# =============================================================================
# DATABASE — Initialisation (Phase 3: uses get_conn from db.py)
# =============================================================================

def init_db():
    """
    Create all required tables. Supports both SQLite and PostgreSQL.
    Phase 0: user_attempts | Phase 1: users, user_stats
    Phase 2: achievements  | Phase 3: spaced_repetition, question_stats
    """
    conn = get_conn()
    try:
        cur = conn.cursor()

        # Phase 0 — quiz history
        cur.execute(adapt_sql("""
            CREATE TABLE IF NOT EXISTS user_attempts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT,
                user_id     INTEGER DEFAULT NULL,
                question_id TEXT,
                subject     TEXT,
                correct     INTEGER,
                time_taken  REAL,
                emotion     TEXT,
                difficulty  TEXT,
                timestamp   TEXT
            )
        """) if not USE_POSTGRES else """
            CREATE TABLE IF NOT EXISTS user_attempts (
                id          SERIAL PRIMARY KEY,
                session_id  TEXT,
                user_id     INTEGER DEFAULT NULL,
                question_id TEXT,
                subject     TEXT,
                correct     INTEGER,
                time_taken  REAL,
                emotion     TEXT,
                difficulty  TEXT,
                timestamp   TEXT
            )
        """)

        # Migration safety — add multimodal columns to user_attempts
        for col_name, col_def in [
            ('user_id', 'INTEGER DEFAULT NULL'),
            ('emotion_confidence', 'REAL DEFAULT 0.75'),
            ('emotion_stability', 'REAL DEFAULT 0.75'),
            ('attention_score', 'INTEGER DEFAULT 75'),
            ('cognitive_state', 'TEXT DEFAULT "engaged"'),
            ('cognitive_reasons', 'TEXT DEFAULT ""')
        ]:
            try:
                cur.execute(f"ALTER TABLE user_attempts ADD COLUMN {col_name} {col_def}")
            except Exception:
                pass

        # Phase 1 — user accounts
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            {pk},
                username      TEXT    UNIQUE NOT NULL,
                email         TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    NOT NULL
            )
        """.format(pk="SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"))

        # Phase 1 — XP / streak / stats
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_stats (
                id              {pk},
                user_id         INTEGER UNIQUE NOT NULL,
                xp              INTEGER DEFAULT 0,
                level           INTEGER DEFAULT 1,
                streak          INTEGER DEFAULT 0,
                longest_streak  INTEGER DEFAULT 0,
                last_login      TEXT    DEFAULT '',
                total_correct   INTEGER DEFAULT 0,
                total_questions INTEGER DEFAULT 0,
                correct_streak  INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """.format(pk="SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"))

        # Phase 2 — achievement badges
        cur.execute("""
            CREATE TABLE IF NOT EXISTS achievements (
                id         {pk},
                user_id    INTEGER NOT NULL,
                badge_id   TEXT    NOT NULL,
                earned_at  TEXT    NOT NULL,
                UNIQUE(user_id, badge_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """.format(pk="SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"))

        # Phase 3 — spaced repetition schedule
        cur.execute("""
            CREATE TABLE IF NOT EXISTS spaced_repetition (
                id               {pk},
                user_id          INTEGER NOT NULL,
                question_id      TEXT    NOT NULL,
                subject          TEXT    NOT NULL,
                next_review_date TEXT    NOT NULL,
                interval_days    INTEGER DEFAULT 1,
                review_count     INTEGER DEFAULT 0,
                UNIQUE(user_id, question_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """.format(pk="SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"))

        # Phase 3 — per-question analytics
        cur.execute("""
            CREATE TABLE IF NOT EXISTS question_stats (
                question_id    TEXT PRIMARY KEY,
                subject        TEXT,
                total_attempts INTEGER DEFAULT 0,
                wrong_count    INTEGER DEFAULT 0,
                avg_time       REAL    DEFAULT 0.0,
                last_updated   TEXT
            )
        """)

        # ML — prediction log (one row per question served)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id                   {pk},
                user_id              INTEGER,
                session_id           TEXT,
                question_id          TEXT,
                emotion              TEXT,
                time_taken           REAL,
                difficulty           TEXT,
                prediction           INTEGER,
                confidence           REAL,
                proba_correct        REAL,
                suggested_difficulty TEXT,
                actual_correct       INTEGER DEFAULT NULL,
                timestamp            TEXT
            )
        """.format(pk="SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"))

        conn.commit()
        init_adaptive_tables()
        db_info()
        print("[DB] All tables ready (Multimodal Phase 3+).")
    except Exception as e:
        print(f"[DB INIT ERROR] {e}")
        conn.rollback()
    finally:
        conn.close()


def log_attempt(session_id, question_id, subject, correct, time_taken, emotion, difficulty,
                user_id=None, emotion_confidence=0.75, emotion_stability=0.75,
                attention_score=75, cognitive_state='engaged', cognitive_reasons=''):
    """Save one quiz attempt to the database with multimodal derived telemetry."""
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute(adapt_sql("""
            INSERT INTO user_attempts
               (session_id, user_id, question_id, subject, correct, time_taken, emotion, difficulty, timestamp,
                emotion_confidence, emotion_stability, attention_score, cognitive_state, cognitive_reasons)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""),
            (session_id, user_id, question_id, subject,
             1 if correct else 0, round(float(time_taken), 2),
             emotion, difficulty, datetime.now().isoformat(),
             round(float(emotion_confidence), 3), round(float(emotion_stability), 3),
             int(attention_score), str(cognitive_state), str(cognitive_reasons))
        )
        conn.commit()
        conn.close()
        # Phase 3: update per-question analytics in background
        update_question_stats(question_id, subject, correct, time_taken)
    except Exception as e:
        # Fallback to simple insert if schema is in transition
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute(adapt_sql("""
                INSERT INTO user_attempts
                   (session_id, user_id, question_id, subject, correct, time_taken, emotion, difficulty, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""),
                (session_id, user_id, question_id, subject,
                 1 if correct else 0, round(float(time_taken), 2),
                 emotion, difficulty, datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
            update_question_stats(question_id, subject, correct, time_taken)
        except Exception as e2:
            print(f"[DB ERROR] log_attempt fallback failed: {e2}")


# =============================================================================
# PHASE 1 — AUTH HELPERS
# =============================================================================

def login_required(f):
    """
    Route decorator — redirects to /login if the user is not authenticated.
    Usage: @login_required above any route function.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'error')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


def validate_username(username):
    """Username must be 3-20 characters: letters, digits, underscore only."""
    return bool(re.match(r'^[a-zA-Z0-9_]{3,20}$', username))


def validate_email(email):
    """Basic email sanity check."""
    return '@' in email and '.' in email.split('@')[-1]


# =============================================================================
# PHASE 1 — XP, LEVEL & STREAK HELPERS
# =============================================================================

def calculate_level(xp):
    """
    Return the LEVEL_THRESHOLDS tuple that matches the given XP total.

    Args:
        xp (int): Total XP earned.

    Returns:
        tuple: (min_xp, level_number, level_name, emoji)
    """
    current = LEVEL_THRESHOLDS[0]
    for threshold in LEVEL_THRESHOLDS:
        if xp >= threshold[0]:
            current = threshold
        else:
            break
    return current


def get_user_profile(user_id):
    """
    Fetch a user's full profile including XP, level progress, and streak.

    Args:
        user_id (int): The user's DB id.

    Returns:
        dict | None: Profile data ready for templates, or None if not found.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            user  = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
            stats = conn.execute('SELECT * FROM user_stats WHERE user_id = ?', (user_id,)).fetchone()

        if not user or not stats:
            return None

        xp         = stats['xp']
        level_info = calculate_level(xp)
        level_idx  = LEVEL_THRESHOLDS.index(level_info)

        # XP progress towards the NEXT level
        if level_idx < len(LEVEL_THRESHOLDS) - 1:
            current_xp  = level_info[0]
            next_xp     = LEVEL_THRESHOLDS[level_idx + 1][0]
            xp_in_level = xp - current_xp
            xp_needed   = next_xp - current_xp
            xp_percent  = round(min(100.0, xp_in_level / xp_needed * 100), 1)
            xp_to_next  = next_xp - xp
        else:
            xp_percent = 100.0
            xp_to_next = 0

        return {
            'id':              user['id'],
            'username':        user['username'],
            'email':           user['email'],
            'created_at':      user['created_at'],
            'xp':              xp,
            'level':           level_info[1],
            'level_name':      level_info[2],
            'level_emoji':     level_info[3],
            'xp_percent':      xp_percent,
            'xp_to_next':      xp_to_next,
            'streak':          stats['streak'],
            'longest_streak':  stats['longest_streak'],
            'total_correct':   stats['total_correct'],
            'total_questions': stats['total_questions'],
            'correct_streak':  stats['correct_streak'],
        }
    except Exception as e:
        print(f"[PROFILE ERROR] {e}")
        return None


def get_safe_profile(user_id: int) -> dict:
    """
    Like get_user_profile() but ALWAYS returns a dict (never None).
    Auto-creates a user_stats row for brand-new users and falls back to
    safe defaults when the DB query fails — prevents Jinja UndefinedError.
    """
    # Ensure user_stats row exists
    try:
        with sqlite3.connect(DB_PATH) as _c:
            if not _c.execute('SELECT id FROM user_stats WHERE user_id=?', (user_id,)).fetchone():
                _c.execute(
                    'INSERT OR IGNORE INTO user_stats '
                    '(user_id,xp,level,streak,longest_streak,last_login,'
                    'total_correct,total_questions,correct_streak) '
                    'VALUES (?,0,1,0,0,"",0,0,0)', (user_id,)
                )
                _c.commit()
    except Exception:
        pass

    profile = get_user_profile(user_id)
    if profile is not None:
        return profile

    # Absolute fallback
    try:
        with sqlite3.connect(DB_PATH) as _c:
            _c.row_factory = sqlite3.Row
            u = _c.execute('SELECT username,email,created_at FROM users WHERE id=?', (user_id,)).fetchone()
        uname = u['username'] if u else 'User'
        email = u['email']    if u else ''
        cat   = u['created_at'] if u else ''
    except Exception:
        uname, email, cat = 'User', '', ''

    return {
        'id': user_id, 'username': uname, 'email': email, 'created_at': cat,
        'xp': 0, 'level': 1, 'level_name': 'Beginner', 'level_emoji': '🌱',
        'xp_percent': 0.0, 'xp_to_next': 100, 'streak': 0,
        'longest_streak': 0, 'total_correct': 0, 'total_questions': 0,
        'correct_streak': 0,
    }


def award_xp(user_id, correct):
    """
    Award XP to a user after a quiz answer and update their correct streak.

    XP breakdown:
        +2  for attempting
        +10 for a correct answer
        +25 bonus every 5 consecutive correct answers

    Args:
        user_id (int): The user to update.
        correct (bool): Whether the answer was correct.

    Returns:
        dict: {xp_earned, total_xp, level, level_name, level_emoji,
               xp_percent, streak_bonus, correct_streak}
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            stats = conn.execute(
                'SELECT xp, correct_streak FROM user_stats WHERE user_id = ?', (user_id,)
            ).fetchone()

            if not stats:
                return {}

            old_streak     = stats['correct_streak']
            new_streak     = (old_streak + 1) if correct else 0
            xp_earned      = XP_PER_ATTEMPT + (XP_PER_CORRECT if correct else 0)
            streak_bonus   = 0

            # Every 5 consecutive correct answers earns a bonus
            if correct and new_streak > 0 and new_streak % 5 == 0:
                xp_earned    += XP_STREAK_BONUS
                streak_bonus  = XP_STREAK_BONUS

            new_total_xp = stats['xp'] + xp_earned

            conn.execute("""
                UPDATE user_stats
                SET xp              = xp + ?,
                    correct_streak  = ?,
                    total_correct   = total_correct + ?,
                    total_questions = total_questions + 1
                WHERE user_id = ?
            """, (xp_earned, new_streak, 1 if correct else 0, user_id))
            conn.commit()

        level_info = calculate_level(new_total_xp)
        level_idx  = LEVEL_THRESHOLDS.index(level_info)

        if level_idx < len(LEVEL_THRESHOLDS) - 1:
            current_xp  = level_info[0]
            next_xp     = LEVEL_THRESHOLDS[level_idx + 1][0]
            xp_in_level = new_total_xp - current_xp
            xp_needed   = next_xp - current_xp
            xp_percent  = round(min(100.0, xp_in_level / xp_needed * 100), 1)
        else:
            xp_percent = 100.0

        # Check achievements after awarding XP
        new_badges = check_achievements(user_id, {
            'correct': correct,
            'time_taken': 0,  # passed separately when called from quiz
            'total_xp': new_total_xp,
            'level': level_info[1],
            'correct_streak': new_streak,
        })

        return {
            'xp_earned':    xp_earned,
            'total_xp':     new_total_xp,
            'level':        level_info[1],
            'level_name':   level_info[2],
            'level_emoji':  level_info[3],
            'xp_percent':   xp_percent,
            'streak_bonus': streak_bonus,
            'correct_streak': new_streak,
            'new_badges':   new_badges,
        }
    except Exception as e:
        print(f"[XP ERROR] {e}")
        return {}


# =============================================================================
# PHASE 2 — ACHIEVEMENT SYSTEM
# =============================================================================

def check_achievements(user_id, context):
    """
    Check and award any newly earned badges for this user.

    Args:
        user_id (int): The user to check.
        context (dict): Keys: correct, time_taken, total_xp, level,
                        correct_streak, streak (login streak).

    Returns:
        list[dict]: Newly awarded badges [{badge_id, emoji, name, description}]
    """
    new_badges = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            stats = conn.execute(
                'SELECT * FROM user_stats WHERE user_id = ?', (user_id,)
            ).fetchone()
            existing = {r['badge_id'] for r in conn.execute(
                'SELECT badge_id FROM achievements WHERE user_id = ?', (user_id,)
            ).fetchall()}

            if not stats:
                return []

            total_q   = stats['total_questions']
            total_c   = stats['total_correct']
            streak    = stats['streak']
            level     = context.get('level', 1)
            correct   = context.get('correct', False)
            time_t    = context.get('time_taken', 99)

            # Define badge conditions
            candidates = []

            if total_q >= 1:
                candidates.append('first_step')
            if streak >= 5:
                candidates.append('on_fire')
            if streak >= 10:
                candidates.append('streak_master')
            if level >= 5:
                candidates.append('scholar')
            if level >= 6:
                candidates.append('expert_level')
            if total_q >= 50:
                candidates.append('brain_power')
            if total_q >= 100:
                candidates.append('century')
            if correct and time_t > 0 and time_t <= 5:
                candidates.append('speed_demon')
            # Perfect score: >=5 questions answered with 100% accuracy
            if total_q >= 5 and total_c == total_q:
                candidates.append('perfect_score')

            # Check leaderboard #1
            top = conn.execute(
                'SELECT user_id FROM user_stats ORDER BY xp DESC LIMIT 1'
            ).fetchone()
            if top and top['user_id'] == user_id:
                candidates.append('top_learner')

            badge_map = {b[0]: b for b in ALL_BADGES}
            for badge_id in candidates:
                if badge_id not in existing and badge_id in badge_map:
                    conn.execute(
                        'INSERT OR IGNORE INTO achievements (user_id, badge_id, earned_at) VALUES (?, ?, ?)',
                        (user_id, badge_id, datetime.now().isoformat())
                    )
                    b = badge_map[badge_id]
                    new_badges.append({'badge_id': b[0], 'emoji': b[1], 'name': b[2], 'description': b[3]})
                    print(f"[BADGE] {user_id} earned: {badge_id}")
            conn.commit()
    except Exception as e:
        print(f"[ACHIEVEMENT ERROR] {e}")
    return new_badges


def update_daily_streak(user_id):
    """
    Update the user's login streak based on today's date vs their last login.

    Rules:
        - Same day       → no change
        - Consecutive day → streak + 1
        - Gap > 1 day    → streak resets to 1

    Args:
        user_id (int): The user to update.

    Returns:
        int: Updated streak count.
    """
    try:
        today_str     = date.today().isoformat()
        yesterday_str = (date.today() - timedelta(days=1)).isoformat()

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            stats = conn.execute(
                'SELECT streak, longest_streak, last_login FROM user_stats WHERE user_id = ?',
                (user_id,)
            ).fetchone()

            if not stats:
                return 1

            last_login = stats['last_login'] or ''
            streak     = stats['streak']

            if last_login == today_str:
                return streak            # Already counted today
            elif last_login == yesterday_str:
                streak += 1             # Consecutive day
            else:
                streak = 1              # Streak broken or first login

            longest = max(streak, stats['longest_streak'])

            conn.execute("""
                UPDATE user_stats
                SET streak = ?, longest_streak = ?, last_login = ?
                WHERE user_id = ?
            """, (streak, longest, today_str, user_id))
            conn.commit()

        return streak
    except Exception as e:
        print(f"[STREAK ERROR] {e}")
        return 1


# =============================================================================
# PHASE 3 — GEMINI AI QUESTION GENERATOR
# =============================================================================

def validate_question_dict(q_dict: Any) -> bool:
    """Validates that a question dictionary has the required schema and valid options."""
    if not isinstance(q_dict, dict):
        return False
    if not q_dict.get("title") or not q_dict.get("quiz"):
        return False
    quiz = q_dict["quiz"]
    if not isinstance(quiz, dict):
        return False
    if not quiz.get("q"):
        return False
    opts = quiz.get("opts")
    if not isinstance(opts, list) or len(opts) < 2:
        return False
    try:
        ans = int(quiz.get("ans"))
        if ans < 0 or ans >= len(opts):
            return False
    except (ValueError, TypeError):
        return False
    return True


def generate_question(topic, difficulty="medium", reinforcement_concepts=None, previous_ids=None):
    """
    Generate an AI-powered MCQ using Google Gemini (google.genai SDK).
    Validates structure before presentation and returns a validated question dict,
    or None on failure.
    """
    if not GEMINI_AVAILABLE or _GEMINI_CLIENT is None:
        return None
    try:
        reinforce_clause = f' Focus especially on reinforcing key concepts: "{reinforcement_concepts}".' if reinforcement_concepts else ''
        avoid_clause = f' Avoid questions similar to recently completed question IDs: {previous_ids}.' if previous_ids else ''

        prompt = f"""Create a {difficulty} multiple-choice question about "{topic}" for a programming learner.{reinforce_clause}{avoid_clause}
Return ONLY valid JSON (no markdown, no extra text) in this exact format:
{{
  "title": "Short question title",
  "body": "2-3 sentence explanation of the concept",
  "code": "optional short code example or empty string",
  "quiz": {{
    "q": "The question text",
    "opts": ["Option A", "Option B", "Option C"],
    "ans": 1,
    "explanation": "Why the correct answer is right"
  }}
}}
The ans field is the 0-based index of the correct option."""

        response = _GEMINI_CLIENT.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        text = response.text.strip()

        # Strip markdown fences if Gemini wraps in ```json
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()

        data = json.loads(text)
        candidate = {
            "id":           f"ai_{topic[:3].lower()}_{int(time.time())}",
            "diff":         difficulty,
            "title":        data.get("title", "AI Question"),
            "body":         data.get("body", ""),
            "code":         data.get("code", ""),
            "quiz": {
                "q":           data["quiz"]["q"],
                "opts":        data["quiz"]["opts"],
                "ans":         int(data["quiz"]["ans"]),
                "explanation": data["quiz"].get("explanation", ""),
            },
            "ai_generated": True,
        }

        if validate_question_dict(candidate):
            return candidate
        else:
            print(f"[GEMINI WARNING] Generated question failed schema validation: {data}")
            return None
    except Exception as e:
        print(f"[GEMINI ERROR] generate_question: {e}")
        return None



# =============================================================================
# PHASE 3 — SPACED REPETITION SYSTEM
# =============================================================================

def schedule_review(user_id, question_id, subject, was_correct):
    """
    Schedule (or update) a spaced-repetition review for a question.

    Algorithm:
      - Wrong answer → schedule review for next_review_date = today + 1 day
      - Correct on review → double the interval (1→3→7→14 days)
      - Correct on first attempt (no existing row) → do nothing

    Args:
        user_id (int):     Logged-in user.
        question_id (str): Question ID from the session.
        subject (str):     Subject name.
        was_correct (bool): Whether this attempt was correct.
    """
    try:
        conn = get_conn()
        cur  = conn.cursor()
        today = date.today().isoformat()

        # Check if a review row already exists for this question
        cur.execute(adapt_sql(
            "SELECT interval_days, review_count FROM spaced_repetition "
            "WHERE user_id = ? AND question_id = ?"
        ), (user_id, question_id))
        row = cur.fetchone()

        if was_correct:
            if row:
                # Correct on a review — double the interval, cap at 30 days
                new_interval = min(int(row["interval_days"]) * 2, 30)
                next_date    = (date.today() + timedelta(days=new_interval)).isoformat()
                cur.execute(adapt_sql("""
                    UPDATE spaced_repetition
                    SET interval_days = ?, next_review_date = ?, review_count = review_count + 1
                    WHERE user_id = ? AND question_id = ?
                """), (new_interval, next_date, user_id, question_id))
            # else: correct on first try — no review needed
        else:
            # Wrong answer — (re)schedule with 1-day interval
            if row:
                cur.execute(adapt_sql("""
                    UPDATE spaced_repetition
                    SET interval_days = 1, next_review_date = ?, review_count = review_count + 1
                    WHERE user_id = ? AND question_id = ?
                """), ((date.today() + timedelta(days=1)).isoformat(), user_id, question_id))
            else:
                cur.execute(adapt_sql("""
                    INSERT INTO spaced_repetition
                        (user_id, question_id, subject, next_review_date, interval_days, review_count)
                    VALUES (?, ?, ?, ?, 1, 0)
                """), (user_id, question_id, subject,
                       (date.today() + timedelta(days=1)).isoformat()))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[SPACED REP ERROR] schedule_review: {e}")


def get_due_reviews(user_id):
    """
    Return a list of question IDs due for review today (or overdue).

    Args:
        user_id (int): The logged-in user.

    Returns:
        list[dict]: Each item: {question_id, subject}
    """
    try:
        today = date.today().isoformat()
        conn  = get_conn()
        cur   = conn.cursor()
        cur.execute(adapt_sql("""
            SELECT question_id, subject FROM spaced_repetition
            WHERE user_id = ? AND next_review_date <= ?
            ORDER BY next_review_date ASC
            LIMIT 5
        """), (user_id, today))
        rows = cur.fetchall()
        conn.close()
        return [{"question_id": r["question_id"], "subject": r["subject"]} for r in rows]
    except Exception as e:
        print(f"[SPACED REP ERROR] get_due_reviews: {e}")
        return []


# =============================================================================
# PHASE 3 — PER-QUESTION ANALYTICS
# =============================================================================

def update_question_stats(question_id, subject, correct, time_taken):
    """
    Upsert per-question analytics into the question_stats table.
    Tracks total attempts, wrong count, and rolling average time.

    Args:
        question_id (str): The question identifier.
        subject (str):     Subject category.
        correct (bool):    Whether this attempt was correct.
        time_taken (float): Seconds taken to answer.
    """
    try:
        conn = get_conn()
        cur  = conn.cursor()
        now  = datetime.now().isoformat()

        # Fetch existing row
        cur.execute(adapt_sql(
            "SELECT total_attempts, avg_time FROM question_stats WHERE question_id = ?"
        ), (question_id,))
        row = cur.fetchone()

        if row:
            # Update existing stats with rolling average
            new_total   = int(row["total_attempts"]) + 1
            old_avg     = float(row["avg_time"])
            new_avg     = round(old_avg + (time_taken - old_avg) / new_total, 3)
            wrong_delta = 0 if correct else 1
            cur.execute(adapt_sql("""
                UPDATE question_stats
                SET total_attempts = ?,
                    wrong_count    = wrong_count + ?,
                    avg_time       = ?,
                    last_updated   = ?
                WHERE question_id = ?
            """), (new_total, wrong_delta, new_avg, now, question_id))
        else:
            # Insert new row
            cur.execute(adapt_sql("""
                INSERT INTO question_stats
                    (question_id, subject, total_attempts, wrong_count, avg_time, last_updated)
                VALUES (?, ?, 1, ?, ?, ?)
            """), (question_id, subject, 0 if correct else 1, round(time_taken, 3), now))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[QSTATS ERROR] update_question_stats: {e}")


# =============================================================================
# FEATURE 1 — EMOTION SMOOTHING
# =============================================================================

SMOOTHING_WINDOW = 10  # Number of recent frames to consider


def smooth_emotion(sid, raw_emotion):
    """
    Return the most common emotion from the last SMOOTHING_WINDOW frames.
    Prevents noisy frame-by-frame flickers in the detected emotion.
    """
    if sid not in sessions:
        return raw_emotion
    sessions[sid]['emotion_window'].append(raw_emotion)
    window = list(sessions[sid]['emotion_window'])
    return Counter(window).most_common(1)[0][0] if window else raw_emotion


# =============================================================================
# FEATURE 5 — RECOMMENDATION SYSTEM
# =============================================================================

def get_recommendation(session_id=None):
    """
    Query user_attempts and return a personalised recommendation string.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            if session_id:
                rows = conn.execute(
                    'SELECT * FROM user_attempts WHERE session_id = ?', (session_id,)
                ).fetchall()
            else:
                rows = conn.execute('SELECT * FROM user_attempts').fetchall()

        if not rows:
            return "Start answering questions to receive personalised recommendations!"

        total    = len(rows)
        correct  = sum(1 for r in rows if r['correct'] == 1)
        accuracy = correct / total if total > 0 else 0

        topic_errors = {}
        for r in rows:
            if r['correct'] == 0:
                subj = r['subject']
                topic_errors[subj] = topic_errors.get(subj, 0) + 1

        struggling = [t for t, cnt in topic_errors.items() if cnt >= 3]

        if struggling:
            return f"Revision needed in: {', '.join(struggling)}. Revisit the basics!"
        elif accuracy < 0.50:
            return (f"Accuracy: {round(accuracy*100)}%. "
                    "Try easier topics or re-read content before answering.")
        elif accuracy >= 0.85:
            return (f"Excellent! {round(accuracy*100)}% accuracy. "
                    "Challenge yourself with harder questions!")
        else:
            return f"Good progress! {round(accuracy*100)}% accuracy. Keep practising!"
    except Exception as e:
        print(f"[REC ERROR] {e}")
        return "Keep learning — you are doing great!"


# =============================================================================
# FEATURE 4 — INTERVENTION SYSTEM
# =============================================================================

def get_intervention(state):
    """
    Return a contextual intervention message based on the cognitive state.
    """
    interventions = {
        'bored': {
            'message': ("You seem under-challenged! Try the quiz without looking "
                        "at the example, or switch to a harder topic."),
            'type':   'challenge',
            'icon':   'rocket',
            'active': True,
        },
        'anxious': {
            'message': ("Take a breath — you've got this! Re-read the explanation "
                        "slowly. There is no time pressure here."),
            'type':   'hint',
            'icon':   'calm',
            'active': True,
        },
        'overloaded': {
            'message': ("This looks like a lot. Step away for 2 minutes, "
                        "then come back. Difficulty has been eased."),
            'type':   'break',
            'icon':   'break',
            'active': True,
        },
        'distracted': {
            'message': ("Stay focused! Close other tabs, silence notifications, "
                        "and give this lesson 5 full minutes."),
            'type':   'focus',
            'icon':   'focus',
            'active': True,
        },
    }
    return interventions.get(state, {'message': '', 'type': 'none', 'icon': '', 'active': False})


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def new_session(sid, subject):
    """Initialise an in-memory learning session for a user."""
    sessions[sid] = {
        "subject":               subject,
        "difficulty":            "easy",
        "emotion_log":           deque(maxlen=50),
        "cognitive_log":         deque(maxlen=50),
        "attention_log":         deque(maxlen=50),
        "performance":           [],
        "answered_ids":          [],
        "current_id":            None,
        "last_emotion":          None,
        "bored_count":           0,
        "question_start":        time.time(),
        "correct":               0,
        "total_q":               0,
        "timeline":              [],
        "start":                 datetime.now().isoformat(),
        "response_times":        [],
        "hesitation_count":      0,
        "error_streak":          0,
        "consecutive_wrong":     0,
        "consecutive_correct":   0,
        "effort_log":            deque(maxlen=50),
        "effort_score":          50,
        "effort_level":          "optimal",
        "last_emotions":         deque(maxlen=10),
        "emotion_window":        deque(maxlen=SMOOTHING_WINDOW),
        "current_emotion":       "neutral",
        "current_emotion_data":  {},
        "current_attention_data":{},
        "current_cognitive_data":{},
        "current_telemetry":     {},
        "current_hand_gesture":  "none",
        "hand_gesture_log":      deque(maxlen=30),
        "gaze_away_score":       0.2,
        "current_gaze":          "unknown",
        "gaze_log":              deque(maxlen=30),
        "voice_stress_score":    0.2,
        "voice_log":             deque(maxlen=30),
        # Feature 1 — attention module
        "attention_detail":      {},
        # Feature 5 — smart break tracking
        "stress_state_log":      deque(maxlen=90),   # ~3 min at 2s intervals
        "break_suggested":       False,
        # Feature 7 — difficulty progress tracking
        "difficulty_log":        [],
    }
    # Initialize session hysteresis tracker
    get_adaptation_tracker(sid, 'easy')


# =============================================================================
# QUESTION SELECTION & ADAPTIVE LEARNING
# =============================================================================

def get_question(session, cog, emotion):
    """Pick the next question based on weak topics, adaptive difficulty, and ML prediction."""
    subj   = session["subject"]
    levels = ["easy", "medium", "hard"]
    diff   = session["difficulty"]

    # Calculate recent performance telemetry
    perf_list = session.get("performance", [])
    recent_acc = (sum(perf_list[-4:]) / len(perf_list[-4:])) if perf_list else 0.70
    sess_acc   = (session["correct"] / max(session["total_q"], 1)) if session.get("total_q", 0) > 0 else 0.70
    last_resp_time = session.get("response_times") and list(session["response_times"])[-1] or 12.0

    # ML Prediction query for behavior tracking
    ml_result = predict_next_correct(
        emotion            = emotion,
        time_taken         = last_resp_time,
        difficulty         = session["difficulty"],
        emotion_confidence = session.get("current_emotion_data", {}).get("confidence", 0.75),
        emotion_stability  = session.get("current_emotion_data", {}).get("stability", 0.75),
        attention_score    = session.get("current_attention_data", {}).get("attention_score", 75),
        blink_rate         = session.get("current_attention_data", {}).get("blink_rate", 16.0),
        recent_accuracy    = recent_acc
    )
    session["last_ml_prediction"] = ml_result

    session["last_emotion"] = emotion
    pool = ALL_QUESTIONS.get(subj, [])

    # If current question is still being answered, return it
    if session["current_id"]:
        for q in pool:
            if q["id"] == session["current_id"]:
                q = q.copy()
                q["adaptation"] = {
                    "state":      cog,
                    "difficulty": diff,
                    "hint":       cog in ["overloaded", "struggling", "anxious"],
                    "break":      cog in ["overloaded", "distracted"],
                    "msg":        _adapt_msg(cog),
                    "state_info": COGNITIVE_STATES.get(cog, COGNITIVE_STATES['engaged']),
                }
                return q

    # Prioritize weak-topic questions when appropriate
    user_id = session.get("user_id")
    prioritized_pool = prioritize_question_pool(pool, user_id, subj, cog)

    answered   = set(session["answered_ids"])
    candidates = [q for q in prioritized_pool if q["diff"] == diff and q["id"] not in answered]
    if not candidates:
        candidates = [q for q in prioritized_pool if q["id"] not in answered]
    if not candidates:
        session["answered_ids"] = []
        candidates = [q for q in prioritized_pool if q["diff"] == diff] or prioritized_pool

    chosen = random.choice(candidates) if candidates else pool[0]
    session["current_id"]     = chosen["id"]
    session["question_start"] = time.time()
    chosen = chosen.copy()
    chosen["adaptation"] = {
        "state":      cog,
        "difficulty": diff,
        "hint":       cog in ["overloaded", "anxious"],
        "break":      cog in ["overloaded", "distracted"],
        "msg":        _adapt_msg(cog),
        "state_info": COGNITIVE_STATES[cog],
    }
    return chosen


def _adapt_msg(cog):
    msgs = {
        'flow':       'You are in the zone! Harder content loaded.',
        'engaged':    'Great focus! Keep going.',
        'bored':      'You seem bored! Play a quick game to refresh!',
        'overloaded': 'Too much? Easier content + take a break.',
        'anxious':    'Relax! Easier content loaded with hints.',
        'distracted': 'Stay focused! A quick activity is ready.',
    }
    return msgs.get(cog, 'Keep learning!')


# =============================================================================
# EMOTION & ATTENTION HELPERS (Phase 4 Upgraded)
# =============================================================================

def analyze_emotion(frame_b64):
    """Wrapper forwarding to the modern infer_emotion engine."""
    return infer_emotion(frame_b64=frame_b64, session_id='default')


def fake_emotion():
    """Return neutral fallback dict when no frame is available."""
    return infer_emotion(frame_b64='', session_id='default')


def attention_score(img):
    """Score 0-1 based on modern composite attention engine."""
    res = detect_attention(img=img, session_id='default')
    return round(float(res['attention_score']) / 100.0, 3)


def get_cognitive(emotion, attention, perf, bored_count, looking_away: bool = False):
    """Wrapper calling evaluate_cognitive_state."""
    emo_data = {'emotion': emotion, 'confidence': 0.75, 'stability': 0.75, 'face_detected': True, 'is_uncertain': False}
    att_data = {'attention_score': int(attention * 100) if attention <= 1.0 else int(attention),
                'looking_away': looking_away, 'eye_state': 'open', 'head_pose': {'looking_forward': not looking_away}}
    perf_data = {'recent_accuracy': (sum(perf[-3:]) / len(perf[-3:])) if perf else 0.70, 'total_questions': len(perf)}
    res = evaluate_cognitive_state(emo_data, att_data, perf_data)
    return res['state']


# =============================================================================
# FEATURE 5 — SMART BREAK RECOMMENDATION
# =============================================================================

# Number of consecutive stress frames (at 2s each) before suggesting a break
_BREAK_TRIGGER_FRAMES = 45   # ~90 seconds of sustained stress


def check_break_needed(s: dict) -> dict:
    """
    Track sustained overloaded/anxious states and recommend a break when
    the student has been in a stress state for ~3 continuous minutes.

    Args:
        s: In-memory session dict.

    Returns:
        dict: {break_suggestion: bool, message: str, duration_secs: int}
    """
    cog_log = list(s.get('stress_state_log', []))
    if not cog_log:
        return {'break_suggestion': False, 'message': '', 'duration_secs': 0}

    # Count trailing consecutive stress frames
    stress_states = {'overloaded', 'anxious'}
    consecutive   = 0
    for state in reversed(cog_log):
        if state in stress_states:
            consecutive += 1
        else:
            break

    duration_secs = consecutive * 2   # 2s per frame

    if consecutive >= _BREAK_TRIGGER_FRAMES and not s.get('break_suggested', False):
        s['break_suggested'] = True
        return {
            'break_suggestion': True,
            'message':          'Take a 2-minute break ☕ You’ve been under stress for a while. '  
                                'Step away, breathe, and come back refreshed.',
            'duration_secs':    duration_secs,
        }

    # Reset flag once student leaves stress state
    if consecutive == 0:
        s['break_suggested'] = False

    return {'break_suggestion': False, 'message': '', 'duration_secs': duration_secs}


def get_engagement(s):
    """Calculate 0-100 engagement score from attention, accuracy and emotion."""
    al = list(s['attention_log'])
    if not al:
        return 50
    # att is already 0-100; normalize to 0-1 before combining
    att  = (sum(al[-10:]) / min(len(al), 10)) / 100.0
    perf = s['correct'] / max(s['total_q'], 1)
    el   = list(s['emotion_log'])[-5:]
    pos  = sum(1 for e in el if e in ['happy', 'surprise', 'neutral']) / max(len(el), 1)
    return round(min(100, (att * 0.5 + perf * 0.3 + pos * 0.2) * 100), 1)


# =============================================================================
# PHASE 1 — AUTH ROUTES
# =============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """
    GET  → Show the login form.
    POST → Validate credentials, set session, redirect to main app.
    """
    # Already logged in — go straight to the app
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Please enter your username and password.', 'error')
            return redirect(url_for('login_page'))

        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                user = conn.execute(
                    'SELECT * FROM users WHERE username = ?', (username,)
                ).fetchone()

            if user and check_password_hash(user['password_hash'], password):
                session['user_id']  = user['id']
                session['username'] = user['username']
                # Update daily login streak
                update_daily_streak(user['id'])
                return redirect(url_for('index'))
            else:
                flash('Incorrect username or password. Please try again.', 'error')
        except Exception as e:
            print(f"[LOGIN ERROR] {e}")
            flash('Something went wrong. Please try again.', 'error')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    """
    GET  → Show the registration form.
    POST → Validate input, create user, redirect to login.
    """
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username  = request.form.get('username', '').strip()
        email     = request.form.get('email', '').strip().lower()
        password  = request.form.get('password', '')
        confirm   = request.form.get('confirm_password', '')

        # ── Validate ─────────────────────────────────────────────────────────
        errors = []
        if not validate_username(username):
            errors.append('Username must be 3-20 characters (letters, digits, underscore).')
        if not validate_email(email):
            errors.append('Please enter a valid email address.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm:
            errors.append('Passwords do not match.')

        if errors:
            for err in errors:
                flash(err, 'error')
            return redirect(url_for('register_page'))

        # ── Create account ────────────────────────────────────────────────────
        try:
            password_hash = generate_password_hash(password)
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.execute(
                    'INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)',
                    (username, email, password_hash, datetime.now().isoformat())
                )
                new_user_id = cursor.lastrowid
                # Create initial stats row for this user
                conn.execute(
                    """INSERT INTO user_stats
                       (user_id, xp, level, streak, longest_streak, last_login,
                        total_correct, total_questions, correct_streak)
                       VALUES (?, 0, 1, 0, 0, ?, 0, 0, 0)""",
                    (new_user_id, date.today().isoformat())
                )
                conn.commit()

            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login_page'))

        except sqlite3.IntegrityError:
            flash('Username or email already taken. Please choose another.', 'error')
        except Exception as e:
            print(f"[REGISTER ERROR] {e}")
            flash('Registration failed. Please try again.', 'error')

    return render_template('register.html')


@app.route('/logout')
def logout():
    """Clear the session and redirect to login."""
    sid = session.get('session_id', 'demo')
    clear_session_tracker(sid)
    clear_attention_session(sid)
    clear_adaptation_tracker(sid)
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login_page'))


# =============================================================================
# MAIN ROUTES
# =============================================================================

@app.route('/')
@login_required
def index():
    """Main learning interface — requires login. Passes user profile to template."""
    profile = get_safe_profile(session['user_id'])
    return render_template('index.html', user=profile)


@app.route('/dashboard')
@login_required
def dashboard():
    """Progress dashboard — requires login. Passes user profile to template."""
    profile = get_safe_profile(session['user_id'])
    return render_template('dashboard.html', user=profile)


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/api/session/start', methods=['POST'])
def start():
    """Start or restart a learning session."""
    d   = request.json or {}
    sid = d.get('session_id', f"s{int(time.time())}")
    new_session(sid, d.get('subject', 'Python Basics'))
    return jsonify({"session_id": sid, "status": "started"})


@app.route('/api/session/calibrate', methods=['POST'])
def calibrate_route():
    """Calibrate user baseline resting head pose and eye openness."""
    d = request.json or {}
    sid = d.get('session_id', 'demo')
    samples = d.get('samples', [])
    res = calibrate_session(sid, samples)
    return jsonify(res)


@app.route('/api/debug/telemetry/<sid>')
def debug_telemetry_route(sid):
    """Returns real-time raw vision, landmark, and cognitive telemetry for debug mode."""
    s = sessions.get(sid, {})
    return jsonify(s.get('current_telemetry', {}))


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Multimodal Cognitive Analysis Endpoint — called every ~2 seconds.
    1. Camera Quality & Lighting Pre-check
    2. Face Detection, Eye-Angle Alignment & ONNX Emotion Inference
    3. Landmark-based EAR Blink Tracking & solvePnP 3D Head Pose
    4. Multimodal Cognitive-State Engine with Explainability
    5. Adaptive Learning Hysteresis & Weak Topic Routing
    """
    d = request.json or {}
    t_req_start = time.perf_counter()
    sid = d.get('session_id', 'demo')
    if sid not in sessions:
        new_session(sid, d.get('subject', 'Python Basics'))
    s = sessions[sid]

    frame_b64 = d.get('frame', '')
    t0 = time.perf_counter()
    img = decode_base64_frame(frame_b64) if frame_b64 else None
    t_decode_ms = round((time.perf_counter() - t0) * 1000, 2)

    # 1. Emotion Inference with Temporal Fusion
    er = infer_emotion(frame_b64=frame_b64, img=img, session_id=sid)
    s['current_emotion'] = er['emotion']
    s['current_emotion_data'] = er
    s['emotion_log'].append(er['emotion'])

    # 2. Attention & 3D Head Pose
    t0 = time.perf_counter()
    attention_detail = detect_attention(frame_b64=frame_b64, img=img, session_id=sid, primary_face=er.get('primary_face'))
    t_att_ms = round((time.perf_counter() - t0) * 1000, 2)
    s['current_attention_data'] = attention_detail
    s['attention_detail'] = attention_detail
    s['attention_log'].append(attention_detail['attention_score'])

    # 3. Camera Quality Assessment
    face_rect = er.get('primary_face', {}).get('bbox') if er.get('primary_face') else None
    quality = assess_camera_quality(img, face_rect)

    # 4. Student Performance Context
    perf_list = s.get('performance', [])
    recent_acc = (sum(perf_list[-4:]) / len(perf_list[-4:])) if perf_list else 0.70
    sess_acc = (s['correct'] / max(s['total_q'], 1)) if s.get('total_q', 0) > 0 else 0.70
    last_resp_time = s.get("response_times") and list(s["response_times"])[-1] or 0.0

    perf_context = {
        'recent_accuracy': recent_acc,
        'session_accuracy': sess_acc,
        'consecutive_wrong': s.get('consecutive_wrong', 0),
        'consecutive_correct': s.get('consecutive_correct', 0),
        'last_response_time': last_resp_time,
        'total_questions': s.get('total_q', 0),
        'difficulty': s.get('difficulty', 'easy'),
        'is_weak_topic': False
    }

    # 5. Multimodal Cognitive State Reasoning
    t0 = time.perf_counter()
    cog_eval = evaluate_cognitive_state(
        emotion_data=er,
        attention_data=attention_detail,
        performance_data=perf_context,
        previous_state=s.get('current_cog', 'engaged')
    )
    t_cog_ms = round((time.perf_counter() - t0) * 1000, 2)
    cog = cog_eval['state']
    s['current_cog'] = cog
    s['current_cognitive_data'] = cog_eval
    s['cognitive_log'].append(cog)

    # 6. Adaptation Hysteresis & Cooldown
    tracker = get_adaptation_tracker(sid, s['difficulty'])
    new_diff, diff_changed, adapt_reason = tracker.evaluate_adaptation(
        cognitive_state=cog,
        recent_accuracy=recent_acc,
        session_accuracy=sess_acc,
        consecutive_wrong=s.get('consecutive_wrong', 0),
        consecutive_correct=s.get('consecutive_correct', 0),
        time_taken=last_resp_time
    )
    if diff_changed:
        s['difficulty'] = new_diff
        s['current_id'] = None
        s.setdefault('difficulty_log', []).append({
            'timestamp': datetime.now().isoformat(),
            'difficulty': new_diff,
            'reason': adapt_reason
        })

    # Track stress / fatigue / emotional struggle for adaptive break & game triggers
    s.setdefault('stress_state_log', deque(maxlen=90)).append(cog)
    break_info = check_break_needed(s)

    # Multi-factor stress & fatigue counter (Angry, Sad, Fear, Struggling, Overloaded, Anxious, Bored)
    is_stressed_or_fatigued = (
        er['emotion'] in ['angry', 'sad', 'fear'] or
        cog in ['overloaded', 'struggling', 'anxious', 'bored'] or
        s.get('consecutive_wrong', 0) >= 2
    )

    if er.get('face_detected', False) and is_stressed_or_fatigued:
        s['stress_count'] = s.get('stress_count', 0) + 1
    else:
        s['stress_count'] = max(0, s.get('stress_count', 0) - 1)

    # Trigger relaxing game after 3 consistent frames of stress, sadness, anger, or boredom
    show_game = (
        er.get('face_detected', False) and
        s.get('stress_count', 0) >= 3 and
        not s.get('game_played_recently', False)
    )

    focus_mode = (cog == 'distracted')

    eng = get_engagement(s)
    q = get_question(s, cog, er['emotion'])

    # Append timeline entry
    s['timeline'].append({
        "t": datetime.now().isoformat(),
        "emotion": er['emotion'],
        "confidence": er['confidence'],
        "stability": er['stability'],
        "cog": cog,
        "att": attention_detail['attention_score'],
        "eng": eng,
    })

    # Store telemetry snapshot for debug mode & teacher view
    s['current_telemetry'] = {
        'face_detected': er['face_detected'],
        'face_status': er['face_status'],
        'camera_quality': quality,
        'emotion': er,
        'attention': attention_detail,
        'cognitive': cog_eval,
        'adaptation': {
            'difficulty': s['difficulty'],
            'changed': diff_changed,
            'reason': adapt_reason
        },
        'stats': {
            'difficulty': s['difficulty'],
            'correct': s['correct'],
            'total': s['total_q'],
            'performance': round(s['correct'] / max(s['total_q'], 1) * 100, 1),
        }
    }

    # Cached Personalized recommendation (prevents redundant DB queries on every video frame)
    uid = session.get('user_id')
    now_ts = time.time()
    if uid:
        cached_rec = s.get('cached_personal_rec')
        cached_ts = s.get('cached_personal_rec_ts', 0.0)
        if not cached_rec or (now_ts - cached_ts > 30.0):
            personal_rec = get_personalized_recommendation(uid)
            s['cached_personal_rec'] = personal_rec
            s['cached_personal_rec_ts'] = now_ts
        else:
            personal_rec = cached_rec
    else:
        personal_rec = get_recommendation(sid)

    t_total_backend_ms = round((time.perf_counter() - t_req_start) * 1000, 2)
    timing_info = {
        "decode_ms": t_decode_ms,
        "face_detect_ms": er.get('timing', {}).get('face_detect_ms', 0.0),
        "onnx_inference_ms": er.get('timing', {}).get('onnx_inference_ms', 0.0),
        "emotion_total_ms": er.get('timing', {}).get('total_ms', 0.0),
        "attention_ms": t_att_ms,
        "cognitive_ms": t_cog_ms,
        "total_backend_ms": t_total_backend_ms
    }

    return jsonify({
        "emotion": er,
        "cognitive": cog_eval,
        "camera_quality": quality,
        "content": q,
        "engagement": eng,
        "show_game": show_game,
        "intervention": get_intervention(cog),
        "recommendation": personal_rec,
        "effort": compute_cognitive_effort(s),
        "ml_prediction": s.get("last_ml_prediction", {"available": False}),
        "attention": attention_detail,
        "adaptation": {
            "difficulty": s['difficulty'],
            "changed": diff_changed,
            "reason": adapt_reason
        },
        "adaptive_decision": s.get('last_adaptive_decision', {
            "action": "maintain_difficulty",
            "topic": s.get('subject', 'Python Basics'),
            "current_difficulty": s['difficulty'],
            "recommended_difficulty": s['difficulty'],
            "reason": adapt_reason,
            "support_type": "none",
            "confidence": "medium"
        }),
        "break_info": break_info,
        "focus_mode": focus_mode,
        "timing": timing_info,
        "stats": {
            "difficulty": s['difficulty'],
            "correct": s['correct'],
            "total": s['total_q'],
        }
    })


@app.route('/api/quiz', methods=['POST'])
def quiz():
    """
    Record a quiz answer.
    - Idempotency check prevents duplicate attempts on client retry
    - Logs to user_attempts DB (Phase 0)
    - Awards XP to logged-in user (Phase 1)
    - Evaluates multimodal adaptive learning decision & difficulty state
    - Phase 3: schedules spaced repetition review for wrong answers
    """
    d   = request.json or {}
    sid = d.get('session_id', 'demo')
    if sid not in sessions:
        new_session(sid, 'Python Basics')
    s = sessions[sid]
    s.pop('cached_personal_rec', None)

    correct    = bool(d.get('correct', False))
    time_taken = float(d.get('time_taken', 0.0))
    qid        = s.get('current_id') or d.get('question_id') or 'unknown'
    subject    = s.get('subject') or d.get('subject') or 'Unknown'

    # Idempotency check: prevent duplicate attempts for the same question/submission
    attempt_token = d.get('attempt_token') or f"{sid}:{qid}:{d.get('chosen_answer')}:{round(time_taken,1)}:{correct}"
    s.setdefault('recent_attempt_tokens', deque(maxlen=30))
    if attempt_token in s['recent_attempt_tokens']:
        cached_dec = s.get('last_adaptive_decision', {})
        return jsonify({
            "status": "duplicate_ignored",
            "message": "Attempt already recorded",
            "duplicate": True,
            "correct": correct,
            "score": round(s['correct'] / max(s['total_q'], 1) * 100, 1),
            "total": s['total_q'],
            "correct_count": s['correct'],
            "adaptive_decision": cached_dec,
            "current_difficulty": s.get('difficulty', 'easy')
        })
    s['recent_attempt_tokens'].append(attempt_token)

    s['total_q'] += 1
    if correct:
        s['correct'] += 1
        s['consecutive_correct'] = s.get('consecutive_correct', 0) + 1
        s['consecutive_wrong'] = 0
    else:
        s['consecutive_wrong'] = s.get('consecutive_wrong', 0) + 1
        s['consecutive_correct'] = 0

    s['performance'].append(1.0 if correct else 0.0)
    tracker = get_adaptation_tracker(sid, s.get('difficulty', 'easy'))
    tracker.record_question_answered()

    user_id = session.get('user_id')

    # Log attempt to DB with rich multimodal telemetry
    emo_data = s.get('current_emotion_data', {})
    att_data = s.get('current_attention_data', {})
    cog_data = s.get('current_cognitive_data', {})

    log_attempt(
        session_id         = sid,
        question_id        = qid,
        subject            = subject,
        correct            = correct,
        time_taken         = time_taken,
        emotion            = s.get('current_emotion', 'neutral'),
        difficulty         = s.get('difficulty', 'easy'),
        user_id            = user_id,
        emotion_confidence = emo_data.get('confidence', 0.75),
        emotion_stability  = emo_data.get('stability', 0.75),
        attention_score    = att_data.get('attention_score', 75),
        cognitive_state    = s.get('current_cog', 'engaged'),
        cognitive_reasons  = '; '.join(cog_data.get('reasons', []))
    )

    if s['current_id'] and s['current_id'] not in s['answered_ids']:
        s['answered_ids'].append(s['current_id'])
    s['current_id']     = None
    s['question_start'] = time.time()

    # ── Evaluate Multimodal Adaptive Learning Decision ───────────────────────
    topic_info = get_user_topic_difficulty(user_id, subject, default_diff=s.get('difficulty', 'easy'))
    perf_list = s.get('performance', [])
    recent_acc = (sum(perf_list[-5:]) / len(perf_list[-5:])) if perf_list else (1.0 if correct else 0.0)
    due_reviews = len(get_due_reviews(user_id)) if user_id else 0

    adaptive_dec = evaluate_learning_decision(
        topic                  = subject,
        current_difficulty     = s.get('difficulty', 'easy'),
        total_topic_attempts   = topic_info.get('total_attempts', 0) + 1,
        recent_accuracy        = recent_acc,
        consecutive_correct    = s.get('consecutive_correct', 0),
        consecutive_wrong      = s.get('consecutive_wrong', 0),
        current_response_time  = time_taken,
        baseline_response_time = topic_info.get('avg_response_time', 12.0),
        emotion_data           = emo_data,
        attention_data         = att_data,
        revision_due           = (due_reviews > 0),
        recent_attempts        = [{'correct': int(x)} for x in perf_list[-5:]]
    )

    # Persist topic difficulty state and record decision for evaluation & audit
    update_user_topic_difficulty(
        user_id        = user_id,
        topic          = subject,
        new_difficulty = adaptive_dec['recommended_difficulty'],
        is_correct     = correct,
        time_taken     = time_taken
    )

    record_adaptive_decision(
        user_id          = user_id,
        session_id       = sid,
        topic            = subject,
        previous_diff    = adaptive_dec['current_difficulty'],
        recommended_diff = adaptive_dec['recommended_difficulty'],
        action           = adaptive_dec['action'],
        reason           = adaptive_dec['reason'],
        support_type     = adaptive_dec['support_type'],
        confidence       = adaptive_dec['confidence'],
        signals          = adaptive_dec.get('signals_used')
    )

    if adaptive_dec['recommended_difficulty'] != s.get('difficulty'):
        s['difficulty'] = adaptive_dec['recommended_difficulty']
        s.setdefault('difficulty_log', []).append({
            'timestamp': datetime.now().isoformat(),
            'difficulty': s['difficulty'],
            'reason': adaptive_dec['reason']
        })

    s['last_adaptive_decision'] = adaptive_dec

    # Phase 1: Award XP + check achievements
    xp_data = {}
    if user_id:
        xp_data = award_xp(user_id, correct)
        check_achievements(user_id, {
            'correct': correct,
            'time_taken': time_taken,
            'level': xp_data.get('level', 1),
        })
        # Phase 3: Schedule spaced repetition review
        schedule_review(user_id, qid, subject, correct)

        # ML: backfill actual_correct into the most recent prediction log row
        try:
            conn = get_conn()
            conn.execute(adapt_sql("""
                UPDATE ml_predictions
                SET actual_correct = ?
                WHERE session_id = ? AND question_id = ? AND actual_correct IS NULL
                ORDER BY timestamp DESC
                LIMIT 1
            """), (1 if correct else 0, sid, qid))
            conn.commit()
            conn.close()
        except Exception:
            pass

    # Phase 3: Tell frontend how many reviews are due
    due_count = len(get_due_reviews(user_id)) if user_id else 0

    # Feature 3 — AI explanation for wrong answers
    explanation_data = {}
    if not correct and d.get('question_title'):
        explanation_data = generate_explanation(
            question_title = d.get('question_title', ''),
            question_body  = d.get('question_body',  ''),
            correct_answer = d.get('correct_answer', ''),
            wrong_answer   = d.get('chosen_answer',  ''),
            subject        = subject,
        )

    response = {
        "correct":            correct,
        "score":              round(s['correct'] / s['total_q'] * 100, 1),
        "total":              s['total_q'],
        "correct_count":      s['correct'],
        "due_reviews":        due_count,
        "explanation":        explanation_data,   # Feature 3
        "adaptive_decision":  adaptive_dec,
        "current_difficulty": s.get('difficulty', 'easy'),
    }
    response.update(xp_data)
    return jsonify(response)


# ── Adaptive Learning Analytics & Audit Endpoints ────────────────────────────

@app.route('/api/adaptive/history')
@login_required
def adaptive_history_api():
    """Return historical adaptive decisions for the authenticated student."""
    uid = session['user_id']
    limit = request.args.get('limit', 50, type=int)
    history = get_adaptive_history(uid, limit=limit)
    return jsonify({'decisions': history, 'count': len(history)})


@app.route('/api/adaptive/summary')
@login_required
def adaptive_summary_api():
    """Return aggregated adaptive metrics and difficulty breakdown for the authenticated student."""
    uid = session['user_id']
    summary = get_adaptive_summary(uid)
    return jsonify(summary)


@app.route('/api/topic/difficulty')
@login_required
def topic_difficulty_api():
    """Return topic difficulty for the authenticated student."""
    uid = session['user_id']
    topic = request.args.get('topic', 'Python Basics')
    diff_data = get_user_topic_difficulty(uid, topic)
    return jsonify(diff_data)



@app.route('/api/user/stats')
@login_required
def user_stats_api():
    """Return the current logged-in user's profile as JSON (for live XP bar updates)."""
    profile = get_safe_profile(session['user_id'])
    if not profile:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(profile)


@app.route('/api/game/closed', methods=['POST'])
def game_closed():
    """Reset boredom counter after the mini-game is closed."""
    d   = request.json or {}
    sid = d.get('session_id', 'demo')
    if sid in sessions:
        sessions[sid]['bored_count']    = 0
        sessions[sid]['question_start'] = time.time()
    return jsonify({'status': 'ok'})


@app.route('/api/analytics/<sid>')
def analytics(sid):
    """Raw session analytics endpoint (unchanged)."""
    if sid not in sessions:
        return jsonify({"error": "Not found"}), 404
    s  = sessions[sid]
    el = list(s['emotion_log'])
    cl = list(s['cognitive_log'])
    return jsonify({
        "emotion_dist":   {e: el.count(e) for e in set(el)},
        "cognitive_dist": {c: cl.count(c) for c in set(cl)},
        "avg_attention":  round(sum(s['attention_log']) / max(len(s['attention_log']), 1), 3),
        "performance":    round(s['correct'] / max(s['total_q'], 1) * 100, 1),
        "timeline":       s['timeline'][-50:],
        "total_q":        s['total_q'],
        "correct":        s['correct'],
        "difficulty":     s['difficulty'],
    })


@app.route('/dashboard')
@login_required
def dashboard_page():
    """Dashboard — already defined above, this duplicate guard is a safety net."""
    profile = get_safe_profile(session['user_id'])
    return render_template('dashboard.html', user=profile)


@app.route('/api/dashboard/stats')
@login_required
def dashboard_stats():
    """
    Per-user learning analytics endpoint.
    Returns both legacy dashboard stats and upgraded topic mastery,
    trends, revision completion, before-and-after comparison, and explainable recommendations.
    """
    try:
        uid = session['user_id']
        range_filter = request.args.get('range', 'all')

        conn = get_conn()
        cur = conn.cursor()

        # Query all attempts for authenticated student
        cur.execute(adapt_sql(
            'SELECT * FROM user_attempts WHERE user_id = ? ORDER BY timestamp ASC'
        ), (uid,))
        rows = cur.fetchall()
        attempts = [dict(r) for r in rows]

        # Query spaced repetition records
        cur.execute(adapt_sql(
            'SELECT * FROM spaced_repetition WHERE user_id = ?'
        ), (uid,))
        spaced_rows = [dict(r) for r in cur.fetchall()]

        # Query skill assessment history
        cur.execute(adapt_sql(
            'SELECT * FROM skill_assessments WHERE user_id = ? ORDER BY assessed_at ASC'
        ), (uid,))
        skill_rows = [dict(r) for r in cur.fetchall()]

        conn.close()

        # Generate comprehensive learning analytics report via analytics_engine
        report = get_student_analytics_report(
            uid, attempts, spaced_rows, skill_rows, date_range=range_filter
        )

        if not attempts:
            return jsonify({
                "accuracy_over_time":   [],
                "emotion_distribution": {},
                "difficulty_progress":  [],
                "subject_stats":        {},
                "avg_time":             0,
                "total_questions":      0,
                "overall_accuracy":     0,
                "recommendation":       report['recommendations'][0]['reason'] if report['recommendations'] else get_recommendation(),
                "summary":              report['summary'],
                "analytics_summary":    report['summary'],
                "topic_mastery":        report['topic_mastery'],
                "trend_analysis":       report['trends'],
                "revision_analytics":   report['revision'],
                "before_after":         report['before_after'],
                "explainable_recommendations": report['recommendations'],
                "active_range":         range_filter
            })

        # Backward-compatible fields
        accuracy_over_time = []
        running_correct = 0
        for i, row in enumerate(attempts, 1):
            running_correct += int(row.get('correct') or 0)
            accuracy_over_time.append({
                "timestamp": str(row.get('timestamp') or '')[:16].replace('T', ' '),
                "accuracy":  round(running_correct / i * 100, 1),
            })

        emotion_distribution = {}
        for row in attempts:
            emo = row.get('emotion') or 'neutral'
            emotion_distribution[emo] = emotion_distribution.get(emo, 0) + 1

        diff_map = {'easy': 1, 'medium': 2, 'hard': 3}
        difficulty_progress = [
            {
                "timestamp":      str(row.get('timestamp') or '')[:16].replace('T', ' '),
                "difficulty_num": diff_map.get(row.get('difficulty'), 1),
                "difficulty":     row.get('difficulty') or 'easy',
            }
            for row in attempts
        ]

        subject_stats = {}
        for row in attempts:
            subj = row.get('subject') or 'Unknown'
            if subj not in subject_stats:
                subject_stats[subj] = {'total': 0, 'correct': 0}
            subject_stats[subj]['total'] += 1
            subject_stats[subj]['correct'] += int(row.get('correct') or 0)

        top_rec_text = report['recommendations'][0]['reason'] if report['recommendations'] else get_recommendation()

        return jsonify({
            # Legacy compatibility
            "accuracy_over_time":   accuracy_over_time,
            "emotion_distribution": emotion_distribution,
            "difficulty_progress":  difficulty_progress,
            "subject_stats":        subject_stats,
            "avg_time":             report['summary']['overall_avg_time'],
            "total_questions":      report['summary']['total_questions'],
            "overall_accuracy":     report['summary']['overall_accuracy'],
            "recommendation":       top_rec_text,

            # Upgraded Analytics & Learning Reports (Part 1)
            "summary":                    report['summary'],
            "analytics_summary":          report['summary'],
            "topic_mastery":              report['topic_mastery'],
            "trend_analysis":             report['trends'],
            "revision_analytics":         report['revision'],
            "before_after":               report['before_after'],
            "explainable_recommendations": report['recommendations'],
            "active_range":               range_filter
        })

    except Exception as e:
        print(f"[DASHBOARD ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/detailed')
@login_required
def analytics_detailed():
    """Dedicated endpoint for detailed learning analytics and explainable recommendations."""
    return dashboard_stats()



# =============================================================================
# FEATURE 4 — EMOTION HISTORY API
# =============================================================================

@app.route('/api/emotion-history')
@login_required
def emotion_history_api():
    """
    Return timestamped emotion + cognitive state history for the current user.
    GET /api/emotion-history?limit=100

    Reads from user_attempts (persisted) so data survives page reloads.
    Frontend can use this to plot an emotion-vs-time graph.
    """
    try:
        uid   = session['user_id']
        limit = min(int(request.args.get('limit', 100)), 500)
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT timestamp, emotion, difficulty FROM user_attempts "
                "WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (uid, limit)
            ).fetchall()

        history = []
        for r in reversed(rows):   # chronological order
            emo = r['emotion'] or 'neutral'
            history.append({
                'timestamp':     r['timestamp'][:19].replace('T', ' '),
                'emotion':       emo,
                'cognitive_state': EMOTION_TO_COGNITIVE.get(emo, 'engaged'),
            })
        return jsonify({'history': history, 'count': len(history)})
    except Exception as e:
        print(f"[EMOTION HISTORY ERROR] {e}")
        return jsonify({'history': [], 'count': 0}), 500


# =============================================================================
# FEATURE 2 — WEAK TOPICS / PERSONALIZATION ROUTES
# =============================================================================

@app.route('/api/weak-topics')
@login_required
def weak_topics_api():
    """
    Return subjects where the user's accuracy is below 60%.
    GET /api/weak-topics
    """
    uid  = session['user_id']
    weak = get_weak_topics(uid)
    rec  = get_personalized_recommendation(uid)
    return jsonify({'weak_topics': weak, 'recommendation': rec})


@app.route('/api/subject-accuracy')
@login_required
def subject_accuracy_api():
    """
    Return per-subject accuracy breakdown for the current user.
    GET /api/subject-accuracy
    """
    uid  = session['user_id']
    data = get_subject_accuracy(uid)
    # Convert to list for easier frontend consumption
    result = [
        {'subject': subj, **stats}
        for subj, stats in sorted(data.items(), key=lambda x: x[1]['accuracy'])
    ]
    return jsonify({'subjects': result})


# =============================================================================
# FEATURE 6 — LEADERBOARD IMPROVEMENTS (weekly + subject-wise)
# =============================================================================

@app.route('/api/leaderboard/weekly')
@login_required
def weekly_leaderboard_api():
    """
    Return top 10 users ranked by XP earned in the last 7 days.
    XP is approximated from correct answers * XP_PER_CORRECT in user_attempts.
    GET /api/leaderboard/weekly
    """
    try:
        uid      = session['user_id']
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT u.id, u.username,
                       COUNT(*)     AS total_q,
                       SUM(a.correct) AS correct_count
                FROM   user_attempts a
                JOIN   users u ON u.id = a.user_id
                WHERE  a.timestamp >= ?
                GROUP  BY u.id, u.username
                ORDER  BY correct_count DESC
                LIMIT  10
            """, (week_ago,)).fetchall()

        result = []
        for rank, row in enumerate(rows, 1):
            weekly_xp = int(row['correct_count'] or 0) * XP_PER_CORRECT + \
                        int(row['total_q'] or 0) * XP_PER_ATTEMPT
            result.append({
                'rank':       rank,
                'user_id':    row['id'],
                'username':   row['username'],
                'weekly_xp':  weekly_xp,
                'correct':    int(row['correct_count'] or 0),
                'total_q':    int(row['total_q'] or 0),
                'is_me':      row['id'] == uid,
            })
        return jsonify(result)
    except Exception as e:
        print(f"[WEEKLY LB ERROR] {e}")
        return jsonify([]), 500


@app.route('/api/leaderboard/subject')
@login_required
def subject_leaderboard_api():
    """
    Return top 10 users for a specific subject ranked by accuracy.
    GET /api/leaderboard/subject?subject=Python+Basics
    """
    try:
        uid     = session['user_id']
        subject = request.args.get('subject', 'Python Basics')
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT u.id, u.username,
                       COUNT(*)        AS total_q,
                       SUM(a.correct)  AS correct_count
                FROM   user_attempts a
                JOIN   users u ON u.id = a.user_id
                WHERE  a.subject = ?
                GROUP  BY u.id, u.username
                HAVING total_q >= 3
                ORDER  BY CAST(correct_count AS REAL) / total_q DESC
                LIMIT  10
            """, (subject,)).fetchall()

        result = []
        for rank, row in enumerate(rows, 1):
            total   = int(row['total_q'] or 1)
            correct = int(row['correct_count'] or 0)
            result.append({
                'rank':     rank,
                'user_id':  row['id'],
                'username': row['username'],
                'accuracy': round(correct / total * 100, 1),
                'correct':  correct,
                'total_q':  total,
                'is_me':    row['id'] == uid,
            })
        return jsonify({'subject': subject, 'leaderboard': result})
    except Exception as e:
        print(f"[SUBJECT LB ERROR] {e}")
        return jsonify({'subject': subject, 'leaderboard': []}), 500


# =============================================================================
# FEATURE 7 — REAL-TIME DIFFICULTY PROGRESS API
# =============================================================================

@app.route('/api/difficulty-progress')
@login_required
def difficulty_progress_api():
    """
    Return difficulty changes over time for the current user.
    Combines in-session live data with persisted DB history.
    GET /api/difficulty-progress?session_id=<sid>
    """
    try:
        uid = session['user_id']
        sid = request.args.get('session_id', '')

        # DB history (persisted across sessions)
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT timestamp, difficulty FROM user_attempts "
                "WHERE user_id = ? ORDER BY timestamp ASC LIMIT 200",
                (uid,)
            ).fetchall()

        diff_map = {'easy': 1, 'medium': 2, 'hard': 3}
        history  = [
            {
                'timestamp':      r['timestamp'][:19].replace('T', ' '),
                'difficulty':     r['difficulty'],
                'difficulty_num': diff_map.get(r['difficulty'], 1),
            }
            for r in rows
        ]

        # Append live in-session log if available
        if sid and sid in sessions:
            for entry in sessions[sid].get('difficulty_log', []):
                history.append({
                    'timestamp':      entry['timestamp'][:19].replace('T', ' '),
                    'difficulty':     entry['difficulty'],
                    'difficulty_num': diff_map.get(entry['difficulty'], 1),
                    'live':           True,
                })

        return jsonify({'progress': history, 'count': len(history)})
    except Exception as e:
        print(f"[DIFF PROGRESS ERROR] {e}")
        return jsonify({'progress': [], 'count': 0}), 500


# =============================================================================
# FEATURE 1 — ATTENTION DETAIL API
# =============================================================================

@app.route('/api/attention/<sid>')
def attention_api(sid):
    """
    Return the latest attention detection result for a session.
    GET /api/attention/<session_id>
    """
    if sid not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify(sessions[sid].get('attention_detail', {}))


# =============================================================================
# PHASE 2 — LEADERBOARD, PROFILE, ACHIEVEMENTS ROUTES
# =============================================================================

@app.route('/leaderboard')
@login_required
def leaderboard_page():
    """Leaderboard page — top users ranked by XP."""
    profile = get_safe_profile(session['user_id'])
    return render_template('leaderboard.html', user=profile)


@app.route('/profile')
@login_required
def profile_page():
    """User profile page — personal stats, badges, XP."""
    profile = get_safe_profile(session['user_id'])
    return render_template('profile.html', user=profile)


@app.route('/api/leaderboard')
@login_required
def api_leaderboard():
    """Return top 10 users by XP with rank info."""
    try:
        uid = session['user_id']
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT u.id, u.username, s.xp, s.level, s.streak, s.total_correct, s.total_questions
                FROM users u
                JOIN user_stats s ON u.id = s.user_id
                ORDER BY s.xp DESC
                LIMIT 10
            """).fetchall()

        result = []
        for rank, row in enumerate(rows, 1):
            level_info = calculate_level(row['xp'])
            accuracy   = round(row['total_correct'] / max(row['total_questions'], 1) * 100, 1)
            result.append({
                'rank':       rank,
                'user_id':    row['id'],
                'username':   row['username'],
                'xp':         row['xp'],
                'level':      level_info[1],
                'level_name': level_info[2],
                'level_emoji':level_info[3],
                'streak':     row['streak'],
                'accuracy':   accuracy,
                'is_me':      row['id'] == uid,
            })
        return jsonify(result)
    except Exception as e:
        print(f"[LEADERBOARD ERROR] {e}")
        return jsonify([]), 500


@app.route('/api/achievements')
@login_required
def api_achievements():
    """Return all badges — earned and locked — for the current user."""
    try:
        uid = session['user_id']
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            earned_rows = conn.execute(
                'SELECT badge_id, earned_at FROM achievements WHERE user_id = ?', (uid,)
            ).fetchall()
        earned = {r['badge_id']: r['earned_at'] for r in earned_rows}

        result = []
        for badge_id, emoji, name, description in ALL_BADGES:
            result.append({
                'badge_id':    badge_id,
                'emoji':       emoji,
                'name':        name,
                'description': description,
                'earned':      badge_id in earned,
                'earned_at':   earned.get(badge_id, None),
            })
        return jsonify(result)
    except Exception as e:
        print(f"[ACHIEVEMENTS ERROR] {e}")
        return jsonify([]), 500


# =============================================================================
# PHASE 3 — NEW ROUTES
# =============================================================================

@app.route('/generate-question')
@login_required
@limiter.limit("10 per minute")  # Rate-limit Gemini calls to avoid quota burn
def generate_question_route():
    """
    Generate an AI question via Gemini API.
    GET /generate-question?topic=Python+Basics&difficulty=medium&reinforce=loops&prev_ids=ai_pyt_1,ai_pyt_2

    Returns the question as JSON (same format as static questions).
    Falls back cleanly to a verified static question if Gemini is unavailable or rate-limited.
    """
    topic       = request.args.get('topic', 'Python Basics')
    difficulty  = request.args.get('difficulty', 'medium')
    reinforce   = request.args.get('reinforce', '')
    previous_ids = request.args.get('prev_ids', '')

    # Try Gemini first
    q = generate_question(topic, difficulty, reinforcement_concepts=reinforce, previous_ids=previous_ids)

    if q is None:
        # Safe fallback: pick from topic pool, matching topic, or foundational pool
        pool = ALL_QUESTIONS.get(topic, [])
        if not pool:
            for k, v in ALL_QUESTIONS.items():
                if k.lower() in topic.lower() or topic.lower() in k.lower():
                    pool = v
                    break
        if not pool:
            pool = ALL_QUESTIONS.get('Python Basics', [])

        if pool:
            import random as _r
            candidates = [x for x in pool if x.get('diff') == difficulty] or pool
            q = _r.choice(candidates).copy()
            q['ai_generated'] = False
            q['fallback']     = True
            q['fallback_reason'] = 'AI service temporarily unavailable; served from verified question bank.'
        else:
            return jsonify({'error': f'No questions found for topic: {topic}'}), 404

    return jsonify(q)



@app.route('/api/analytics/hardest-questions')
@login_required
def hardest_questions():
    """
    Return the top 10 hardest questions globally (sorted by wrong_rate).
    Used by the Dashboard 'Hardest Questions' widget.
    """
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT question_id, subject, total_attempts, wrong_count, avg_time
            FROM question_stats
            WHERE total_attempts >= 3
            ORDER BY CAST(wrong_count AS REAL) / total_attempts DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        conn.close()
        result = []
        for r in rows:
            wrong_rate = round(r['wrong_count'] / max(r['total_attempts'], 1) * 100, 1)
            result.append({
                'question_id':    r['question_id'],
                'subject':        r['subject'],
                'total_attempts': r['total_attempts'],
                'wrong_count':    r['wrong_count'],
                'wrong_rate':     wrong_rate,
                'avg_time':       round(r['avg_time'], 1),
            })
        return jsonify(result)
    except Exception as e:
        print(f"[HARDEST Q ERROR] {e}")
        return jsonify([]), 500


@app.route('/api/spaced-repetition/due')
@login_required
def due_reviews_api():
    """
    Return questions due for spaced-repetition review for the logged-in user.
    GET /api/spaced-repetition/due
    """
    uid  = session['user_id']
    due  = get_due_reviews(uid)
    return jsonify({'due_count': len(due), 'questions': due})


@app.route('/api/gemini/status')
@login_required
def gemini_status():
    """Quick endpoint to check if Gemini AI is configured and available."""
    return jsonify({
        'available': GEMINI_AVAILABLE,
        'message': 'Gemini AI is ready!' if GEMINI_AVAILABLE
                   else 'Set GEMINI_API_KEY in your .env file to enable AI questions.',
    })


# =============================================================================
# ML MODEL ROUTES
# =============================================================================

@app.route('/api/ml-model-info')
@login_required
def ml_model_info():
    """
    Return metadata about the loaded ML prediction model.
    GET /api/ml-model-info
    """
    return jsonify(get_model_info())


@app.route('/api/predict-next', methods=['POST'])
@login_required
def predict_next_api():
    """
    Predict whether the current user will answer the next question correctly.
    POST body: {session_id} (optional — uses session emotion/difficulty if omitted)

    Returns prediction dict + logs to ml_predictions table.
    """
    d   = request.json or {}
    sid = d.get('session_id', '')
    s   = sessions.get(sid, {})

    emotion    = s.get('current_emotion', d.get('emotion', 'neutral'))
    difficulty = s.get('difficulty',      d.get('difficulty', 'easy'))
    rt_list    = list(s.get('response_times', []))
    time_taken = rt_list[-1] if rt_list else float(d.get('time_taken', 10.0))

    result = predict_next_correct(emotion, time_taken, difficulty)

    # Log prediction to DB
    if result['available']:
        try:
            conn = get_conn()
            conn.execute(adapt_sql("""
                INSERT INTO ml_predictions
                    (user_id, session_id, question_id, emotion, time_taken,
                     difficulty, prediction, confidence, proba_correct,
                     suggested_difficulty, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """), (
                session.get('user_id'),
                sid,
                s.get('current_id', 'unknown'),
                emotion, time_taken, difficulty,
                result['prediction'],
                result['confidence'],
                result['proba_correct'],
                result['suggested_difficulty'],
                datetime.now().isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[ML LOG ERROR] {e}")

    return jsonify(result)


@app.route('/api/retrain', methods=['POST'])
@login_required
def retrain_model():
    """
    Trigger model retraining from the latest user_attempts data.
    POST /api/retrain
    Body (optional): {model_type: 'lr' | 'rf', min_samples: 10}

    Runs synchronously — for large datasets consider a background task.
    Returns new accuracy or error message.
    """
    import subprocess
    d          = request.json or {}
    model_type = d.get('model_type', 'lr')
    min_samples = str(d.get('min_samples', 10))

    if model_type not in ('lr', 'rf'):
        return jsonify({'error': 'model_type must be lr or rf'}), 400

    try:
        result = subprocess.run(
            ['python', 'train_model.py', '--model', model_type,
             '--min-samples', min_samples],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode != 0:
            return jsonify({
                'success': False,
                'error':   result.stderr or result.stdout,
            }), 500

        # Reload the freshly saved model into memory
        reloaded = reload_model()
        info     = get_model_info()
        return jsonify({
            'success':    True,
            'reloaded':   reloaded,
            'model_info': info,
            'output':     result.stdout[-800:],  # last 800 chars of training log
        })
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Training timed out (>120s)'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# COGNITIVE EFFORT MODELLING (Major Feature)
# =============================================================================
# Cognitive Effort = mental workload score (0-100) computed from:
#   - Response time relative to baseline
#   - Error streak (consecutive wrong answers)
#   - Hesitation count (long pauses before answering)
#   - Emotion-based arousal
#   - Difficulty level weight

EFFORT_WEIGHTS = {
    'response_time': 0.18,   # slow = high effort
    'error_streak':  0.15,   # repeated errors = high effort
    'hesitation':    0.12,   # pauses = high effort
    'emotion':       0.12,   # anxious/angry = high effort
    'difficulty':    0.08,   # hard questions = higher baseline
    'hand_gesture':  0.12,   # hand gesture stress signal
    'gaze_focus':    0.13,   # looking away = distracted
    'voice_stress':  0.10,   # voice pitch/volume = stress
}

EMOTION_AROUSAL = {
    'happy': 0.1, 'neutral': 0.2, 'surprise': 0.3,
    'sad': 0.4,   'disgust': 0.5, 'fear': 0.7, 'angry': 0.8,
}

DIFF_WEIGHT = {'easy': 0.2, 'medium': 0.5, 'hard': 0.8}

# Hand gesture → cognitive effort signal (0.0 = calm, 1.0 = high stress)
# Detected by MediaPipe Hands on the frontend and sent via /api/hand-gesture
HAND_GESTURE_EFFORT = {
    'open_palm':  0.1,   # relaxed, low effort
    'pointing':   0.3,   # focused, moderate effort
    'thumbs_up':  0.1,   # positive, low effort
    'thumbs_down':0.6,   # frustrated, high effort
    'fist':       0.8,   # stressed/tense, very high effort
    'fidget':     0.7,   # restless, high effort
    'none':       0.3,   # no hand detected — neutral
}


def compute_cognitive_effort(session_data):
    """
    Compute a 0-100 Cognitive Effort Score for the current session state.

    Higher score = more mental effort / cognitive load.
    Zones:
        0-30  : Low effort (bored / under-challenged)
        31-60 : Optimal effort (flow zone)
        61-80 : High effort (challenging but manageable)
        81-100: Overload (intervention needed)

    Components:
        - Response time  (how long to answer)
        - Error streak   (consecutive wrong answers)
        - Hesitation     (pauses > 20s)
        - Emotion arousal (facial expression stress level)
        - Difficulty     (current question difficulty)
        - Hand gesture   (MediaPipe hand pose stress signal)

    Args:
        session_data (dict): The in-memory session dict.

    Returns:
        dict: {score, zone, label, color, components}
    """
    s = session_data

    # 1. Response time component (normalised 0-1)
    rt_list = list(s.get('response_times', []))
    if rt_list:
        avg_rt = sum(rt_list) / len(rt_list)
        rt_score = min(1.0, max(0.0, (avg_rt - 5) / 25))
    else:
        rt_score = 0.3

    # 2. Error streak component
    err_score = min(1.0, s.get('error_streak', 0) / 5)

    # 3. Hesitation component
    hes_score = min(1.0, s.get('hesitation_count', 0) / max(s.get('total_q', 1), 1))

    # 4. Emotion arousal component
    emo_score = EMOTION_AROUSAL.get(s.get('current_emotion', 'neutral'), 0.2)

    # 5. Difficulty component
    diff_score = DIFF_WEIGHT.get(s.get('difficulty', 'easy'), 0.3)

    # 6. Hand gesture component
    current_gesture = s.get('current_hand_gesture', 'none')
    hand_score      = HAND_GESTURE_EFFORT.get(current_gesture, 0.3)

    # 7. Gaze focus component (NEW — Eye Tracking)
    # 0.0 = looking at content, 1.0 = looking away
    gaze_score = s.get('gaze_away_score', 0.2)

    # 8. Voice stress component (NEW — Microphone Analysis)
    # 0.0 = calm voice, 1.0 = stressed voice
    voice_score = s.get('voice_stress_score', 0.2)

    # Weighted sum → 0-100
    raw = (
        rt_score    * EFFORT_WEIGHTS['response_time'] +
        err_score   * EFFORT_WEIGHTS['error_streak']  +
        hes_score   * EFFORT_WEIGHTS['hesitation']    +
        emo_score   * EFFORT_WEIGHTS['emotion']       +
        diff_score  * EFFORT_WEIGHTS['difficulty']    +
        hand_score  * EFFORT_WEIGHTS['hand_gesture']  +
        gaze_score  * EFFORT_WEIGHTS['gaze_focus']    +
        voice_score * EFFORT_WEIGHTS['voice_stress']
    )
    score = round(min(100, max(0, raw * 100)), 1)

    if score <= 30:
        zone, label, color = 'low',      'Under-Challenged',   '#0099CC'
    elif score <= 60:
        zone, label, color = 'optimal',  'Optimal Load',       '#00C851'
    elif score <= 80:
        zone, label, color = 'high',     'High Cognitive Load', '#FF8800'
    else:
        zone, label, color = 'overload', 'Cognitive Overload',  '#FF4444'

    return {
        'score':  score,
        'zone':   zone,
        'label':  label,
        'color':  color,
        'hand_gesture': current_gesture,
        'components': {
            'response_time':   round(rt_score   * 100, 1),
            'error_streak':    round(err_score   * 100, 1),
            'hesitation':      round(hes_score   * 100, 1),
            'emotion_arousal': round(emo_score   * 100, 1),
            'difficulty':      round(diff_score  * 100, 1),
            'hand_gesture':    round(hand_score  * 100, 1),
            'gaze_focus':      round(gaze_score  * 100, 1),
            'voice_stress':    round(voice_score * 100, 1),
        }
    }


def update_session_effort(sid, time_taken, correct):
    """
    Update session-level effort tracking signals after each quiz answer.
    Call this from the /api/quiz route.
    """
    if sid not in sessions:
        return
    s = sessions[sid]

    rt = list(s.get('response_times', deque(maxlen=20)))
    rt.append(time_taken)
    s['response_times'] = deque(rt[-20:], maxlen=20)

    if time_taken > 20:
        s['hesitation_count'] = s.get('hesitation_count', 0) + 1

    s['error_streak'] = 0 if correct else s.get('error_streak', 0) + 1

    effort = compute_cognitive_effort(s)
    s.setdefault('effort_log', deque(maxlen=50)).append({
        't':      datetime.now().isoformat(),
        'score':  effort['score'],
        'zone':   effort['zone'],
        'gesture': s.get('current_hand_gesture', 'none'),
    })
    s['effort_score'] = effort['score']
    s['effort_zone']  = effort['zone']


def get_effort_intervention(effort_data):
    """
    Return a targeted intervention based on cognitive effort zone.
    """
    zone = effort_data.get('zone', 'optimal')
    interventions = {
        'low': {
            'message': 'You are breezing through this! Try harder questions or skip hints.',
            'action':  'increase_difficulty',
            'icon':    '🚀',
        },
        'optimal': {
            'message': 'Perfect cognitive load — you are in the learning zone!',
            'action':  'maintain',
            'icon':    '✅',
        },
        'high': {
            'message': 'High mental effort detected. Take a short pause before continuing.',
            'action':  'suggest_break',
            'icon':    '⚠️',
        },
        'overload': {
            'message': 'Cognitive overload! Switching to easier content and adding hints.',
            'action':  'reduce_difficulty',
            'icon':    '🆘',
        },
    }
    return interventions.get(zone, interventions['optimal'])


# =============================================================================
# LEARNING PATH ENGINE (Major Feature)
# =============================================================================
# Structured curriculum with prerequisites and mastery tracking.
# Each node: {id, subject, topic, prerequisites, mastery_threshold}

LEARNING_PATHS = {
    'Python Basics': [
        {'node': 'py_intro',    'title': 'Python Introduction',  'prereqs': [],              'questions': ['p1','p2','p3'], 'mastery': 0.7},
        {'node': 'py_control',  'title': 'Control Flow',         'prereqs': ['py_intro'],    'questions': ['p4','p5','p6'], 'mastery': 0.7},
        {'node': 'py_data',     'title': 'Data Structures',      'prereqs': ['py_control'],  'questions': ['p7'],          'mastery': 0.7},
        {'node': 'py_oop',      'title': 'OOP & Classes',        'prereqs': ['py_data'],     'questions': ['p8'],          'mastery': 0.8},
        {'node': 'py_advanced', 'title': 'Advanced Python',      'prereqs': ['py_oop'],      'questions': ['p9','p10'],    'mastery': 0.8},
    ],
    'Machine Learning': [
        {'node': 'ml_intro',    'title': 'ML Fundamentals',      'prereqs': [],              'questions': ['ml1','ml2'],   'mastery': 0.7},
        {'node': 'ml_models',   'title': 'ML Models',            'prereqs': ['ml_intro'],    'questions': ['ml3','ml4'],   'mastery': 0.7},
        {'node': 'ml_deep',     'title': 'Deep Learning',        'prereqs': ['ml_models'],   'questions': ['ml5'],         'mastery': 0.8},
    ],
    'Data Science': [
        {'node': 'ds_intro',    'title': 'Data Science Basics',  'prereqs': [],              'questions': ['d1','d2'],     'mastery': 0.7},
        {'node': 'ds_tools',    'title': 'NumPy & Pandas',       'prereqs': ['ds_intro'],    'questions': ['d3','d4'],     'mastery': 0.7},
        {'node': 'ds_ml',       'title': 'ML Pipeline',          'prereqs': ['ds_tools'],    'questions': ['d5'],          'mastery': 0.8},
    ],
}


def get_node_mastery(user_id, node_id, question_ids):
    """
    Calculate mastery (0.0-1.0) for a learning path node based on
    the user's recent correct answers for that node's questions.
    """
    if not user_id or not question_ids:
        return 0.0
    try:
        with sqlite3.connect(DB_PATH) as conn:
            placeholders = ','.join('?' * len(question_ids))
            rows = conn.execute(
                f"SELECT question_id, correct FROM user_attempts "
                f"WHERE user_id=? AND question_id IN ({placeholders}) "
                f"ORDER BY timestamp DESC",
                [user_id] + question_ids
            ).fetchall()
        if not rows:
            return 0.0
        # Use most recent attempt per question
        seen = {}
        for qid, correct in rows:
            if qid not in seen:
                seen[qid] = correct
        return round(sum(seen.values()) / len(question_ids), 2)
    except Exception as e:
        print(f"[PATH ERROR] {e}")
        return 0.0


def get_learning_path_status(user_id, subject):
    """
    Return the full learning path for a subject with mastery and unlock status.

    Returns:
        list[dict]: Each node with mastery, unlocked, completed fields.
    """
    path = LEARNING_PATHS.get(subject, [])
    if not path:
        return []

    node_mastery = {}
    result = []

    for node in path:
        mastery = get_node_mastery(user_id, node['node'], node['questions'])
        node_mastery[node['node']] = mastery

        # Unlocked if all prerequisites are mastered
        prereqs_met = all(
            node_mastery.get(p, 0.0) >= LEARNING_PATHS[subject][
                next(i for i, n in enumerate(path) if n['node'] == p)
            ]['mastery']
            for p in node['prereqs']
        ) if node['prereqs'] else True

        completed = mastery >= node['mastery']

        result.append({
            'node':       node['node'],
            'title':      node['title'],
            'prereqs':    node['prereqs'],
            'mastery':    mastery,
            'threshold':  node['mastery'],
            'unlocked':   prereqs_met,
            'completed':  completed,
            'questions':  node['questions'],
        })

    return result


def get_next_recommended_node(user_id, subject):
    """
    Return the next unlocked, incomplete node in the learning path.
    """
    path_status = get_learning_path_status(user_id, subject)
    for node in path_status:
        if node['unlocked'] and not node['completed']:
            return node
    # All completed — return last node
    return path_status[-1] if path_status else None


# =============================================================================
# PREDICTIVE PERFORMANCE ANALYTICS (Major Feature)
# =============================================================================
# Uses a simple logistic-regression-style model trained on session features
# to predict: struggle probability, dropout risk, and recommended action.

def predict_performance(user_id, session_data=None):
    """
    Predict student performance risk using session + historical features.

    Features used:
        - Recent accuracy (last 10 questions)
        - Average response time
        - Cognitive effort score
        - Error streak
        - Login streak
        - Current difficulty level

    Returns:
        dict: {struggle_prob, dropout_risk, predicted_grade, recommendation, confidence}
    """
    try:
        features = {}

        # Historical features from DB
        if user_id:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                recent = conn.execute(
                    "SELECT correct, time_taken, difficulty FROM user_attempts "
                    "WHERE user_id=? ORDER BY timestamp DESC LIMIT 20",
                    (user_id,)
                ).fetchall()
                stats = conn.execute(
                    "SELECT streak, total_correct, total_questions FROM user_stats WHERE user_id=?",
                    (user_id,)
                ).fetchone()

            if recent:
                features['recent_accuracy'] = sum(r['correct'] for r in recent) / len(recent)
                features['avg_time']        = sum(r['time_taken'] for r in recent) / len(recent)
                diff_map = {'easy': 1, 'medium': 2, 'hard': 3}
                features['avg_difficulty']  = sum(diff_map.get(r['difficulty'], 1) for r in recent) / len(recent)
            else:
                features['recent_accuracy'] = 0.5
                features['avg_time']        = 10.0
                features['avg_difficulty']  = 1.0

            features['streak'] = stats['streak'] if stats else 0
            features['total_q'] = stats['total_questions'] if stats else 0
        else:
            features = {'recent_accuracy': 0.5, 'avg_time': 10.0, 'avg_difficulty': 1.0, 'streak': 0, 'total_q': 0}

        # Session features
        if session_data:
            features['effort_score']  = session_data.get('effort_score', 50) / 100
            features['error_streak']  = min(session_data.get('error_streak', 0) / 5, 1.0)
            features['session_acc']   = session_data.get('correct', 0) / max(session_data.get('total_q', 1), 1)
        else:
            features['effort_score']  = 0.5
            features['error_streak']  = 0.0
            features['session_acc']   = features['recent_accuracy']

        # Logistic-regression-style struggle probability
        # Higher = more likely to struggle
        struggle_logit = (
            -2.0 * features['recent_accuracy']     +  # low accuracy → struggle
             0.05 * features['avg_time']            +  # slow responses → struggle
             1.5  * features['error_streak']        +  # error streak → struggle
             0.8  * features['effort_score']        +  # high effort → struggle
            -0.3  * min(features['streak'] / 10, 1) +  # good streak → less struggle
             0.5                                        # bias
        )
        struggle_prob = round(1 / (1 + math.exp(-struggle_logit)), 3)

        # Dropout risk (based on streak + engagement)
        dropout_logit = (
            -0.5 * min(features['streak'] / 7, 1)  +
            -1.0 * features['recent_accuracy']      +
             0.3 * features['effort_score']         +
             0.2                                     # bias
        )
        dropout_risk = round(1 / (1 + math.exp(-dropout_logit)), 3)

        # Predicted grade (A/B/C/D/F)
        acc = features['recent_accuracy']
        if acc >= 0.90:   predicted_grade = 'A'
        elif acc >= 0.75: predicted_grade = 'B'
        elif acc >= 0.60: predicted_grade = 'C'
        elif acc >= 0.45: predicted_grade = 'D'
        else:             predicted_grade = 'F'

        # Recommendation
        if struggle_prob > 0.7:
            recommendation = 'Immediate intervention needed — switch to easier content and add hints.'
        elif struggle_prob > 0.5:
            recommendation = 'Student showing signs of struggle — review previous topics.'
        elif dropout_risk > 0.6:
            recommendation = 'Engagement dropping — try a game break or switch subjects.'
        elif acc >= 0.85:
            recommendation = 'Excellent performance — increase difficulty to maintain challenge.'
        else:
            recommendation = 'On track — continue current learning path.'

        confidence = round(min(0.95, 0.5 + features['total_q'] / 100), 2)

        return {
            'struggle_probability': struggle_prob,
            'dropout_risk':         dropout_risk,
            'predicted_grade':      predicted_grade,
            'recommendation':       recommendation,
            'confidence':           confidence,
            'features':             {k: round(v, 3) if isinstance(v, float) else v for k, v in features.items()},
        }
    except Exception as e:
        print(f"[PREDICT ERROR] {e}")
        return {
            'struggle_probability': 0.5,
            'dropout_risk':         0.3,
            'predicted_grade':      'C',
            'recommendation':       'Keep learning!',
            'confidence':           0.1,
            'features':             {},
        }


# =============================================================================
# PHASE 3 — NEW ROUTES (continued)
# =============================================================================

@app.route('/api/hand-gesture', methods=['POST'])
def hand_gesture_api():
    """
    Receive hand gesture detected by MediaPipe Hands on the frontend.
    POST body: {session_id, gesture}

    Gesture values: open_palm | pointing | thumbs_up | thumbs_down | fist | fidget | none

    The gesture is stored in the session and used by compute_cognitive_effort
    as the hand_gesture component of the cognitive effort score.
    """
    d       = request.json or {}
    sid     = d.get('session_id', 'demo')
    gesture = d.get('gesture', 'none')

    if gesture not in HAND_GESTURE_EFFORT:
        gesture = 'none'

    if sid in sessions:
        sessions[sid]['current_hand_gesture'] = gesture
        # Log gesture to hand history (last 30 frames)
        sessions[sid].setdefault('hand_gesture_log', deque(maxlen=30)).append(gesture)

    # Return updated effort score immediately
    effort = compute_cognitive_effort(sessions[sid]) if sid in sessions else {}
    return jsonify({
        'gesture':      gesture,
        'effort_score': effort.get('score', 50),
        'effort_zone':  effort.get('zone', 'optimal'),
        'effort_color': effort.get('color', '#00C851'),
    })


@app.route('/api/cognitive-effort/<sid>')
def cognitive_effort_api(sid):
    """
    Return real-time cognitive effort score for a session.
    GET /api/cognitive-effort/<session_id>
    """
    if sid not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    effort = compute_cognitive_effort(sessions[sid])
    effort['intervention'] = get_effort_intervention(effort)
    effort['history'] = list(sessions[sid].get('effort_log', []))[-20:]
    effort['gesture_history'] = list(sessions[sid].get('hand_gesture_log', []))[-10:]
    return jsonify(effort)


@app.route('/api/learning-path')
@login_required
def learning_path_api():
    """
    Return the learning path status for the current user.
    GET /api/learning-path?subject=Python+Basics
    """
    subject = request.args.get('subject', 'Python Basics')
    uid     = session['user_id']
    path    = get_learning_path_status(uid, subject)
    next_node = get_next_recommended_node(uid, subject)
    return jsonify({
        'subject':   subject,
        'path':      path,
        'next_node': next_node,
        'available_subjects': list(LEARNING_PATHS.keys()),
    })


@app.route('/api/predict')
@login_required
def predict_api():
    """
    Return performance prediction for the current user.
    GET /api/predict?session_id=<sid>
    """
    uid = session['user_id']
    sid = request.args.get('session_id')
    s   = sessions.get(sid) if sid else None
    prediction = predict_performance(uid, s)
    return jsonify(prediction)


@app.route('/api/report')
@login_required
def session_report_api():
    """
    Generate a comprehensive session/progress report for the current user.
    Returns JSON suitable for display or PDF export.
    """
    uid = session['user_id']
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            attempts = conn.execute(
                "SELECT * FROM user_attempts WHERE user_id=? ORDER BY timestamp DESC LIMIT 100",
                (uid,)
            ).fetchall()
            stats = conn.execute(
                "SELECT * FROM user_stats WHERE user_id=?", (uid,)
            ).fetchone()
            user = conn.execute(
                "SELECT username, email, created_at FROM users WHERE id=?", (uid,)
            ).fetchone()
            badges = conn.execute(
                "SELECT badge_id, earned_at FROM achievements WHERE user_id=?", (uid,)
            ).fetchall()

        if not attempts:
            return jsonify({'error': 'No data yet'}), 404

        total   = len(attempts)
        correct = sum(1 for a in attempts if a['correct'] == 1)
        times   = [a['time_taken'] for a in attempts if a['time_taken'] > 0]

        subject_breakdown = {}
        for a in attempts:
            s = a['subject'] or 'Unknown'
            subject_breakdown.setdefault(s, {'total': 0, 'correct': 0})
            subject_breakdown[s]['total']   += 1
            subject_breakdown[s]['correct'] += a['correct']

        for s in subject_breakdown:
            d = subject_breakdown[s]
            d['accuracy'] = round(d['correct'] / d['total'] * 100, 1) if d['total'] else 0

        emotion_dist = {}
        for a in attempts:
            e = a['emotion'] or 'neutral'
            emotion_dist[e] = emotion_dist.get(e, 0) + 1

        prediction = predict_performance(uid)

        return jsonify({
            'user': {
                'username':   user['username'],
                'email':      user['email'],
                'joined':     user['created_at'][:10],
                'xp':         stats['xp'] if stats else 0,
                'level':      stats['level'] if stats else 1,
                'streak':     stats['streak'] if stats else 0,
            },
            'summary': {
                'total_questions': total,
                'correct':         correct,
                'accuracy':        round(correct / total * 100, 1),
                'avg_time':        round(sum(times) / len(times), 1) if times else 0,
                'badges_earned':   len(badges),
            },
            'subject_breakdown': subject_breakdown,
            'emotion_distribution': emotion_dist,
            'prediction':          prediction,
            'generated_at':        datetime.now().isoformat(),
        })
    except Exception as e:
        print(f"[REPORT ERROR] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/cognitive-effort')
@login_required
def cognitive_effort_page():
    """Cognitive Effort Modelling page."""
    profile = get_safe_profile(session['user_id'])
    return render_template('cognitive_effort.html', user=profile)


@app.route('/learning-path')
@login_required
def learning_path_page():
    """Learning Path page."""
    profile = get_safe_profile(session['user_id'])
    return render_template('learning_path.html', user=profile)


# =============================================================================
# FEATURE: VOICE STRESS DETECTION — Microphone Analysis API
# =============================================================================

@app.route('/api/voice-stress', methods=['POST'])
def voice_stress_api():
    """
    Receive voice stress features from Web Audio API on the frontend.
    POST body: {session_id, rms_energy, pitch_estimate, speech_rate, silence_ratio}

    rms_energy    : 0.0-1.0 (volume level)
    pitch_estimate: 0.0-1.0 (normalised pitch, high = stressed)
    speech_rate   : 0.0-1.0 (fast = anxious)
    silence_ratio : 0.0-1.0 (long silences = hesitant)

    Computes voice_stress_score and stores in session.
    """
    d             = request.json or {}
    sid           = d.get('session_id', 'demo')
    rms_energy    = float(d.get('rms_energy',    0.3))
    pitch_est     = float(d.get('pitch_estimate', 0.3))
    speech_rate   = float(d.get('speech_rate',   0.3))
    silence_ratio = float(d.get('silence_ratio', 0.3))

    # Weighted voice stress score
    # High pitch + fast speech + low volume + long silence = stressed
    voice_stress = round(
        pitch_est     * 0.35 +
        speech_rate   * 0.25 +
        (1 - rms_energy) * 0.20 +   # low volume = nervous
        silence_ratio * 0.20,
        3
    )
    voice_stress = min(1.0, max(0.0, voice_stress))

    # Classify stress level
    if voice_stress < 0.3:
        stress_label = 'Calm'
        stress_color = '#00C851'
    elif voice_stress < 0.6:
        stress_label = 'Mild Stress'
        stress_color = '#FF8800'
    else:
        stress_label = 'High Stress'
        stress_color = '#FF4444'

    if sid in sessions:
        sessions[sid]['voice_stress_score'] = voice_stress
        sessions[sid].setdefault('voice_log', deque(maxlen=30)).append({
            't':     datetime.now().isoformat(),
            'score': round(voice_stress * 100, 1),
            'label': stress_label,
        })

    effort = compute_cognitive_effort(sessions[sid]) if sid in sessions else {}
    return jsonify({
        'voice_stress':  round(voice_stress * 100, 1),
        'stress_label':  stress_label,
        'stress_color':  stress_color,
        'effort_score':  effort.get('score', 50),
        'effort_zone':   effort.get('zone', 'optimal'),
        'components': {
            'pitch':    round(pitch_est     * 100, 1),
            'speed':    round(speech_rate   * 100, 1),
            'volume':   round(rms_energy    * 100, 1),
            'silence':  round(silence_ratio * 100, 1),
        }
    })


# =============================================================================
# FEATURE: TEACHER DASHBOARD — Live Class Monitoring
# =============================================================================

# In-memory teacher session store: {teacher_id: {class_code, students:[]}}
teacher_classes = {}


@app.route('/teacher')
@login_required
def teacher_dashboard():
    """Teacher live class monitoring dashboard."""
    profile = get_safe_profile(session['user_id'])
    return render_template('teacher.html', user=profile)


@app.route('/api/teacher/class-status')
@login_required
def teacher_class_status():
    """
    Return live status of all active student sessions.
    Shows emotion, cognitive state, effort score, performance for each student.
    Used by the teacher dashboard to monitor the whole class in real-time.
    """
    try:
        uid = session['user_id']
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            users = conn.execute(
                "SELECT id, username FROM users WHERE id != ?", (uid,)
            ).fetchall()

        students = []
        for u in users:
            # Find active session for this user
            user_session = None
            for sid, s in sessions.items():
                # Match session to user via recent DB attempt
                pass

            # Get latest stats from DB
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                stats = conn.execute(
                    "SELECT * FROM user_stats WHERE user_id=?", (u['id'],)
                ).fetchone()
                recent = conn.execute(
                    "SELECT emotion, correct, difficulty, timestamp FROM user_attempts "
                    "WHERE user_id=? ORDER BY timestamp DESC LIMIT 5",
                    (u['id'],)
                ).fetchall()

            if not stats:
                continue

            # Determine current emotion from most recent attempt
            last_emotion = recent[0]['emotion'] if recent else 'neutral'
            cog_state    = recent[0]['cognitive_state'] if (recent and 'cognitive_state' in recent[0].keys() and recent[0]['cognitive_state']) else EMOTION_TO_COGNITIVE.get(last_emotion, 'engaged')
            cog_reasons  = recent[0]['cognitive_reasons'] if (recent and 'cognitive_reasons' in recent[0].keys() and recent[0]['cognitive_reasons']) else 'Active student session'
            att_score    = int(recent[0]['attention_score']) if (recent and 'attention_score' in recent[0].keys() and recent[0]['attention_score'] is not None) else 80
            emo_conf     = float(recent[0]['emotion_confidence']) if (recent and 'emotion_confidence' in recent[0].keys() and recent[0]['emotion_confidence'] is not None) else 0.80
            emo_stab     = float(recent[0]['emotion_stability']) if (recent and 'emotion_stability' in recent[0].keys() and recent[0]['emotion_stability'] is not None) else 0.78
            curr_diff    = recent[0]['difficulty'] if (recent and 'difficulty' in recent[0].keys() and recent[0]['difficulty']) else 'easy'

            # Recent accuracy
            if recent:
                acc = round(sum(r['correct'] for r in recent) / len(recent) * 100, 1)
            else:
                acc = 0

            # Alert level
            if cog_state in ['overloaded', 'struggling']:
                alert = 'danger'
            elif cog_state in ['distracted', 'bored', 'anxious']:
                alert = 'warning'
            else:
                alert = 'ok'

            level_info = calculate_level(stats['xp'])

            students.append({
                'user_id':         u['id'],
                'username':        u['username'],
                'emotion':         last_emotion,
                'emotion_conf':    round(emo_conf * 100),
                'emotion_stab':    round(emo_stab * 100),
                'attention_score': att_score,
                'cog_state':       cog_state,
                'cog_info':        COGNITIVE_STATES.get(cog_state, COGNITIVE_STATES['engaged']),
                'cog_reasons':     cog_reasons,
                'difficulty':      curr_diff,
                'accuracy':        acc,
                'xp':              stats['xp'],
                'level':           level_info[1],
                'level_emoji':     level_info[3],
                'streak':          stats['streak'],
                'total_q':         stats['total_questions'],
                'alert':           alert,
                'last_active':     recent[0]['timestamp'][:16].replace('T', ' ') if recent else 'Never',
            })

        # Class-wide stats
        total_students  = len(students)
        in_flow         = sum(1 for s in students if s['cog_state'] == 'flow')
        struggling      = sum(1 for s in students if s['alert'] == 'danger')
        avg_accuracy    = round(sum(s['accuracy'] for s in students) / max(total_students, 1), 1)

        EMOTION_EMOJI = {
            'happy': '😄', 'sad': '😢', 'angry': '😠',
            'fear': '😨', 'surprise': '😲', 'disgust': '🤢', 'neutral': '😐'
        }
        for s in students:
            s['emotion_emoji'] = EMOTION_EMOJI.get(s['emotion'], '😐')

        return jsonify({
            'students':       students,
            'class_stats': {
                'total':        total_students,
                'in_flow':      in_flow,
                'struggling':   struggling,
                'avg_accuracy': avg_accuracy,
                'engagement':   round(in_flow / max(total_students, 1) * 100, 1),
            }
        })
    except Exception as e:
        print(f"[TEACHER ERROR] {e}")
        return jsonify({'students': [], 'class_stats': {}}), 500


@app.route('/api/teacher/send-hint', methods=['POST'])
@login_required
def teacher_send_hint():
    """
    Teacher sends a hint/message to a specific student.
    POST body: {target_user_id, message}
    Message is stored and picked up by student's next /api/analyze call.
    """
    d              = request.json or {}
    target_user_id = d.get('target_user_id')
    message        = d.get('message', '')

    if not target_user_id or not message:
        return jsonify({'error': 'Missing target_user_id or message'}), 400

    # Store teacher message — student picks it up on next analyze call
    if 'teacher_messages' not in sessions:
        sessions['teacher_messages'] = {}
    sessions['teacher_messages'][str(target_user_id)] = {
        'message':   message,
        'from':      session.get('username', 'Teacher'),
        'timestamp': datetime.now().isoformat(),
    }
    return jsonify({'status': 'sent', 'to': target_user_id, 'message': message})


# =============================================================================
# STARTUP
# =============================================================================

# This block runs both when:
#   python app.py        (development)
#   gunicorn app:app     (production — init_db called at module load)
init_db()  # Always run at import time so Gunicorn workers have tables ready

if False:  # app.run moved to end of file so all routes register first
    print("\n" + "="*55)
    print("  EmotionLearnAI - Phase 3 Production Build")
    print("="*55)
    print(f"  Gemini AI   : {'[ON]  Enabled' if GEMINI_AVAILABLE else '[OFF] Set GEMINI_API_KEY in .env'}")
    print(f"  Database    : {'PostgreSQL' if USE_POSTGRES else 'SQLite (local)'}")
    print("  URLs        : http://localhost:5000")
    print("              : http://localhost:5000/dashboard")
    print("              : http://localhost:5000/leaderboard")
    print("              : http://localhost:5000/profile")
    print("              : http://localhost:5000/generate-question?topic=Python+Basics")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)


# =============================================================================
# MULTI-DOMAIN EXTENSION — Phase 1: Role/Domain Layer
# =============================================================================

def init_multidomain_db():
    """Create new tables for multi-domain features. Never modifies existing tables."""
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS interviews (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                job_role     TEXT NOT NULL,
                interview_type TEXT NOT NULL,
                status       TEXT DEFAULT 'in_progress',
                total_questions INTEGER DEFAULT 0,
                answered     INTEGER DEFAULT 0,
                created_at   TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS interview_answers (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id INTEGER NOT NULL,
                question_no  INTEGER NOT NULL,
                question_text TEXT NOT NULL,
                user_answer  TEXT NOT NULL,
                score        REAL DEFAULT 0,
                feedback     TEXT DEFAULT '',
                time_taken   REAL DEFAULT 0,
                emotion      TEXT DEFAULT 'neutral',
                attention    INTEGER DEFAULT 75,
                created_at   TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS skill_assessments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                category     TEXT NOT NULL,
                score        REAL DEFAULT 0,
                total_q      INTEGER DEFAULT 0,
                correct_q    INTEGER DEFAULT 0,
                assessed_at  TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_programs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT NOT NULL,
                description  TEXT DEFAULT '',
                topics       TEXT DEFAULT '',
                created_by   INTEGER,
                created_at   TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_enrollments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                program_id   INTEGER NOT NULL,
                progress     REAL DEFAULT 0,
                score        REAL DEFAULT 0,
                enrolled_at  TEXT NOT NULL,
                UNIQUE(user_id, program_id)
            )
        """)

        conn.commit()
        conn.close()
        print("[DB] Multi-domain tables ready.")
    except Exception as e:
        print(f"[DB MULTIDOMAIN ERROR] {e}")


# Job roles with required skills
JOB_ROLES = {
    "Software Developer": ["Python", "Data Structures", "Algorithms", "OOP", "Git", "SQL"],
    "Frontend Developer": ["HTML/CSS", "JavaScript", "React", "REST APIs", "Git"],
    "Backend Developer": ["Python/Node.js", "REST APIs", "SQL", "Authentication", "Git"],
    "Full Stack Developer": ["HTML/CSS", "JavaScript", "React", "Python", "SQL", "REST APIs", "Git"],
    "Data Analyst": ["Python", "SQL", "Pandas", "Data Visualization", "Statistics"],
    "Data Scientist": ["Python", "ML", "Deep Learning", "Statistics", "SQL", "Data Visualization"],
    "Cybersecurity Analyst": ["Networking", "Encryption", "Linux", "Ethical Hacking", "SQL"],
    "Cloud Engineer": ["AWS/Azure/GCP", "Linux", "Networking", "Docker", "Python"],
}

# Technical skill assessment question bank
SKILL_QUESTIONS = {
    "Data Structures": [
        {"id": "ds1", "q": "What is the time complexity of binary search?", "opts": ["O(n)", "O(log n)", "O(n^2)"], "ans": 1},
        {"id": "ds2", "q": "Which data structure uses LIFO order?", "opts": ["Queue", "Stack", "Array"], "ans": 1},
        {"id": "ds3", "q": "What is a linked list?", "opts": ["Array with fixed size", "Nodes linked by pointers", "Hash table"], "ans": 1},
    ],
    "Algorithms": [
        {"id": "al1", "q": "Which sorting algorithm has O(n log n) average case?", "opts": ["Bubble Sort", "Merge Sort", "Insertion Sort"], "ans": 1},
        {"id": "al2", "q": "What does BFS stand for?", "opts": ["Best First Search", "Breadth First Search", "Binary Fast Search"], "ans": 1},
        {"id": "al3", "q": "Dynamic programming solves problems by:", "opts": ["Brute force", "Storing subproblem results", "Random guessing"], "ans": 1},
    ],
    "DBMS": [
        {"id": "db1", "q": "ACID stands for:", "opts": ["Atomicity Consistency Isolation Durability", "Array Cache Index Data", "None"], "ans": 0},
        {"id": "db2", "q": "Which SQL clause filters grouped results?", "opts": ["WHERE", "HAVING", "ORDER BY"], "ans": 1},
        {"id": "db3", "q": "A primary key must be:", "opts": ["Nullable", "Unique and not null", "Duplicate allowed"], "ans": 1},
    ],
    "Networking": [
        {"id": "nw1", "q": "HTTP operates at which OSI layer?", "opts": ["Transport", "Application", "Network"], "ans": 1},
        {"id": "nw2", "q": "What does DNS do?", "opts": ["Encrypts data", "Resolves domain names to IPs", "Routes packets"], "ans": 1},
        {"id": "nw3", "q": "TCP provides:", "opts": ["Unreliable delivery", "Reliable ordered delivery", "Broadcast only"], "ans": 1},
    ],
    "Operating Systems": [
        {"id": "os1", "q": "What is a deadlock?", "opts": ["Fast process", "Processes waiting for each other forever", "Memory overflow"], "ans": 1},
        {"id": "os2", "q": "Virtual memory allows:", "opts": ["Faster CPU", "More RAM than physically available", "GPU acceleration"], "ans": 1},
        {"id": "os3", "q": "A process vs thread: threads share:", "opts": ["Nothing", "Memory space of the process", "CPU only"], "ans": 1},
    ],
}

# Interview question bank per type
INTERVIEW_QUESTIONS = {
    "HR Interview": [
        "Tell me about yourself and your background.",
        "What are your greatest strengths and weaknesses?",
        "Where do you see yourself in 5 years?",
        "Why do you want to work for this company?",
        "Describe a challenging situation and how you handled it.",
    ],
    "Technical Interview": [
        "Explain the difference between a stack and a queue.",
        "What is object-oriented programming? Name its four pillars.",
        "Explain what REST API is and how it works.",
        "What is the difference between SQL and NoSQL databases?",
        "Explain time complexity and give an example.",
    ],
    "Behavioral Interview": [
        "Tell me about a time you worked in a team under pressure.",
        "Describe a situation where you had to learn something quickly.",
        "Give an example of a time you showed leadership.",
        "Tell me about a time you failed and what you learned.",
        "Describe how you prioritize tasks when you have multiple deadlines.",
    ],
    "Communication Assessment": [
        "Explain a complex technical concept to a non-technical person.",
        "How do you handle disagreements with colleagues?",
        "Describe your approach to giving and receiving feedback.",
        "How do you ensure clear communication in a remote team?",
        "Tell me about a presentation you gave and how it went.",
    ],
}


# =============================================================================
# MULTI-DOMAIN PAGE ROUTES
# =============================================================================

@app.route('/select-role')
@login_required
def select_role():
    """Multi-domain role selection hub."""
    profile = get_safe_profile(session['user_id'])
    return render_template('role_select.html', user=profile)


@app.route('/interview')
@login_required
def interview_page():
    """AI Interview Assessment page."""
    profile = get_safe_profile(session['user_id'])
    return render_template('interview.html', user=profile, job_roles=list(JOB_ROLES.keys()))


@app.route('/skill-assessment')
@login_required
def skill_assessment_page():
    """Technical Skill Assessment page."""
    profile = get_safe_profile(session['user_id'])
    return render_template('skill_assessment.html', user=profile, categories=list(SKILL_QUESTIONS.keys()))


@app.route('/employee-training')
@login_required
def employee_training_page():
    """Employee Training page."""
    profile = get_safe_profile(session['user_id'])
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            programs = conn.execute("SELECT * FROM training_programs ORDER BY created_at DESC").fetchall()
            programs = [dict(p) for p in programs]
    except Exception:
        programs = []
    return render_template('employee_training.html', user=profile, programs=programs)


@app.route('/candidate-dashboard')
@login_required
def candidate_dashboard_page():
    """Candidate dashboard showing interview results and skill gaps."""
    profile = get_safe_profile(session['user_id'])
    return render_template('candidate_dashboard.html', user=profile)


# =============================================================================
# MULTI-DOMAIN API ROUTES
# =============================================================================

@app.route('/api/job-roles')
@login_required
def api_job_roles():
    """Return all job roles with required skills."""
    result = [{"role": role, "skills": skills} for role, skills in JOB_ROLES.items()]
    return jsonify(result)


@app.route('/api/interview/start', methods=['POST'])
@login_required
def api_interview_start():
    """Start a new interview session."""
    d = request.json or {}
    job_role = d.get('job_role', 'Software Developer')
    interview_type = d.get('interview_type', 'Technical Interview')
    uid = session['user_id']

    questions = INTERVIEW_QUESTIONS.get(interview_type, INTERVIEW_QUESTIONS['Technical Interview'])

    # Use Gemini to generate additional questions if available
    ai_questions = []
    if GEMINI_AVAILABLE and _GEMINI_CLIENT:
        try:
            prompt = f'Generate 3 {interview_type} questions for a {job_role} position. Return ONLY a JSON array of strings. Example: ["Q1", "Q2", "Q3"]'
            response = _GEMINI_CLIENT.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            text = response.text.strip().strip('`').strip()
            if text.startswith('json'):
                text = text[4:].strip()
            ai_questions = json.loads(text)
            if not isinstance(ai_questions, list):
                ai_questions = []
        except Exception:
            ai_questions = []

    all_questions = questions + ai_questions
    random.shuffle(all_questions)
    selected = all_questions[:5]

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "INSERT INTO interviews (user_id, job_role, interview_type, total_questions, created_at) VALUES (?,?,?,?,?)",
                (uid, job_role, interview_type, len(selected), datetime.now().isoformat())
            )
            interview_id = cur.lastrowid
            conn.commit()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'interview_id': interview_id,
        'job_role': job_role,
        'interview_type': interview_type,
        'questions': selected,
        'total': len(selected),
    })


@app.route('/api/interview/answer', methods=['POST'])
@login_required
def api_interview_answer():
    """Submit an answer for an interview question."""
    d = request.json or {}
    interview_id = d.get('interview_id')
    question_no = d.get('question_no', 1)
    question_text = d.get('question_text', '')
    user_answer = d.get('user_answer', '')
    time_taken = float(d.get('time_taken', 0))
    emotion = d.get('emotion', 'neutral')
    attention = int(d.get('attention', 75))

    if not user_answer.strip():
        return jsonify({'score': 0, 'feedback': 'No answer provided.', 'saved': False})

    # Score the answer using Gemini if available
    score = 0.0
    feedback = ''
    if GEMINI_AVAILABLE and _GEMINI_CLIENT:
        try:
            prompt = (
                f'Interview question: "{question_text}"\n'
                f'Candidate answer: "{user_answer}"\n\n'
                f'Score this answer from 0 to 10 based on relevance, clarity, and completeness.\n'
                f'Return ONLY valid JSON: {{"score": 7, "feedback": "Brief feedback here"}}'
            )
            response = _GEMINI_CLIENT.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            text = response.text.strip().strip('`').strip()
            if text.startswith('json'):
                text = text[4:].strip()
            result = json.loads(text)
            score = float(result.get('score', 5)) / 10.0
            feedback = result.get('feedback', '')
        except Exception:
            score = 0.5
            feedback = 'Answer recorded. AI evaluation unavailable.'
    else:
        # Simple keyword-based scoring fallback
        word_count = len(user_answer.split())
        score = min(1.0, word_count / 50.0)
        feedback = 'Answer recorded. Add GEMINI_API_KEY for AI-powered evaluation.'

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO interview_answers (interview_id, question_no, question_text, user_answer, score, feedback, time_taken, emotion, attention, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (interview_id, question_no, question_text, user_answer, score, feedback, time_taken, emotion, attention, datetime.now().isoformat())
            )
            conn.execute(
                "UPDATE interviews SET answered = answered + 1 WHERE id = ?",
                (interview_id,)
            )
            conn.commit()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'score': round(score * 10, 1), 'feedback': feedback, 'saved': True})


@app.route('/api/interview/report/<int:interview_id>')
@login_required
def api_interview_report(interview_id):
    """Generate interview report with scores and skill gap analysis."""
    uid = session['user_id']
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            interview = conn.execute(
                "SELECT * FROM interviews WHERE id=? AND user_id=?", (interview_id, uid)
            ).fetchone()
            if not interview:
                return jsonify({'error': 'Interview not found'}), 404
            answers = conn.execute(
                "SELECT * FROM interview_answers WHERE interview_id=? ORDER BY question_no",
                (interview_id,)
            ).fetchall()

        answers = [dict(a) for a in answers]
        total_score = sum(a['score'] for a in answers)
        avg_score = round(total_score / max(len(answers), 1) * 10, 1)
        avg_time = round(sum(a['time_taken'] for a in answers) / max(len(answers), 1), 1)

        # Emotion distribution during interview
        emotion_dist = {}
        for a in answers:
            e = a['emotion'] or 'neutral'
            emotion_dist[e] = emotion_dist.get(e, 0) + 1

        # Skill gap vs job role
        job_role = dict(interview)['job_role']
        required_skills = JOB_ROLES.get(job_role, [])
        skill_gap = {
            'required': required_skills,
            'assessed_score': avg_score,
            'recommendation': (
                f'Focus on strengthening: {", ".join(required_skills[:3])}.'
                if avg_score < 6 else
                f'Good foundation in {job_role} skills. Practice advanced topics.'
            )
        }

        return jsonify({
            'interview': dict(interview),
            'answers': answers,
            'summary': {
                'total_questions': len(answers),
                'avg_score': avg_score,
                'avg_time': avg_time,
                'emotion_distribution': emotion_dist,
            },
            'skill_gap': skill_gap,
            'generated_at': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/interview/history')
@login_required
def api_interview_history():
    """Return all interviews for the current user."""
    uid = session['user_id']
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM interviews WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
                (uid,)
            ).fetchall()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify([]), 500


@app.route('/api/skill-assessment/submit', methods=['POST'])
@login_required
def api_skill_assessment_submit():
    """Submit skill assessment results."""
    d = request.json or {}
    uid = session['user_id']
    category = d.get('category', 'Data Structures')
    answers = d.get('answers', [])

    questions = SKILL_QUESTIONS.get(category, [])
    correct = 0
    for ans in answers:
        qid = ans.get('id')
        chosen = ans.get('chosen')
        for q in questions:
            if q['id'] == qid and chosen == q['ans']:
                correct += 1
                break

    total = len(questions)
    score = round(correct / max(total, 1) * 100, 1)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO skill_assessments (user_id, category, score, total_q, correct_q, assessed_at) VALUES (?,?,?,?,?,?)",
                (uid, category, score, total, correct, datetime.now().isoformat())
            )
            conn.commit()
    except Exception as e:
        pass

    return jsonify({
        'category': category,
        'score': score,
        'correct': correct,
        'total': total,
        'level': 'Strong' if score >= 75 else ('Medium' if score >= 50 else 'Needs Practice'),
    })


@app.route('/api/skill-assessment/history')
@login_required
def api_skill_assessment_history():
    """Return skill assessment history for the current user."""
    uid = session['user_id']
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM skill_assessments WHERE user_id=? ORDER BY assessed_at DESC",
                (uid,)
            ).fetchall()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify([])


@app.route('/api/skill-gap')
@login_required
def api_skill_gap():
    """Return skill gap analysis for the current user based on assessments."""
    uid = session['user_id']
    job_role = request.args.get('job_role', 'Software Developer')
    required = JOB_ROLES.get(job_role, [])

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT category, AVG(score) as avg_score FROM skill_assessments WHERE user_id=? GROUP BY category",
                (uid,)
            ).fetchall()
        assessed = {r['category']: round(r['avg_score'], 1) for r in rows}
    except Exception:
        assessed = {}

    gaps = []
    strengths = []
    for skill in required:
        score = assessed.get(skill, None)
        if score is None:
            gaps.append({'skill': skill, 'status': 'Not Assessed', 'score': None})
        elif score >= 75:
            strengths.append({'skill': skill, 'status': 'Strong', 'score': score})
        elif score >= 50:
            gaps.append({'skill': skill, 'status': 'Medium', 'score': score})
        else:
            gaps.append({'skill': skill, 'status': 'Weak', 'score': score})

    return jsonify({
        'job_role': job_role,
        'required_skills': required,
        'strengths': strengths,
        'gaps': gaps,
        'recommendation': (
            f'Practice: {", ".join(g["skill"] for g in gaps[:3])}.'
            if gaps else 'Great skill coverage for this role!'
        )
    })


@app.route('/api/training/programs')
@login_required
def api_training_programs():
    """Return all training programs."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM training_programs ORDER BY created_at DESC").fetchall()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify([])


@app.route('/api/training/enroll', methods=['POST'])
@login_required
def api_training_enroll():
    """Enroll current user in a training program."""
    d = request.json or {}
    uid = session['user_id']
    program_id = d.get('program_id')
    if not program_id:
        return jsonify({'error': 'program_id required'}), 400
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO training_enrollments (user_id, program_id, enrolled_at) VALUES (?,?,?)",
                (uid, program_id, datetime.now().isoformat())
            )
            conn.commit()
        return jsonify({'status': 'enrolled', 'program_id': program_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/progress', methods=['POST'])
@login_required
def api_training_progress():
    """Update training progress for current user."""
    d = request.json or {}
    uid = session['user_id']
    program_id = d.get('program_id')
    progress = float(d.get('progress', 0))
    score = float(d.get('score', 0))
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE training_enrollments SET progress=?, score=? WHERE user_id=? AND program_id=?",
                (progress, score, uid, program_id)
            )
            conn.commit()
        return jsonify({'status': 'updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/my-enrollments')
@login_required
def api_my_enrollments():
    """Return training enrollments for current user."""
    uid = session['user_id']
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT e.*, p.name, p.description, p.topics
                   FROM training_enrollments e
                   JOIN training_programs p ON p.id = e.program_id
                   WHERE e.user_id=? ORDER BY e.enrolled_at DESC""",
                (uid,)
            ).fetchall()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify([])


@app.route('/api/training/create', methods=['POST'])
@login_required
def api_training_create():
    """Create a new training program (trainer/admin)."""
    d = request.json or {}
    uid = session['user_id']
    name = d.get('name', '').strip()
    description = d.get('description', '')
    topics = d.get('topics', '')
    if not name:
        return jsonify({'error': 'Program name required'}), 400
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "INSERT INTO training_programs (name, description, topics, created_by, created_at) VALUES (?,?,?,?,?)",
                (name, description, topics, uid, datetime.now().isoformat())
            )
            conn.commit()
            return jsonify({'status': 'created', 'program_id': cur.lastrowid})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run multi-domain DB init
init_multidomain_db()


@app.route('/api/skill-assessment/questions')
@login_required
def api_skill_questions():
    """Return questions for a skill assessment category."""
    category = request.args.get('category', 'Data Structures')
    questions = SKILL_QUESTIONS.get(category, [])
    return jsonify({'category': category, 'questions': questions})


@app.route('/interview-report/<int:interview_id>')
@login_required
def interview_report_page(interview_id):
    """Interview report detail page."""
    profile = get_safe_profile(session['user_id'])
    return render_template('interview_report.html', user=profile, interview_id=interview_id)


@app.route('/recruiter-dashboard')
@login_required
def recruiter_dashboard_page():
    """Recruiter dashboard — candidate assessment results and skill coverage."""
    profile = get_safe_profile(session['user_id'])
    return render_template('recruiter_dashboard.html', user=profile)


@app.route('/admin-dashboard')
@login_required
def admin_dashboard_page():
    """Admin/Trainer dashboard — platform overview, users, programs, reports."""
    profile = get_safe_profile(session['user_id'])
    return render_template('admin_dashboard.html', user=profile)



# ===== FIXED MAIN ENTRY POINT =====
# app.run() is here so ALL routes (including multi-domain) are registered first
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  EmotionLearnAI - Multi-Domain Platform")
    print("="*55)
    import os as _os
    _gemini = _os.getenv('GEMINI_API_KEY', '')
    print(f"  Gemini AI   : {'[ON]  Enabled' if _gemini else '[OFF] Set GEMINI_API_KEY in .env'}")
    print("  URLs        : http://localhost:5000")
    print("              : http://localhost:5000/select-role")
    print("              : http://localhost:5000/interview")
    print("              : http://localhost:5000/skill-assessment")
    print("              : http://localhost:5000/employee-training")
    print("              : http://localhost:5000/recruiter-dashboard")
    print("              : http://localhost:5000/admin-dashboard")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
