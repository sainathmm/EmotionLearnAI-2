# ============================================================
# Gunicorn Production Server Configuration
# ============================================================
# Run with: gunicorn --config gunicorn.conf.py app:app
# ============================================================
import os

# ── Workers ───────────────────────────────────────────────────
# Rule of thumb: 2 * CPU cores + 1
# Use 2 for free-tier hosting (Render/Railway have 0.5–1 CPU)
workers = int(os.getenv("WEB_CONCURRENCY", 2))

# ── Worker type ───────────────────────────────────────────────
# sync = default, safe for SQLite
# gevent/gthread = for async; requires: pip install gevent
worker_class = "sync"

# ── Binding ───────────────────────────────────────────────────
# Render/Railway inject PORT automatically
port = int(os.getenv("PORT", 5000))
bind = f"0.0.0.0:{port}"

# ── Timeouts ─────────────────────────────────────────────────
# 120s for emotion analysis (camera frames can be slow)
timeout = 120
keepalive = 5

# ── Logging ──────────────────────────────────────────────────
accesslog = "-"    # stdout
errorlog  = "-"    # stderr
loglevel  = os.getenv("LOG_LEVEL", "info")

# ── Process name ─────────────────────────────────────────────
proc_name = "emotionlearnai"
