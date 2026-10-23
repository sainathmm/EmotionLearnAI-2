# ============================================================
# EmotionLearnAI — Dockerfile
# ============================================================
# Build:  docker build -t emotionlearnai .
# Run:    docker run -p 5000:5000 --env-file .env emotionlearnai
# ============================================================

# ── Base image: slim Python 3.11 ─────────────────────────────
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────
# libgl1 and libglib2.0-0 are required by OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────
# Copy requirements first so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────
COPY . .

# ── Expose port ───────────────────────────────────────────────
EXPOSE 5000

# ── Run with Gunicorn ─────────────────────────────────────────
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
