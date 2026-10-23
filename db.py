"""
db.py — Database Abstraction Layer (Phase 3)
============================================
Provides a unified interface for both PostgreSQL and SQLite.

Usage in app.py:
    from db import get_conn, placeholder, adapt_row

The rest of app.py keeps the same row['field'] access pattern because:
  - SQLite  → sqlite3.Row     → supports row['field']
  - Postgres → RealDictCursor → supports row['field']
"""

import os
import sqlite3

# ── Try to import psycopg2 (PostgreSQL driver) ────────────────────────────────
try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_OK = True
except ImportError:
    _PSYCOPG2_OK = False

# ── Detect which backend to use ───────────────────────────────────────────────
DB_PATH     = os.path.join(os.path.dirname(__file__), 'database.db')
DATABASE_URL = os.getenv("DATABASE_URL", "")
USE_POSTGRES = bool(DATABASE_URL and _PSYCOPG2_OK)


def get_conn():
    """
    Return an open database connection.

    - If DATABASE_URL env var is set AND psycopg2 is installed → PostgreSQL
    - Otherwise → SQLite (uses local database.db)

    Always call conn.close() when done, or use as a context manager.
    Rows from both backends support dict-style access: row['column_name']
    """
    if USE_POSTGRES:
        conn = psycopg2.connect(
            DATABASE_URL,
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def ph():
    """
    Return the correct SQL placeholder string.
    - PostgreSQL uses %s
    - SQLite    uses ?
    """
    return "%s" if USE_POSTGRES else "?"


def adapt_sql(sql):
    """
    Automatically convert ? placeholders to %s for PostgreSQL.
    Allows writing all SQL with ? and calling adapt_sql() when needed.
    """
    if USE_POSTGRES:
        return sql.replace("?", "%s")
    return sql


def last_insert_id(cursor, table=None, pk="id"):
    """
    Get the last inserted row ID in a backend-agnostic way.
    PostgreSQL: add RETURNING id to INSERT and fetch it.
    SQLite:     use cursor.lastrowid.
    """
    if USE_POSTGRES:
        row = cursor.fetchone()
        return row[pk] if row else None
    return cursor.lastrowid


def db_info():
    """Print which DB backend is active (called at startup)."""
    if USE_POSTGRES:
        print(f"[DB] Backend: PostgreSQL | {DATABASE_URL[:40]}...")
    else:
        print(f"[DB] Backend: SQLite | {DB_PATH}")
