"""
Offline-First SQLite Sync Layer
===============================
SQLite-backed write queue for farm intelligence with LWW CRDT conflict resolution.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_DB_PATH = Path(os.getenv("OFFLINE_SYNC_DB_PATH", "farm_intelligence_sync.db"))
_LOCK = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema():
    with _LOCK:
        conn = _get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS farm_intelligence_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    device_id TEXT NOT NULL,
                    synced INTEGER DEFAULT 0,
                    sync_attempts INTEGER DEFAULT 0,
                    last_error TEXT,
                    conflict_vector TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_synced ON farm_intelligence_queue(synced)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_uid ON farm_intelligence_queue(uid)")
            conn.commit()
            logger.info("Offline sync schema initialized at %s", _DB_PATH)
        finally:
            conn.close()


def queue_write(uid: str, payload: dict, device_id: str) -> int:
    created_at = time.time()
    conflict_vector = json.dumps({"timestamp": created_at, "device_id": device_id})
    payload_json = json.dumps(payload, default=str)

    with _LOCK:
        conn = _get_conn()
        try:
            cursor = conn.execute(
                """
                INSERT INTO farm_intelligence_queue (uid, payload, created_at, device_id, conflict_vector)
                VALUES (?, ?, ?, ?, ?)
                """,
                (uid, payload_json, created_at, device_id, conflict_vector),
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.info("Queued farm intelligence write id=%s for uid=%s", row_id, uid)
            return row_id
        finally:
            conn.close()


def get_pending(limit: int = 100, max_attempts: int = 5) -> List[dict]:
    with _LOCK:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM farm_intelligence_queue
                WHERE synced = 0 AND sync_attempts < ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (max_attempts, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def mark_synced(row_id: int):
    with _LOCK:
        conn = _get_conn()
        try:
            conn.execute(
                "UPDATE farm_intelligence_queue SET synced = 1 WHERE id = ?",
                (row_id,),
            )
            conn.commit()
        finally:
            conn.close()


def mark_failed(row_id: int, error: str):
    with _LOCK:
        conn = _get_conn()
        try:
            conn.execute(
                """
                UPDATE farm_intelligence_queue
                SET sync_attempts = sync_attempts + 1, last_error = ?
                WHERE id = ?
                """,
                (error, row_id),
            )
            conn.commit()
        finally:
            conn.close()


def prune_synced(older_than_seconds: int = 86400):
    cutoff = time.time() - older_than_seconds
    with _LOCK:
        conn = _get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM farm_intelligence_queue WHERE synced = 1 AND created_at < ?",
                (cutoff,),
            )
            conn.commit()
            if cursor.rowcount:
                logger.info("Pruned %d old synced records", cursor.rowcount)
        finally:
            conn.close()


def get_sync_stats() -> dict:
    with _LOCK:
        conn = _get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM farm_intelligence_queue").fetchone()[0]
            pending = conn.execute("SELECT COUNT(*) FROM farm_intelligence_queue WHERE synced = 0").fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM farm_intelligence_queue WHERE synced = 0 AND sync_attempts >= 5"
            ).fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(created_at) FROM farm_intelligence_queue WHERE synced = 0"
            ).fetchone()[0]
        finally:
            conn.close()

    return {
        "total_queued": total,
        "pending_sync": pending,
        "failed_permanently": failed,
        "oldest_pending_seconds": round(time.time() - oldest, 2) if oldest else None,
        "db_path": str(_DB_PATH),
    }


def resolve_conflict(local_vector: dict, remote_vector: dict) -> bool:
    """
    LWW CRDT: return True if local wins, False if remote wins.
    """
    if local_vector["timestamp"] > remote_vector["timestamp"]:
        return True
    if local_vector["timestamp"] < remote_vector["timestamp"]:
        return False
    return local_vector["device_id"] > remote_vector["device_id"]