"""
Sync Worker
===========
Background thread that polls connectivity and flushes the SQLite queue to Firestore.
"""

import hashlib
import json
import logging
import os
import threading
import time
from typing import Optional

from persistence.offline_sync import get_pending, mark_synced, mark_failed, resolve_conflict, prune_synced

logger = logging.getLogger(__name__)

POLL_INTERVAL = int(os.getenv("SYNC_WORKER_POLL_INTERVAL", "10"))
BATCH_SIZE = int(os.getenv("SYNC_WORKER_BATCH_SIZE", "100"))
MAX_ATTEMPTS = int(os.getenv("SYNC_WORKER_MAX_ATTEMPTS", "5"))


class SyncWorker:
    def __init__(self, db_firestore):
        self.db = db_firestore
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._tick_count = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info("Sync worker started (interval=%ds, batch=%d)", POLL_INTERVAL, BATCH_SIZE)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Sync worker stopped")

    def _loop(self):
        while self.running:
            try:
                self._tick()
            except Exception:
                logger.exception("Sync worker tick failed")
            time.sleep(POLL_INTERVAL)

    def _tick(self):
        if self.db is None:
            logger.debug("Firestore unavailable, skipping sync tick")
            return

        self._tick_count += 1
        if self._tick_count % 360 == 0:
            prune_synced()

        pending = get_pending(limit=BATCH_SIZE, max_attempts=MAX_ATTEMPTS)
        if not pending:
            return

        logger.info("Syncing %d pending farm intelligence entries", len(pending))

        for item in pending:
            try:
                self._sync_item(item)
            except Exception as exc:
                logger.exception("Failed syncing item %s", item["id"])
                mark_failed(item["id"], str(exc)[:500])

    def _sync_item(self, item: dict):
        uid = item["uid"]
        payload = json.loads(item["payload"])
        local_vector = json.loads(item["conflict_vector"])

        doc_id = hashlib.sha256(
            f"{uid}:{item['created_at']}:{item['device_id']}".encode()
        ).hexdigest()

        doc_ref = self.db.collection("users").document(uid).collection("farm_intelligence_history").document(doc_id)

        remote_doc = doc_ref.get()
        if remote_doc.exists:
            remote_data = remote_doc.to_dict() or {}
            remote_vector = remote_data.get("_conflict_vector")
            if remote_vector and not resolve_conflict(local_vector, remote_vector):
                logger.info("Remote wins for doc %s, dropping local", doc_id)
                mark_synced(item["id"])
                return

        payload["_conflict_vector"] = local_vector
        payload["_synced_at"] = time.time()
        doc_ref.set(payload)
        mark_synced(item["id"])
        logger.info("Synced farm intelligence doc %s for uid=%s", doc_id, uid)


_worker: Optional[SyncWorker] = None


def get_sync_worker(db_firestore=None) -> SyncWorker:
    global _worker
    if _worker is None:
        _worker = SyncWorker(db_firestore)
    return _worker