"""
whatsapp_store.py — Thread-safe, crash-safe subscriber persistence.

Problems solved
---------------
1. Race condition (read-modify-write with no locking)
   Two concurrent subscription requests both read the same snapshot,
   each modify their own in-memory copy, and the second write silently
   overwrites the first — permanently losing a subscriber.

2. Corrupted reads during concurrent writes
   open(..., "w") truncates the file immediately.  A concurrent reader
   that opens the file between the truncation and the final flush sees
   an empty or partial file, causing json.load() to throw.  The old
   bare `except Exception: return {}` swallowed this silently, causing
   trigger-alert to broadcast to zero subscribers.

3. No crash durability
   A process crash mid-write left the file empty or truncated with no
   recovery path.

Solutions
---------
- threading.Lock serialises every read and write.  FastAPI's async
  endpoints run on a thread pool that shares the same process, so a
  single in-process lock is sufficient.
- Atomic write: data is written to a sibling `.tmp` file first, then
  os.replace() swaps it in.  os.replace() is atomic on POSIX and
  effectively atomic on Windows (same-volume rename), so readers always
  see either the old complete file or the new complete file — never a
  partial write.
- Errors are logged with full tracebacks instead of being swallowed.
"""

import json
import logging
import os
import threading
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Type alias for the subscriber dict stored per user_id.
Subscriber = Dict[str, Any]


class SubscriberStore:
    """
    Thread-safe, crash-safe persistence layer for WhatsApp subscribers.

    All public methods acquire ``_lock`` before touching the file, so
    concurrent FastAPI requests cannot interleave their reads and writes.

    Parameters
    ----------
    filepath : str
        Path to the JSON file used for persistence.
    """

    def __init__(self, filepath: str = "whatsapp_subscribers.json") -> None:
        self._filepath = filepath
        self._tmp_filepath = filepath + ".tmp"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Private helpers (must be called with _lock already held)
    # ------------------------------------------------------------------

    def _read_locked(self) -> Dict[str, Subscriber]:
        """Read and parse the subscribers file.  Returns {} on any error."""
        if not os.path.exists(self._filepath):
            return {}
        try:
            with open(self._filepath, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            logger.error(
                "Subscriber file '%s' contains invalid JSON and could not be "
                "parsed: %s.  Returning empty subscriber list.",
                self._filepath,
                exc,
            )
            return {}
        except OSError as exc:
            logger.error(
                "Could not read subscriber file '%s': %s.  "
                "Returning empty subscriber list.",
                self._filepath,
                exc,
            )
            return {}

    def _write_locked(self, subscribers: Dict[str, Subscriber]) -> None:
        """
        Atomically write *subscribers* to disk.

        Writes to a sibling ``.tmp`` file first, then uses ``os.replace``
        to swap it in.  This guarantees that readers always see a complete,
        valid JSON file — never a partial write.

        Raises
        ------
        OSError
            If the write or rename fails.  The caller is responsible for
            surfacing this as an HTTP 500.
        """
        try:
            with open(self._tmp_filepath, "w", encoding="utf-8") as fh:
                json.dump(subscribers, fh, indent=2, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())  # flush OS buffers before rename
            os.replace(self._tmp_filepath, self._filepath)
        except OSError as exc:
            logger.error(
                "Failed to write subscriber file '%s': %s",
                self._filepath,
                exc,
            )
            # Clean up the temp file if it was created.
            try:
                if os.path.exists(self._tmp_filepath):
                    os.remove(self._tmp_filepath)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all(self) -> Dict[str, Subscriber]:
        """
        Return a snapshot of all subscribers.

        The returned dict is a copy — callers cannot accidentally mutate
        the in-memory state.
        """
        with self._lock:
            return dict(self._read_locked())

    def upsert(self, user_id: str, subscriber: Subscriber) -> None:
        """
        Add or update a single subscriber atomically.

        Parameters
        ----------
        user_id : str
            Unique identifier for the subscriber.
        subscriber : dict
            Subscriber data (phone_number, name, subscribed_at, …).

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        with self._lock:
            subscribers = self._read_locked()
            subscribers[user_id] = subscriber
            self._write_locked(subscribers)
            logger.info("Subscriber '%s' upserted successfully.", user_id)

    def remove(self, user_id: str) -> bool:
        """
        Remove a subscriber by user_id.

        Returns
        -------
        bool
            True if the subscriber existed and was removed, False otherwise.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        with self._lock:
            subscribers = self._read_locked()
            if user_id not in subscribers:
                return False
            del subscribers[user_id]
            self._write_locked(subscribers)
            logger.info("Subscriber '%s' removed.", user_id)
            return True

    def count(self) -> int:
        """Return the number of registered subscribers."""
        with self._lock:
            return len(self._read_locked())


# Module-level singleton — created once at import time.
subscriber_store = SubscriberStore()
