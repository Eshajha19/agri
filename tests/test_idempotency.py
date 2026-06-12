"""Tests for idempotency-key deduplication in prediction endpoints."""

import json
from unittest.mock import MagicMock, patch

import pytest
from main import _IdempotencyStore


class TestIdempotencyStore:
    def test_fingerprint_deterministic(self):
        store = _IdempotencyStore()
        fp1 = store.fingerprint("key-1", '{"a":1}')
        fp2 = store.fingerprint("key-1", '{"a":1}')
        assert fp1 == fp2

    def test_fingerprint_differs_for_different_payloads(self):
        store = _IdempotencyStore()
        fp1 = store.fingerprint("key-1", '{"a":1}')
        fp2 = store.fingerprint("key-1", '{"a":2}')
        assert fp1 != fp2

    def test_start_returns_true_for_new_key(self):
        store = _IdempotencyStore()
        assert store.start("abc", "task-1") is True

    def test_start_returns_false_for_duplicate(self):
        store = _IdempotencyStore()
        store.start("abc", "task-1")
        assert store.start("abc", "task-2") is False

    def test_get_returns_none_for_unknown(self):
        store = _IdempotencyStore()
        assert store.get("unknown") is None

    def test_get_returns_in_progress(self):
        store = _IdempotencyStore()
        store.start("abc", "task-1")
        task_id, result = store.get("abc")
        assert task_id == "task-1"
        assert result is None

    def test_complete_stores_result(self):
        store = _IdempotencyStore()
        store.start("abc", "task-1")
        store.complete("abc", {"predicted_ExpYield": 2500.0})
        task_id, result = store.get("abc")
        assert task_id == "task-1"
        assert result == {"predicted_ExpYield": 2500.0}

    def test_evict_expired_entries(self):
        store = _IdempotencyStore()
        store.start("abc", "task-1")
        # Manually expire the entry
        import datetime
        past = datetime.datetime.now() - datetime.timedelta(hours=2)
        store._store["abc"] = ("task-1", None, past)
        store._evict()
        assert store.get("abc") is None

    def test_complete_nonexistent_is_noop(self):
        store = _IdempotencyStore()
        store.complete("nonexistent", {"x": 1})  # should not raise
        assert store.get("nonexistent") is None
