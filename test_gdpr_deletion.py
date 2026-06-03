from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gdpr_deletion import GDPRDeletionManager, DeletionTarget


class _FakeStore:
    def __init__(self, records: dict[str, list[str]]):
        self.records = records

    def delete_records(self, uid: str):
        deleted = len(self.records.get(uid, []))
        self.records.pop(uid, None)
        return {"deleted": deleted, "retained": 0 if deleted else 1, "notes": "deleted" if deleted else "missing"}


def test_deletion_request_respects_retention_window(tmp_path):
    manager = GDPRDeletionManager(
        request_log_path=tmp_path / "requests.jsonl",
        audit_log_path=tmp_path / "audit.jsonl",
    )

    now = datetime(2026, 5, 29, tzinfo=timezone.utc)
    request = manager.create_request("uid-123", retention_days=7, now=now)

    assert request["status"] == "pending_retention"
    assert request["uid"] == "uid-123"
    assert request["retention_until"].startswith("2026-06-05")
    assert manager.due_requests(now=now) == []
    assert manager.request_log_path.exists()
    assert manager.audit_log_path.exists()


def test_process_due_deletion_executes_targets_and_records_audit(tmp_path):
    manager = GDPRDeletionManager(
        request_log_path=tmp_path / "requests.jsonl",
        audit_log_path=tmp_path / "audit.jsonl",
    )

    store = _FakeStore({"uid-123": ["finance-1", "finance-2"]})
    request = manager.create_request(
        "uid-123",
        retention_days=0,
        now=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
    )

    targets = [
        DeletionTarget(name="finance_applications", delete=store.delete_records),
        DeletionTarget(name="immutable_ledger", delete=None, retain_reason="retained_by_policy"),
    ]

    processed = manager.execute_request(
        request["request_id"],
        targets,
        now=datetime(2026, 5, 29, 13, 0, tzinfo=timezone.utc),
    )

    assert processed["status"] == "completed"
    assert processed["deleted_entities"] == ["finance_applications"]
    assert "immutable_ledger" in processed["retained_entities"]
    assert store.records.get("uid-123") is None

    audit_lines = [line for line in Path(manager.audit_log_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(audit_lines) == 2
    parsed = [json.loads(line) for line in audit_lines]
    assert parsed[-1]["action"] == "deletion_completed"
    assert parsed[-1]["details"]["deleted_entities"] == ["finance_applications"]


def test_due_requests_only_include_elapsed_items(tmp_path):
    manager = GDPRDeletionManager(
        request_log_path=tmp_path / "requests.jsonl",
        audit_log_path=tmp_path / "audit.jsonl",
    )

    now = datetime(2026, 5, 29, tzinfo=timezone.utc)
    manager.create_request("uid-a", retention_days=0, now=now)
    manager.create_request("uid-b", retention_days=5, now=now)

    due = manager.due_requests(now=now + timedelta(hours=1))
    assert [item["uid"] for item in due] == ["uid-a"]


def test_in_memory_notification_cleanup_on_deletion(tmp_path):
    manager = GDPRDeletionManager(
        request_log_path=tmp_path / "requests.jsonl",
        audit_log_path=tmp_path / "audit.jsonl",
    )

    cleaned_uids = []
    def mock_cleanup_hook(uid: str):
        cleaned_uids.append(uid)

    manager.register_post_deletion_hook(mock_cleanup_hook)

    request = manager.create_request(
        "uid-123",
        retention_days=0,
        now=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
    )

    # Execute request
    manager.execute_request(
        request["request_id"],
        [],
        now=datetime(2026, 5, 29, 13, 0, tzinfo=timezone.utc),
    )

    assert "uid-123" in cleaned_uids
