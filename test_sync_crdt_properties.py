from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random

from sync_conflict_resolver import CRDTResolver, DocumentVersion, SyncManager


BASE_TIME = datetime(2026, 5, 29, tzinfo=timezone.utc)


def _make_doc(doc_id: str, client_id: str, offset_seconds: int, data: dict) -> DocumentVersion:
    timestamp = (BASE_TIME + timedelta(seconds=offset_seconds)).isoformat()
    return DocumentVersion(doc_id=doc_id, data=data, client_id=client_id, timestamp=timestamp)


def _random_doc(rng: random.Random, doc_id: str, client_id: str, offset_seconds: int) -> DocumentVersion:
    data = {
        "name": rng.choice(["Alpha", "Beta", "Gamma", "Delta"]),
        "age": rng.randint(18, 80),
        "city": rng.choice(["Pune", "Mumbai", "Delhi", "Chennai"]),
        "active": rng.choice([True, False]),
    }
    return _make_doc(doc_id, client_id, offset_seconds, data)


def _assert_same_merge(left: DocumentVersion, right: DocumentVersion):
    merged_lr, conflicts_lr = CRDTResolver.merge(left, right)
    merged_rl, conflicts_rl = CRDTResolver.merge(right, left)

    assert merged_lr.data == merged_rl.data
    assert set(conflicts_lr) == set(conflicts_rl)
    return merged_lr


def test_crdt_merge_is_idempotent_across_random_documents():
    rng = random.Random(42)
    for index in range(50):
        doc = _random_doc(rng, f"doc-{index}", f"client-{index % 3}", index)
        merged, conflicts = CRDTResolver.merge(doc, doc)

        assert merged.data == doc.data
        assert conflicts == []


def test_crdt_merge_is_commutative_across_random_documents():
    rng = random.Random(7)
    for index in range(50):
        left = _random_doc(rng, f"doc-{index}", f"client-{index % 3}", index)
        right = _random_doc(rng, f"doc-{index}", f"client-{(index + 1) % 3}", index + 1)

        _assert_same_merge(left, right)


def test_crdt_merge_is_associative_across_random_documents():
    rng = random.Random(101)
    for index in range(25):
        a = _random_doc(rng, f"doc-{index}", "client-a", index)
        b = _random_doc(rng, f"doc-{index}", "client-b", index + 1)
        c = _random_doc(rng, f"doc-{index}", "client-c", index + 2)

        left_pair, left_conflicts = CRDTResolver.merge(a, b)
        left_result, left_again = CRDTResolver.merge(left_pair, c)

        right_pair, right_conflicts = CRDTResolver.merge(b, c)
        right_result, right_again = CRDTResolver.merge(a, right_pair)

        assert left_result.data == right_result.data
        assert set(left_conflicts + left_again) == set(right_conflicts + right_again)


def test_sync_manager_uses_crdt_mode_by_default():
    manager = SyncManager()

    manager.on_local_change("docs/1", {"title": "One", "status": "draft"}, "client-1")
    data, has_conflict, conflicts = manager.on_server_update(
        "docs/1",
        {"title": "One", "status": "published"},
    )

    assert manager.use_crdt is True
    assert data["title"] == "One"
    assert data["status"] in {"draft", "published"}
    assert isinstance(has_conflict, bool)
    assert isinstance(conflicts, list)
