from datetime import datetime

from sync_conflict_resolver import DocumentVersion, CRDTResolver


def test_lww_merge_timestamp_order():
    t1 = "2023-01-01T00:00:00"
    t2 = "2023-01-01T00:01:00"

    local = DocumentVersion(doc_id="doc1", data={"x": 1}, client_id="clientA", timestamp=t1)
    server = DocumentVersion(doc_id="doc1", data={"x": 2}, client_id="clientB", timestamp=t2)

    merged, conflicts = CRDTResolver.merge(local, server)

    assert merged.data["x"] == 2
    assert "x" in conflicts


def test_lww_merge_equal_timestamp_clientid_tiebreak():
    ts = "2023-01-01T00:00:00"

    # client ids chosen to test deterministic tie-break
    local = DocumentVersion(doc_id="doc1", data={"k": "A"}, client_id="a", timestamp=ts)
    server = DocumentVersion(doc_id="doc1", data={"k": "B"}, client_id="b", timestamp=ts)

    merged, conflicts = CRDTResolver.merge(local, server)

    # 'b' >= 'a' so server should win in tie
    assert merged.data["k"] == "B"
    assert "k" in conflicts
