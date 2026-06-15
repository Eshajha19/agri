from backend.sustainability_analytics import SustainabilityAnalytics
from backend.storage_backend import FileStorage

def test_file_storage_consistency(tmp_path):
    path = tmp_path / "history.json"
    backend = FileStorage(str(path))
    sa = SustainabilityAnalytics(backend=backend)

    sa._append_history("user1", {"msg": "hello"})
    data = backend.load()
    assert "user1" in data
    assert data["user1"][0]["msg"] == "hello"
