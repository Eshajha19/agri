from collections import OrderedDict
import threading
from backend.storage_backend import FileStorage, StorageBackend

class SustainabilityAnalytics:
    def __init__(self, backend: StorageBackend = None) -> None:
        self.backend = backend or FileStorage("sustainability_history.json")
        self._history: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict(self.backend.load())
        self._history_lock = threading.Lock()

    def _append_history(self, user_id: str, record: Dict[str, Any]) -> None:
        with self._history_lock:
            if user_id not in self._history:
                self._history[user_id] = []
            self._history[user_id].append(record)
            self.backend.save(self._history)   # ✅ single persistence call
