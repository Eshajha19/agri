from typing import Dict, List, Any
import os, json

class StorageBackend:
    def load(self) -> Dict[str, List[Dict[str, Any]]]:
        raise NotImplementedError

    def save(self, history: Dict[str, List[Dict[str, Any]]]) -> None:
        raise NotImplementedError


class FileStorage(StorageBackend):
    def __init__(self, path: str):
        self.path = path

    def load(self) -> Dict[str, List[Dict[str, Any]]]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.exception("Firestore operation failed: %s", exc)
            return {}


    def save(self, history: Dict[str, List[Dict[str, Any]]]) -> None:
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)


class FirestoreStorage(StorageBackend):
    def __init__(self):
        import firebase_admin
        from firebase_admin import firestore
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        self.client = firestore.client()

    def load(self) -> Dict[str, List[Dict[str, Any]]]:
        docs = self.client.collection("sustainability_history").stream()
        history = {}
        for doc in docs:
            history[doc.id] = doc.to_dict().get("records", [])
        return history

    def save(self, history: Dict[str, List[Dict[str, Any]]]) -> None:
        for user_id, records in history.items():
            self.client.collection("sustainability_history").document(user_id).set({"records": records})
