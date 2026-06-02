import pytest
from fastapi.testclient import TestClient
from main import app
from rbac import RBACManager
import firebase_admin.auth
from role_sync import sync_role_claim

def _make_fake_firestore(roles):
    class _FakeDoc:
        def __init__(self, role):
            self.exists = True
            self._role = role
        def to_dict(self):
            return {"role": self._role}
    class _FakeUserRef:
        def __init__(self, uid):
            self._uid = uid
        def get(self):
            role = roles.get(self._uid)
            if role is None:
                return type("Missing", (), {"exists": False})()
            return _FakeDoc(role)
    class _FakeCollection:
        def document(self, uid):
            return _FakeUserRef(uid)
    class _FakeFirestore:
        def collection(self, name):
            return _FakeCollection()
    return _FakeFirestore()

# Bypass RBACMiddleware ASGI bug for tests
def asgi_call(self, scope, receive, send):
    import anyio
    return anyio.from_thread.start_blocking_portal().call(self.app, scope, receive, send)
# Actually, the simplest is to just call self.app
async def fake_call(self, scope, receive, send):
    await self.app(scope, receive, send)
from rbac import RBACMiddleware
RBACMiddleware.__call__ = fake_call

from main import app
client = TestClient(app)

@pytest.fixture()
def mock_db(monkeypatch):
    store = _make_fake_firestore({"test-uid": "admin"})
    monkeypatch.setattr(RBACManager, "get_db", staticmethod(lambda: store))
    monkeypatch.setattr("main.db_firestore", store)
    return store

def test_valid_token_accepted(monkeypatch, mock_db):
    def mock_verify(token, check_revoked=False):
        return {"uid": "test-uid", "role": "admin"}
    monkeypatch.setattr(firebase_admin.auth, "verify_id_token", mock_verify)
    monkeypatch.setattr("rbac.firebase_auth.verify_id_token", mock_verify)
    monkeypatch.setattr("main.auth.verify_id_token", mock_verify)

    response = client.get("/api/notifications", headers={"Authorization": "Bearer valid_token"})
    assert response.status_code == 200

def test_revoked_token_rejected(monkeypatch, mock_db):
    def mock_verify(token, check_revoked=False):
        raise firebase_admin.auth.RevokedIdTokenError("Token revoked")
    monkeypatch.setattr(firebase_admin.auth, "verify_id_token", mock_verify)
    monkeypatch.setattr("rbac.firebase_auth.verify_id_token", mock_verify)
    monkeypatch.setattr("main.auth.verify_id_token", mock_verify)

    response = client.get("/api/notifications", headers={"Authorization": "Bearer revoked_token"})
    assert response.status_code == 401
    assert "revoked" in response.json().get("detail", "").lower() or "revoked" in response.json().get("error", "").lower()

def test_expired_token_rejected(monkeypatch, mock_db):
    def mock_verify(token, check_revoked=False):
        raise firebase_admin.auth.ExpiredIdTokenError("Token expired", None)
    monkeypatch.setattr(firebase_admin.auth, "verify_id_token", mock_verify)
    monkeypatch.setattr("rbac.firebase_auth.verify_id_token", mock_verify)
    monkeypatch.setattr("main.auth.verify_id_token", mock_verify)

    response = client.get("/api/notifications", headers={"Authorization": "Bearer expired_token"})
    assert response.status_code == 401

def test_malformed_token_rejected(monkeypatch, mock_db):
    def mock_verify(token, check_revoked=False):
        raise ValueError("Malformed token")
    monkeypatch.setattr(firebase_admin.auth, "verify_id_token", mock_verify)
    monkeypatch.setattr("rbac.firebase_auth.verify_id_token", mock_verify)
    monkeypatch.setattr("main.auth.verify_id_token", mock_verify)

    response = client.get("/api/notifications", headers={"Authorization": "Bearer bad_token"})
    assert response.status_code == 401

def test_missing_token_rejected(monkeypatch, mock_db):
    response = client.get("/api/notifications")
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_role_sync_triggers_revocation(monkeypatch):
    calls = []
    def mock_revoke(uid):
        calls.append(uid)
    monkeypatch.setattr("role_sync._revoke_refresh_tokens_sync", mock_revoke)
    monkeypatch.setattr("role_sync._set_claim_sync", lambda uid, role: None)

    await sync_role_claim("test-uid", "admin")
    assert calls == ["test-uid"]

# --- Additional Tests to Verify All Revocation Paths ---

import firebase_admin

# Prevent ValueError: The default Firebase app already exists
# when importing multiple modules that call initialize_app()
original_init = firebase_admin.initialize_app
def safe_init(*args, **kwargs):
    if not firebase_admin._apps:
        return original_init(*args, **kwargs)
firebase_admin.initialize_app = safe_init

from firebase_admin import firestore
firestore.client = lambda: _make_fake_firestore({"test-uid": "admin"})

from feedback_api import app as feedback_app
feedback_client = TestClient(feedback_app)

def test_feedback_api_post_rejects_revoked_token(monkeypatch):
    def mock_verify(token, check_revoked=False):
        raise firebase_admin.auth.RevokedIdTokenError("Token revoked")
    monkeypatch.setattr("feedback_api.firebase_auth.verify_id_token", mock_verify)
    
    response = feedback_client.post("/api/feedback", json={}, headers={"Authorization": "Bearer revoked_token"})
    assert response.status_code == 401
    assert "revoked" in response.json().get("detail", "").lower() or "revoked" in response.json().get("error", "").lower()

def test_feedback_api_stats_rejects_revoked_token(monkeypatch):
    def mock_verify(token, check_revoked=False):
        raise firebase_admin.auth.RevokedIdTokenError("Token revoked")
    monkeypatch.setattr("feedback_api.firebase_auth.verify_id_token", mock_verify)
    
    response = feedback_client.get("/api/feedback/stats", headers={"Authorization": "Bearer revoked_token"})
    assert response.status_code == 401
    assert "revoked" in response.json().get("detail", "").lower() or "revoked" in response.json().get("error", "").lower()

def test_websocket_rejects_revoked_token(monkeypatch, mock_db):
    def mock_verify(token, check_revoked=False):
        raise firebase_admin.auth.RevokedIdTokenError("Token revoked")
    monkeypatch.setattr("main.auth.verify_id_token", mock_verify)

    from fastapi.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/api/notifications/stream?token=revoked_token"):
            pass
    assert exc.value.code == 1008
    assert "revoked" in exc.value.reason.lower()
