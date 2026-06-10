import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

firebase_admin_stub = SimpleNamespace(
    _apps=[],
    initialize_app=lambda: None,
    credentials=SimpleNamespace(),
    auth=SimpleNamespace(verify_id_token=lambda token: {"uid": token}),
    firestore=SimpleNamespace(client=lambda: None),
    storage=SimpleNamespace(bucket=lambda *args, **kwargs: None),
)

whatsapp_service_stub = SimpleNamespace(
    send_whatsapp_message=lambda *args, **kwargs: {"success": True, "status": "success"},
    format_alert_message=lambda alert_type, message: message,
)

def _make_reportlab_stub():
    reportlab_module = ModuleType("reportlab")
    lib_module = ModuleType("reportlab.lib")
    pagesizes_module = ModuleType("reportlab.lib.pagesizes")
    pdfgen_module = ModuleType("reportlab.pdfgen")
    canvas_module = ModuleType("reportlab.pdfgen.canvas")
    colors_module = ModuleType("reportlab.lib.colors")
    units_module = ModuleType("reportlab.lib.units")

    class _Canvas:
        def __init__(self, *args, **kwargs):
            pass

        def setFont(self, *args, **kwargs):
            pass

        def setFillColor(self, *args, **kwargs):
            pass

        def drawCentredString(self, *args, **kwargs):
            pass

        def setStrokeColor(self, *args, **kwargs):
            pass

        def line(self, *args, **kwargs):
            pass

        def drawString(self, *args, **kwargs):
            pass

        def showPage(self):
            pass

        def save(self):
            pass

    canvas_module.Canvas = _Canvas
    pagesizes_module.letter = (612, 792)
    colors_module.green = object()
    colors_module.black = object()
    units_module.inch = 72

    reportlab_module.lib = lib_module
    reportlab_module.pdfgen = pdfgen_module
    lib_module.pagesizes = pagesizes_module
    lib_module.colors = colors_module
    lib_module.units = units_module
    pdfgen_module.canvas = canvas_module

    return {
        "reportlab": reportlab_module,
        "reportlab.lib": lib_module,
        "reportlab.lib.pagesizes": pagesizes_module,
        "reportlab.pdfgen": pdfgen_module,
        "reportlab.pdfgen.canvas": canvas_module,
        "reportlab.lib.colors": colors_module,
        "reportlab.lib.units": units_module,
    }

sys.modules.setdefault("firebase_admin", firebase_admin_stub)
sys.modules.setdefault("firebase_admin.credentials", firebase_admin_stub.credentials)
sys.modules.setdefault("firebase_admin.auth", firebase_admin_stub.auth)
sys.modules.setdefault("firebase_admin.firestore", firebase_admin_stub.firestore)
sys.modules.setdefault("firebase_admin.storage", firebase_admin_stub.storage)
sys.modules.setdefault("whatsapp_service", whatsapp_service_stub)
sys.modules.setdefault("cv2", ModuleType("cv2"))
sys.modules.setdefault("aiohttp", ModuleType("aiohttp"))
joblib_stub = ModuleType("joblib")
joblib_stub.load = lambda *args, **kwargs: None
joblib_stub.dump = lambda *args, **kwargs: None

sys.modules.setdefault("joblib", joblib_stub)
sys.modules.update(_make_reportlab_stub())

import main
from rbac_audit import rbac_audit_trail, validate_required_roles
from rbac import AuthContext, RBACManager


class _FakeDoc:
    def __init__(self, role):
        self._role = role

    @property
    def exists(self):
        return True

    def to_dict(self):
        return {"role": self._role}


class _FakeDocumentRef:
    def __init__(self, role):
        self._role = role

    def get(self):
        return _FakeDoc(self._role)


class _FakeCollection:
    def __init__(self, roles):
        self._roles = roles

    def document(self, uid):
        return _FakeDocumentRef(self._roles.get(uid, "farmer"))


class _FakeFirestore:
    def __init__(self, roles):
        self._roles = roles

    def collection(self, name):
        return _FakeCollection(self._roles)


@pytest.fixture(autouse=True)
def _reset_audit_trail(tmp_path, monkeypatch):
    rbac_audit_trail.clear()
    monkeypatch.setattr(rbac_audit_trail, "log_path", Path(tmp_path / "rbac_audit.jsonl"))
    yield
    rbac_audit_trail.clear()


@pytest.fixture()
def client(monkeypatch):
    role_map = {
        "admin-user": "admin",
        "expert-user": "expert",
        "farmer-user": "farmer",
    }

    async def _fake_resolve_auth_context(request, allow_unauthenticated=False):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.replace("Bearer ", "").strip()
        if not token:
            raise main.HTTPException(status_code=401, detail="Missing or invalid authentication token")
        role = role_map.get(token, "farmer")
        return AuthContext(uid=token, role=role, roles=(role,), tenant_id="tenant-1")

    monkeypatch.setattr(main.RBACManager, "resolve_auth_context", _fake_resolve_auth_context)
    return TestClient(main.app)


def test_validate_required_roles_normalizes_and_deduplicates():
    assert validate_required_roles(["Admin", "expert", "admin"]) == ["admin", "expert"]


def test_validate_required_roles_rejects_unknown_role():
    with pytest.raises(ValueError):
        validate_required_roles(["pilot"])


def test_admin_audit_endpoint_records_allowed_access(monkeypatch):
    async def _fake_resolve_auth_context(request, allow_unauthenticated=False):
        return AuthContext(uid="admin-user", role="admin", roles=("admin",), tenant_id="tenant-1")

    monkeypatch.setattr(main.RBACManager, "resolve_auth_context", _fake_resolve_auth_context)

    async def _call():
        request = SimpleNamespace(method="GET", url=SimpleNamespace(path="/api/admin/rbac-audit"), headers={}, client=SimpleNamespace(host="127.0.0.1"))
        return await main.verify_role(request, required_roles=["admin", "expert"])

    import asyncio

    token_data = asyncio.run(_call())
    assert token_data["role"] == "admin"

    events = rbac_audit_trail.snapshot(limit=10)
    assert events
    assert events[-1]["outcome"] == "allowed"
    assert events[-1]["required_roles"] == ["admin", "expert"]


def test_admin_audit_endpoint_records_denied_access(monkeypatch):
    async def _fake_resolve_auth_context(request, allow_unauthenticated=False):
        return AuthContext(uid="farmer-user", role="farmer", roles=("farmer",), tenant_id="tenant-1")

    monkeypatch.setattr(main.RBACManager, "resolve_auth_context", _fake_resolve_auth_context)

    async def _call():
        request = SimpleNamespace(method="GET", url=SimpleNamespace(path="/api/admin/rbac-audit"), headers={}, client=SimpleNamespace(host="127.0.0.1"))
        return await main.verify_role(request, required_roles=["admin", "expert"])

    import asyncio

    with pytest.raises(main.HTTPException) as exc:
        asyncio.run(_call())

    assert exc.value.status_code == 403
    events = rbac_audit_trail.snapshot(limit=10)
    assert events
    assert events[-1]["outcome"] == "denied"
    assert events[-1]["reason"] == "insufficient_permissions"
    assert events[-1]["role"] == "farmer"


def test_verify_role_returns_roles_and_tenant(client, monkeypatch):
    async def _fake_resolve_auth_context(request, allow_unauthenticated=False):
        return AuthContext(
            uid="admin-user",
            role="admin",
            roles=("admin", "expert"),
            tenant_id="tenant-1",
        )

    monkeypatch.setattr(main.RBACManager, "resolve_auth_context", _fake_resolve_auth_context)

    async def _call():
        request = SimpleNamespace(method="GET", url=SimpleNamespace(path="/x"), headers={}, client=SimpleNamespace(host="127.0.0.1"))
        return await main.verify_role(request, required_roles=["admin"], required_tenant_id="tenant-1")

    import asyncio

    token_data = asyncio.run(_call())
    assert token_data["uid"] == "admin-user"
    assert token_data["role"] == "admin"
    assert token_data["roles"] == ["admin", "expert"]
    assert token_data["tenant_id"] == "tenant-1"


def test_verify_role_enforces_required_tenant(client, monkeypatch):
    async def _fake_resolve_auth_context(request, allow_unauthenticated=False):
        return AuthContext(
            uid="expert-user",
            role="expert",
            roles=("expert",),
            tenant_id="tenant-a",
        )

    monkeypatch.setattr(main.RBACManager, "resolve_auth_context", _fake_resolve_auth_context)

    async def _call():
        request = SimpleNamespace(method="GET", url=SimpleNamespace(path="/x"), headers={}, client=SimpleNamespace(host="127.0.0.1"))
        return await main.verify_role(request, required_roles=["expert"], required_tenant_id="tenant-b")

    with pytest.raises(main.HTTPException) as exc:
        import asyncio

        asyncio.run(_call())

    assert exc.value.status_code == 403


def test_admin_expert_override_helper_respects_tenant_boundary():
    admin_ctx = AuthContext(uid="u1", role="admin", roles=("admin",), tenant_id="tenant-1")
    expert_ctx = AuthContext(uid="u2", role="expert", roles=("expert",), tenant_id="tenant-2")

    assert RBACManager.can_admin_or_expert_override(admin_ctx, resource_tenant_id="tenant-1") is True
    assert RBACManager.can_admin_or_expert_override(expert_ctx, resource_tenant_id="tenant-1") is False