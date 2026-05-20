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
sys.modules.setdefault("whatsapp_service", whatsapp_service_stub)
joblib_stub = ModuleType("joblib")
joblib_stub.load = lambda *args, **kwargs: None
joblib_stub.dump = lambda *args, **kwargs: None

numpy_stub = ModuleType("numpy")
numpy_stub.ndarray = type("ndarray", (), {})
numpy_stub.array = lambda *args, **kwargs: []
numpy_stub.asarray = lambda *args, **kwargs: []

pandas_stub = ModuleType("pandas")
pandas_stub.DataFrame = type("DataFrame", (), {})
pandas_stub.Series = type("Series", (), {})

sys.modules.setdefault("joblib", joblib_stub)
sys.modules.setdefault("numpy", numpy_stub)
sys.modules.setdefault("pandas", pandas_stub)
sys.modules.update(_make_reportlab_stub())

import main
from rbac_audit import rbac_audit_trail, validate_required_roles


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
    monkeypatch.setattr(main.firebase_auth, "verify_id_token", lambda token: {"uid": token})
    monkeypatch.setattr(
        main,
        "db_firestore",
        _FakeFirestore(
            {
                "admin-user": "admin",
                "expert-user": "expert",
                "farmer-user": "farmer",
            }
        ),
    )
    return TestClient(main.app)


def test_validate_required_roles_normalizes_and_deduplicates():
    assert validate_required_roles(["Admin", "expert", "admin"]) == ["admin", "expert"]


def test_validate_required_roles_rejects_unknown_role():
    with pytest.raises(ValueError):
        validate_required_roles(["pilot"])


def test_admin_audit_endpoint_records_allowed_access(client):
    response = client.get(
        "/api/admin/rbac-audit",
        headers={"Authorization": "Bearer admin-user"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]
    assert payload["data"][-1]["outcome"] == "allowed"
    assert payload["data"][-1]["required_roles"] == ["admin", "expert"]


def test_admin_audit_endpoint_records_denied_access(client):
    response = client.get(
        "/api/admin/rbac-audit",
        headers={"Authorization": "Bearer farmer-user"},
    )

    assert response.status_code == 403
    events = rbac_audit_trail.snapshot(limit=10)
    assert events
    assert events[-1]["outcome"] == "denied"
    assert events[-1]["reason"] == "insufficient_permissions"
    assert events[-1]["role"] == "farmer"