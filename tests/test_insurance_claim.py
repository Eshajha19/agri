import pytest
from fastapi.testclient import TestClient
from backend.routers import insurance
import io

@pytest.fixture(autouse=True)
def mock_auth(monkeypatch):
    """Fixture to bypass authentication during testing."""
    async def fake_verify(request):
        return {"uid": "test-uid-123"}
    monkeypatch.setattr(insurance, "verify_role_fn", fake_verify)

def test_submit_claim_success(client: TestClient):
    """Test submitting a claim normally returns status 'Submitted'."""
    data = {
        "farmer_name": "John Doe",
        "crop_type": "Wheat",
        "season": "Rabi",
        "location": "Punjab",
        "farm_area": "5 acres",
        "damage_cause": "Drought",
    }
    # Create a dummy image file for upload
    files = [("images", ("test.jpg", io.BytesIO(b"dummy image content"), "image/jpeg"))]

    response = client.post("/api/insurance/claim", data=data, files=files)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    assert res_data["claim"]["status"] == "Submitted"
    assert res_data["claim"]["rejection_details"] is None

def test_submit_claim_rejected(client: TestClient):
    """Test submitting a claim with simulate_rejection=true returns status 'Rejected'."""
    data = {
        "farmer_name": "John Doe",
        "crop_type": "Wheat",
        "season": "Rabi",
        "location": "Punjab",
        "farm_area": "5 acres",
        "damage_cause": "Drought",
        "simulate_rejection": "true",
    }
    files = [("images", ("test.jpg", io.BytesIO(b"dummy image content"), "image/jpeg"))]

    response = client.post("/api/insurance/claim", data=data, files=files)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    assert res_data["claim"]["status"] == "Rejected"
    
    # Assert multiple rejection reasons are returned
    details = res_data["claim"]["rejection_details"]
    assert isinstance(details, list)
    assert len(details) == 3
    assert details[0]["reason"] == "Missing land ownership proof"
    assert details[0]["recommended_action"] == "Upload Khasra document (Land record details)"

def test_list_claims(client: TestClient):
    """Test retrieving user's claims list."""
    # Retrieve claims
    response = client.get("/api/insurance/claims")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    assert isinstance(res_data["claims"], list)

def test_export_pdf_report(client: TestClient):
    """Test downloading the PDF report for a claim."""
    data = {
        "farmer_name": "Jane Doe",
        "crop_type": "Rice",
        "season": "Kharif",
        "location": "Haryana",
        "farm_area": "10 acres",
        "damage_cause": "Flood",
        "simulate_rejection": "true",
    }
    files = [("images", ("test.jpg", io.BytesIO(b"dummy image content"), "image/jpeg"))]

    post_resp = client.post("/api/insurance/claim", data=data, files=files)
    claim_id = post_resp.json()["claim"]["claim_id"]

    # Export PDF
    pdf_resp = client.get(f"/api/insurance/claim/{claim_id}/export")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert len(pdf_resp.content) > 0
