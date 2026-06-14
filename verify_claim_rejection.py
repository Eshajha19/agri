import sys
import os
import asyncio

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.routers import insurance

# Mock Request class
class MockRequest:
    pass

# Mock UploadFile class
class MockUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content
        self.content_type = "image/jpeg"

    async def read(self):
        return self.content

async def run_verification():
    print("=" * 80)
    print("INSURANCE CLAIM REJECTION MODULE VERIFICATION")
    print("=" * 80)

    # Mock verify_role_fn
    async def fake_verify(request):
        return {"uid": "test-uid-123"}
    insurance.init_insurance(fake_verify)

    # 1. Test normal submission
    print("\n[TEST 1] Submitting a normal claim...")
    img = MockUploadFile("damage.jpg", b"fake image data")
    req = MockRequest()
    
    res = await insurance.submit_insurance_claim(
        request=req,
        farmer_name="Aman",
        crop_type="Rice",
        season="Kharif",
        location="Punjab",
        farm_area="2.5 acres",
        damage_cause="Flood",
        images=[img],
        simulate_rejection="false"
    )
    
    assert res["success"] is True
    assert res["claim"]["status"] == "Submitted"
    assert res["claim"]["rejection_details"] is None
    print("[OK] Normal claim submission successful (Status: Submitted)")

    # 2. Test rejected submission
    print("\n[TEST 2] Submitting a claim simulating rejection...")
    res_rejected = await insurance.submit_insurance_claim(
        request=req,
        farmer_name="Aman",
        crop_type="Rice",
        season="Kharif",
        location="Punjab",
        farm_area="2.5 acres",
        damage_cause="Flood",
        images=[img],
        simulate_rejection="true"
    )
    
    assert res_rejected["success"] is True
    claim = res_rejected["claim"]
    assert claim["status"] == "Rejected"
    assert len(claim["rejection_details"]) == 3
    print("[OK] Rejected claim simulation successful (Status: Rejected)")
    print("[OK] Rejection details present:")
    for d in claim["rejection_details"]:
        print(f"  - Reason: {d['reason']}")
        print(f"    Recommended Action: {d['recommended_action']}")

    # 3. Test list claims
    print("\n[TEST 3] Listing user claims...")
    list_res = await insurance.list_claims(request=req)
    assert list_res["success"] is True
    assert len(list_res["claims"]) >= 2
    print(f"[OK] Retrieved {len(list_res['claims'])} claims successfully")

    # 4. Test PDF generation with rejection details
    print("\n[TEST 4] Generating PDF report for rejected claim...")
    pdf_bytes = insurance._generate_claim_pdf(claim)
    assert len(pdf_bytes) > 0
    print(f"[OK] PDF report generated successfully ({len(pdf_bytes)} bytes)")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_verification())
