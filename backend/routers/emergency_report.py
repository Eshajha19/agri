from fastapi import APIRouter, UploadFile, File, Form
from datetime import datetime
import os

router = APIRouter()

UPLOAD_DIR = "uploads/emergency_reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/api/emergency-report")
async def emergency_report(
    event: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    photo: UploadFile = File(...),
):
    # Save photo securely
    file_path = os.path.join(UPLOAD_DIR, f"{datetime.utcnow().isoformat()}_{photo.filename}")
    with open(file_path, "wb") as f:
        f.write(await photo.read())

    # Build report object
    report = {
        "event": event,
        "location": {"lat": lat, "lon": lon},
        "photo_path": file_path,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "Pending Review"
    }

    # TODO: Save to database (Postgres, Mongo, etc.)
    # db.insert("emergency_reports", report)

    return {"message": "Emergency report submitted", "report": report}
