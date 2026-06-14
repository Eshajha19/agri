"""
Regional Benchmarking Router
============================
Federated regional yield benchmarking with statistical significance testing,
peer comparison, and automated PDF report generation.

Endpoints
---------
GET  /api/benchmark/regional-stats      — aggregated yield stats by region and crop
POST /api/benchmark/farmer-percentile   — compute farmer's rank within cohort
POST /api/benchmark/significance-test   — run statistical test against baseline
POST /api/benchmark/report/generate   — generate and sign PDF report
GET  /api/benchmark/report/{report_id} — download previously generated report
"""

import hashlib
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/benchmark", tags=["regional benchmarking"])

_DB = None
_VERIFY_ROLE_FN = None
_GET_SIGNING_KEYS_FN = None


def init_regional_benchmarking(db_client, verify_role_fn, get_signing_keys_fn):
    global _DB, _VERIFY_ROLE_FN, _GET_SIGNING_KEYS_FN
    _DB = db_client
    _VERIFY_ROLE_FN = verify_role_fn
    _GET_SIGNING_KEYS_FN = get_signing_keys_fn


def _get_db():
    if _DB is not None:
        return _DB
    try:
        import firebase_admin
        from firebase_admin import firestore
        if firebase_admin._apps:
            return firestore.client()
    except Exception:
        pass
    return None


class PercentileRequest(BaseModel):
    farmer_yield: float = Field(..., gt=0)
    region: str = Field(..., min_length=1, max_length=50)
    crop_type: str = Field(..., min_length=1, max_length=50)


class SignificanceRequest(BaseModel):
    farmer_yield: float = Field(..., gt=0)
    region: str = Field(..., min_length=1, max_length=50)
    crop_type: str = Field(..., min_length=1, max_length=50)


class ReportRequest(BaseModel):
    farmer_yield: float = Field(..., gt=0)
    region: str = Field(..., min_length=1, max_length=50)
    crop_type: str = Field(..., min_length=1, max_length=50)


# =============================================================================
# PDF HELPERS (mirror reports.py pattern)
# =============================================================================

def _build_benchmark_pdf(report_data: dict, signature_hex: str, cert_id: str) -> bytes:
    """Render a regional benchmark report as a signed PDF."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Header
    c.setFillColor(colors.HexColor("#2E7D32"))
    c.rect(0, height - 80, width, 80, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(inch, height - 45, "Fasal Saathi")
    c.setFont("Helvetica", 12)
    c.drawString(inch, height - 65, "Regional Yield Benchmark Report")

    # Cert ID
    c.setFillColor(colors.HexColor("#1B5E20"))
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(width - inch, height - 95, f"Report ID: {cert_id}")
    c.drawRightString(width - inch, height - 110, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    y = height - 150

    # Farmer Performance
    c.setFillColor(colors.HexColor("#2E7D32"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "Farmer Performance")
    c.line(inch, y - 4, width - inch, y - 4)

    y -= 24
    c.setFont("Helvetica", 11)
    perf = report_data.get("percentile", {})
    sig = report_data.get("significance_test", {})

    fields = [
        ("Your Yield", f"₹{report_data.get('farmer_yield', 0):,.2f} /quintal proxy"),
        ("Region", report_data.get("region", "Unknown")),
        ("Crop Type", report_data.get("crop_type", "Unknown")),
        ("Percentile Rank", f"{perf.get('percentile', 0):.1f}% (Rank {perf.get('rank', 0)}/{perf.get('cohort_size', 0)})"),
        ("Statistical Significance", "Significant" if sig.get('overall_significant') else "Not Significant"),
        ("Interpretation", sig.get('interpretation', 'unknown').replace('_', ' ').title()),
    ]

    for label, value in fields:
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(inch, y, f"{label}:")
        c.setFillColor(colors.black)
        c.drawString(3 * inch, y, str(value))
        y -= 20

    # Regional Aggregates
    y -= 20
    c.setFillColor(colors.HexColor("#2E7D32"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "Regional Baseline")
    c.line(inch, y - 4, width - inch, y - 4)

    y -= 24
    agg = report_data.get("regional_aggregates", {})
    agg_fields = [
        ("Cohort Size", str(agg.get("sample_size", 0))),
        ("Mean Yield", f"₹{agg.get('mean_yield', 0):,.2f}"),
        ("Median Yield", f"₹{agg.get('median_yield', 0):,.2f}"),
        ("Std Deviation", f"₹{agg.get('std_yield', 0):,.2f}"),
        ("90% CI Lower", f"₹{agg.get('confidence_interval', {}).get('lower', 0):,.2f}"),
        ("90% CI Upper", f"₹{agg.get('confidence_interval', {}).get('upper', 0):,.2f}"),
        ("CV", str(agg.get("coefficient_of_variation", 0))),
    ]

    c.setFont("Helvetica", 11)
    for label, value in agg_fields:
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(inch, y, f"{label}:")
        c.setFillColor(colors.black)
        c.drawString(3 * inch, y, str(value))
        y -= 20

    # Top Performers
    y -= 20
    c.setFillColor(colors.HexColor("#2E7D32"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "Top Performers in Region")
    c.line(inch, y - 4, width - inch, y - 4)

    y -= 24
    c.setFont("Helvetica", 11)
    top = report_data.get("top_performers", [])
    for i, t in enumerate(top[:5]):
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(inch, y, f"Rank {t.get('rank', i+1)}:")
        c.setFillColor(colors.black)
        c.drawString(3 * inch, y, f"Yield proxy ₹{t.get('yield_proxy', 0):,.2f} — {t.get('season', 'Unknown')}")
        y -= 18

    # Signature
    y -= 20
    c.setFillColor(colors.HexColor("#2E7D32"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "Cryptographic Signature (Ed25519)")
    c.line(inch, y - 4, width - inch, y - 4)

    y -= 24
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#333333"))
    chunk_size = 64
    sig_lines = [signature_hex[i:i + chunk_size] for i in range(0, len(signature_hex), chunk_size)]
    for line in sig_lines:
        c.drawString(inch, y, line)
        y -= 12

    # Footer
    c.setFillColor(colors.HexColor("#EEEEEE"))
    c.rect(0, 0, width, 50, fill=True, stroke=False)
    c.setFillColor(colors.HexColor("#555555"))
    c.setFont("Helvetica", 8)
    c.drawCentredString(width / 2, 30, "This document is cryptographically signed and cannot be altered after generation.")
    c.drawCentredString(width / 2, 18, "Verify authenticity at: https://fasalsaathi.in/verify")

    c.save()
    return buf.getvalue()


def _sign_benchmark_report(private_key: Ed25519PrivateKey, report_data: dict, cert_id: str) -> str:
    """Return hex-encoded Ed25519 signature over canonical report payload."""
    payload = json.dumps({
        "cert_id": cert_id,
        "report_id": report_data.get("report_id"),
        "farmer_yield": report_data.get("farmer_yield"),
        "region": report_data.get("region"),
        "crop_type": report_data.get("crop_type"),
        "percentile": report_data.get("percentile", {}).get("percentile"),
        "generated_at": report_data.get("generated_at"),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")

    signature_bytes = private_key.sign(payload)
    return signature_bytes.hex()


def _make_cert_id(report_id: str) -> str:
    raw = f"{report_id}|{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10].upper()
    return f"BENCH-{digest}"


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/regional-stats")
async def get_regional_stats(region: Optional[str] = None):
    """
    Aggregated yield statistics by region and crop.
    """
    try:
        db = _get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        from ml.regional_analytics import get_regional_analytics

        analytics = get_regional_analytics()
        result = analytics.aggregate_by_region_crop(db, region)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "success": True,
            "data": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Regional stats failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/farmer-percentile")
async def get_farmer_percentile(request: Request, data: PercentileRequest):
    """
    Compute farmer's percentile rank within regional cohort.
    """
    try:
        db = _get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        from ml.regional_analytics import get_regional_analytics

        analytics = get_regional_analytics()
        result = analytics.compute_percentile(db, data.farmer_yield, data.region, data.crop_type)

        if result is None:
            raise HTTPException(status_code=400, detail="Insufficient cohort data for percentile computation")

        return {
            "success": True,
            "data": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Percentile computation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/significance-test")
async def run_significance_test(request: Request, data: SignificanceRequest):
    """
    Run t-test and Mann-Whitney U test against regional baseline.
    """
    try:
        db = _get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        from ml.regional_analytics import get_regional_analytics

        analytics = get_regional_analytics()
        result = analytics.significance_test(db, data.farmer_yield, data.region, data.crop_type)

        if result is None:
            raise HTTPException(status_code=400, detail="Insufficient cohort data for significance testing")

        return {
            "success": True,
            "data": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Significance test failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/generate")
async def generate_benchmark_report(request: Request, data: ReportRequest):
    """
    Generate and sign PDF benchmark report.
    """
    try:
        if not all([_VERIFY_ROLE_FN, _GET_SIGNING_KEYS_FN]):
            raise HTTPException(status_code=500, detail="Not initialized")

        token_data = await _VERIFY_ROLE_FN(request)
        farmer_uid = token_data["uid"]

        db = _get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        from ml.regional_analytics import get_regional_analytics

        analytics = get_regional_analytics()
        report = analytics.generate_report(db, farmer_uid, data.farmer_yield, data.region, data.crop_type)

        if not report:
            raise HTTPException(status_code=500, detail="Failed to generate report")

        # Sign
        private_key = _GET_SIGNING_KEYS_FN()
        cert_id = _make_cert_id(report["report_id"])
        signature_hex = _sign_benchmark_report(private_key, report, cert_id)
        pdf_bytes = _build_benchmark_pdf(report, signature_hex, cert_id)

        # Save PDF
        pdf_path = Path("regional_reports") / f"{report['report_id']}.pdf"
        pdf_path.parent.mkdir(exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Save ownership metadata
        metadata_path = Path("regional_reports") / f"{report['report_id']}.meta.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "owner_uid": farmer_uid,
                    "report_id": report["report_id"],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
            )

        return {
            "success": True,
            "report_id": report["report_id"],
            "cert_id": cert_id,
            "download_url": f"/api/benchmark/report/{report['report_id']}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Report generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/{report_id}")
async def download_report(request: Request, report_id: str):
    """
    Download previously generated benchmark report.
    """
    try:
        if not _VERIFY_ROLE_FN:
            raise HTTPException(
                status_code=500,
                detail="Authentication service unavailable"
            )

        token_data = await _VERIFY_ROLE_FN(request)
        uid = token_data["uid"]

        metadata_path = Path("regional_reports") / f"{report_id}.meta.json"

        if not metadata_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Report metadata not found"
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        owner_uid = metadata.get("owner_uid")

        if owner_uid != uid:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this report"
            )
        pdf_path = Path("regional_reports") / f"{report_id}.pdf"
        if not pdf_path.exists():
            # Try JSON fallback
            json_path = Path("regional_reports") / f"{report_id}.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "success": True,
                    "report": data,
                    "note": "PDF not found, returning JSON data",
                }
            raise HTTPException(status_code=404, detail="Report not found")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="FasalSaathi_Benchmark_{report_id}.pdf"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Report download failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))