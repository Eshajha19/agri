"""Crop Quality Grading Router"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field, validator
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class CropQualityGradingRequest(BaseModel):
    crop_type: str = Field(..., min_length=1, max_length=50)
    image_base64: str = Field(..., min_length=100)

    @validator("image_base64")
    def validate_image_size(cls, v):
        # 10MB maximum payload size for Base64 (10 * 1024 * 1024 * 4 / 3 ≈ 13981013 chars)
        # Cap at 14,000,000 characters to prevent Memory Exhaustion DoS
        MAX_BASE64_SIZE = 14000000
        if len(v) > MAX_BASE64_SIZE:
            raise ValueError("Image payload size exceeds the maximum limit of 10MB")
        return v

class CropQualityBatchRequest(BaseModel):
    crop_type: str = Field(..., min_length=1, max_length=50)
    images_base64: list = Field(..., min_items=1, max_items=100)

    @validator("images_base64")
    def validate_batch_images_size(cls, v):
        # 10MB limit per image, 50MB total batch size limit
        MAX_BASE64_SIZE = 14000000
        MAX_TOTAL_SIZE = 70000000
        total_size = 0
        for img in v:
            if not isinstance(img, str):
                raise ValueError("Each image in the batch must be a base64 encoded string")
            if len(img) > MAX_BASE64_SIZE:
                raise ValueError("An image payload in the batch exceeds the maximum limit of 10MB")
            total_size += len(img)
        if total_size > MAX_TOTAL_SIZE:
            raise ValueError("Total batch payload size exceeds the maximum limit of 50MB")
        return v

class QualityTrendsRequest(BaseModel):
    crop_type: str = Field(..., min_length=1, max_length=50)
    days: int = Field(default=7, ge=1, le=30)

crop_quality_grader = None
rbac_manager = None
Permission = None

def init_quality(cqg, rbac, perm):
    global crop_quality_grader, rbac_manager, Permission
    crop_quality_grader = cqg
    rbac_manager = rbac
    Permission = perm

@router.post("/assess-single")
async def assess_single_crop(request: Request, data: CropQualityGradingRequest):
    """Assess a single crop image. Requires QUALITY_ASSESS permission."""
    if crop_quality_grader is None or rbac_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await rbac_manager.raise_if_unauthorized(request, [Permission.QUALITY_ASSESS], require_all=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
    try:
        import base64
        image_bytes = base64.b64decode(data.image_base64)
        result = crop_quality_grader.assess_crop_image(image_bytes, data.crop_type)
        return {"success": True, "crop_type": data.crop_type, "assessment": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("assess_single_crop error: %s", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during quality assessment.")

@router.post("/assess-batch")
async def assess_batch_crops(request: Request, data: CropQualityBatchRequest):
    """Assess a batch of crop images. Requires QUALITY_ASSESS permission."""
    if crop_quality_grader is None or rbac_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await rbac_manager.raise_if_unauthorized(request, [Permission.QUALITY_ASSESS], require_all=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
    try:
        import base64
        from datetime import datetime

        # Decode and assess one image at a time so peak in-process memory is
        # O(single image) rather than O(total batch size).
        #
        # The previous implementation decoded all images into image_bytes_list
        # before calling batch_grade_crops, holding up to ~37.5 MB of raw
        # bytes simultaneously (50 MB base64 cap × ~0.75 decode ratio).
        # Under concurrent load this multiplied across requests.
        #
        # batch_grade_crops itself iterates images sequentially, so passing
        # a pre-built list provided no parallelism benefit — only memory cost.
        # We replicate its per-image logic here and discard each decoded bytes
        # object immediately after assessment, keeping only the result dict.
        assessments = []
        for idx, img in enumerate(data.images_base64):
            try:
                image_bytes = base64.b64decode(img)
            except Exception as decode_err:
                logger.error("Batch decode error at index %d: %s", idx, decode_err)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 encoding at image index {idx}",
                )
            try:
                assessment = crop_quality_grader.assess_crop_image(image_bytes, data.crop_type)
                from dataclasses import asdict
                assessments.append(asdict(assessment) if hasattr(assessment, '__dataclass_fields__') else assessment)
            except Exception as assess_err:
                assessments.append({
                    "error": str(assess_err),
                    "index": idx,
                    "timestamp": datetime.now().isoformat(),
                })
            finally:
                # Explicitly release the decoded bytes for this image before
                # moving to the next one.
                del image_bytes

        return {"success": True, "crop_type": data.crop_type, "batch_results": assessments}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("assess_batch_crops error: %s", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during batch assessment.")

@router.post("/trends")
async def get_quality_trends(request: Request, data: QualityTrendsRequest):
    """Get quality trends. Requires QUALITY_READ permission."""
    if crop_quality_grader is None or rbac_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await rbac_manager.raise_if_unauthorized(request, [Permission.QUALITY_READ], require_all=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
    try:
        trends = crop_quality_grader.get_quality_trends(data.crop_type, data.days)
        return {"success": True, "crop_type": data.crop_type, "days": data.days, "trends": trends}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("get_quality_trends error: %s", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred fetching quality trends.")

@router.get("/supported-crops")
async def get_supported_crops(request: Request):
    """Get supported crop types. Requires QUALITY_READ permission."""
    if crop_quality_grader is None or rbac_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await rbac_manager.raise_if_unauthorized(request, [Permission.QUALITY_READ], require_all=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
    crops = crop_quality_grader.supported_crops if hasattr(crop_quality_grader, 'supported_crops') else []
    return {"success": True, "supported_crops": crops}

@router.post("/market-price")
async def calculate_market_price(request: Request, data: CropQualityGradingRequest):
    """Calculate market price from crop image. Requires QUALITY_ASSESS permission."""
    if crop_quality_grader is None or rbac_manager is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await rbac_manager.raise_if_unauthorized(request, [Permission.QUALITY_ASSESS], require_all=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
    try:
        import base64
        image_bytes = base64.b64decode(data.image_base64)
        assessment = crop_quality_grader.assess_crop_image(image_bytes, data.crop_type)
        return {"success": True, "crop_type": data.crop_type, "grade": getattr(assessment, 'grade', 'A'), "assessment": assessment}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("calculate_market_price error: %s", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred calculating market price.")
