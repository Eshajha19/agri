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
    if crop_quality_grader is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        import base64
        image_bytes = base64.b64decode(data.image_base64)
        result = crop_quality_grader.assess_crop_image(data.crop_type, image_bytes)
        return {"success": True, "crop_type": data.crop_type, "assessment": result}
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/assess-batch")
async def assess_batch_crops(request: Request, data: CropQualityBatchRequest):
    if crop_quality_grader is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        import base64
        image_bytes_list = [base64.b64decode(img) for img in data.images_base64]
        results = crop_quality_grader.batch_grade_crops(data.crop_type, image_bytes_list)
        return {"success": True, "crop_type": data.crop_type, "batch_results": results}
    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/trends")
async def get_quality_trends(request: Request, data: QualityTrendsRequest):
    if crop_quality_grader is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        trends = crop_quality_grader.get_quality_trends(data.crop_type, data.days)
        return {"success": True, "crop_type": data.crop_type, "days": data.days, "trends": trends}
    except Exception as e:
        logger.error(f"Trends error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/supported-crops")
async def get_supported_crops(request: Request):
    if crop_quality_grader is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    crops = crop_quality_grader.supported_crops if hasattr(crop_quality_grader, 'supported_crops') else []
    return {"success": True, "supported_crops": crops}

@router.post("/market-price")
async def calculate_market_price(request: Request, data: CropQualityGradingRequest):
    if crop_quality_grader is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        import base64
        image_bytes = base64.b64decode(data.image_base64)
        assessment = crop_quality_grader.assess_crop_image(data.crop_type, image_bytes)
        return {"success": True, "crop_type": data.crop_type, "grade": getattr(assessment, 'grade', 'A'), "assessment": assessment}
    except Exception as e:
        logger.error(f"Price error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
