import pytest
from pydantic import ValidationError
from main import CropQualityGradingRequest, CropQualityBatchRequest, GeminiImageRequest

def test_single_crop_valid_base64():
    # 100 character valid string (min length is 100)
    valid_b64 = "A" * 150
    req = CropQualityGradingRequest(crop_type="tomato", image_base64=valid_b64)
    assert req.image_base64 == valid_b64

def test_single_crop_oversized_base64():
    # 14,000,001 characters (exceeds the 14,000,000 limit)
    oversized_b64 = "A" * 14000001
    with pytest.raises(ValidationError) as exc_info:
        CropQualityGradingRequest(crop_type="tomato", image_base64=oversized_b64)
    assert "Image payload size exceeds the maximum limit of 10MB" in str(exc_info.value)

def test_batch_crop_valid_base64():
    img1 = "A" * 200
    img2 = "B" * 200
    req = CropQualityBatchRequest(crop_type="tomato", images_base64=[img1, img2])
    assert req.images_base64 == [img1, img2]

def test_batch_crop_oversized_individual_image():
    oversized_b64 = "A" * 14000001
    with pytest.raises(ValidationError) as exc_info:
        CropQualityBatchRequest(crop_type="tomato", images_base64=[oversized_b64])
    assert "An image payload in the batch exceeds the maximum limit of 10MB" in str(exc_info.value)

def test_batch_crop_oversized_total_payload():
    # 5 images of 10,000,000 characters each (total 50,000,000, which is fine)
    # 8 images of 10,000,000 characters each (total 80,000,000, exceeds 70,000,000 limit)
    images = ["A" * 10000000 for _ in range(8)]
    with pytest.raises(ValidationError) as exc_info:
        CropQualityBatchRequest(crop_type="tomato", images_base64=images)
    assert "Total batch payload size exceeds the maximum limit of 50MB" in str(exc_info.value)

def test_gemini_image_valid_base64():
    valid_b64 = "A" * 100
    req = GeminiImageRequest(image_base64=valid_b64, mime_type="image/png", prompt="Analyze this plant")
    assert req.image_base64 == valid_b64

def test_gemini_image_oversized_base64():
    oversized_b64 = "A" * 14000001
    with pytest.raises(ValidationError) as exc_info:
        GeminiImageRequest(image_base64=oversized_b64, mime_type="image/png", prompt="Analyze this plant")
    assert "Image payload size exceeds the maximum limit of 10MB" in str(exc_info.value)
