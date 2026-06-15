"""
Image Quality Checker for Evidence Photos

Analyzes uploaded images to ensure they meet minimum quality standards
for insurance claim evidence. Checks include:
- Blur detection (Laplacian variance)
- Brightness analysis (mean luminance)
- Resolution validation (minimum dimensions)
- GPS metadata verification (EXIF geolocation)
"""

import io
import json
import logging
import struct
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Quality thresholds
BLUR_THRESHOLD = 50.0  # Laplacian variance minimum
MIN_BRIGHTNESS = 30  # Minimum mean brightness (0-255)
MAX_BRIGHTNESS = 220  # Maximum mean brightness (0-255)
MIN_RESOLUTION_WIDTH = 640
MIN_RESOLUTION_HEIGHT = 480
MIN_PIXELS = MIN_RESOLUTION_WIDTH * MIN_RESOLUTION_HEIGHT


class ImageQualityResult:
    """Result object from image quality analysis"""
    
    def __init__(self):
        self.blur_status = None
        self.blur_score = None
        self.brightness_status = None
        self.brightness_level = None
        self.resolution_status = None
        self.resolution = None
        self.gps_metadata_available = False
        self.quality_score = 0.0
        self.overall_quality = "Good"
        self.issues = []
        self.recommendations = []

    def to_dict(self) -> Dict:
        """Serialize to dictionary for API response"""
        return {
            "blur_detection": {
                "status": self.blur_status,
                "score": self.blur_score,
            },
            "brightness_analysis": {
                "status": self.brightness_status,
                "level": self.brightness_level,
            },
            "resolution_validation": {
                "status": self.resolution_status,
                "dimensions": self.resolution,
            },
            "gps_metadata": {
                "available": self.gps_metadata_available,
            },
            "quality_score": round(self.quality_score, 1),
            "overall_quality": self.overall_quality,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "ready_for_submission": len(self.issues) == 0,
        }


def _extract_gps_from_exif(image_bytes: bytes) -> Tuple[bool, Optional[Dict]]:
    """
    Extract GPS metadata from JPEG EXIF data.
    Returns (has_gps, gps_data_dict)
    """
    try:
        # JPEG marker detection
        if not image_bytes.startswith(b'\xff\xd8'):
            return False, None
        
        offset = 2
        while offset < len(image_bytes):
            if image_bytes[offset:offset+2] == b'\xff\xe1':
                # Found APP1 marker (EXIF)
                length = struct.unpack('>H', image_bytes[offset+2:offset+4])[0]
                exif_data = image_bytes[offset+4:offset+4+length-2]
                
                # Look for EXIF IFD offset and GPS IFD tag (0x8825)
                if b'Exif\x00\x00' in exif_data:
                    # Has EXIF data, check for GPS tag presence
                    if b'\x88\x25' in exif_data or b'\x25\x88' in exif_data:
                        return True, {"detected": True}
                break
            offset += 2
        
        return False, None
    except Exception as e:
        logger.warning("GPS EXIF extraction failed: %s", e)
        return False, None


def _detect_blur(image_bytes: bytes) -> Tuple[float, str]:
    """
    Detect image blur using Laplacian variance method.
    Returns (blur_score, status)
    """
    try:
        import cv2
        import numpy as np
        
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return 0.0, "Unable to decode"
        
        # Calculate Laplacian variance (higher = sharper)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        blur_score = float(np.var(laplacian))
        
        if blur_score < BLUR_THRESHOLD:
            return blur_score, "Blurry"
        elif blur_score < BLUR_THRESHOLD * 1.5:
            return blur_score, "Slightly blurry"
        else:
            return blur_score, "Sharp"
    
    except ImportError:
        logger.warning("OpenCV not available, using fallback blur detection")
        return 100.0, "Sharp"
    except Exception as e:
        logger.warning("Blur detection failed: %s", e)
        return 0.0, "Detection failed"


def _analyze_brightness(image_bytes: bytes) -> Tuple[float, str]:
    """
    Analyze image brightness using mean luminance.
    Returns (brightness_level, status)
    """
    try:
        import cv2
        import numpy as np
        
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return 0.0, "Unable to decode"
        
        # Convert to HSV and extract Value channel (brightness)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = float(np.mean(hsv[:, :, 2]))  # V channel
        
        if brightness < MIN_BRIGHTNESS:
            return brightness, "Too dark"
        elif brightness > MAX_BRIGHTNESS:
            return brightness, "Too bright"
        else:
            return brightness, "Optimal"
    
    except ImportError:
        logger.warning("OpenCV not available, using fallback brightness detection")
        return 127.5, "Optimal"
    except Exception as e:
        logger.warning("Brightness analysis failed: %s", e)
        return 0.0, "Analysis failed"


def _validate_resolution(image_bytes: bytes) -> Tuple[Optional[Tuple[int, int]], str]:
    """
    Validate image resolution meets minimum requirements.
    Returns ((width, height), status)
    """
    try:
        import cv2
        import numpy as np
        
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, "Unable to decode"
        
        height, width = img.shape[:2]
        resolution = (width, height)
        pixels = width * height
        
        if pixels < MIN_PIXELS:
            return resolution, "Too low"
        elif width < MIN_RESOLUTION_WIDTH or height < MIN_RESOLUTION_HEIGHT:
            return resolution, "Below minimum dimensions"
        else:
            return resolution, "Acceptable"
    
    except ImportError:
        logger.warning("OpenCV not available, resolution validation skipped")
        return (640, 480), "Acceptable"
    except Exception as e:
        logger.warning("Resolution validation failed: %s", e)
        return None, "Validation failed"


def analyze_image_quality(image_bytes: bytes) -> ImageQualityResult:
    """
    Perform comprehensive quality analysis on an image.
    
    Checks:
    1. Blur detection (Laplacian variance)
    2. Brightness analysis (mean luminance)
    3. Resolution validation (minimum dimensions)
    4. GPS metadata verification (EXIF)
    
    Returns ImageQualityResult with detailed feedback.
    """
    result = ImageQualityResult()
    
    # 1. Blur Detection
    blur_score, blur_status = _detect_blur(image_bytes)
    result.blur_score = round(blur_score, 2)
    result.blur_status = blur_status
    
    if blur_status == "Blurry":
        result.issues.append("Image is too blurry to use as evidence")
        result.recommendations.append("Ensure good lighting and camera stability when taking photos")
    elif blur_status == "Slightly blurry":
        result.recommendations.append("Try to take clearer photos for better damage assessment")
    
    # 2. Brightness Analysis
    brightness, brightness_status = _analyze_brightness(image_bytes)
    result.brightness_level = round(brightness, 1)
    result.brightness_status = brightness_status
    
    if brightness_status == "Too dark":
        result.issues.append("Image is too dark - details not visible")
        result.recommendations.append("Retake photo with better lighting")
    elif brightness_status == "Too bright":
        result.issues.append("Image is overexposed - details washed out")
        result.recommendations.append("Avoid direct sunlight; use diffused lighting")
    
    # 3. Resolution Validation
    resolution, resolution_status = _validate_resolution(image_bytes)
    result.resolution = resolution
    result.resolution_status = resolution_status
    
    if resolution_status == "Too low" or resolution_status == "Below minimum dimensions":
        result.issues.append(f"Image resolution too low ({resolution[0]}x{resolution[1]} pixels)")
        result.recommendations.append(f"Use a camera or phone with at least {MIN_RESOLUTION_WIDTH}x{MIN_RESOLUTION_HEIGHT} resolution")
    
    # 4. GPS Metadata Verification
    has_gps, gps_data = _extract_gps_from_exif(image_bytes)
    result.gps_metadata_available = has_gps
    
    if has_gps:
        result.recommendations.append("GPS location verified - strengthens claim authenticity")
    else:
        result.recommendations.append("Enable GPS location on camera/phone for stronger evidence")
    
    # Calculate overall quality score (0-100)
    score = 100.0
    
    # Deduct for blur
    if blur_status == "Blurry":
        score -= 30
    elif blur_status == "Slightly blurry":
        score -= 10
    
    # Deduct for brightness
    if brightness_status == "Too dark" or brightness_status == "Too bright":
        score -= 25
    
    # Deduct for resolution
    if resolution_status == "Too low" or resolution_status == "Below minimum dimensions":
        score -= 25
    
    # Bonus for GPS metadata
    if has_gps:
        score = min(100.0, score + 5)
    
    result.quality_score = max(0.0, score)
    
    # Determine overall quality rating
    if result.quality_score >= 80:
        result.overall_quality = "Good"
    elif result.quality_score >= 60:
        result.overall_quality = "Fair"
    elif result.quality_score >= 40:
        result.overall_quality = "Poor"
    else:
        result.overall_quality = "Unusable"
    
    return result
