"""
Advanced Crop Quality Grading System
Uses computer vision and ML to automatically grade harvested crops
"""

import numpy as np
import cv2
from PIL import Image
import io
import json
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import base64
from dataclasses import dataclass, asdict

# Quality grading parameters for different crops
CROP_QUALITY_PARAMS = {
    "tomato": {
        "color_ranges": {"red": (120, 180), "green": (40, 80), "blue": (30, 70)},
        "size_range": (40, 150),  # mm
        "shape_uniformity_threshold": 0.75,
        "defect_threshold": 10,  # percentage
    },
    "potato": {
        "color_ranges": {"red": (100, 140), "green": (90, 130), "blue": (70, 110)},
        "size_range": (50, 200),
        "shape_uniformity_threshold": 0.70,
        "defect_threshold": 15,
    },
    "grain": {
        "color_ranges": {"red": (150, 200), "green": (130, 180), "blue": (80, 130)},
        "size_range": (5, 15),
        "shape_uniformity_threshold": 0.80,
        "defect_threshold": 8,
    },
    "fruit": {
        "color_ranges": {"red": (140, 220), "green": (50, 150), "blue": (30, 100)},
        "size_range": (50, 200),
        "shape_uniformity_threshold": 0.78,
        "defect_threshold": 12,
    },
}

# Market grade mapping
GRADE_MAPPING = {
    "A": {"min_score": 90, "label": "Premium", "price_multiplier": 1.4},
    "B": {"min_score": 75, "label": "Good", "price_multiplier": 1.2},
    "C": {"min_score": 60, "label": "Standard", "price_multiplier": 1.0},
    "D": {"min_score": 40, "label": "Below Average", "price_multiplier": 0.7},
    "F": {"min_score": 0, "label": "Reject", "price_multiplier": 0.0},
}


@dataclass
class QualityAssessment:
    """Quality assessment result for a crop"""

    crop_type: str
    grade: str
    score: float
    size_quality: float
    color_quality: float
    shape_quality: float
    defect_percentage: float
    market_price_adjustment: float
    recommendations: List[str]
    timestamp: str
    confidence: float


class CropQualityGrader:
    """Main crop quality grading system"""

    # Maximum number of assessments retained in the in-process history.
    # Each QualityAssessment is a small dataclass (~200 bytes), so 1 000
    # entries consume roughly 200 KB — a safe upper bound for a long-running
    # process.  When the cap is reached the oldest entry is automatically
    # evicted by the deque before the new one is appended.
    _MAX_HISTORY = 1_000

    def __init__(self):
        self.supported_crops = list(CROP_QUALITY_PARAMS.keys())
        # Bounded deque: oldest assessments are evicted automatically when
        # the cap is reached, preventing unbounded memory growth.
        self.quality_history: deque = deque(maxlen=self._MAX_HISTORY)

    def assess_crop_image(
        self, image_data: bytes, crop_type: str
    ) -> QualityAssessment:
        """
        Assess crop quality from image data
        
        Args:
            image_data: Image bytes (PNG, JPG)
            crop_type: Type of crop (tomato, potato, grain, fruit)
            
        Returns:
            QualityAssessment object with detailed grading
        """
        if crop_type.lower() not in self.supported_crops:
            raise ValueError(
                f"Unsupported crop type: {crop_type}. Supported: {self.supported_crops}"
            )

        # Load image
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image data")

        # Get crop params
        params = CROP_QUALITY_PARAMS[crop_type.lower()]

        # Analyze image
        size_quality = self._assess_size(image, params)
        color_quality = self._assess_color(image, params)
        shape_quality = self._assess_shape(image, params)
        defect_percentage = self._detect_defects(image, params)

        # Calculate overall score (0-100)
        overall_score = (
            size_quality * 0.25
            + color_quality * 0.35
            + shape_quality * 0.25
            + (100 - defect_percentage) * 0.15
        )

        # Determine grade
        grade = self._get_grade(overall_score)
        price_adjustment = GRADE_MAPPING[grade]["price_multiplier"]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            size_quality, color_quality, shape_quality, defect_percentage
        )

        assessment = QualityAssessment(
            crop_type=crop_type.lower(),
            grade=grade,
            score=round(overall_score, 2),
            size_quality=round(size_quality, 2),
            color_quality=round(color_quality, 2),
            shape_quality=round(shape_quality, 2),
            defect_percentage=round(defect_percentage, 2),
            market_price_adjustment=price_adjustment,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            confidence=round(min(95, 70 + (overall_score / 100) * 25), 2),
        )

        # Store in history
        self.quality_history.append(assessment)

        return assessment

    def _assess_size(self, image: np.ndarray, params: Dict) -> float:
        """Assess size uniformity of crops in image"""
        if not isinstance(image, np.ndarray):
            return 50.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use Otsu's thresholding to adapt to varying lighting conditions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return 50.0

        sizes = [cv2.contourArea(c) for c in contours]
        if len(sizes) < 2:
            return 80.0

        # Calculate size uniformity
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        uniformity = max(0.0, 100 - (std_size / mean_size * 100)) if mean_size > 0 else 50.0
        return float(min(100, uniformity))

    def _assess_color(self, image: np.ndarray, params: Dict) -> float:
        """Assess color quality against crop-specific target ranges"""
        if not isinstance(image, np.ndarray):
            return 50.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        avg_h = np.mean(h)
        avg_s = np.mean(s)
        avg_v = np.mean(v)

        # Saturation and value scores
        saturation_score = min(100, (avg_s / 255) * 120)
        brightness_score = min(100, (avg_v / 255) * 110)

        # Hue matching score using crop-specific color ranges
        hue_score = 50.0  # default neutral
        color_ranges = params.get("color_ranges", {})
        if color_ranges:
            # Check if avg_h falls within expected ranges for the crop
            # color_ranges has red, green, blue keys but we're in HSV - use hue equivalents
            # For simplicity, map typical HSV hue ranges: red=0-15/165-180, green=35-85, blue=85-130
            # Check each channel's expected range
            in_range_count = 0
            total_checks = 0
            
            # Red hue wraps around 0/180 in HSV
            red_min, red_max = color_ranges.get("red", (0, 0))
            if red_min <= red_max:
                in_range = red_min <= avg_h <= red_max
            else:  # wraps around 0
                in_range = avg_h >= red_min or avg_h <= red_max
            if red_min or red_max:  # only check if range is defined
                in_range_count += 1 if in_range else 0
                total_checks += 1
            
            green_min, green_max = color_ranges.get("green", (0, 0))
            if green_min or green_max:
                in_range_count += 1 if green_min <= avg_h <= green_max else 0
                total_checks += 1
            
            blue_min, blue_max = color_ranges.get("blue", (0, 0))
            if blue_min or blue_max:
                in_range_count += 1 if blue_min <= avg_h <= blue_max else 0
                total_checks += 1
            
            if total_checks > 0:
                hue_score = (in_range_count / total_checks) * 100

        # Combine: saturation (40%), brightness (30%), hue match (30%)
        color_quality_score = (saturation_score * 0.4 + brightness_score * 0.3 + hue_score * 0.3)

        return float(min(100, color_quality_score))

    def _assess_shape(self, image: np.ndarray, params: Dict) -> float:
        """Assess shape uniformity"""
        if not isinstance(image, np.ndarray):
            return 50.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return 50.0

        # Calculate circularity for each contour
        circularities = []
        for contour in contours:
            if len(contour) > 5:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    circularities.append(min(1.0, circularity))

        if not circularities:
            return 50.0

        avg_circularity = np.mean(circularities)
        shape_quality = avg_circularity * 100

        return min(100, shape_quality)

    def _detect_defects(self, image: np.ndarray, params: Dict) -> float:
        """Detect defects in crops using crop region segmentation"""
        if not isinstance(image, np.ndarray):
            return 10.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Segment crop region using adaptive threshold to exclude background
        _, crop_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert if background is larger than foreground (common case)
        if np.count_nonzero(crop_mask) > crop_mask.size / 2:
            crop_mask = cv2.bitwise_not(crop_mask)

        # Use edge detection on crop region only
        edges = cv2.Canny(gray, 50, 150)
        # Mask edges to crop region only
        crop_edges = cv2.bitwise_and(edges, edges, mask=crop_mask)

        crop_pixels = np.count_nonzero(crop_mask)
        defect_pixels = np.count_nonzero(crop_edges)

        if crop_pixels == 0:
            return 10.0

        defect_percentage = (defect_pixels / crop_pixels) * 100

        return min(100, defect_percentage * 2)  # Scale defects

    def _get_grade(self, score: float) -> str:
        """Determine grade based on score"""
        # Sort grades by their minimum score in descending order
        for grade_key in ["A", "B", "C", "D", "F"]:
            if score >= GRADE_MAPPING[grade_key]["min_score"]:
                return grade_key
        return "F"

    def _generate_recommendations(
        self,
        size_quality: float,
        color_quality: float,
        shape_quality: float,
        defect_percentage: float,
    ) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []

        if size_quality < 70:
            recommendations.append(
                "Improve size uniformity during harvest and sorting"
            )
        if color_quality < 70:
            recommendations.append("Optimize ripeness and storage conditions")
        if shape_quality < 70:
            recommendations.append("Improve crop handling to reduce shape deformities")
        if defect_percentage > 20:
            recommendations.append("Reduce physical damage during harvesting")

        if not recommendations:
            recommendations.append("Excellent quality! Maintain current practices")

        return recommendations

    def batch_grade_crops(
        self, images_data: List[bytes], crop_type: str
    ) -> Dict:
        """
        Grade multiple crops in batch.

        Args:
            images_data: List of image bytes. Each entry must be non-empty bytes.
            crop_type: Type of crop being assessed.

        Returns:
            Dictionary with per-image assessments and aggregate batch statistics.
        """
        if not images_data:
            return {
                "assessments": [],
                "batch_statistics": {
                    "total_crops": 0,
                    "graded_crops": 0,
                    "failed_crops": 0,
                    "average_score": 0,
                    "grade_distribution": {},
                    "average_price_adjustment": 0,
                },
                "crop_type": crop_type,
                "timestamp": datetime.now().isoformat(),
            }

        assessments = []
        for idx, image_data in enumerate(images_data):
            if not isinstance(image_data, (bytes, bytearray)) or len(image_data) == 0:
                assessments.append({
                    "error": f"Image at index {idx} is empty or not valid bytes",
                    "index": idx,
                    "timestamp": datetime.now().isoformat(),
                })
                continue
            try:
                assessment = self.assess_crop_image(image_data, crop_type)
                assessments.append(asdict(assessment))
            except Exception as e:
                assessments.append(
                    {"error": str(e), "index": idx, "timestamp": datetime.now().isoformat()}
                )

        # Calculate batch statistics
        valid_assessments = [a for a in assessments if "error" not in a]
        failed_count = len(assessments) - len(valid_assessments)
        if valid_assessments:
            scores = [a["score"] for a in valid_assessments]
            grades = [a["grade"] for a in valid_assessments]
            batch_stats = {
                "total_crops": len(images_data),
                "graded_crops": len(valid_assessments),
                "failed_crops": failed_count,
                "average_score": round(np.mean(scores), 2),
                "grade_distribution": {
                    g: grades.count(g) for g in set(grades)
                },
                "average_price_adjustment": round(
                    np.mean([a["market_price_adjustment"] for a in valid_assessments]),
                    3,
                ),
            }
        else:
            batch_stats = {
                "total_crops": len(images_data),
                "graded_crops": 0,
                "failed_crops": failed_count,
                "average_score": 0,
                "grade_distribution": {},
                "average_price_adjustment": 0,
            }

        return {
            "assessments": assessments,
            "batch_statistics": batch_stats,
            "crop_type": crop_type,
            "timestamp": datetime.now().isoformat(),
        }

    def get_quality_trends(self, crop_type: str, days: int = 7) -> Dict:
        """Get quality trends over the specified number of days.

        The ``days`` parameter was previously accepted and validated at the
        API layer (ge=1, le=30) but was never used inside this method — the
        filter ``[a for a in self.quality_history if a.crop_type == ...]``
        returned all history for the crop type regardless of age.  A caller
        requesting ``days=1`` received the same result as ``days=30``, and
        the response included ``"days": data.days`` from the router,
        implying the window was respected when it was not.

        Fix: parse each assessment's ``timestamp`` field and exclude entries
        older than ``days`` calendar days from the current UTC time.
        """
        from datetime import timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        recent_assessments = []
        for a in self.quality_history:
            if a.crop_type != crop_type.lower():
                continue
            try:
                # timestamp is stored as datetime.now().isoformat() — naive
                # local time.  Parse it and treat as UTC for comparison.
                ts = datetime.fromisoformat(a.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    recent_assessments.append(a)
            except (ValueError, TypeError):
                # Malformed timestamp — include the assessment rather than
                # silently dropping it.
                recent_assessments.append(a)

        if not recent_assessments:
            return {"error": "No assessment history"}

        scores = [a.score for a in recent_assessments]
        grades = [a.grade for a in recent_assessments]

        return {
            "crop_type": crop_type.lower(),
            "days": days,
            "assessments_count": len(recent_assessments),
            "average_score": round(np.mean(scores), 2),
            "score_trend": scores[-5:],  # Last 5 scores within the window
            "grade_distribution": {g: grades.count(g) for g in set(grades)},
            "latest_assessment": asdict(recent_assessments[-1]),
        }
