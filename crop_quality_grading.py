"""
Advanced Crop Quality Grading System
Uses computer vision and ML to automatically grade harvested crops
"""

import numpy as np
import cv2
from PIL import Image
import io
import json
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
        # Aspect ratio = minor/major axis (0=line, 1=perfect circle).
        # Tomatoes are near-spherical so the ratio stays close to 1.
        "aspect_ratio_range": (0.80, 1.00),
        "defect_threshold": 10,  # percentage
    },
    "potato": {
        "color_ranges": {"red": (100, 140), "green": (90, 130), "blue": (70, 110)},
        "size_range": (50, 200),
        "shape_uniformity_threshold": 0.70,
        # Potatoes are mildly oblong; accept a wider range toward elongation.
        "aspect_ratio_range": (0.55, 1.00),
        "defect_threshold": 15,
    },
    "grain": {
        "color_ranges": {"red": (150, 200), "green": (130, 180), "blue": (80, 130)},
        "size_range": (5, 15),
        "shape_uniformity_threshold": 0.80,
        # Grains (rice, wheat) are distinctly elongated kernels.
        "aspect_ratio_range": (0.25, 0.65),
        "defect_threshold": 8,
    },
    "fruit": {
        "color_ranges": {"red": (140, 220), "green": (50, 150), "blue": (30, 100)},
        "size_range": (50, 200),
        "shape_uniformity_threshold": 0.78,
        # General fruit (apple, mango) range from round to slightly oblong.
        "aspect_ratio_range": (0.65, 1.00),
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

    def __init__(self):
        self.supported_crops = list(CROP_QUALITY_PARAMS.keys())
        self.quality_history = []

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
        contours, _ = cv2.findContours(
            cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
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
        """Assess color quality"""
        if not isinstance(image, np.ndarray):
            return 50.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_quality_score = 0.0

        # Check dominant color in expected range
        h, s, v = cv2.split(hsv)
        avg_h = np.mean(h)
        avg_s = np.mean(s)
        avg_v = np.mean(v)

        # Saturation and value scores
        saturation_score = min(100, (avg_s / 255) * 120)
        brightness_score = min(100, (avg_v / 255) * 110)

        color_quality_score = (saturation_score * 0.6 + brightness_score * 0.4)

        return float(min(100, color_quality_score))

    def _assess_shape(self, image: np.ndarray, params: Dict) -> float:
        """Assess shape quality against the crop-specific expected aspect ratio range.

        Rather than applying a universal circularity metric (which penalises
        naturally elongated crops such as grains or oblong potatoes), this method
        measures the minor/major axis ratio of each detected contour via
        ``cv2.fitEllipse`` and scores how well it falls within the
        ``aspect_ratio_range`` configured in ``CROP_QUALITY_PARAMS``.

        Ratio convention: ``minor_axis / major_axis`` ∈ (0, 1].
        A ratio of 1 means a perfect circle; values approaching 0 indicate
        high elongation.  Each contour whose ratio lies within ``[low, high]``
        scores 100; contours outside the range are penalised proportionally
        to their deviation scaled by the range width.  The final score is the
        mean across all valid contours.
        """
        if not isinstance(image, np.ndarray):
            return 50.0

        aspect_ratio_range = params.get("aspect_ratio_range")
        if not aspect_ratio_range:
            # No range configured — fall back to neutral score
            return 50.0

        low, high = aspect_ratio_range
        range_width = max(high - low, 0.01)  # guard against zero-width ranges

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return 50.0

        contour_scores: List[float] = []
        for contour in contours:
            # fitEllipse requires at least 5 points
            if len(contour) < 5:
                continue
            try:
                _, (minor_axis, major_axis), _ = cv2.fitEllipse(contour)
            except cv2.error:
                continue
            if major_axis <= 0:
                continue

            ratio = min(minor_axis, major_axis) / max(minor_axis, major_axis)

            if low <= ratio <= high:
                score = 100.0
            elif ratio < low:
                deviation = low - ratio
                score = max(0.0, 100.0 - (deviation / range_width) * 100.0)
            else:  # ratio > high
                deviation = ratio - high
                score = max(0.0, 100.0 - (deviation / range_width) * 100.0)

            contour_scores.append(score)

        if not contour_scores:
            return 50.0

        return float(min(100.0, np.mean(contour_scores)))

    def _detect_defects(self, image: np.ndarray, params: Dict) -> float:
        """Detect defects in crops"""
        if not isinstance(image, np.ndarray):
            return 10.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use edge detection to find defects
        edges = cv2.Canny(gray, 50, 150)
        total_pixels = edges.shape[0] * edges.shape[1]
        defect_pixels = np.count_nonzero(edges)

        defect_percentage = (defect_pixels / total_pixels) * 100

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
        """Get quality trends over time"""
        recent_assessments = [
            a
            for a in self.quality_history
            if a.crop_type == crop_type.lower()
        ]

        if not recent_assessments:
            return {"error": "No assessment history"}

        scores = [a.score for a in recent_assessments]
        grades = [a.grade for a in recent_assessments]

        return {
            "crop_type": crop_type.lower(),
            "assessments_count": len(recent_assessments),
            "average_score": round(np.mean(scores), 2),
            "score_trend": scores[-5:],  # Last 5 scores
            "grade_distribution": {g: grades.count(g) for g in set(grades)},
            "latest_assessment": asdict(recent_assessments[-1]),
        }
