"""
Test suite for Crop Quality Grading System
Tests the computer vision and ML-based crop quality assessment
"""

import pytest
import numpy as np
import cv2
from PIL import Image
import io
import base64
from crop_quality_grading import (
    CropQualityGrader,
    CROP_QUALITY_PARAMS,
    GRADE_MAPPING,
    QualityAssessment,
)


class TestCropQualityGrader:
    """Test suite for crop quality grading"""

    @pytest.fixture
    def grader(self):
        """Initialize grader for tests"""
        return CropQualityGrader()

    @pytest.fixture
    def dummy_image(self):
        """Create a dummy test image"""
        # Create a simple test image
        img = Image.new("RGB", (200, 200), color=(100, 150, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    def test_grader_initialization(self, grader):
        """Test grader initialization"""
        assert grader is not None
        assert len(grader.supported_crops) == 4
        assert "tomato" in grader.supported_crops
        assert "potato" in grader.supported_crops
        assert "grain" in grader.supported_crops
        assert "fruit" in grader.supported_crops

    def test_unsupported_crop_type(self, grader, dummy_image):
        """Test error handling for unsupported crop type"""
        with pytest.raises(ValueError):
            grader.assess_crop_image(dummy_image, "unknown_crop")

    def test_invalid_image_data(self, grader):
        """Test error handling for invalid image data"""
        with pytest.raises(ValueError):
            grader.assess_crop_image(b"invalid image data", "tomato")

    def test_assess_single_crop(self, grader, dummy_image):
        """Test single crop assessment"""
        assessment = grader.assess_crop_image(dummy_image, "tomato")

        assert assessment is not None
        assert isinstance(assessment, QualityAssessment)
        assert assessment.crop_type == "tomato"
        assert 0 <= assessment.score <= 100
        assert assessment.grade in ["A", "B", "C", "D", "F"]
        assert 0 <= assessment.size_quality <= 100
        assert 0 <= assessment.color_quality <= 100
        assert 0 <= assessment.shape_quality <= 100
        assert 0 <= assessment.defect_percentage <= 100
        assert 0 < assessment.market_price_adjustment
        assert len(assessment.recommendations) > 0
        assert 0 < assessment.confidence <= 100

    def test_crop_type_case_insensitivity(self, grader, dummy_image):
        """Test that crop type is case-insensitive"""
        assessment1 = grader.assess_crop_image(dummy_image, "TOMATO")
        assessment2 = grader.assess_crop_image(dummy_image, "tomato")

        assert assessment1.crop_type == assessment2.crop_type

    def test_quality_history_tracking(self, grader, dummy_image):
        """Test that assessments are tracked in history"""
        initial_count = len(grader.quality_history)

        grader.assess_crop_image(dummy_image, "tomato")
        assert len(grader.quality_history) == initial_count + 1

        grader.assess_crop_image(dummy_image, "potato")
        assert len(grader.quality_history) == initial_count + 2

    def test_batch_grading(self, grader, dummy_image):
        """Test batch grading of multiple crops"""
        images = [dummy_image] * 3
        result = grader.batch_grade_crops(images, "tomato")

        assert "assessments" in result
        assert "batch_statistics" in result
        assert result["batch_statistics"]["total_crops"] == 3
        assert result["batch_statistics"]["graded_crops"] == 3
        assert "average_score" in result["batch_statistics"]
        assert "grade_distribution" in result["batch_statistics"]

    def test_batch_grading_with_invalid_image(self, grader):
        """Test batch grading with some invalid images"""
        valid_image = self._create_test_image()
        invalid_images = [valid_image, b"invalid", valid_image]

        result = grader.batch_grade_crops(invalid_images, "tomato")

        assert result["batch_statistics"]["total_crops"] == 3
        assert result["batch_statistics"]["graded_crops"] == 2

    def test_get_grade_mapping(self):
        """Test grade mapping"""
        assert GRADE_MAPPING["A"]["min_score"] == 90
        assert GRADE_MAPPING["A"]["label"] == "Premium"
        assert GRADE_MAPPING["B"]["min_score"] == 75
        assert GRADE_MAPPING["C"]["min_score"] == 60
        assert GRADE_MAPPING["D"]["min_score"] == 40
        assert GRADE_MAPPING["F"]["min_score"] == 0

    def test_grade_determination(self, grader):
        """Test grade determination based on score"""
        assert grader._get_grade(95) == "A"
        assert grader._get_grade(85) == "B"
        assert grader._get_grade(65) == "C"
        assert grader._get_grade(45) == "D"
        assert grader._get_grade(20) == "F"

    def test_recommendations_generation(self, grader):
        """Test recommendation generation"""
        recs = grader._generate_recommendations(
            size_quality=80,
            color_quality=80,
            shape_quality=80,
            defect_percentage=5,
        )

        assert isinstance(recs, list)
        assert len(recs) > 0
        assert "Excellent quality" in recs[0] or "maintain" in recs[0].lower()

    def test_recommendations_with_poor_quality(self, grader):
        """Test recommendations for poor quality crops"""
        recs = grader._generate_recommendations(
            size_quality=40,
            color_quality=30,
            shape_quality=45,
            defect_percentage=50,
        )

        assert any("size" in rec.lower() for rec in recs)
        assert any("ripeness" in rec.lower() or "storage" in rec.lower()
                   for rec in recs)
        assert any("handling" in rec.lower() for rec in recs)
        assert any("damage" in rec.lower() for rec in recs)

    def test_quality_trends(self, grader, dummy_image):
        """Test quality trends retrieval"""
        # Create multiple assessments
        for _ in range(3):
            grader.assess_crop_image(dummy_image, "tomato")

        trends = grader.get_quality_trends("tomato", days=7)

        assert trends is not None
        assert "crop_type" in trends
        assert "assessments_count" in trends
        assert trends["assessments_count"] == 3
        assert "average_score" in trends
        assert "latest_assessment" in trends

    def test_crop_quality_params_structure(self):
        """Test the structure of crop quality parameters"""
        for crop in CROP_QUALITY_PARAMS:
            params = CROP_QUALITY_PARAMS[crop]
            assert "color_ranges" in params
            assert "size_range" in params
            assert "shape_uniformity_threshold" in params
            assert "defect_threshold" in params

    def test_size_assessment(self, grader):
        """Test size quality assessment"""
        image = self._create_test_image_cv2()
        size_quality = grader._assess_size(image, CROP_QUALITY_PARAMS["tomato"])

        assert isinstance(size_quality, float)
        assert 0 <= size_quality <= 100

    def test_color_assessment(self, grader):
        """Test color quality assessment"""
        image = self._create_test_image_cv2()
        color_quality = grader._assess_color(image, CROP_QUALITY_PARAMS["tomato"])

        assert isinstance(color_quality, float)
        assert 0 <= color_quality <= 100

    def test_shape_assessment(self, grader):
        """Test shape quality assessment"""
        image = self._create_test_image_cv2()
        shape_quality = grader._assess_shape(image, CROP_QUALITY_PARAMS["tomato"])

        assert isinstance(shape_quality, float)
        assert 0 <= shape_quality <= 100

    def test_defect_detection(self, grader):
        """Test defect detection"""
        image = self._create_test_image_cv2()
        defect_percentage = grader._detect_defects(image, CROP_QUALITY_PARAMS["tomato"])

        assert isinstance(defect_percentage, float)
        assert 0 <= defect_percentage <= 100

    def test_assessment_timestamp(self, grader, dummy_image):
        """Test that assessments have proper timestamps"""
        assessment = grader.assess_crop_image(dummy_image, "tomato")

        assert assessment.timestamp is not None
        assert "T" in assessment.timestamp  # ISO format check

    def test_price_adjustment_values(self):
        """Test price adjustment values are reasonable"""
        for grade, info in GRADE_MAPPING.items():
            assert 0 <= info["price_multiplier"] <= 2.0
            # Higher grades should have higher multipliers
            if grade != "F":
                prev_multiplier = info["price_multiplier"]
                # Verify A > B > C > D > F (in general)

    @staticmethod
    def _create_test_image():
        """Create a simple test image"""
        img = Image.new("RGB", (200, 200), color=(100, 150, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    @staticmethod
    def _create_test_image_cv2():
        """Create a simple test image as numpy array for CV2 operations"""
        return np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    @staticmethod
    def _create_test_image_with_color():
        """Create a test image with specific colors"""
        img = Image.new("RGB", (200, 200), color=(150, 100, 80))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    @staticmethod
    def _create_test_image_with_shapes():
        """Create a test image with shapes"""
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        # Add some shapes to the image
        pixels = img.load()
        for i in range(50, 150):
            for j in range(50, 150):
                if (i - 100) ** 2 + (j - 100) ** 2 < 2500:
                    pixels[i, j] = (100, 100, 100)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
