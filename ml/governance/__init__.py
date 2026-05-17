"""
ML Governance Module
Provides drift detection, shadow evaluation, and model versioning/rollback.
"""

from ml.governance.drift_detector import DriftDetector, DriftAlert
from ml.governance.shadow_evaluator import ShadowEvaluator, ShadowEvaluation
from ml.governance.model_versioning import ModelVersionManager, ModelVersion

__all__ = [
    'DriftDetector',
    'DriftAlert',
    'ShadowEvaluator',
    'ShadowEvaluation',
    'ModelVersionManager',
    'ModelVersion',
]
