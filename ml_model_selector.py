"""
ML Model Selection & Routing Engine
Routes predictions to appropriate model version based on feature flags and A/B tests
"""

import logging
import random
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelSelection:
    """Selected model for prediction"""
    model_id: str
    model_version: str
    model_path: str
    test_id: Optional[str] = None
    arm_id: Optional[str] = None
    reason: str = "default"


class ModelSelector:
    """Routes predictions to models based on rules and tests"""
    
    def __init__(self):
        self.feature_flags = {}  # flag_name -> {enabled, rollout_percentage, model_id}
        self.user_segments = {}  # user_id -> list of segments
        self.model_rules = {}  # model_name -> rule_set
        self.A_B_tests = {}  # model_name -> active test
    
    def register_feature_flag(
        self,
        flag_name: str,
        model_id: str,
        enabled: bool = True,
        rollout_percentage: int = 0,
        regions: list = None,
        user_segments: list = None
    ):
        """Register feature flag for model"""
        
        self.feature_flags[flag_name] = {
            "name": flag_name,
            "model_id": model_id,
            "enabled": enabled,
            "rollout_percentage": rollout_percentage,
            "regions": regions or [],
            "user_segments": user_segments or []
        }
        
        logger.info(f"Registered feature flag: {flag_name} -> {model_id}")
    
    def segment_user(self, user_id: str, segment_name: str):
        """Add user to segment"""
        
        if user_id not in self.user_segments:
            self.user_segments[user_id] = []
        
        if segment_name not in self.user_segments[user_id]:
            self.user_segments[user_id].append(segment_name)
    
    def is_flag_enabled(
        self,
        flag_name: str,
        user_id: str = None,
        region: str = None
    ) -> bool:
        """Check if feature flag is enabled for user"""
        
        if flag_name not in self.feature_flags:
            return False
        
        flag = self.feature_flags[flag_name]
        
        if not flag["enabled"]:
            return False
        
        # Check rollout percentage
        if flag["rollout_percentage"] > 0:
            # Use deterministic hashing (SHA-256) so assignments survive restarts
            if user_id:
                import hashlib
                digest = hashlib.sha256(f"{user_id}:{flag_name}".encode()).hexdigest()
                hash_val = int(digest[:8], 16) % 100
                if hash_val >= flag["rollout_percentage"]:
                    return False
        
        # Check region
        if flag["regions"] and region and region not in flag["regions"]:
            return False
        
        # Check user segments
        if flag["user_segments"] and user_id:
            user_segs = self.user_segments.get(user_id, [])
            if not any(seg in flag["user_segments"] for seg in user_segs):
                return False
        
        return True
    
    def select_model(
        self,
        model_name: str,
        active_models: Dict,
        ab_test_manager: Optional[Any] = None,
        user_id: str = None,
        region: str = None,
        request_context: Dict = None
    ) -> ModelSelection:
        """
        Select model for prediction
        
        Args:
            model_name: Name of the model to select
            active_models: Dict of active models {model_id: model_version}
            ab_test_manager: A/B test manager instance
            user_id: User ID for segmentation
            region: Region for rollout
            request_context: Additional request context
        
        Returns:
            ModelSelection with selected model
        """
        
        request_context = request_context or {}
        
        # Check if user is in A/B test
        if ab_test_manager:
            ab_test = ab_test_manager.active_tests.get(model_name)
            if ab_test and ab_test.status.value == "running":
                # Select arm for this request
                arm = ab_test.select_arm()
                if arm:
                    # Get model from active models
                    if arm.model_id in active_models:
                        return ModelSelection(
                            model_id=arm.model_id,
                            model_version=arm.model_version,
                            model_path=active_models[arm.model_id]["path"],
                            test_id=ab_test.test_id,
                            arm_id=arm.model_id,
                            reason="A/B_test"
                        )
        
        # Check feature flags
        for flag_name, flag in self.feature_flags.items():
            if self.is_flag_enabled(flag_name, user_id, region):
                if flag["model_id"] in active_models:
                    return ModelSelection(
                        model_id=flag["model_id"],
                        model_version=active_models[flag["model_id"]]["version"],
                        model_path=active_models[flag["model_id"]]["path"],
                        reason=f"feature_flag:{flag_name}"
                    )
        
        # Use canary traffic allocation if available
        canary_data = request_context.get("canary_traffic")
        if canary_data:
            rand = random.random()
            cumulative = 0
            
            for model_id, traffic_pct in canary_data.items():
                cumulative += traffic_pct / 100
                if rand < cumulative:
                    if model_id in active_models:
                        return ModelSelection(
                            model_id=model_id,
                            model_version=active_models[model_id]["version"],
                            model_path=active_models[model_id]["path"],
                            reason="canary_traffic"
                        )
        
        # Default to primary model
        primary_models = [
            (mid, mv) for mid, mv in active_models.items()
            if mv.get("status") == "production"
        ]
        
        if primary_models:
            model_id, model_data = primary_models[0]
            return ModelSelection(
                model_id=model_id,
                model_version=model_data["version"],
                model_path=model_data["path"],
                reason="primary_production"
            )
        
        # Fallback to any active model
        if active_models:
            model_id, model_data = list(active_models.items())[0]
            return ModelSelection(
                model_id=model_id,
                model_version=model_data["version"],
                model_path=model_data["path"],
                reason="fallback"
            )
        
        raise ValueError(f"No models available for {model_name}")
    
    def record_selection_outcome(
        self,
        selection: ModelSelection,
        success: bool,
        metrics: Dict,
        ab_test_manager: Optional[Any] = None
    ):
        """Record outcome of model selection"""
        
        # Update A/B test if applicable
        if selection.test_id and ab_test_manager:
            ab_test_manager.record_outcome(
                selection.test_id,
                selection.arm_id,
                success,
                metrics
            )
        
        logger.debug(f"Recorded selection outcome: {selection.model_id}, success={success}")
    
    def to_dict(self) -> Dict:
        """Export configuration"""
        return {
            "feature_flags": self.feature_flags,
            "user_segments": self.user_segments,
            "A_B_tests_count": len(self.A_B_tests)
        }


# Global model selector instance
_model_selector: Optional[ModelSelector] = None


def get_model_selector() -> ModelSelector:
    """Get or create global model selector"""
    global _model_selector
    
    if _model_selector is None:
        _model_selector = ModelSelector()
    
    return _model_selector
