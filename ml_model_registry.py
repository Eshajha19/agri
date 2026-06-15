"""
ML Model Versioning & Registry System
Manages model versions, deployment history, and metadata
"""

import logging
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import uuid

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status"""
    DRAFT = "draft"
    CANARY = "canary"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    ROLLED_BACK = "rolled_back"


class ModelVersion:
    """Model version metadata and registry"""
    
    def __init__(
        self,
        model_name: str,
        version: str,
        model_path: str,
        status: ModelStatus = ModelStatus.DRAFT,
        created_by: str = "system",
        description: str = None,
        metrics: Dict[str, float] = None
    ):
        self.model_id = str(uuid.uuid4())
        self.model_name = model_name
        self.version = version
        self.model_path = model_path
        self.status = status
        self.created_by = created_by
        self.description = description
        self.metrics = metrics or {}
        self.created_at = datetime.now().isoformat()
        self.deployed_at = None
        self.rollback_reason = None
        self.canary_traffic_percentage = 0
        self.deployment_history: List[Dict] = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "model_path": self.model_path,
            "status": self.status.value,
            "created_by": self.created_by,
            "description": self.description,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "deployed_at": self.deployed_at,
            "canary_traffic_percentage": self.canary_traffic_percentage,
            "rollback_reason": self.rollback_reason,
            "deployment_history": self.deployment_history
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'ModelVersion':
        """Create from dictionary"""
        version = ModelVersion(
            model_name=data["model_name"],
            version=data["version"],
            model_path=data["model_path"],
            status=ModelStatus(data.get("status", "draft")),
            created_by=data.get("created_by", "system"),
            description=data.get("description"),
            metrics=data.get("metrics", {})
        )
        version.model_id = data["model_id"]
        version.created_at = data["created_at"]
        version.deployed_at = data.get("deployed_at")
        version.canary_traffic_percentage = data.get("canary_traffic_percentage", 0)
        version.rollback_reason = data.get("rollback_reason")
        version.deployment_history = data.get("deployment_history", [])
        return version


class ModelRegistry:
    """Central registry for all model versions"""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, ModelVersion]] = {}  # model_name -> version -> ModelVersion
        self.active_models: Dict[str, ModelVersion] = {}  # model_name -> active_version
        self.deployment_log: List[Dict] = []
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        created_by: str = "system",
        description: str = None,
        metrics: Dict[str, float] = None
    ) -> ModelVersion:
        """Register new model version"""
        
        model = ModelVersion(
            model_name=model_name,
            version=version,
            model_path=model_path,
            created_by=created_by,
            description=description,
            metrics=metrics
        )
        
        if model_name not in self.models:
            self.models[model_name] = {}
        
        self.models[model_name][version] = model
        
        logger.info(f"Registered model {model_name}:{version} (ID: {model.model_id})")
        return model
    
    def get_model_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version"""
        return self.models.get(model_name, {}).get(version)
    
    def get_active_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get currently active model"""
        return self.active_models.get(model_name)
    
    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model"""
        return list(self.models.get(model_name, {}).values())
    
    def promote_to_canary(
        self,
        model_name: str,
        version: str,
        traffic_percentage: int = 5
    ) -> bool:
        """Promote model to canary (5% traffic)"""
        
        model = self.get_model_version(model_name, version)
        if not model:
            logger.error(f"Model {model_name}:{version} not found")
            return False
        
        model.status = ModelStatus.CANARY
        model.canary_traffic_percentage = traffic_percentage
        model.deployed_at = datetime.now().isoformat()
        
        self._log_deployment(model_name, version, "canary", traffic_percentage)
        logger.info(f"Promoted {model_name}:{version} to CANARY ({traffic_percentage}% traffic)")
        
        return True
    
    def promote_to_staging(
        self,
        model_name: str,
        version: str,
        traffic_percentage: int = 25
    ) -> bool:
        """Promote model to staging (25% traffic)"""
        
        model = self.get_model_version(model_name, version)
        if not model:
            return False
        
        model.status = ModelStatus.STAGING
        model.canary_traffic_percentage = traffic_percentage
        
        self._log_deployment(model_name, version, "staging", traffic_percentage)
        logger.info(f"Promoted {model_name}:{version} to STAGING ({traffic_percentage}% traffic)")
        
        return True
    
    def promote_to_production(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """Promote model to production (100% traffic)"""
        
        model = self.get_model_version(model_name, version)
        if not model:
            return False
        
        # Archive previous production model
        if model_name in self.active_models:
            old_model = self.active_models[model_name]
            old_model.status = ModelStatus.ARCHIVED
        
        model.status = ModelStatus.PRODUCTION
        model.canary_traffic_percentage = 100
        model.deployed_at = datetime.now().isoformat()
        self.active_models[model_name] = model
        
        self._log_deployment(model_name, version, "production", 100)
        logger.info(f"Promoted {model_name}:{version} to PRODUCTION")
        
        return True
    
    def rollback(
        self,
        model_name: str,
        reason: str = "Performance degradation"
    ) -> bool:
        """Rollback to previous production model"""
        
        current = self.active_models.get(model_name)
        if not current:
            logger.error(f"No active model for {model_name}")
            return False
        
        current.status = ModelStatus.ROLLED_BACK
        current.rollback_reason = reason
        
        # Find the version that was most recently promoted to production
        # before the current model (using the deployment log rather than
        # picking the most recently archived model, which may have been
        # a canary or experimental build that never served production).
        production_logs = [
            log for log in self.deployment_log
            if log["model_name"] == model_name
            and log["version"] != current.version
            and log["action"] in ("production", "rollback")
        ]
        production_logs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        previous = None
        if production_logs:
            prev_version = production_logs[0]["version"]
            previous = self.get_model_version(model_name, prev_version)
            if previous:
                previous.status = ModelStatus.PRODUCTION
                previous.canary_traffic_percentage = 100
                self.active_models[model_name] = previous
                
                self._log_deployment(model_name, previous.version, "rollback", 100, reason)
                logger.warning(f"Rolled back {model_name} to {previous.version}: {reason}")
                return True
        
        logger.error(f"No previous production model found for {model_name}")
        return False
    
    def _log_deployment(
        self,
        model_name: str,
        version: str,
        action: str,
        traffic: int,
        reason: str = None
    ):
        """Log deployment event"""
        self.deployment_log.append({
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "version": version,
            "action": action,
            "traffic_percentage": traffic,
            "reason": reason
        })
    
    def get_deployment_history(
        self,
        model_name: str,
        limit: int = 20
    ) -> List[Dict]:
        """Get deployment history for a model"""
        return [
            log for log in self.deployment_log
            if log["model_name"] == model_name
        ][-limit:]
    
    def export_registry(self) -> Dict:
        """Export entire registry as JSON"""
        return {
            "models": {
                name: {
                    version: model.to_dict()
                    for version, model in versions.items()
                }
                for name, versions in self.models.items()
            },
            "active_models": {
                name: model.to_dict()
                for name, model in self.active_models.items()
            },
            "deployment_log": self.deployment_log
        }


# Global registry instance
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create global model registry"""
    global _model_registry
    
    if _model_registry is None:
        _model_registry = ModelRegistry()
    
    return _model_registry
