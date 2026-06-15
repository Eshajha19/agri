"""
Model Versioning & Rollback Module
Manages model versions and enables safe rollback to previous versions.
"""
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a model version"""
    version_id: str
    model_name: str
    model_path: str
    created_at: str
    promoted_at: Optional[str]
    is_production: bool
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelVersionManager:
    """
    Manages model versions and enables rollback.
    Keeps track of model history and performance metrics.
    """
    
    def __init__(self, versions_dir: str = "model_versions"):
        """
        Initialize ModelVersionManager
        
        Args:
            versions_dir: Directory to store version metadata
        """
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(exist_ok=True)
        
        self.versions: Dict[str, ModelVersion] = {}
        self.production_version: Optional[str] = None
        self.version_history: List[Dict[str, Any]] = []
        
        self._load_versions()
    
    def _load_versions(self):
        """Load existing versions from disk"""
        versions_file = self.versions_dir / "versions.json"
        if versions_file.exists():
            try:
                with open(versions_file, 'r') as f:
                    data = json.load(f)
                    self.versions = {
                        v_id: ModelVersion(**v) for v_id, v in data.get('versions', {}).items()
                    }
                    self.production_version = data.get('production_version')
                    self.version_history = data.get('version_history', [])
                logger.info(f"Loaded {len(self.versions)} model versions from disk")
            except Exception as e:
                logger.error(f"Error loading versions: {e}")
    
    def _save_versions(self):
        """Save versions to disk"""
        versions_file = self.versions_dir / "versions.json"
        try:
            data = {
                'versions': {v_id: v.to_dict() for v_id, v in self.versions.items()},
                'production_version': self.production_version,
                'version_history': self.version_history,
            }
            with open(versions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving versions: {e}")
    
    def register_version(
        self,
        model_name: str,
        model_path: str,
        performance_metrics: Dict[str, float],
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Register a new model version
        
        Args:
            model_name: Name of the model (e.g., 'xgboost', 'lstm')
            model_path: Path to the model file
            performance_metrics: Dict of metrics (e.g., {'rmse': 0.15, 'r2': 0.85})
            metadata: Additional metadata about the version
        
        Returns:
            Version ID
        """
        version_id = f"{model_name}_v{len([v for v in self.versions.values() if v.model_name == model_name]) + 1}"
        
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            model_path=model_path,
            created_at=datetime.now().isoformat(),
            promoted_at=None,
            is_production=False,
            performance_metrics=performance_metrics,
            metadata=metadata or {},
        )
        
        self.versions[version_id] = version
        
        self.version_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'register',
            'version_id': version_id,
            'details': f"Registered {model_name} with RMSE={performance_metrics.get('rmse', 'N/A')}"
        })
        
        self._save_versions()
        logger.info(f"Registered version: {version_id}")
        
        return version_id
    
    def promote_version(self, version_id: str) -> bool:
        """
        Promote a version to production
        
        Args:
            version_id: Version to promote
        
        Returns:
            True if successful
        """
        if version_id not in self.versions:
            logger.error(f"Version not found: {version_id}")
            return False
        
        # Demote current production version
        if self.production_version:
            old_version = self.versions.get(self.production_version)
            if old_version is not None:
                old_version.is_production = False
            
            self.version_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'demote',
                'version_id': self.production_version,
                'details': f"Demoted {self.production_version}" + (" (was in production for X days)" if old_version else " (version no longer exists)")
            })
        
        # Promote new version
        version = self.versions[version_id]
        version.is_production = True
        version.promoted_at = datetime.now().isoformat()
        self.production_version = version_id
        
        self.version_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'promote',
            'version_id': version_id,
            'details': f"Promoted {version_id} to production (RMSE={version.performance_metrics.get('rmse', 'N/A')})"
        })
        
        self._save_versions()
        logger.info(f"Promoted version: {version_id}")
        
        return True
    
    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a previous model version
        
        Args:
            version_id: Version to rollback to
        
        Returns:
            True if successful
        """
        if version_id not in self.versions:
            logger.error(f"Version not found: {version_id}")
            return False
        
        # Record current production before rollback
        previous_production = self.production_version
        
        # Perform rollback
        success = self.promote_version(version_id)
        
        if success:
            self.version_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'rollback',
                'version_id': version_id,
                'details': f"Rolled back from {previous_production} to {version_id}"
            })
            logger.warning(f"Rolled back to version: {version_id}")
        
        return success
    
    def get_production_version(self) -> Optional[ModelVersion]:
        """Get current production version"""
        if self.production_version:
            return self.versions.get(self.production_version)
        return None
    
    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get details about a specific version"""
        if version_id in self.versions:
            return self.versions[version_id].to_dict()
        return None
    
    def list_versions(
        self,
        model_name: str = None,
        include_production_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """List all versions, optionally filtered"""
        versions = list(self.versions.values())
        
        if model_name:
            versions = [v for v in versions if v.model_name == model_name]
        
        if include_production_only:
            versions = [v for v in versions if v.is_production]
        
        return [v.to_dict() for v in versions]
    
    def get_version_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get history of version management actions"""
        return self.version_history[-limit:]
    
    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict[str, Any]:
        """Compare two versions"""
        v1 = self.versions.get(version_id_1)
        v2 = self.versions.get(version_id_2)
        
        if not v1 or not v2:
            return {'error': 'One or both versions not found'}
        
        # Compare performance metrics
        metrics_comparison = {}
        all_metrics = set(v1.performance_metrics.keys()) | set(v2.performance_metrics.keys())
        
        for metric in all_metrics:
            m1 = v1.performance_metrics.get(metric, 0)
            m2 = v2.performance_metrics.get(metric, 0)
            diff = m2 - m1
            diff_pct = (diff / m1 * 100) if m1 != 0 else 0
            
            metrics_comparison[metric] = {
                'version_1': m1,
                'version_2': m2,
                'difference': diff,
                'difference_percent': diff_pct,
                'better_version': 'v1' if (metric in ['r2', 'accuracy'] and m1 > m2) or (metric in ['rmse', 'mae'] and m1 < m2) else 'v2'
            }
        
        return {
            'version_1': v1.to_dict(),
            'version_2': v2.to_dict(),
            'metrics_comparison': metrics_comparison,
        }
    
    def cleanup_old_versions(self, keep_count: int = 5, model_name: str = None) -> int:
        """
        Remove old versions, keeping the most recent N versions
        
        Args:
            keep_count: Number of recent versions to keep
            model_name: Optional model name to filter by
        
        Returns:
            Number of versions deleted
        """
        versions_to_check = [
            v for v in self.versions.values()
            if model_name is None or v.model_name == model_name
        ]
        
        # Sort by creation date
        versions_to_check.sort(key=lambda x: x.created_at, reverse=True)
        
        to_delete = []
        deleted_count = 0
        
        for version in versions_to_check[keep_count:]:
            if not version.is_production:  # Never delete production version
                to_delete.append(version.version_id)
                deleted_count += 1
        
        for version_id in to_delete:
            del self.versions[version_id]
            logger.info(f"Deleted old version: {version_id}")
        
        if deleted_count > 0:
            self._save_versions()
        
        return deleted_count
