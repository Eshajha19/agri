"""
ML Model Management API Routes
Endpoints for model registration, deployment, and monitoring
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import ML systems
from ml_model_registry import get_model_registry, ModelVersion, ModelStatus
from ml_ab_testing import get_ab_test_manager, Arm
from ml_metrics import get_metrics_collector, CanaryMonitor
from ml_model_selector import get_model_selector

router = APIRouter(prefix="/api/ml", tags=["ml"])

# Initialize systems
model_registry = get_model_registry()
ab_test_manager = get_ab_test_manager()
metrics_collector = get_metrics_collector()
model_selector = get_model_selector()
canary_monitor = CanaryMonitor(metrics_collector)


@router.get("/models")
async def list_models():
    """List all registered model versions"""
    try:
        registry_data = model_registry.export_registry()
        all_models = []
        
        for model_name, versions in registry_data["models"].items():
            for version, model_data in versions.items():
                all_models.append(model_data)
        
        return {
            "success": True,
            "models": all_models,
            "total": len(all_models)
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register")
async def register_model(
    model_name: str,
    version: str,
    model_path: str,
    created_by: str = "system",
    description: str = None,
    metrics: Dict = None
):
    """Register new model version"""
    try:
        model = model_registry.register_model(
            model_name=model_name,
            version=version,
            model_path=model_path,
            created_by=created_by,
            description=description,
            metrics=metrics or {}
        )
        
        return {
            "success": True,
            "model_id": model.model_id,
            "message": f"Model {model_name}:{version} registered"
        }
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/promote-canary")
async def promote_canary(
    model_name: str,
    model_id: str,
    traffic_percentage: int = 5
):
    """Promote model to canary (5% traffic)"""
    try:
        # Find the version from model_id
        model = None
        for v in model_registry.list_versions(model_name):
            if v.model_id == model_id:
                model = v
                break
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        success = model_registry.promote_to_canary(model_name, model.version, traffic_percentage)
        
        if success:
            # Start canary monitoring
            current_production = model_registry.get_active_model(model_name)
            if current_production:
                canary_monitor.start_canary(
                    model.model_id,
                    current_production.model_id,
                    traffic_percentage
                )
            
            return {
                "success": True,
                "message": f"Model promoted to canary ({traffic_percentage}% traffic)"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to promote model")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error promoting to canary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/promote-staging")
async def promote_staging(
    model_name: str,
    model_id: str,
    traffic_percentage: int = 25
):
    """Promote model to staging (25% traffic)"""
    try:
        model = None
        for v in model_registry.list_versions(model_name):
            if v.model_id == model_id:
                model = v
                break
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        success = model_registry.promote_to_staging(model_name, model.version, traffic_percentage)
        
        if success:
            return {
                "success": True,
                "message": f"Model promoted to staging ({traffic_percentage}% traffic)"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to promote model")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error promoting to staging: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/promote-production")
async def promote_production(
    model_name: str,
    model_id: str
):
    """Promote model to production (100% traffic)"""
    try:
        model = None
        for v in model_registry.list_versions(model_name):
            if v.model_id == model_id:
                model = v
                break
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        success = model_registry.promote_to_production(model_name, model.version)
        
        if success:
            # Record baseline for monitoring
            metrics_collector.record_baseline(model.model_id)
            
            return {
                "success": True,
                "message": f"Model promoted to production"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to promote model")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error promoting to production: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback")
async def rollback_model(
    model_name: str,
    reason: str = "Performance degradation"
):
    """Rollback to previous production model"""
    try:
        success = model_registry.rollback(model_name, reason)
        
        if success:
            return {
                "success": True,
                "message": f"Model rolled back: {reason}"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to rollback model")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """Get current metrics for all models"""
    try:
        export = metrics_collector.export_metrics()
        return export
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{model_id}")
async def get_model_metrics(model_id: str):
    """Get metrics for specific model"""
    try:
        metrics = metrics_collector.get_metrics(model_id)
        
        return {
            "success": True,
            "model_id": model_id,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record-prediction")
async def record_prediction(
    model_id: str,
    model_version: str,
    y_true: float,
    y_pred: float,
    latency: float
):
    """Record model prediction for metrics tracking"""
    try:
        metrics_collector.record_prediction(
            model_id=model_id,
            model_version=model_version,
            y_true=y_true,
            y_pred=y_pred,
            latency=latency
        )
        
        return {"success": True}
    except Exception as e:
        logger.error(f"Error recording prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-tests")
async def list_ab_tests():
    """List all active A/B tests"""
    try:
        active = ab_test_manager.list_active_tests()
        completed = ab_test_manager.list_completed_tests()
        
        return {
            "success": True,
            "tests": active,
            "completed_tests": completed
        }
    except Exception as e:
        logger.error(f"Error listing A/B tests: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-test/create")
async def create_ab_test(
    test_name: str,
    model_name: str,
    control_model_id: str,
    control_model_version: str,
    variant_model_id: str,
    variant_model_version: str,
    confidence_threshold: float = 0.95,
    min_samples: int = 1000
):
    """Create new A/B test"""
    try:
        control_arm = Arm(control_model_id, control_model_version, "Control")
        variant_arm = Arm(variant_model_id, variant_model_version, "Variant")
        
        test = ab_test_manager.create_test(
            test_name=test_name,
            model_name=model_name,
            control_arm=control_arm,
            variant_arm=variant_arm,
            confidence_threshold=confidence_threshold,
            min_samples=min_samples
        )
        
        ab_test_manager.start_test(test.test_id)
        
        return {
            "success": True,
            "test_id": test.test_id,
            "message": f"A/B test created and started"
        }
    except Exception as e:
        logger.error(f"Error creating A/B test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-test/{test_id}")
async def get_ab_test_results(test_id: str):
    """Get A/B test results"""
    try:
        results = ab_test_manager.get_test_results(test_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Test not found")
        
        return {
            "success": True,
            "test": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting A/B test results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployment-history/{model_name}")
async def get_deployment_history(
    model_name: str,
    limit: int = 20
):
    """Get deployment history for model"""
    try:
        history = model_registry.get_deployment_history(model_name, limit)
        
        return {
            "success": True,
            "model_name": model_name,
            "history": history
        }
    except Exception as e:
        logger.error(f"Error getting deployment history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/canary-health/{model_id}")
async def check_canary_health(model_id: str):
    """Check canary deployment health"""
    try:
        healthy, reason = canary_monitor.check_canary_health(model_id)
        
        return {
            "success": True,
            "model_id": model_id,
            "healthy": healthy,
            "reason": reason
        }
    except Exception as e:
        logger.error(f"Error checking canary health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select-model")
async def select_model(
    model_name: str,
    user_id: str = None,
    region: str = None
):
    """Select best model for prediction"""
    try:
        active_models = model_registry.active_models
        
        if not active_models:
            raise HTTPException(status_code=400, detail="No active models available")
        
        active_dict = {}
        for mid, m in active_models.items():
            entry = {
                "version": m.version,
                "path": m.model_path,
                "status": m.status.value,
            }
            # Key by model_name so name-based lookups work
            active_dict[mid] = entry
            # Also key by model_id (UUID) so feature flags and A/B tests
            # that reference UUIDs can resolve correctly
            active_dict[m.model_id] = entry
        
        selection = model_selector.select_model(
            model_name=model_name,
            active_models=active_dict,
            ab_test_manager=ab_test_manager,
            user_id=user_id,
            region=region
        )
        
        return {
            "success": True,
            "model_id": selection.model_id,
            "model_version": selection.model_version,
            "model_path": selection.model_path,
            "reason": selection.reason,
            "test_id": selection.test_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry")
async def export_registry():
    """Export entire model registry"""
    try:
        registry = model_registry.export_registry()
        return {
            "success": True,
            "registry": registry
        }
    except Exception as e:
        logger.error(f"Error exporting registry: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
