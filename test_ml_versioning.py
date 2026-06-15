"""
Tests for ML Model Versioning & A/B Testing Infrastructure
"""

import pytest
from ml_model_registry import (
    ModelRegistry, ModelVersion, ModelStatus, get_model_registry
)
from ml_ab_testing import (
    ABTest, Arm, ABTestManager, TestStatus, get_ab_test_manager
)
from ml_metrics import (
    MetricsCollector, CanaryMonitor, get_metrics_collector
)
from ml_model_selector import ModelSelector, get_model_selector


class TestModelVersion:
    """Test ModelVersion class"""
    
    def test_create_model_version(self):
        """Test creating a model version"""
        model = ModelVersion(
            model_name="yield_prediction",
            version="1.0",
            model_path="/models/yield_v1.0.joblib",
            created_by="data_team"
        )
        
        assert model.model_name == "yield_prediction"
        assert model.version == "1.0"
        assert model.status == ModelStatus.DRAFT
        assert model.model_id is not None
    
    def test_model_version_to_dict(self):
        """Test converting model to dictionary"""
        model = ModelVersion(
            model_name="yield_prediction",
            version="1.0",
            model_path="/models/yield_v1.0.joblib"
        )
        
        data = model.to_dict()
        assert data["model_name"] == "yield_prediction"
        assert data["status"] == "draft"
    
    def test_model_version_from_dict(self):
        """Test creating model from dictionary"""
        original = ModelVersion(
            model_name="yield_prediction",
            version="1.0",
            model_path="/models/yield_v1.0.joblib"
        )
        
        data = original.to_dict()
        restored = ModelVersion.from_dict(data)
        
        assert restored.model_name == original.model_name
        assert restored.version == original.version


class TestModelRegistry:
    """Test ModelRegistry"""
    
    def test_register_model(self):
        """Test registering a model"""
        registry = ModelRegistry()
        
        model = registry.register_model(
            model_name="yield_prediction",
            version="1.0",
            model_path="/models/yield_v1.0.joblib"
        )
        
        assert model.model_id is not None
        assert registry.get_model_version("yield_prediction", "1.0") == model
    
    def test_promote_to_canary(self):
        """Test promoting model to canary"""
        registry = ModelRegistry()
        
        model = registry.register_model(
            model_name="yield_prediction",
            version="1.0",
            model_path="/models/yield_v1.0.joblib"
        )
        
        success = registry.promote_to_canary("yield_prediction", "1.0", 5)
        assert success
        assert model.status == ModelStatus.CANARY
        assert model.canary_traffic_percentage == 5
    
    def test_promote_to_production(self):
        """Test promoting model to production"""
        registry = ModelRegistry()
        
        model = registry.register_model(
            model_name="yield_prediction",
            version="1.0",
            model_path="/models/yield_v1.0.joblib"
        )
        
        success = registry.promote_to_production("yield_prediction", "1.0")
        assert success
        assert model.status == ModelStatus.PRODUCTION
        assert registry.get_active_model("yield_prediction") == model
    
    def test_promote_to_production_rollback_tracking(self):
        """Test that promoting a new version to production records rollback metadata"""
        registry = ModelRegistry()
        
        model_v1 = registry.register_model(
            model_name="yield_prediction",
            version="1.0",
            model_path="/models/yield_v1.0.joblib"
        )
        registry.promote_to_production("yield_prediction", "1.0")
        
        model_v2 = registry.register_model(
            model_name="yield_prediction",
            version="2.0",
            model_path="/models/yield_v2.0.joblib"
        )
        success = registry.promote_to_production("yield_prediction", "2.0")
        
        assert success
        assert model_v2.status == ModelStatus.PRODUCTION
        assert model_v1.status == ModelStatus.ARCHIVED
        assert model_v2.replaced_version == "1.0"
        assert model_v2.rollback_reference == "1.0"
        
        # Verify deployment history on the model
        assert any(
            h["action"] == "production_promotion" and h["replaced_version"] == "1.0"
            for h in model_v2.deployment_history
        )
        
        # Verify central deployment log
        history = registry.get_deployment_history("yield_prediction")
        production_log = [h for h in history if h["action"] == "production" and h["version"] == "2.0"][0]
        assert production_log["replaced_version"] == "1.0"
        assert production_log["rollback_reference"] == "1.0"
        assert production_log["metadata"]["previous_status"] == "archived"

    def test_rollback_using_metadata(self):
        """Test that rollback correctly identifies previous version via metadata"""
        registry = ModelRegistry()
        
        model_v1 = registry.register_model(
            model_name="yield_prediction",
            version="1.0",
            model_path="/models/yield_v1.0.joblib"
        )
        registry.promote_to_production("yield_prediction", "1.0")
        
        model_v2 = registry.register_model(
            model_name="yield_prediction",
            version="2.0",
            model_path="/models/yield_v2.0.joblib"
        )
        registry.promote_to_production("yield_prediction", "2.0")
        
        # Trigger rollback
        success = registry.rollback("yield_prediction", reason="high latency")
        
        assert success
        assert registry.get_active_model("yield_prediction") == model_v1
        assert model_v1.status == ModelStatus.PRODUCTION
        assert model_v2.status == ModelStatus.ROLLED_BACK
        assert model_v2.rollback_reason == "high latency"

    def test_deployment_history(self):
        """Test tracking deployment history"""
        registry = ModelRegistry()
        
        model = registry.register_model(
            model_name="yield_prediction",
            version="1.0",
            model_path="/models/yield_v1.0.joblib"
        )
        
        registry.promote_to_canary("yield_prediction", "1.0", 5)
        registry.promote_to_production("yield_prediction", "1.0")
        
        history = registry.get_deployment_history("yield_prediction")
        assert len(history) >= 2
        assert history[-1]["action"] in ["canary", "production"]


class TestArm:
    """Test A/B test Arm"""
    
    def test_create_arm(self):
        """Test creating an arm"""
        arm = Arm(model_id="model_1", model_version="1.0", name="Control")
        
        assert arm.model_id == "model_1"
        assert arm.successes == 0
        assert arm.failures == 0
    
    def test_record_prediction(self):
        """Test recording prediction metrics"""
        arm = Arm(model_id="model_1", model_version="1.0")
        
        arm.record_prediction(mae=0.5, rmse=0.25, latency=50)
        
        assert arm.predictions == 1
        assert arm.mae_sum == 0.5
        assert arm.latency_sum == 50
    
    def test_get_mean_metric(self):
        """Test getting mean metrics"""
        arm = Arm(model_id="model_1", model_version="1.0")
        
        arm.record_prediction(mae=0.5, rmse=0.25, latency=50)
        arm.record_prediction(mae=0.6, rmse=0.36, latency=60)
        
        assert abs(arm.get_mean_metric("mae") - 0.55) < 0.01
        assert abs(arm.get_mean_metric("latency") - 55) < 1
    
    def test_record_outcome(self):
        """Test recording trial outcome"""
        arm = Arm(model_id="model_1", model_version="1.0")
        
        arm.record_outcome(True)
        arm.record_outcome(True)
        arm.record_outcome(False)
        
        assert arm.successes == 2
        assert arm.failures == 1
        assert arm.total_trials == 3


class TestABTest:
    """Test A/B Testing"""
    
    def test_create_ab_test(self):
        """Test creating A/B test"""
        control = Arm("model_1", "1.0", "Control")
        variant = Arm("model_2", "2.0", "Variant")
        
        test = ABTest(
            test_name="yield_model_test",
            model_name="yield_prediction",
            control_arm=control,
            variant_arm=variant
        )
        
        assert test.test_name == "yield_model_test"
        assert test.status == TestStatus.SETUP
    
    def test_start_ab_test(self):
        """Test starting A/B test"""
        control = Arm("model_1", "1.0", "Control")
        variant = Arm("model_2", "2.0", "Variant")
        
        test = ABTest(
            test_name="test",
            model_name="model",
            control_arm=control,
            variant_arm=variant
        )
        
        test.start()
        assert test.status == TestStatus.RUNNING
    
    def test_select_arm_thompson_sampling(self):
        """Test Thompson sampling arm selection"""
        control = Arm("model_1", "1.0", "Control")
        variant = Arm("model_2", "2.0", "Variant")
        
        # Give variant more successes
        for _ in range(10):
            variant.record_outcome(True)
        
        for _ in range(3):
            control.record_outcome(True)
        
        test = ABTest(
            test_name="test",
            model_name="model",
            control_arm=control,
            variant_arm=variant
        )
        
        test.start()
        
        # Select arms multiple times - variant should be selected more often
        selections = [test.select_arm() for _ in range(100)]
        variant_count = sum(1 for arm in selections if arm == variant)
        
        # Variant should be selected more often (but not always due to randomness)
        assert variant_count > 30


class TestABTestManager:
    """Test A/B Test Manager"""
    
    def test_create_test(self):
        """Test creating test"""
        manager = ABTestManager()
        
        control = Arm("model_1", "1.0", "Control")
        variant = Arm("model_2", "2.0", "Variant")
        
        test = manager.create_test(
            test_name="test",
            model_name="model",
            control_arm=control,
            variant_arm=variant
        )
        
        assert test.test_id in manager.active_tests
    
    def test_record_outcome(self):
        """Test recording outcome"""
        manager = ABTestManager()
        
        control = Arm("model_1", "1.0", "Control")
        variant = Arm("model_2", "2.0", "Variant")
        
        test = manager.create_test(
            test_name="test",
            model_name="model",
            control_arm=control,
            variant_arm=variant,
            min_samples=5
        )
        
        manager.start_test(test.test_id)
        
        # Record outcomes
        for _ in range(5):
            manager.record_outcome(
                test.test_id,
                control.model_id,
                True,
                {"mae": 0.1, "rmse": 0.01, "latency": 50}
            )


class TestMetricsCollector:
    """Test Metrics Collection"""
    
    def test_record_prediction(self):
        """Test recording prediction"""
        collector = MetricsCollector()
        
        collector.record_prediction(
            model_id="model_1",
            model_version="1.0",
            y_true=100,
            y_pred=98,
            latency=50
        )
        
        assert len(collector.predictions["model_1"]) == 1
    
    def test_get_metrics(self):
        """Test getting metrics"""
        collector = MetricsCollector()
        
        for i in range(10):
            collector.record_prediction(
                model_id="model_1",
                model_version="1.0",
                y_true=100,
                y_pred=98 + i * 0.1,
                latency=50 + i
            )
        
        metrics = collector.get_metrics("model_1")
        
        assert metrics["predictions"] == 10
        assert metrics["mae"] is not None
        assert metrics["latency"] is not None
    
    def test_compare_models(self):
        """Test comparing two models"""
        collector = MetricsCollector()
        
        # Model 1 - better performance
        for _ in range(10):
            collector.record_prediction("model_1", "1.0", 100, 99, 50)
        
        # Model 2 - worse performance
        for _ in range(10):
            collector.record_prediction("model_2", "1.0", 100, 95, 100)
        
        comparison = collector.compare_models("model_1", "model_2")
        
        assert comparison["model_1_better_mae"]
        assert comparison["model_1_better_latency"]


class TestCanaryMonitor:
    """Test Canary Monitoring"""
    
    def test_start_canary(self):
        """Test starting canary"""
        collector = MetricsCollector()
        monitor = CanaryMonitor(collector)
        
        monitor.start_canary(
            model_id="model_new",
            baseline_model_id="model_old",
            traffic_percentage=5
        )
        
        assert "model_new" in monitor.canary_deployments
        assert monitor.canary_deployments["model_new"]["status"] == "monitoring"
    
    def test_check_canary_health(self):
        """Test checking canary health"""
        collector = MetricsCollector()
        monitor = CanaryMonitor(collector)
        
        # Record baseline
        for _ in range(20):
            collector.record_prediction("model_old", "1.0", 100, 99, 50)
        
        monitor.start_canary("model_new", "model_old", 5)
        
        # Record canary data
        for _ in range(20):
            collector.record_prediction("model_new", "2.0", 100, 99.1, 50)
        
        healthy, reason = monitor.check_canary_health("model_new")
        assert healthy  # Should be healthy since performance is similar


class TestModelSelector:
    """Test Model Selector"""
    
    def test_register_feature_flag(self):
        """Test registering feature flag"""
        selector = ModelSelector()
        
        selector.register_feature_flag(
            flag_name="use_v2",
            model_id="model_v2",
            enabled=True,
            rollout_percentage=10
        )
        
        assert "use_v2" in selector.feature_flags
    
    def test_is_flag_enabled(self):
        """Test checking flag"""
        selector = ModelSelector()
        
        selector.register_feature_flag(
            flag_name="use_v2",
            model_id="model_v2",
            enabled=True,
            rollout_percentage=100
        )
        
        assert selector.is_flag_enabled("use_v2", user_id="user_1")
    
    def test_segment_user(self):
        """Test user segmentation"""
        selector = ModelSelector()
        
        selector.segment_user("user_1", "beta_testers")
        selector.register_feature_flag(
            flag_name="beta_feature",
            model_id="model_beta",
            enabled=True,
            user_segments=["beta_testers"]
        )
        
        assert selector.is_flag_enabled("beta_feature", "user_1")


class TestGlobalSingletons:
    """Test global singleton instances"""
    
    def test_get_model_registry(self):
        """Test getting global model registry"""
        registry1 = get_model_registry()
        registry2 = get_model_registry()
        
        assert registry1 is registry2
    
    def test_get_ab_test_manager(self):
        """Test getting global A/B test manager"""
        manager1 = get_ab_test_manager()
        manager2 = get_ab_test_manager()
        
        assert manager1 is manager2
    
    def test_get_metrics_collector(self):
        """Test getting global metrics collector"""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2
    
    def test_get_model_selector(self):
        """Test getting global model selector"""
        selector1 = get_model_selector()
        selector2 = get_model_selector()
        
        assert selector1 is selector2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
