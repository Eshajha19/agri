"""
Tests for ML Governance Pipeline (Drift Detection, Shadow Evaluation, Rollback Safety)
"""
import pytest
import numpy as np
from datetime import datetime
from ml.governance import DriftDetector, ShadowEvaluator, ModelVersionManager


class TestDriftDetector:
    """Tests for drift detection"""
    
    @pytest.fixture
    def drift_detector(self):
        return DriftDetector(
            window_size=100,
            prediction_drift_threshold=0.2,
            input_drift_threshold=0.15
        )
    
    def test_baseline_setting(self, drift_detector):
        """Test setting baseline statistics"""
        baseline_preds = [100, 102, 98, 101, 99, 103, 100, 102, 98, 99]
        drift_detector.set_baseline('model_v1', baseline_preds)
        
        assert 'model_v1' in drift_detector.baseline_stats
        assert drift_detector.baseline_stats['model_v1']['mean'] > 0
        assert drift_detector.baseline_stats['model_v1']['std'] > 0
    
    def test_no_drift_detection(self, drift_detector):
        """Test normal predictions without drift"""
        baseline_preds = [100.0] * 20
        drift_detector.set_baseline('model_v1', baseline_preds)
        
        # Predictions close to baseline should not trigger drift
        for _ in range(50):
            is_drift, alert = drift_detector.check_prediction_drift('model_v1', 100.5)
        
        # Should not detect drift in stable predictions
        assert len(drift_detector.alerts) == 0 or all(a.severity == 'low' for a in drift_detector.alerts)
    
    def test_prediction_drift_detection(self, drift_detector):
        """Test detection of significant prediction drift"""
        baseline_preds = [100.0] * 10
        drift_detector.set_baseline('model_v1', baseline_preds)
        
        # Add predictions that deviate significantly
        for _ in range(30):
            drift_detector.check_prediction_drift('model_v1', 130.0)  # 30% higher
        
        # Should detect drift eventually
        assert any(a.drift_type == 'prediction' for a in drift_detector.alerts)
    
    def test_input_drift_detection(self, drift_detector):
        """Test detection of input distribution drift"""
        baseline_preds = [100.0] * 10
        drift_detector.set_baseline('model_v1', baseline_preds)
        
        # Record input features that drift
        for i in range(20):
            drift_detector.check_input_drift('model_v1', {'rainfall': 200 + i*2, 'temp': 35})
        
        # Verify input history is being tracked
        assert 'model_v1' in drift_detector.input_history
    
    def test_drift_severity_calculation(self, drift_detector):
        """Test severity level calculation"""
        assert drift_detector._calculate_severity(0.6) == 'high'
        assert drift_detector._calculate_severity(0.35) == 'medium'
        assert drift_detector._calculate_severity(0.1) == 'low'
    
    def test_get_alerts(self, drift_detector):
        """Test retrieving drift alerts"""
        baseline_preds = [100.0] * 10
        drift_detector.set_baseline('model_v1', baseline_preds)
        
        # Generate some alerts
        for _ in range(20):
            drift_detector.check_prediction_drift('model_v1', 150.0)
        
        alerts = drift_detector.get_alerts('model_v1')
        assert len(alerts) > 0
        assert all('timestamp' in a for a in alerts)
    
    def test_clear_alerts(self, drift_detector):
        """Test clearing alerts"""
        baseline_preds = [100.0] * 10
        drift_detector.set_baseline('model_v1', baseline_preds)
        
        for _ in range(20):
            drift_detector.check_prediction_drift('model_v1', 150.0)
        
        assert len(drift_detector.alerts) > 0
        drift_detector.clear_alerts()
        assert len(drift_detector.alerts) == 0


class TestShadowEvaluator:
    """Tests for shadow evaluation"""
    
    @pytest.fixture
    def shadow_evaluator(self):
        return ShadowEvaluator(
            min_samples=10,
            error_improvement_threshold=0.05,
            confidence_threshold=0.85
        )
    
    def test_start_shadow_evaluation(self, shadow_evaluator):
        """Test starting shadow evaluation"""
        eval_id = shadow_evaluator.start_shadow_evaluation('prod_v1', 'candidate_v2')
        
        assert eval_id in shadow_evaluator.active_evaluations
        assert shadow_evaluator.active_evaluations[eval_id]['production_model'] == 'prod_v1'
    
    def test_record_predictions(self, shadow_evaluator):
        """Test recording predictions"""
        eval_id = shadow_evaluator.start_shadow_evaluation('prod_v1', 'candidate_v2')
        
        shadow_evaluator.record_predictions(eval_id, 100.0, 101.0, 102.0)
        shadow_evaluator.record_predictions(eval_id, 105.0, 103.0, 104.0)
        
        session = shadow_evaluator.active_evaluations[eval_id]
        assert len(session['production_predictions']) == 2
        assert len(session['candidate_predictions']) == 2
    
    def test_candidate_promotes_on_improvement(self, shadow_evaluator):
        """Test candidate promotion when error reduces"""
        eval_id = shadow_evaluator.start_shadow_evaluation('prod_v1', 'candidate_v2')
        
        # Production model: error = 5
        # Candidate model: error = 3 (40% improvement)
        for i in range(15):
            shadow_evaluator.record_predictions(
                eval_id,
                100 + i,        # production prediction
                100 + i + 0.5,  # candidate prediction (better)
                100 + i + 5.0,  # actual value
            )
        
        result = shadow_evaluator.evaluate_candidate(eval_id)
        
        assert result is not None
        assert result.candidate_better
        assert result.recommendation == 'promote'
    
    def test_candidate_kept_monitoring(self, shadow_evaluator):
        """Test candidate kept for monitoring when slightly better"""
        eval_id = shadow_evaluator.start_shadow_evaluation('prod_v1', 'candidate_v2')
        
        # Candidate slightly better (2% improvement)
        for i in range(15):
            shadow_evaluator.record_predictions(
                eval_id,
                100 + i + 2.0,
                100 + i + 1.9,  # slightly better
                100 + i,
            )
        
        result = shadow_evaluator.evaluate_candidate(eval_id)
        
        assert result is not None
        assert not result.candidate_better
        assert result.recommendation == 'keep_monitoring'
    
    def test_candidate_rejected_on_worse_performance(self, shadow_evaluator):
        """Test candidate rejection when performance is worse"""
        eval_id = shadow_evaluator.start_shadow_evaluation('prod_v1', 'candidate_v2')
        
        # Production: error = 0.5, Candidate: error = 2.0 (worse)
        for i in range(15):
            shadow_evaluator.record_predictions(
                eval_id,
                100 + i,
                100 + i + 2.0,  # much worse
                100 + i + 0.5,
            )
        
        result = shadow_evaluator.evaluate_candidate(eval_id)
        
        assert result is not None
        assert not result.candidate_better
        assert result.recommendation == 'reject'
    
    def test_evaluation_status(self, shadow_evaluator):
        """Test checking evaluation status"""
        eval_id = shadow_evaluator.start_shadow_evaluation('prod_v1', 'candidate_v2')
        
        status = shadow_evaluator.get_evaluation_status(eval_id)
        assert status['status'] == 'in_progress'
        assert status['samples_collected'] == 0
        assert status['ready_for_evaluation'] is False
        
        # Add samples
        for i in range(15):
            shadow_evaluator.record_predictions(eval_id, 100, 101, 102)
        
        status = shadow_evaluator.get_evaluation_status(eval_id)
        assert status['samples_collected'] == 15
        assert status['ready_for_evaluation'] is True
    
    def test_cleanup_evaluation(self, shadow_evaluator):
        """Test cleanup of evaluation session"""
        eval_id = shadow_evaluator.start_shadow_evaluation('prod_v1', 'candidate_v2')
        assert eval_id in shadow_evaluator.active_evaluations
        
        shadow_evaluator.cleanup_evaluation(eval_id)
        assert eval_id not in shadow_evaluator.active_evaluations


class TestModelVersionManager:
    """Tests for model versioning and rollback"""
    
    @pytest.fixture
    def version_manager(self, tmp_path):
        return ModelVersionManager(versions_dir=str(tmp_path / 'versions'))
    
    def test_register_version(self, version_manager):
        """Test registering a new model version"""
        version_id = version_manager.register_version(
            model_name='xgboost',
            model_path='/path/to/model.joblib',
            performance_metrics={'rmse': 0.15, 'r2': 0.85},
            metadata={'author': 'ml-team'}
        )
        
        assert version_id in version_manager.versions
        assert not version_manager.versions[version_id].is_production
    
    def test_promote_version(self, version_manager):
        """Test promoting a version to production"""
        version_id = version_manager.register_version(
            'xgboost',
            '/path/to/model.joblib',
            {'rmse': 0.15, 'r2': 0.85}
        )
        
        success = version_manager.promote_version(version_id)
        
        assert success
        assert version_manager.production_version == version_id
        assert version_manager.versions[version_id].is_production
    
    def test_rollback_to_previous_version(self, version_manager):
        """Test rollback to previous version"""
        # Register and promote v1
        v1_id = version_manager.register_version(
            'xgboost',
            '/path/to/model_v1.joblib',
            {'rmse': 0.20, 'r2': 0.80}
        )
        version_manager.promote_version(v1_id)
        
        # Register and promote v2
        v2_id = version_manager.register_version(
            'xgboost',
            '/path/to/model_v2.joblib',
            {'rmse': 0.15, 'r2': 0.85}
        )
        version_manager.promote_version(v2_id)
        
        assert version_manager.production_version == v2_id
        
        # Rollback to v1
        version_manager.rollback_to_version(v1_id)
        
        assert version_manager.production_version == v1_id
        assert version_manager.versions[v1_id].is_production
    
    def test_get_production_version(self, version_manager):
        """Test getting production version"""
        assert version_manager.get_production_version() is None
        
        version_id = version_manager.register_version(
            'xgboost',
            '/path/to/model.joblib',
            {'rmse': 0.15}
        )
        version_manager.promote_version(version_id)
        
        prod_version = version_manager.get_production_version()
        assert prod_version is not None
        assert prod_version.version_id == version_id
    
    def test_list_versions(self, version_manager):
        """Test listing versions"""
        v1 = version_manager.register_version('xgboost', '/path/v1.joblib', {'rmse': 0.20})
        v2 = version_manager.register_version('xgboost', '/path/v2.joblib', {'rmse': 0.15})
        v3 = version_manager.register_version('lstm', '/path/v3.joblib', {'rmse': 0.18})
        
        all_versions = version_manager.list_versions()
        assert len(all_versions) == 3
        
        xgboost_versions = version_manager.list_versions(model_name='xgboost')
        assert len(xgboost_versions) == 2
    
    def test_compare_versions(self, version_manager):
        """Test comparing two versions"""
        v1 = version_manager.register_version('xgboost', '/path/v1.joblib', {'rmse': 0.20, 'r2': 0.80})
        v2 = version_manager.register_version('xgboost', '/path/v2.joblib', {'rmse': 0.15, 'r2': 0.85})
        
        comparison = version_manager.compare_versions(v1, v2)
        
        assert 'metrics_comparison' in comparison
        assert 'rmse' in comparison['metrics_comparison']
        assert comparison['metrics_comparison']['rmse']['difference'] == -0.05
    
    def test_cleanup_old_versions(self, version_manager):
        """Test cleanup of old versions"""
        versions = []
        for i in range(10):
            v = version_manager.register_version(
                'xgboost',
                f'/path/v{i}.joblib',
                {'rmse': 0.20 - i*0.01}
            )
            versions.append(v)
        
        # Keep only 3 recent versions
        deleted = version_manager.cleanup_old_versions(keep_count=3)
        
        assert deleted == 7
        assert len([v for v in version_manager.list_versions(model_name='xgboost')]) == 3
    
    def test_version_history(self, version_manager):
        """Test version management history"""
        v1 = version_manager.register_version('xgboost', '/path/v1.joblib', {'rmse': 0.20})
        version_manager.promote_version(v1)
        
        v2 = version_manager.register_version('xgboost', '/path/v2.joblib', {'rmse': 0.15})
        version_manager.promote_version(v2)
        version_manager.rollback_to_version(v1)
        
        history = version_manager.get_version_history()
        
        assert any(h['action'] == 'register' for h in history)
        assert any(h['action'] == 'promote' for h in history)
        assert any(h['action'] == 'rollback' for h in history)


class TestGovernanceRbacRoutes:
    """RBAC enforcement for ML governance HTTP routes."""

    @pytest.fixture
    def app(self):
        from fastapi import FastAPI
        from backend.routers.governance import router, init_governance

        app = FastAPI()
        app.include_router(router, prefix="/api/ml-governance")

        mock_dd = DriftDetector()
        mock_se = ShadowEvaluator(min_samples=5)
        mock_vm = ModelVersionManager(versions_dir="./test_versions_rbac")

        async def allow_auth_fn(request, required_roles=None):
            return {"uid": "test-user", "role": "admin"}

        init_governance(mock_dd, mock_se, mock_vm, auth_fn=allow_auth_fn)
        app.state._mock_dd = mock_dd
        app.state._mock_se = mock_se
        app.state._mock_vm = mock_vm
        yield app

    @pytest.fixture
    def client(self, app):
        from fastapi.testclient import TestClient
        yield TestClient(app)

    AUTH_ROUTES = {
        "GET": ["/api/ml-governance/drift/alerts"],
        "POST": [
            "/api/ml-governance/drift/check",
            "/api/ml-governance/shadow/record",
            "/api/ml-governance/versions/list",
        ],
    }
    ADMIN_ROUTES = {
        "POST": [
            "/api/ml-governance/drift/baseline",
            "/api/ml-governance/shadow/start",
            "/api/ml-governance/shadow/evaluate",
            "/api/ml-governance/versions/register",
            "/api/ml-governance/versions/promote",
            "/api/ml-governance/versions/rollback",
        ],
        "GET": [
            "/api/ml-governance/versions/production",
            "/api/ml-governance/versions/compare",
            "/api/ml-governance/status",
        ],
    }

    def test_unauthenticated_requests_return_401(self, client):
        for method, routes in {**self.AUTH_ROUTES, **self.ADMIN_ROUTES}.items():
            for route in routes:
                resp = client.request(method, route)
                assert resp.status_code in (401, 403, 503), (
                    f"{method} {route} expected 401/403/503, got {resp.status_code}"
                )

    def test_unauthorized_role_returns_403(self, client, app):
        from backend.routers.governance import init_governance
        from fastapi import HTTPException

        async def deny_auth_fn(request, required_roles=None):
            raise HTTPException(status_code=403, detail="Access denied")

        init_governance(
            app.state._mock_dd, app.state._mock_se, app.state._mock_vm,
            auth_fn=deny_auth_fn,
        )

        for method, routes in self.ADMIN_ROUTES.items():
            for route in routes:
                resp = client.request(
                    method, route,
                    headers={"Authorization": "Bearer faketoken"},
                )
                assert resp.status_code == 403, (
                    f"{method} {route} expected 403, got {resp.status_code}"
                )


class TestGovernancePipeline:
    """Integration tests for complete governance pipeline"""
    
    def test_full_governance_workflow(self):
        """Test complete workflow: register -> shadow eval -> promote -> drift detection"""
        drift_detector = DriftDetector()
        shadow_evaluator = ShadowEvaluator(min_samples=5)
        version_manager = ModelVersionManager(versions_dir='./test_versions')
        
        # 1. Register versions
        prod_v = version_manager.register_version('xgboost', '/prod.joblib', {'rmse': 0.20})
        cand_v = version_manager.register_version('xgboost', '/cand.joblib', {'rmse': 0.15})
        
        # 2. Promote production version
        version_manager.promote_version(prod_v)
        
        # 3. Shadow evaluation
        eval_id = shadow_evaluator.start_shadow_evaluation(prod_v, cand_v)
        for i in range(10):
            shadow_evaluator.record_predictions(eval_id, 100 + i, 100.5 + i, 100 + i + 0.2)
        
        result = shadow_evaluator.evaluate_candidate(eval_id)
        assert result.recommendation == 'promote'
        
        # 4. Promote candidate
        version_manager.promote_version(cand_v)
        assert version_manager.production_version == cand_v
        
        # 5. Setup drift detection
        baseline_preds = [100 + i for i in range(20)]
        drift_detector.set_baseline(cand_v, baseline_preds)
        
        # 6. Monitor predictions (no drift)
        for i in range(15):
            drift_detector.check_prediction_drift(cand_v, 100.5 + i)
        
        alerts = drift_detector.get_alerts(cand_v)
        # With stable predictions, should have minimal alerts
        assert len(alerts) < 5
        
        print("✅ Full governance pipeline workflow completed successfully")


class TestGovernanceAuditLogging:
    """Audit events are emitted on register / promote / rollback."""

    @pytest.fixture
    def app(self):
        from fastapi import FastAPI
        from backend.routers.governance import router, init_governance

        app = FastAPI()
        app.include_router(router, prefix="/api/ml-governance")

        mock_dd = DriftDetector()
        mock_se = ShadowEvaluator(min_samples=5)
        mock_vm = ModelVersionManager(versions_dir="./test_versions_audit")

        async def auth_fn(request, required_roles=None):
            return {"uid": "auditor", "role": "admin"}

        init_governance(mock_dd, mock_se, mock_vm, auth_fn=auth_fn)
        app.state._mock_dd = mock_dd
        app.state._mock_se = mock_se
        app.state._mock_vm = mock_vm
        yield app

    @pytest.fixture
    def client(self, app):
        from fastapi.testclient import TestClient
        yield TestClient(app)

    def _reg_version(self, client, model="xgboost"):
        resp = client.post(
            "/api/ml-governance/versions/register",
            json={"model_name": model, "model_path": "/m.joblib", "rmse": 0.15},
            headers={"Authorization": "Bearer t"},
        )
        assert resp.status_code == 200
        return resp.json()["version_id"]

    def test_register_emits_audit_event(self, client):
        from rbac_audit import rbac_audit_trail

        rbac_audit_trail.clear()
        self._reg_version(client)

        snap = rbac_audit_trail.snapshot(limit=10)
        actions = [e["action"] for e in snap]
        assert "model_version.register" in actions

    def test_promote_emits_audit_event(self, client):
        from rbac_audit import rbac_audit_trail

        rbac_audit_trail.clear()
        vid = self._reg_version(client)

        client.post(
            "/api/ml-governance/versions/promote",
            params={"version_id": vid},
            headers={"Authorization": "Bearer t"},
        )

        snap = rbac_audit_trail.snapshot(limit=10)
        actions = [e["action"] for e in snap]
        assert "model_version.promote" in actions

    def test_rollback_emits_audit_event(self, client):
        from rbac_audit import rbac_audit_trail

        rbac_audit_trail.clear()
        v1 = self._reg_version(client)
        v2 = self._reg_version(client, model="xgboost")

        client.post(
            "/api/ml-governance/versions/promote",
            params={"version_id": v2},
            headers={"Authorization": "Bearer t"},
        )
        client.post(
            "/api/ml-governance/versions/rollback",
            params={"version_id": v1},
            headers={"Authorization": "Bearer t"},
        )

        snap = rbac_audit_trail.snapshot(limit=10)
        actions = [e["action"] for e in snap]
        assert "model_version.rollback" in actions

    def test_audit_event_contains_actor_uid(self, client):
        from rbac_audit import rbac_audit_trail

        rbac_audit_trail.clear()
        self._reg_version(client)

        snap = rbac_audit_trail.snapshot(limit=10)
        register_events = [e for e in snap if e["action"] == "model_version.register"]
        assert len(register_events) >= 1
        assert register_events[0]["uid"] == "auditor"
        assert register_events[0]["role"] == "admin"

    def test_audit_event_reason_contains_version_id(self, client):
        from rbac_audit import rbac_audit_trail

        rbac_audit_trail.clear()
        vid = self._reg_version(client)

        snap = rbac_audit_trail.snapshot(limit=10)
        register_events = [e for e in snap if e["action"] == "model_version.register"]
        assert len(register_events) >= 1
        reason = register_events[0].get("reason", "")
        assert vid in reason
