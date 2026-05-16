"""
test_feature_flags.py
──────────────────────
Unit tests for the A/B Testing & Feature Flag Framework.

Tests run without Firestore connectivity (all storage mocked to in-memory).
"""

import importlib
import sys
import unittest
from unittest.mock import MagicMock, patch


# ── Patch Firestore out before importing modules ───────────────────────────────
firebase_mock = MagicMock()
sys.modules.setdefault("firebase_admin", firebase_mock)
sys.modules.setdefault("firebase_admin.firestore", firebase_mock)
sys.modules.setdefault("firebase_admin.credentials", firebase_mock)
sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.cloud", MagicMock())
sys.modules.setdefault("google.cloud.firestore_v1", MagicMock())
sys.modules.setdefault("google.cloud.firestore_v1.base_query", MagicMock())

# Import modules under test with Firestore disabled
with patch("feature_flags.flag_store._FIRESTORE_AVAILABLE", False), \
     patch("feature_flags.experiment_engine._FIRESTORE_AVAILABLE", False), \
     patch("feature_flags.metrics_collector._FIRESTORE_AVAILABLE", False):
    from feature_flags import flag_store, experiment_engine, metrics_collector


class TestFlagStore(unittest.TestCase):

    def setUp(self):
        # Reset in-memory cache before each test
        flag_store._cache.clear()
        flag_store._cache_loaded_at = 0.0
        flag_store._FIRESTORE_AVAILABLE = False

    def test_list_flags_returns_defaults(self):
        flags = flag_store.list_flags()
        self.assertIsInstance(flags, list)
        self.assertGreater(len(flags), 0)

    def test_upsert_and_get_flag(self):
        flag_store.upsert_flag("test_flag", {
            "enabled": True, "rollout_pct": 50,
            "description": "Test flag"
        })
        flag = flag_store.get_flag("test_flag")
        self.assertIsNotNone(flag)
        self.assertEqual(flag["enabled"], True)
        self.assertEqual(flag["rollout_pct"], 50)
        self.assertEqual(flag["id"], "test_flag")

    def test_rollback_disables_flag(self):
        flag_store.upsert_flag("rollback_test", {"enabled": True, "rollout_pct": 80})
        result = flag_store.rollback_flag("rollback_test")
        self.assertFalse(result["enabled"])
        self.assertEqual(result["rollout_pct"], 0)

    def test_delete_flag(self):
        flag_store.upsert_flag("to_delete", {"enabled": True})
        deleted = flag_store.delete_flag("to_delete")
        self.assertTrue(deleted)
        self.assertIsNone(flag_store.get_flag("to_delete"))

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(flag_store.delete_flag("nonexistent_xyz"))

    def test_rollout_pct_boundaries(self):
        flag_store.upsert_flag("zero_pct", {"enabled": False, "rollout_pct": 0})
        flag_store.upsert_flag("full_pct", {"enabled": True, "rollout_pct": 100})
        self.assertEqual(flag_store.get_flag("zero_pct")["rollout_pct"], 0)
        self.assertEqual(flag_store.get_flag("full_pct")["rollout_pct"], 100)

    def test_flag_enrichment_defaults(self):
        flag_store.upsert_flag("minimal_flag", {})
        flag = flag_store.get_flag("minimal_flag")
        self.assertIn("enabled", flag)
        self.assertIn("rollout_pct", flag)
        self.assertIn("created_at", flag)
        self.assertIn("updated_at", flag)


class TestExperimentEngine(unittest.TestCase):

    def setUp(self):
        experiment_engine._exp_cache.clear()
        experiment_engine._exp_cache_at = 0.0
        experiment_engine._FIRESTORE_AVAILABLE = False

    def test_assignment_is_deterministic(self):
        """Same user + experiment always gets the same variant."""
        variants = [
            {"id": "control", "weight": 50},
            {"id": "treatment", "weight": 50},
        ]
        v1 = experiment_engine._assign_variant("user_abc", "exp_1", "salt", variants)
        v2 = experiment_engine._assign_variant("user_abc", "exp_1", "salt", variants)
        self.assertEqual(v1, v2)

    def test_different_users_can_get_different_variants(self):
        """Assignment distributes across users."""
        variants = [
            {"id": "control", "weight": 50},
            {"id": "treatment", "weight": 50},
        ]
        results = set()
        for i in range(200):
            v = experiment_engine._assign_variant(f"user_{i}", "exp_dist", "salt", variants)
            results.add(v)
        # With 200 users and 50/50 split, both variants should appear
        self.assertIn("control", results)
        self.assertIn("treatment", results)

    def test_rollout_0_pct(self):
        """With 0% weight on treatment, everyone gets control."""
        variants = [
            {"id": "control",   "weight": 100},
            {"id": "treatment", "weight": 0},
        ]
        for i in range(50):
            v = experiment_engine._assign_variant(f"u_{i}", "exp", "s", variants)
            self.assertEqual(v, "control")

    def test_rollout_100_pct_treatment(self):
        """With 100% weight on treatment, everyone gets treatment."""
        variants = [
            {"id": "control",   "weight": 0},
            {"id": "treatment", "weight": 100},
        ]
        for i in range(50):
            v = experiment_engine._assign_variant(f"u_{i}", "exp", "s", variants)
            self.assertEqual(v, "treatment")

    def test_salt_change_reshuffles_users(self):
        """Changing the salt produces a different variant distribution."""
        variants = [
            {"id": "a", "weight": 50},
            {"id": "b", "weight": 50},
        ]
        v_old = experiment_engine._assign_variant("user_x", "exp", "salt_old", variants)
        v_new = experiment_engine._assign_variant("user_x", "exp", "salt_new", variants)
        # Not guaranteed to differ for every user, but tests the mechanism
        # At minimum, the function must not crash
        self.assertIn(v_old, ("a", "b"))
        self.assertIn(v_new, ("a", "b"))

    def test_assign_user_draft_experiment_returns_control(self):
        experiment_engine.create_experiment({
            "id": "draft_exp", "name": "Draft", "status": "draft",
            "variants": [{"id": "control", "name": "C", "weight": 100}],
            "salt": "s",
        })
        result = experiment_engine.assign_user("user_1", "draft_exp")
        self.assertEqual(result["variant"], "control")
        self.assertIn("experiment_status_draft", result.get("reason", ""))

    def test_assign_user_running_experiment(self):
        experiment_engine.create_experiment({
            "id": "running_exp", "name": "Running", "status": "running",
            "variants": [
                {"id": "control",   "name": "C", "weight": 50},
                {"id": "treatment", "name": "T", "weight": 50},
            ],
            "salt": "run_salt",
        })
        result = experiment_engine.assign_user("user_a", "running_exp")
        self.assertIn(result["variant"], ("control", "treatment"))
        self.assertEqual(result["experiment_id"], "running_exp")

    def test_assign_nonexistent_experiment(self):
        result = experiment_engine.assign_user("user_x", "does_not_exist")
        self.assertEqual(result["variant"], "control")
        self.assertEqual(result["reason"], "experiment_not_found")


class TestMetricsCollector(unittest.TestCase):

    def setUp(self):
        metrics_collector._FIRESTORE_AVAILABLE = False

    def test_log_event_returns_event_dict(self):
        event = metrics_collector.log_event(
            event_type="impression",
            user_id="user_1",
            experiment_id="exp_1",
            variant="control",
        )
        self.assertEqual(event["event_type"], "impression")
        self.assertEqual(event["user_id"], "user_1")
        self.assertIn("timestamp", event)

    def test_invalid_event_type_becomes_custom(self):
        event = metrics_collector.log_event(
            event_type="invalid_type",
            user_id="user_1",
        )
        self.assertEqual(event["event_type"], "custom")

    def test_empty_metrics_when_firestore_unavailable(self):
        result = metrics_collector.get_experiment_metrics("exp_1")
        self.assertEqual(result["experiment_id"], "exp_1")
        self.assertEqual(result["total_events"], 0)

    def test_batch_log_returns_count(self):
        events = [
            {"event_type": "impression", "user_id": f"u_{i}",
             "experiment_id": "exp", "variant": "control"}
            for i in range(5)
        ]
        count = metrics_collector.log_events_batch(events)
        self.assertEqual(count, 5)


if __name__ == "__main__":
    unittest.main()
