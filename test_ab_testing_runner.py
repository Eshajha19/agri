"""Tests for the A/B testing runner, metrics pipeline, and winner promotion."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch


firebase_mock = MagicMock()
sys.modules.setdefault("firebase_admin", firebase_mock)
sys.modules.setdefault("firebase_admin.firestore", firebase_mock)
sys.modules.setdefault("firebase_admin.credentials", firebase_mock)
sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.cloud", MagicMock())
sys.modules.setdefault("google.cloud.firestore_v1", MagicMock())
sys.modules.setdefault("google.cloud.firestore_v1.base_query", MagicMock())


with patch("feature_flags.flag_store._FIRESTORE_AVAILABLE", False), \
     patch("feature_flags.experiment_engine._FIRESTORE_AVAILABLE", False), \
     patch("feature_flags.metrics_collector._FIRESTORE_AVAILABLE", False):
    from feature_flags import experiment_engine, metrics_collector
    from feature_flags.ab_testing_runner import ABTestingRunner


class TestABTestingRunner(unittest.TestCase):
    def setUp(self):
        experiment_engine._exp_cache.clear()
        experiment_engine._exp_cache_at = 0.0
        experiment_engine._FIRESTORE_AVAILABLE = False
        metrics_collector.clear_events()
        metrics_collector._FIRESTORE_AVAILABLE = False

    def _create_experiment(self, status: str = "running"):
        return experiment_engine.create_experiment({
            "id": "yield_ab",
            "name": "Yield AB",
            "status": status,
            "variants": [
                {"id": "control", "name": "Control", "weight": 50},
                {"id": "treatment", "name": "Treatment", "weight": 50},
            ],
            "salt": "salt_1",
        })

    def test_set_traffic_split_updates_weights(self):
        self._create_experiment()
        runner = ABTestingRunner("yield_ab")

        updated = runner.set_traffic_split({"control": 20, "treatment": 80})

        self.assertEqual(updated["traffic_split"], {"control": 20, "treatment": 80})
        self.assertEqual(updated["variants"][0]["weight"], 20)
        self.assertEqual(updated["variants"][1]["weight"], 80)

    def test_metrics_pipeline_aggregates_in_memory_events(self):
        self._create_experiment()
        runner = ABTestingRunner("yield_ab")

        runner.log_event("impression", "u1", "control")
        runner.log_event("conversion", "u1", "control")
        runner.log_event("impression", "u2", "treatment")

        metrics = runner.get_metrics()

        self.assertEqual(metrics["experiment_id"], "yield_ab")
        self.assertEqual(metrics["total_events"], 3)
        self.assertEqual(metrics["variants"]["control"]["impressions"], 1)
        self.assertEqual(metrics["variants"]["control"]["conversions"], 1)
        self.assertEqual(metrics["variants"]["treatment"]["impressions"], 1)

    def test_auto_promotes_winner_when_lift_is_clear(self):
        self._create_experiment()
        runner = ABTestingRunner(
            "yield_ab",
            min_impressions_per_variant=10,
            min_total_events=20,
            min_absolute_lift_pct=10.0,
        )

        for i in range(20):
            runner.log_event("impression", f"c{i}", "control")
            runner.log_event("impression", f"t{i}", "treatment")

        for i in range(3):
            runner.log_event("conversion", f"c{i}", "control")
        for i in range(12):
            runner.log_event("conversion", f"t{i}", "treatment")

        decision = runner.process()

        self.assertTrue(decision.promoted)
        self.assertEqual(decision.winner_variant, "treatment")

        experiment = experiment_engine.get_experiment("yield_ab")
        self.assertEqual(experiment["status"], "completed")
        self.assertEqual(experiment["winner_variant"], "treatment")
        self.assertEqual(experiment["traffic_split"], {"control": 0, "treatment": 100})

        reassigned = experiment_engine.assign_user("new_user", "yield_ab")
        self.assertEqual(reassigned["variant"], "treatment")
        self.assertEqual(reassigned["reason"], "winner_promoted")


if __name__ == "__main__":
    unittest.main()
