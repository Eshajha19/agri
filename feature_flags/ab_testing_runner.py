"""A/B testing runner with traffic split, metrics pipeline, and promotion logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from feature_flags import experiment_engine, metrics_collector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PromotionDecision:
    experiment_id: str
    winner_variant: Optional[str]
    runner_up_variant: Optional[str]
    promoted: bool
    reason: str
    metrics: Dict[str, Any]


class ABTestingRunner:
    """Coordinates assignments, metrics, and automatic winner promotion."""

    def __init__(
        self,
        experiment_id: str,
        *,
        min_impressions_per_variant: int = 50,
        min_total_events: int = 100,
        min_absolute_lift_pct: float = 5.0,
    ) -> None:
        self.experiment_id = experiment_id
        self.min_impressions_per_variant = max(1, int(min_impressions_per_variant))
        self.min_total_events = max(1, int(min_total_events))
        self.min_absolute_lift_pct = float(min_absolute_lift_pct)

    def get_experiment(self) -> Optional[Dict[str, Any]]:
        return experiment_engine.get_experiment(self.experiment_id)

    def set_traffic_split(self, traffic_split: Dict[str, int]) -> Dict[str, Any]:
        experiment = experiment_engine.set_traffic_split(self.experiment_id, traffic_split)
        if experiment is None:
            raise KeyError(f"Experiment '{self.experiment_id}' not found")
        return experiment

    def assign_user(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assignment = experiment_engine.assign_user(user_id, self.experiment_id)
        metrics_collector.log_event(
            event_type="impression",
            user_id=user_id,
            experiment_id=self.experiment_id,
            variant=assignment.get("variant"),
            metadata={**(metadata or {}), "assignment_reason": assignment.get("reason")},
            session_id=session_id,
        )
        return assignment

    def log_event(
        self,
        event_type: str,
        user_id: str,
        variant: str | None = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        return metrics_collector.log_event(
            event_type=event_type,
            user_id=user_id,
            experiment_id=self.experiment_id,
            variant=variant,
            metadata=metadata,
            session_id=session_id,
        )

    def record_assignment(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assign_user(user_id, session_id=session_id, metadata=metadata)

    def get_metrics(self) -> Dict[str, Any]:
        return metrics_collector.get_experiment_metrics(self.experiment_id)

    def evaluate_winner(self) -> PromotionDecision:
        experiment = self.get_experiment()
        metrics = self.get_metrics()

        if not experiment:
            return PromotionDecision(self.experiment_id, None, None, False, "experiment_not_found", metrics)

        if experiment.get("status") == "completed" and experiment.get("winner_variant"):
            return PromotionDecision(
                self.experiment_id,
                experiment.get("winner_variant"),
                None,
                False,
                "already_promoted",
                metrics,
            )
        if experiment.get("status") != "running":
            return PromotionDecision(
                self.experiment_id,
                None,
                None,
                False,
                f"experiment_status_{experiment.get('status')}",
                metrics,
            )


        variants = metrics.get("variants", {}) or {}
        if len(variants) < 2 or metrics.get("total_events", 0) < self.min_total_events:
            return PromotionDecision(self.experiment_id, None, None, False, "insufficient_data", metrics)

        ranked = sorted(
            variants.items(),
            key=lambda item: (
                item[1].get("conversion_rate", 0.0),
                -item[1].get("errors", 0),
                item[1].get("impressions", 0),
            ),
            reverse=True,
        )

        winner_variant, winner_metrics = ranked[0]
        runner_up_variant, runner_up_metrics = ranked[1]

        if winner_metrics.get("impressions", 0) < self.min_impressions_per_variant:
            return PromotionDecision(self.experiment_id, None, None, False, "winner_under_sampled", metrics)
        if runner_up_metrics.get("impressions", 0) < self.min_impressions_per_variant:
            return PromotionDecision(self.experiment_id, None, None, False, "runner_up_under_sampled", metrics)

        lift = winner_metrics.get("conversion_rate", 0.0) - runner_up_metrics.get("conversion_rate", 0.0)
        if lift < self.min_absolute_lift_pct:
            return PromotionDecision(self.experiment_id, None, runner_up_variant, False, "lift_below_threshold", metrics)

        promoted = experiment_engine.promote_winner(
            self.experiment_id,
            winner_variant,
            reason=f"auto_promotion_lift_{lift:.2f}",
        )
        if promoted is None:
            return PromotionDecision(self.experiment_id, None, runner_up_variant, False, "promotion_failed", metrics)

        logger.info(
            "Promoted winner '%s' for experiment '%s' with lift %.2f",
            winner_variant,
            self.experiment_id,
            lift,
        )
        return PromotionDecision(self.experiment_id, winner_variant, runner_up_variant, True, "promoted", metrics)

    def process(self) -> PromotionDecision:
        return self.evaluate_winner()

    def log_batch(self, events: list[Dict[str, Any]]) -> int:
        normalized = []
        for event in events:
            normalized.append({
                "event_type": event.get("event_type", "custom"),
                "user_id": event.get("user_id", "anonymous"),
                "experiment_id": self.experiment_id,
                "variant": event.get("variant"),
                "flag_id": event.get("flag_id"),
                "metadata": event.get("metadata", {}),
                "session_id": event.get("session_id"),
                "timestamp": event.get("timestamp"),
            })
        return metrics_collector.log_events_batch(normalized)


def get_runner(experiment_id: str, **kwargs: Any) -> ABTestingRunner:
    return ABTestingRunner(experiment_id, **kwargs)
