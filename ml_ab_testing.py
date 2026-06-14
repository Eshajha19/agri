"""A/B Testing Engine with Multi-Armed Bandits
Implements Thompson sampling for optimal model allocation."""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    SETUP = "setup"
    RUNNING = "running"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class Arm:
    model_id: str
    model_version: str
    name: Optional[str] = None
    successes: int = 0
    failures: int = 0
    total_trials: int = 0
    predictions: int = 0
    mae_sum: float = 0.0
    rmse_sum: float = 0.0
    latency_sum: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        self.name = self.name or self.model_version

    @property
    def alpha(self) -> int:
        return self.successes + 1

    @property
    def beta(self) -> int:
        return self.failures + 1

    @property
    def mean_mae(self) -> float:
        return self.mae_sum / self.predictions if self.predictions else 0.0

    @property
    def mean_rmse(self) -> float:
        return self.rmse_sum / self.predictions if self.predictions else 0.0

    @property
    def mean_latency(self) -> float:
        return self.latency_sum / self.predictions if self.predictions else 0.0

    def record_prediction(self, mae: float, rmse: float, latency: float) -> None:
        self.predictions += 1
        self.mae_sum += mae
        self.rmse_sum += rmse
        self.latency_sum += latency

    def record_outcome(self, success: bool) -> None:
        self.total_trials += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def sample_score(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "name": self.name,
            "successes": self.successes,
            "failures": self.failures,
            "total_trials": self.total_trials,
            "predictions": self.predictions,
            "mae": self.mean_mae,
            "rmse": self.mean_rmse,
            "latency": self.mean_latency,
            "created_at": self.created_at,
        }


@dataclass
class ABTest:
    test_name: str
    model_name: str
    control_arm: Arm
    variant_arm: Arm
    confidence_threshold: float = 0.95
    min_samples: int = 1000
    test_id: str = field(init=False)
    status: TestStatus = field(default=TestStatus.SETUP)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    current_allocation: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.test_id = f"{self.test_name}_{int(datetime.now().timestamp() * 1000)}"
        self.current_allocation = {
            self.control_arm.model_id: 0.5,
            self.variant_arm.model_id: 0.5,
        }

    def start(self) -> None:
        self.status = TestStatus.RUNNING
        self.started_at = datetime.now().isoformat()
        logger.info("Started A/B test: %s", self.test_name)

    def select_arm(self) -> Arm:
        """Select arm using Thompson sampling"""
        control_prob = self.control_arm.sample_from_distribution()
        variant_prob = self.variant_arm.sample_from_distribution()
        
        # Thompson sampling — fair random tie-break when probabilities are equal
        if control_prob > variant_prob:
            return self.control_arm
        elif variant_prob > control_prob:
            return self.variant_arm
        else:
            return self.control_arm if random.random() < 0.5 else self.variant_arm
    
    def record_arm_outcome(self, arm_id: str, success: bool, metrics: Dict):
        """Record outcome for an arm"""
        arm = self.control_arm if arm_id == self.control_arm.model_id else self.variant_arm
        
        arm.record_outcome(success)
        if {"mae", "rmse", "latency"}.issubset(metrics):
            arm.record_prediction(metrics["mae"], metrics["rmse"], metrics["latency"])
        self._update_allocation()
    
    def _update_allocation(self):
        """Update traffic allocation based on Thompson sampling.

        Averages multiple samples from each arm's Beta distribution to
        reduce variance and prevent excessive allocation fluctuations.
        """
        if self.status != TestStatus.RUNNING:
            return

        n_samples = 100
        control_total = 0.0
        variant_total = 0.0
        for _ in range(n_samples):
            control_total += self.control_arm.sample_from_distribution()
            variant_total += self.variant_arm.sample_from_distribution()

        control_score = control_total / n_samples
        variant_score = variant_total / n_samples

        total_score = control_score + variant_score
        if total_score > 0:
            self.current_allocation[self.control_arm.model_id] = control_score / total_score
            self.current_allocation[self.variant_arm.model_id] = variant_score / total_score
    
    def _probability_variant_better(self) -> float:
        """Compute P(variant > control) via Monte Carlo from Beta posteriors."""
        n = 10000
        control_samples = np.random.beta(
            self.control_arm.successes + 1,
            self.control_arm.failures + 1,
            size=n,
        )
        variant_samples = np.random.beta(
            self.variant_arm.successes + 1,
            self.variant_arm.failures + 1,
            size=n,
        )
        return float(np.mean(variant_samples > control_samples))

    def get_winner(self) -> Optional[Arm]:
        """Determine winner using a two-sample Z-test for proportions."""
        n1 = self.control_arm.total_trials
        n2 = self.variant_arm.total_trials

        if (n1 + n2) < self.min_samples or n1 == 0 or n2 == 0:
            return None

        p1 = self.control_arm.successes / n1
        p2 = self.variant_arm.successes / n2

        # Pooled proportion under the null hypothesis (p1 == p2)
        p_pooled = (self.control_arm.successes + self.variant_arm.successes) / (n1 + n2)
        if p_pooled <= 0 or p_pooled >= 1:
            return None

        # Standard error of the difference
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
        if se == 0:
            return None

        # Z statistic
        z_stat = (p2 - p1) / se

        # Map confidence threshold to two-tailed critical z-values
        if self.confidence_threshold >= 0.99:
            z_crit = 2.576
        elif self.confidence_threshold >= 0.95:
            z_crit = 1.960
        else:
            z_crit = 1.645  # 90% confidence

        if abs(z_stat) > z_crit:
            return self.variant_arm if p2 > p1 else self.control_arm

        return None

    def end(self) -> None:
        self.status = TestStatus.COMPLETED
        self.ended_at = datetime.now().isoformat()
        logger.info("Ended A/B test: %s", self.test_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "model_name": self.model_name,
            "status": self.status.value,
            "control_arm": self.control_arm.to_dict(),
            "variant_arm": self.variant_arm.to_dict(),
            "current_allocation": self.current_allocation,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


class ABTestManager:
    def __init__(self) -> None:
        self.active_tests: Dict[str, ABTest] = {}
        self.completed_tests: List[ABTest] = []

    def create_test(
        self,
        test_name: str,
        model_name: str,
        control_arm: Arm,
        variant_arm: Arm,
        confidence_threshold: float = 0.95,
        min_samples: int = 1000,
    ) -> ABTest:
        test = ABTest(
            test_name=test_name,
            model_name=model_name,
            control_arm=control_arm,
            variant_arm=variant_arm,
            confidence_threshold=confidence_threshold,
            min_samples=min_samples,
        )
        self.active_tests[test.test_id] = test
        logger.info("Created A/B test: %s (ID: %s)", test_name, test.test_id)
        return test

    def get_test(self, test_id: str) -> Optional[ABTest]:
        return self.active_tests.get(test_id)

    def start_test(self, test_id: str) -> bool:
        test = self.get_test(test_id)
        if not test:
            return False
        test.start()
        return True

    def select_arm(self, test_id: str) -> Optional[Arm]:
        test = self.get_test(test_id)
        if not test or test.status != TestStatus.RUNNING:
            return None
        return test.select_arm()

    def record_outcome(self, test_id: str, arm_id: str, success: bool, metrics: Dict[str, float]) -> bool:
        test = self.get_test(test_id)
        if not test:
            return False
        test.record_arm_outcome(arm_id, success, metrics)
        winner = test.get_winner()
        if winner:
            test.end()
            self.active_tests.pop(test_id, None)
            self.completed_tests.append(test)
            logger.info("A/B test %s completed. Winner: %s", test_id, winner.name)
        return True

    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        test = self.get_test(test_id) or next(
            (t for t in self.completed_tests if t.test_id == test_id),
            None,
        )
        return test.to_dict() if test else None

    def list_active_tests(self) -> List[Dict[str, Any]]:
        return [test.to_dict() for test in self.active_tests.values()]

    def list_completed_tests(self) -> List[Dict[str, Any]]:
        return [test.to_dict() for test in self.completed_tests]


_ab_test_manager: Optional[ABTestManager] = None


def get_ab_test_manager() -> ABTestManager:
    global _ab_test_manager
    if _ab_test_manager is None:
        _ab_test_manager = ABTestManager()
    return _ab_test_manager
    
    def select_arm(self, test_id: str) -> Optional[Arm]:
        """Select arm for a request"""
        test = self.get_test(test_id)
        if not test or test.status != TestStatus.RUNNING:
            return None
        
        return test.select_arm()
    
    def record_outcome(self, test_id: str, arm_id: str, success: bool, metrics: Dict) -> bool:
        """Record outcome for arm"""
        test = self.get_test(test_id)
        if not test:
            return False
        
        test.record_arm_outcome(arm_id, success, metrics)
        
        # Check if test is complete
        winner = test.get_winner()
        if winner:
            test.end()
            self.active_tests.pop(test_id, None)
            self.completed_tests.append(test)
            logger.info(f"A/B test {test_id} completed. Winner: {winner.name}")
        
        return True
    
    def get_test_results(self, test_id: str) -> Optional[Dict]:
        """Get test results"""
        test = self.get_test(test_id) or next(
            (t for t in self.completed_tests if t.test_id == test_id), None
        )
        
        if not test:
            return None
        
        return test.to_dict()
    
    def list_active_tests(self) -> List[Dict]:
        """List all active tests"""
        return [test.to_dict() for test in self.active_tests.values()]
    
    def list_completed_tests(self) -> List[Dict]:
        """List completed tests"""
        return [test.to_dict() for test in self.completed_tests]


# Global A/B test manager instance
_ab_test_manager: Optional[ABTestManager] = None


def get_ab_test_manager() -> ABTestManager:
    """Get or create global A/B test manager"""
    global _ab_test_manager
    
    if _ab_test_manager is None:
        _ab_test_manager = ABTestManager()
    
    return _ab_test_manager
