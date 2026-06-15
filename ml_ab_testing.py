"""
A/B Testing Engine with Multi-Armed Bandits
Implements Thompson sampling for optimal model allocation
"""

import logging
import random
import math
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status"""
    SETUP = "setup"
    RUNNING = "running"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Arm:
    """A/B test arm (model variant)"""
    
    def __init__(self, model_id: str, model_version: str, name: str = None):
        self.model_id = model_id
        self.model_version = model_version
        self.name = name or f"{model_version}"
        
        # Thompson sampling parameters (Beta distribution)
        self.successes = 0  # Alpha parameter
        self.failures = 0   # Beta parameter
        self.total_trials = 0
        
        # Metrics
        self.predictions = 0
        self.mae_sum = 0.0
        self.rmse_sum = 0.0
        self.latency_sum = 0.0
        self.latency_count = 0
        
        self.created_at = datetime.now().isoformat()
    
    def get_mean_metric(self, metric_name: str) -> float:
        """Get mean value for a metric"""
        if metric_name == "mae":
            return self.mae_sum / max(self.predictions, 1)
        elif metric_name == "rmse":
            return self.rmse_sum / max(self.predictions, 1)
        elif metric_name == "latency":
            return self.latency_sum / max(self.latency_count, 1)
        return 0.0
    
    def record_prediction(self, mae: float, rmse: float, latency: float):
        """Record prediction metrics"""
        self.predictions += 1
        self.mae_sum += mae
        self.rmse_sum += rmse
        self.latency_sum += latency
        self.latency_count += 1
    
    def record_outcome(self, success: bool):
        """Record trial outcome"""
        self.total_trials += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
    
    def record_trial(self, success: bool, mae: float = 0.0, rmse: float = 0.0, latency: float = 0.0):
        """Record outcome and prediction metrics atomically so counters never drift."""
        self.total_trials += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.predictions += 1
        self.mae_sum += mae
        self.rmse_sum += rmse
        self.latency_sum += latency
        self.latency_count += 1
    
    def sample_from_distribution(self) -> float:
        """Sample success probability from Beta distribution (Thompson sampling)"""
        # Use Beta(alpha, beta) where alpha = successes, beta = failures
        alpha = self.successes + 1  # Add pseudocount
        beta = self.failures + 1
        
        if alpha <= 0 or beta <= 0:
            return 0.5
        
        # Sample from Beta distribution
        return np.random.beta(alpha, beta)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "name": self.name,
            "successes": self.successes,
            "failures": self.failures,
            "total_trials": self.total_trials,
            "predictions": self.predictions,
            "mae": self.get_mean_metric("mae"),
            "rmse": self.get_mean_metric("rmse"),
            "latency": self.get_mean_metric("latency"),
            "created_at": self.created_at
        }


class ABTest:
    """A/B test for comparing model versions"""
    
    def __init__(
        self,
        test_name: str,
        model_name: str,
        control_arm: Arm,
        variant_arm: Arm,
        confidence_threshold: float = 0.95,
        min_samples: int = 1000
    ):
        self.test_id = f"{test_name}_{datetime.now().timestamp()}"
        self.test_name = test_name
        self.model_name = model_name
        self.control_arm = control_arm
        self.variant_arm = variant_arm
        self.confidence_threshold = confidence_threshold
        self.min_samples = min_samples
        
        self.status = TestStatus.SETUP
        self.created_at = datetime.now().isoformat()
        self.started_at = None
        self.ended_at = None
        
        # Traffic allocation
        self.current_allocation = {
            control_arm.model_id: 0.5,
            variant_arm.model_id: 0.5
        }
    
    def start(self):
        """Start the A/B test"""
        self.status = TestStatus.RUNNING
        self.started_at = datetime.now().isoformat()
        logger.info(f"Started A/B test: {self.test_name}")
    
    def select_arm(self) -> Arm:
        """Select arm using Thompson sampling"""
        control_prob = self.control_arm.sample_from_distribution()
        variant_prob = self.variant_arm.sample_from_distribution()
        
        # Use Thompson sampling to select
        if control_prob > variant_prob:
            return self.control_arm
        else:
            return self.variant_arm
    
    def record_arm_outcome(self, arm_id: str, success: bool, metrics: Dict):
        """Record outcome for an arm"""
        arm = self.control_arm if arm_id == self.control_arm.model_id else self.variant_arm
        
        arm.record_trial(
            success,
            mae=metrics.get("mae", 0.0),
            rmse=metrics.get("rmse", 0.0),
            latency=metrics.get("latency", 0.0),
        )
        
        # Update allocation based on Thompson sampling
        self._update_allocation()
    
    def _update_allocation(self):
        """Update traffic allocation based on Thompson sampling"""
        if self.status != TestStatus.RUNNING:
            return
        
        control_score = self.control_arm.sample_from_distribution()
        variant_score = self.variant_arm.sample_from_distribution()
        
        total_score = control_score + variant_score
        if total_score > 0:
            self.current_allocation[self.control_arm.model_id] = control_score / total_score
            self.current_allocation[self.variant_arm.model_id] = variant_score / total_score
    
    def get_winner(self) -> Optional[Arm]:
        """Determine winner with confidence threshold"""
        total_trials = self.control_arm.total_trials + self.variant_arm.total_trials
        
        if total_trials < self.min_samples:
            return None
        
        control_mean = self.control_arm.successes / max(self.control_arm.total_trials, 1)
        variant_mean = self.variant_arm.successes / max(self.variant_arm.total_trials, 1)
        
        # Simple confidence check
        if abs(control_mean - variant_mean) > (1 - self.confidence_threshold):
            return self.variant_arm if variant_mean > control_mean else self.control_arm
        
        return None
    
    def end(self):
        """End the A/B test"""
        self.status = TestStatus.COMPLETED
        self.ended_at = datetime.now().isoformat()
        logger.info(f"Ended A/B test: {self.test_name}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
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
            "ended_at": self.ended_at
        }


class ABTestManager:
    """Manages A/B tests for model comparison"""
    
    def __init__(self):
        self.active_tests: Dict[str, ABTest] = {}
        self.completed_tests: List[ABTest] = []
    
    def create_test(
        self,
        test_name: str,
        model_name: str,
        control_arm: Arm,
        variant_arm: Arm,
        confidence_threshold: float = 0.95,
        min_samples: int = 1000
    ) -> ABTest:
        """Create new A/B test"""
        
        test = ABTest(
            test_name=test_name,
            model_name=model_name,
            control_arm=control_arm,
            variant_arm=variant_arm,
            confidence_threshold=confidence_threshold,
            min_samples=min_samples
        )
        
        self.active_tests[test.test_id] = test
        logger.info(f"Created A/B test: {test_name} (ID: {test.test_id})")
        
        return test
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get A/B test by ID"""
        return self.active_tests.get(test_id)
    
    def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        test = self.get_test(test_id)
        if not test:
            return False
        
        test.start()
        return True
    
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
