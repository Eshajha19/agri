"""
Shadow Evaluation Module
Runs new models alongside production models to evaluate performance before promotion.
"""
import logging
from collections import deque
import threading
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShadowEvaluation:
    """Results from shadow evaluation comparing two models"""
    timestamp: str
    production_model: str
    candidate_model: str
    samples_evaluated: int
    
    # Metrics
    production_mean_error: float
    candidate_mean_error: float
    error_reduction: float  # percentage
    
    # Performance comparison
    candidate_better: bool
    recommendation: str  # 'promote', 'keep_monitoring', 'reject'
    
    # Confidence metrics
    confidence_score: float  # 0-1
    min_sample_requirement_met: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ShadowEvaluator:
    """
    Evaluates candidate models against production models.
    Runs models in parallel on same data without affecting production.
    """
    
    def __init__(
        self,
        min_samples: int = 50,
        error_improvement_threshold: float = 0.05,  # 5% improvement required
        confidence_threshold: float = 0.85,
    ):
        """
        Initialize ShadowEvaluator
        
        Args:
            min_samples: Minimum predictions to collect before evaluation
            error_improvement_threshold: Candidate must improve error by this %
            confidence_threshold: Minimum confidence (0-1) for recommendation
        """
        self.min_samples = min_samples
        self.error_improvement_threshold = error_improvement_threshold
        self.confidence_threshold = confidence_threshold
        
        # Tracking
        # Bounded deque: oldest evaluations are discarded once the limit is
        # reached, preventing unbounded heap growth in long-running deployments.
        self.evaluations: deque = deque(maxlen=500)
        self.active_evaluations: Dict[str, Dict[str, Any]] = {}  # eval_id -> data
        self._lock = threading.Lock()
        self._drift_callbacks: List = []

    def on_drift_detected(self, callback) -> None:
        """Register a callback to be invoked when candidate performance is degraded (drifted)."""
        self._drift_callbacks.append(callback)

    def _fire_drift_callbacks(self, alert: Dict[str, Any]) -> None:
        """Invoke all registered callbacks with the alert dict."""
        for cb in self._drift_callbacks:
            try:
                cb(alert)
            except Exception as exc:
                logger.error("Shadow callback %r raised an error: %s", cb, exc, exc_info=True)
    
    def start_shadow_evaluation(
        self,
        production_model_name: str,
        candidate_model_name: str,
        eval_id: str = None,
    ) -> str:
        """
        Start a shadow evaluation session
        
        Args:
            production_model_name: Name of current production model
            candidate_model_name: Name of candidate model to test
            eval_id: Optional evaluation ID (auto-generated if not provided)
        
        Returns:
            Evaluation session ID
        """
        if eval_id is None:
            eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        async with self._lock:
            self.active_evaluations[eval_id] = {
                'production_model': production_model_name,
                'candidate_model': candidate_model_name,
                'production_predictions': deque(maxlen=10000),
                'candidate_predictions': deque(maxlen=10000),
                'actual_values': deque(maxlen=10000),
                'started_at': datetime.now(),
            }
        
        logger.info(f"Started shadow evaluation '{eval_id}': {production_model_name} vs {candidate_model_name}")
        return eval_id
    
    def record_predictions(
        self,
        eval_id: str,
        production_prediction: float,
        candidate_prediction: float,
        actual_value: float,
    ) -> None:
        """
        Record predictions from both models for comparison
        
        Args:
            eval_id: Evaluation session ID
            production_prediction: Prediction from production model
            candidate_prediction: Prediction from candidate model
            actual_value: Actual observed value
        """
        if eval_id not in self.active_evaluations:
            raise ValueError(f"Unknown evaluation ID: {eval_id}")
        
            eval_session = self.active_evaluations[eval_id]
            eval_session['production_predictions'].append(production_prediction)
            eval_session['candidate_predictions'].append(candidate_prediction)
            eval_session['actual_values'].append(actual_value)
    
    def evaluate_candidate(self, eval_id: str) -> Optional[ShadowEvaluation]:
        """
        Evaluate the candidate model against production model
        
        Args:
            eval_id: Evaluation session ID
        
        Returns:
            ShadowEvaluation results or None if not enough samples
        """
        if eval_id not in self.active_evaluations:
            logger.warning(f"Unknown evaluation ID: {eval_id}")
            return None
        
        eval_session = self.active_evaluations[eval_id]
        production_preds = np.array(eval_session['production_predictions'])
        candidate_preds = np.array(eval_session['candidate_predictions'])
        actual_vals = np.array(eval_session['actual_values'])
        
        n_samples = len(actual_vals)
        
        # Check minimum samples
        if n_samples < self.min_samples:
            logger.info(f"Evaluation {eval_id}: Only {n_samples}/{self.min_samples} samples collected")
            return None
        
        # Calculate errors
        production_errors = np.abs(production_preds - actual_vals)
        candidate_errors = np.abs(candidate_preds - actual_vals)
        
        prod_mean_error = float(np.mean(production_errors))
        cand_mean_error = float(np.mean(candidate_errors))
        
        # Calculate improvement
        error_reduction = (prod_mean_error - cand_mean_error) / (prod_mean_error + 1e-10)
        
        # Calculate confidence based on consistency
        variance_improvement = float(
            np.std(production_errors) - np.std(candidate_errors)
        ) / (np.std(production_errors) + 1e-10)
        
        # Confidence score: higher if consistent improvement across metrics
        confidence_score = float(
            min(1.0, (abs(error_reduction) + abs(variance_improvement)) / 2.0)
        )
        
        # Determine recommendation (confidence threshold enforced)
        candidate_better = error_reduction > self.error_improvement_threshold
        
        if candidate_better and confidence_score >= self.confidence_threshold:
            recommendation = 'promote'
        elif error_reduction > 0:
            recommendation = 'keep_monitoring'
        else:
            recommendation = 'reject'
        
        result = ShadowEvaluation(
            timestamp=datetime.now().isoformat(),
            production_model=eval_session['production_model'],
            candidate_model=eval_session['candidate_model'],
            samples_evaluated=n_samples,
            production_mean_error=prod_mean_error,
            candidate_mean_error=cand_mean_error,
            error_reduction=error_reduction,
            candidate_better=candidate_better,
            recommendation=recommendation,
            confidence_score=confidence_score,
            min_sample_requirement_met=n_samples >= self.min_samples,
        )
        
            self.evaluations.append(result)
            if recommendation == 'reject':
                alert = {
                    "timestamp": result.timestamp,
                    "model_name": result.candidate_model,
                    "drift_type": "shadow_performance_degradation",
                    "severity": "high",
                    "metric_value": result.candidate_mean_error,
                    "threshold": result.production_mean_error,
                    "details": f"Candidate error {result.candidate_mean_error:.4f} is worse than production error {result.production_mean_error:.4f}"
                }
                self._fire_drift_callbacks(alert)
        # cleanup_evaluation acquires self._lock itself; call it outside the
        # with block to avoid deadlock with a non-reentrant threading.Lock.
        self.cleanup_evaluation(eval_id)
        logger.info(
            f"Evaluation {eval_id} complete: {recommendation.upper()} "
            f"(error reduction: {error_reduction:.2%}, confidence: {confidence_score:.2%})"
        )
        
        return result
    
    def get_evaluation_status(self, eval_id: str) -> Dict[str, Any]:
        """Get current status of an evaluation"""
        async with self._lock:
            if eval_id not in self.active_evaluations:
                return {'status': 'not_found'}
        
            session = self.active_evaluations[eval_id]
            n_samples = len(session['actual_values'])
        
        return {
            'status': 'in_progress',
            'production_model': session['production_model'],
            'candidate_model': session['candidate_model'],
            'samples_collected': n_samples,
            'samples_needed': self.min_samples,
            'ready_for_evaluation': n_samples >= self.min_samples,
            'progress': min(1.0, n_samples / self.min_samples),
        }
    
    def get_evaluations(
        self,
        candidate_model: str = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get past evaluations"""
        async with self._lock:
            evals = self.evaluations if candidate_model is None else [
                e for e in self.evaluations if e.candidate_model == candidate_model
            ]
            return [e.to_dict() for e in evals[-limit:]]
    
    def cleanup_evaluation(self, eval_id: str) -> None:
        """Clean up completed evaluation from memory"""
        async with self._lock:
            if eval_id in self.active_evaluations:
                del self.active_evaluations[eval_id]
                logger.info(f"Cleaned up evaluation: {eval_id}")
