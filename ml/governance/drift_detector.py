"""
Drift Detection Module
Monitors model prediction drift and data distribution changes.
"""
import logging
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Represents a detected drift alert"""
    timestamp: str
    model_name: str
    drift_type: str  # 'prediction' or 'input'
    severity: str    # 'low', 'medium', 'high'
    metric_value: float
    threshold: float
    details: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging"""
        return asdict(self)


class DriftDetector:
    """
    Detects data drift and prediction drift for models.
    
    Drift Types:
    - Prediction Drift: Model predictions deviating from expected range
    - Input Drift: Input feature distributions changing
    """
    
    def __init__(
        self,
        window_size: int = 100,
        prediction_drift_threshold: float = 0.2,
        input_drift_threshold: float = 0.15,
    ):
        """
        Initialize DriftDetector
        
        Args:
            window_size: Number of recent predictions to track
            prediction_drift_threshold: Alert if drift > this value
            input_drift_threshold: Alert if input drift > this value
        """
        self.window_size = window_size
        self.prediction_drift_threshold = prediction_drift_threshold
        self.input_drift_threshold = input_drift_threshold
        
        # Tracking storage
        self.prediction_history: Dict[str, deque] = {}  # model_name -> deque of predictions
        self.input_history: Dict[str, deque] = {}       # model_name -> deque of input stats
        self.baseline_stats: Dict[str, Dict[str, Any]] = {}  # model_name -> baseline
        self.alerts: List[DriftAlert] = []

        # Registered callbacks fired when drift is detected.
        # Each callable receives a single dict (DriftAlert.to_dict()) so that
        # callers can trigger rollback, send notifications, or log externally
        # without coupling this module to any infrastructure code.
        self._drift_callbacks: List = []

    def on_drift_detected(self, callback) -> None:
        """Register a callback to be invoked whenever drift breaches a threshold.

        The callback is called with a single argument: a dict representation of
        the DriftAlert that triggered it (see DriftAlert.to_dict()).

        Example::

            def my_handler(alert: dict):
                logger.critical("Drift detected: %s", alert)

            detector.on_drift_detected(my_handler)
        """
        self._drift_callbacks.append(callback)

    def _fire_drift_callbacks(self, alert: DriftAlert) -> None:
        """Invoke all registered drift callbacks with the alert dict.

        Exceptions raised by individual callbacks are logged and suppressed so
        that a broken callback never silences the detection result.
        """
        alert_dict = alert.to_dict()
        for cb in self._drift_callbacks:
            try:
                cb(alert_dict)
            except Exception as exc:  # pragma: no cover
                logger.error("Drift callback %r raised an error: %s", cb, exc, exc_info=True)

    def set_baseline(self, model_name: str, baseline_predictions: List[float]):
        """
        Set baseline statistics for a model (from training set)
        
        Args:
            model_name: Name of the model
            baseline_predictions: List of predictions from training set
        """
        baseline_array = np.array(baseline_predictions)
        
        self.baseline_stats[model_name] = {
            'mean': float(np.mean(baseline_array)),
            'std': float(np.std(baseline_array)),
            'min': float(np.min(baseline_array)),
            'max': float(np.max(baseline_array)),
            'median': float(np.median(baseline_array)),
        }
        
        # Initialize history tracking
        self.prediction_history[model_name] = deque(maxlen=self.window_size)
        self.input_history[model_name] = deque(maxlen=self.window_size)
        
        logger.info(f"Baseline set for model '{model_name}': mean={self.baseline_stats[model_name]['mean']:.2f}")
    
    def check_prediction_drift(
        self,
        model_name: str,
        prediction: float,
        actual_value: float = None,
    ) -> Tuple[bool, DriftAlert]:
        """
        Check if a prediction indicates drift
        
        Args:
            model_name: Name of the model
            prediction: Current prediction
            actual_value: Actual observed value (optional)
        
        Returns:
            Tuple of (drift_detected, alert_or_none)
        """
        if model_name not in self.baseline_stats:
            logger.warning(f"No baseline for model '{model_name}', skipping drift check")
            return False, None
        
        baseline = self.baseline_stats[model_name]
        self.prediction_history[model_name].append(prediction)
        
        # Calculate z-score deviation from baseline mean
        z_score = abs((prediction - baseline['mean']) / (baseline['std'] + 1e-10))
        
        # Get recent mean and std from window
        recent_preds = list(self.prediction_history[model_name])
        if len(recent_preds) < 10:  # Need minimum samples
            return False, None
        
        recent_mean = np.mean(recent_preds)
        recent_std = np.std(recent_preds)
        
        # Calculate drift as percentage change from baseline
        drift_magnitude = abs(recent_mean - baseline['mean']) / (baseline['mean'] + 1e-10)
        
        if drift_magnitude > self.prediction_drift_threshold:
            alert = DriftAlert(
                timestamp=datetime.now(timezone.utc).isoformat(),
                model_name=model_name,
                drift_type='prediction',
                severity=self._calculate_severity(drift_magnitude),
                metric_value=drift_magnitude,
                threshold=self.prediction_drift_threshold,
                details=f"Recent mean {recent_mean:.2f} vs baseline {baseline['mean']:.2f}"
            )
            self.alerts.append(alert)
            self._fire_drift_callbacks(alert)
            logger.warning(f"Prediction drift detected for {model_name}: {drift_magnitude:.2%}")
            return True, alert
        
        return False, None
    
    def check_input_drift(
        self,
        model_name: str,
        input_features: Dict[str, Any],
    ) -> Tuple[bool, DriftAlert]:
        """
        Check if input features indicate drift
        
        Args:
            model_name: Name of the model
            input_features: Dictionary of input features
        
        Returns:
            Tuple of (drift_detected, alert_or_none)
        """
        if model_name not in self.baseline_stats:
            logger.warning(f"No baseline for model '{model_name}', skipping input drift check")
            return False, None
        
        # Calculate feature statistics
        numeric_values = []
        for key, value in input_features.items():
            if isinstance(value, (int, float)):
                numeric_values.append(value)
        
        if not numeric_values:
            return False, None
        
        input_mean = np.mean(numeric_values)
        input_std = np.std(numeric_values)
        
        self.input_history[model_name].append({'mean': input_mean, 'std': input_std})
        
        # Need minimum samples for comparison
        if len(self.input_history[model_name]) < 5:
            return False, None
        
        # Compare recent input distribution to baseline predictions distribution
        baseline = self.baseline_stats[model_name]
        recent_inputs = list(self.input_history[model_name])
        recent_input_mean = np.mean([x['mean'] for x in recent_inputs])
        
        # Kolmogorov-Smirnov-like check using mean shift
        drift_magnitude = abs(recent_input_mean - baseline['mean']) / (baseline['mean'] + 1e-10)
        
        if drift_magnitude > self.input_drift_threshold:
            alert = DriftAlert(
                timestamp=datetime.now(timezone.utc).isoformat(),
                model_name=model_name,
                drift_type='input',
                severity=self._calculate_severity(drift_magnitude),
                metric_value=drift_magnitude,
                threshold=self.input_drift_threshold,
                details=f"Input mean {recent_input_mean:.2f} vs baseline {baseline['mean']:.2f}"
            )
            self.alerts.append(alert)
            self._fire_drift_callbacks(alert)
            logger.warning(f"Input drift detected for {model_name}: {drift_magnitude:.2%}")
            return True, alert
        
        return False, None
    
    @staticmethod
    def _calculate_severity(magnitude: float) -> str:
        """Calculate severity level based on magnitude"""
        if magnitude > 0.5:
            return 'high'
        elif magnitude > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def get_alerts(self, model_name: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent drift alerts"""
        alerts = self.alerts if model_name is None else [a for a in self.alerts if a.model_name == model_name]
        return [a.to_dict() for a in alerts[-limit:]]
    
    def clear_alerts(self):
        """Clear all stored alerts"""
        self.alerts.clear()
        logger.info("All drift alerts cleared")
