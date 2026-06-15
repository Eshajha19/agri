"""
ML Model Metrics Collection & Analysis
Tracks MAE, RMSE, latency, and generates performance reports
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and analyzes model performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector
        
        Args:
            window_size: Number of predictions to keep in memory
        """
        self.window_size = window_size
        self.predictions = defaultdict(list)  # model_id -> list of predictions
        self.performance_history = defaultdict(list)  # model_id -> list of hourly snapshots
    
    def record_prediction(
        self,
        model_id: str,
        model_version: str,
        y_true: float,
        y_pred: float,
        latency: float,
        timestamp: str = None
    ):
        """Record a single prediction"""
        
        mae = abs(y_true - y_pred)
        rmse = (y_true - y_pred) ** 2
        
        prediction = {
            "model_id": model_id,
            "model_version": model_version,
            "y_true": y_true,
            "y_pred": y_pred,
            "mae": mae,
            "rmse": rmse,
            "latency": latency,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        
        self.predictions[model_id].append(prediction)
        
        # Keep only recent predictions (sliding window)
        if len(self.predictions[model_id]) > self.window_size:
            self.predictions[model_id] = self.predictions[model_id][-self.window_size:]
    
    def get_metrics(self, model_id: str) -> Dict:
        """Get current metrics for a model"""
        
        if model_id not in self.predictions or not self.predictions[model_id]:
            return {
                "model_id": model_id,
                "predictions": 0,
                "mae": None,
                "rmse": None,
                "latency": None,
                "mae_percentile_95": None,
                "latency_percentile_95": None
            }
        
        predictions = self.predictions[model_id]
        
        maes = [p["mae"] for p in predictions]
        rmses = [p["rmse"] for p in predictions]
        latencies = [p["latency"] for p in predictions]
        
        def percentile(data, p):
            sorted_data = sorted(data)
            index = int((p / 100) * len(sorted_data))
            return sorted_data[min(index, len(sorted_data) - 1)]
        
        return {
            "model_id": model_id,
            "predictions": len(predictions),
            "mae": statistics.mean(maes),
            "mae_std": statistics.stdev(maes) if len(maes) > 1 else 0,
            "rmse": math.sqrt(statistics.mean(rmses)),
            "latency": statistics.mean(latencies),
            "latency_std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "mae_percentile_95": percentile(maes, 95),
            "latency_percentile_95": percentile(latencies, 95),
            "latency_min": min(latencies),
            "latency_max": max(latencies)
        }
    
    def compare_models(
        self,
        model_id_1: str,
        model_id_2: str
    ) -> Dict:
        """Compare two models side by side"""
        
        metrics_1 = self.get_metrics(model_id_1)
        metrics_2 = self.get_metrics(model_id_2)
        
        return {
            "model_1": metrics_1,
            "model_2": metrics_2,
            "mae_difference": (metrics_1["mae"] or 0) - (metrics_2["mae"] or 0),
            "latency_difference": (metrics_1["latency"] or 0) - (metrics_2["latency"] or 0),
            "model_1_better_mae": (metrics_1["mae"] or float('inf')) < (metrics_2["mae"] or float('inf')),
            "model_1_better_latency": (metrics_1["latency"] or float('inf')) < (metrics_2["latency"] or float('inf'))
        }
    
    def detect_performance_degradation(
        self,
        model_id: str,
        mae_threshold: float = None,
        latency_threshold: float = None
    ) -> Tuple[bool, str]:
        """
        Detect if model performance has degraded
        
        Returns:
            (has_degraded, reason)
        """
        
        if model_id not in self.performance_history:
            return False, "No baseline"
        
        current_metrics = self.get_metrics(model_id)
        history = self.performance_history[model_id]
        
        if not history:
            return False, "No baseline"
        
        baseline = history[0]
        
        # Check MAE degradation
        current_mae = current_metrics.get("mae")
        baseline_mae = baseline.get("mae")
        if current_mae is not None and baseline_mae is not None:
            mae_increase = ((current_mae - baseline_mae) / baseline_mae) * 100
            if mae_increase > 10:  # 10% increase
                return True, f"MAE increased by {mae_increase:.2f}%"
        
        # Check latency degradation
        current_latency = current_metrics.get("latency")
        baseline_latency = baseline.get("latency")
        if current_latency is not None and baseline_latency is not None:
            latency_increase = ((current_latency - baseline_latency) / baseline_latency) * 100
            if latency_increase > 15:  # 15% increase
                return True, f"Latency increased by {latency_increase:.2f}%"
        
        return False, "Performance stable"
    
    def record_baseline(self, model_id: str):
        """Record current metrics as baseline"""
        metrics = self.get_metrics(model_id)
        self.performance_history[model_id].insert(0, metrics)
        
        # Keep only last 30 baselines
        if len(self.performance_history[model_id]) > 30:
            self.performance_history[model_id] = self.performance_history[model_id][:30]
        
        mae = metrics.get("mae")
        mae_str = f"MAE={mae:.4f}" if mae is not None else "MAE=N/A"
        logger.info(f"Recorded baseline for {model_id}: {mae_str}")
    
    def get_performance_report(self, model_id: str) -> Dict:
        """Generate performance report"""
        
        return {
            "model_id": model_id,
            "current_metrics": self.get_metrics(model_id),
            "history": self.performance_history.get(model_id, []),
            "timestamp": datetime.now().isoformat()
        }
    
    def export_metrics(self) -> Dict:
        """Export all metrics"""
        return {
            "models": {
                model_id: self.get_metrics(model_id)
                for model_id in self.predictions.keys()
            },
            "timestamp": datetime.now().isoformat()
        }


class CanaryMonitor:
    """Monitors canary deployments and manages rollbacks"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.canary_deployments = {}  # model_id -> deployment_info
        self.rollback_triggers = defaultdict(list)
    
    def start_canary(
        self,
        model_id: str,
        baseline_model_id: str,
        traffic_percentage: int = 5,
        error_threshold: float = 0.1
    ):
        """Start canary monitoring for a model"""
        
        self.canary_deployments[model_id] = {
            "model_id": model_id,
            "baseline_model_id": baseline_model_id,
            "traffic_percentage": traffic_percentage,
            "error_threshold": error_threshold,
            "started_at": datetime.now().isoformat(),
            "status": "monitoring",
            "errors": 0
        }
        
        # Record baseline
        self.metrics.record_baseline(baseline_model_id)
        
        logger.info(f"Started canary for {model_id} with {traffic_percentage}% traffic")
    
    def check_canary_health(self, model_id: str) -> Tuple[bool, str]:
        """Check if canary is healthy"""
        
        if model_id not in self.canary_deployments:
            return True, "Not in canary"
        
        canary = self.canary_deployments[model_id]
        current_metrics = self.metrics.get_metrics(model_id)
        baseline_metrics = self.metrics.get_metrics(canary["baseline_model_id"])
        
        # Compare error rates
        current_mae = current_metrics.get("mae")
        baseline_mae = baseline_metrics.get("mae")
        if current_mae is not None and baseline_mae is not None:
            mae_increase = current_mae / baseline_mae

            if mae_increase > 1.2:  # 20% worse
                return False, f"MAE 20% worse than baseline (ratio: {mae_increase:.2f})"
        
        # Check prediction count
        if current_metrics["predictions"] < 10:
            return True, "Insufficient data"
        
        return True, "Canary healthy"
    
    def promote_canary(self, model_id: str) -> bool:
        """Promote canary to production"""
        
        if model_id not in self.canary_deployments:
            return False
        
        canary = self.canary_deployments[model_id]
        canary["status"] = "promoted"
        canary["promoted_at"] = datetime.now().isoformat()
        
        logger.info(f"Promoted canary {model_id} to production")
        return True
    
    def rollback_canary(self, model_id: str, reason: str) -> bool:
        """Rollback canary deployment"""
        
        if model_id not in self.canary_deployments:
            return False
        
        canary = self.canary_deployments[model_id]
        canary["status"] = "rolled_back"
        canary["rolled_back_at"] = datetime.now().isoformat()
        canary["rollback_reason"] = reason
        
        logger.warning(f"Rolled back canary {model_id}: {reason}")
        return True


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector"""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    
    return _metrics_collector


# Import math for sqrt
import math
