"""
Celery Predictive Auto-scaler
=============================
Polls Redis queue depth every 30s and integrates seasonal traffic
forecasts from PriceForecaster to scale worker processes up/down.
"""

import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional

import redis
from celery import Celery

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

MIN_WORKERS = int(os.getenv("CELERY_MIN_WORKERS", "2"))
MAX_WORKERS = int(os.getenv("CELERY_MAX_WORKERS", "20"))
QUEUE_DEPTH_THRESHOLD = int(os.getenv("CELERY_QUEUE_DEPTH_THRESHOLD", "100"))
POLL_INTERVAL = int(os.getenv("CELERY_AUTOSCALE_POLL_INTERVAL", "30"))
IDLE_TIMEOUT = int(os.getenv("CELERY_IDLE_TIMEOUT", "300"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Queues to monitor
_MONITORED_QUEUES = [
    "celery",
    "predict_yield_task",
    "predict_yield_lag_task",
    "predict_yield_trend_task",
]


# =============================================================================
# AUTOSCALER
# =============================================================================

class CeleryAutoscaler:
    """
    Predictive auto-scaler for Celery workers.
    """

    def __init__(self, celery_app: Celery, price_forecaster=None):
        self.celery_app = celery_app
        self.price_forecaster = price_forecaster
        self.redis_client = redis.from_url(REDIS_URL)
        self.workers: List[subprocess.Popen] = []
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_scale_time = datetime.min
        self.last_reason = "init"
        self.current_queue_depth = 0
        self.predicted_demand = 1.0
        self.target_workers = MIN_WORKERS

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    def start(self):
        """Start the autoscaler daemon thread."""
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info("Autoscaler started (min=%d, max=%d, threshold=%d)",
                    MIN_WORKERS, MAX_WORKERS, QUEUE_DEPTH_THRESHOLD)

    def stop(self):
        """Stop the daemon and terminate all managed workers."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        with self.lock:
            for proc in self.workers:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                except Exception:
                    logger.exception("Error terminating worker PID %d", proc.pid)
            self.workers.clear()
        logger.info("Autoscaler stopped and all workers terminated")

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------

    def _loop(self):
        while self.running:
            try:
                self._tick()
            except Exception:
                logger.exception("Autoscaler tick failed")
            time.sleep(POLL_INTERVAL)

    def _tick(self):
        """Single scaling decision."""
        self._reap_dead_workers()

        queue_depth = self._get_queue_depth()
        self.current_queue_depth = queue_depth

        seasonal_multiplier = 1.0
        if self.price_forecaster:
            try:
                seasonal_multiplier = self.price_forecaster.get_seasonal_demand_signal()
            except Exception:
                logger.exception("Seasonal signal failed")
        self.predicted_demand = seasonal_multiplier

        target = self._calculate_target_workers(queue_depth, seasonal_multiplier)
        self.target_workers = target

        self._scale_to(target)

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------

    def _get_queue_depth(self) -> int:
        """Sum Redis list lengths for all monitored queues."""
        total = 0
        for queue in _MONITORED_QUEUES:
            try:
                total += self.redis_client.llen(queue)
            except Exception:
                pass
        return total

    def _calculate_target_workers(self, queue_depth: int, seasonal_multiplier: float) -> int:
        """
        Calculate desired worker count based on queue depth and seasonal forecast.
        """
        target = MIN_WORKERS

        if queue_depth > QUEUE_DEPTH_THRESHOLD:
            # Proportional scale: each extra worker handles ~threshold/2 tasks
            extra_needed = (queue_depth - QUEUE_DEPTH_THRESHOLD) / (QUEUE_DEPTH_THRESHOLD / 2)
            target = MIN_WORKERS + int(extra_needed)
            self.last_reason = "depth_threshold"
        elif seasonal_multiplier > 1.5:
            # Pre-warm before predicted harvest traffic
            target = int(MIN_WORKERS * seasonal_multiplier)
            self.last_reason = "seasonal_forecast"
        else:
            # Scale down if idle for too long
            idle_seconds = (datetime.now() - self.last_scale_time).total_seconds()
            if idle_seconds > IDLE_TIMEOUT and len(self.workers) > MIN_WORKERS:
                target = MIN_WORKERS
                self.last_reason = "idle_timeout"
            else:
                self.last_reason = "stable"

        return max(MIN_WORKERS, min(MAX_WORKERS, target))

    # -------------------------------------------------------------------------
    # WORKER MANAGEMENT
    # -------------------------------------------------------------------------

    def _reap_dead_workers(self):
        """Remove crashed workers from the managed list."""
        with self.lock:
            alive = [p for p in self.workers if p.poll() is None]
            dead_count = len(self.workers) - len(alive)
            self.workers = alive
        if dead_count:
            logger.info("Reaped %d dead workers", dead_count)

    def _scale_to(self, target: int):
        """Spawn or kill workers to reach target count."""
        with self.lock:
            current = len(self.workers)

        if current == target:
            return

        if target > current:
            for _ in range(target - current):
                self._spawn_worker()
        else:
            for _ in range(current - target):
                self._kill_worker()

        self.last_scale_time = datetime.now()
        logger.info("Scaled workers %d -> %d (reason: %s)", current, target, self.last_reason)

    def _spawn_worker(self):
        """Launch a new Celery worker subprocess."""
        cmd = [
            "python", "-m", "celery",
            "-A", "celery_worker",
            "worker",
            "--loglevel=info",
            "--concurrency=4",
            "--without-gossip",
            "--without-mingle",
            "--without-heartbeat",
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self.lock:
                self.workers.append(proc)
            logger.info("Spawned worker PID %d", proc.pid)
        except Exception:
            logger.exception("Failed to spawn worker")

    def _kill_worker(self):
        """Gracefully terminate one worker, force-kill if necessary."""
        with self.lock:
            if not self.workers:
                return
            proc = self.workers.pop()

        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            logger.info("Killed worker PID %d", proc.pid)
        except Exception:
            logger.exception("Failed to kill worker PID %d", proc.pid)

    # -------------------------------------------------------------------------
    # STATUS
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current autoscale metrics for /health/autoscale."""
        with self.lock:
            active_workers = [p.pid for p in self.workers if p.poll() is None]
        return {
            "current_workers": len(active_workers),
            "target_workers": self.target_workers,
            "queue_depth": self.current_queue_depth,
            "predicted_demand": round(self.predicted_demand, 2),
            "last_scale_reason": self.last_reason,
            "last_scale_time": self.last_scale_time.isoformat() if self.last_scale_time != datetime.min else None,
            "min_workers": MIN_WORKERS,
            "max_workers": MAX_WORKERS,
            "queue_depth_threshold": QUEUE_DEPTH_THRESHOLD,
            "poll_interval": POLL_INTERVAL,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_autoscaler: Optional[CeleryAutoscaler] = None


def get_autoscaler(celery_app: Celery = None, price_forecaster=None) -> CeleryAutoscaler:
    global _autoscaler
    if _autoscaler is None:
        _autoscaler = CeleryAutoscaler(celery_app, price_forecaster)
    return _autoscaler