"""
Resource usage monitoring for production safety.

Tracks memory and CPU usage, logs warnings when thresholds are exceeded,
and exposes a /metrics endpoint for observability.
"""
import os
import logging
import threading
import time
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

# Defaults — overridable via env vars
MEMORY_WARN_MB = int(os.getenv("MEMORY_WARN_MB", "450"))
MEMORY_CRITICAL_MB = int(os.getenv("MEMORY_CRITICAL_MB", "500"))
CPU_WARN_PERCENT = float(os.getenv("CPU_WARN_PERCENT", "85"))
POLL_INTERVAL_SECONDS = int(os.getenv("RESOURCE_POLL_INTERVAL", "30"))


class ResourceMonitor:
    """Background thread that polls memory/CPU and logs warnings."""

    def __init__(
        self,
        memory_warn_mb: int = MEMORY_WARN_MB,
        memory_critical_mb: int = MEMORY_CRITICAL_MB,
        cpu_warn_percent: float = CPU_WARN_PERCENT,
        poll_interval: int = POLL_INTERVAL_SECONDS,
    ):
        self.memory_warn_mb = memory_warn_mb
        self.memory_critical_mb = memory_critical_mb
        self.cpu_warn_percent = cpu_warn_percent
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_snapshot: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="resource-monitor"
        )
        self._thread.start()
        logger.info(
            "Resource monitor started (warnmem=%dMB, critmem=%dMB, warncpu=%.0f%%, poll=%ds)",
            self.memory_warn_mb,
            self.memory_critical_mb,
            self.cpu_warn_percent,
            self.poll_interval,
        )

    def stop(self) -> None:
        """Signal the monitoring thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def snapshot(self) -> dict:
        """Return the most recent resource reading."""
        if not self._last_snapshot:
            self._last_snapshot = self._collect()
        return self._last_snapshot

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect(self) -> dict:
        proc = psutil.Process()
        mem = proc.memory_info()
        vm = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        return {
            "process_rss_mb": round(mem.rss / (1024 * 1024), 1),
            "process_vms_mb": round(mem.vms / (1024 * 1024), 1),
            "system_memory_percent": vm.percent,
            "system_memory_available_mb": round(vm.available / (1024 * 1024), 1),
            "cpu_percent": cpu_percent,
        }

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                snap = self._collect()
                self._last_snapshot = snap
                self._check_thresholds(snap)
            except Exception as exc:
                logger.warning("Resource monitor poll failed: %s", exc)
            self._stop_event.wait(self.poll_interval)

    def _check_thresholds(self, snap: dict) -> None:
        rss_mb = snap["process_rss_mb"]
        cpu = snap["cpu_percent"]

        if rss_mb >= self.memory_critical_mb:
            logger.critical(
                "CRITICAL: Process memory %.0fMB exceeds hard limit %dMB — "
                "instance may be OOM-killed by Render",
                rss_mb,
                self.memory_critical_mb,
            )
        elif rss_mb >= self.memory_warn_mb:
            logger.warning(
                "Memory usage %.0fMB exceeds warn threshold %dMB",
                rss_mb,
                self.memory_warn_mb,
            )

        if cpu >= self.cpu_warn_percent:
            logger.warning(
                "CPU usage %.1f%% exceeds warn threshold %.0f%%",
                cpu,
                self.cpu_warn_percent,
            )


# Module-level singleton
resource_monitor = ResourceMonitor()
