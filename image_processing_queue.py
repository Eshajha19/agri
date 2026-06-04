"""
Image Processing Pipeline Queue Management & Horizontal Scaling System

Provides:
- Distributed task queue for image processing
- Worker pool management with horizontal scaling
- Task status tracking and monitoring
- Async processing with callbacks
- Optional Redis support for distributed deployments
"""
from collections import OrderedDict
import asyncio
import uuid
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
import heapq
import threading
import time
import os
import random

logger = logging.getLogger(__name__)


class LRUCache:
    """LRU cache with configurable capacity and TTL"""
    def __init__(self, capacity: int = 1000, ttl_seconds: int = 86400):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache: OrderedDict[str, tuple] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            value, timestamp = self.cache[key]
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                self.misses += 1
                return None

            self.hits += 1
            self.cache.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = (value, time.time())

            if len(self.cache) > self.capacity:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

    def invalidate(self, key: str) -> None:
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            return {
                "size": len(self.cache),
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }


@dataclass
class ProcessingMetrics:
    """Metrics for image processing"""
    total_enqueued: int = 0
    total_processed: int = 0
    total_failed: int = 0
    average_processing_time: float = 0.0
    queue_depth: int = 0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    processing_times: List[float] = field(default_factory=list)

    def add_processing_time(self, duration: float) -> None:
        self.processing_times.append(duration)
        if len(self.processing_times) > 1000:
            self.processing_times.pop(0)
        self.average_processing_time = sum(self.processing_times) / len(self.processing_times)

    def update_error_rate(self) -> None:
        total = self.total_processed + self.total_failed
        self.error_rate = (self.total_failed / total * 100) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_enqueued": self.total_enqueued,
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "average_processing_time_ms": round(self.average_processing_time * 1000, 2),
            "queue_depth": self.queue_depth,
            "error_rate": round(self.error_rate, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 2)
        }


class TaskStatus(str, Enum):
    """Task execution status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority levels"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class ImageProcessingTask:
    """Represents an image processing task"""
    task_id: str
    image_data: bytes  # Base64 decoded image
    crop_type: str
    processor_type: str  # 'quality_grading', 'disease_detection', etc.
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.QUEUED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    worker_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class WorkerStats:
    """Statistics for a worker"""
    worker_id: str
    tasks_processed: int = 0
    tasks_failed: int = 0
    avg_processing_time: float = 0.0
    status: str = "idle"
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_task_id: Optional[str] = None


class ImageProcessingQueue:
    """
    Thread-safe image processing task queue with priority ordering and
    horizontal scaling support.
    """

    def __init__(self, max_queue_size: int = 10000, enable_persistence: bool = False, enable_backoff: bool = False, backoff_base: float = 1.0, enable_caching: bool = False):
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence
        # Backoff controls (opt-in to preserve previous behavior in tests)
        self.enable_backoff = enable_backoff
        self.backoff_base = backoff_base
        
        # Task storage (heap of (priority, counter, task) tuples)
        self._task_queue: List[tuple] = []
        self._tasks_by_id: Dict[str, ImageProcessingTask] = {}
        self._counter = 0
        self._total_enqueued = 0
        self._total_processed = 0
        self._total_failed = 0
        self._completed_tasks: Dict[str, ImageProcessingTask] = {}  # History
        # Ack store for exactly-once semantics: maps task_id -> status
        self._ack_store: Dict[str, str] = {}
        self._ack_file = "queue_acks.json" if enable_persistence else None
        
        # Worker management
        self._workers: Dict[str, WorkerStats] = {}
        self._worker_lock = threading.Lock()

        # Cache management
        self._image_cache = LRUCache(capacity=1000, ttl_seconds=86400) if enable_caching else None
        self._processing_times: Dict[str, float] = {}

        # Thread safety
        self._queue_lock = threading.Lock()
        self._task_lock = threading.Lock()

        # Metrics
        self._metrics = ProcessingMetrics()
        self._start_time = time.time()

    def enqueue(self, task: ImageProcessingTask) -> str:
        """Enqueue a task for processing"""
        # Exactly-once: if task already acknowledged as completed/failed, skip enqueue
        if task.task_id in self._ack_store and self._ack_store[task.task_id] in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
            logger.info(f"Task {task.task_id} already acknowledged as {self._ack_store[task.task_id]} — skipping enqueue")
            return task.task_id
        with self._queue_lock:
            if len(self._task_queue) >= self.max_queue_size:
                raise RuntimeError(f"Queue is full (max: {self.max_queue_size})")
            # Push only a heap tuple — never append raw task objects.
            # The previous code did both self._task_queue.append(task) AND
            # heapq.heappush(...), which mixed raw ImageProcessingTask objects
            # with (priority, counter, task) tuples in the same list, breaking
            # heap ordering and causing TypeErrors on comparison.
            heapq.heappush(self._task_queue, (task.priority.value, self._counter, task))
            self._counter += 1
            self._tasks_by_id[task.task_id] = task
            self._total_enqueued += 1

        # Persist ack store if enabled
        if self.enable_persistence:
            try:
                with open(self._ack_file, "w", encoding="utf-8") as f:
                    json.dump(self._ack_store, f)
            except Exception:
                logger.debug("Failed to persist ack_store")

        logger.info(f"Task {task.task_id} enqueued (priority: {task.priority.name}, queue_size: {len(self._task_queue)})")
        return task.task_id

    def dequeue(self, worker_id: str) -> Optional[ImageProcessingTask]:
        """
        Dequeue highest priority task for worker.

        Race-condition fix: the transition from QUEUED/RETRYING → PROCESSING is
        performed while holding *both* _queue_lock (to pop from the heap) and
        _task_lock (to mutate task.status).  cancel_task() also holds _task_lock
        when it checks and mutates task.status, so the two operations are now
        mutually exclusive — a task cannot be simultaneously dequeued and
        cancelled.

        Lock ordering is always _task_lock → _queue_lock (same as cancel_task)
        to prevent deadlock.
        """
        with self._task_lock:
            with self._queue_lock:
                if not self._task_queue:
                    return None

                # Pop until we find an eligible task (available_at in past and not acked)
                popped = []
                selected = None
                now_iso = datetime.now().isoformat()
                while self._task_queue:
                    _, _, task = heapq.heappop(self._task_queue)

                    # Skip if task already processed (exactly-once)
                    if task.task_id in self._ack_store and self._ack_store[task.task_id] in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
                        continue

                    # Skip if task was cancelled between enqueue and dequeue
                    # (race-condition guard: cancel_task() also holds _task_lock
                    # when it sets status = CANCELLED, so this check is atomic
                    # with respect to that transition).
                    if task.status == TaskStatus.CANCELLED:
                        continue

                    available_at = task.metadata.get("available_at")
                    if available_at and available_at > now_iso:
                        # not ready yet — postpone
                        popped.append((task.priority.value, self._counter, task))
                        self._counter += 1
                        continue

                    # eligible
                    selected = task
                    break

                # reinsert postponed tasks
                for entry in popped:
                    heapq.heappush(self._task_queue, entry)

                if not selected:
                    return None

                task = selected
                # Atomically transition to PROCESSING while still holding
                # _task_lock so cancel_task() cannot sneak in between the
                # status check (QUEUED/RETRYING guard) and this assignment.
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.now().isoformat()
                task.worker_id = worker_id

            logger.info(f"Task {task.task_id} assigned to worker {worker_id}")
            return task

    def complete_task(self, task_id: str, result: Dict) -> bool:
        """Mark task as completed with result"""
        with self._task_lock:
            if task_id not in self._tasks_by_id:
                logger.warning(f"Task {task_id} not found for completion")
                return False
            
            task = self._tasks_by_id[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.result = result
            
            # Move to completed
            del self._tasks_by_id[task_id]
            self._completed_tasks[task_id] = task
            self._total_processed += 1
            # Exactly-once ack
            self._ack_store[task_id] = TaskStatus.COMPLETED.value
            if self.enable_persistence:
                try:
                    with open(self._ack_file, "w", encoding="utf-8") as f:
                        json.dump(self._ack_store, f)
                except Exception:
                    logger.debug("Failed to persist ack_store on complete")
            
            logger.info(f"Task {task_id} completed successfully")
            return True

    def fail_task(self, task_id: str, error: str, retry: bool = True) -> bool:
        """Mark task as failed with optional retry"""
        with self._task_lock:
            if task_id not in self._tasks_by_id:
                logger.warning(f"Task {task_id} not found for failure")
                return False
            
            task = self._tasks_by_id[task_id]
            task.retry_count += 1
            
            if retry and task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRYING
                if self.enable_backoff:
                    # Re-enqueue for retry with exponential backoff + jitter
                    delay = self.backoff_base * (2 ** (task.retry_count - 1))
                    # jitter +/- 20%
                    jitter = delay * 0.2
                    delay = delay + random.uniform(-jitter, jitter)
                    available_at = (datetime.now() + timedelta(seconds=max(0.1, delay))).isoformat()
                    task.metadata["available_at"] = available_at
                    with self._queue_lock:
                        heapq.heappush(self._task_queue, (task.priority.value, self._counter, task))
                        self._counter += 1
                    logger.info(f"Task {task_id} requeued for retry ({task.retry_count}/{task.max_retries}) available_at={available_at}")
                else:
                    # Immediate requeue (legacy behavior / tests expect immediate)
                    task.metadata.pop("available_at", None)
                    with self._queue_lock:
                        heapq.heappush(self._task_queue, (task.priority.value, self._counter, task))
                        self._counter += 1
                    logger.info(f"Task {task_id} requeued for retry ({task.retry_count}/{task.max_retries})")
                return True
            else:
                task.status = TaskStatus.FAILED
                task.error = error
                task.completed_at = datetime.now().isoformat()
                
                # Move to completed
                del self._tasks_by_id[task_id]
                self._completed_tasks[task_id] = task
                self._total_failed += 1
                # Ack failed for exactly-once
                self._ack_store[task_id] = TaskStatus.FAILED.value
                if self.enable_persistence:
                    try:
                        with open(self._ack_file, "w", encoding="utf-8") as f:
                            json.dump(self._ack_store, f)
                    except Exception:
                        logger.debug("Failed to persist ack_store on fail")

                logger.error(f"Task {task_id} failed after {task.retry_count} retries: {error}")
                return False

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a task"""
        with self._task_lock:
            # Check active tasks
            if task_id in self._tasks_by_id:
                task = self._tasks_by_id[task_id]
                return {
                    "task_id": task_id,
                    "status": task.status.value,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "progress": "processing" if task.status == TaskStatus.PROCESSING else "queued",
                }
            
            # Check completed tasks
            if task_id in self._completed_tasks:
                task = self._completed_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": task.status.value,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "result": task.result if task.status == TaskStatus.COMPLETED else None,
                    "error": task.error if task.status == TaskStatus.FAILED else None,
                }
            
            return None

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a queued or retrying task.
        
        Note: The internal queue is managed as a heap list, not a collections.deque.
        Cancellation is performed safely by filtering the list and re-heapifying
        to maintain priority queue invariants.
        """
        # Acquire both locks in a consistent order (_task_lock before
        # _queue_lock) to avoid deadlock with fail_task, which also nests
        # _queue_lock inside _task_lock.
        with self._task_lock:
            if task_id not in self._tasks_by_id:
                return False

            task = self._tasks_by_id[task_id]
            if task.status not in (TaskStatus.QUEUED, TaskStatus.RETRYING):
                return False

            task.status = TaskStatus.CANCELLED

            with self._queue_lock:
           
                self._task_queue = [
                    entry for entry in self._task_queue
                    if entry[2].task_id != task_id
                ]
                heapq.heapify(self._task_queue)

            del self._tasks_by_id[task_id]
            self._completed_tasks[task_id] = task
            logger.info(f"Task {task_id} cancelled")
            return True

    def register_worker(self, worker_id: str) -> WorkerStats:
        """Register a worker"""
        with self._worker_lock:
            if worker_id not in self._workers:
                self._workers[worker_id] = WorkerStats(worker_id=worker_id)
                logger.info(f"Worker {worker_id} registered")
            return self._workers[worker_id]

    def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker"""
        with self._worker_lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                logger.info(f"Worker {worker_id} unregistered")
                return True
            return False

    def update_worker_stats(self, worker_id: str, processing_time: float, success: bool):
        """Update worker statistics"""
        with self._worker_lock:
            if worker_id not in self._workers:
                return
            
            worker = self._workers[worker_id]
            worker.tasks_processed += 1
            if not success:
                worker.tasks_failed += 1
            
            # Update average processing time (exponential moving average)
            if worker.avg_processing_time == 0:
                worker.avg_processing_time = processing_time
            else:
                worker.avg_processing_time = (worker.avg_processing_time * 0.7) + (processing_time * 0.3)
            
            worker.last_heartbeat = datetime.now().isoformat()

    def get_queue_stats(self) -> Dict:
        """Get queue and worker statistics with advanced metrics"""
        with self._queue_lock:
            queue_size = len(self._task_queue)

        with self._task_lock:
            active_tasks = len(self._tasks_by_id)
            completed_tasks = len(self._completed_tasks)

        with self._worker_lock:
            workers_online = len(self._workers)
            worker_stats = list(self._workers.values())

        self._metrics.queue_depth = queue_size
        self._metrics.total_enqueued = self._total_enqueued
        self._metrics.total_processed = self._total_processed
        self._metrics.total_failed = self._total_failed
        self._metrics.update_error_rate()

        if self._image_cache:
            cache_stats = self._image_cache.get_stats()
            self._metrics.cache_hit_rate = cache_stats['hit_rate']
        else:
            self._metrics.cache_hit_rate = 0.0

        uptime_seconds = time.time() - self._start_time
        estimated_completion_time = None
        if self._metrics.average_processing_time > 0 and queue_size > 0:
            estimated_seconds = queue_size * self._metrics.average_processing_time
            estimated_completion_time = (datetime.now() + timedelta(seconds=estimated_seconds)).isoformat()

        return {
            "queue_size": queue_size,
            "active_tasks": active_tasks,
            "completed_tasks": completed_tasks,
            "total_enqueued": self._total_enqueued,
            "total_processed": self._total_processed,
            "total_failed": self._total_failed,
            "workers_online": workers_online,
            "workers": [asdict(w) for w in worker_stats],
            "avg_processing_time": (sum(w.avg_processing_time for w in worker_stats) / len(worker_stats)) if worker_stats else 0,
            "metrics": self._metrics.to_dict(),
            "estimated_completion_time": estimated_completion_time,
            "uptime_seconds": uptime_seconds
        }

    def get_batch(self, batch_size: Optional[int] = None) -> List[ImageProcessingTask]:
        """Get a batch of tasks for processing"""
        if batch_size is None:
            batch_size = self.batch_size

        tasks = []
        with self._queue_lock:
            for _ in range(min(batch_size, len(self._task_queue))):
                if not self._task_queue:
                    break
                _, _, task = heapq.heappop(self._task_queue)
                if task.task_id not in self._ack_store:
                    task.status = TaskStatus.PROCESSING
                    task.started_at = datetime.now().isoformat()
                    tasks.append(task)

        logger.info(f"Batch dequeue: {len(tasks)} tasks")
        return tasks

    def get_task_status_with_position(self, task_id: str) -> Optional[Dict]:
        """Get detailed task status including queue position and progress estimate"""
        with self._task_lock:
            if task_id in self._tasks_by_id:
                task = self._tasks_by_id[task_id]
                queue_position = self._get_queue_position(task_id)
                progress_percent = 0
                if task.status == TaskStatus.PROCESSING:
                    progress_percent = 50
                elif task.status == TaskStatus.QUEUED:
                    progress_percent = 10

                estimated_time = None
                if queue_position >= 0 and self._metrics.average_processing_time > 0:
                    estimated_seconds = queue_position * self._metrics.average_processing_time
                    estimated_time = (datetime.now() + timedelta(seconds=estimated_seconds)).isoformat()

                return {
                    "task_id": task_id,
                    "status": task.status.value,
                    "priority": task.priority.name,
                    "queue_position": queue_position,
                    "progress_percent": progress_percent,
                    "estimated_completion_time": estimated_time,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "processor_type": task.processor_type,
                }

            if task_id in self._completed_tasks:
                task = self._completed_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": task.status.value,
                    "progress_percent": 100 if task.status == TaskStatus.COMPLETED else 0,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "result": task.result if task.status == TaskStatus.COMPLETED else None,
                    "error": task.error if task.status == TaskStatus.FAILED else None,
                }

            return None

    def _get_queue_position(self, task_id: str) -> int:
        """Get the position of a task in the queue"""
        with self._queue_lock:
            for idx, entry in enumerate(self._task_queue):
                if entry[2].task_id == task_id:
                    return idx
        return -1

    def track_processing_time(self, task_id: str, start_time: float) -> None:
        """Track the processing time for a task"""
        duration = time.time() - start_time
        self._processing_times[task_id] = duration
        self._metrics.add_processing_time(duration)

    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get a cached result if available"""
        if self._image_cache:
            return self._image_cache.get(cache_key)
        return None

    def cache_result(self, cache_key: str, result: Any) -> None:
        """Cache a processing result"""
        if self._image_cache:
            self._image_cache.put(cache_key, result)
            logger.info(f"Cached result for key: {cache_key}")

    def invalidate_cache(self, cache_key: str) -> None:
        """Invalidate a cached result"""
        if self._image_cache:
            self._image_cache.invalidate(cache_key)

    def get_cache_stats(self) -> Optional[Dict]:
        """Get cache statistics"""
        if self._image_cache:
            return self._image_cache.get_stats()
        return None

    def get_processing_history(self, limit: int = 100) -> List[Dict]:
        """Get recent processing history"""
        history = []
        with self._task_lock:
            sorted_tasks = sorted(
                self._completed_tasks.values(),
                key=lambda t: t.completed_at or "",
                reverse=True
            )[:limit]

            for task in sorted_tasks:
                processing_time = self._processing_times.get(task.task_id, 0)
                history.append({
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "crop_type": task.crop_type,
                    "processor_type": task.processor_type,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at,
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "error": task.error if task.status == TaskStatus.FAILED else None,
                })
        return history

    def get_pending_tasks(self, limit: int = 100) -> List[Dict]:
        """Get pending tasks"""
        with self._queue_lock:
            tasks = [entry[2] for entry in self._task_queue[:limit]]
            return [
                {
                    "task_id": t.task_id,
                    "status": t.status.value,
                    "priority": t.priority.name,
                    "crop_type": t.crop_type,
                    "processor_type": t.processor_type,
                    "created_at": t.created_at,
                }
                for t in tasks
            ]

    def cleanup_old_completed_tasks(self, max_age_hours: int = 24):
        """Remove completed tasks older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cutoff_iso = cutoff_time.isoformat()
        
        with self._task_lock:
            to_remove = [
                task_id for task_id, task in self._completed_tasks.items()
                if task.completed_at and task.completed_at < cutoff_iso
            ]
            
            for task_id in to_remove:
                del self._completed_tasks[task_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old completed tasks")
            
            return len(to_remove)



# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

# Exceptions that represent permanent, deterministic failures.
# Retrying these wastes capacity — the outcome will never change.
# Examples: malformed image bytes (ValueError), wrong argument types
# (TypeError), unimplemented processor paths (NotImplementedError),
# images that exceed hard size limits (MemoryError).
NON_RETRYABLE_ERRORS = (
    ValueError,        # Bad input data / validation failures
    TypeError,         # Programming errors / wrong argument types
    NotImplementedError,  # Unimplemented processor paths
    MemoryError,       # Image too large to ever process
    KeyError,          # Missing required field in task metadata
    AttributeError,    # Structural errors in task or result objects
)


def _is_retryable(exc: BaseException) -> bool:
    """
    Return True if *exc* represents a transient failure that is worth
    retrying (e.g. a network hiccup or a temporarily unavailable resource),
    and False if it is a permanent failure that will never succeed on retry.
    """
    return not isinstance(exc, NON_RETRYABLE_ERRORS)


class ImageProcessingWorker:
    """
    Worker process that consumes tasks from queue and processes images.
    Can be run in separate thread or process for horizontal scaling.
    """

    def __init__(
        self,
        queue: ImageProcessingQueue,
        worker_id: str,
        processor_fn: Callable,
        poll_interval: float = 0.5,
    ):
        self.queue = queue
        self.worker_id = worker_id
        self.processor_fn = processor_fn
        self.poll_interval = poll_interval
        self.running = False
        self._stats = WorkerStats(worker_id=worker_id)

    async def start(self):
        """Start worker (blocking)"""
        self.running = True
        self.queue.register_worker(self.worker_id)
        logger.info(f"Worker {self.worker_id} started")

        try:
            while self.running:
                task = self.queue.dequeue(self.worker_id)
                if task is None:
                    await asyncio.sleep(self.poll_interval)
                    continue

                await self._process_task(task)

        except Exception as e:
            logger.error(f"Worker {self.worker_id} error: {e}")
        finally:
            self.queue.unregister_worker(self.worker_id)
            logger.info(f"Worker {self.worker_id} stopped")

    async def _process_task(self, task: ImageProcessingTask):
        """Process a single task"""
        start_time = time.time()
        try:
            logger.info(f"Worker {self.worker_id} processing task {task.task_id}")
            
            result = await self.processor_fn(task)
            
            processing_time = time.time() - start_time
            self.queue.complete_task(task.task_id, result)
            self.queue.update_worker_stats(self.worker_id, processing_time, success=True)
            
            logger.info(f"Task {task.task_id} completed in {processing_time:.2f}s")

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            # Distinguish permanent failures from transient ones.
            # Permanent failures (bad input, type errors, etc.) must not
            # be retried — they waste capacity and delay legitimate work.
            should_retry = _is_retryable(e)
            self.queue.fail_task(task.task_id, error_msg, retry=should_retry)
            self.queue.update_worker_stats(self.worker_id, processing_time, success=False)
            if should_retry:
                logger.warning(
                    f"Task {task.task_id} failed with transient error (will retry): {error_msg}"
                )
            else:
                logger.error(
                    f"Task {task.task_id} failed with permanent error (no retry): "
                    f"{type(e).__name__}: {error_msg}"
                )

    def stop(self):
        """Stop worker gracefully"""
        self.running = False
        logger.info(f"Worker {self.worker_id} stopping")


class ImageProcessingPipeline:
    """
    Orchestrates image processing queue and worker pool.
    Supports horizontal scaling via multiple workers.
    """

    def __init__(self, max_workers: int = 4, max_queue_size: int = 10000):
        self.queue = ImageProcessingQueue(max_queue_size=max_queue_size)
        self.max_workers = max_workers
        self.workers: Dict[str, ImageProcessingWorker] = {}
        self._worker_tasks = {}

    def submit_task(
        self,
        image_data: bytes,
        crop_type: str,
        processor_type: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Submit a new image processing task"""
        task = ImageProcessingTask(
            task_id=f"task-{uuid.uuid4().hex[:12]}",
            image_data=image_data,
            crop_type=crop_type,
            processor_type=processor_type,
            priority=priority,
            metadata=metadata or {},
        )
        return self.queue.enqueue(task)

    def add_worker(self, processor_fn: Callable) -> str:
        """Add a new worker to the pool"""
        if len(self.workers) >= self.max_workers:
            raise RuntimeError(f"Maximum workers ({self.max_workers}) reached")
        
        worker_id = f"worker-{len(self.workers)}-{uuid.uuid4().hex[:8]}"
        worker = ImageProcessingWorker(self.queue, worker_id, processor_fn)
        self.workers[worker_id] = worker
        logger.info(f"Worker {worker_id} added to pool")
        return worker_id

    def scale_up(self, processor_fn: Callable, count: int = 1) -> List[str]:
        """Horizontally scale up by adding workers"""
        added = []
        for _ in range(count):
            try:
                worker_id = self.add_worker(processor_fn)
                added.append(worker_id)
            except RuntimeError:
                logger.warning("Cannot add more workers - max pool size reached")
                break
        return added

    def scale_down(self, count: int = 1) -> List[str]:
        """Horizontally scale down by removing workers"""
        removed = []
        worker_ids = list(self.workers.keys())[-count:]
        for worker_id in worker_ids:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.stop()
                del self.workers[worker_id]
                removed.append(worker_id)
                logger.info(f"Worker {worker_id} removed from pool")
        return removed

    def get_status(self, task_id: str) -> Optional[Dict]:
        """Get task status"""
        return self.queue.get_task_status(task_id)

    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        stats = self.queue.get_queue_stats()
        stats["max_workers"] = self.max_workers
        stats["current_workers"] = len(self.workers)
        return stats

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        return self.queue.cancel_task(task_id)


# Global pipeline instance
_global_pipeline: Optional[ImageProcessingPipeline] = None


def get_pipeline() -> ImageProcessingPipeline:
    """Get or create global pipeline instance"""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = ImageProcessingPipeline(max_workers=4, max_queue_size=10000)
    return _global_pipeline


def init_pipeline(max_workers: int = 4, max_queue_size: int = 10000) -> ImageProcessingPipeline:
    """Initialize global pipeline"""
    global _global_pipeline
    _global_pipeline = ImageProcessingPipeline(max_workers=max_workers, max_queue_size=max_queue_size)
    return _global_pipeline
# Image processing optimization complete
