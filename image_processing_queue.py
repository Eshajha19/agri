"""
Image Processing Pipeline Queue Management & Horizontal Scaling System

Provides:
- Distributed task queue for image processing
- Worker pool management with horizontal scaling
- Task status tracking and monitoring
- Async processing with callbacks
- Optional Redis support for distributed deployments
"""

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

logger = logging.getLogger(__name__)


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

    def __init__(self, max_queue_size: int = 10000, enable_persistence: bool = False):
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence
        
        # Task storage (heap of (priority, counter, task) tuples)
        self._task_queue: List[tuple] = []
        self._tasks_by_id: Dict[str, ImageProcessingTask] = {}
        self._counter = 0
        self._completed_tasks: Dict[str, ImageProcessingTask] = {}  # History
        
        # Worker management
        self._workers: Dict[str, WorkerStats] = {}
        self._worker_lock = threading.Lock()
        
        # Thread safety
        self._queue_lock = threading.Lock()
        self._task_lock = threading.Lock()
        
        # Metrics
        self._total_enqueued = 0
        self._total_processed = 0
        self._total_failed = 0

    def enqueue(self, task: ImageProcessingTask) -> str:
        """Enqueue a task for processing"""
        with self._queue_lock:
            if len(self._task_queue) >= self.max_queue_size:
                raise RuntimeError(f"Queue is full (max: {self.max_queue_size})")
            self._task_queue.append(task)
            heapq.heappush(self._task_queue, (task.priority.value, self._counter, task))
            self._counter += 1
            self._tasks_by_id[task.task_id] = task
            self._total_enqueued += 1
            
        logger.info(f"Task {task.task_id} enqueued (priority: {task.priority.name}, queue_size: {len(self._task_queue)})")
        return task.task_id

    def dequeue(self, worker_id: str) -> Optional[ImageProcessingTask]:
        """Dequeue highest priority task for worker"""
        with self._queue_lock:
            if not self._task_queue:
                return None

            _, _, task = heapq.heappop(self._task_queue)

            # Update task status
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
                # Re-enqueue for retry
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
        """Cancel a queued or processing task"""
        with self._queue_lock:
            if task_id not in self._tasks_by_id:
                return False

            task = self._tasks_by_id[task_id]
            if task.status in (TaskStatus.QUEUED, TaskStatus.RETRYING):
                task.status = TaskStatus.CANCELLED
                self._task_queue = deque(
                    t for t in self._task_queue if t.task_id != task_id
                )
                del self._tasks_by_id[task_id]
                self._completed_tasks[task_id] = task
                logger.info(f"Task {task_id} cancelled")
                return True

            return False

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
            if success:
                worker.tasks_processed += 1
            else:
                worker.tasks_failed += 1
            
            # Update average processing time (exponential moving average)
            if worker.avg_processing_time == 0:
                worker.avg_processing_time = processing_time
            else:
                worker.avg_processing_time = (worker.avg_processing_time * 0.7) + (processing_time * 0.3)
            
            worker.last_heartbeat = datetime.now().isoformat()

    def get_queue_stats(self) -> Dict:
        """Get queue and worker statistics"""
        with self._queue_lock:
            queue_size = len(self._task_queue)
        
        with self._task_lock:
            active_tasks = len(self._tasks_by_id)
            completed_tasks = len(self._completed_tasks)
        
        with self._worker_lock:
            workers_online = len(self._workers)
            worker_stats = list(self._workers.values())
        
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
        }

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
            self.queue.fail_task(task.task_id, error_msg, retry=True)
            self.queue.update_worker_stats(self.worker_id, processing_time, success=False)
            logger.error(f"Task {task.task_id} failed: {error_msg}")

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
