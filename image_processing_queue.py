"""
Image Processing Pipeline Queue Management & Horizontal Scaling System

Provides:
- Thread-safe priority queue
- Worker pool with horizontal scaling
- Retry + backoff support
- Task lifecycle tracking
- Optional persistence + caching hooks
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
import random
from PIL import Image, ExifTags
import io

logger = logging.getLogger(__name__)


# -----------------------------
# LRU Cache
# -----------------------------
class LRUCache:
    def __init__(self, capacity: int = 1000, ttl_seconds: int = 86400):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache: OrderedDict[str, tuple] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            if key not in self.cache:
                return None

            value, ts = self.cache[key]
            if time.time() - ts > self.ttl:
                del self.cache[key]
                return None

            self.cache.move_to_end(key)
            return value

    def put(self, key: str, value: Any):
        with self.lock:
            self.cache[key] = (value, time.time())
            self.cache.move_to_end(key)

            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


# -----------------------------
# Enums
# -----------------------------
class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


# -----------------------------
# Task Model
# -----------------------------
@dataclass
class ImageProcessingTask:
    task_id: str
    image_data: bytes
    crop_type: str
    processor_type: str
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
    orientation_metadata: Optional[Dict] = None


# -----------------------------
# Worker Stats
# -----------------------------
@dataclass
class WorkerStats:
    worker_id: str
    tasks_processed: int = 0
    tasks_failed: int = 0
    avg_processing_time: float = 0.0
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())


# -----------------------------
# Queue Core
# -----------------------------
class ImageProcessingQueue:
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size

        self._task_queue: List[tuple] = []
        self._tasks_by_id: Dict[str, ImageProcessingTask] = {}
        self._completed_tasks: Dict[str, ImageProcessingTask] = {}

        self._ack_store: Dict[str, str] = {}

        self._counter = 0

        self._queue_lock = threading.Lock()
        self._task_lock = threading.Lock()

        self._total_enqueued = 0
        self._total_processed = 0
        self._total_failed = 0

    def enqueue(self, task: ImageProcessingTask) -> str:
        """Enqueue a task for processing"""
        with self._queue_lock:
            if len(self._task_queue) >= self.max_queue_size:
                raise RuntimeError(f"Queue is full (max: {self.max_queue_size})")
            heapq.heappush(self._task_queue, (task.priority.value, self._counter, task))
            self._counter += 1
            self._tasks_by_id[task.task_id] = task
            self._total_enqueued += 1
            
        logger.info(f"Task {task.task_id} enqueued (priority: {task.priority.name}, queue_size: {len(self._task_queue)})")
        return task.task_id

    self._enqueue_task(task)
    # -------------------------
    # Dequeue
    # -------------------------
    def dequeue(self, worker_id: str) -> Optional[ImageProcessingTask]:
        with self._queue_lock:
            while self._task_queue:
                _, _, task = heapq.heappop(self._task_queue)

                if self._skip_task(task):
                    continue

                self._set_status(task, TaskStatus.PROCESSING)
                task.worker_id = worker_id
                return task

        return None

    # -------------------------
    # Complete
    # -------------------------
    def complete_task(self, task_id: str, result: Dict) -> bool:
        with self._task_lock:
            task = self._tasks_by_id.get(task_id)
            if not task:
                return False

            self._set_status(task, TaskStatus.COMPLETED)
            task.result = result
            task.image_data = b""

            del self._tasks_by_id[task_id]
            self._completed_tasks[task_id] = task

            self._ack_store[task_id] = TaskStatus.COMPLETED.value
            self._total_processed += 1

        self._persist_ack()
        return True

    # -------------------------
    # Fail
    # -------------------------
    def fail_task(self, task_id: str, error: str, retry: bool = True) -> bool:
        """Mark task as failed with optional retry"""
        need_requeue = False
        with self._task_lock:
            task = self._tasks_by_id.get(task_id)
            if not task:
                return False

            task = self._tasks_by_id[task_id]
            task.retry_count += 1

            if retry and task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRYING
                need_requeue = True
            else:
                task.status = TaskStatus.FAILED
                task.error = error
                task.completed_at = datetime.now().isoformat()
                del self._tasks_by_id[task_id]
                self._completed_tasks[task_id] = task
                self._total_failed += 1
                logger.error(f"Task {task_id} failed after {task.retry_count} retries: {error}")
                return False

        if need_requeue:
            with self._queue_lock:
                heapq.heappush(self._task_queue, (task.priority.value, self._counter, task))
                self._counter += 1
            logger.info(f"Task {task_id} requeued for retry ({task.retry_count}/{task.max_retries})")
            return True

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

            del self._tasks_by_id[task_id]
            self._completed_tasks[task_id] = task
            self._total_failed += 1

        return False

    # -------------------------
    # Cancel
    # -------------------------
    def cancel_task(self, task_id: str) -> bool:
        with self._task_lock:
            task = self._tasks_by_id.get(task_id)
            if not task:
                return False

            task = self._tasks_by_id[task_id]
            if task.status in (TaskStatus.QUEUED, TaskStatus.RETRYING):
                task.status = TaskStatus.CANCELLED
                self._task_queue = [
                    entry for entry in self._task_queue if entry[2].task_id != task_id
                ]
                del self._tasks_by_id[task_id]
                self._completed_tasks[task_id] = task
                logger.info(f"Task {task_id} cancelled")
                return True

            self._set_status(task, TaskStatus.CANCELLED)
            task.image_data = b""

            del self._tasks_by_id[task_id]
            self._completed_tasks[task_id] = task

        with self._queue_lock:
            self._task_queue = [
                e for e in self._task_queue if e[2].task_id != task_id
            ]
            heapq.heapify(self._task_queue)

        return True

    # -------------------------
    # Stats
    # -------------------------
    def get_stats(self) -> Dict:
        with self._queue_lock:
            qsize = len(self._task_queue)

        with self._task_lock:
            active = len(self._tasks_by_id)
            done = len(self._completed_tasks)

        return {
            "queue_size": qsize,
            "active": active,
            "completed": done,
            "enqueued": self._total_enqueued,
            "processed": self._total_processed,
            "failed": self._total_failed,
        }


# -----------------------------
# Worker
# -----------------------------
class ImageProcessingWorker:
    def __init__(self, queue: ImageProcessingQueue, worker_id: str, processor_fn: Callable):
        self.queue = queue
        self.worker_id = worker_id
        self.processor_fn = processor_fn
        self.running = True

    async def start(self):
        while self.running:
            task = self.queue.dequeue(self.worker_id)

            if not task:
                await asyncio.sleep(0.3)
                continue

            await self._process(task)

    async def _process(self, task: ImageProcessingTask):
        start = time.time()
        try:
            result = await self.processor_fn(task)
            self.queue.complete_task(task.task_id, result)

        except Exception as e:
            self.queue.fail_task(task.task_id, str(e), retry=True)

        finally:
            _ = time.time() - start

    def stop(self):
        self.running = False


# -----------------------------
# Pipeline
# -----------------------------
class ImageProcessingPipeline:
    def __init__(self, max_workers: int = 4):
        self.queue = ImageProcessingQueue()
        self.workers: Dict[str, ImageProcessingWorker] = {}
        self.max_workers = max_workers

    def _extract_exif_orientation(self, image_data: bytes) -> tuple[int, Optional[Dict]]:
        """Extract EXIF orientation from raw image bytes. Returns (orientation, metadata_dict)."""
        try:
            img = Image.open(io.BytesIO(image_data))
            exif = img._getexif()
            if exif is None:
                return 1, None

            orientation_tag = next(
                (tag for tag, name in ExifTags.TAGS.items() if name == "Orientation"),
                None,
            )
            if orientation_tag is None:
                return 1, None

            orientation = exif.get(orientation_tag, 1)
            metadata = {
                "original_orientation": orientation,
                "width": img.width,
                "height": img.height,
                "format": img.format,
            }
            return orientation, metadata
        except Exception as exc:
            logger.warning("EXIF extraction failed: %s", exc)
            return 1, None

    def _normalize_orientation(self, image_data: bytes, orientation: int) -> bytes:
        """Apply rotation/flop based on EXIF orientation tag, return normalized JPEG bytes."""
        if orientation == 1:
            return image_data  # Normal, no change

        try:
            img = Image.open(io.BytesIO(image_data))

            # Orientation mapping: https://jdhao.github.io/2019/07/31/image_rotation_exif_info/
            transforms = {
                2: (Image.FLIP_LEFT_RIGHT,),
                3: (Image.ROTATE_180,),
                4: (Image.FLIP_TOP_BOTTOM,),
                5: (Image.TRANSPOSE,),  # Mirror across top-left diagonal
                6: (Image.ROTATE_270,),  # 90° CW
                7: (Image.TRANSVERSE,),  # Mirror across top-right diagonal
                8: (Image.ROTATE_90,),    # 90° CCW
            }

            for transform in transforms.get(orientation, ()):
                img = img.transpose(transform)

            # Strip EXIF and save as JPEG
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=95)
            normalized = output.getvalue()

            logger.info(
                "Normalized EXIF orientation %d for image (size: %d -> %d bytes)",
                orientation,
                len(image_data),
                len(normalized),
            )
            return normalized
        except Exception as exc:
            logger.error("Orientation normalization failed: %s", exc)
            return image_data  # Fallback to original

    def submit(self, image_data: bytes, **kwargs) -> str:
        orientation, orientation_meta = self._extract_exif_orientation(image_data)

        if orientation != 1 and orientation_meta:
            logger.warning(
                "Image submitted with EXIF orientation %d (expected 1). Normalizing before queueing.",
                orientation,
            )
            image_data = self._normalize_orientation(image_data, orientation)

        task = ImageProcessingTask(
            task_id=f"task-{uuid.uuid4().hex[:12]}",
            image_data=image_data,
            orientation_metadata=orientation_meta,
            **kwargs,
        )
        return self.queue.enqueue(task)

    def add_worker(self, processor_fn: Callable):
        if len(self.workers) >= self.max_workers:
            raise RuntimeError("Max workers reached")

        wid = f"worker-{len(self.workers)}"
        worker = ImageProcessingWorker(self.queue, wid, processor_fn)
        self.workers[wid] = worker
        return wid
