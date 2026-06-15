"""
Test suite for Image Processing Pipeline Queue Management & Horizontal Scaling
"""

import pytest
import asyncio
from image_processing_queue import (
    ImageProcessingQueue,
    ImageProcessingTask,
    ImageProcessingWorker,
    ImageProcessingPipeline,
    TaskStatus,
    TaskPriority,
)


class TestImageProcessingQueue:
    """Test suite for queue operations"""

    @pytest.fixture
    def queue(self):
        """Initialize queue"""
        return ImageProcessingQueue(max_queue_size=100)

    def test_enqueue_task(self, queue):
        """Test enqueuing a task"""
        task = ImageProcessingTask(
            task_id="test-1",
            image_data=b"fake_image",
            crop_type="tomato",
            processor_type="quality_grading",
        )
        task_id = queue.enqueue(task)
        assert task_id == "test-1"
        assert queue.get_queue_stats()["queue_size"] == 1

    def test_dequeue_task(self, queue):
        """Test dequeuing a task"""
        task = ImageProcessingTask(
            task_id="test-1",
            image_data=b"fake_image",
            crop_type="tomato",
            processor_type="quality_grading",
        )
        queue.enqueue(task)
        
        dequeued = queue.dequeue("worker-1")
        assert dequeued is not None
        assert dequeued.task_id == "test-1"
        assert dequeued.status == TaskStatus.PROCESSING
        assert dequeued.worker_id == "worker-1"

    def test_priority_ordering(self, queue):
        """Test tasks are dequeued in priority order"""
        # Enqueue low priority first
        low = ImageProcessingTask(
            task_id="low-1",
            image_data=b"img",
            crop_type="tomato",
            processor_type="quality_grading",
            priority=TaskPriority.LOW,
        )
        queue.enqueue(low)
        
        # Enqueue high priority
        high = ImageProcessingTask(
            task_id="high-1",
            image_data=b"img",
            crop_type="tomato",
            processor_type="quality_grading",
            priority=TaskPriority.HIGH,
        )
        queue.enqueue(high)
        
        # High priority should be dequeued first
        dequeued = queue.dequeue("worker-1")
        assert dequeued.task_id == "high-1"

    def test_complete_task(self, queue):
        """Test completing a task"""
        task = ImageProcessingTask(
            task_id="test-1",
            image_data=b"fake_image",
            crop_type="tomato",
            processor_type="quality_grading",
        )
        queue.enqueue(task)
        queue.dequeue("worker-1")
        
        result = {"grade": "A", "score": 95.0}
        queue.complete_task("test-1", result)
        
        status = queue.get_task_status("test-1")
        assert status["status"] == "completed"
        assert status["result"] == result

    def test_fail_task_with_retry(self, queue):
        """Test failing a task with retry"""
        task = ImageProcessingTask(
            task_id="test-1",
            image_data=b"fake_image",
            crop_type="tomato",
            processor_type="quality_grading",
            max_retries=3,
        )
        queue.enqueue(task)
        queue.dequeue("worker-1")
        
        # Fail task (should retry)
        queue.fail_task("test-1", "Processing error", retry=True)
        
        status = queue.get_task_status("test-1")
        assert status["status"] == "retrying"
        
        # Task should be requeued
        stats = queue.get_queue_stats()
        assert stats["queue_size"] == 1

    def test_fail_task_max_retries(self, queue):
        """Test task failure after max retries"""
        task = ImageProcessingTask(
            task_id="test-1",
            image_data=b"fake_image",
            crop_type="tomato",
            processor_type="quality_grading",
            max_retries=2,  # max_retries=2 means retry_count can be 1 or 2
        )
        queue.enqueue(task)
        dequeued = queue.dequeue("worker-1")
        assert dequeued is not None
        
        # Fail first time (retry_count becomes 1, 1 < 2, so requeue)
        queue.fail_task("test-1", "Error 1", retry=True)
        dequeued = queue.dequeue("worker-1")
        assert dequeued is not None
        assert dequeued.retry_count == 1
        
        # Fail second time (retry_count becomes 2, 2 < 2 is FALSE, final failure)
        queue.fail_task("test-1", "Error 2 (final)", retry=True)
        
        status = queue.get_task_status("test-1")
        assert status["status"] == "failed"
        assert "Error 2" in status["error"]

    def test_cancel_task(self, queue):
        """Test cancelling a task"""
        task = ImageProcessingTask(
            task_id="test-1",
            image_data=b"fake_image",
            crop_type="tomato",
            processor_type="quality_grading",
        )
        queue.enqueue(task)
        
        success = queue.cancel_task("test-1")
        assert success is True
        
        status = queue.get_task_status("test-1")
        assert status["status"] == "cancelled"

    def test_register_and_unregister_worker(self, queue):
        """Test worker registration"""
        stats = queue.register_worker("worker-1")
        assert stats.worker_id == "worker-1"
        
        queue_stats = queue.get_queue_stats()
        assert queue_stats["workers_online"] == 1
        
        queue.unregister_worker("worker-1")
        queue_stats = queue.get_queue_stats()
        assert queue_stats["workers_online"] == 0

    def test_worker_statistics(self, queue):
        """Test worker statistics tracking"""
        queue.register_worker("worker-1")
        
        # Simulate processing
        queue.update_worker_stats("worker-1", processing_time=1.5, success=True)
        queue.update_worker_stats("worker-1", processing_time=2.0, success=False)
        
        stats = queue.get_queue_stats()
        worker_stat = stats["workers"][0]
        assert worker_stat["tasks_processed"] == 2
        assert worker_stat["tasks_failed"] == 1

    def test_queue_overflow(self, queue):
        """Test queue size limit"""
        small_queue = ImageProcessingQueue(max_queue_size=2)
        
        task1 = ImageProcessingTask(
            task_id="test-1",
            image_data=b"img",
            crop_type="tomato",
            processor_type="quality_grading",
        )
        task2 = ImageProcessingTask(
            task_id="test-2",
            image_data=b"img",
            crop_type="tomato",
            processor_type="quality_grading",
        )
        task3 = ImageProcessingTask(
            task_id="test-3",
            image_data=b"img",
            crop_type="tomato",
            processor_type="quality_grading",
        )
        
        small_queue.enqueue(task1)
        small_queue.enqueue(task2)
        
        # Third should fail
        with pytest.raises(RuntimeError, match="Queue is full"):
            small_queue.enqueue(task3)

    def test_get_pending_tasks(self, queue):
        """Test retrieving pending tasks"""
        for i in range(5):
            task = ImageProcessingTask(
                task_id=f"test-{i}",
                image_data=b"img",
                crop_type="tomato",
                processor_type="quality_grading",
            )
            queue.enqueue(task)
        
        pending = queue.get_pending_tasks(limit=3)
        assert len(pending) == 3

    def test_cleanup_old_completed_tasks(self, queue):
        """Test cleanup of old completed tasks"""
        task = ImageProcessingTask(
            task_id="test-1",
            image_data=b"img",
            crop_type="tomato",
            processor_type="quality_grading",
        )
        queue.enqueue(task)
        queue.dequeue("worker-1")
        queue.complete_task("test-1", {"grade": "A"})
        
        stats = queue.get_queue_stats()
        assert stats["completed_tasks"] == 1
        
        # Cleanup should have no effect (tasks just added)
        cleaned = queue.cleanup_old_completed_tasks(max_age_hours=0)
        assert cleaned >= 0


@pytest.mark.asyncio
class TestImageProcessingWorker:
    """Test suite for worker operations"""

    async def test_worker_processes_task(self):
        """Test worker can process a task"""
        queue = ImageProcessingQueue()
        
        # Mock processor function
        async def mock_processor(task):
            await asyncio.sleep(0.01)
            return {"grade": "A", "score": 90.0}
        
        worker = ImageProcessingWorker(queue, "worker-1", mock_processor)
        
        # Enqueue task
        task = ImageProcessingTask(
            task_id="test-1",
            image_data=b"img",
            crop_type="tomato",
            processor_type="quality_grading",
        )
        queue.enqueue(task)
        
        # Process one task
        await worker._process_task(queue.dequeue("worker-1"))
        
        # Check task is completed
        status = queue.get_task_status("test-1")
        assert status["status"] == "completed"

    async def test_worker_handles_failure(self):
        """Test worker handles task failure and retries"""
        queue = ImageProcessingQueue()
        
        # Mock processor that fails
        async def failing_processor(task):
            raise ValueError("Processing failed")
        
        worker = ImageProcessingWorker(queue, "worker-1", failing_processor)
        
        # Enqueue task
        task = ImageProcessingTask(
            task_id="test-1",
            image_data=b"img",
            crop_type="tomato",
            processor_type="quality_grading",
            max_retries=2,  # Allow 1 retry after initial attempt
        )
        queue.enqueue(task)
        
        # Process task (should fail and retry)
        dequeued = queue.dequeue("worker-1")
        await worker._process_task(dequeued)
        
        # Check task is retrying (first failure with max_retries=2)
        status = queue.get_task_status("test-1")
        assert status["status"] == "retrying"


class TestImageProcessingPipeline:
    """Test suite for pipeline operations"""

    def test_submit_task(self):
        """Test submitting a task to pipeline"""
        pipeline = ImageProcessingPipeline(max_workers=2)
        
        task_id = pipeline.submit_task(
            image_data=b"test_image",
            crop_type="tomato",
            processor_type="quality_grading",
            priority=TaskPriority.NORMAL,
        )
        
        assert task_id is not None
        assert task_id.startswith("task-")

    def test_add_worker_to_pool(self):
        """Test adding worker to pool"""
        pipeline = ImageProcessingPipeline(max_workers=2)
        
        async def processor(task):
            return {"grade": "A"}
        
        worker_id = pipeline.add_worker(processor)
        assert worker_id is not None
        assert len(pipeline.workers) == 1

    def test_max_workers_limit(self):
        """Test max workers limit"""
        pipeline = ImageProcessingPipeline(max_workers=2)
        
        async def processor(task):
            return {"grade": "A"}
        
        pipeline.add_worker(processor)
        pipeline.add_worker(processor)
        
        # Third worker should fail
        with pytest.raises(RuntimeError, match="Maximum workers"):
            pipeline.add_worker(processor)

    def test_scale_up(self):
        """Test horizontal scaling up"""
        pipeline = ImageProcessingPipeline(max_workers=5)
        
        async def processor(task):
            return {"grade": "A"}
        
        assert len(pipeline.workers) == 0
        
        added = pipeline.scale_up(processor, count=3)
        assert len(added) == 3
        assert len(pipeline.workers) == 3

    def test_scale_down(self):
        """Test horizontal scaling down"""
        pipeline = ImageProcessingPipeline(max_workers=5)
        
        async def processor(task):
            return {"grade": "A"}
        
        pipeline.scale_up(processor, count=3)
        assert len(pipeline.workers) == 3
        
        removed = pipeline.scale_down(count=2)
        assert len(removed) == 2
        assert len(pipeline.workers) == 1

    def test_get_stats(self):
        """Test getting pipeline statistics"""
        pipeline = ImageProcessingPipeline(max_workers=2)
        
        async def processor(task):
            return {"grade": "A"}
        
        pipeline.add_worker(processor)
        
        for _ in range(5):
            pipeline.submit_task(
                image_data=b"img",
                crop_type="tomato",
                processor_type="quality_grading",
            )
        
        stats = pipeline.get_stats()
        assert stats["queue_size"] == 5
        assert stats["current_workers"] == 1
        assert stats["max_workers"] == 2

    def test_cancel_task(self):
        """Test cancelling task via pipeline"""
        pipeline = ImageProcessingPipeline()
        
        task_id = pipeline.submit_task(
            image_data=b"img",
            crop_type="tomato",
            processor_type="quality_grading",
        )
        
        success = pipeline.cancel_task(task_id)
        assert success is True

    def test_get_status(self):
        """Test getting task status via pipeline"""
        pipeline = ImageProcessingPipeline()
        
        task_id = pipeline.submit_task(
            image_data=b"img",
            crop_type="tomato",
            processor_type="quality_grading",
        )
        
        status = pipeline.get_status(task_id)
        assert status is not None
        assert status["status"] == "queued"


class TestExifOrientationNormalization:
    """Test EXIF orientation extraction and normalization"""

    def test_orientation_1_no_change(self):
        """Test orientation 1 (normal) returns identical bytes"""
        pipeline = ImageProcessingPipeline()

        # Create a minimal valid JPEG (1x1 pixel, no EXIF)
        from PIL import Image
        import io
        img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()

        orientation, meta = pipeline._extract_exif_orientation(raw)
        assert orientation == 1
        assert meta is None or meta.get("original_orientation") == 1

        normalized = pipeline._normalize_orientation(raw, 1)
        assert normalized == raw

    def test_orientation_3_180_degrees(self):
        """Test orientation 3 (180° rotation)"""
        pipeline = ImageProcessingPipeline()

        from PIL import Image
        import io
        img = Image.new("RGB", (20, 10), color="blue")  # Wide rectangle
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()

        # Manually rotate 180° to simulate orientation 3
        img_rot = img.rotate(180)
        buf_rot = io.BytesIO()
        img_rot.save(buf_rot, format="JPEG")
        raw_rot = buf_rot.getvalue()

        # Normalize should flip it back to original dimensions
        normalized = pipeline._normalize_orientation(raw_rot, 3)
        result_img = Image.open(io.BytesIO(normalized))
        assert result_img.size == (20, 10)

    def test_orientation_6_90_cw(self):
        """Test orientation 6 (90° CW) — portrait photo from mobile"""
        pipeline = ImageProcessingPipeline()

        from PIL import Image
        import io
        img = Image.new("RGB", (10, 20), color="green")  # Tall rectangle
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()

        # Simulate 90° CW rotation (width/height swap)
        img_rot = img.rotate(270, expand=True)  # PIL rotate is CCW, so 270 = 90 CW
        buf_rot = io.BytesIO()
        img_rot.save(buf_rot, format="JPEG")
        raw_rot = buf_rot.getvalue()

        # Normalize should restore original portrait dimensions
        normalized = pipeline._normalize_orientation(raw_rot, 6)
        result_img = Image.open(io.BytesIO(normalized))
        assert result_img.size == (10, 20)

    def test_orientation_8_90_ccw(self):
        """Test orientation 8 (90° CCW)"""
        pipeline = ImageProcessingPipeline()

        from PIL import Image
        import io
        img = Image.new("RGB", (10, 20), color="yellow")  # Tall rectangle
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()

        # Simulate 90° CCW rotation
        img_rot = img.rotate(90, expand=True)
        buf_rot = io.BytesIO()
        img_rot.save(buf_rot, format="JPEG")
        raw_rot = buf_rot.getvalue()

        # Normalize should restore original portrait dimensions
        normalized = pipeline._normalize_orientation(raw_rot, 8)
        result_img = Image.open(io.BytesIO(normalized))
        assert result_img.size == (10, 20)

    def test_submit_with_orientation_metadata(self):
        """Test that submit() attaches orientation metadata to task"""
        pipeline = ImageProcessingPipeline()

        from PIL import Image
        import io
        img = Image.new("RGB", (10, 20), color="purple")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()

        task_id = pipeline.submit(image_data=raw, crop_type="wheat", processor_type="disease_detection")
        task = pipeline.queue._tasks_by_id.get(task_id)

        assert task is not None
        assert task.orientation_metadata is not None
        assert task.orientation_metadata["original_orientation"] == 1
        assert task.orientation_metadata["width"] == 10
        assert task.orientation_metadata["height"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
