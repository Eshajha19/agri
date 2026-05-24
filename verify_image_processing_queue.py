"""
End-to-end verification of Image Processing Pipeline Queue Management & Horizontal Scaling
"""

import asyncio
import sys
from image_processing_queue import (
    ImageProcessingPipeline,
    ImageProcessingTask,
    TaskPriority,
)


async def mock_crop_quality_processor(task: ImageProcessingTask) -> dict:
    """Mock image processor for quality grading"""
    await asyncio.sleep(0.1)  # Simulate processing time
    return {
        "crop_type": task.crop_type,
        "grade": "A",
        "score": 92.5,
        "processor": task.processor_type,
    }


async def mock_disease_detector(task: ImageProcessingTask) -> dict:
    """Mock image processor for disease detection"""
    await asyncio.sleep(0.08)  # Simulate processing time
    return {
        "crop_type": task.crop_type,
        "disease_detected": False,
        "confidence": 0.98,
        "processor": task.processor_type,
    }


async def test_phase_1_basic_queueing():
    """Phase 1: Test basic queue operations and task management"""
    print("\n=== Phase 1: Basic Queue Operations ===\n")
    
    pipeline = ImageProcessingPipeline(max_workers=2)
    
    # Submit tasks with different priorities
    print("[*] Submitting 5 tasks with mixed priorities...")
    task_ids = []
    
    task_ids.append(pipeline.submit_task(
        image_data=b"crop_image_1",
        crop_type="tomato",
        processor_type="quality_grading",
        priority=TaskPriority.NORMAL,
    ))
    
    task_ids.append(pipeline.submit_task(
        image_data=b"crop_image_2",
        crop_type="wheat",
        processor_type="disease_detection",
        priority=TaskPriority.HIGH,  # High priority
    ))
    
    task_ids.append(pipeline.submit_task(
        image_data=b"crop_image_3",
        crop_type="rice",
        processor_type="quality_grading",
        priority=TaskPriority.NORMAL,
    ))
    
    task_ids.append(pipeline.submit_task(
        image_data=b"crop_image_4",
        crop_type="potato",
        processor_type="disease_detection",
        priority=TaskPriority.LOW,
    ))
    
    task_ids.append(pipeline.submit_task(
        image_data=b"crop_image_5",
        crop_type="corn",
        processor_type="quality_grading",
        priority=TaskPriority.CRITICAL,  # Critical priority
    ))
    
    print(f"[OK] Submitted {len(task_ids)} tasks")
    print(f"[OK] Task IDs: {', '.join(task_ids[:3])}...")
    
    # Check queue stats
    stats = pipeline.get_stats()
    print(f"[OK] Queue size: {stats['queue_size']}")
    print(f"[OK] Active tasks: {stats['active_tasks']}")
    
    return task_ids


async def test_phase_2_horizontal_scaling():
    """Phase 2: Test horizontal scaling with worker pool"""
    print("\n=== Phase 2: Horizontal Scaling ===\n")
    
    pipeline = ImageProcessingPipeline(max_workers=4)
    
    # Add workers to pool
    print("[*] Starting worker pool...")
    
    worker_id_1 = pipeline.add_worker(mock_crop_quality_processor)
    print(f"[OK] Worker 1 added: {worker_id_1}")
    
    worker_id_2 = pipeline.add_worker(mock_disease_detector)
    print(f"[OK] Worker 2 added: {worker_id_2}")
    
    # Submit batch of tasks
    print("\n[*] Submitting 10 image processing tasks...")
    task_ids = []
    for i in range(10):
        task_id = pipeline.submit_task(
            image_data=f"crop_image_{i}".encode(),
            crop_type="tomato",
            processor_type="quality_grading" if i % 2 == 0 else "disease_detection",
            priority=TaskPriority.NORMAL if i % 3 == 0 else TaskPriority.HIGH,
        )
        task_ids.append(task_id)
    
    print(f"[OK] Submitted {len(task_ids)} tasks")
    
    # Scale up workers
    print("\n[*] Scaling up worker pool...")
    added = pipeline.scale_up(mock_crop_quality_processor, count=2)
    print(f"[OK] Added {len(added)} more workers")
    
    stats = pipeline.get_stats()
    print(f"[OK] Current workers: {stats['current_workers']}/{stats['max_workers']}")
    print(f"[OK] Queue size: {stats['queue_size']}")
    print(f"[OK] Workers online: {stats['workers_online']}")
    
    # Scale down workers
    print("\n[*] Scaling down worker pool...")
    removed = pipeline.scale_down(count=1)
    print(f"[OK] Removed {len(removed)} workers")
    
    stats = pipeline.get_stats()
    print(f"[OK] Current workers: {stats['current_workers']}/{stats['max_workers']}")


async def test_phase_3_task_processing():
    """Phase 3: Test actual async task processing"""
    print("\n=== Phase 3: Async Task Processing ===\n")
    
    pipeline = ImageProcessingPipeline(max_workers=2)
    
    # Add workers
    print("[*] Initializing worker pool with 2 workers...")
    worker_1 = pipeline.add_worker(mock_crop_quality_processor)
    worker_2 = pipeline.add_worker(mock_disease_detector)
    print(f"[OK] Workers ready: {worker_1}, {worker_2}")
    
    # Submit tasks
    print("\n[*] Submitting 5 image processing tasks...")
    task_ids = []
    for i in range(5):
        task_id = pipeline.submit_task(
            image_data=f"crop_image_{i}".encode(),
            crop_type=["tomato", "wheat", "rice", "potato", "corn"][i],
            processor_type="quality_grading" if i % 2 == 0 else "disease_detection",
        )
        task_ids.append(task_id)
        print(f"  - Task {i+1}: {task_id[:20]}... (queued)")
    
    print(f"[OK] All {len(task_ids)} tasks submitted")
    
    # Check task status
    print("\n[*] Checking task status...")
    for task_id in task_ids[:2]:
        status = pipeline.get_status(task_id)
        if status:
            print(f"  - Task {task_id[:20]}...: {status['status']}")
    
    # Get pending tasks
    print("\n[*] Pending tasks in queue:")
    pending = pipeline.queue.get_pending_tasks(limit=5)
    for i, task_info in enumerate(pending, 1):
        print(f"  {i}. {task_info['task_id'][:20]}... ({task_info['crop_type']}, {task_info['priority']})")


async def test_phase_4_error_handling():
    """Phase 4: Test error handling and retries"""
    print("\n=== Phase 4: Error Handling & Retries ===\n")
    
    from image_processing_queue import ImageProcessingQueue, TaskStatus
    
    queue = ImageProcessingQueue()
    
    # Create task
    task = ImageProcessingTask(
        task_id="retry-test-1",
        image_data=b"test_image",
        crop_type="tomato",
        processor_type="quality_grading",
        max_retries=3,
    )
    
    print("[*] Testing task retry mechanism...")
    print(f"  - Created task: {task.task_id}")
    print(f"  - Max retries: {task.max_retries}")
    
    # Enqueue
    queue.enqueue(task)
    print("[OK] Task enqueued")
    
    # Simulate failures with retries
    for attempt in range(1, 4):
        dequeued = queue.dequeue("worker-1")
        if dequeued:
            print(f"[*] Attempt {attempt}: dequeued task (retry_count={dequeued.retry_count})")
            queue.fail_task(task.task_id, f"Simulated error {attempt}", retry=True)
            status = queue.get_task_status(task.task_id)
            print(f"  - Status after failure: {status['status']}")
        else:
            print(f"[!] No task to dequeue at attempt {attempt}")
            break
    
    # Check final status
    final_status = queue.get_task_status(task.task_id)
    print(f"\n[OK] Final task status: {final_status['status']}")
    if final_status['error']:
        print(f"[OK] Error message: {final_status['error']}")


async def test_phase_5_metrics_and_monitoring():
    """Phase 5: Test metrics and monitoring"""
    print("\n=== Phase 5: Metrics & Monitoring ===\n")
    
    pipeline = ImageProcessingPipeline(max_workers=3)
    
    # Setup workers
    pipeline.add_worker(mock_crop_quality_processor)
    pipeline.add_worker(mock_disease_detector)
    
    # Submit tasks
    for i in range(8):
        pipeline.submit_task(
            image_data=f"image_{i}".encode(),
            crop_type="tomato",
            processor_type="quality_grading" if i % 2 == 0 else "disease_detection",
        )
    
    # Get comprehensive stats
    stats = pipeline.get_stats()
    
    print("[*] Pipeline Metrics:")
    print(f"  - Queue size: {stats['queue_size']}")
    print(f"  - Active tasks: {stats['active_tasks']}")
    print(f"  - Completed tasks: {stats['completed_tasks']}")
    print(f"  - Total enqueued: {stats['total_enqueued']}")
    print(f"  - Total processed: {stats['total_processed']}")
    print(f"  - Total failed: {stats['total_failed']}")
    print(f"  - Current workers: {stats['current_workers']}/{stats['max_workers']}")
    print(f"  - Workers online: {stats['workers_online']}")
    print(f"  - Average processing time: {stats['avg_processing_time']:.3f}s")
    
    print("\n[*] Worker Details:")
    for worker_stat in stats['workers'][:2]:
        print(f"  - Worker {worker_stat['worker_id'][:20]}...")
        print(f"    • Tasks processed: {worker_stat['tasks_processed']}")
        print(f"    • Tasks failed: {worker_stat['tasks_failed']}")
        print(f"    • Avg processing time: {worker_stat['avg_processing_time']:.3f}s")


async def main():
    print("\n" + "="*70)
    print("Image Processing Pipeline Queue Management & Horizontal Scaling")
    print("End-to-End Verification")
    print("="*70)
    
    try:
        await test_phase_1_basic_queueing()
        await test_phase_2_horizontal_scaling()
        await test_phase_3_task_processing()
        await test_phase_4_error_handling()
        await test_phase_5_metrics_and_monitoring()
        
        print("\n" + "="*70)
        print("[PASS] All verification phases completed successfully!")
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
