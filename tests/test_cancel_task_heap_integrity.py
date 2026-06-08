"""
Regression tests for: Queue Becomes Permanently Invalid After Task Cancellation
Bug: cancel_task() replaced the heap-backed list with a collections.deque,
     causing all subsequent heapq operations to fail permanently.

These tests verify:
1. _task_queue remains a list (not a deque) after cancel_task().
2. The heap invariant is preserved after cancellation.
3. heappush and heappop continue to work correctly after cancellation.
4. Multiple sequential cancellations do not corrupt the queue.
5. Remaining tasks are dequeued in correct priority order after cancellation.
"""

import heapq
import pytest

from image_processing_queue import ImageProcessingQueue, ImageProcessingTask, TaskPriority, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_id: str, priority: TaskPriority = TaskPriority.NORMAL) -> ImageProcessingTask:
    return ImageProcessingTask(
        task_id=task_id,
        image_data=b"fake-image-bytes",
        crop_type="wheat",
        processor_type="disease_detection",
        priority=priority,
    )


def _make_queue() -> ImageProcessingQueue:
    """Return a fresh queue with caching disabled."""
    return ImageProcessingQueue(enable_persistence=False)


# ---------------------------------------------------------------------------
# Test 1: _task_queue is still a list after cancellation
# ---------------------------------------------------------------------------

def test_queue_type_is_list_after_cancel():
    """
    After cancel_task(), _task_queue must remain a plain list so that
    heapq operations continue to work. Previously the code replaced the
    list with a deque, which broke all subsequent heap calls.
    """
    q = _make_queue()
    task = _make_task("task-type-check")
    q.enqueue(task)

    cancelled = q.cancel_task("task-type-check")

    assert cancelled is True
    assert isinstance(q._task_queue, list), (
        "_task_queue must remain a list after cancellation; "
        "replacing it with a deque breaks all heapq operations."
    )


# ---------------------------------------------------------------------------
# Test 2: heap invariant is preserved after cancellation
# ---------------------------------------------------------------------------

def test_heap_invariant_preserved_after_cancel():
    """
    After cancel_task(), heapq.heapify() must be called so that the
    remaining entries still satisfy the min-heap property.
    heapq.nsmallest with n == len(...) is the standard way to verify this.
    """
    q = _make_queue()
    q.enqueue(_make_task("t1", TaskPriority.LOW))
    q.enqueue(_make_task("t2", TaskPriority.NORMAL))
    q.enqueue(_make_task("t3", TaskPriority.HIGH))
    q.enqueue(_make_task("t4", TaskPriority.CRITICAL))

    q.cancel_task("t2")  # cancel the NORMAL priority task

    # heapq.nlargest / nsmallest requires a valid heap for efficiency
    # but we can directly verify the invariant with _siftup check:
    heap = q._task_queue
    for i in range(1, len(heap)):
        parent = (i - 1) // 2
        assert heap[parent] <= heap[i], (
            f"Heap invariant violated at index {i}: "
            f"parent[{parent}]={heap[parent]} > child[{i}]={heap[i]}"
        )


# ---------------------------------------------------------------------------
# Test 3: heappush works correctly after cancellation
# ---------------------------------------------------------------------------

def test_heappush_works_after_cancel():
    """
    Verify that a new task can be pushed onto the queue after a cancellation
    without raising a TypeError (which would happen if _task_queue were a deque).
    """
    q = _make_queue()
    q.enqueue(_make_task("t-push-1", TaskPriority.NORMAL))
    q.cancel_task("t-push-1")

    # This must not raise TypeError: 'deque' object does not support item assignment
    new_task = _make_task("t-push-2", TaskPriority.HIGH)
    try:
        q.enqueue(new_task)
    except TypeError as exc:
        pytest.fail(f"heappush raised TypeError after cancellation: {exc}")

    assert "t-push-2" in q._tasks_by_id


# ---------------------------------------------------------------------------
# Test 4: heappop (via dequeue) works correctly after cancellation
# ---------------------------------------------------------------------------

def test_dequeue_works_after_cancel():
    """
    Verify that dequeue() (which calls heapq.heappop internally) works
    correctly after a cancellation. A deque-corrupted queue would raise
    TypeError on heappop.
    """
    q = _make_queue()
    q.enqueue(_make_task("t-pop-1", TaskPriority.NORMAL))
    q.enqueue(_make_task("t-pop-2", TaskPriority.HIGH))

    q.cancel_task("t-pop-1")

    # Register a worker so dequeue() can proceed
    q.register_worker("worker-1")

    try:
        task = q.dequeue("worker-1")
    except TypeError as exc:
        pytest.fail(f"heappop raised TypeError after cancellation: {exc}")

    assert task is not None
    assert task.task_id == "t-pop-2"


# ---------------------------------------------------------------------------
# Test 5: multiple sequential cancellations do not corrupt queue
# ---------------------------------------------------------------------------

def test_multiple_cancellations_keep_queue_valid():
    """
    Cancel several tasks in sequence and verify the queue remains intact
    with correct heap ordering for the survivors.
    """
    q = _make_queue()
    ids = [f"t-multi-{i}" for i in range(6)]
    priorities = [
        TaskPriority.LOW,
        TaskPriority.NORMAL,
        TaskPriority.HIGH,
        TaskPriority.CRITICAL,
        TaskPriority.NORMAL,
        TaskPriority.LOW,
    ]
    for tid, pri in zip(ids, priorities):
        q.enqueue(_make_task(tid, pri))

    # Cancel three tasks spread across the heap
    q.cancel_task("t-multi-0")
    q.cancel_task("t-multi-2")
    q.cancel_task("t-multi-4")

    # Queue must still be a list
    assert isinstance(q._task_queue, list)

    # Heap invariant must still hold
    heap = q._task_queue
    for i in range(1, len(heap)):
        parent = (i - 1) // 2
        assert heap[parent] <= heap[i], (
            f"Heap corrupted after multiple cancellations at index {i}"
        )

    # Only the non-cancelled tasks remain
    remaining_ids = {entry[2].task_id for entry in heap}
    assert remaining_ids == {"t-multi-1", "t-multi-3", "t-multi-5"}


# ---------------------------------------------------------------------------
# Test 6: cancelled tasks are not returned by dequeue
# ---------------------------------------------------------------------------

def test_cancelled_task_not_dequeued():
    """
    Ensure a cancelled task is never handed to a worker by dequeue().
    """
    q = _make_queue()
    q.enqueue(_make_task("t-skip", TaskPriority.CRITICAL))
    q.enqueue(_make_task("t-keep", TaskPriority.HIGH))

    q.cancel_task("t-skip")
    q.register_worker("worker-x")

    task = q.dequeue("worker-x")
    assert task is not None
    assert task.task_id == "t-keep", (
        "Dequeue should skip cancelled tasks and return the next valid one."
    )
