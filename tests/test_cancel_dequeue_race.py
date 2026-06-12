"""
Regression tests for: Task Cancellation Race Allows Cancelled Tasks to
Continue Processing in image_processing_queue.py

Root cause:
    dequeue() held only _queue_lock when transitioning task.status to
    PROCESSING.  cancel_task() held only _task_lock when checking/setting
    task.status.  Because these are two *different* locks, a window existed
    where cancel_task() could see the task as QUEUED and mark it CANCELLED
    *while* dequeue() was simultaneously popping and promoting it to PROCESSING.
    The result: the worker received a task whose internal state was CANCELLED,
    producing inconsistent lifecycle tracking.

Fix:
    dequeue() now acquires _task_lock (outer) then _queue_lock (inner) —
    the same ordering as cancel_task() — before checking and mutating
    task.status.  The status transition to PROCESSING is therefore atomic
    with respect to cancel_task(), eliminating the race.

Tests:
    1. Sequential: cancel before dequeue returns None.
    2. Sequential: dequeue before cancel returns False.
    3. Sequential: dequeued task is never in CANCELLED state.
    4. Concurrent (threaded): cancelled tasks never reach a worker.
    5. Concurrent (threaded): task status is never simultaneously PROCESSING
       and CANCELLED on the same object.
"""

import threading
import time
import pytest

from image_processing_queue import (
    ImageProcessingQueue,
    ImageProcessingTask,
    TaskPriority,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_id: str, priority: TaskPriority = TaskPriority.NORMAL) -> ImageProcessingTask:
    return ImageProcessingTask(
        task_id=task_id,
        image_data=b"img",
        crop_type="rice",
        processor_type="disease_detection",
        priority=priority,
    )


def _make_queue() -> ImageProcessingQueue:
    return ImageProcessingQueue(enable_persistence=False)


# ---------------------------------------------------------------------------
# Test 1: cancel before dequeue — worker gets nothing
# ---------------------------------------------------------------------------

def test_cancel_before_dequeue_returns_none():
    """
    If cancel_task() completes before dequeue() is called, the task must
    not be returned to any worker.
    """
    q = _make_queue()
    q.register_worker("w1")
    task = _make_task("t-cancel-first")
    q.enqueue(task)

    cancelled = q.cancel_task("t-cancel-first")
    assert cancelled is True

    result = q.dequeue("w1")
    assert result is None, "Cancelled task must not be returned by dequeue()."


# ---------------------------------------------------------------------------
# Test 2: dequeue before cancel — cancel_task returns False
# ---------------------------------------------------------------------------

def test_dequeue_before_cancel_returns_false():
    """
    Once dequeue() has promoted a task to PROCESSING, cancel_task() must
    return False — it is too late to cancel.
    """
    q = _make_queue()
    q.register_worker("w1")
    task = _make_task("t-dequeue-first")
    q.enqueue(task)

    result = q.dequeue("w1")
    assert result is not None
    assert result.status == TaskStatus.PROCESSING

    cancelled = q.cancel_task("t-dequeue-first")
    assert cancelled is False, (
        "cancel_task() must return False for a task already in PROCESSING state."
    )


# ---------------------------------------------------------------------------
# Test 3: dequeued task is never CANCELLED
# ---------------------------------------------------------------------------

def test_dequeued_task_status_is_processing():
    """
    A task returned by dequeue() must have status == PROCESSING, never
    CANCELLED.  The race condition allowed CANCELLED tasks to slip through.
    """
    q = _make_queue()
    q.register_worker("w1")
    task = _make_task("t-status-check")
    q.enqueue(task)

    result = q.dequeue("w1")
    assert result is not None
    assert result.status == TaskStatus.PROCESSING, (
        f"Dequeued task must be PROCESSING, got {result.status}"
    )
    assert result.status != TaskStatus.CANCELLED


# ---------------------------------------------------------------------------
# Test 4: concurrent threads — cancelled tasks never reach a worker
# ---------------------------------------------------------------------------

def test_concurrent_cancel_and_dequeue_no_cancelled_task_reaches_worker():
    """
    Simulate the race: one thread cancels, another dequeues, both at the
    same time.  Over many repetitions, a cancelled task must *never* be
    handed to a worker.
    """
    violations = []

    for i in range(200):
        q = _make_queue()
        q.register_worker("w1")
        task = _make_task(f"t-race-{i}")
        q.enqueue(task)

        dequeued_tasks = []
        barrier = threading.Barrier(2)

        def do_dequeue():
            barrier.wait()
            t = q.dequeue("w1")
            if t is not None:
                dequeued_tasks.append(t)

        def do_cancel():
            barrier.wait()
            q.cancel_task(f"t-race-{i}")

        t1 = threading.Thread(target=do_dequeue)
        t2 = threading.Thread(target=do_cancel)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        for dt in dequeued_tasks:
            if dt.status == TaskStatus.CANCELLED:
                violations.append(
                    f"Iteration {i}: dequeued task has status CANCELLED"
                )

    assert not violations, (
        f"Race condition detected in {len(violations)} iteration(s):\n"
        + "\n".join(violations[:5])
    )


# ---------------------------------------------------------------------------
# Test 5: concurrent threads — task is never both PROCESSING and CANCELLED
# ---------------------------------------------------------------------------

def test_concurrent_status_is_never_both_processing_and_cancelled():
    """
    After both threads finish, the task's final status must be exclusively
    PROCESSING (dequeue won) or CANCELLED (cancel won) — never an impossible
    intermediate that switches between the two.
    """
    for i in range(200):
        q = _make_queue()
        q.register_worker("w1")
        task = _make_task(f"t-atomic-{i}")
        q.enqueue(task)

        barrier = threading.Barrier(2)
        results = {}

        def do_dequeue():
            barrier.wait()
            results["dequeued"] = q.dequeue("w1")

        def do_cancel():
            barrier.wait()
            results["cancelled"] = q.cancel_task(f"t-atomic-{i}")

        t1 = threading.Thread(target=do_dequeue)
        t2 = threading.Thread(target=do_cancel)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        dequeued = results.get("dequeued")
        cancelled = results.get("cancelled")

        # Exactly one of the two must have "won"
        if dequeued is not None:
            # dequeue won: cancel must have failed
            assert not cancelled, (
                f"Iteration {i}: task was both dequeued AND cancelled — "
                "race condition still present."
            )
            assert dequeued.status == TaskStatus.PROCESSING
        else:
            # cancel won: task must not have reached a worker
            assert cancelled is True
