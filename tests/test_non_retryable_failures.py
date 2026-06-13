"""
Regression tests for: Worker Retries Non-Retryable Failures Unnecessarily
in image_processing_queue.py

Root cause:
    _process_task() called self.queue.fail_task(..., retry=True) for *every*
    exception, including permanent failures such as ValueError (malformed
    image), TypeError (bad argument), and NotImplementedError.  This caused
    deterministic failures to burn through all max_retries slots before
    finally being marked FAILED, wasting worker capacity and delaying
    legitimate work.

Fix:
    Introduced NON_RETRYABLE_ERRORS (a tuple of permanent exception types)
    and _is_retryable() helper.  _process_task() now passes
    retry=_is_retryable(exc) so that permanent errors are immediately
    marked FAILED without consuming any retry slots.

Tests:
    1. Permanent error (ValueError) → task FAILED immediately, retry_count == 1.
    2. Permanent error (TypeError) → no retry.
    3. Permanent error (NotImplementedError) → no retry.
    4. Transient error (IOError) → task is requeued for retry.
    5. Permanent error does not consume retry budget of other tasks.
    6. _is_retryable() returns correct values for all classified types.
"""

import asyncio
import pytest

from image_processing_queue import (
    ImageProcessingQueue,
    ImageProcessingTask,
    ImageProcessingWorker,
    TaskPriority,
    TaskStatus,
    NON_RETRYABLE_ERRORS,
    _is_retryable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_id: str, max_retries: int = 3) -> ImageProcessingTask:
    t = ImageProcessingTask(
        task_id=task_id,
        image_data=b"img",
        crop_type="wheat",
        processor_type="disease_detection",
        priority=TaskPriority.NORMAL,
    )
    t.max_retries = max_retries
    return t


def _make_queue() -> ImageProcessingQueue:
    return ImageProcessingQueue(enable_persistence=False)


async def _run_one_task(queue: ImageProcessingQueue, task: ImageProcessingTask, processor_fn):
    """Enqueue, dequeue, and process exactly one task via ImageProcessingWorker."""
    queue.register_worker("w1")
    queue.enqueue(task)
    worker = ImageProcessingWorker(queue, "w1", processor_fn)
    dequeued = queue.dequeue("w1")
    assert dequeued is not None
    await worker._process_task(dequeued)


# ---------------------------------------------------------------------------
# Test 1: ValueError → immediately FAILED, retry_count == 1
# ---------------------------------------------------------------------------

def test_permanent_value_error_fails_immediately():
    """
    A ValueError (e.g. malformed image bytes) is a permanent failure.
    The task must be marked FAILED after the first attempt without retrying.
    """
    q = _make_queue()
    task = _make_task("t-val-err", max_retries=3)

    async def bad_processor(t):
        raise ValueError("Invalid image format: cannot decode bytes")

    asyncio.run(_run_one_task(q, task, bad_processor))

    status = q.get_task_status("t-val-err")
    assert status is not None
    assert status["status"] == TaskStatus.FAILED.value, (
        "ValueError must cause immediate FAILED status, not retry."
    )
    # retry_count should be 1 (the initial attempt), not > 1
    failed_task = q._completed_tasks.get("t-val-err")
    assert failed_task is not None
    assert failed_task.retry_count == 1, (
        f"Expected retry_count=1 (no retries), got {failed_task.retry_count}"
    )


# ---------------------------------------------------------------------------
# Test 2: TypeError → immediately FAILED
# ---------------------------------------------------------------------------

def test_permanent_type_error_fails_immediately():
    """TypeError (programming error / wrong arg type) must not be retried."""
    q = _make_queue()
    task = _make_task("t-type-err", max_retries=3)

    async def bad_processor(t):
        raise TypeError("processor_type must be str, got int")

    asyncio.run(_run_one_task(q, task, bad_processor))

    status = q.get_task_status("t-type-err")
    assert status["status"] == TaskStatus.FAILED.value


# ---------------------------------------------------------------------------
# Test 3: NotImplementedError → immediately FAILED
# ---------------------------------------------------------------------------

def test_permanent_not_implemented_fails_immediately():
    """Unimplemented processor paths must not be retried."""
    q = _make_queue()
    task = _make_task("t-not-impl", max_retries=3)

    async def bad_processor(t):
        raise NotImplementedError("Processor 'legacy_v1' is not supported")

    asyncio.run(_run_one_task(q, task, bad_processor))

    status = q.get_task_status("t-not-impl")
    assert status["status"] == TaskStatus.FAILED.value


# ---------------------------------------------------------------------------
# Test 4: IOError → task is requeued for retry (transient)
# ---------------------------------------------------------------------------

def test_transient_io_error_allows_retry():
    """
    IOError is a transient failure (network / disk issue).
    The task must be requeued for retry, not immediately failed.
    """
    q = _make_queue()
    task = _make_task("t-io-err", max_retries=3)

    async def flaky_processor(t):
        raise IOError("Upstream model service unavailable")

    asyncio.run(_run_one_task(q, task, flaky_processor))

    # Task should still be in active tasks (requeued as RETRYING)
    active = q._tasks_by_id.get("t-io-err")
    assert active is not None, "Transient IOError must requeue the task, not fail it."
    assert active.status == TaskStatus.RETRYING
    assert active.retry_count == 1


# ---------------------------------------------------------------------------
# Test 5: Transient TimeoutError → requeued for retry
# ---------------------------------------------------------------------------

def test_transient_timeout_error_allows_retry():
    """TimeoutError (model inference timeout) is transient and must be retried."""
    q = _make_queue()
    task = _make_task("t-timeout", max_retries=3)

    async def slow_processor(t):
        raise TimeoutError("Model inference timed out after 30s")

    asyncio.run(_run_one_task(q, task, slow_processor))

    active = q._tasks_by_id.get("t-timeout")
    assert active is not None
    assert active.status == TaskStatus.RETRYING


# ---------------------------------------------------------------------------
# Test 6: _is_retryable() classification accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc, expected_retryable", [
    (ValueError("bad data"),           False),
    (TypeError("wrong type"),          False),
    (NotImplementedError("not impl"),  False),
    (MemoryError("out of memory"),     False),
    (KeyError("missing_key"),          False),
    (AttributeError("no attr"),        False),
    (IOError("disk error"),            True),
    (TimeoutError("timeout"),          True),
    (ConnectionError("conn refused"),  True),
    (RuntimeError("transient"),        True),
    (Exception("generic"),             True),
])
def test_is_retryable_classification(exc, expected_retryable):
    result = _is_retryable(exc)
    assert result == expected_retryable, (
        f"_is_retryable({type(exc).__name__}) returned {result}, "
        f"expected {expected_retryable}"
    )


# ---------------------------------------------------------------------------
# Test 7: permanent error does not consume retry budget of sibling tasks
# ---------------------------------------------------------------------------

def test_permanent_error_does_not_affect_other_tasks():
    """
    When one task fails permanently, other tasks in the queue must be
    unaffected — their retry budgets must remain intact.
    """
    q = _make_queue()
    q.register_worker("w1")

    task_bad = _make_task("t-perm-bad", max_retries=3)
    task_good = _make_task("t-perm-good", max_retries=3)

    q.enqueue(task_bad)
    q.enqueue(task_good)

    async def bad_processor(t):
        raise ValueError("malformed image")

    async def run_bad():
        worker = ImageProcessingWorker(q, "w1", bad_processor)
        dequeued = q.dequeue("w1")
        await worker._process_task(dequeued)

    asyncio.run(run_bad())

    # t-perm-bad must be permanently failed
    bad_status = q.get_task_status("t-perm-bad")
    assert bad_status["status"] == TaskStatus.FAILED.value

    # t-perm-good must still be queued and untouched
    good_task = q._tasks_by_id.get("t-perm-good")
    assert good_task is not None
    assert good_task.status == TaskStatus.QUEUED
    assert good_task.retry_count == 0
