"""Tests for Celery task error mapping — broker down, timeout, success."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import celery.exceptions as _celery_exc


@pytest.fixture
def mock_task():
    t = MagicMock()
    t.get = MagicMock()
    return t


class FakeCeleryTimeout(_celery_exc.TimeoutError):
    pass


# We test _run_celery_task logic via its internal exception branches.


def test_broker_down_returns_503():
    """ConnectionError / OSError during task.get → 503."""
    from main import _run_celery_task

    task = MagicMock()
    task.get.side_effect = ConnectionError("Broker unreachable")

    with pytest.raises(HTTPException) as exc_info:
        # synchronous call inside our async test is fine — we only
        # check the exception that _run_celery_task raises before
        # awaiting.
        #
        # We patch run_in_executor to raise the error synchronously.
        async def run():
            return await _run_celery_task(task, timeout=30)

        coro = run()
        # Drive the coroutine until it hits run_in_executor
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor.side_effect = ConnectionError("Broker down")
            try:
                coro.send(None)
            except StopIteration:
                pass
            except HTTPException as e:
                assert e.status_code == 503
                return
        pytest.fail("Expected HTTPException 503")


def test_successful_task():
    """Successful task.get returns the result."""
    from main import _run_celery_task

    task = MagicMock()
    task.get.return_value = {"predicted_ExpYield": 2500.0}

    async def run():
        with patch("asyncio.get_running_loop") as mock_loop:
            loop = MagicMock()
            loop.run_in_executor = AsyncMock(return_value={"predicted_ExpYield": 2500.0})
            mock_loop.return_value = loop
            return await _run_celery_task(task, timeout=30)

    import asyncio
    result = asyncio.run(run())
    assert result["predicted_ExpYield"] == 2500.0
