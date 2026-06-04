"""
Differential privacy accounting helpers for DP-SGD proof-of-concept training.

These formulas use a conservative closed-form approximation suitable for
configuration and logging in research workflows. For strict production-grade
accounting, rely on framework-specific accountants (for example, Opacus RDP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacyBudget:
    """Container for privacy budget values used in training and reporting."""

    epsilon: float
    delta: float
    noise_multiplier: float
    sample_rate: float
    steps: int


def _validate_inputs(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> None:
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be > 0")
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("sample_rate must be in (0, 1]")
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")


def approximate_epsilon(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    """Approximate epsilon for Poisson-sampled DP-SGD.

    Uses the closed-form approximation:
      epsilon ~= q * sqrt(2 * T * ln(1 / delta)) / sigma
    where q is sample rate, T is optimizer steps, sigma is noise multiplier.
    """
    _validate_inputs(noise_multiplier, sample_rate, steps, delta)
    return (sample_rate * math.sqrt(2.0 * steps * math.log(1.0 / delta))) / noise_multiplier


def noise_multiplier_for_target_epsilon(
    *,
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    """Compute a noise multiplier that approximately satisfies target epsilon."""
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be > 0")
    _validate_inputs(1.0, sample_rate, steps, delta)
    return (sample_rate * math.sqrt(2.0 * steps * math.log(1.0 / delta))) / target_epsilon


def build_privacy_budget(
    *,
    epsilon: float,
    delta: float,
    sample_rate: float,
    steps: int,
) -> PrivacyBudget:
    """Build a complete budget record by deriving the needed noise multiplier."""
    noise_multiplier = noise_multiplier_for_target_epsilon(
        target_epsilon=epsilon,
        sample_rate=sample_rate,
        steps=steps,
        delta=delta,
    )
    return PrivacyBudget(
        epsilon=epsilon,
        delta=delta,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        steps=steps,
    )
