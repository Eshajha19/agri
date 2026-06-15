from ml.dp_privacy import (
    approximate_epsilon,
    noise_multiplier_for_target_epsilon,
    build_privacy_budget,
)


def test_epsilon_decreases_with_more_noise():
    low_noise_eps = approximate_epsilon(
        noise_multiplier=0.8,
        sample_rate=0.02,
        steps=1000,
        delta=1e-5,
    )
    high_noise_eps = approximate_epsilon(
        noise_multiplier=1.6,
        sample_rate=0.02,
        steps=1000,
        delta=1e-5,
    )

    assert high_noise_eps < low_noise_eps


def test_epsilon_increases_with_more_steps():
    short_run = approximate_epsilon(
        noise_multiplier=1.2,
        sample_rate=0.02,
        steps=500,
        delta=1e-5,
    )
    long_run = approximate_epsilon(
        noise_multiplier=1.2,
        sample_rate=0.02,
        steps=2000,
        delta=1e-5,
    )

    assert long_run > short_run


def test_target_epsilon_inverse_noise_estimate():
    target_epsilon = 3.0
    sample_rate = 0.01
    steps = 1500
    delta = 1e-5

    noise = noise_multiplier_for_target_epsilon(
        target_epsilon=target_epsilon,
        sample_rate=sample_rate,
        steps=steps,
        delta=delta,
    )

    recovered = approximate_epsilon(
        noise_multiplier=noise,
        sample_rate=sample_rate,
        steps=steps,
        delta=delta,
    )

    # Inverse should approximately recover the target in this closed-form model.
    assert abs(recovered - target_epsilon) < 1e-9


def test_build_privacy_budget_contains_expected_values():
    budget = build_privacy_budget(
        epsilon=2.5,
        delta=1e-6,
        sample_rate=0.015,
        steps=1200,
    )

    assert budget.epsilon == 2.5
    assert budget.delta == 1e-6
    assert budget.sample_rate == 0.015
    assert budget.steps == 1200
    assert budget.noise_multiplier > 0
