"""Test the `_counting` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import pytest

from skcomplex.systems.mathematics.stochastic_processes import (
    BernoulliProcessSystem,
    PoissonProcessSystem,
)

from ..conftest import _run


def test_bernoulli_and_poisson_processes_count_at_their_rates():
    """Test that Bernoulli successes average `p n` and Poisson counts average and vary as `rate t`."""
    successes = _run(BernoulliProcessSystem(100, p=0.3), 100).get_agents('trials', 100).values['successes']
    assert successes.mean() == pytest.approx(30, rel=0.05)
    counts = _run(PoissonProcessSystem(500, rate=2.0), 20).get_values('counters.count')[-1]
    assert counts.mean() == pytest.approx(40, rel=0.05)
    assert counts.var() == pytest.approx(40, rel=0.2)
