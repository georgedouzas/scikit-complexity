"""Test the `_counting` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import pytest

from skcomplex.systems.mathematics.stochastic_processes import (
    bernoulli_process,
    poisson_process,
)

from ..conftest import _run


def test_bernoulli_and_poisson_processes_count_at_their_rates():
    """Test that Bernoulli successes average `p n` and Poisson counts average and vary as `rate t`."""
    successes = _run(bernoulli_process(100, p=0.3), 100).agents('trials', 100).values['successes']
    assert successes.mean() == pytest.approx(30, rel=0.05)
    counts = _run(poisson_process(500, rate=2.0), 20).values('counters.count')[-1]
    assert counts.mean() == pytest.approx(40, rel=0.05)
    assert counts.var() == pytest.approx(40, rel=0.2)
