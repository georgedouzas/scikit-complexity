"""Test the `_duopoly` module of `skcomplex.systems.economics.markets`."""

import numpy as np
import pytest

from skcomplex.systems.economics.markets import (
    CournotDuopolySystem,
)

from ..conftest import _agents, _run


def test_cournot_duopoly_converges_to_the_nash_equilibrium():
    """Test that best responses without errors converge to 50 units each at the price 500, and errors keep noise."""
    exact = _run(
        CournotDuopolySystem().prepare().system_,
        30,
        observables=CournotDuopolySystem().get_run_settings()['observables'],
    )
    assert exact.get_values('total')[-1] == pytest.approx(100.0, abs=1e-3)
    assert exact.get_values('price')[-1] == pytest.approx(500.0, abs=1e-2)
    assert np.allclose(_agents(exact, 'firms', 30)['output'], 50.0, atol=1e-3)
    noisy = _run(
        CournotDuopolySystem(error=0.1).prepare().system_,
        30,
        observables=CournotDuopolySystem().get_run_settings()['observables'],
    ).get_values('total')[10:]
    assert noisy.std() > 0.5
    assert np.all(np.abs(noisy - 100.0) < 25.0)
