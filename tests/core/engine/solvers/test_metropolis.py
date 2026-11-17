"""Test the `_metropolis` module of `skcomplex.core.engine.solvers`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

from .test_objective import Well


def test_metropolis_samples_the_boltzmann_weight_of_a_well():
    """Test that the states of a well `(x - 2)^2` at temperature 0.5 have mean 2 and variance 0.25."""
    solver = MetropolisSolver(temperature=0.5, step=0.5)
    system = System(elements=[Element('ball', values={'x': 2.0})], interactions=[Well('well')], solver=solver)
    positions = system.prepare().simulate(until=20000, dt=1.0, random_state=0).get_values('ball.x')
    assert np.mean(positions) == pytest.approx(2.0, abs=0.05)
    assert np.var(positions) == pytest.approx(0.25, abs=0.03)


@pytest.mark.parametrize('setting', ['temperature', 'step'])
def test_metropolis_rejects_a_setting_that_is_not_positive(setting):
    """Test that a temperature or a step of zero raises an error naming it."""
    solver = MetropolisSolver(**{setting: 0.0})
    system = System(elements=[Element('ball', values={'x': 0.0})], interactions=[Well('well')], solver=solver)
    with pytest.raises(ValidationError, match=f'{setting} of MetropolisSolver must be positive'):
        system.prepare().simulate(until=1.0)
