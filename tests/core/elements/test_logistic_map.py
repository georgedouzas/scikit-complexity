"""Test the `_logistic_map` module of `skcomplex.core.elements`."""

import numpy as np
import pytest

from skcomplex.core.elements import LogisticMapElement
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


def _orbit(r, steps=2000):
    """Return the orbit of the logistic map from 0.2 at a growth parameter."""
    system = System(elements=[LogisticMapElement('map', x=0.2, r=r)])
    return np.asarray(Simulation(system).check().prepare().run(until=steps).trajectory_.values('map.x'))


def test_logistic_map_iterates_its_equation():
    """Test that the map iterates `x -> r x (1 - x)`."""
    orbit = _orbit(3.7, steps=10)
    np.testing.assert_allclose(orbit[1:], 3.7 * orbit[:-1] * (1 - orbit[:-1]))


def test_logistic_map_converges_to_its_fixed_point_below_three():
    """Test that below `r = 3` the map converges to the fixed point `1 - 1 / r`."""
    assert _orbit(2.9)[-1] == pytest.approx(1 - 1 / 2.9, abs=1e-6)


def test_logistic_map_doubles_its_period_above_three():
    """Test that above `r = 3` the map settles on the cycle of period 2 of the closed form."""
    r = 3.2
    orbit = _orbit(r)
    root = np.sqrt((r - 3) * (r + 1))
    expected = sorted([(r + 1 - root) / (2 * r), (r + 1 + root) / (2 * r)])
    np.testing.assert_allclose(sorted(orbit[-2:]), expected, atol=1e-9)
    assert orbit[-1] == pytest.approx(orbit[-3], abs=1e-9)
