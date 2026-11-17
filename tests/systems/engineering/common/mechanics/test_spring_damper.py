"""Test the `_spring_damper` module of `skcomplex.systems.engineering.common.mechanics`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.engineering.common.mechanics import SpringDamperInteraction


def test_prepare_keeps_the_momentum_and_damps_the_relative_motion():
    """Test that the spring and damper keep the momentum at 0 and damp the gap of the masses."""
    body = Element('body', values={'a': 1.0, 'b': 0.0, 'ma': 1.0, 'mb': 2.0})
    joint = SpringDamperInteraction('joint', target='body', states=('a', 'b'), masses=('ma', 'mb'), k=4.0, b=0.5)
    system = System(elements=[body], interactions=[joint], solver=ScipyODESolver(rtol=1e-9, atol=1e-11))
    trajectory = system.prepare().simulate(until=30.0, dt=1.0)
    momentum = trajectory.get_values("body.a'") + 2.0 * trajectory.get_values("body.b'")
    np.testing.assert_allclose(momentum, 0.0, atol=1e-7)
    gap = trajectory.get_values('body.a') - trajectory.get_values('body.b')
    assert abs(gap[-1]) < 0.01


@pytest.mark.parametrize('names', [{'states': ('a',)}, {'masses': ('m1', 'm2', 'm3')}])
def test_prepare_rejects_states_or_masses_that_are_not_two(names):
    """Test that `prepare` rejects a spring and damper on one state or with three masses."""
    with pytest.raises(ValidationError, match='Give two names'):
        SpringDamperInteraction('joint', target='body', **names).prepare()
