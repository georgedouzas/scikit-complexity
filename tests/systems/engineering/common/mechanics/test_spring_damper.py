"""Test the `_spring_damper` module of `skcomplex.systems.engineering.common.mechanics`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.systems.engineering.common.mechanics import SpringDamperInteraction


class TwoMassElement(Element):
    """Two masses with their own coordinates, joined by a spring and a damper."""

    def __init__(
        self,
        label: str,
        *,
        a: Annotated[object, StateSpec(order=2)] = 1.0,
        a_d1: object = 0.0,
        b: Annotated[object, StateSpec(order=2)] = 0.0,
        b_d1: object = 0.0,
        k: Annotated[float, ParameterSpec()] = 4.0,
        c: Annotated[float, ParameterSpec()] = 0.5,
        ma: Annotated[float, ParameterSpec()] = 1.0,
        mb: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label)
        self.a = a
        self.a_d1 = a_d1
        self.b = b
        self.b_d1 = b_d1
        self.k = k
        self.c = c
        self.ma = ma
        self.mb = mb


def test_run_keeps_the_momentum_and_damps_the_relative_motion():
    """Test that the spring and damper keep the momentum at 0 and damp the gap of the masses."""
    joint = SpringDamperInteraction(
        'joint',
        target='body',
        states=('a', 'b'),
        stiffness='k',
        damping='c',
        divisor={'a': 'ma', 'b': 'mb'},
    )
    system = CompositeSystem(elements=[TwoMassElement('body')], interactions=[joint])
    simulation = simulate(system, until=30.0, solver=ScipyODESolver(rtol=1e-9, atol=1e-11), dt=1.0)
    trajectory = simulation
    momentum = trajectory.values('body.a_d1') + 2.0 * trajectory.values('body.b_d1')
    np.testing.assert_allclose(momentum, 0.0, atol=1e-7)
    gap = trajectory.values('body.a') - trajectory.values('body.b')
    assert abs(gap[-1]) < 0.01


def test_check_rejects_states_that_are_not_two():
    """Test that `check` rejects a spring and damper on one state."""
    with pytest.raises(ValidationError, match='two states'):
        SpringDamperInteraction('joint', target='body', states=('a',)).prepare()
