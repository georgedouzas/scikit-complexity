"""Test the `_orders` module of `skcomplex.core.engine.compile.passes`."""

from typing import Annotated

import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import Parameter, State, StateSpec, VisualSpec
from skcomplex.core.systems import System


class ParticleElement(Element):
    """Particle whose position changes at the rate of its velocity."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(rate='v')] = 0.0,
        v: State[float] = 0.0,
        m: Parameter[float] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.m = m


class LostElement(Element):
    """Element whose state names a rate it does not have."""

    def __init__(self, label: str, visual: VisualSpec | None = None, *, x: Annotated[float, StateSpec(rate='w')] = 0.0):
        super().__init__(label, visual=visual)
        self.x = x


def test_intrinsic_laws_move_a_free_particle():
    """A particle with no interaction moves at its constant velocity."""
    trajectory = System(elements=[ParticleElement('ball', x=1.0, v=2.0)]).prepare().simulate(until=1.0, dt=0.5)
    assert trajectory.get_values('ball.x')[-1] == pytest.approx(3.0)
    assert trajectory.get_values('ball.v')[-1] == pytest.approx(2.0)


def test_intrinsic_laws_add_to_interaction_rates():
    """A falling particle follows the exact solution of free fall."""
    gravity = RateInteraction('gravity', target='ball', state='v', rate=-9.81)
    system = System(elements=[ParticleElement('ball', x=10.0)], interactions=[gravity]).prepare()
    trajectory = system.simulate(until=1.4, dt=0.01)
    exact = 10.0 - 9.81 * trajectory.times**2 / 2
    assert trajectory.get_values('ball.x') == pytest.approx(exact, abs=1e-9)


def test_intrinsic_laws_move_every_member():
    """Every member of a group moves at its own velocity."""
    gas = GroupElement('gas', element=ParticleElement, size=3, values={'x': [0.0, 1.0, 2.0], 'v': [1.0, 2.0, 3.0]})
    trajectory = System(elements=[gas]).prepare().simulate(until=1.0, dt=0.5)
    assert trajectory.get_values('gas.x')[-1].tolist() == pytest.approx([1.0, 3.0, 5.0])


def test_check_rate_rejects_an_unknown_rate():
    """A rate that names no other state raises an error that lists the states."""
    with pytest.raises(ValidationError, match="rate 'w', which must be another of its states: x"):
        System(elements=[LostElement('lost')]).prepare()
