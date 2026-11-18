"""Test reading the recorded values of a trajectory by name."""

from typing import Annotated

import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.experiments._replicate import _replicate
from skcomplex.core.specs import StateSpec
from skcomplex.core.systems import System
from skcomplex.systems.biology.ecology import LotkaVolterraSystem


def test_trajectory_reads_values_by_name():
    """Test that a trajectory returns the values of a state by name and is not indexed."""
    trajectory = LotkaVolterraSystem().prepare().system_.prepare().simulate(until=1.0, dt=0.1)
    values = trajectory.get_values('prey.x')
    assert values.shape[0] == trajectory.times.size
    with pytest.raises(TypeError):
        trajectory['prey.x']


def test_member_keeps_the_parameters_of_interactions():
    """Test that a member of a replicated system reads its own states and the shared parameters of interactions."""
    ensemble = _replicate(LotkaVolterraSystem().prepare().system_, 2, {'prey.x': [20.0, 40.0]})
    trajectory = simulate(ensemble, until=1.0, dt=0.1)
    member = trajectory.get_member(1, PassCompiler().prepare(LotkaVolterraSystem().prepare().system_).run().problem_)
    assert float(member.get_values('prey.x')[0]) == 40.0
    assert float(member.get_values('growth.r')[0]) == 1.1


class FallingElement(Element):
    """Element whose height is of order two."""

    def __init__(self, label: str, *, h: Annotated[float, StateSpec(order=2)] = 1.0):
        super().__init__(label)
        self.h = h


def _falling_trajectory() -> Trajectory:
    """Return the trajectory of a body falling for one second."""
    gravity = RateInteraction('gravity', target='body', state='h', rate=-9.81)
    system = System(elements=[FallingElement('body')], interactions=[gravity]).prepare()
    return system.simulate(until=1.0, dt=0.5, initial_state={"body.h'": 2.0})


def test_names_show_derivatives_with_primes():
    """The names of a trajectory show derivatives with primes."""
    assert _falling_trajectory().names[:2] == ['body.h', "body.h'"]


def test_get_values_reads_a_derivative_by_its_prime():
    """A derivative is read by its name with a prime, and the start takes the value given by that name."""
    trajectory = _falling_trajectory()
    assert trajectory.get_values("body.h'")[0] == pytest.approx(2.0)
    assert trajectory.get_values("body.h'")[-1] == pytest.approx(2.0 - 9.81)


def test_get_values_lists_names_with_primes_when_one_is_unknown():
    """An unknown name raises an error that lists the names with primes."""
    with pytest.raises(KeyError, match=r"body\.h'"):
        _falling_trajectory().get_values('body.v')
