"""Test the `_regrowth` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import RegrowthInteraction
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec


class ResourceElement(Element):
    """Cell that holds a resource with a capacity and a clock."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        resource: Annotated[float, StateSpec()] = 0.0,
        capacity: Annotated[float, StateSpec()] = 1.0,
        clock: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.resource = resource
        self.capacity = capacity
        self.clock = clock


def _regrow(until: int, values: dict[str, object], **arguments: object) -> Trajectory:
    """Return the trajectory of two cells whose resource regrows."""
    cells = GroupElement('cells', element=ResourceElement, size=2, values=values)
    regrowth = RegrowthInteraction('regrowth', resources='cells', **arguments)
    system = System(elements=[cells], interactions=[regrowth])
    return Simulation(system, random_state=0).check().prepare().run(until=until).trajectory_


def _resources(trajectory: Trajectory, frame: int) -> list[float]:
    """Return the resources of the cells at a frame."""
    return trajectory.agents('cells', frame).values['resource'].tolist()


def test_regrow_restores_the_capacity_after_the_delay():
    """Test that a resource below its capacity is restored when its clock ends, and the clock restarts."""
    values = {'resource': [0.0, 1.0], 'clock': [3.0, 3.0]}
    trajectory = _regrow(3, values, form='delay', delay=5.0)
    assert _resources(trajectory, 2) == [0.0, 1.0]
    assert _resources(trajectory, 3) == [1.0, 1.0]
    assert trajectory.agents('cells', 3).values['clock'].tolist() == [5.0, 3.0]


def test_regrow_adds_the_rate_up_to_the_capacity():
    """Test that linear regrowth adds the rate every step and stops at the capacity of every cell."""
    values = {'resource': [0.0, 0.0], 'capacity': [1.0, 4.0]}
    trajectory = _regrow(3, values, form='linear', rate=1.5)
    assert _resources(trajectory, 1) == [1.0, 1.5]
    assert _resources(trajectory, 3) == [1.0, 4.0]


def test_regrow_grows_logistically_towards_the_capacity():
    """Test that logistic regrowth follows `r' = r + rate r (1 - r / K)` and approaches the capacity."""
    values = {'resource': [0.1, 0.0], 'capacity': [2.0, 2.0]}
    trajectory = _regrow(60, values, form='logistic', rate=0.5)
    expected = 0.1
    for _ in range(5):
        expected += 0.5 * expected * (1 - expected / 2.0)
    assert _resources(trajectory, 5)[0] == pytest.approx(expected)
    assert _resources(trajectory, 60) == pytest.approx([2.0, 0.0])
    assert np.all(np.diff([_resources(trajectory, frame)[0] for frame in range(61)]) >= 0)


@pytest.mark.parametrize('arguments', [{'form': 'cubic'}, {'resource': 1}, {'capacity': None}, {'clock': 2}])
def test_check_rejects_invalid_arguments(arguments):
    """Test that an unknown form and a state name that is not a string are rejected."""
    with pytest.raises(ValidationError):
        RegrowthInteraction('regrowth', resources='cells', **arguments).check()
