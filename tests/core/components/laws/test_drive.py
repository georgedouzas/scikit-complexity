"""Test the `_drive` module of `skcomplex.core.components.laws`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import DriveLaw
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import RoleSpec, StateSpec


def test_drive_targets_its_lattice_state():
    """Test that a drive targets the one lattice state it changes."""
    height = var('height')
    assert DriveLaw(height, lambda values, *, random: values + 1).prepare().targets_ == (height,)


def test_drive_needs_a_callable_function():
    """Test that a drive rejects a function that cannot be called."""
    with pytest.raises(ValidationError, match='callable'):
        DriveLaw(var('height'), 1.0).prepare()


class ChainCoordinateElement(Element):
    """Coordinate of a point of a chain."""

    def __init__(self, label, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


class ReplayInteraction(Interaction):
    """Drive that sets a chain to one member of an ensemble per step."""

    def __init__(self, label, *, target: Annotated[str, RoleSpec()], ensemble=None):
        super().__init__(label)
        self.target = target
        self.ensemble = ensemble

    def _get_laws(self, namespace):
        """Return the replay."""
        last = len(self.ensemble) - 1
        return [
            DriveLaw(
                self._get_member(namespace, 'target').x,
                lambda values, *, time: self.ensemble[min(int(time), last)],
            ),
        ]


def test_drive_replays_an_ensemble_by_the_time_of_the_step():
    """Test that positions at frame 10 are member 10, and positions at frame 40 are member 37 of 38 members."""
    ensemble = np.random.default_rng(0).normal(size=(38, 20, 3))
    chain = LatticeElement(
        'chain',
        element=ChainCoordinateElement,
        space=LatticeSpace((20, 3), boundary='fixed'),
        values={'x': ensemble[0]},
    )
    replay = ReplayInteraction('replay', target='chain', ensemble=ensemble)
    system = System(elements=[chain], interactions=[replay], time=DiscreteTime())
    frames = Simulation(system, random_state=0).prepare().run(until=40).trajectory_.lattice('chain.x')
    assert frames[10] == pytest.approx(ensemble[10])
    assert frames[40] == pytest.approx(ensemble[37])


def test_drive_passes_time_only_to_a_function_that_names_it():
    """Test that a function of any keywords receives the generator and no time."""
    seen = {}

    def drive(values, **keywords):
        seen.update(keywords)
        return values

    chain = LatticeElement('chain', element=ChainCoordinateElement, space=LatticeSpace((2, 3), boundary='fixed'))
    interaction = _DriveInteraction('drive', target='chain', function=drive)
    system = System(elements=[chain], interactions=[interaction], time=DiscreteTime())
    Simulation(system, random_state=0).prepare().run(until=1)
    assert 'time' not in seen
    assert 'random' in seen


class _DriveInteraction(Interaction):
    """Interaction with one drive of a given function."""

    def __init__(self, label, *, target: Annotated[str, RoleSpec()], function=None):
        super().__init__(label)
        self.target = target
        self.function = function

    def _get_laws(self, namespace):
        """Return the drive."""
        return [DriveLaw(self._get_member(namespace, 'target').x, self.function)]
