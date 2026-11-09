"""Test the `_array_update` module of `skcomplex.core.components.laws`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ArrayUpdateLaw
from skcomplex.core.components.spaces import CellComplexSpace, DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import MatrixGroupSpec, RoleSpec, StateSpec


def test_drive_targets_its_lattice_state():
    """Test that a drive targets the one lattice state it changes."""
    height = var('height')
    assert ArrayUpdateLaw(height, lambda values, *, random: values + 1).prepare().targets_ == (height,)


def test_drive_needs_a_callable_function():
    """Test that a drive rejects a function that cannot be called."""
    with pytest.raises(ValidationError, match='callable'):
        ArrayUpdateLaw(var('height'), 1.0).prepare()


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
            ArrayUpdateLaw(
                self._get_member(namespace, 'target').x,
                lambda values, *, time: self.ensemble[min(int(time), last)],
            ),
        ]


def test_drive_replays_an_ensemble_by_the_time_of_the_step():
    """Test that positions at frame 10 are member 10, and positions at frame 40 are member 37 of 38 members."""
    ensemble = np.random.default_rng(0).normal(size=(38, 20, 3))
    chain = GroupElement(
        'chain',
        element=ChainCoordinateElement,
        space=LatticeSpace((20, 3), boundary='fixed'),
        values={'x': ensemble[0]},
        on='sites',
    )
    replay = ReplayInteraction('replay', target='chain', ensemble=ensemble)
    system = CompositeSystem(elements=[chain], interactions=[replay], time=DiscreteTime())
    frames = simulate(system, until=40, random_state=0).lattice('chain.x')
    assert frames[10] == pytest.approx(ensemble[10])
    assert frames[40] == pytest.approx(ensemble[37])


def test_drive_passes_time_only_to_a_function_that_names_it():
    """Test that a function of any keywords receives the generator and no time."""
    seen = {}

    def drive(values, **keywords):
        seen.update(keywords)
        return values

    chain = GroupElement(
        'chain',
        element=ChainCoordinateElement,
        space=LatticeSpace((2, 3), boundary='fixed'),
        on='sites',
    )
    interaction = _DriveInteraction('drive', target='chain', function=drive)
    system = CompositeSystem(elements=[chain], interactions=[interaction], time=DiscreteTime())
    simulate(system, until=1, random_state=0)
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
        return [ArrayUpdateLaw(self._get_member(namespace, 'target').x, self.function)]


class PairElement(Element):
    """Link with a matrix of U(1), two real components."""

    def __init__(self, label, U: Annotated[object, StateSpec(group=MatrixGroupSpec('U', 1))] = (1.0, 0.0)):
        super().__init__(label)
        self.U = U


class RotationInteraction(Interaction):
    """Drive that rotates the phase of every link by its number of faces."""

    def __init__(self, label, *, target: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.target = target

    def _get_laws(self, namespace):
        """Return the rotation."""
        return [ArrayUpdateLaw(self._get_member(namespace, 'target').U, self._rotate)]

    def _rotate(self, values, *, space):
        """Return the components rotated by the number of faces of every link."""
        group = MatrixGroupSpec('U', 1)
        turns = np.exp(1j * (space.cofaces_[1] < space.boundaries_.shape[0]).sum(axis=1))
        return group.components_of(turns[:, None, None] * group.matrices_of(values))


def test_drive_of_a_matrix_state_receives_the_components_on_a_last_axis_and_the_space():
    """Test that a drive changes all components of every link, read with the space of the group."""
    space = CellComplexSpace(LatticeSpace((3, 3), boundary='fixed'))
    links = GroupElement('links', element=PairElement, space=space, on='edges')
    system = CompositeSystem(
        elements=[links],
        interactions=[RotationInteraction('turn', target='links')],
        time=DiscreteTime(),
    )
    values = simulate(system, until=1, random_state=0).values('links.U')
    faces = (space.cofaces_[1] < space.boundaries_.shape[0]).sum(axis=1)
    assert values.shape == (2, 12, 2)
    assert np.allclose(values[1, :, 0] + 1j * values[1, :, 1], np.exp(1j * faces))


class PartialInteraction(Interaction):
    """Drive that targets one component of a matrix state."""

    def __init__(self, label, *, target: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.target = target

    def _get_laws(self, namespace):
        """Return a drive of the first component."""
        return [ArrayUpdateLaw(self._get_member(namespace, 'target').U[0], lambda values, *, random: values)]


def test_drive_of_one_component_of_a_matrix_state_raises_an_error():
    """Test that a drive that targets one of the components of a matrix state names the whole state."""
    space = CellComplexSpace(LatticeSpace((3, 3)))
    links = GroupElement('links', element=PairElement, space=space, on='edges')
    system = CompositeSystem(
        elements=[links],
        interactions=[PartialInteraction('part', target='links')],
        time=DiscreteTime(),
    )
    with pytest.raises(ValidationError, match='Target the whole state'):
        simulate(system, until=0)
