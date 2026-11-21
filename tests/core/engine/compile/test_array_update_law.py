"""Test the `_array_update_law` module of `skcomplex.core.engine.compile`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components import MatrixGroupSpec
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import BaseSpace, CellComplexSpace, LatticeSpace
from skcomplex.core.engine.compile._array_update_law import ArrayUpdateLaw
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role

U1 = MatrixGroupSpec('U', 1)


def test_drive_targets_its_lattice_state():
    """Test that a drive targets the one lattice state it changes."""
    height = var('height')
    assert ArrayUpdateLaw(height, lambda values, **kwargs: values + 1)._get_targets() == (height,)


def test_drive_needs_a_callable_function():
    """Test that a drive rejects a function that cannot be called."""
    with pytest.raises(ValidationError, match='callable'):
        ArrayUpdateLaw(var('height'), 1.0)._check()


class ReplayInteraction(Interaction):
    """Drive that sets a chain to one member of an ensemble per step."""

    def __init__(self, label, *, target: Role, ensemble=None):
        super().__init__(label)
        self.target = target
        self.ensemble = ensemble

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the replay."""
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context.target.x, self._replay, arrays=True)]

    def _replay(self, values, *, time, **kwargs):
        """Return the member of the ensemble of the step, or the last member after the ensemble ends."""
        return self.ensemble[min(int(time), len(self.ensemble) - 1)]


def test_drive_replays_an_ensemble_by_the_time_of_the_step():
    """Test that positions at frame 10 are member 10, and positions at frame 40 are member 37 of 38 members."""
    ensemble = np.random.default_rng(0).normal(size=(38, 20, 3))
    chain = Element('chain', values={'x': ensemble[0]}, space=LatticeSpace((20, 3), boundary='fixed'))
    replay = ReplayInteraction('replay', target='chain', ensemble=ensemble)
    system = System(elements=[chain], interactions=[replay]).prepare()
    frames = system.simulate(until=40, random_state=0).get_lattice('chain.x')
    assert frames[10] == pytest.approx(ensemble[10])
    assert frames[40] == pytest.approx(ensemble[37])


@pytest.mark.parametrize('stochastic', [False, True])
def test_drive_receives_the_keywords_of_every_rule(stochastic):
    """Test that a drive receives time, space, degree, and random, the generator if stochastic."""
    seen = {}

    def drive(values, **keywords):
        seen.update(keywords)
        return values

    space = LatticeSpace((2, 3), boundary='fixed')
    chain = Element('chain', values={'x': 0.0}, space=space)
    interaction = _DriveInteraction('drive', target='chain', function=drive, stochastic=stochastic)
    System(elements=[chain], interactions=[interaction]).prepare().simulate(until=1, random_state=0)
    assert set(seen) == {'time', 'space', 'degree', 'random'}
    assert (seen['time'], seen['space'], seen['degree']) == (1.0, space, None)
    assert isinstance(seen['random'], np.random.Generator) if stochastic else seen['random'] is None


class _DriveInteraction(Interaction):
    """Interaction with one drive of a given function."""

    def __init__(self, label, *, target: Role, function=None, stochastic=False):
        super().__init__(label)
        self.target = target
        self.function = function
        self.stochastic = stochastic

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the drive."""
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context.target.x, self.function, arrays=True, stochastic=self.stochastic)]


class PairElement(Element):
    """Links with a matrix of U(1), two real components."""

    def __init__(
        self,
        label,
        *,
        U: Annotated[object, U1] = (1.0, 0.0),
        space: BaseSpace | None = None,
        on: str | None = None,
    ):
        super().__init__(label, space=space, on=on)
        self.U = U


class RotationInteraction(Interaction):
    """Drive that rotates the phase of every link by its number of faces."""

    def __init__(self, label, *, target: Role):
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rotation."""
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context.target.U, self._rotate, arrays=True)]

    def _rotate(self, values, *, space, **kwargs):
        """Return the components rotated by the number of faces of every link."""
        turns = np.exp(1j * (space.cofaces_[1] < space.boundaries_.shape[0]).sum(axis=1))
        return U1.get_components_of(turns[:, None, None] * U1.get_matrices_of(values))


def test_drive_of_a_matrix_state_receives_the_components_on_a_last_axis_and_the_space():
    """Test that a drive changes all components of every link, read with the space of the group."""
    space = CellComplexSpace(LatticeSpace((3, 3), boundary='fixed'))
    links = PairElement('links', U=np.tile((1.0, 0.0), (12, 1)), space=space, on='edges')
    system = System(elements=[links], interactions=[RotationInteraction('turn', target='links')]).prepare()
    values = system.simulate(until=1, random_state=0).get_values('links.U')
    faces = (space.cofaces_[1] < space.boundaries_.shape[0]).sum(axis=1)
    assert values.shape == (2, 12, 2)
    assert np.allclose(values[1, :, 0] + 1j * values[1, :, 1], np.exp(1j * faces))


class PartialInteraction(Interaction):
    """Drive that targets one component of a matrix state."""

    def __init__(self, label, *, target: Role):
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set a drive of the first component."""
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context.target.U[0], lambda values, **kwargs: values, arrays=True)]


def test_drive_of_one_component_of_a_matrix_state_raises_an_error():
    """Test that a drive that targets one of the components of a matrix state names the whole state."""
    links = PairElement(
        'links',
        U=np.tile((1.0, 0.0), (18, 1)),
        space=CellComplexSpace(LatticeSpace((3, 3))),
        on='edges',
    )
    system = System(elements=[links], interactions=[PartialInteraction('part', target='links')])
    with pytest.raises(ValidationError, match='Target the whole state'):
        system.prepare().simulate(until=0)
