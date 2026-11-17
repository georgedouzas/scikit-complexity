"""Test the `_multilevel_selection` module of `skcomplex.systems.social.cooperation`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import ConservationInteraction
from skcomplex.core.components.laws.utils import coordinate, moment
from skcomplex.core.components.spaces import AxisSpec, FiniteVolumeSpec, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver, SSPRKSolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec
from skcomplex.systems.social.cooperation import (
    NonlinearPairwiseGroupConflictSystem,
    SimplexSpace,
    TrimorphicMultilevelSelectionSystem,
    TwoLevelReplicatorSystem,
)

SIZE = 12


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x


def _simplex(size: int = SIZE) -> SimplexSpace:
    """Return the 2-simplex of three frequencies at one position."""
    axes = [AxisSpec('x', size, 0.0, 1.0), AxisSpec('y', size, 0.0, 1.0)]
    return SimplexSpace(LatticeSpace((1,)), axes).prepare()


def replicator_drift(_: SimpleNamespace) -> list[object]:
    """Return the replicator dynamics of three strategies whose payoffs make the third dominate the first two."""
    x, y = coordinate('x'), coordinate('y')
    first, second = 1 - 2 * x - y, 2 - x - 3 * y
    return [0, x * ((1 - x) * first - y * second), y * ((1 - y) * second - x * first)]


def total(s: SimpleNamespace) -> object:
    """Return the integral of the density over the simplex."""
    return moment(s.x, 0, axes=('x', 'y'))


def test_simplex_space_cuts_cells_along_the_slanted_face() -> None:
    """Hold whole cells below the slanted face, half cells along it, and empty cells beyond it."""
    volumes = _simplex(10).volumes_.ravel()
    assert (volumes == 1.0).sum() == 45
    assert (volumes == 0.5).sum() == 10
    assert (volumes == 0.0).sum() == 45
    assert volumes.sum() / 100 == pytest.approx(0.5)


def test_simplex_space_closes_the_faces_beyond_the_simplex() -> None:
    """Open the faces between cells inside the simplex and close the faces to the empty cells."""
    space = _simplex(4)
    np.testing.assert_array_equal(space.apertures_[0], np.ones((1, 1, 1)))
    np.testing.assert_array_equal(space.apertures_[1][0, :, 0], [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(space.apertures_[2][0, 3, :], [1, 0, 0, 0, 0])


def test_simplex_space_holds_the_volume_of_a_three_dimensional_simplex() -> None:
    """Sum the cell volumes of three class axes to the volume one sixth of the 3-simplex."""
    axes = [AxisSpec(name, 6, 0.0, 1.0) for name in ('x', 'y', 'z')]
    space = SimplexSpace(LatticeSpace((1,)), axes).prepare()
    assert space.volumes_.sum() / 6**3 == pytest.approx(1 / 6)


@pytest.mark.parametrize(
    'axes',
    [
        [AxisSpec('x', 10, 0.0, 1.0), AxisSpec('y', 12, 0.0, 1.0)],
        [AxisSpec('x', 10, 0.0, 2.0), AxisSpec('y', 10, 0.0, 2.0)],
        [AxisSpec('x', 10, 0.0, 1.0, periodic=True), AxisSpec('y', 10, 0.0, 1.0)],
    ],
)
def test_simplex_space_rejects_axes_that_are_not_unit_intervals_of_one_size(axes: list[AxisSpec]) -> None:
    """Raise a named error for axes of different sizes, ranges other than `[0, 1]`, or periodic axes."""
    with pytest.raises(ValidationError, match=r'intervals on \[0, 1\]'):
        SimplexSpace(LatticeSpace((1,)), axes).prepare()


def test_moment_over_a_simplex_weights_every_cell_by_its_volume() -> None:
    """Integrate a uniform density of two over the 2-simplex to one."""
    space = _simplex()
    start = 2.0 * (space.volumes_ > 0)
    groups = FieldElement('groups', element=AmountElement, space=space, values={'x': start})
    observables = [Observable('total', total, group='groups')]
    trajectory = simulate(CompositeSystem(elements=[groups]), until=0.1, dt=0.1, observables=observables)
    values = np.asarray(trajectory.get_values('total'), dtype=float)
    assert np.abs(values - 1.0).max() < 1e-12


@pytest.mark.parametrize('solver', [RK4Solver(dt=0.01), SSPRKSolver()], ids=['rk4', 'ssprk'])
def test_replicator_flow_on_a_simplex_conserves_the_density(solver: object) -> None:
    """Keep the total of a uniform density on the simplex within 1e-10 and the empty cells at zero."""
    space = _simplex()
    start = 2.0 * (space.volumes_ > 0)
    groups = FieldElement(
        'groups',
        element=AmountElement,
        space=space,
        values={'x': start},
        discretization='finite-volume',
        scheme=FiniteVolumeSpec(limiter=None),
    )
    flow = ConservationInteraction(
        'flow',
        target='groups',
        state='x',
        flux=lambda s: [component * s.target.x for component in replicator_drift(s)],
    )
    trajectory = simulate(CompositeSystem(elements=[groups], interactions=[flow]), until=2.0, dt=0.5, solver=solver)
    final = trajectory.get_values('groups.x')[-1].reshape(space.shape_)
    weights = space.volumes_ / SIZE**2
    assert abs((final * weights).sum() - 1.0) < 1e-10
    assert np.all(final[space.volumes_ == 0] == 0.0)
    assert not np.allclose(final, start)


def _final_density(system, until, dt=0.02):
    """Return the initial and final density of a system of group compositions."""
    trajectory = simulate(system, until=until, dt=until, solver=RK4Solver(dt=dt))
    values = trajectory.get_values('groups.x')
    return values[0], values[-1]


@pytest.mark.parametrize('victory', ['fermi', 'local', 'tullock'])
def test_nonlinear_pairwise_group_conflict_keeps_the_total(victory):
    """Test that every victory probability keeps the integral of the density at one."""
    start, final = _final_density(
        NonlinearPairwiseGroupConflictSystem(40, victory=victory, exponent=2.0).prepare().system_,
        3.0,
    )
    assert abs(final.sum() / 40 - 1.0) < 1e-10
    assert not np.allclose(final, start)


def test_nonlinear_pairwise_group_conflict_strong_conflict_sustains_punishers():
    """Test that the mean fraction of punishers ends higher with strong than with weak between-group conflict."""
    centres = (np.arange(50) + 0.5) / 50
    means = {}
    for strength in (0.1, 2.0):
        _, final = _final_density(
            NonlinearPairwiseGroupConflictSystem(50, group_strength=strength).prepare().system_,
            10.0,
        )
        means[strength] = float((final * centres).sum() / 50)
    assert means[2.0] > 0.8
    assert means[0.1] < 0.1


@pytest.mark.parametrize('victory', ['defectors', 'payoff'])
def test_trimorphic_multilevel_selection_lowers_defection_and_keeps_the_total(victory):
    """Test that the mean fraction of defectors falls on the simplex while the integral stays one."""
    system = TrimorphicMultilevelSelectionSystem(16, victory=victory).prepare().system_
    space = system.elements[0].space.prepare()
    weights = space.volumes_.ravel() / 16**2
    defectors = np.broadcast_to(space.coordinates_['y'][None, None, :], space.shape_).ravel()
    start, final = _final_density(system, 5.0)
    assert abs((final * weights).sum() - 1.0) < 1e-10
    assert (final * weights * defectors).sum() < 0.8 * (start * weights * defectors).sum()


def test_multilevel_selection_rejects_an_unknown_victory_probability():
    """Test that an unknown victory probability raises a named error listing the known ones."""
    with pytest.raises(ValidationError, match='fermi'):
        NonlinearPairwiseGroupConflictSystem(victory='majority').prepare()
    with pytest.raises(ValidationError, match='defectors'):
        TrimorphicMultilevelSelectionSystem(victory='fermi').prepare()


@pytest.mark.parametrize(('strength', 'expected'), [(2.0, 0.75), (1.0, 0.5)])
def test_two_level_replicator_reaches_the_mean_fraction_of_punishers(strength, expected):
    """Test that above `lambda* = c + q - p` the mean fraction of punishers tends to `1 - (c + q - p) / lambda`."""
    _, final = _final_density(TwoLevelReplicatorSystem(group_strength=strength).prepare().system_, 20.0, dt=0.01)
    final = final.ravel()
    centres = (np.arange(final.size) + 0.5) / final.size
    assert np.sum(centres * final) / np.sum(final) == pytest.approx(expected, abs=0.02)
    assert np.mean(final) == pytest.approx(1.0, rel=1e-3)


def test_two_level_replicator_loses_the_punishers_below_the_threshold():
    """Test that groups concentrate at all defectors below `lambda*`, and at all punishers if `p > c + q`."""
    centres = (np.arange(100) + 0.5) / 100
    _, weak = _final_density(TwoLevelReplicatorSystem(group_strength=0.3).prepare().system_, 20.0, dt=0.01)
    assert np.sum(centres * weak.ravel()) / np.sum(weak) < 0.05
    _, strong = _final_density(TwoLevelReplicatorSystem(punishment=1.5).prepare().system_, 20.0, dt=0.01)
    assert np.sum(centres * strong.ravel()) / np.sum(strong) > 0.95


def test_two_level_replicator_rejects_a_non_additive_victory():
    """Test that only additively separable victory probabilities reduce to a two-level replicator equation."""
    with pytest.raises(ValidationError, match='Unknown victory'):
        TwoLevelReplicatorSystem(victory='fermi').prepare()
