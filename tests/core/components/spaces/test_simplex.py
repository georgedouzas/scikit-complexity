"""Test simplex spaces, phase spaces whose class axes are the free frequencies of a simplex."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components.elements import DensityElement, FieldElement
from skcomplex.core.components.interactions import AdvectionInteraction
from skcomplex.core.components.laws.utils import coordinate, moment
from skcomplex.core.components.spaces import LatticeSpace, SimplexSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver, SSPRKSolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import AxisSpec, ObservableSpec

SIZE = 12


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
    return moment(s.f, 0, axes=('x', 'y'))


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
    groups = FieldElement('groups', element=DensityElement, space=space, values={'f': start})
    observables = [ObservableSpec('total', total, group='groups')]
    trajectory = simulate(System(elements=[groups]), until=0.1, dt=0.1, observables=observables)
    values = np.asarray(trajectory.values('total'), dtype=float)
    assert np.abs(values - 1.0).max() < 1e-12


@pytest.mark.parametrize('solver', [RK4Solver(dt=0.01), SSPRKSolver()], ids=['rk4', 'ssprk'])
def test_replicator_flow_on_a_simplex_conserves_the_density(solver: object) -> None:
    """Keep the total of a uniform density on the simplex within 1e-10 and the empty cells at zero."""
    space = _simplex()
    start = 2.0 * (space.volumes_ > 0)
    groups = FieldElement(
        'groups',
        element=DensityElement,
        space=space,
        values={'f': start},
        discretization='finite-volume',
    )
    flow = AdvectionInteraction('flow', target='groups', state='f', drift=replicator_drift, limiter=None)
    trajectory = simulate(System(elements=[groups], interactions=[flow]), until=2.0, dt=0.5, solver=solver)
    final = trajectory.values('groups.f')[-1].reshape(space.shape_)
    weights = space.volumes_ / SIZE**2
    assert abs((final * weights).sum() - 1.0) < 1e-10
    assert np.all(final[space.volumes_ == 0] == 0.0)
    assert not np.allclose(final, start)
