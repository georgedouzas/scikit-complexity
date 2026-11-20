"""Test phase spaces, the product of a position lattice and class axes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import coordinate, divergence
from skcomplex.core.components.spaces import (
    AxisSpec,
    BaseSpace,
    FiniteVolumeSpec,
    LatticeNeighborhood,
    LatticeSpace,
    PhaseSpace,
)
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import SpaceTooLargeError, ValidationError
from skcomplex.core.systems import System


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount, at every site of its space."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x


def _transport(gas, flux, until, dt):
    """Return the run of a field whose amount `x` changes by minus the divergence of a flux of its namespace."""
    drift = Interaction('drift', roles={'target': 'gas'}, laws=lambda s: [RateLaw(s.target.x, -divergence(flux(s)))])
    system = System(elements=[gas], interactions=[drift], solver=RK4Solver(dt=dt))
    return system.prepare().simulate(until=until)


def test_prepare_holds_one_value_per_point_of_the_product() -> None:
    """Hold 8192 values for 64 positions and 128 class points, with initial values over position and class."""
    space = PhaseSpace(LatticeSpace((64,)), [AxisSpec('v', 128, -6.0, 6.0)]).prepare()
    values = np.ones((64, 128))
    values[3, 5] = 2.0
    still = Interaction('still', roles={'target': 'gas'}, laws=lambda s: [RateLaw(s.target.x, 0)])
    system = System(elements=[AmountElement('gas', x=values, space=space)], interactions=[still]).prepare()
    assert system._problem.states.size == 8192
    assert space.class_axes_ == {'v': 1}
    assert space.position_axes_ == (0,)
    assert space.spacings_ == (1.0, 12.0 / 128)
    problem = system.simulate(until=0, dt=0.1).problem
    assert problem.initial_state[3 * 128 + 5] == 2.0


def test_prepare_places_coordinates_at_cell_centres() -> None:
    """Place the coordinates of an axis at the centres of its cells."""
    space = PhaseSpace(LatticeSpace(2), [AxisSpec('v', 4, 0.0, 2.0)]).prepare()
    assert space.coordinates_['v'].tolist() == [0.25, 0.75, 1.25, 1.75]


def test_check_rejects_position_or_axes_it_cannot_take() -> None:
    """Reject a triangular position, no axes, and axes that share a name."""
    axis = AxisSpec('v', 4, 0.0, 2.0)
    with pytest.raises(ValidationError, match='square lattice'):
        PhaseSpace(LatticeSpace((2, 2), geometry='triangular'), [axis]).prepare()
    with pytest.raises(ValidationError, match='one or more class axes'):
        PhaseSpace(LatticeSpace(2), []).prepare()
    with pytest.raises(ValidationError, match='distinct names'):
        PhaseSpace(LatticeSpace(2), [axis, axis]).prepare()


def test_prepare_raises_above_limit() -> None:
    """Raise a named error that holds the size the space would take above its limit."""
    space = PhaseSpace(LatticeSpace((1000,)), [AxisSpec('v', 1000, -1.0, 1.0)], limit=100)
    with pytest.raises(SpaceTooLargeError, match='1000000 sites'):
        space.prepare()


def test_check_reports_space_size_issue_with_size() -> None:
    """Report a phase space above its limit through the compile report, with the size it would take."""
    space = PhaseSpace(LatticeSpace((10,)), [AxisSpec('v', 10, -1.0, 1.0)], limit=50)
    report = PassCompiler().prepare(System(elements=[AmountElement('gas', space=space)])).report_
    issues = [issue for issue in report.issues if issue.code == 'space-size']
    assert issues
    assert '100 sites' in issues[0].message


def test_index_table_keeps_edges_per_axis() -> None:
    """Join a periodic position axis and reflect at the ends of an interval class axis."""
    space = PhaseSpace(LatticeSpace(3), [AxisSpec('v', 2, 0.0, 1.0)]).prepare()
    ahead = space._get_index_table(LatticeNeighborhood(((1, 0),)))[:, 0]
    above = space._get_index_table(LatticeNeighborhood(((0, 1),)))[:, 0]
    assert ahead.tolist() == [2, 3, 4, 5, 0, 1]
    assert above.tolist() == [1, 1, 3, 3, 5, 5]


@pytest.mark.parametrize(
    ('discretization', 'options'),
    [
        ('finite-volume', {'scheme': FiniteVolumeSpec(limiter='van-leer')}),
        ('finite-difference', {}),
        ('spectral', {}),
    ],
)
def test_periodic_class_axis_returns_after_one_turn(discretization: str, options: dict[str, object]) -> None:
    """Return a profile on a periodic axis of 32 angles to its start after one turn at constant drift."""
    axis = AxisSpec('theta', 32, 0.0, 2 * np.pi, periodic=True)
    space = PhaseSpace(LatticeSpace((1,), discretization=discretization, **options), [axis]).prepare()
    start = 1.0 + 0.5 * np.cos(space.coordinates_['theta'])
    gas = AmountElement('gas', x=start, space=space)
    trajectory = _transport(gas, lambda s: [0, 1.0 * s.target.x], 2 * np.pi, 0.02)
    final = trajectory.get_values('gas.x')[-1]
    assert np.abs(final - start).max() < 0.05


@pytest.mark.parametrize(
    ('discretization', 'options'),
    [
        ('finite-volume', {'scheme': FiniteVolumeSpec(limiter='van-leer')}),
        ('finite-difference', {}),
        ('spectral', {}),
    ],
)
def test_free_streaming_shifts_every_class_slice(discretization: str, options: dict[str, object]) -> None:
    """Shift every class slice by `v t` and keep the total within 1e-12 relative error."""
    size, length = 64, 2 * np.pi
    position = LatticeSpace((size,), spacing=length / size, discretization=discretization, **options)
    space = PhaseSpace(position, [AxisSpec('v', 16, -1.0, 1.0)]).prepare()
    v = space.coordinates_['v']
    x = np.arange(size) * length / size
    start = np.outer(2.0 + np.sin(x), np.exp(-(v**2)))
    gas = AmountElement('gas', x=start, space=space)
    trajectory = _transport(gas, lambda s: [coordinate('v') * s.target.x, 0], 1.0, 0.01)
    final = trajectory.get_values('gas.x')[-1].reshape(size, 16)
    exact = np.stack([(2.0 + np.sin(x - speed)) * np.exp(-(speed**2)) for speed in v], axis=1)
    assert np.abs(final - exact).max() < 0.02
    assert abs(final.sum() / start.sum() - 1.0) < 1e-12


def test_zero_drift_leaves_classes_unchanged() -> None:
    """Leave a spatially uniform field unchanged under a drift of zero along its class axis."""
    position = LatticeSpace((4,), discretization='finite-volume')
    space = PhaseSpace(position, [AxisSpec('v', 16, -3.0, 3.0)]).prepare()
    start = np.tile(np.exp(-(space.coordinates_['v'] ** 2)), (4, 1))
    trajectory = _transport(AmountElement('gas', x=start, space=space), lambda s: [0, 0], 1.0, 0.1)
    final = trajectory.get_values('gas.x')[-1].reshape(4, 16)
    assert np.array_equal(final, start)


def test_transport_conserves_total_through_interval_edges() -> None:
    """Keep the total when a drift along an interval class axis pushes the density into its edge."""
    position = LatticeSpace((1,), discretization='finite-volume')
    space = PhaseSpace(position, [AxisSpec('v', 32, 0.0, 1.0)]).prepare()
    trajectory = _transport(AmountElement('gas', x=1.0, space=space), lambda s: [0, 1.0 * s.target.x], 0.5, 0.005)
    final = trajectory.get_values('gas.x')[-1]
    assert abs(final.sum() / 32 - 1.0) < 1e-12
    assert final[-1] > 1.5


def test_phase_space_takes_the_discretization_of_its_position() -> None:
    """Test that a phase space takes the discretization, the scheme, and the edges of its position lattice."""
    scheme = FiniteVolumeSpec(riemann='godunov')
    position = LatticeSpace(4, boundary={'x': 'zero-flux'}, discretization='finite-volume', scheme=scheme)
    space = PhaseSpace(position, [AxisSpec('v', 2, -1.0, 1.0)]).prepare()
    assert (space.discretization, space.scheme, space.boundary) == ('finite-volume', scheme, {'x': 'zero-flux'})
    assert space.axes_periodic_ == (True, False)
