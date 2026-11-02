"""Test phase spaces, the product of a position lattice and class axes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import pytest

from skcomplex.core.elements import DensityElement, FieldElement
from skcomplex.core.interactions import AdvectionInteraction
from skcomplex.core.laws.utils import coordinate
from skcomplex.core.spaces import LatticeNeighborhood, LatticeSpace, PhaseSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import PassCompiler
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.errors import SpaceTooLargeError, ValidationError
from skcomplex.specs import AxisSpec


def test_prepare_holds_one_value_per_point_of_the_product() -> None:
    """Hold 8192 values for 64 positions and 128 class points, with initial values over position and class."""
    space = PhaseSpace(LatticeSpace((64,)), [AxisSpec('v', 128, -6.0, 6.0)]).check().prepare()
    values = np.ones((64, 128))
    values[3, 5] = 2.0
    field = FieldElement('gas', element=DensityElement, space=space, values={'f': values}).check().prepare()
    assert field.size_ == 8192
    assert space.class_axes_ == {'v': 1}
    assert space.position_axes_ == (0,)
    assert space.spacings_ == (1.0, 12.0 / 128)
    simulation = Simulation(System(elements=[field]), dt=0.1).check().prepare()
    assert simulation.problem_.initial_state[3 * 128 + 5] == 2.0


def test_prepare_places_coordinates_at_cell_centres() -> None:
    """Place the coordinates of an axis at the centres of its cells."""
    space = PhaseSpace(LatticeSpace(2), [AxisSpec('v', 4, 0.0, 2.0)]).check().prepare()
    assert space.coordinates_['v'].tolist() == [0.25, 0.75, 1.25, 1.75]


def test_check_rejects_position_or_axes_it_cannot_take() -> None:
    """Reject a triangular position, no axes, and axes that share a name."""
    axis = AxisSpec('v', 4, 0.0, 2.0)
    with pytest.raises(ValidationError, match='square lattice'):
        PhaseSpace(LatticeSpace((2, 2), geometry='triangular'), [axis]).check()
    with pytest.raises(ValidationError, match='one or more class axes'):
        PhaseSpace(LatticeSpace(2), []).check()
    with pytest.raises(ValidationError, match='distinct names'):
        PhaseSpace(LatticeSpace(2), [axis, axis]).check()


def test_prepare_raises_above_limit() -> None:
    """Raise a named error that holds the size the space would take above its limit."""
    space = PhaseSpace(LatticeSpace((1000,)), [AxisSpec('v', 1000, -1.0, 1.0)], limit=100).check()
    with pytest.raises(SpaceTooLargeError, match='1000000 sites'):
        space.prepare()


def test_check_reports_space_size_issue_with_size() -> None:
    """Report a phase space above its limit through the compile report, with the size it would take."""
    space = PhaseSpace(LatticeSpace((10,)), [AxisSpec('v', 10, -1.0, 1.0)], limit=50)
    field = FieldElement('gas', element=DensityElement, space=space)
    report = PassCompiler().check().prepare(System(elements=[field])).report_
    issues = [issue for issue in report.issues if issue.code == 'space-size']
    assert issues
    assert '100 sites' in issues[0].message


def test_index_table_keeps_edges_per_axis() -> None:
    """Join a periodic position axis and reflect at the ends of an interval class axis."""
    space = PhaseSpace(LatticeSpace(3), [AxisSpec('v', 2, 0.0, 1.0)]).check().prepare()
    ahead = space._index_table(LatticeNeighborhood(((1, 0),)))[:, 0]
    above = space._index_table(LatticeNeighborhood(((0, 1),)))[:, 0]
    assert ahead.tolist() == [2, 3, 4, 5, 0, 1]
    assert above.tolist() == [1, 1, 3, 3, 5, 5]


@pytest.mark.parametrize(
    ('discretization', 'options'),
    [
        ('finite-volume', {'limiter': 'van-leer'}),
        ('finite-difference', {'riemann': None}),
        ('spectral', {'riemann': None}),
    ],
)
def test_periodic_class_axis_returns_after_one_turn(discretization: str, options: dict[str, object]) -> None:
    """Return a profile on a periodic axis of 32 angles to its start after one turn at constant drift."""
    axis = AxisSpec('theta', 32, 0.0, 2 * np.pi, periodic=True)
    space = PhaseSpace(LatticeSpace((1,)), [axis]).check().prepare()
    start = 1.0 + 0.5 * np.cos(space.coordinates_['theta'])
    field = FieldElement('gas', element=DensityElement, space=space, values={'f': start}, discretization=discretization)
    drift = AdvectionInteraction('turn', target='gas', state='f', drift=lambda _: [0, 1.0], **options)
    simulation = Simulation(System(elements=[field], interactions=[drift]), solver=RK4Solver(dt=0.02)).check().prepare()
    final = simulation.run(until=2 * np.pi).trajectory_.values('gas.f')[-1]
    assert np.abs(final - start).max() < 0.05


@pytest.mark.parametrize(
    ('discretization', 'options'),
    [
        ('finite-volume', {'limiter': 'van-leer'}),
        ('finite-difference', {'riemann': None}),
        ('spectral', {'riemann': None}),
    ],
)
def test_free_streaming_shifts_every_class_slice(discretization: str, options: dict[str, object]) -> None:
    """Shift every class slice by `v t` and keep the total within 1e-12 relative error."""
    size, length = 64, 2 * np.pi
    space = PhaseSpace(LatticeSpace((size,), spacing=length / size), [AxisSpec('v', 16, -1.0, 1.0)]).check().prepare()
    v = space.coordinates_['v']
    x = np.arange(size) * length / size
    start = np.outer(2.0 + np.sin(x), np.exp(-(v**2)))
    field = FieldElement('gas', element=DensityElement, space=space, values={'f': start}, discretization=discretization)
    drift = AdvectionInteraction('stream', target='gas', state='f', drift=lambda _: [coordinate('v'), 0], **options)
    simulation = Simulation(System(elements=[field], interactions=[drift]), solver=RK4Solver(dt=0.01)).check().prepare()
    final = simulation.run(until=1.0).trajectory_.values('gas.f')[-1].reshape(size, 16)
    exact = np.stack([(2.0 + np.sin(x - speed)) * np.exp(-(speed**2)) for speed in v], axis=1)
    assert np.abs(final - exact).max() < 0.02
    assert abs(final.sum() / start.sum() - 1.0) < 1e-12


def test_zero_drift_leaves_classes_unchanged() -> None:
    """Leave a spatially uniform field unchanged under a drift of zero along its class axis."""
    space = PhaseSpace(LatticeSpace((4,)), [AxisSpec('v', 16, -3.0, 3.0)]).check().prepare()
    start = np.tile(np.exp(-(space.coordinates_['v'] ** 2)), (4, 1))
    field = FieldElement(
        'gas',
        element=DensityElement,
        space=space,
        values={'f': start},
        discretization='finite-volume',
    )
    drift = AdvectionInteraction('still', target='gas', state='f', drift=lambda _: [0, 0])
    simulation = Simulation(System(elements=[field], interactions=[drift]), solver=RK4Solver(dt=0.1)).check().prepare()
    final = simulation.run(until=1.0).trajectory_.values('gas.f')[-1].reshape(4, 16)
    assert np.array_equal(final, start)


def test_transport_conserves_total_through_interval_edges() -> None:
    """Keep the total when a drift along an interval class axis pushes the density into its edge."""
    space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('v', 32, 0.0, 1.0)]).check().prepare()
    field = FieldElement('gas', element=DensityElement, space=space, values={'f': 1.0}, discretization='finite-volume')
    drift = AdvectionInteraction('push', target='gas', state='f', drift=lambda _: [0, 1.0])
    simulation = (
        Simulation(System(elements=[field], interactions=[drift]), solver=RK4Solver(dt=0.005)).check().prepare()
    )
    final = simulation.run(until=0.5).trajectory_.values('gas.f')[-1]
    assert abs(final.sum() / 32 - 1.0) < 1e-12
    assert final[-1] > 1.5
