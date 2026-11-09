"""Test phase spaces, the product of a position lattice and class axes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import ConservationInteraction
from skcomplex.core.components.laws.utils import coordinate
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace, PhaseSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import SpaceTooLargeError, ValidationError
from skcomplex.core.specs import AxisSpec, StateSpec, VisualSpec


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


def test_prepare_holds_one_value_per_point_of_the_product() -> None:
    """Hold 8192 values for 64 positions and 128 class points, with initial values over position and class."""
    space = PhaseSpace(LatticeSpace((64,)), [AxisSpec('v', 128, -6.0, 6.0)]).prepare()
    values = np.ones((64, 128))
    values[3, 5] = 2.0
    field = FieldElement('gas', element=AmountElement, space=space, values={'x': values}).prepare()
    assert field.size_ == 8192
    assert space.class_axes_ == {'v': 1}
    assert space.position_axes_ == (0,)
    assert space.spacings_ == (1.0, 12.0 / 128)
    problem = simulate(CompositeSystem(elements=[field]), until=0, dt=0.1).problem
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
    field = FieldElement('gas', element=AmountElement, space=space)
    report = PassCompiler().prepare(CompositeSystem(elements=[field])).report_
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
        ('finite-volume', {'limiter': 'van-leer'}),
        ('finite-difference', {'riemann': None}),
        ('spectral', {'riemann': None}),
    ],
)
def test_periodic_class_axis_returns_after_one_turn(discretization: str, options: dict[str, object]) -> None:
    """Return a profile on a periodic axis of 32 angles to its start after one turn at constant drift."""
    axis = AxisSpec('theta', 32, 0.0, 2 * np.pi, periodic=True)
    space = PhaseSpace(LatticeSpace((1,)), [axis]).prepare()
    start = 1.0 + 0.5 * np.cos(space.coordinates_['theta'])
    field = FieldElement(
        'gas',
        element=AmountElement,
        space=space,
        values={'x': start},
        discretization=discretization,
    )
    drift = ConservationInteraction('turn', target='gas', state='x', flux=lambda s: [0, 1.0 * s.target.x], **options)
    trajectory = simulate(
        CompositeSystem(elements=[field], interactions=[drift]),
        until=2 * np.pi,
        solver=RK4Solver(dt=0.02),
    )
    final = trajectory.get_values('gas.x')[-1]
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
    space = PhaseSpace(LatticeSpace((size,), spacing=length / size), [AxisSpec('v', 16, -1.0, 1.0)]).prepare()
    v = space.coordinates_['v']
    x = np.arange(size) * length / size
    start = np.outer(2.0 + np.sin(x), np.exp(-(v**2)))
    field = FieldElement(
        'gas',
        element=AmountElement,
        space=space,
        values={'x': start},
        discretization=discretization,
    )
    drift = ConservationInteraction(
        'stream',
        target='gas',
        state='x',
        flux=lambda s: [coordinate('v') * s.target.x, 0],
        **options,
    )
    trajectory = simulate(CompositeSystem(elements=[field], interactions=[drift]), until=1.0, solver=RK4Solver(dt=0.01))
    final = trajectory.get_values('gas.x')[-1].reshape(size, 16)
    exact = np.stack([(2.0 + np.sin(x - speed)) * np.exp(-(speed**2)) for speed in v], axis=1)
    assert np.abs(final - exact).max() < 0.02
    assert abs(final.sum() / start.sum() - 1.0) < 1e-12


def test_zero_drift_leaves_classes_unchanged() -> None:
    """Leave a spatially uniform field unchanged under a drift of zero along its class axis."""
    space = PhaseSpace(LatticeSpace((4,)), [AxisSpec('v', 16, -3.0, 3.0)]).prepare()
    start = np.tile(np.exp(-(space.coordinates_['v'] ** 2)), (4, 1))
    field = FieldElement(
        'gas',
        element=AmountElement,
        space=space,
        values={'x': start},
        discretization='finite-volume',
    )
    drift = ConservationInteraction('still', target='gas', state='x', flux=lambda s: [0, 0])
    trajectory = simulate(CompositeSystem(elements=[field], interactions=[drift]), until=1.0, solver=RK4Solver(dt=0.1))
    final = trajectory.get_values('gas.x')[-1].reshape(4, 16)
    assert np.array_equal(final, start)


def test_transport_conserves_total_through_interval_edges() -> None:
    """Keep the total when a drift along an interval class axis pushes the density into its edge."""
    space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('v', 32, 0.0, 1.0)]).prepare()
    field = FieldElement(
        'gas',
        element=AmountElement,
        space=space,
        values={'x': 1.0},
        discretization='finite-volume',
    )
    drift = ConservationInteraction('push', target='gas', state='x', flux=lambda s: [0, 1.0 * s.target.x])
    trajectory = simulate(
        CompositeSystem(elements=[field], interactions=[drift]),
        until=0.5,
        solver=RK4Solver(dt=0.005),
    )
    final = trajectory.get_values('gas.x')[-1]
    assert abs(final.sum() / 32 - 1.0) < 1e-12
    assert final[-1] > 1.5
