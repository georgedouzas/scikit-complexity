"""Test what the laws of a phase space read: the class coordinates, the moments, and the equilibria."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, var
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import ConservationInteraction, RateInteraction
from skcomplex.core.components.laws.utils import (
    coordinate,
    gradient,
    inverse_helmholtz,
    maximum_entropy,
    moment,
    reduce_axes,
)
from skcomplex.core.components.laws.utils._phase import coordinates_of
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import AxisSpec, ObservableSpec, StateSpec, VisualSpec


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



def density(namespace: SimpleNamespace) -> object:
    """Return the density moment of the field over its velocity axis."""
    return moment(namespace.x, 0, axes=('v',))


def electric_energy(namespace: SimpleNamespace) -> object:
    """Return the electric field squared solved from the density moment."""
    return gradient(inverse_helmholtz(1 - moment(namespace.x, 0, axes=('v',))), 1)[0] ** 2


def potential_force(namespace: SimpleNamespace) -> list[object]:
    """Return the drift of a Vlasov-Poisson field, the velocity along position and the force along velocity."""
    return [
        coordinate('v'),
        -gradient(inverse_helmholtz(1 - moment(namespace.target.x, 0, axes=("v",))), 1)[0],
    ]


def test_coordinate_rejects_a_name_that_is_no_identifier() -> None:
    """Raise a named error for an axis name that is no identifier."""
    with pytest.raises(ValidationError, match='identifier'):
        coordinate('not a name')
    with pytest.raises(ValidationError, match='sequence of names'):
        coordinates_of('v')


def test_coordinate_lowering_gives_maxwellian_start_unit_density() -> None:
    """Give a Maxwellian start the density 1 at every position within the quadrature error of the grid."""
    space = PhaseSpace(LatticeSpace((4,)), [AxisSpec('v', 128, -6.0, 6.0)]).prepare()
    v = space.coordinates_['v']
    start = np.tile(np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi), (4, 1))
    field = FieldElement('gas', element=AmountElement, space=space, values={'x': start})
    observables = [ObservableSpec('rho', density, group='gas')]
    trajectory = simulate(CompositeSystem(elements=[field]), until=0.1, dt=0.1, observables=observables)
    values = np.asarray(trajectory.values('rho'))
    assert np.abs(values - 1.0).max() < 1e-6


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_coordinate_lowers_on_numpy_and_numba(backend: object) -> None:
    """Stream by the same stencils and rates that read a coordinate on the NumPy and the Numba backends."""
    space = PhaseSpace(LatticeSpace((8,), spacing=0.5), [AxisSpec('v', 4, -1.0, 1.0)]).prepare()
    start = np.outer(2.0 + np.sin(np.arange(8)), np.ones(4))
    gas = FieldElement('gas', element=AmountElement, space=space, values={'x': start})
    stream = ConservationInteraction(
        's', target='gas', state='x', flux=lambda s: [coordinate('v') * s.target.x, 0], riemann=None,
    )
    grow = RateInteraction('g', target='gas', state='x', rate=lambda s: coordinate('v') * s.target.x)
    system = CompositeSystem(elements=[gas], interactions=[stream, grow])
    trajectory = simulate(system, until=0.1, backend=backend, solver=RK4Solver(dt=0.01))
    final = trajectory.values('gas.x')[-1].reshape(8, 4)
    grown = np.exp(0.1 * space.coordinates_['v'])
    assert final.sum(axis=0) == pytest.approx(start.sum(axis=0) * grown, rel=1e-3)


def test_coordinate_outside_phase_space_is_a_law_space_issue() -> None:
    """Report a coordinate read by a law whose group lives on a lattice without class axes."""
    field = FieldElement('gas', element=AmountElement, space=LatticeSpace((4,)))
    rate = RateInteraction('rate', target='gas', state='x', rate=lambda _: coordinate('v'))
    report = PassCompiler().prepare(CompositeSystem(elements=[field], interactions=[rate])).report_
    assert any(issue.code == 'law-space' and 'class axes v' in issue.message for issue in report.issues)


def test_coordinate_of_missing_axis_names_axes_held() -> None:
    """Report a coordinate of an axis the phase space does not hold, with the axes it holds."""
    space = PhaseSpace(LatticeSpace((4,)), [AxisSpec('v', 4, -1.0, 1.0)])
    field = FieldElement('gas', element=AmountElement, space=space)
    rate = RateInteraction('rate', target='gas', state='x', rate=lambda _: coordinate('w'))
    report = PassCompiler().prepare(CompositeSystem(elements=[field], interactions=[rate])).report_
    issues = [issue for issue in report.issues if issue.code == 'law-space']
    assert issues
    assert "['v']" in issues[0].message


def test_moment_outside_phase_space_is_a_law_space_issue() -> None:
    """Report a moment over axes in a law whose group lives on a lattice without class axes."""
    field = FieldElement('gas', element=AmountElement, space=LatticeSpace((4,)))
    rate = RateInteraction('rate', target='gas', state='x', rate=lambda s: moment(s.target.x, 0, axes=('v',)))
    report = PassCompiler().prepare(CompositeSystem(elements=[field], interactions=[rate])).report_
    assert any(issue.code == 'law-space' for issue in report.issues)


def test_moment_rejects_coordinates_with_axes_and_odd_order_over_two_axes() -> None:
    """Raise named errors for coordinates next to axes and an odd order over two axes."""
    with pytest.raises(ValidationError, match='no other coordinates'):
        moment(var('x'), 0, [0.0, 1.0], axes=('v',))
    with pytest.raises(ValidationError, match='even order'):
        moment(var('x'), 1, axes=('v', 'w'))


def test_moment_with_axes_is_reduce_axes_of_coordinates() -> None:
    """Write a moment over an axis as the integral of the expression times the coordinate raised to its order."""
    f = var('x')
    assert moment(f, 1, axes=('v',)) == reduce_axes(f * coordinate('v'), ('v',))


def test_moments_of_a_shifted_maxwellian_return_density_mean_and_temperature() -> None:
    """Return the density 2, the mean velocity 0.5, and the temperature 1.5 at every position within 1e-8."""
    size = 3
    space = PhaseSpace(LatticeSpace((size,)), [AxisSpec('v', 256, -16.0, 16.0)]).prepare()
    v = space.coordinates_['v']
    start = 2.0 * np.exp(-((v - 0.5) ** 2) / 3.0) / np.sqrt(3.0 * np.pi)
    field = FieldElement('gas', element=AmountElement, space=space, values={'x': np.tile(start, (size, 1))})
    observables = [
        ObservableSpec('rho', density, group='gas'),
        ObservableSpec('mean', mean_at, group='gas'),
        ObservableSpec('temperature', temperature_at, group='gas'),
    ]
    trajectory = simulate(CompositeSystem(elements=[field]), until=0.1, dt=0.1, observables=observables)
    assert np.abs(np.asarray(trajectory.values('rho'))[-1] - 2.0).max() < 1e-8
    assert np.abs(np.asarray(trajectory.values('mean'))[-1] - 0.5).max() < 1e-8
    assert np.abs(np.asarray(trajectory.values('temperature'))[-1] - 1.5).max() < 1e-8


def mean_at(namespace: SimpleNamespace) -> object:
    """Return the mean velocity at every position."""
    return moment(namespace.x, 1, axes=('v',), normalized=True)


def temperature_at(namespace: SimpleNamespace) -> object:
    """Return the variance of the velocity at every position."""
    return moment(namespace.x, 2, axes=('v',), central=True)


def test_rate_over_position_reads_moment_of_phase_field_through_operators() -> None:
    """Lower a drift that reads the gradient of the potential of its own density moment."""
    space = PhaseSpace(LatticeSpace((16,), spacing=4 * np.pi / 16), [AxisSpec('v', 32, -6.0, 6.0)]).prepare()
    v = space.coordinates_['v']
    x = np.arange(16) * 4 * np.pi / 16
    start = np.outer(1.0 + 0.01 * np.cos(0.5 * x), np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi))
    field = FieldElement(
        'gas',
        element=AmountElement,
        space=space,
        values={'x': start},
        discretization='finite-volume',
    )
    drift = ConservationInteraction(
        'vlasov', target='gas', state='x', flux=lambda s: [component * s.target.x for component in potential_force(s)],
    )
    system = CompositeSystem(elements=[field], interactions=[drift])
    final = simulate(system, until=0.5, solver=RK4Solver(dt=0.05)).values('gas.x')[-1]
    assert abs(final.sum() / start.sum() - 1.0) < 1e-12


def test_drift_reads_field_over_position_through_sources() -> None:
    """Accelerate only the classes at the positions where a field over the position lattice is nonzero."""
    lattice = LatticeSpace((2,))
    space = PhaseSpace(lattice, [AxisSpec('v', 32, -4.0, 4.0)]).prepare()
    start = np.tile(np.exp(-(space.coordinates_['v'] ** 2)), (2, 1))
    gas = FieldElement(
        'gas',
        element=AmountElement,
        space=space,
        values={'x': start},
        discretization='finite-volume',
    )
    force = FieldElement('force', element=AmountElement, space=lattice, values={'x': [1.0, 0.0]})
    drift = ConservationInteraction(
        'push', target='gas', state='x', flux=lambda s: [0, s.sources[0].x * s.target.x], sources=['force'],
    )
    system = CompositeSystem(elements=[gas, force], interactions=[drift], space=space)
    final = simulate(system, until=0.5, solver=RK4Solver(dt=0.01))
    values = final.values('gas.x')[-1].reshape(2, 32)
    assert np.array_equal(values[1], start[1])
    assert np.average(space.coordinates_['v'], weights=values[0]) == pytest.approx(0.5, abs=0.01)


def test_rate_over_position_reads_moment_through_sources() -> None:
    """Grow a field over the position lattice at the rate of the density moment of a phase-space field."""
    lattice = LatticeSpace((3,))
    space = PhaseSpace(lattice, [AxisSpec('v', 8, -1.0, 1.0)]).prepare()
    start = np.ones((3, 8)) * np.array([[1.0], [2.0], [3.0]])
    gas = FieldElement('gas', element=AmountElement, space=space, values={'x': start})
    level = FieldElement('level', element=AmountElement, space=lattice)
    rate = RateInteraction(
        'grow',
        target='level',
        state='x',
        rate=lambda s: moment(s.sources[0].x, 0, axes=('v',)),
        sources=['gas'],
    )
    system = CompositeSystem(elements=[gas, level], interactions=[rate], space=space)
    final = simulate(system, until=1.0, solver=RK4Solver(dt=0.1))
    assert final.values('level.x')[-1] == pytest.approx([2.0, 4.0, 6.0])


def test_maximum_entropy_rejects_an_order_above_two() -> None:
    """Raise a named error for an order of a maximum-entropy density that is not 0, 1, or 2."""
    with pytest.raises(ValidationError, match='0, 1, or 2'):
        maximum_entropy(var('x'), ('v',), order=3)


def test_relaxation_to_maximum_entropy_conserves_moments_and_raises_entropy() -> None:
    """Keep density, momentum, and energy within 1e-12, converge to a Maxwellian, and never lower the entropy."""
    space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('v', 128, -16.0, 16.0)]).prepare()
    v = space.coordinates_['v']
    width = space.spacings_[1]
    start = np.exp(-((v - 2.0) ** 2) / 2) + 0.5 * np.exp(-((v + 2.0) ** 2) / 2)
    gas = FieldElement('gas', element=AmountElement, space=space, values={'x': start})
    relax = RateInteraction(
        'bgk',
        target='gas',
        state='x',
        rate=lambda s: maximum_entropy(s.target.x, axes=('v',)) - s.target.x,
    )
    system = CompositeSystem(elements=[gas], interactions=[relax])
    trajectory = simulate(system, until=50.0, dt=1.0, solver=RK4Solver(dt=0.1))
    values = np.asarray(trajectory.values('gas.x'))
    moments = values @ np.stack([np.ones(v.size), v, v**2], axis=1) * width
    assert np.abs(moments[-1] / moments[0] - 1.0).max() < 1e-12
    entropy = -np.sum(values * np.log(values), axis=1)
    assert np.all(np.diff(entropy) >= -1e-12)
    density, mean = moments[0, 0], moments[0, 1] / moments[0, 0]
    temperature = moments[0, 2] / density - mean**2
    exact = density * np.exp(-((v - mean) ** 2) / (2 * temperature)) / np.sqrt(2 * np.pi * temperature)
    assert np.abs(values[-1] - exact).max() < 1e-5


def test_maximum_entropy_of_two_axes_keeps_every_moment_at_every_position() -> None:
    """Keep the moments of a density over two class axes at two positions in the equilibrium."""
    axes = [AxisSpec('a', 24, -6.0, 6.0), AxisSpec('b', 24, -6.0, 6.0)]
    space = PhaseSpace(LatticeSpace((2,)), axes).prepare()
    a, b = np.meshgrid(space.coordinates_['a'], space.coordinates_['b'], indexing='ij')
    first = np.exp(-((a - 1.0) ** 2) - b**2) + np.exp(-((a + 1.0) ** 2) - (b - 1.0) ** 2)
    start = np.stack([first, 2.0 * first[::-1]])
    gas = FieldElement('gas', element=AmountElement, space=space, values={'x': start})
    relax = RateInteraction(
        'bgk',
        target='gas',
        state='x',
        rate=lambda s: maximum_entropy(s.target.x, axes=('a', 'b')) - s.target.x,
    )
    trajectory = simulate(CompositeSystem(elements=[gas], interactions=[relax]), until=1.0, solver=RK4Solver(dt=0.2))
    final = trajectory.values('gas.x')[-1].reshape(start.shape)
    for basis in (np.ones(a.shape), a, b, a**2 + b**2):
        assert np.sum(final * basis, axis=(1, 2)) == pytest.approx(np.sum(start * basis, axis=(1, 2)), rel=1e-11)


def test_landau_damping_decays_at_the_linear_rate() -> None:
    """Damp the electric energy at the linear Landau rate 0.1533 within 5 percent up to time 30."""
    positions, velocities, length = 64, 128, 4 * np.pi
    axis = AxisSpec('v', velocities, -6.0, 6.0)
    space = PhaseSpace(LatticeSpace((positions,), spacing=length / positions), [axis]).prepare()
    v = space.coordinates_['v']
    x = np.arange(positions) * length / positions
    start = np.outer(1.0 + 0.01 * np.cos(0.5 * x), np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi))
    field = FieldElement(
        'gas',
        element=AmountElement,
        space=space,
        values={'x': start},
        discretization='finite-difference',
    )
    drift = ConservationInteraction(
        'vlasov',
        target='gas',
        state='x',
        flux=lambda s: [component * s.target.x for component in potential_force(s)],
        riemann=None,
    )
    observables = [ObservableSpec('energy', electric_energy, group='gas')]
    system = CompositeSystem(elements=[field], interactions=[drift])
    trajectory = simulate(system, until=30.0, dt=0.1, solver=RK4Solver(dt=0.05), observables=observables)
    energy = np.asarray(trajectory.values('energy'), dtype=float)
    times = np.asarray(trajectory.times)
    energy = energy.reshape(times.size, -1).sum(axis=1)
    peaks = [
        index
        for index in range(1, times.size - 1)
        if energy[index] > energy[index - 1] and energy[index] > energy[index + 1]
    ]
    slope = np.polyfit(times[peaks], np.log(energy[peaks]), 1)[0]
    assert abs(-slope / 2 - 0.1533) < 0.05 * 0.1533
