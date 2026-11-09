"""Test the semi-Lagrangian solver of fields whose fluxes are linear drifts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components.elements import FieldElement, NonnegativeElement
from skcomplex.core.components.interactions import ConservationInteraction
from skcomplex.core.components.laws.utils import coordinate, gradient, inverse_helmholtz, moment
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace, SimplexSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import SemiLagrangianSolver
from skcomplex.core.errors import SolverError
from skcomplex.core.specs import AxisSpec


def poisson_drift(namespace: SimpleNamespace) -> list[object]:
    """Return the drift of a Vlasov-Poisson field."""
    density = moment(namespace.target.x, 0, axes=('v',))
    return [coordinate('v'), -gradient(inverse_helmholtz(1 - density), 1)[0]]


def test_large_transport_steps_stay_stable_and_conserve_total() -> None:
    """Stay bounded at a Courant number of 5 and keep the total within 1e-12 relative error."""
    size, length = 32, 2 * np.pi
    spacing = length / size
    space = PhaseSpace(LatticeSpace((size,), spacing=spacing), [AxisSpec('v', 8, 0.5, 1.5)]).prepare()
    start = np.outer(2.0 + np.sin(np.arange(size) * spacing), np.ones(8))
    gas = FieldElement(
        'gas',
        element=NonnegativeElement,
        space=space,
        values={'x': start},
        discretization='finite-volume',
    )
    stream = ConservationInteraction('s', target='gas', state='x', flux=lambda s: [coordinate('v') * s.target.x, 0])
    solver = SemiLagrangianSolver(dt=5 * spacing)
    system = CompositeSystem(elements=[gas], interactions=[stream])
    trajectory = simulate(system, until=4.0, dt=5 * spacing, solver=solver)
    final = trajectory.values('gas.x')[-1].reshape(size, 8)
    assert np.abs(final).max() < 1.01 * start.max()
    assert abs(final.sum() / start.sum() - 1.0) < 1e-12
    exact = np.stack(
        [2.0 + np.sin(np.arange(size) * spacing - speed * 4.0) for speed in space.coordinates_['v']],
        axis=1,
    )
    assert np.abs(final - exact).max() < 1e-10


def test_nonlinear_flux_raises_a_named_error() -> None:
    """Raise a named error saying that the flux of Burgers' equation is not a linear drift."""
    space = LatticeSpace((16,), spacing=0.1)
    values = {'x': 1.0 + np.sin(2 * np.pi * np.arange(16) / 16)}
    wave = FieldElement('wave', element=NonnegativeElement, space=space, values=values, discretization='finite-volume')
    burgers = ConservationInteraction('b', target='wave', state='x', flux=lambda s: s.target.x**2 / 2)
    with pytest.raises(SolverError, match='not a linear drift'):
        simulate(
            CompositeSystem(elements=[wave], interactions=[burgers]),
            until=0.1,
            solver=SemiLagrangianSolver(dt=0.1),
        )


def test_partial_cells_raise_a_named_error() -> None:
    """Raise a named error saying that a field on a simplex has partial cells that whole slices cannot shift."""
    space = SimplexSpace(LatticeSpace((1,)), [AxisSpec('x', 8, 0.0, 1.0), AxisSpec('y', 8, 0.0, 1.0)])
    values = {'x': 2.0 * (space.prepare().volumes_ > 0)}
    groups = FieldElement(
        'groups',
        element=NonnegativeElement,
        space=space,
        values=values,
        discretization='finite-volume',
    )
    flow = ConservationInteraction(
        'flow', target='groups', state='x', flux=lambda s: [0, 1.0 * s.target.x, -1.0 * s.target.x],
    )
    with pytest.raises(SolverError, match='partial cells'):
        simulate(
            CompositeSystem(elements=[groups], interactions=[flow]),
            until=0.1,
            solver=SemiLagrangianSolver(dt=0.1),
        )


def test_landau_damping_with_large_steps() -> None:
    """Damp the electric energy at the linear rate within 5 percent at a step that an explicit scheme cannot take."""
    positions, velocities, length = 32, 64, 4 * np.pi
    axis = AxisSpec('v', velocities, -6.0, 6.0)
    space = PhaseSpace(LatticeSpace((positions,), spacing=length / positions), [axis]).prepare()
    v = space.coordinates_['v']
    x = np.arange(positions) * length / positions
    start = np.outer(1.0 + 0.01 * np.cos(0.5 * x), np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi))
    gas = FieldElement(
        'gas',
        element=NonnegativeElement,
        space=space,
        values={'x': start},
        discretization='finite-volume',
    )
    drift = ConservationInteraction(
        'vlasov', target='gas', state='x', flux=lambda s: [component * s.target.x for component in poisson_drift(s)],
    )
    system = CompositeSystem(elements=[gas], interactions=[drift])
    solver = SemiLagrangianSolver(dt=0.25)
    trajectory = simulate(system, until=30.0, dt=0.25, solver=solver)
    values = np.asarray(trajectory.values('gas.x')).reshape(-1, positions, velocities)
    density = values.sum(axis=2) * space.spacings_[1]
    field = np.fft.ifft(np.fft.fft(density - density.mean(axis=1, keepdims=True), axis=1), axis=1).real
    amplitude = np.abs(np.fft.rfft(field, axis=1)[:, 1])
    times = np.asarray(trajectory.times)
    peaks = [
        index
        for index in range(1, times.size - 1)
        if amplitude[index] > amplitude[index - 1] and amplitude[index] > amplitude[index + 1]
    ]
    slope = np.polyfit(times[peaks], np.log(amplitude[peaks]), 1)[0]
    assert abs(-slope - 0.1533) < 0.05 * 0.1533
