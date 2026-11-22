"""Test the semi-Lagrangian solver of fields whose fluxes are linear drifts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import coordinate, divergence, gradient, inverse_helmholtz, moment
from skcomplex.core.components.spaces import AxisSpec, BaseSpace, LatticeSpace, PhaseSpace
from skcomplex.core.engine.solvers import SemiLagrangianSolver
from skcomplex.core.errors import SolverError
from skcomplex.core.systems import System
from skcomplex.systems.social.cooperation import SimplexSpace


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount, or one per site of a space."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[npt.ArrayLike, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x


def _transport(label: str, target: str, flux: Callable[[SimpleNamespace], list[object]]) -> Interaction:
    """Return the transport of the amount of a field by the divergence of a flux, a function of the namespace."""
    return Interaction(
        label,
        roles={'target': target},
        laws=lambda s: [RateLaw(s.target.x.diff(s.t), -divergence(flux(s)))],
    )


def poisson_drift(namespace: SimpleNamespace) -> list[object]:
    """Return the drift of a Vlasov-Poisson field."""
    density = moment(namespace.target.x, 0, axes=('v',))
    return [coordinate('v'), -gradient(inverse_helmholtz(1 - density), 1)[0]]


def test_large_transport_steps_stay_stable_and_conserve_total() -> None:
    """Stay bounded at a Courant number of 5 and keep the total within 1e-12 relative error."""
    size, length = 32, 2 * np.pi
    spacing = length / size
    position = LatticeSpace((size,), spacing=spacing, discretization='finite-volume')
    space = PhaseSpace(position, [AxisSpec('v', 8, 0.5, 1.5)]).prepare()
    start = np.outer(2.0 + np.sin(np.arange(size) * spacing), np.ones(8))
    gas = AmountElement('gas', x=start, space=space)
    stream = _transport('s', 'gas', lambda s: [coordinate('v') * s.target.x, 0])
    system = System(elements=[gas], interactions=[stream], solver=SemiLagrangianSolver(dt=5 * spacing))
    trajectory = system.prepare().simulate(until=4.0, dt=5 * spacing)
    final = trajectory.get_values('gas.x')[-1].reshape(size, 8)
    assert np.abs(final).max() < 1.01 * start.max()
    assert abs(final.sum() / start.sum() - 1.0) < 1e-12
    exact = np.stack(
        [2.0 + np.sin(np.arange(size) * spacing - speed * 4.0) for speed in space.coordinates_['v']],
        axis=1,
    )
    assert np.abs(final - exact).max() < 1e-10


def test_nonlinear_flux_raises_a_named_error() -> None:
    """Raise a named error saying that the flux of Burgers' equation is not a linear drift."""
    space = LatticeSpace((16,), spacing=0.1, discretization='finite-volume')
    wave = AmountElement('wave', x=1.0 + np.sin(2 * np.pi * np.arange(16) / 16), space=space)
    burgers = _transport('b', 'wave', lambda s: [s.target.x**2 / 2])
    system = System(elements=[wave], interactions=[burgers], solver=SemiLagrangianSolver(dt=0.1))
    with pytest.raises(SolverError, match='not a linear drift'):
        system.prepare().simulate(until=0.1)


def test_partial_cells_raise_a_named_error() -> None:
    """Raise a named error saying that a field on a simplex has partial cells that whole slices cannot shift."""
    position = LatticeSpace((1,), discretization='finite-volume')
    space = SimplexSpace(position, [AxisSpec('x', 8, 0.0, 1.0), AxisSpec('y', 8, 0.0, 1.0)])
    groups = AmountElement('groups', x=2.0 * (space.prepare().volumes_ > 0), space=space)
    flow = _transport('flow', 'groups', lambda s: [0, 1.0 * s.target.x, -1.0 * s.target.x])
    system = System(elements=[groups], interactions=[flow], solver=SemiLagrangianSolver(dt=0.1))
    with pytest.raises(SolverError, match='partial cells'):
        system.prepare().simulate(until=0.1)


def test_landau_damping_with_large_steps() -> None:
    """Damp the electric energy at the linear rate within 5 percent at a step that an explicit scheme cannot take."""
    positions, velocities, length = 32, 64, 4 * np.pi
    axis = AxisSpec('v', velocities, -6.0, 6.0)
    position = LatticeSpace((positions,), spacing=length / positions, discretization='finite-volume')
    space = PhaseSpace(position, [axis]).prepare()
    v = space.coordinates_['v']
    x = np.arange(positions) * length / positions
    start = np.outer(1.0 + 0.01 * np.cos(0.5 * x), np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi))
    gas = AmountElement('gas', x=start, space=space)
    drift = _transport('vlasov', 'gas', lambda s: [component * s.target.x for component in poisson_drift(s)])
    system = System(elements=[gas], interactions=[drift], solver=SemiLagrangianSolver(dt=0.25))
    trajectory = system.prepare().simulate(until=30.0, dt=0.25)
    values = np.asarray(trajectory.get_values('gas.x')).reshape(-1, positions, velocities)
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
