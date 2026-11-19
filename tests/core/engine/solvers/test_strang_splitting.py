"""Test the Strang splitting of transport and collisions."""

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
from skcomplex.core.components.laws.utils import coordinate, divergence, maximum_entropy
from skcomplex.core.components.spaces import AxisSpec, BaseSpace, LatticeSpace, PhaseSpace
from skcomplex.core.engine.solvers import EulerMaruyamaSolver, RK4Solver, SemiLagrangianSolver, StrangSplittingSolver
from skcomplex.core.errors import SolverError
from skcomplex.core.systems import System


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


def equilibrium(namespace: SimpleNamespace) -> object:
    """Return the maximum-entropy equilibrium of the field."""
    return maximum_entropy(namespace.target.x, axes=('v',))


def _relaxation() -> Interaction:
    """Return the relaxation of the gas towards its equilibrium at the time scale 0.5."""
    return Interaction(
        'r',
        roles={'target': 'gas'},
        laws=lambda s: [RateLaw(s.target.x.diff(s.t), (equilibrium(s) - s.target.x) / 0.5)],
    )


def _streaming() -> Interaction:
    """Return the streaming of the gas at its velocity."""
    return _transport('s', 'gas', lambda s: [coordinate('v') * s.target.x, 0])


def bgk_final(step: float) -> npt.NDArray[np.float64]:
    """Return the final state of a BGK field run with split steps of a length."""
    size, length = 16, 2 * np.pi
    position = LatticeSpace((size,), spacing=length / size, discretization='finite-volume')
    space = PhaseSpace(position, [AxisSpec('v', 32, -8.0, 8.0)]).prepare()
    v, x = space.coordinates_['v'], np.arange(size) * length / size
    start = np.outer(1.0 + 0.5 * np.sin(x), np.exp(-(v**2) / 2)) + np.outer(
        1.0 + 0.5 * np.cos(x),
        np.exp(-((v - 1) ** 2)),
    )
    gas = AmountElement('gas', x=start, space=space)
    solver = StrangSplittingSolver(SemiLagrangianSolver(dt=step), RK4Solver(dt=step), dt=step)
    system = System(elements=[gas], interactions=[_streaming(), _relaxation()], solver=solver)
    return system.prepare().simulate(until=1.0, dt=1.0).get_values('gas.x')[-1]


def test_splitting_is_second_order_in_the_time_step() -> None:
    """Shrink the error against a reference at step `h / 64` by a factor between 3.5 and 4.5 from `h` to `h / 2`."""
    step = 0.1
    reference = bgk_final(step / 64)
    coarse, fine = bgk_final(step), bgk_final(step / 2)
    ratio = np.abs(coarse - reference).max() / np.abs(fine - reference).max()
    assert 3.5 < ratio < 4.5


def test_prepare_rejects_a_problem_without_fluxes() -> None:
    """Raise a named error for a problem with no flux to split from its sources."""
    space = PhaseSpace(LatticeSpace((2,)), [AxisSpec('v', 4, -1.0, 1.0)])
    gas = AmountElement('gas', x=1.0, space=space)
    system = System(elements=[gas], interactions=[_relaxation()], solver=StrangSplittingSolver(dt=0.1))
    with pytest.raises(SolverError, match='no flux'):
        system.prepare().simulate(until=0.1)


def test_prepare_rejects_an_inner_solver_that_lacks_its_part() -> None:
    """Raise a named error for a transport solver that does not advance flux laws."""
    space = PhaseSpace(LatticeSpace((2,), discretization='finite-volume'), [AxisSpec('v', 4, -1.0, 1.0)])
    gas = AmountElement('gas', x=1.0, space=space)
    solver = StrangSplittingSolver(EulerMaruyamaSolver(dt=0.1), RK4Solver(dt=0.1), dt=0.1)
    system = System(elements=[gas], interactions=[_streaming()], solver=solver)
    with pytest.raises(SolverError):
        system.prepare().simulate(until=0.1)
