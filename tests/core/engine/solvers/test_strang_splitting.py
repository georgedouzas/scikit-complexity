"""Test the Strang splitting of transport and collisions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

from skcomplex.core.components.elements import DensityElement, FieldElement
from skcomplex.core.components.interactions import AdvectionInteraction, RelaxationInteraction
from skcomplex.core.components.laws.utils import coordinate, maximum_entropy
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver, RK4Solver, SemiLagrangianSolver, StrangSplittingSolver
from skcomplex.core.errors import SolverError
from skcomplex.core.specs import AxisSpec


def equilibrium(namespace: SimpleNamespace) -> object:
    """Return the maximum-entropy equilibrium of the field."""
    return maximum_entropy(namespace.target.f, axes=('v',))


def bgk_final(step: float) -> npt.NDArray[np.float64]:
    """Return the final state of a BGK field run with split steps of a length."""
    size, length = 16, 2 * np.pi
    space = PhaseSpace(LatticeSpace((size,), spacing=length / size), [AxisSpec('v', 32, -8.0, 8.0)]).prepare()
    v, x = space.coordinates_['v'], np.arange(size) * length / size
    start = np.outer(1.0 + 0.5 * np.sin(x), np.exp(-(v**2) / 2)) + np.outer(
        1.0 + 0.5 * np.cos(x),
        np.exp(-((v - 1) ** 2)),
    )
    gas = FieldElement('gas', element=DensityElement, space=space, values={'f': start}, discretization='finite-volume')
    stream = AdvectionInteraction('s', target='gas', state='f', drift=lambda _: [coordinate('v'), 0])
    relax = RelaxationInteraction('r', target='gas', state='f', goal=equilibrium, time=0.5)
    solver = StrangSplittingSolver(SemiLagrangianSolver(dt=step), RK4Solver(dt=step), dt=step)
    system = System(elements=[gas], interactions=[stream, relax])
    return simulate(system, until=1.0, dt=1.0, solver=solver).values('gas.f')[-1]


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
    gas = FieldElement('gas', element=DensityElement, space=space, values={'f': 1.0})
    relax = RelaxationInteraction('r', target='gas', state='f', goal=equilibrium, time=0.5)
    with pytest.raises(SolverError, match='no flux'):
        simulate(System(elements=[gas], interactions=[relax]), until=0.1, solver=StrangSplittingSolver(dt=0.1))


def test_prepare_rejects_an_inner_solver_that_lacks_its_part() -> None:
    """Raise a named error for a transport solver that does not advance flux laws."""
    space = PhaseSpace(LatticeSpace((2,)), [AxisSpec('v', 4, -1.0, 1.0)])
    gas = FieldElement('gas', element=DensityElement, space=space, values={'f': 1.0}, discretization='finite-volume')
    stream = AdvectionInteraction('s', target='gas', state='f', drift=lambda _: [coordinate('v'), 0])
    solver = StrangSplittingSolver(EulerMaruyamaSolver(dt=0.1), RK4Solver(dt=0.1), dt=0.1)
    with pytest.raises(SolverError):
        simulate(System(elements=[gas], interactions=[stream]), until=0.1, solver=solver)
