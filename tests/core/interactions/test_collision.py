"""Test the binary collision integral of a density over the class axes of a phase space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
import pytest

from skcomplex.core.elements import DensityElement, FieldElement
from skcomplex.core.interactions import CollisionInteraction
from skcomplex.core.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import PassCompiler
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.errors import ValidationError
from skcomplex.specs import AxisSpec

SIZE, HALF_WIDTH = 32, 8.0


def maxwell(_speed: object, _cosine: object) -> float:
    """Return the cross-section of Maxwell molecules in two velocity dimensions."""
    return 1 / (2 * np.pi)


def bkw(time: float, vx: npt.NDArray[np.float64], vy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the Bobylev-Krook-Wu solution of Maxwell molecules in two velocity dimensions."""
    scale = 1 - np.exp(-time / 8) / 2
    squared = vx**2 + vy**2
    return (
        np.exp(-squared / (2 * scale)) / (2 * np.pi * scale**2) * (2 * scale - 1 + (1 - scale) / (2 * scale) * squared)
    )


def velocity_space() -> tuple[PhaseSpace, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a phase space of one position and two velocity axes, with the velocity grids."""
    axes = [AxisSpec('vx', SIZE, -HALF_WIDTH, HALF_WIDTH), AxisSpec('vy', SIZE, -HALF_WIDTH, HALF_WIDTH)]
    space = PhaseSpace(LatticeSpace((1,)), axes).check().prepare()
    vx, vy = np.meshgrid(space.coordinates_['vx'], space.coordinates_['vy'], indexing='ij')
    return space, vx, vy


def test_check_rejects_values_it_cannot_take() -> None:
    """Raise named errors for a state that is no name and functions that are not callable."""
    with pytest.raises(ValidationError, match='no name'):
        CollisionInteraction('c', target='gas', state='1f', cross_section=maxwell).check()
    with pytest.raises(ValidationError, match='cross-section'):
        CollisionInteraction('c', target='gas', cross_section=1.0).check()
    with pytest.raises(ValidationError, match='factor'):
        CollisionInteraction('c', target='gas', cross_section=maxwell, factor=2.0).check()


def test_bkw_solution_within_1e_4_and_density_constant() -> None:
    """Match the exact BKW solution within 1e-4 at time 5, and keep the density within 1e-12."""
    space, vx, vy = velocity_space()
    start = bkw(0.0, vx, vy)
    gas = FieldElement('gas', element=DensityElement, space=space, values={'f': start[None]})
    collide = CollisionInteraction('c', target='gas', cross_section=maxwell)
    simulation = Simulation(System(elements=[gas], interactions=[collide]), solver=RK4Solver(dt=0.1)).check().prepare()
    final = simulation.run(until=5.0).trajectory_.values('gas.f')[-1].reshape(SIZE, SIZE)
    assert np.abs(final - bkw(5.0, vx, vy)).max() < 1e-4
    assert abs(final.sum() / start.sum() - 1.0) < 1e-12


def test_density_factor_speeds_relaxation_twice() -> None:
    """Reach every intermediate state in half the time, within 1 percent, with a density factor of 2."""
    space, vx, vy = velocity_space()
    start = bkw(0.0, vx, vy)
    gas = FieldElement('gas', element=DensityElement, space=space, values={'f': start[None]})
    plain = CollisionInteraction('c', target='gas', cross_section=maxwell)
    faster = CollisionInteraction('c', target='gas', cross_section=maxwell, factor=lambda _: 2.0)
    slow = Simulation(System(elements=[gas], interactions=[plain]), solver=RK4Solver(dt=0.1)).check().prepare()
    fast = Simulation(System(elements=[gas], interactions=[faster]), solver=RK4Solver(dt=0.05)).check().prepare()
    first = slow.run(until=2.0).trajectory_.values('gas.f')[-1]
    second = fast.run(until=1.0).trajectory_.values('gas.f')[-1]
    assert np.abs(first - second).max() / np.abs(first - start.ravel()).max() < 0.01


def test_collision_without_velocity_axes_is_a_law_space_issue() -> None:
    """Report a collision on a lattice without class axes, with the fix of an interval `AxisSpec`."""
    gas = FieldElement('gas', element=DensityElement, space=LatticeSpace((4,)))
    collide = CollisionInteraction('c', target='gas', cross_section=maxwell)
    report = PassCompiler().check().prepare(System(elements=[gas], interactions=[collide])).report_
    issues = [issue for issue in report.issues if issue.code == 'law-space']
    assert issues
    assert 'AxisSpec' in issues[0].fix


def test_collision_on_periodic_axis_is_a_law_space_issue() -> None:
    """Report a collision on a phase space whose class axes are not two intervals."""
    axes = [AxisSpec('vx', 8, -1.0, 1.0), AxisSpec('vy', 8, 0.0, 6.0, periodic=True)]
    gas = FieldElement('gas', element=DensityElement, space=PhaseSpace(LatticeSpace((1,)), axes))
    collide = CollisionInteraction('c', target='gas', cross_section=maxwell)
    report = PassCompiler().check().prepare(System(elements=[gas], interactions=[collide])).report_
    assert any(issue.code == 'law-space' for issue in report.issues)
