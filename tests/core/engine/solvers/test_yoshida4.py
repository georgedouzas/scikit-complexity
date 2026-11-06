"""Test the `Yoshida4Solver` class."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import HamiltonianInteraction, RateInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import Yoshida4Solver
from skcomplex.core.errors import SolverError
from skcomplex.core.specs import StateSpec


class PlaneElement(Element):
    """Element with two canonical pairs of order 1."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        y: Annotated[object, StateSpec()] = 0.0,
        px: Annotated[object, StateSpec()] = 0.0,
        py: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.px = px
        self.py = py


def _oscillator() -> System:
    """Return a harmonic oscillator in the plane written with a Hamiltonian."""
    energy = HamiltonianInteraction(
        'energy',
        target='mass',
        coordinates=('x', 'y'),
        momenta=('px', 'py'),
        hamiltonian=lambda s: (s.target.px**2 + s.target.py**2 + s.target.x**2 + s.target.y**2) / 2,
    )
    return System(elements=[PlaneElement('mass')], interactions=[energy])


def _final_error(backend: type, dt: float) -> float:
    """Return the error of the oscillator at time 10 for an internal step."""
    simulation = Simulation(_oscillator(), backend=backend(), solver=Yoshida4Solver(dt=dt), dt=10.0)
    trajectory = simulation.prepare().run(until=10.0).trajectory_
    return float(abs(trajectory.values('mass.x')[-1] - np.cos(10.0)))


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend], ids=['numpy', 'numba'])
def test_run_converges_at_fourth_order(backend):
    """Test that halving the step shrinks the error of a Hamiltonian oscillator by a factor between 14 and 18."""
    ratio = _final_error(backend, 0.1) / _final_error(backend, 0.05)
    assert 14 < ratio < 18


def test_run_bounds_the_energy_of_a_long_run():
    """Test that the energy of the oscillator drifts by less than 1e-6 over a thousand periods."""
    simulation = Simulation(_oscillator(), solver=Yoshida4Solver(dt=0.05), dt=100.0)
    trajectory = simulation.prepare().run(until=6000.0).trajectory_
    energy = sum(trajectory.values(f'mass.{name}') ** 2 for name in ('x', 'y', 'px', 'py')) / 2
    assert np.ptp(energy) < 1e-6


def _rotating_energy(s):
    """Return the Hamiltonian of a mass in a rotating frame with a cubic potential."""
    kinetic = (s.target.px**2 + s.target.py**2) / 2
    return kinetic + s.target.py * s.target.x - s.target.px * s.target.y + s.target.x**2 * s.target.y


def test_prepare_rejects_a_hamiltonian_that_couples_momenta():
    """Test that a Coriolis term with a potential, whose momentum rates read momenta, raises an error with a fix."""
    rotating = HamiltonianInteraction(
        'rotating',
        target='mass',
        coordinates=('x', 'y'),
        momenta=('px', 'py'),
        hamiltonian=_rotating_energy,
    )
    system = System(elements=[PlaneElement('mass')], interactions=[rotating])
    with pytest.raises(SolverError, match=r'rate of mass\.p[xy] reads .*mass\.p[xy].*RK4Solver'):
        Yoshida4Solver(dt=0.1).prepare(PassCompiler().prepare(system).run().problem_)


def test_prepare_rejects_a_state_that_reads_itself():
    """Test that a damping rate, which reads its own state, admits no split."""
    damping = RateInteraction('damping', target='mass', state='px', rate=lambda s: -s.target.px)
    system = _oscillator()
    system = System(elements=system.elements, interactions=[*system.interactions, damping])
    with pytest.raises(SolverError, match=r'rate of mass\.px reads mass\.px'):
        Yoshida4Solver(dt=0.1).prepare(PassCompiler().prepare(system).run().problem_)
