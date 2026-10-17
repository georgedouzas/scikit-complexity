"""Benchmark the Numba backend against the NumPy backend and real time."""

import time

import numpy as np
import pytest

from skcomplex.compile import compile_system
from skcomplex.core import Domain, parameter
from skcomplex.model import EuclideanSpace, System
from skcomplex.models.physics import PairPotential, Particle, ReflectingBox, ideal_gas
from skcomplex.runtime import RealTime, Simulation
from skcomplex.solvers import VelocityVerlet


class SoftSphere(PairPotential):
    """Soft repulsion `k * (cutoff - r) ** 2 / 2`."""

    k: float = parameter(100.0, domain=Domain.POSITIVE)

    def pair_energy(self, s, r):
        """Return the soft repulsion energy."""
        return s.k * (s.cutoff - r) ** 2 / 2


def _step_time(problem, repeats=20):
    """Return the mean time of one drift and boundary evaluation after a warm-up call."""
    x, p = problem.initial_state, problem.initial_parameters
    problem.drift(0.0, x, p)
    problem.boundary(0.0, x, p)
    start = time.perf_counter()
    for _ in range(repeats):
        problem.drift(0.0, x, p)
        problem.boundary(0.0, x, p)
    return (time.perf_counter() - start) / repeats


@pytest.mark.benchmark
def test_numba_is_three_times_faster_on_million_member_gas():
    """Test that one step of a million-member gas takes Numba at most a third of the NumPy time."""
    gas = ideal_gas(1_000_000, random_state=0)
    numpy_time = _step_time(compile_system(gas, backend='numpy'))
    numba_time = _step_time(compile_system(gas, backend='numba'))
    assert numba_time <= numpy_time / 3


@pytest.mark.benchmark
def test_interacting_gas_of_100000_runs_in_real_time():
    """Test that 100000 soft spheres with velocity Verlet keep a real-time clock within 0.2 seconds."""
    n = 100_000
    size = float(np.sqrt(n))
    rng = np.random.default_rng(0)
    system = System(
        elements=[Particle.group('gas', n, x=rng.uniform(0, size, (n, 2)), v=rng.normal(0, 1, (n, 2)))],
        interactions=[
            SoftSphere('soft', first='gas', second='gas', cutoff=1.0),
            ReflectingBox('box', target='gas', size=size),
        ],
        space=EuclideanSpace(2),
    )
    simulation = Simulation(system, solver=VelocityVerlet(dt=0.01), dt=0.01, clock=RealTime(speed=1.0), record=False)
    simulation.run(until=2)
    assert simulation.clock_.max_lag_ <= 0.2
