"""Test the `VelocityVerletSolver` class."""

import time
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, oo, sqrt
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import ContinuousSpace, RadiusNeighborhood
from skcomplex.core.engine.solvers import (
    StepSpec,
    VelocityVerletSolver,
)
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class SoftSphereInteraction(Interaction):
    """Soft repulsion `k * (cutoff - r) ** 2 / 2` between the members of a group within the cutoff."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        cutoff: float = 1.0,
        k: Annotated[float, RealSet.open(0, oo)] = 100.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.cutoff = cutoff
        self.k = k

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the force of the neighbours within the cutoff on every member."""
        super()._prepare(context)
        s, x = context, context.target.x
        separation = x - s.target.other.x
        r = sqrt(separation.dot_product(separation))
        distance = SR.var('r', domain='positive')
        energy = s.k * (self.cutoff - distance) ** 2 / 2
        slope = energy.diff(distance).subs({distance: r})
        force = neighbors(-slope * separation / r, RadiusNeighborhood(self.cutoff))
        self.laws_ = [RateLaw(x.diff(s.t, 2), force)]


def test_a_free_ball_moves_at_its_constant_speed():
    """A ball without forces keeps its speed."""
    ball = Element('ball', values={'x': [0.0], "x'": [2.0]}, space=ContinuousSpace(n_dim=1))
    problem = System(elements=[ball]).prepare().simulate(until=0).problem
    stepper = VelocityVerletSolver(dt=0.1, random_state=0).prepare(problem)
    step = StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)
    assert stepper.run(step).state_.round(10).tolist() == [2.0, 2.0]


@pytest.mark.benchmark
def test_interacting_gas_of_100000_runs_in_real_time():
    """Test that 100000 soft spheres with velocity Verlet keep a real-time clock within 0.2 seconds."""
    n = 100_000
    size = float(np.sqrt(n))
    rng = np.random.default_rng(0)
    gas = Element('gas', values={'x': rng.uniform(0, size, (n, 2)), "x'": rng.normal(0, 1, (n, 2))}, size=n)
    system = System(
        elements=[gas],
        interactions=[SoftSphereInteraction('soft', target='gas', cutoff=1.0)],
        space=ContinuousSpace(extent=(size, size), boundary='reflecting'),
        solver=VelocityVerletSolver(dt=0.01),
    )
    problem = system.prepare().simulate(until=0).problem
    solver = VelocityVerletSolver(dt=0.01, until=2.0).prepare(problem)
    state, lags = solver.initial_state_, []
    start = time.perf_counter()
    for frame in range(200):
        step = StepSpec(frame * 0.01, state, problem.initial_parameters, (frame + 1) * 0.01)
        state = solver.run(step, final=frame == 199).state_
        lags.append(time.perf_counter() - start - (frame + 1) * 0.01)
    assert max(lags) <= 0.2
