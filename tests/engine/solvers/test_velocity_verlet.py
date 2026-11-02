"""Test the `VelocityVerletSolver` class."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.elements import GroupElement, PointElement
from skcomplex.core.interactions import PairCouplingInteraction
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import RealTimeClock, Simulation
from skcomplex.engine.solvers import (
    VelocityVerletSolver,
)
from skcomplex.specs import ParameterSpec


class SoftSphereInteraction(PairCouplingInteraction):
    """Soft repulsion `k * (cutoff - r) ** 2 / 2`."""

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        period: float | None = None,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            potential=self._get_pair_energy,
            period=period,
        )
        self.k = k

    def _get_pair_energy(self, s, r):
        """Return the soft repulsion energy."""
        return s.k * (s.cutoff - r) ** 2 / 2


@pytest.mark.benchmark
def test_interacting_gas_of_100000_runs_in_real_time():
    """Test that 100000 soft spheres with velocity Verlet keep a real-time clock within 0.2 seconds."""
    n = 100_000
    size = float(np.sqrt(n))
    rng = np.random.default_rng(0)
    system = System(
        elements=[
            GroupElement(
                'gas',
                element=PointElement,
                size=n,
                values={'x': rng.uniform(0, size, (n, 2)), 'x_d1': rng.normal(0, 1, (n, 2))},
            ),
        ],
        interactions=[
            SoftSphereInteraction('soft', first='gas', second='gas', cutoff=1.0),
        ],
        space=ContinuousSpace(extent=(size, size), boundary='reflecting'),
    )
    simulation = (
        Simulation(
            system,
            solver=VelocityVerletSolver(dt=0.01),
            dt=0.01,
            clock=RealTimeClock(speed=1.0),
            record=False,
        )
        .check()
        .prepare()
    )
    simulation.run(until=2)
    assert simulation.clock_.max_lag_ <= 0.2
