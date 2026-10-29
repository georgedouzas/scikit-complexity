"""Test the `_pair_potential` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, var, vector
from sage.sets.real_set import RealSet

from skcomplex.core.elements import GroupElement, ParticleElement
from skcomplex.core.interactions import PairPotentialInteraction
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.specs import ParameterSpec


class SoftSphereInteraction(PairPotentialInteraction):
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
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 50.0,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position, period=period)
        self.k = k

    def _get_pair_energy(self, s, r):
        """Return the soft repulsion energy."""
        return s.k * (s.cutoff - r) ** 2 / 2


def test_pair_forces_are_minus_the_derivative_of_the_energy_and_opposite():
    """Test that the forces on the two particles of a pair are opposite and push them apart."""
    x1, x2, v1, v2, k, cutoff = var('x1 x2 v1 v2 k cutoff')
    first = SimpleNamespace(x=vector([x1]), v=vector([v1]), m=1)
    second = SimpleNamespace(x=vector([x2]), v=vector([v2]), m=2)
    interaction = SoftSphereInteraction('soft', 'gas', 'gas').check().prepare()
    push, pull = interaction.run(SimpleNamespace(first=first, second=second, k=k, cutoff=cutoff)).laws_
    values = {x1: 0.0, x2: 0.5, k: 2.0, cutoff: 1.0}
    assert (push.target, pull.target) == (first.v, second.v)
    assert list(push.value.subs(values)) == [-1.0]
    assert list(pull.value.subs(values)) == [0.5]


@pytest.mark.slow
def test_pair_interaction_conserves_momentum():
    """Test that soft-sphere collisions inside an isolated group keep the total momentum."""
    rng = np.random.default_rng(0)
    values = {'x': rng.uniform(0, 3, (200, 2)), 'v': rng.normal(0, 1, (200, 2))}
    system = System(
        elements=[GroupElement('gas', element=ParticleElement, size=200, values=values)],
        interactions=[SoftSphereInteraction('soft', first='gas', second='gas', cutoff=0.2)],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = (
        Simulation(system, solver=ScipyODESolver(rtol=1e-9, atol=1e-12), dt=0.05).check().run(until=2).trajectory_
    )
    momentum = trajectory.values('gas.v').sum(axis=1)
    assert np.allclose(momentum, momentum[0], atol=1e-8)
