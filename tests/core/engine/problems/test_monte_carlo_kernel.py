"""Test the `_monte_carlo_kernel` module of `skcomplex.core.engine.problems`."""

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.systems import System


def _driven(s):
    """Return the objective of a spin in a field that turns from down to up at `t = 1.5`, over the temperature."""
    return [ObjectiveLaw(-s.target.s * (s.t - 1.5) / s.target.T, s.target.s)]


def test_monte_carlo_kernel_objective_reads_the_time():
    """Test that a field that turns up at `t = 1.5` holds every spin down at first and up afterwards."""
    ring = Element('ring', values={'s': np.ones(16), 'T': 0.01}, space=LatticeSpace(16))
    field = Interaction('field', roles={'target': 'ring'}, laws=_driven)
    solver = MetropolisSolver(moves={'ring.s': 'flip'})
    system = System(elements=[ring], interactions=[field], solver=solver).prepare()
    spins = system.simulate(until=3, random_state=0).get_values('ring.s')
    assert np.all(spins[1] == -1.0)
    assert np.all(spins[3] == 1.0)
