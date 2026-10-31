"""Test the `_gompertz_growth` module of `skcomplex.models.biology.common.populations`."""

import numpy as np

from skcomplex.core.elements import CompartmentElement
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.common.populations import GompertzGrowthInteraction


def _run(interaction, n0, until=4.0, dt=0.01):
    """Return the times and the population of one compartment under one interaction."""
    system = System(elements=[CompartmentElement('population', n=n0)], interactions=[interaction])
    trajectory = Simulation(system, dt=dt).check().prepare().run(until=until).trajectory_
    return np.asarray(trajectory.times), np.asarray(trajectory.values('population.n'))


def test_gompertz_growth_follows_its_closed_solution():
    """Test that Gompertz growth follows `K exp(log(n0 / K) exp(-r t))`."""
    times, population = _run(GompertzGrowthInteraction('growth', target='population', r=0.5, K=100.0), 2.0)
    np.testing.assert_allclose(population, 100.0 * np.exp(np.log(0.02) * np.exp(-0.5 * times)), rtol=1e-4)
