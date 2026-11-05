"""Test the `_power_law_growth` module of `skcomplex.models.biology.common.populations`."""

import numpy as np

from skcomplex.core.elements import CompartmentElement
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.common.populations import PowerLawGrowthInteraction


def _run(interaction, n0, until=4.0, dt=0.01):
    """Return the times and the population of one compartment under one interaction."""
    system = System(elements=[CompartmentElement('population', n=n0)], interactions=[interaction])
    trajectory = Simulation(system, dt=dt).prepare().run(until=until).trajectory_
    return np.asarray(trajectory.times), np.asarray(trajectory.values('population.n'))


def test_power_law_growth_follows_its_closed_solution():
    """Test that power-law growth with the exponent 1/2 follows `(sqrt(n0) + r t / 2) ** 2`."""
    times, population = _run(PowerLawGrowthInteraction('growth', target='population', r=0.5, p=0.5), 2.0)
    np.testing.assert_allclose(population, (np.sqrt(2.0) + 0.25 * times) ** 2, rtol=1e-4)
