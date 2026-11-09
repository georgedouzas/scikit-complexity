"""Test the `_exponential_growth` module of `skcomplex.systems.biology.common.populations`."""

import numpy as np

from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.common.kinetics import CompartmentElement
from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction


def _run(interaction, n0, until=4.0, dt=0.01):
    """Return the times and the population of one compartment under one interaction."""
    system = System(elements=[CompartmentElement('population', x=n0)], interactions=[interaction])
    trajectory = simulate(system, dt=dt, until=until)
    return np.asarray(trajectory.times), np.asarray(trajectory.values('population.x'))


def test_exponential_growth_follows_its_closed_solution():
    """Test that exponential growth follows `n0 exp(r t)`."""
    times, population = _run(ExponentialGrowthInteraction('growth', target='population', r=0.3), 2.0)
    np.testing.assert_allclose(population, 2.0 * np.exp(0.3 * times), rtol=1e-4)
