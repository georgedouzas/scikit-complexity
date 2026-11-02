"""Test the `_mortality` module of `skcomplex.models.biology.common.populations`."""

import numpy as np

from skcomplex.core.elements import CompartmentElement
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.common.populations import MortalityInteraction


def _run(interaction, n0, until=4.0, dt=0.01):
    """Return the times and the population of one compartment under one interaction."""
    system = System(elements=[CompartmentElement('population', n=n0)], interactions=[interaction])
    trajectory = Simulation(system, dt=dt).check().prepare().run(until=until).trajectory_
    return np.asarray(trajectory.times), np.asarray(trajectory.values('population.n'))


def test_mortality_decays_exponentially():
    """Test that mortality follows `n0 exp(-d t)`."""
    times, population = _run(MortalityInteraction('death', target='population', d=0.4), 5.0)
    np.testing.assert_allclose(population, 5.0 * np.exp(-0.4 * times), rtol=1e-4)
