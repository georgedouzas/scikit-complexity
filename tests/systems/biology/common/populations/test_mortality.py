"""Test the `_mortality` module of `skcomplex.systems.biology.common.populations`."""

import numpy as np

from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.common.kinetics import CompartmentElement
from skcomplex.systems.biology.common.populations import MortalityInteraction


def _run(interaction, n0, until=4.0, dt=0.01):
    """Return the times and the population of one compartment under one interaction."""
    system = System(elements=[CompartmentElement('population', x=n0)], interactions=[interaction])
    trajectory = simulate(system, dt=dt, until=until)
    return np.asarray(trajectory.times), np.asarray(trajectory.values('population.x'))


def test_mortality_decays_exponentially():
    """Test that mortality follows `n0 exp(-d t)`."""
    times, population = _run(MortalityInteraction('death', target='population', d=0.4), 5.0)
    np.testing.assert_allclose(population, 5.0 * np.exp(-0.4 * times), rtol=1e-4)
