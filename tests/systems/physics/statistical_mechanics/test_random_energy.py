"""Test the `_random_energy` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.experiments import sweep
from skcomplex.systems.physics.statistical_mechanics import RandomEnergySystem


def _ensemble(**parameters: object) -> CompositeSystem:
    """Return the parts of a random energy system for its model parameters, as an experiment runs them."""
    return RandomEnergySystem(**parameters).prepare().system_


@pytest.mark.slow
def test_random_energy_system_glass_transition() -> None:
    """Test that the entropy over 10 seeds follows the theory above the glass temperature and freezes below it."""
    temperatures = [2.0, 1.0, 0.8, 0.3]
    bounds = {'n_spins': [20], 'temperature': temperatures, 'random_state': list(range(10))}
    result = sweep(_ensemble, bounds, until=1.0, outputs=['ensemble.entropy'])
    entropies = result.series('ensemble.entropy')[:, 0]
    mean = entropies.reshape(len(temperatures), 10).mean(axis=1)
    theory = np.log(2) - 1 / (4 * np.array(temperatures[:3]) ** 2)
    assert np.all(np.abs(mean[:3] - theory) < [0.02, 0.02, 0.05])
    assert mean[3] < 0.1
