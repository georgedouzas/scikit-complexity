"""Test the `_logistic_growth` module of `skcomplex.systems.biology.common.populations`."""

import numpy as np
import pytest

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.common.kinetics import CompartmentElement
from skcomplex.systems.biology.common.populations import LogisticGrowthInteraction, PopulationElement


def _run(interaction, n0, until=4.0, dt=0.01):
    """Return the times and the population of one compartment under one interaction."""
    system = CompositeSystem(elements=[CompartmentElement('population', x=n0)], interactions=[interaction])
    trajectory = simulate(system, dt=dt, until=until)
    return np.asarray(trajectory.times), np.asarray(trajectory.values('population.x'))


def test_logistic_growth_follows_its_closed_solution():
    """Test that Verhulst's logistic growth follows its sigmoid towards the capacity."""
    interaction = LogisticGrowthInteraction('growth', target='population', r=1.0, K=100.0)
    times, population = _run(interaction, 2.0)
    np.testing.assert_allclose(population, 100.0 / (1 + 49.0 * np.exp(-times)), rtol=1e-4)


def test_logistic_growth_has_no_exponent_parameter():
    """Test that Verhulst's logistic takes the rate, the capacity, and the delay as parameters."""
    params = LogisticGrowthInteraction('growth', target='population').get_params()
    assert {'r', 'K', 'tau'} <= set(params)
    assert 'p' not in params


def test_logistic_growth_saturates():
    """Test that logistic growth approaches the carrying capacity."""
    system = CompositeSystem(
        elements=[PopulationElement('p', x=5)],
        interactions=[LogisticGrowthInteraction('g', target='p', r=1.0, K=200)],
    )
    assert simulate(system, dt=1.0, until=30).values('p.x')[-1] == pytest.approx(200, rel=1e-3)
