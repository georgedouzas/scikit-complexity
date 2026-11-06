"""Test the `_urban_scaling` module of `skcomplex.systems.social.organizations`."""

import numpy as np
import pytest

from skcomplex.core.analysis import ScalingAnalysis
from skcomplex.core.experiments import Experiment, GridDesign
from skcomplex.core.specs import ObservableSpec
from skcomplex.systems.social.organizations import gravity_urban_scaling

POPULATIONS = [100, 200, 400, 800, 1600]


def _totals(alpha: float, gamma: float, seeds: list[int]) -> np.ndarray:
    """Return the total interaction of every population, averaged over seeds."""
    total = ObservableSpec('total_contacts', group='residents', variable='contacts', reduce='sum')
    bounds = {'population': POPULATIONS, 'alpha': [alpha], 'gamma': [gamma], 'random_state': seeds}
    experiment = Experiment(
        gravity_urban_scaling,
        GridDesign(),
        bounds,
        until=1.0,
        outputs=['total_contacts'],
        observables=[total],
    )
    series = experiment.prepare().run().result_.series('total_contacts')[:, 0]
    return series.reshape(len(POPULATIONS), len(seeds)).mean(axis=1)


def _exponent(totals: np.ndarray) -> float:
    """Return the exponent of the total interaction against the population."""
    scaling = ScalingAnalysis('power', parameters=POPULATIONS, critical=0.0, above=True)
    return float(np.ravel(scaling.prepare(totals).run().exponent_)[0])


@pytest.mark.parametrize(('alpha', 'gamma'), [(1.0, 1.0), (1.2, 0.5)])
def test_gravity_urban_scaling_exponent(alpha: float, gamma: float) -> None:
    """Test that the fitted exponent of the total interaction is within 0.05 of `2 - alpha gamma / 2`."""
    assert _exponent(_totals(alpha, gamma, [0])) == pytest.approx(2 - alpha * gamma / 2, abs=0.05)


@pytest.mark.slow
def test_gravity_urban_scaling_exponent_over_seeds() -> None:
    """Test that near `gamma = 2`, whose close pairs make single cities heavy-tailed, the exponent holds over seeds."""
    assert _exponent(_totals(0.8, 1.5, list(range(40)))) == pytest.approx(1.4, abs=0.05)
