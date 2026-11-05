"""Test the `_speciation` module of `skcomplex.models.biology.ecology`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.biology.ecology import neutral_biodiversity


def test_neutral_biodiversity_species_number_matches_ewens_sum():
    """Test that the mean number of species matches S_N = sum_n theta / (theta + n) within 5%."""
    size, nu = 100, 0.02
    theta = (size - 1) * nu / (1 - nu)
    trajectory = (
        Simulation(
            built := neutral_biodiversity(size, nu=nu, random_state=0),
            random_state=0,
            observables=describe('neutral_biodiversity').observables_of(built),
        )
        .prepare()
        .run(until=3000)
        .trajectory_
    )
    expected = np.sum(theta / (theta + np.arange(size)))
    assert trajectory.values('richness')[500:].mean() == pytest.approx(expected, rel=0.05)
