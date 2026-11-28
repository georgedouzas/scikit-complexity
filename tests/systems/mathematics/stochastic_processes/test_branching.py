"""Test the `_branching` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems.mathematics.stochastic_processes import (
    GaltonWatsonSystem,
)

from ..conftest import _run


def test_galton_watson_grows_geometrically_on_average():
    """Test that the mean generation size is `Z0 m^n`, and that subcritical processes die out."""
    sizes = [
        _run(GaltonWatsonSystem(50, m=1.2), 5, random_state=seed).get_values('population.count')[-1]
        for seed in range(30)
    ]
    assert np.mean(sizes) == pytest.approx(50 * 1.2**5, rel=0.15)
    assert _run(GaltonWatsonSystem(10, m=0.5), 30).get_values('population.count')[-1] == 0


def test_galton_watson_geometric_offspring_scale_by_their_mean():
    """Test that geometric offspring of mean 0.8 give generation 3 a mean of 0.8 cubed."""
    for individuals in (True, False):
        system = GaltonWatsonSystem(200, m=0.8, offspring='geometric', individuals=individuals).prepare()
        sizes = [system.simulate(3, random_state=seed).get_values('population.count')[-1] for seed in range(100)]
        assert np.mean(sizes) / 200 == pytest.approx(0.8**3, rel=0.1)


def test_galton_watson_finite_offspring_have_their_mean():
    """Test that the offspring probabilities `[0.25, 0.5, 0.25]` have the mean 1."""
    for individuals in (True, False):
        system = GaltonWatsonSystem(1, offspring=[0.25, 0.5, 0.25], individuals=individuals).prepare()
        sizes = [system.simulate(1, random_state=seed).get_values('population.count')[-1] for seed in range(800)]
        assert np.mean(sizes) == pytest.approx(1, abs=0.07)


def test_galton_watson_counts_match_the_individuals():
    """Test that counts have the mean sizes of the individuals and the mean `m^n`."""
    counts = GaltonWatsonSystem(1, m=2.5, individuals=False).prepare()
    individuals = GaltonWatsonSystem(1, m=2.5).prepare()
    by_count = [counts.simulate(5, random_state=seed).get_values('population.count')[-1] for seed in range(200)]
    by_agent = [individuals.simulate(5, random_state=seed).get_values('population.count')[-1] for seed in range(200)]
    assert np.mean(by_count) == pytest.approx(np.mean(by_agent), rel=0.2)
    sizes = [counts.simulate(12, random_state=seed).get_values('population.count')[-1] for seed in range(400)]
    assert np.mean(sizes) == pytest.approx(2.5**12, rel=0.15)


def test_galton_watson_rejects_offspring_that_are_no_law():
    """Test that an unknown law and probabilities not summing to 1 raise an error."""
    with pytest.raises(ValidationError, match='Unknown offspring law'):
        GaltonWatsonSystem(offspring='binomial').prepare()
    with pytest.raises(ValidationError, match='sum to 1'):
        GaltonWatsonSystem(offspring=[0.5, 0.2], individuals=False).prepare()
