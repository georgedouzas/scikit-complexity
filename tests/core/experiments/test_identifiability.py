"""Test the `_identifiability` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import IdentifiabilityResult, identifiability
from skcomplex.core.experiments._identifiability import _get_interval, _get_verdict
from skcomplex.systems.biology.ecology import LogisticGrowthSystem

BOUNDS = {'r': (0.1, 2.0), 'K': (50.0, 500.0)}
NOISE = 1 + 0.02 * np.random.default_rng(0).standard_normal(30)
SATURATING = np.linspace(0.0, 20.0, 30)
EARLY = np.linspace(0.0, 3.0, 30)


def _get_saturating_data():
    """Return 30 logistic counts with 2% noise that reach the capacity 100."""
    return {'population.x': (SATURATING, 100 / (1 + 9 * np.exp(-0.5 * SATURATING)) * NOISE)}


def test_identifiability_rejects_an_unknown_method():
    """Test that an unknown method raises an error naming both methods."""
    with pytest.raises(ValidationError, match=r"'profile'.*'refits'"):
        identifiability(LogisticGrowthSystem(), _get_saturating_data(), BOUNDS, method='fisher')


@pytest.mark.parametrize(
    ('settings', 'match'),
    [({'level': 1.5}, 'level'), ({'noise': (0.0,)}, 'noise'), ({'n': 0}, 'n = 0')],
)
def test_identifiability_rejects_bad_settings(settings, match):
    """Test that a level outside (0, 1), a noise level that is not positive, or a bad count raise an error."""
    with pytest.raises(ValidationError, match=match):
        identifiability(LogisticGrowthSystem(), _get_saturating_data(), BOUNDS, **settings)


def test_get_interval_interpolates_the_crossings_and_leaves_open_sides():
    """Test that the interval ends where the ratio crosses the threshold, and is open where it never does."""
    values = np.asarray([3.0, 0.0, 1.0, 2.0, -1.0])
    ratios = np.asarray([0.0, 0.5, 1.0, 0.5, 0.0])
    low, high = _get_interval(values, ratios, 0.25)
    assert low == pytest.approx(-0.5)
    assert high == pytest.approx(2.5)
    low, high = _get_interval(values[:-1], ratios[:-1], 0.25)
    assert np.isnan(low)
    assert high == pytest.approx(2.5)


def test_get_verdict_follows_the_thresholds_of_the_level():
    """Test that errors up to the level are strong, up to ten times it weak, and above none."""
    assert [_get_verdict(error, 0.01) for error in (0.005, 0.05, 0.2)] == ['strong', 'weak', 'none']


@pytest.mark.slow
def test_profile_of_saturating_data_identifies_both_parameters():
    """Test that profiles of data through saturation find both parameters, with 95% intervals that hold the truth."""
    result = identifiability(LogisticGrowthSystem(), _get_saturating_data(), BOUNDS, random_state=0)
    assert isinstance(result, IdentifiabilityResult)
    assert result.verdicts == {'r': 'identifiable', 'K': 'identifiable'}
    for name, truth in (('r', 0.5), ('K', 100.0)):
        low, high = result.intervals[name]
        assert low < truth < high
        values, ratios = result.profiles[name]
        assert values.shape == ratios.shape
    assert result.errors is None


@pytest.mark.slow
def test_profile_of_early_data_leaves_the_capacity_open():
    """Test that data below a tenth of the capacity leave it not identifiable, its interval open at the top."""
    data = {'population.x': (EARLY, 100 / (1 + 99 * np.exp(-0.5 * EARLY)) * NOISE)}
    result = identifiability(LogisticGrowthSystem(n=1.0), data, BOUNDS, random_state=0)
    assert result.verdicts['K'] == 'not identifiable'
    assert np.isnan(result.intervals['K'][1])


@pytest.mark.slow
def test_refits_of_saturating_data_find_both_parameters():
    """Test that 100 refits at the noise level 0.01 call both parameters strongly or weakly identifiable."""
    result = identifiability(
        LogisticGrowthSystem(),
        _get_saturating_data(),
        BOUNDS,
        method='refits',
        noise=(0.01,),
        n=100,
        random_state=0,
    )
    assert set(result.verdicts.values()) <= {'strong', 'weak'}
    assert set(result.errors['r']) == {0.01}
    assert result.profiles is None
    assert result.intervals is None


@pytest.mark.slow
def test_refits_with_a_seed_are_reproducible():
    """Test that two refits with the seed 3 give the same relative errors, also on threads."""
    settings = {'method': 'refits', 'noise': (0.01, 0.05), 'n': 5, 'random_state': 3}
    first = identifiability(LogisticGrowthSystem(), _get_saturating_data(), BOUNDS, **settings)
    second = identifiability(LogisticGrowthSystem(), _get_saturating_data(), BOUNDS, executor='thread', **settings)
    assert first.errors == second.errors
