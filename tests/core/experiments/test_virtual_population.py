"""Test the `_virtual_population` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest
from scipy import stats

from skcomplex.core.errors import ExperimentError, ValidationError
from skcomplex.core.experiments import VirtualPopulationResult, virtual_population
from skcomplex.core.experiments._virtual_population import _get_inclusion
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.biology.oncology import ImmunoOncologyQspSystem

QSP_BOUNDS = {
    'growth': (0.005, 0.1),
    'kill': (1e-11, 5e-10),
    'recruitment': (0.002, 0.05),
    'treg_recruitment': (0.0005, 0.01),
}
BASELINE = 60.0


def log_ratio(trajectory):
    """Return the log ratio of effector to regulatory T cells at the end of a run."""
    return float(np.log(trajectory.get_values('effector_treg_ratio')[-1]))


def _get_patient():
    """Return a small tumour whose treatment starts at the baseline."""
    return ImmunoOncologyQspSystem(cancer=1e8, effector=1e6, treg=1e6, start=BASELINE)


def test_virtual_population_keeps_the_rows_inside_the_constraints():
    """Test that plausible logistic populations reach the range, with their share as the acceptance rate."""
    result = virtual_population(
        LogisticGrowthSystem(),
        {'r': (0.1, 1.0)},
        ['population.x'],
        {'population.x': (50.0, 100.0)},
        n=20,
        until=5.0,
        random_state=0,
    )
    assert isinstance(result, VirtualPopulationResult)
    assert list(result.plausible.columns) == ['r', 'population.x']
    assert result.plausible['population.x'].between(50.0, 100.0).all()
    assert result.acceptance == len(result.plausible) / 20
    assert result.selected is None
    assert result.inclusion is None


def test_virtual_population_names_the_constraint_that_rejects_most_rows():
    """Test that a population with no plausible row raises an error naming the strictest constraint."""
    with pytest.raises(ExperimentError, match=r"'population.x' in \(1000, 2000\) rejected 10 of 10"):
        virtual_population(
            LogisticGrowthSystem(),
            {'r': (0.1, 1.0)},
            ['population.x'],
            {'population.x': (1000.0, 2000.0)},
            n=10,
            until=5.0,
            random_state=0,
        )


@pytest.mark.parametrize(
    ('settings', 'match'),
    [
        ({'constraints': {'size': (0.0, 1.0)}}, 'names no output'),
        ({'constraints': {'population.x': (2.0, 1.0)}}, 'low < high'),
        ({'design': 'grid'}, 'Unknown design'),
        ({'target': {'a': [1.0], 'b': [2.0]}}, 'one output'),
        ({'target': {'size': [1.0, 2.0]}}, 'names no output'),
    ],
)
def test_virtual_population_rejects_bad_settings(settings, match):
    """Test that unknown outputs, bad ranges, grid designs, and targets of several outputs raise an error."""
    arguments = {'constraints': {'population.x': (0.0, 200.0)}, **settings}
    with pytest.raises(ValidationError, match=match):
        virtual_population(
            LogisticGrowthSystem(),
            {'r': (0.1, 1.0)},
            ['population.x'],
            n=10,
            until=5.0,
            random_state=0,
            **arguments,
        )


def test_virtual_population_needs_ten_plausible_rows_for_a_target():
    """Test that selection by inclusion with fewer than ten plausible rows raises an error."""
    with pytest.raises(ExperimentError, match='at least 10 plausible rows'):
        virtual_population(
            LogisticGrowthSystem(),
            {'r': (0.1, 1.0)},
            ['population.x'],
            {'population.x': (0.0, 200.0)},
            n=5,
            target={'population.x': [20.0, 30.0]},
            until=5.0,
            random_state=0,
        )


def test_get_inclusion_peaks_at_one_where_the_target_is_dense():
    """Test that the probability of inclusion is largest near the observed values and scaled to one."""
    values = np.linspace(-3.0, 3.0, 61)
    inclusion = _get_inclusion(np.random.default_rng(0).normal(1.0, 0.3, 200), values)
    assert inclusion.max() == 1.0
    assert abs(values[np.argmax(inclusion)] - 1.0) < 0.5


@pytest.mark.slow
def test_virtual_patients_have_a_plausible_baseline_diameter():
    """Test that plausible patients of the QSP system have a baseline diameter between 1 and 10 centimetres."""
    result = virtual_population(
        _get_patient(),
        QSP_BOUNDS,
        ['diameter', log_ratio],
        {'diameter': (1.0, 10.0)},
        n=500,
        until=BASELINE,
        random_state=0,
    )
    assert result.plausible['diameter'].between(1.0, 10.0).all()
    assert result.acceptance == len(result.plausible) / 500
    assert 0 < result.acceptance < 1


@pytest.mark.slow
def test_virtual_patients_match_observed_log_ratios():
    """Test that patients selected by inclusion match 100 observed log ratios within a KS statistic of 0.15."""
    target = np.random.default_rng(1).normal(1.5, 0.4, 100)
    result = virtual_population(
        _get_patient(),
        QSP_BOUNDS,
        ['diameter', log_ratio],
        {'diameter': (1.0, 10.0)},
        n=2000,
        target={'log_ratio': target},
        until=BASELINE,
        random_state=0,
    )
    assert result.inclusion.shape == (len(result.plausible),)
    assert stats.ks_2samp(result.selected['log_ratio'], target).statistic < 0.15
