"""Test the `_sensitivity` module of `skcomplex.core.experiments`."""

import math

import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components import Element
from skcomplex.core.experiments import (
    sensitivity,
)
from skcomplex.core.system import Observable, System

Point = Element.define('Point', parameters={'x1': 0.0, 'x2': 0.0, 'x3': 0.0})


def _ishigami(s):
    """Return the Ishigami function of the point's coordinates."""
    return sin(s.p.x1) + 7 * sin(s.p.x2) ** 2 + 0.1 * s.p.x3**4 * sin(s.p.x1)


def test_sobol_indices_of_ishigami():
    """Test that Sobol first-order indices of the Ishigami function match the known values within 0.05."""
    system = System(components=[Point('p')])
    bounds = {f'p.x{index}': (-math.pi, math.pi) for index in (1, 2, 3)}
    observables = [Observable('y', _ishigami)]
    result = sensitivity(system, bounds, 'y', method='sobol', n=2048, until=0, random_state=0, observables=observables)
    expected = {'p.x1': 0.3139, 'p.x2': 0.4424, 'p.x3': 0.0}
    for name, value in expected.items():
        assert result.indices[name]['S1'] == pytest.approx(value, abs=0.05)
    assert result.ranking()[0] == 'p.x1'
    assert result.indices['p.x1']['ST'] == pytest.approx(0.5576, abs=0.05)


def test_morris_and_one_at_a_time_rank_parameters():
    """Test that Morris screening and one-at-a-time effects rank the carrying capacity first for the final size."""
    bounds = {'growth.r': (0.5, 1.0), 'growth.K': (50.0, 150.0)}
    morris = sensitivity('logistic_growth', bounds, 'population.n', method='morris', n=10, until=20, random_state=0)
    oat = sensitivity('logistic_growth', bounds, 'population.n', method='oat', until=20)
    assert morris.ranking()[0] == 'growth.K'
    assert oat.ranking()[0] == 'growth.K'
