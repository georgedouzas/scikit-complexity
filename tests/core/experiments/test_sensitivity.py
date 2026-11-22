"""Test the `_sensitivity` module of `skcomplex.core.experiments`."""

import math

import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.errors import AnalysisError
from skcomplex.core.experiments import sensitivity
from skcomplex.core.systems import System
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.mathematics.dynamical_systems import HenonMapInteraction


def _ishigami(s):
    """Return the Ishigami function of the point's coordinates."""
    return sin(s.p.x1) + 7 * sin(s.p.x2) ** 2 + 0.1 * s.p.x3**4 * sin(s.p.x1)


@pytest.mark.slow
def test_sobol_indices_of_ishigami():
    """Test that Sobol first-order indices of the Ishigami function match the known values within 0.05."""
    system = System(elements=[Element('p', values={'x1': 0.0, 'x2': 0.0, 'x3': 0.0})])
    bounds = {f'p.x{index}': (-math.pi, math.pi) for index in (1, 2, 3)}
    observables = [Observable('y', _ishigami)]
    analysis = sensitivity(system, bounds, output='y', n=2048, until=0, observables=observables, random_state=0)
    expected = {'p.x1': 0.3139, 'p.x2': 0.4424, 'p.x3': 0.0}
    for name, value in expected.items():
        assert analysis.first_order[name] == pytest.approx(value, abs=0.05)
    assert analysis.total_order['p.x1'] == pytest.approx(0.5576, abs=0.05)
    assert analysis.ranking[0] == 'p.x1'


@pytest.mark.parametrize('method', ['morris', 'oat'])
def test_morris_and_one_at_a_time_rank_parameters(method):
    """Test that Morris screening and one-at-a-time effects rank the carrying capacity first for the final size."""
    bounds = {'growth.r': (0.5, 1.0), 'growth.K': (50.0, 150.0)}
    analysis = sensitivity(
        LogisticGrowthSystem(),
        bounds,
        output='population.x',
        method=method,
        until=20,
        random_state=0,
    )
    assert analysis.ranking[0] == 'growth.K'
    assert analysis.inputs.shape == (len(analysis.outputs), 2)


def test_model_parameters_are_inputs():
    """Test that model parameters rank like the dotted paths of the parameters they set."""
    bounds = {'r': (0.5, 1.0), 'K': (50.0, 150.0)}
    analysis = sensitivity(LogisticGrowthSystem(), bounds, output='population.x', method='oat', until=20)
    assert analysis.ranking == ['K', 'r']


def test_sensitivity_check_rejects_an_unknown_method_and_a_missing_output():
    """Test that an unknown method or a missing output fail with a named error."""
    with pytest.raises(AnalysisError, match='Unknown sensitivity method'):
        sensitivity(LogisticGrowthSystem(), {'r': (0.5, 1.0)}, method='fast', output='y')
    with pytest.raises(AnalysisError, match='output = None'):
        sensitivity(LogisticGrowthSystem(), {'r': (0.5, 1.0)}, output=None)


def _henon():
    """Return the Hénon map as a `System` of one element and one interaction."""
    return System(
        'henon_map',
        elements=[Element('map', values={'x': 0.0, 'y': 0.0})],
        interactions=[HenonMapInteraction('henon', target='map')],
    )


def test_sensitivity_of_a_system_by_nested_name_and_path():
    """Test that the inputs of a `System` may be nested parameters and dotted paths, and that both move the output."""
    bounds = {'henon__a': (1.1, 1.3), 'henon.b': (0.2, 0.3)}
    result = sensitivity(_henon(), bounds, output='map.x', method='oat', until=3)
    assert sorted(result.ranking) == ['henon.b', 'henon__a']
    assert all(effect != 0 for effect in result.effects.values())
