"""Test the `_sensitivity` module of `skcomplex.core.experiments`."""

import math
from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.systems import System
from skcomplex.core.errors import AnalysisError
from skcomplex.core.experiments import sensitivity
from skcomplex.core.specs import ObservableSpec, ParameterSpec, VisualSpec
from skcomplex.systems.biology.ecology import LogisticGrowthSystem


class PointElement(Element):
    """Point element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x1: Annotated[object, ParameterSpec()] = 0.0,
        x2: Annotated[object, ParameterSpec()] = 0.0,
        x3: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3


def _ishigami(s):
    """Return the Ishigami function of the point's coordinates."""
    return sin(s.p.x1) + 7 * sin(s.p.x2) ** 2 + 0.1 * s.p.x3**4 * sin(s.p.x1)


@pytest.mark.slow
def test_sobol_indices_of_ishigami():
    """Test that Sobol first-order indices of the Ishigami function match the known values within 0.05."""
    system = System(elements=[PointElement('p')])
    bounds = {f'p.x{index}': (-math.pi, math.pi) for index in (1, 2, 3)}
    observables = [ObservableSpec('y', _ishigami)]
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
        output='population.n',
        method=method,
        until=20,
        random_state=0,
    )
    assert analysis.ranking[0] == 'growth.K'
    assert analysis.inputs.shape == (len(analysis.outputs), 2)


def test_model_parameters_are_inputs():
    """Test that model parameters rank like the dotted paths of the parameters they set."""
    bounds = {'r': (0.5, 1.0), 'K': (50.0, 150.0)}
    analysis = sensitivity(LogisticGrowthSystem(), bounds, output='population.n', method='oat', until=20)
    assert analysis.ranking == ['K', 'r']


def test_sensitivity_check_rejects_an_unknown_method_and_a_missing_output():
    """Test that an unknown method or a missing output fail with a named error."""
    with pytest.raises(AnalysisError, match='Unknown sensitivity method'):
        sensitivity(LogisticGrowthSystem(), {'r': (0.5, 1.0)}, method='fast', output='y')
    with pytest.raises(AnalysisError, match='output = None'):
        sensitivity(LogisticGrowthSystem(), {'r': (0.5, 1.0)}, output=None)
