"""Test the `_flow` module of `skcomplex.systems.general.dynamics`."""

import numpy as np
import pytest

from skcomplex.core.analysis import loops
from skcomplex.core.components.elements import Element
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.general.dynamics import FlowInteraction


def _population(*flows):
    """Return a population whose flows are given."""
    population = Element('population', values={'n': 10.0, 'm': 0.0, 'b': 0.1, 'd': 0.05})
    return System(elements=[population], interactions=list(flows))


def test_flow_takes_from_one_stock_what_it_gives_to_the_other():
    """Test that a flow between two stocks keeps their total and is recorded under its label."""
    deaths = FlowInteraction(
        'deaths',
        target='population',
        leaves='n',
        enters='m',
        amount=lambda s: s.target.d * s.target.n,
    )
    trajectory = _population(deaths).prepare().simulate(until=2.0, dt=1.0)
    n, m = trajectory.get_values('population.n'), trajectory.get_values('population.m')
    np.testing.assert_allclose(n + m, 10.0)
    np.testing.assert_allclose(trajectory.get_values('population.deaths'), 0.05 * n)


def test_flows_with_the_environment_close_a_loop_through_their_stock():
    """Test that births reinforce and deaths balance a population, as loops through each flow."""
    births = FlowInteraction('births', target='population', enters='n', amount=lambda s: s.target.b * s.target.n)
    deaths = FlowInteraction('deaths', target='population', leaves='n', amount=lambda s: s.target.d * s.target.n)
    analysis = loops(_population(births, deaths).prepare())
    assert dict(zip(analysis.loops, analysis.polarities, strict=True)) == {
        ('population.n', 'population.births'): 'reinforcing',
        ('population.n', 'population.deaths'): 'balancing',
    }
    assert analysis.graph.edges['population.deaths', 'population.n']['sign'] == -1


def test_flow_of_no_stock_is_rejected():
    """Test that a flow that neither leaves nor enters a stock names itself."""
    with pytest.raises(ValidationError, match="'leak' neither leaves nor enters"):
        _population(FlowInteraction('leak', target='population', amount=lambda s: 1.0)).prepare()
