"""Test the `_flows` module of `skcomplex.core.engine.compile`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.analysis import loops
from skcomplex.core.components.elements import Element, Quantity
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import ConservationError, LabelError, ValidationError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System

RATES = {'births': 0.1, 'deaths': 0.05}


class PeopleElement(Element):
    """Two stocks of people, the living and the dead."""

    def __init__(
        self,
        label: str,
        *,
        n: Annotated[float, Quantity('people')] = 10.0,
        m: Annotated[float, Quantity('people')] = 0.0,
    ) -> None:
        super().__init__(label)
        self.n = n
        self.m = m


class LedgerElement(Element):
    """Two stocks of people and a stock of money."""

    def __init__(
        self,
        label: str,
        *,
        n: Annotated[float, Quantity('people')] = 1.0,
        m: Annotated[float, Quantity('people')] = 0.0,
        k: Annotated[float, Quantity('money')] = 0.0,
    ) -> None:
        super().__init__(label)
        self.n = n
        self.m = m
        self.k = k


class TermsInteraction(Interaction):
    """An interaction that returns the flows of a function of its namespace."""

    def __init__(self, label: str, *, target: Role, terms=None) -> None:
        super().__init__(label)
        self.target = target
        self.terms = terms

    def _prepare(self, context=None):
        """Set the flows."""
        super()._prepare(context)
        if hasattr(context, 'target'):
            self.laws_ = self.terms(context.target)


def _births_and_deaths(s):
    """Return births into the living and deaths from the living into the dead."""
    return [
        RateLaw({s.n: +1}, RATES['births'] * s.n, label='births'),
        RateLaw({s.n: -1, s.m: +1}, RATES['deaths'] * s.n, label='deaths'),
    ]


def _system(terms, *elements):
    """Return a prepared system of the elements and one interaction of the flows."""
    elements = elements or (PeopleElement('population'),)
    interaction = TermsInteraction('process', target=elements[0].label, terms=terms)
    return System(elements=list(elements), interactions=[interaction]).prepare()


def test_flows_are_recorded_under_the_element_of_their_stocks():
    """Test that the trajectory holds every labelled flow by its label, under the element of its stocks."""
    trajectory = _system(_births_and_deaths).simulate(until=2.0, dt=1.0)
    n = trajectory.get_values('population.n')
    np.testing.assert_allclose(trajectory.get_values('population.births'), RATES['births'] * n)
    np.testing.assert_allclose(trajectory.get_values('population.deaths'), RATES['deaths'] * n)
    np.testing.assert_allclose(n, 10.0 * np.exp((RATES['births'] - RATES['deaths']) * np.arange(3)), rtol=1e-4)


def test_flow_matrix_and_loops_read_the_signs_of_the_flows():
    """Test that a flow with only a sink comes from the environment and one with both moves between stocks."""
    system = _system(_births_and_deaths)
    trajectory = system.simulate(until=1.0, dt=1.0)
    matrix = loops(trajectory, kind='flow-matrix')
    assert matrix.flows == ('population.births', 'population.deaths')
    assert matrix.holders == ('environment', 'population.n', 'population.m')
    n = trajectory.get_values('population.n')[-1]
    np.testing.assert_allclose(
        matrix.matrix,
        [[-RATES['births'] * n, RATES['births'] * n, 0.0], [0.0, -RATES['deaths'] * n, RATES['deaths'] * n]],
    )
    analysis = loops(system)
    assert dict(zip(analysis.loops, analysis.polarities, strict=True)) == {
        ('population.n', 'population.births'): 'reinforcing',
        ('population.n', 'population.deaths'): 'balancing',
    }
    assert analysis.graph.edges['population.deaths', 'population.m']['sign'] == 1


def test_a_flow_out_of_the_system_has_only_its_source():
    """Test that a flow that maps one stock to -1 is an outflow, recorded as a positive amount."""
    trajectory = _system(lambda s: [RateLaw({s.n: -1}, 0.5 * s.n, label='deaths')]).simulate(until=1.0, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('population.deaths'), 0.5 * trajectory.get_values('population.n'))
    matrix = loops(trajectory, kind='flow-matrix')
    assert matrix.holders == ('population.n', 'environment')


def test_a_flow_between_elements_is_reported_under_its_interaction():
    """Test that a flow whose stocks belong to two elements takes the path of the interaction that sets it."""

    class TwoTanksInteraction(Interaction):
        """A flow from the upper tank to the lower one."""

        def __init__(self, label: str, *, upper: Role, lower: Role) -> None:
            super().__init__(label)
            self.upper = upper
            self.lower = lower

        def _prepare(self, context=None):
            """Set the flow."""
            super()._prepare(context)
            if hasattr(context, 'upper'):
                self.laws_ = [RateLaw({context.upper.n: -1, context.lower.n: +1}, context.upper.n / 2, label='drain')]

    tanks = [PeopleElement('upper'), PeopleElement('lower', n=0.0)]
    system = System(elements=tanks, interactions=[TwoTanksInteraction('pipe', upper='upper', lower='lower')])
    trajectory = system.prepare().simulate(until=1.0, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('pipe.drain'), trajectory.get_values('upper.n') / 2)


@pytest.mark.parametrize(
    ('terms', 'error', 'match'),
    [
        (
            lambda s: [RateLaw({s.n: -1}, s.n, label='f'), RateLaw({s.m: +1}, 2 * s.n, label='f')],
            ValidationError,
            'move',
        ),
        (lambda s: [RateLaw({s.n: -1}, s.n, label='f'), RateLaw({s.m: -1}, s.n, label='f')], ValidationError, 'Rename'),
        (lambda s: [RateLaw({s.n: -1, s.m: -1}, s.n, label='f')], ValidationError, 'signs'),
        (lambda s: [RateLaw({s.n: 2}, s.n, label='f')], ValidationError, 'signs'),
        (lambda s: [RateLaw({s.n: -1, s.m: +1}, s.n)], ValidationError, 'no label'),
        (lambda s: [RateLaw({s.n: -1, s.k: +1}, s.n, label='f')], ConservationError, 'quantities'),
    ],
)
def test_flows_that_break_the_rules_are_rejected(terms, error, match):
    """Test that unequal amounts, two sources, wrong signs, no label, or two quantities raise an error."""
    with pytest.raises(error, match=match):
        _system(terms, LedgerElement('population'))


def test_two_flows_with_one_label_on_one_element_are_rejected():
    """Test that two interactions setting a flow of the same label on one element raise a label error."""
    interactions = [
        TermsInteraction(label, target='population', terms=lambda s: [RateLaw({s.n: +1}, 1.0, label='births')])
        for label in ('first', 'second')
    ]
    with pytest.raises(LabelError, match='births'):
        System(elements=[PeopleElement('population')], interactions=interactions).prepare()


def test_an_inflow_that_sage_writes_with_a_minus_sign_enters_its_stock():
    """Test that an inflow such as `x (1 - y)`, which Sage writes as `-x (y - 1)`, is drawn into its stock."""

    def logistic(s):
        return [RateLaw({s.n: +1}, s.n * (1 - s.m), label='growth')]

    system = _system(logistic, PeopleElement('population', n=1.0, m=0.5))
    graph = loops(system).graph
    assert graph.edges['population.growth', 'population.n']['sign'] == 1
    assert ('population.growth', 'environment') not in graph.edges
    trajectory = system.simulate(until=1.0, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('population.n'), [1.0, np.exp(0.5)], rtol=1e-4)
