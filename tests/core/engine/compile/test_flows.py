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
    """An interaction that returns the rate terms of a function of its namespace."""

    def __init__(self, label: str, *, target: Role, terms=None) -> None:
        super().__init__(label)
        self.target = target
        self.terms = terms

    def _prepare(self, context=None):
        """Set the rate terms."""
        super()._prepare(context)
        if hasattr(context, 'target'):
            self.laws_ = self.terms(context.target)


def _births_and_deaths(s):
    """Return births into the living and deaths from the living into the dead."""
    return [
        RateLaw(s.n, RATES['births'] * s.n, label='births'),
        RateLaw(s.n, -RATES['deaths'] * s.n, label='deaths'),
        RateLaw(s.m, RATES['deaths'] * s.n, label='deaths'),
    ]


def _system(terms, *elements):
    """Return a prepared system of the elements and one interaction of the terms."""
    elements = elements or (PeopleElement('population'),)
    interaction = TermsInteraction('process', target=elements[0].label, terms=terms)
    return System(elements=list(elements), interactions=[interaction]).prepare()


def test_labelled_rate_terms_are_recorded_as_flows_under_their_element():
    """Test that the trajectory holds every labelled flow by its label, under the element of its stocks."""
    trajectory = _system(_births_and_deaths).simulate(until=2.0, dt=1.0)
    n = trajectory.get_values('population.n')
    np.testing.assert_allclose(trajectory.get_values('population.births'), RATES['births'] * n)
    np.testing.assert_allclose(trajectory.get_values('population.deaths'), RATES['deaths'] * n)
    np.testing.assert_allclose(n, 10.0 * np.exp((RATES['births'] - RATES['deaths']) * np.arange(3)), rtol=1e-4)


def test_flow_matrix_and_loops_read_the_labelled_terms():
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
    """Test that a lone term with a minus sign is an outflow, recorded as a positive amount."""
    trajectory = _system(lambda s: [RateLaw(s.n, -0.5 * s.n, label='deaths')]).simulate(until=1.0, dt=1.0)
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
            """Set the two terms of the flow."""
            super()._prepare(context)
            if hasattr(context, 'upper'):
                amount = context.upper.n / 2
                self.laws_ = [
                    RateLaw(context.upper.n, -amount, label='drain'),
                    RateLaw(context.lower.n, amount, label='drain'),
                ]

    tanks = [PeopleElement('upper'), PeopleElement('lower', n=0.0)]
    system = System(elements=tanks, interactions=[TwoTanksInteraction('pipe', upper='upper', lower='lower')])
    trajectory = system.prepare().simulate(until=1.0, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('pipe.drain'), trajectory.get_values('upper.n') / 2)


@pytest.mark.parametrize(
    ('terms', 'error', 'match'),
    [
        (lambda s: [RateLaw(s.n, -s.n, label='f'), RateLaw(s.m, 2 * s.n, label='f')], ValidationError, 'add up'),
        (lambda s: [RateLaw(s.n, -s.n, label='f'), RateLaw(s.n, s.n, label='f')], ValidationError, 'one term'),
        (lambda s: [RateLaw(s.n, -s.n, label='f'), RateLaw(s.k, s.n, label='f')], ConservationError, 'quantities'),
    ],
)
def test_terms_that_are_no_flow_are_rejected(terms, error, match):
    """Test that terms that do not cancel, change one stock twice, or join two quantities raise an error."""
    with pytest.raises(error, match=match):
        _system(terms, LedgerElement('population'))


def test_two_flows_with_one_label_on_one_element_are_rejected():
    """Test that two interactions setting a flow of the same label on one element raise a label error."""
    interactions = [
        TermsInteraction(label, target='population', terms=lambda s: [RateLaw(s.n, 1.0, label='births')])
        for label in ('first', 'second')
    ]
    with pytest.raises(LabelError, match='births'):
        System(elements=[PeopleElement('population')], interactions=interactions).prepare()
