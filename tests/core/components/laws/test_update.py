"""Test the `_update` module of `skcomplex.core.components.laws`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, var

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import (
    LocalRuleLaw,
    OuterTotalisticSpec,
    RateLaw,
    TopologyRequestSpec,
    TotalisticSpec,
    UpdateLaw,
)
from skcomplex.core.components.laws.utils import bernoulli, lag, shift_member, total, where
from skcomplex.core.components.spaces import LatticeSpace, RadiusNeighborhood
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec
from skcomplex.core.systems import System

ROW = LatticeSpace(5)
AROUND = RadiusNeighborhood(1, center=True)


class EconomyElement(Element):
    """Prices that grow every step, with an adaptive policy rate."""

    def __init__(
        self,
        label: str,
        *,
        price: Annotated[float, StateSpec()] = 1.0,
        rate: Annotated[float, ParameterSpec(adaptive=True)] = 0.02,
    ) -> None:
        super().__init__(label)
        self.price = price
        self.rate = rate

    def _get_laws(self, s):
        """Return the growth of prices."""
        return [UpdateLaw(s.price, s.price * 1.03)]


class PolicyInteraction(Interaction):
    """Rule that raises the policy rate whenever prices exceed a target."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the update of the adaptive rate."""
        return [UpdateLaw(s.target.rate, where(s.target.price > 1.05, s.target.rate + 0.01, s.target.rate))]


class FibonacciElement(Element):
    """Fibonacci sequence, a state with one lag."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, StateSpec(lags=1, initial=(1.0,))] = 1.0,
        x_lag1: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.x_lag1 = x_lag1

    def _get_laws(self, s):
        """Return the recurrence."""
        return [UpdateLaw(s.x, s.x + lag(s.x, 1))]


class IncomeElement(Element):
    """Samuelson's multiplier-accelerator system of income."""

    def __init__(
        self,
        label: str,
        *,
        Y: Annotated[object, StateSpec(lags=1, initial=(100.0,))] = 110.0,
        Y_lag1: float = 100.0,
        c: Annotated[object, ParameterSpec()] = 0.8,
        v: Annotated[object, ParameterSpec()] = 1.5,
        G: Annotated[object, ParameterSpec()] = 20.0,
    ) -> None:
        super().__init__(label)
        self.Y = Y
        self.Y_lag1 = Y_lag1
        self.c = c
        self.v = v
        self.G = G

    def _get_laws(self, s):
        """Return the recurrence of income."""
        return [UpdateLaw(s.Y, s.c * (1 + s.v) * s.Y - s.c * s.v * lag(s.Y, 1) + s.G)]


class PumpElement(Element):
    """Tank filled by a bounded control input."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        u: Annotated[object, ParameterSpec(adaptive=True, bounds=(0, 1))] = 0.0,
        weight: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.u = u
        self.weight = weight

    def _get_laws(self, s):
        """Return the filling of the tank."""
        return [RateLaw(s.x, -s.x + s.u)]


def test_update_runs_into_its_value_and_targets_a_topology_request():
    """Test that an update runs into its new value, and targets a topology request that it fires with a probability."""
    x = var('x')
    assert UpdateLaw(x, 2 * x).prepare().run({'x': 1.5}).value_ == 3.0
    request = TopologyRequestSpec('rewire', keep='target')
    law = UpdateLaw(request, 0.1).prepare()
    assert law.targets_ == (request,)


def test_adaptive_parameter_changes_only_when_the_policy_fires():
    """Test that the adaptive rate changes only at steps where prices exceed the target, and starts from its value."""
    system = CompositeSystem(
        elements=[EconomyElement('economy', rate=0.03)],
        interactions=[PolicyInteraction('bank', target='economy')],
    )
    trajectory = simulate(system, until=4)
    rates, prices = trajectory.get_values('economy.rate'), trajectory.get_values('economy.price')
    assert rates[0] == pytest.approx(0.03)
    np.testing.assert_array_equal(np.diff(rates) > 0, prices[:-1] > 1.05)


def test_fibonacci_recurrence():
    """Test that `x_{t+1} = x_t + x_{t-1}` from 1 and 1 gives the Fibonacci sequence."""
    trajectory = simulate(CompositeSystem(elements=[FibonacciElement('f')]), until=8)
    assert trajectory.get_values('f.x').tolist() == [1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert trajectory.get_values('f.x_lag1').tolist()[:3] == [1, 1, 2]


def test_multiplier_accelerator_oscillates_with_growing_amplitude():
    """Test that Samuelson's system with c v > 1 oscillates around G / (1 - c) with growing amplitude."""
    income = simulate(CompositeSystem(elements=[IncomeElement('economy')]), until=40).get_values('economy.Y')
    deviation = income - 20.0 / (1 - 0.8)
    assert np.any(deviation[1:] * deviation[:-1] < 0)
    assert np.abs(deviation[-10:]).max() > 10 * np.abs(deviation[:10]).max()


def test_adaptive_parameters_compile_with_their_bounds():
    """Test that compiling keeps the bounded adaptive parameters, the controls, and leaves the form an ODE."""
    problem = PassCompiler().prepare(CompositeSystem(elements=[PumpElement('pump')])).run().problem_
    assert problem.form.name == 'ODE'
    assert set(problem.adaptive_parameters) == {'pump.u'}
    assert problem.adaptive_parameters['pump.u'].bounds == (0.0, 1.0)


class CounterElement(Element):
    """Member with a value `x`."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x


class LawsInteraction(Interaction):
    """Interaction whose laws a function of the namespace gives."""

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], laws: object) -> None:
        super().__init__(label)
        self.target = target
        self.laws = laws

    def _get_laws(self, s):
        """Return the laws the function gives."""
        return self.laws(s)


def _counters(laws, initial, until=1):
    """Return the values of the members of a group under the laws of one interaction."""
    group = GroupElement('counters', element=CounterElement, size=len(initial), values={'x': initial})
    system = CompositeSystem(elements=[group], interactions=[LawsInteraction('laws', target='counters', laws=laws)])
    return simulate(system, until=until).get_values('counters.x')


def _increments(s):
    """Return two increments of the value."""
    return [
        UpdateLaw(s.target.x, s.target.x + 1, increment=True),
        UpdateLaw(s.target.x, s.target.x + 2, increment=True),
    ]


def _moves(s):
    """Return the moves of the value from 0 to 1 and from 1 to 2."""
    return [
        UpdateLaw(s.target.x, where(s.target.x == 0, 1, s.target.x), source=0),
        UpdateLaw(s.target.x, where(s.target.x == 1, 2, s.target.x), source=1),
    ]


def test_update_increments_of_one_state_add():
    """Test that two increment laws of one state add their changes in every step."""
    assert _counters(_increments, [0.0, 10.0], until=2)[-1].tolist() == [6.0, 16.0]


def test_update_moves_from_distinct_sources_compose():
    """Test that moves of one state from distinct source values all apply from the values at the start of the step."""
    assert _counters(_moves, [0.0, 1.0, 2.0])[-1].tolist() == [1.0, 2.0, 2.0]


@pytest.mark.parametrize('sources', [(0, 0), (None, 1), (None, None)])
def test_update_laws_that_may_conflict_raise_error(sources):
    """Test that two update laws of one state with the same source, or one without a source, raise an error."""

    def laws(s):
        """Return one move per source."""
        return [UpdateLaw(s.target.x, s.target.x + 1, source=source) for source in sources]

    with pytest.raises(ValidationError, match='second time'):
        _counters(laws, [0.0])


@pytest.mark.parametrize(
    ('order', 'initial', 'expected'),
    [
        (None, [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]),
        (SR(0), [1.0, 1.0, 1.0], [3.0, 5.0, 9.0]),
        ('descending', [1.0, 2.0, 3.0], [16.0, 9.0, 6.0]),
    ],
)
def test_update_order_reads_the_values_set_before(order, initial, expected):
    """Test that ordered members read the values the members before them set, and unordered ones the step start."""

    def laws(s):
        """Return the update of every member to the total of the group."""
        key = -s.target.x if order == 'descending' else order
        return [UpdateLaw(s.target.x, total(s.target.x), order=key)]

    assert _counters(laws, initial)[-1].tolist() == expected


def test_update_check_rejects_an_increment_with_a_source():
    """Test that an increment law with a source raises an error."""
    x = var('x')
    with pytest.raises(ValidationError, match='no source'):
        UpdateLaw(x, x + 1, increment=True, source=0).prepare()


@pytest.mark.parametrize(
    ('rule', 'expected'),
    [
        (TotalisticSpec(lambda total: total % 2), [0, 1, 1, 1, 0]),
        (OuterTotalisticSpec({(0, 1): 1}), [0, 1, 0, 1, 0]),
        (lambda center, left, right: left, [0, 0, 0, 1, 0]),
    ],
)
def test_update_with_a_neighborhood_runs_a_local_rule(rule, expected):
    """Test that an update with a neighbourhood sets every cell from the ordered values of its neighbours."""
    x = var('x')
    law = UpdateLaw(x, rule, neighborhood=AROUND).prepare()
    assert law.run({'space': ROW, 'x': [0, 0, 1, 0, 0]}).value_.tolist() == expected


def test_update_with_a_neighborhood_becomes_a_local_rule_law():
    """Test that a local rule keeps its neighbourhood, reads, schedule, and label, and draws when it names `random`."""
    a, b = var('a b')

    def flip(*values, random):
        """Return the first value flipped."""
        return 1 - values[0]

    law = UpdateLaw((a, b), flip, neighborhood=AROUND, reads=[b], schedule='checkerboard', label='flip')
    rule = law._get_local_rule()
    assert isinstance(rule, LocalRuleLaw)
    assert (rule.targets, rule.over, rule.rule, rule.reads, rule.schedule, rule.stochastic, rule.label) == (
        (a, b),
        AROUND,
        flip,
        (b,),
        'checkerboard',
        True,
        'flip',
    )
    assert law._get_targets() == (a, b)
    assert not UpdateLaw(a, TotalisticSpec({1: 1}), neighborhood=AROUND)._get_local_rule().stochastic


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'neighborhood': 'moore'}, 'Give one'),
        ({'neighborhood': AROUND, 'schedule': 'sometimes'}, 'Unknown schedule'),
        ({'schedule': 'checkerboard'}, 'no neighbourhood'),
        ({'reads': [var('y')]}, 'no neighbourhood'),
        ({'neighborhood': AROUND, 'phase': 'move'}, 'neighbourhood and phase'),
        ({'neighborhood': AROUND, 'increment': True}, 'neighbourhood and increment'),
    ],
)
def test_update_check_rejects_a_bad_local_rule(settings, message):
    """Test that a local rule needs a neighbourhood, a known schedule, and none of the settings of one member."""
    x = var('x')
    with pytest.raises(ValidationError, match=message):
        UpdateLaw(x, TotalisticSpec({1: 1}), **settings).prepare()


class StepTransferInteraction(Interaction):
    """Moves of one unit along a ring of cells in one step, each with probability one half."""

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: object = None) -> None:
        """Set the increment of every cell, which loses its unit and gains the unit of the cell behind it."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            level = context.target.level
            move = bernoulli('move', 0.5)
            leaves = level * (1 - shift_member(level, 1)) * move
            arrives = shift_member(level, -1) * (1 - level) * shift_member(move, -1)
            self.laws_ = [UpdateLaw(level, level - leaves + arrives, increment=True)]


def test_update_increments_move_units_once_per_step_with_one_draw_per_edge():
    """Test that units move along a ring once per step, both ends of an edge reading one draw, so none is lost."""
    cells = Element('cells', values={'level': np.tile([1.0, 0.0], 25)}, space=LatticeSpace(50))
    system = System(elements=[cells], interactions=[StepTransferInteraction('moves', target='cells')]).prepare()
    levels = np.asarray(system.simulate(until=50, random_state=0).get_values('cells.level'))
    assert system.form_ == 'discrete map'
    assert set(np.unique(levels)) == {0.0, 1.0}
    assert set(levels.sum(axis=1)) == {25.0}
    assert (levels[1:] != levels[:-1]).any()
