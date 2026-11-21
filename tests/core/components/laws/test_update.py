"""Test the `_update` module of `skcomplex.core.components.laws`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, var
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import (
    ObjectiveLaw,
    OuterTotalisticSpec,
    RateLaw,
    TopologyRequestSpec,
    TotalisticSpec,
    UpdateLaw,
)
from skcomplex.core.components.laws.utils import bernoulli, shift_member, total, where
from skcomplex.core.components.spaces import LatticeSpace, RadiusNeighborhood
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role

ROW = LatticeSpace(5)
AROUND = RadiusNeighborhood(1, center=True)


def _run(label, values, laws, until, *, size=None):
    """Return the run of one element, or group, under the laws of one interaction, a function of the namespace."""
    rule = Interaction('laws', roles={'target': label}, laws=laws)
    system = System(elements=[Element(label, values=values, size=size)], interactions=[rule])
    return system.prepare().simulate(until=until)


def _policy(s):
    """Return the growth of prices and the rise of the policy rate whenever prices exceed a target."""
    e = s.target
    return [UpdateLaw(e.price, e.price * 1.03), UpdateLaw(e.rate, where(e.price > 1.05, e.rate + 0.01, e.rate))]


def _fibonacci(s):
    """Return the recurrence `x_{t+1} = x_t + x_{t-1}`."""
    x = s.target.x
    return [UpdateLaw(x, x + x(t=s.t - 1))]


def _income(s):
    """Return Samuelson's recurrence of income with `c = 0.8`, `v = 1.5`, and `G = 20`."""
    e = s.target
    return [UpdateLaw(e.Y, e.c * (1 + e.v) * e.Y - e.c * e.v * e.Y(t=s.t - 1) + e.G)]


def test_update_gives_its_value_and_targets_a_topology_request():
    """Test that an update gives its new value, and targets a topology request that it fires with a probability."""
    x = var('x')
    assert UpdateLaw(x, 2 * x)._get_value({'x': 1.5}) == 3.0
    request = TopologyRequestSpec('rewire', keep='target')
    law = UpdateLaw(request, 0.1)
    assert law._get_targets() == (request,)


def test_adaptive_parameter_changes_only_when_the_policy_fires():
    """Test that the adaptive rate changes only at steps where prices exceed the target, and starts from its value."""
    trajectory = _run('economy', {'price': 1.0, 'rate': 0.03}, _policy, 4)
    rates, prices = trajectory.get_values('economy.rate'), trajectory.get_values('economy.price')
    assert rates[0] == pytest.approx(0.03)
    np.testing.assert_array_equal(np.diff(rates) > 0, prices[:-1] > 1.05)


def test_fibonacci_recurrence():
    """Test that `x_{t+1} = x_t + x_{t-1}` from 1 and 1 gives the Fibonacci sequence."""
    trajectory = _run('f', {'x': 1.0, 'x[-1]': 1.0}, _fibonacci, 8)
    assert trajectory.get_values('f.x').tolist() == [1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert trajectory.get_values('f.x[-1]').tolist()[:3] == [1, 1, 2]


def test_multiplier_accelerator_oscillates_with_growing_amplitude():
    """Test that Samuelson's system with c v > 1 oscillates around G / (1 - c) with growing amplitude."""
    values = {'Y': 110.0, 'Y[-1]': 100.0, 'c': 0.8, 'v': 1.5, 'G': 20.0}
    income = _run('economy', values, _income, 40).get_values('economy.Y')
    deviation = income - 20.0 / (1 - 0.8)
    assert np.any(deviation[1:] * deviation[:-1] < 0)
    assert np.abs(deviation[-10:]).max() > 10 * np.abs(deviation[:10]).max()


class PumpElement(Element):
    """Tank filled by a control input bounded to the unit interval."""

    def __init__(self, label: str, *, x: float = 1.0, u: Annotated[float, RealSet.closed(0, 1)] = 0.0) -> None:
        super().__init__(label)
        self.x = x
        self.u = u


def test_adaptive_parameters_compile_with_their_bounds():
    """Test that compiling keeps the bounded adaptive parameters, the controls, and leaves the form an ODE."""

    def laws(s):
        """Return the filling of the tank and a cost of the control."""
        return [RateLaw(s.target.x, -s.target.x + s.target.u), ObjectiveLaw(2.0 * s.target.u**2, controls=[s.target.u])]

    rule = Interaction('fill', roles={'target': 'pump'}, laws=laws)
    problem = System(elements=[PumpElement('pump')], interactions=[rule]).prepare()._problem
    assert problem.form.name == 'ODE'
    assert set(problem.adaptive_parameters) == {'pump.u'}
    assert problem.adaptive_parameters['pump.u'].bounds == (0.0, 1.0)


def _counters(laws, initial, until=1):
    """Return the values of the members of a group under the laws of one interaction."""
    return _run('counters', {'x': initial}, laws, until, size=len(initial)).get_values('counters.x')


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


def _shift(values, **kwargs):
    """Return the values shifted by one cell when they are an array, and 2 for a scalar state."""
    return np.roll(values, 1) if isinstance(values, np.ndarray) else 2.0


def test_update_with_arrays_gives_a_function_of_two_arguments_the_whole_array():
    """Test that `arrays=True` hands the whole array of a row of cells to a function of two arguments."""
    rule = Interaction('shift', roles={'target': 'row'}, laws=lambda s: [UpdateLaw(s.target.x, _shift, arrays=True)])
    row = Element('row', values={'x': [0.0, 1.0, 2.0, 3.0, 4.0]}, space=LatticeSpace(5))
    trajectory = System(elements=[row], interactions=[rule]).prepare().simulate(until=1)
    assert trajectory.get_values('row.x')[-1].tolist() == [4.0, 0.0, 1.0, 2.0, 3.0]


def test_update_without_arrays_calls_a_function_of_two_arguments_with_time_and_values():
    """Test that without `arrays` the same function gives the next value of one state as `f(t, values)`."""
    received = []

    def step(t, values):
        """Return the shift of the values and record what it receives."""
        received.append((t, dict(values)))
        return _shift(values)

    trajectory = _run('counter', {'x': 1.0}, lambda s: [UpdateLaw(s.target.x, step)], 1)
    assert trajectory.get_values('counter.x').tolist() == [1.0, 2.0]
    assert received[0] == (0.0, {'counter.x': 1.0})


@pytest.mark.parametrize(
    ('target', 'value', 'settings'),
    [
        (var('x'), var('x') + 1, {}),
        (TopologyRequestSpec('rewire'), _shift, {}),
        (var('x'), _shift, {'neighborhood': AROUND}),
    ],
)
def test_update_check_rejects_arrays_without_a_function_of_a_state(target, value, settings):
    """Test that `arrays=True` needs a function value, a state target, and no neighbourhood."""
    with pytest.raises(ValidationError, match='arrays=True'):
        UpdateLaw(target, value, arrays=True, **settings)._check()


def test_update_check_rejects_a_score_without_arrays():
    """Test that a score needs an update of the whole array."""
    with pytest.raises(ValidationError, match='has a score'):
        UpdateLaw(var('x'), _shift, score=np.sum)._check()


def test_update_check_rejects_an_increment_with_a_source():
    """Test that an increment law with a source raises an error."""
    x = var('x')
    with pytest.raises(ValidationError, match='no source'):
        UpdateLaw(x, x + 1, increment=True, source=0)._check()


@pytest.mark.parametrize(
    ('rule', 'expected'),
    [
        (TotalisticSpec(lambda total: total % 2), [0, 1, 1, 1, 0]),
        (OuterTotalisticSpec({(0, 1): 1}), [0, 1, 0, 1, 0]),
        (lambda center, left, right, **kwargs: left, [0, 0, 0, 1, 0]),
    ],
)
def test_update_with_a_neighborhood_runs_a_local_rule(rule, expected):
    """Test that an update with a neighbourhood sets every cell from the ordered values of its neighbours."""
    x = var('x')
    law = UpdateLaw(x, rule, neighborhood=AROUND)
    assert law._get_value({'space': ROW, 'x': [0, 0, 1, 0, 0]}).tolist() == expected


def test_update_with_a_neighborhood_becomes_a_local_rule():
    """Test that a local rule keeps the settings of its law, `stochastic` among them."""
    a, b = var('a b')

    def flip(*values, random, **kwargs):
        """Return the first value flipped."""
        return 1 - values[0]

    law = UpdateLaw(
        (a, b),
        flip,
        neighborhood=AROUND,
        reads=[b],
        schedule='checkerboard',
        label='flip',
        stochastic=True,
    )
    rule = law._get_local_rule()
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
    assert not UpdateLaw(a, flip, neighborhood=AROUND)._get_local_rule().stochastic


def _record(stochastic, settings):
    """Return the keywords a rule of a row receives first, and the tags of its system."""
    received = []

    def rule(values, *around, **kwargs):
        """Return the values unchanged and record the keywords."""
        received.append(kwargs)
        return values

    row = Element('row', values={'x': np.zeros(5), 'k': 2.0}, space=ROW)
    law = Interaction(
        'rule',
        roles={'target': 'row'},
        laws=lambda s: [UpdateLaw(s.target.x, rule, stochastic=stochastic, **settings)],
    )
    system = System(elements=[row], interactions=[law])
    tags = system.get_tags()
    system.prepare().simulate(until=1, random_state=0)
    return received[0], tags


@pytest.mark.parametrize('settings', [{'neighborhood': AROUND}, {'arrays': True}])
def test_update_rule_receives_every_keyword(settings):
    """Test that a local rule and an update of a whole array get the fixed keywords."""
    keywords, _ = _record(False, settings)
    assert set(keywords) == {'k', 'time', 'space', 'degree', 'random'}
    assert isinstance(keywords['time'], float)
    assert (keywords['space'], keywords['random']) == (ROW, None)
    assert np.all(np.asarray(keywords['k']) == 2.0)
    if settings.get('arrays'):
        assert keywords['degree'] is None
    else:
        assert keywords['degree'].tolist() == [2.0] * 5


def test_update_rule_with_kwargs_ignores_the_keywords_it_does_not_name():
    """Test that a rule naming none of the keywords runs, as `**kwargs` takes them."""
    rule = Interaction(
        'shift',
        roles={'target': 'row'},
        laws=lambda s: [UpdateLaw(s.target.x, lambda x, left, right, **kwargs: left, neighborhood=AROUND)],
    )
    row = Element('row', values={'x': [0.0, 1.0, 2.0, 3.0, 4.0]}, space=ROW)
    trajectory = System(elements=[row], interactions=[rule]).prepare().simulate(until=1)
    assert trajectory.get_values('row.x')[-1].tolist() == [4.0, 0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    ('settings', 'drawn'),
    [({'neighborhood': AROUND}, np.ndarray), ({'arrays': True}, np.random.Generator)],
)
def test_update_stochastic_rule_receives_random_and_keeps_the_tags(settings, drawn):
    """Test that `stochastic=True` gives `random` and leaves the tags as they are."""
    keywords, tags = _record(True, settings)
    _, deterministic = _record(False, settings)
    assert isinstance(keywords['random'], drawn)
    assert tags == deterministic
    if drawn is np.ndarray:
        assert keywords['random'].shape == (5,)
        assert np.all((keywords['random'] >= 0) & (keywords['random'] < 1))


def test_update_rule_naming_random_without_stochastic_receives_none():
    """Test that a rule naming `random` on a law not stochastic receives `None`."""
    received = []

    def draw(values, *, random, **kwargs):
        """Return the values and record what `random` is."""
        received.append(random)
        return values

    row = Element('row', values={'x': np.zeros(5)}, space=ROW)
    law = Interaction('draw', roles={'target': 'row'}, laws=lambda s: [UpdateLaw(s.target.x, draw, arrays=True)])
    System(elements=[row], interactions=[law]).prepare().simulate(until=1)
    assert received == [None]


@pytest.mark.parametrize(
    ('value', 'settings'),
    [(var('x') + 1, {}), (_shift, {}), (TotalisticSpec({1: 1}), {'neighborhood': AROUND})],
)
def test_update_check_rejects_stochastic_without_a_function_of_cells_or_arrays(value, settings):
    """Test that `stochastic=True` needs a function of cells or of a whole array."""
    with pytest.raises(ValidationError, match='is stochastic'):
        UpdateLaw(var('x'), value, stochastic=True, **settings)._check()


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'neighborhood': 'moore'}, 'Give a neighbourhood'),
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
        UpdateLaw(x, TotalisticSpec({1: 1}), **settings)._check()


class StepTransferInteraction(Interaction):
    """Moves of one unit along a ring of cells in one step, each with probability one half."""

    def __init__(self, label: str, *, target: Role) -> None:
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
