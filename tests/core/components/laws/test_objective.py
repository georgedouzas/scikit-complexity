"""Test the `_objective` module of `skcomplex.core.components.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import ObjectiveLaw
from skcomplex.core.errors import ValidationError


def test_run_evaluates_the_objective():
    """Running an objective gives its value at the given symbol values."""
    x = var('x')
    assert float(ObjectiveLaw((x - 2) ** 2, x).prepare().run({'x': 5.0}).value_) == pytest.approx(9.0)


def test_get_targets_reads_one_or_several_states():
    """An objective is minimised over one state or several."""
    x, y = var('x y')
    assert ObjectiveLaw(x**2, x)._get_targets() == (x,)
    assert ObjectiveLaw(x**2 + y**2, [x, y])._get_targets() == (x, y)


def test_prepare_rejects_a_function():
    """An objective given as a Python function raises an error, since it is minimised with its gradient."""
    with pytest.raises(ValidationError, match='symbolic expression'):
        ObjectiveLaw(lambda s: 0.0, var('x')).prepare()


def test_prepare_rejects_no_targets():
    """An objective with no state to vary raises an error."""
    with pytest.raises(ValidationError, match='names the states'):
        ObjectiveLaw(var('x') ** 2, []).prepare()


def test_get_targets_reads_the_controls():
    """An objective with controls steers them over a run, so they are its targets."""
    x, u = var('x u')
    law = ObjectiveLaw(x**2 + u**2, controls=[u], terminal=x)
    assert law.prepare().targets_ == (u,)
    assert law._get_law_requirements().time[0].__name__ == '_ContinuousTime'


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [({'targets': var('x'), 'controls': [var('u')]}, 'not both'), ({'targets': var('x'), 'terminal': 1}, 'controls')],
)
def test_prepare_rejects_targets_with_controls_or_a_terminal_cost(arguments, match):
    """An objective takes targets or controls, and a terminal cost only with controls."""
    with pytest.raises(ValidationError, match=match):
        ObjectiveLaw(var('x') ** 2, **arguments).prepare()


def test_values_in_order_read_one_value_or_a_sequence():
    """An objective holds one value, or a sequence of values minimised in order, and none asks for any point."""
    x, y = var('x y')
    assert ObjectiveLaw(x, [x, y])._get_values_in_order() == (x,)
    assert ObjectiveLaw([-x, abs(y)], [x, y])._get_values_in_order() == (-x, abs(y))
    assert ObjectiveLaw([], [x])._get_values_in_order() == ()


def test_objective_keeps_its_condition_and_its_policy():
    """An objective stores the condition under which it is sought and what a run does without a feasible point."""
    x, b = var('x b')
    law = ObjectiveLaw(-x, x, active=b > 0, infeasible='zero')
    assert (law.active, law.infeasible) == (b > 0, 'zero')
    assert ObjectiveLaw(-x, x).infeasible == 'raise'
