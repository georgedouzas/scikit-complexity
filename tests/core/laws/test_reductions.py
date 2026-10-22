"""Test the symbolic reductions that laws use over the members of a group."""

import numpy as np
import pytest
import scipy.sparse
from sage.all__sagemath_symbolics import cos, tanh

from skcomplex.core.base import ValidationError, symbol
from skcomplex.core.laws import (
    Symbols,
    both,
    count,
    has_reductions,
    maximum,
    mean,
    minimum,
    reductions_in,
    total,
    variance,
)
from skcomplex.core.system import reduce_values


@pytest.mark.parametrize(
    ('function', 'kind'),
    [(total, 'sum'), (mean, 'mean'), (maximum, 'max'), (minimum, 'min'), (variance, 'var')],
    ids=['total', 'mean', 'maximum', 'minimum', 'variance'],
)
def test_reductions_stay_unevaluated(function, kind):
    """Test that a reduction of a member symbol stays an unevaluated call of its kind."""
    x = symbol('x')
    found = reductions_in(1 + function(x**2))
    assert [item.kind for item in found] == [kind]
    assert str(found[0].argument) == 'x^2'


def test_reductions_survive_simplify():
    """Test that reductions with conditions and weights come back unchanged from Maxima."""
    x = symbol('x')
    weights = np.arange(9.0).reshape(3, 3)
    expression = -x + tanh(total(x, where=x > 0.5)) + 2 * mean(cos(x), where=both(x > 0, x < 1))
    expression += total(x, weights=weights) + count(where=x > 0)
    for simplified in (expression.simplify(), expression.simplify_full()):
        assert {str(item.call) for item in reductions_in(simplified)} == {
            str(item.call) for item in reductions_in(expression)
        }


def test_reductions_record_condition_weights_and_group():
    """Test that a reduction holds its condition, its weights, and the group it names."""
    x, y = symbol('x'), symbol('y')
    weights = scipy.sparse.csr_matrix(np.eye(2))
    (found,) = reductions_in(mean(x, over=Symbols(t=symbol('t'), y=y), where=x > 0, weights=weights))
    assert str(found.condition) == 'x > 0'
    assert str(found.over) == str(y)
    assert (found.weights != weights).nnz == 0
    assert has_reductions(found.call)
    assert not has_reductions(x + 1)


def test_count_takes_no_expression():
    """Test that a count reads only its condition."""
    x = symbol('x')
    (found,) = reductions_in(count(where=x > 0.5))
    assert found.kind == 'count'
    assert found.argument == 1


@pytest.mark.parametrize(
    ('call', 'match'),
    [
        (lambda x: maximum(x, over='group'), 'namespace'),
        (lambda x: total(x, weights=np.ones(3)), 'matrix'),
    ],
    ids=['group name', 'vector weights'],
)
def test_malformed_reductions_are_rejected(call, match):
    """Test that a group given by name and weights that are not a matrix raise errors."""
    with pytest.raises(ValidationError, match=match):
        call(symbol('x'))


def test_count_reduces_values():
    """Test that the reduction table counts members, also none."""
    assert reduce_values([1.0, 2.0], 'count') == 2.0
    assert reduce_values([], 'count') == 0.0
