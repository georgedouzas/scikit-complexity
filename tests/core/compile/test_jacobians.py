"""Test the exact Jacobian of drifts that read reductions, and its sparsity pattern."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin, tanh

from skcomplex.core.base import ValidationError
from skcomplex.core.compile import compile_system
from skcomplex.core.model import Element, Rate, System, count, maximum, mean, minimum, total, variance, where
from skcomplex.models.physics.synchronization import nonlocal_ring_chimera

RNG = np.random.default_rng(1)
PAYOFF = RNG.normal(size=(20, 20))


def _finite_differences(problem, x, p, step=1e-6):
    """Return the central finite-difference Jacobian of a drift."""
    columns = []
    for index in range(x.size):
        shift = np.zeros(x.size)
        shift[index] = step
        columns.append((problem.drift(0.0, x + shift, p) - problem.drift(0.0, x - shift, p)) / (2 * step))
    return np.stack(columns, axis=1)


@pytest.mark.parametrize(
    'laws',
    [
        lambda s: [Rate(s.x, -s.x + tanh(total(s.x)) + mean(s.x**2))],
        lambda s: [
            Rate(s.x, s.x * (total(s.x, weights=PAYOFF) - total(s.x * total(s.x, weights=PAYOFF)))),
            Rate(
                s.y,
                maximum(s.x * s.y)
                + variance(s.y)
                - minimum(sin(s.x))
                + mean(s.y, where=s.x > 0) * count(where=s.y > 0),
            ),
        ],
        lambda s: [Rate(s.x, where(s.x > 0, total(s.y**2), -s.x)), Rate(s.y, -s.y)],
    ],
    ids=['tanh of total', 'every kind', 'condition'],
)
def test_jacobian_against_finite_differences(laws):
    """Test that the exact Jacobian equals central finite differences and that its pattern covers every entry."""
    Member = Element.define('Member', states={'x': 0.0, 'y': 0.0}, laws=laws)
    system = System(elements=[Member.group('group', 20, x=RNG.normal(size=20), y=RNG.normal(size=20))])
    problem = compile_system(system)
    x, p = problem.initial_state, problem.initial_parameters
    exact = problem.jacobian(0.0, x, p).toarray()
    estimate = _finite_differences(problem, x, p)
    np.testing.assert_allclose(exact, estimate, atol=1e-6)
    pattern = problem.jacobian.pattern().toarray()
    assert pattern[np.abs(estimate) > 1e-9].all()


def test_jacobian_of_pair_sums_is_not_derived():
    """Test that the exact Jacobian of a pair interaction raises an error that names the interaction."""
    problem = compile_system(nonlocal_ring_chimera(10, R=2))
    with pytest.raises(ValidationError, match=r'coupling.*finite differences'):
        problem.jacobian(0.0, problem.initial_state, problem.initial_parameters)


def test_pattern_is_omitted_above_the_limit():
    """Test that the pattern of a dense block above the size limit is omitted."""
    Member = Element.define('Wide', states={'x': 0.0}, laws=lambda s: [Rate(s.x, total(s.x))])
    problem = compile_system(System(elements=[Member.group('group', 3000)]))
    assert problem.jacobian.pattern() is None
