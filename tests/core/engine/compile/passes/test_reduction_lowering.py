"""Test the `_reduction_lowering` module of `skcomplex.core.engine.compile.passes`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import count, neighbors, total
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.systems import System

from .....conftest import _weighted


def _step(x, gain, function, reduce):
    """Return the values after one step of a map coupling over every other member of the group."""

    def laws(s):
        """Return the map of every member to the reduction of the pair terms over the other members."""
        pairs = total(function(SimpleNamespace(member=s.target, other=s.target.other)), over=s.target)
        reduced = pairs - function(SimpleNamespace(member=s.target, other=s.target))
        return [UpdateLaw(s.target.x, reduced if reduce == 'sum' else reduced / (count(over=s.target) - 1))]

    members = Element('members', values={'x': x, 'gain': gain}, size=len(x))
    coupling = Interaction('coupling', roles={'target': 'members'}, laws=laws)
    return System(elements=[members], interactions=[coupling]).prepare().simulate(until=1).get_values('members.x')[-1]


def _separable(s):
    """Return a pair term that is a sum of products of member and other factors."""
    return s.member.gain * (s.other.x**2 - 3 * s.member.x * s.other.x) + 2


@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_run_reduces_separable_pairs_over_the_group(reduce):
    """Test that a pair term of products of member and other factors reduces to its sum or mean over the others."""
    x, gain = np.array([0.5, 1.0, 2.0, 4.0, -1.0]), np.array([1.0, 2.0, 0.5, 3.0, 1.5])
    terms = gain[:, None] * (x[None, :] ** 2 - 3 * x[:, None] * x[None, :]) + 2
    totals = terms.sum(axis=1) - np.diag(terms)
    expected = totals if reduce == 'sum' else totals / (len(x) - 1)
    np.testing.assert_allclose(_step(x, gain, _separable, reduce), expected, rtol=1e-12)


def test_run_reduces_pairs_that_do_not_separate_pair_by_pair():
    """Test that a pair term that does not factor into member and other parts still reduces over every pair."""
    x, gain = np.array([0.5, 1.0, 2.0, 4.0]), np.ones(4)
    terms = np.sin(x[None, :] - x[:, None])
    expected = terms.sum(axis=1) - np.diag(terms)
    np.testing.assert_allclose(_step(x, gain, lambda s: sin(s.other.x - s.member.x), 'sum'), expected, atol=1e-12)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_reduction_lowering_weights_of_a_shaped_parameter(backend, reduce):
    """Test that a reduction weighted by a matrix parameter gives every reading member its row times the members."""
    weights = np.random.default_rng(0).normal(size=(4, 2))
    inputs = np.array([1.5, -0.5])
    problem = _weighted(weights, inputs, reduce=reduce).set_params(backend=backend()).prepare()._problem
    rate = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = weights @ inputs / (len(inputs) if reduce == 'mean' else 1)
    np.testing.assert_allclose(problem.states.read(rate, 'hidden.x'), expected, atol=1e-12)


@pytest.mark.parametrize(('kind', 'padded'), [('fixed', 'constant'), ('zero-flux', 'symmetric')])
def test_reduction_lowering_reads_a_state_with_its_own_edges_beyond_them(kind, padded):
    """Test that a neighbour sum reads the fixed value, or the mirrored site, beyond the edges of a state."""
    field = np.random.default_rng(0).random(6)
    boundary = {'u': (kind, 0.5) if kind == 'fixed' else kind}
    rule = Interaction(
        'c',
        roles={'target': 'm'},
        laws=lambda s: [RateLaw(s.target.u, neighbors(s.target.other.u**2, reduce='sum'))],
    )
    element = Element('m', values={'u': field}, space=LatticeSpace(6, boundary=boundary))
    problem = System(elements=[element], interactions=[rule]).prepare()._problem
    rate = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'm.u')
    extended = np.pad(field, 1, mode=padded, **({'constant_values': 0.5} if kind == 'fixed' else {})) ** 2
    assert np.abs(np.ravel(rate) - (extended[:-2] + extended[2:])).max() < 1e-12
