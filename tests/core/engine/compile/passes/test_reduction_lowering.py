"""Test the `_reduction_lowering` module of `skcomplex.core.engine.compile.passes`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec

from .....conftest import _weighted


class ValueElement(Element):
    """Member with a value and a gain."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        gain: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.gain = gain


def _step(x, gain, function, reduce):
    """Return the values after one step of a map coupling over every other member of the group."""
    members = GroupElement('members', element=ValueElement, size=len(x), values={'x': x, 'gain': gain})
    coupling = CouplingInteraction(
        'coupling',
        target='members',
        read='x',
        write='x',
        function=function,
        mode='group',
        reduce=reduce,
        form='map',
    )
    system = System(elements=[members], interactions=[coupling], time=DiscreteTime())
    return simulate(system, until=1).values('members.x')[-1]


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
    problem = PassCompiler(backend=backend()).prepare(_weighted(weights, inputs, reduce=reduce)).run().problem_
    rate = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = weights @ inputs / (len(inputs) if reduce == 'mean' else 1)
    np.testing.assert_allclose(problem.states.read(rate, 'hidden.x'), expected, atol=1e-12)
