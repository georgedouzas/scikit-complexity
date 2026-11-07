"""Test the `_jax` module of `skcomplex.core.engine.backends`."""

import numpy as np
import pytest

from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import JaxBackend, NumpyBackend
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import BackendError, SolverError
from skcomplex.systems.biology.ecology import lotka_volterra
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem

from ....conftest import CallableGrowthInteraction, HoldingElement, _correlated, _weighted


def _final(system, backend, until):
    """Return the last recorded state of a run of a system on a backend."""
    trajectory = simulate(system, until=until, backend=backend, random_state=0)
    return np.array([trajectory.values(name)[-1] for name in trajectory.names])


@pytest.mark.parametrize(('factory', 'until'), [(LogisticMapSystem, 50), (lotka_volterra, 20.0)])
def test_jax_agrees_with_numpy(factory, until):
    """Test that the JAX backend runs `LogisticMapSystem` and `lotka_volterra` as NumPy does, within 1e-10."""
    expected = _final(factory(), NumpyBackend(), until)
    np.testing.assert_allclose(_final(factory(), JaxBackend(), until), expected, atol=1e-10)


@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_jax_weights_of_a_shaped_parameter(reduce):
    """Test that a reduction weighted by a matrix parameter gives every reading member its row times the members."""
    weights = np.random.default_rng(0).normal(size=(4, 2))
    inputs = np.array([1.5, -0.5])
    problem = PassCompiler(backend=JaxBackend()).prepare(_weighted(weights, inputs, reduce=reduce)).run().problem_
    rate = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = weights @ inputs / (len(inputs) if reduce == 'mean' else 1)
    np.testing.assert_allclose(problem.states.read(rate, 'hidden.x'), expected, atol=1e-12)


def test_jax_rejects_noise():
    """Test that stochastic differential equations, which the JAX backend does not write, raise an error."""
    with pytest.raises(SolverError, match='noise-unsupported'):
        PassCompiler(backend=JaxBackend()).prepare(_correlated(None)).run()


def test_jax_rejects_a_callable_law():
    """Test that a law given as a Python callable, which has no JAX form, raises an error."""
    system = System(elements=[HoldingElement('a', n=1.0)], interactions=[CallableGrowthInteraction('g', target='a')])
    with pytest.raises(BackendError, match='Python function'):
        PassCompiler(backend=JaxBackend()).prepare(system).run()
