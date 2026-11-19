"""Test the `_jax` module of `skcomplex.core.engine.backends`."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import bernoulli, categorical
from skcomplex.core.engine.backends import JaxBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import BackendError, SolverError, ValidationError, ValidationWarning
from skcomplex.core.systems import System
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem

from ....conftest import CallableGrowthInteraction, HoldingElement, _correlated, _weighted

KEY = jax.random.PRNGKey(0)


def _final(system, backend, until):
    """Return the last recorded state of a run of a system on a backend."""
    trajectory = system.set_params(backend=backend).prepare().simulate(until, random_state=0)
    return np.array([trajectory.get_values(name)[-1] for name in trajectory.names])


@pytest.mark.parametrize(('factory', 'until'), [(LogisticMapSystem, 50), (LotkaVolterraSystem, 20.0)])
def test_jax_agrees_with_numpy(factory, until):
    """Test that the JAX backend runs `LogisticMapSystem` and `LotkaVolterraSystem` as NumPy does, within 1e-10."""
    expected = _final(factory(), NumpyBackend(), until)
    np.testing.assert_allclose(_final(factory(), JaxBackend(), until), expected, atol=1e-10)


@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_jax_weights_of_a_shaped_parameter(reduce):
    """Test that a reduction weighted by a matrix parameter gives every reading member its row times the members."""
    weights = np.random.default_rng(0).normal(size=(4, 2))
    inputs = np.array([1.5, -0.5])
    problem = _weighted(weights, inputs, reduce=reduce).set_params(backend=JaxBackend()).prepare()._problem
    rate = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = weights @ inputs / (len(inputs) if reduce == 'mean' else 1)
    np.testing.assert_allclose(problem.states.read(rate, 'hidden.x'), expected, atol=1e-12)


def test_jax_rejects_noise():
    """Test that stochastic differential equations, which the JAX backend does not write, raise an error."""
    with pytest.raises(SolverError, match='noise-unsupported'):
        _correlated(None).set_params(backend=JaxBackend()).prepare()


def test_jax_rejects_a_callable_law():
    """Test that a law given as a Python callable, which has no JAX form, raises an error."""
    system = System(
        elements=[HoldingElement('a', n=1.0)],
        interactions=[CallableGrowthInteraction('g', target='a')],
        backend=JaxBackend(),
    )
    with pytest.raises(BackendError, match='Python function'):
        system.prepare()


def _relaxed_run(size, temperature=None, hard=False, kind='bernoulli', key=KEY):
    """Return the JAX run of a group whose members draw a relaxed value, and the probability of 1."""

    def laws(s):
        """Return the update of every member to a relaxed draw of its probability."""
        b, p = s.target.b, s.target.p
        if kind == 'bernoulli':
            return [UpdateLaw(b, bernoulli('b', p, temperature=temperature, hard=hard))]
        return [UpdateLaw(b, categorical('b', [1 - p, p], temperature=temperature, hard=hard))]

    drawn = Element('g', values={'b': np.zeros(size), 'p': np.full(size, 0.3)}, size=size)
    system = System(elements=[drawn], interactions=[Interaction('draw', roles={'target': 'g'}, laws=laws)])
    backend = JaxBackend()
    problem = PassCompiler(backend=backend).prepare(system).run().problem_
    run = JaxBackend.unroll(backend.traced_, FormKind.DISCRETE, np.array([1.0]), draws=problem.draws)
    x0 = jnp.asarray(problem.initial_state)

    def values(p):
        return run(x0, jnp.full(size, p), key)[0]

    return values


@pytest.mark.parametrize('kind', ['bernoulli', 'categorical'])
def test_jax_differentiates_relaxed_draws(kind):
    """Test that the gradient of the mean of relaxed draws agrees with central finite differences."""
    with jax.enable_x64(True):
        values = _relaxed_run(200, temperature=0.5, kind=kind)

        def mean(p):
            return jnp.mean(values(p))

        gradient = float(jax.grad(mean)(0.3))
        step = 1e-6
        expected = (float(mean(0.3 + step)) - float(mean(0.3 - step))) / (2 * step)
    assert gradient == pytest.approx(expected, rel=1e-6)
    assert gradient > 0


def test_jax_relaxed_draws_tend_to_exact_draws():
    """Test that relaxed Bernoulli draws at temperature 0.01 have the mean of exact draws."""
    with jax.enable_x64(True):
        drawn = np.asarray(_relaxed_run(20000, temperature=0.01)(0.3))
    assert abs(drawn.mean() - 0.3) < 0.02


def test_jax_hard_draws_are_exact_with_the_gradient_of_the_relaxation():
    """Test that hard relaxed draws are 0 or 1 and have the gradient of the relaxation."""
    with jax.enable_x64(True):
        hard, soft = _relaxed_run(1000, 0.5, hard=True), _relaxed_run(1000, 0.5)
        drawn = np.asarray(hard(0.3))
        gradients = [float(jax.grad(lambda p, run=run: jnp.mean(run(p)))(0.3)) for run in (hard, soft)]
    assert set(np.unique(drawn)) == {0.0, 1.0}
    assert gradients[0] == pytest.approx(gradients[1], rel=1e-12)


def test_jax_run_of_a_drawing_map_needs_a_key():
    """Test that the run of a map that draws raises an error without a key."""
    with pytest.raises(ValidationError, match='JAX key'):
        _relaxed_run(2, temperature=0.5, key=None)(0.3)


def test_jax_reports_discrete_draws_without_a_temperature():
    """Test that a Bernoulli draw without a temperature gives a `not-differentiable` issue naming it on JAX."""

    flip = Interaction('flip', roles={'target': 'a'}, laws=lambda s: [UpdateLaw(s.target.b, bernoulli('flip', 0.4))])
    system = System(elements=[Element('a', values={'b': 0.0})], interactions=[flip])
    with pytest.warns(ValidationWarning) as caught:
        system.set_params(backend=JaxBackend()).prepare()
    (issue,) = {str(item.message) for item in caught}
    assert issue.endswith('[not-differentiable]')
    assert "'flip'" in issue
    assert 'temperature' in issue
    with warnings.catch_warnings():
        warnings.simplefilter('error', ValidationWarning)
        system.set_params(backend=NumpyBackend()).prepare()
