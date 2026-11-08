"""Test the `_jax` module of `skcomplex.core.engine.backends`."""

from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import bernoulli, categorical
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import JaxBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import BackendError, SolverError, ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.core.types import FormKind
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem

from ....conftest import CallableGrowthInteraction, HoldingElement, _correlated, _weighted


def _final(system, backend, until):
    """Return the last recorded state of a run of a system on a backend."""
    trajectory = simulate(system, until=until, backend=backend, random_state=0)
    return np.array([trajectory.values(name)[-1] for name in trajectory.names])


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


KEY = jax.random.PRNGKey(0)


def _relaxed_run(size, temperature=None, hard=False, kind='bernoulli', key=KEY):
    """Return the JAX run of a group whose members draw a relaxed value, and the probability of 1."""

    class DrawnElement(Element):
        """Element whose state is a relaxed draw of a probability."""

        def __init__(
            self,
            label: str,
            *,
            b: Annotated[float, StateSpec()] = 0.0,
            p: Annotated[float, ParameterSpec()] = 0.3,
        ) -> None:
            super().__init__(label)
            self.b = b
            self.p = p

        def _get_laws(self, s):
            """Return the drawn update."""
            if kind == 'bernoulli':
                return [UpdateLaw(s.b, bernoulli('b', s.p, temperature=temperature, hard=hard))]
            return [UpdateLaw(s.b, categorical('b', [1 - s.p, s.p], temperature=temperature, hard=hard))]

    system = System(elements=[GroupElement('g', element=DrawnElement, size=size)], time=DiscreteTime())
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
