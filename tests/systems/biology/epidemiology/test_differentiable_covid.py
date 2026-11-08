"""Test the `_differentiable_covid` module of `skcomplex.systems.biology.epidemiology`."""

import jax
import jax.numpy as jnp
import numpy as np

from skcomplex.core.engine.backends import JaxBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import DifferentiableCovidAbmSystem

BETAS = ('beta_household', 'beta_school', 'beta_company')


def _total_infections(system):
    """Return the JAX function from the three betas to the total infections of 30 days at fixed uniform values."""
    backend = JaxBackend()
    problem = PassCompiler(backend=backend).prepare(system.prepare().system_).run().problem_
    run = JaxBackend.unroll(backend.traced_, problem.form, np.arange(31.0), draws=problem.draws)
    x0, p0 = jnp.asarray(problem.initial_state), jnp.asarray(problem.initial_parameters)
    columns = [np.arange(problem.parameters.size)[problem.parameters.locate(f'people.{name}')[1]] for name in BETAS]
    new = np.arange(problem.states.size)[problem.states.locate('people.new')[1]]
    key = jax.random.PRNGKey(0)

    def infections(betas):
        p = p0
        for column, beta in zip(columns, betas, strict=True):
            p = p.at[column].set(beta)
        return run(x0, p, key)[:, new].sum()

    return infections


def test_gradient_of_the_infections_agrees_with_finite_differences():
    """Test that the gradient of the total infections in the betas agrees with central finite differences."""
    with jax.enable_x64(True):
        infections = _total_infections(DifferentiableCovidAbmSystem(n_households=40))
        betas = jnp.array([0.9, 0.6, 0.3])
        gradient = np.asarray(jax.grad(infections)(betas))
        step = 1e-6
        expected = [
            (float(infections(betas.at[k].add(step))) - float(infections(betas.at[k].add(-step)))) / (2 * step)
            for k in range(3)
        ]
    np.testing.assert_allclose(gradient, expected, rtol=1e-5)
    assert np.all(gradient > 0)


def test_no_transmission_gives_no_infections():
    """Test that with every beta 0 the daily infections are 0, and NumPy agents stay 0 or 1."""
    system = DifferentiableCovidAbmSystem(beta_household=0.0, beta_school=0.0, beta_company=0.0)
    trajectory = simulate(system, until=30, random_state=0)
    assert np.all(trajectory.values('daily_infections') == 0)
    assert set(np.unique(trajectory.values('people.i'))) <= {0.0, 1.0}
    assert trajectory.values('people.i')[-1].sum() < trajectory.values('people.i')[0].sum()


def test_system_holds_the_tag_differentiable():
    """Test that the relaxed draws make the system differentiable."""
    assert DifferentiableCovidAbmSystem(n_households=5).get_tags().differentiable
