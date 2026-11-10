"""Test the `_differentiable_covid` module of `skcomplex.systems.biology.epidemiology`."""

import jax
import jax.numpy as jnp
import numpy as np

from skcomplex.core.engine.backends import JaxBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.experiments import calibrate
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
    assert np.all(trajectory.get_values('daily_infections') == 0)
    assert set(np.unique(trajectory.get_values('people.i'))) <= {0.0, 1.0}
    assert trajectory.get_values('people.i')[-1].sum() < trajectory.get_values('people.i')[0].sum()


def test_system_holds_the_tag_differentiable():
    """Test that the relaxed draws make the system differentiable."""
    assert DifferentiableCovidAbmSystem(n_households=5).get_tags().differentiable


def test_variational_calibration_recovers_the_three_betas():
    """Test that variational calibration of the mean daily infections recovers the three betas within 0.15."""
    times = np.arange(31.0)
    runs = [simulate(DifferentiableCovidAbmSystem(), until=30, random_state=seed) for seed in range(20)]
    paths = ('cases.children', 'cases.adults')
    data = {path: (times, np.mean([run.get_values(path) for run in runs], axis=0)) for path in paths}
    result = calibrate(
        DifferentiableCovidAbmSystem(),
        data,
        dict.fromkeys(BETAS, (0.0, 2.0)),
        method='variational',
        loss='likelihood',
        n_samples=300,
        budget=4000,
        random_state=0,
    )
    for name, truth in zip(BETAS, (0.9, 0.6, 0.3), strict=True):
        assert abs(result.estimates[name] - truth) < 0.15
    assert result.success
