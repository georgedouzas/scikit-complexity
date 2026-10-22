"""Test the kinetic models of active particles: welfare, immune competition, opinions, evacuation, and criminality."""

import numpy as np
import pytest

from skcomplex.core.compile import validate
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ParticleKinetics
from skcomplex.models import create, describe


def _trajectory(name, until, dt, solver='auto', **options):
    """Return the trajectory of a model run with its standard observables."""
    system = create(name, **options)
    observables = describe(name).observables_of(system)
    return (
        Simulation(system, dt=dt, observables=observables, solver=solver, random_state=0).run(until=until).trajectory_
    )


def test_welfare_policy_lowers_the_gini_coefficient():
    """Test that a policy moving activity from the richest to the poorest classes lowers the stationary Gini."""
    without = _trajectory('kinetic_wealth_welfare', 200, 50.0)
    with_policy = _trajectory('kinetic_wealth_welfare', 200, 50.0, tax=0.2)
    assert with_policy['gini'][-1] < without['gini'][-1] - 0.05
    assert with_policy['gini'][-1] == pytest.approx(with_policy['gini'][-2], abs=1e-4)
    np.testing.assert_allclose(without['density'], 1.0, rtol=1e-10)


def test_wealth_gini_matches_its_definition():
    """Test that the Gini observable equals the mean absolute difference of wealth over twice the mean."""
    trajectory = _trajectory('kinetic_wealth_welfare', 20, 20.0)
    f = np.asarray(trajectory['wealth.f'][-1])
    wealth = np.linspace(0.0, 1.0, f.size)
    expected = (f[:, None] * f[None, :] * np.abs(wealth[:, None] - wealth[None, :])).sum() / (
        2 * f.sum() * (f @ wealth)
    )
    assert trajectory['gini'][-1] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(('activation', 'decays'), [(0.75, True), (1.0, True), (0.0, False), (0.25, False)])
def test_immune_response_above_threshold_clears_the_tumour(activation, decays):
    """Test that the tumour density decays above the immune activation threshold and grows below it."""
    trajectory = _trajectory('immune_competition', 20, 5.0, activation=activation)
    density = trajectory['tumour_density']
    assert bool(density[-1] < density[0]) is decays
    assert bool(density[-1] > density[0]) is not decays


def test_opinions_reach_consensus_or_polarise():
    """Test that a wide confidence leads to consensus and a narrow one with repulsion to polarisation."""
    consensus = _trajectory('kinetic_opinion', 60, 60.0, confidence=6, repulsion=0.0)
    polarised = _trajectory('kinetic_opinion', 60, 60.0, confidence=1, repulsion=0.2)
    assert consensus['opinion_variance'][-1] < 0.1 < consensus['opinion_variance'][0]
    assert polarised['opinion_variance'][-1] > 0.9
    extremes = np.asarray(polarised['opinions.f'][-1])[[0, -1]].sum()
    assert extremes > 0.9


def test_evacuation_empties_the_building():
    """Test that the evacuated fraction grows monotonically to the whole crowd and the density is kept."""
    trajectory = _trajectory('evacuation', 40, 2.0)
    evacuated = trajectory['evacuated']
    assert evacuated[0] == 0.0
    assert np.all(np.diff(evacuated) >= -1e-12)
    assert evacuated[-1] > 0.99
    np.testing.assert_allclose(np.asarray(trajectory['crowd.f']).sum(axis=1), 1.0, rtol=1e-9)


def test_evacuation_runs_with_particles():
    """Test that the particle solver evacuates the building like the gain and loss equations."""
    equations = _trajectory('evacuation', 6, 3.0)
    particles = _trajectory('evacuation', 6, 3.0, solver=ParticleKinetics(n=20_000, dt=0.01))
    np.testing.assert_allclose(particles['evacuated'], equations['evacuated'], atol=0.03)


def test_security_forces_respond_to_criminality():
    """Test that recruitment reads the mean criminality, so security grows while crime is high and crime recedes."""
    trajectory = _trajectory('criminality', 60, 2.0)
    security = trajectory['security_forces']
    assert security.max() > 5 * security[0]
    assert trajectory['mean_criminality'][-1] < trajectory['mean_criminality'][0]
    weak = _trajectory('criminality', 60, 60.0, deterrence=0.0, recruitment=0.0)
    assert weak['mean_criminality'][-1] > 0.5


@pytest.mark.parametrize('name', ['kinetic_wealth_welfare', 'immune_competition', 'kinetic_opinion', 'evacuation'])
def test_kinetic_models_run_with_the_particle_solver(name):
    """Test that the kinetic models of encounters alone validate with the particle solver."""
    assert validate(create(name), solver='particles').ok


def test_criminality_needs_an_ode_solver():
    """Test that the macro-scale recruitment, a rate law, rules out the particle solver."""
    issues = validate(create('criminality'), solver='particles').errors
    assert [issue.code for issue in issues] == ['solver-unsupported']
