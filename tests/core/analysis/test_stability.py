"""Test fixed points, Jacobians, stability, nullclines, and divergence."""

import pytest
from sage.all__sagemath_symbolics import SR

from skcomplex.core.analysis import divergence, fixed_points, jacobian, nullclines, stability
from skcomplex.core.base import AnalysisError
from skcomplex.models.biology.ecology import lotka_volterra
from skcomplex.models.biology.epidemiology import sir
from skcomplex.models.mathematics.dynamical_systems import logistic_map, lorenz
from skcomplex.models.physics.mechanics import harmonic_oscillator


def test_lotka_volterra_centre():
    """Test that the predator-prey interior point is `(d / b, r / a)` with purely imaginary eigenvalues, a centre."""
    system = lotka_volterra()
    points = fixed_points(system)
    d, b, r, a = (SR.var(name) for name in ('d_death', 'b_predation', 'r_growth', 'a_predation'))
    interior = next(point for point in points if point['prey.n'] != 0)
    assert bool(interior['prey.n'] == d / b)
    assert bool(interior['predators.n'] == r / a)
    result = stability(system, interior)
    assert result.classification == 'centre'
    assert abs(result.eigenvalues.real).max() < 1e-9
    assert stability(system, {'prey.n': 0, 'predators.n': 0}).classification == 'saddle'


def test_logistic_map_fixed_point_is_stable():
    """Test that the logistic map's fixed point `1 - 1 / r` is stable at r = 2.5 and 0 is unstable."""
    system = logistic_map(r=2.5)
    classes = {
        round(float(point['map.x'].subs({SR.var('r_map'): 2.5})), 9): stability(system, point).classification
        for point in fixed_points(system)
    }
    assert classes == {0.6: 'stable', 0.0: 'unstable'}


def test_parameters_substitute_before_solving():
    """Test that parameter values given by path give numeric fixed points."""
    points = fixed_points(lorenz(), parameters={'lorenz.rho': 28.0, 'lorenz.beta': 8 / 3})
    assert sorted(round(float(point['lorenz.z']), 6) for point in points) == [0.0, 27.0, 27.0]


def test_volume_preservation():
    """Test that the frictionless harmonic oscillator has divergence 0 and Lorenz contracts volumes."""
    assert divergence(harmonic_oscillator()) == 0
    sigma, beta = SR.var('sigma_lorenz'), SR.var('beta_lorenz')
    assert bool(divergence(lorenz()) == -sigma - beta - 1)


def test_jacobian_and_nullclines():
    """Test the Jacobian's shape and the nullclines of a two-state system."""
    system = lotka_volterra()
    assert jacobian(system).dimensions() == (2, 2)
    assert set(nullclines(system)) == {'prey.n', 'predators.n'}


def test_jump_process_has_no_vector_field():
    """Test that stochastic jump systems are rejected."""
    with pytest.raises(AnalysisError, match='jump process'):
        jacobian(sir(stochastic=True))
