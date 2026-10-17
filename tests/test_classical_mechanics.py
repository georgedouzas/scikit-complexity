"""Test the classical mechanics module."""

import re

import pytest
from sage.all import assume, symbolic_expression, var

from skcomplex.component import EuclideanSpace
from skcomplex.system.physics import (
    ClassicalMechanicsSystem,
    ExternalForce,
    ExternalPotential,
    InternalForce,
    InternalPotential,
    PointParticle,
)


@pytest.mark.parametrize('m', ['m', var('m__point')])
def test_point_particle_wrong_mass_type(m):
    """Test raising an error for wrong mass type."""
    with pytest.raises(TypeError, match='m must be an instance of {int, float}'):
        PointParticle('point', m)


@pytest.mark.parametrize('m', [0.0, -2.0])
def test_point_particle_wrong_mass_value(m):
    """Test raising an error for wrong mass type."""
    with pytest.raises(ValueError, match=f'm == {m}, must be > 0.0.'):
        PointParticle('point', m)


@pytest.mark.parametrize('label', ['point', 'sphere'])
def test_point_particle_default_mass(label):
    """Test point particle class for default mass."""
    point_particle = PointParticle(label)
    assert point_particle.m is None
    assert bool(var(f'm__{label}') == point_particle.m_)


@pytest.mark.parametrize('m', [2.0, 4])
def test_point_particle_numeric_mass(m):
    """Test point particle class for numeric mass."""
    point_particle = PointParticle('point', m)
    assert point_particle.m is m
    assert isinstance(point_particle.m_, float)
    assert point_particle.m_ == m


@pytest.mark.parametrize('F', [4.5, symbolic_expression(6.0)])
def test_internal_force_wrong_force_type(F):
    """Test raising an error for wrong force type."""
    error_msg = f'Parameter `F` should be a list of numeric elements or symbolic expressions. Got `{F}` instead.'
    with pytest.raises(TypeError, match=error_msg):
        InternalForce('gravity', '1', '2', F)


@pytest.mark.parametrize('F', [[5.6, 'F', 3.4], 'F'])
def test_internal_force_wrong_force_value(F):
    """Test raising an error for wrong force value."""
    error_msg = f'Parameter `F` should be a list of numeric elements or symbolic expressions. Got `{F}` instead.'
    with pytest.raises(AssertionError, match=re.escape(error_msg)):
        InternalForce('gravity', '1', '2', F)


@pytest.mark.parametrize('label', ['gravity', 'contact'])
def test_internal_force_default_force(label):
    """Test internal force class for default force."""
    internal_force = InternalForce(label, '1', '2')
    assert internal_force.F is None
    assert bool(var(f'F__{label}__1__2') == internal_force.F_)


@pytest.mark.parametrize(
    'F',
    [
        [1.5, 2, 3.0],
        [0, 4, 2],
        [symbolic_expression('2 * x'), symbolic_expression('x + y^2')],
        [var('x') + var('y'), 5.0],
    ],
)
def test_internal_force(F):
    """Test internal force class."""
    internal_force = InternalForce('internal', '1', '2', F)
    assert internal_force.F is F
    assert F == internal_force.F_


@pytest.mark.parametrize('V', [[4.5], 'V'])
def test_internal_potential_wrong_potential_type(V):
    """Test raising an error for wrong potential type."""
    error_msg = f'Parameter `V` should be a numeric element or symbolic expression. Got `{V}` instead.'
    with pytest.raises(TypeError, match=re.escape(error_msg)):
        InternalPotential('gravity', '1', '2', V)


@pytest.mark.parametrize('label', ['gravity', 'contact'])
def test_internal_potential_default_potential(label):
    """Test internal potential class for default potential."""
    internal_potential = InternalPotential(label, '1', '2')
    assert internal_potential.V is None
    assert bool(var(f'V__{label}__1__2') == internal_potential.V_)


@pytest.mark.parametrize('V', [2.0, var('x') + 5.0, symbolic_expression('x + y^2')])
def test_internal_potential(V):
    """Test internal potential class for default potential."""
    internal_potential = InternalPotential('internal', '1', '2', V)
    assert internal_potential.V is V
    assert symbolic_expression(V) == internal_potential.V_


@pytest.mark.parametrize('F', [1.5, symbolic_expression(2.0)])
def test_external_force_wrong_force_type(F):
    """Test raising an error for wrong force type."""
    error_msg = f'Parameter `F` should be a list of numeric elements or symbolic expressions. Got `{F}` instead.'
    with pytest.raises(TypeError, match=error_msg):
        ExternalForce('gravity', '1', F)


@pytest.mark.parametrize('F', [[1.6, 'Force', 3.4], 'Force'])
def test_external_force_wrong_force_value(F):
    """Test raising an error for wrong force value."""
    error_msg = f'Parameter `F` should be a list of numeric elements or symbolic expressions. Got `{F}` instead.'
    with pytest.raises(AssertionError, match=re.escape(error_msg)):
        ExternalForce('gravity', '1', F)


@pytest.mark.parametrize('label', ['gravity', 'contact'])
def test_external_force_default_force(label):
    """Test external force class for default force."""
    external_force = ExternalForce(label, '1')
    assert external_force.F is None
    assert bool(var(f'F__{label}__1') == external_force.F_)


@pytest.mark.parametrize(
    'F',
    [
        [1.5, 2, 3.0],
        [0, 4, 2],
        [symbolic_expression('2 * x^3'), symbolic_expression('x^2 + y^2')],
        [var('x') + var('y'), 2.0],
    ],
)
def test_external_force(F):
    """Test external force class."""
    external_force = ExternalForce('internal', '1', F)
    assert external_force.F is F
    assert F == external_force.F_


@pytest.mark.parametrize('V', [[4.5], 'V'])
def test_external_potential_wrong_potential_type(V):
    """Test raising an error for wrong potential type."""
    error_msg = f'Parameter `V` should be a numeric element or symbolic expression. Got `{V}` instead.'
    with pytest.raises(TypeError, match=re.escape(error_msg)):
        ExternalPotential('gravity', '1', V)


@pytest.mark.parametrize('label', ['gravity', 'contact'])
def test_external_potential_default_potential(label):
    """Test external potential class for default potential."""
    external_potential = ExternalPotential(label, '1')
    assert external_potential.V is None
    assert bool(var(f'V__{label}__1') == external_potential.V_)


@pytest.mark.parametrize('V', [3.0, var('x') + 1.0, symbolic_expression('x^2 + y^2')])
def test_external_potential(V):
    """Test external potential class for default potential."""
    external_potential = ExternalPotential('internal', '1', V)
    assert external_potential.V is V
    assert symbolic_expression(V) == external_potential.V_


def test_motion_with_constant_velocity():
    """Test the solution of dynamic equations of a free particle."""
    system = ClassicalMechanicsSystem(particles=[PointParticle('1')], space=EuclideanSpace('euclidean', n_dim=3))
    system.simulate()
    solutions = system.dynamic_equations_['solutions']
    assert solutions['1'][0] == symbolic_expression('_K2 * t + _K1')
    assert solutions['1'][1] == symbolic_expression('_K2 * t + _K1')
    assert solutions['1'][2] == symbolic_expression('_K2 * t + _K1')


def test_motion_with_constant_acceleration():
    """Test the solution of dynamic equations under constant external force."""
    system = ClassicalMechanicsSystem(
        particles=[PointParticle('particle')],
        external_interactions=[ExternalForce('constant', 'particle', [var('F'), 0, 0])],
        space=EuclideanSpace('euclidean', n_dim=3),
    )
    system.simulate()
    solutions = system.dynamic_equations_['solutions']
    assert solutions['particle'][0] == symbolic_expression('_K2 * t + 1 / 2 * F * t**2 / m__particle + _K1')
    assert solutions['particle'][1] == symbolic_expression('_K2 * t + _K1')
    assert solutions['particle'][2] == symbolic_expression('_K2 * t + _K1')


def test_motion_one_dimensional_harmonic_oscillator():
    """Test the solution of dynamic equations of harmonic oscillator."""
    k, x__sphere = var('k x__sphere')
    assume(k > 0)
    system = ClassicalMechanicsSystem(
        particles=[PointParticle('sphere')],
        external_interactions=[ExternalForce('elastic', 'sphere', [-k * x__sphere, 0, 0])],
        space=EuclideanSpace('euclidean', n_dim=3),
    )
    system.simulate()
    solutions = system.dynamic_equations_['solutions']
    assert solutions['sphere'][0] == symbolic_expression(
        '_K2 * cos(sqrt(k) * t / sqrt(m__sphere)) + _K1 * sin(sqrt(k) * t / sqrt(m__sphere))',
    )
    assert solutions['sphere'][1] == symbolic_expression('_K2 * t + _K1')
    assert solutions['sphere'][2] == symbolic_expression('_K2 * t + _K1')
