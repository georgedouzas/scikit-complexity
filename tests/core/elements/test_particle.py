"""Test the `_particle` module of `skcomplex.core.elements`."""

from types import SimpleNamespace

from sage.all__sagemath_symbolics import var, vector

from skcomplex.core.elements import ParticleElement
from skcomplex.specs import StateSpec, VisualSpec


def test_particle_declares_its_position_and_velocity():
    """Test that a particle declares a vector position on which boundaries act and its velocity."""
    specs = ParticleElement._get_specs()
    assert specs['x'] == StateSpec(vector=True, position=True)
    assert specs['v'] == StateSpec(vector=True, velocity_of='x')
    assert 0.0 not in specs['m'].domain


def test_particle_moves_with_its_velocity_and_is_drawn_as_points():
    """Test that a particle runs into the kinematic law of its position and is drawn at its positions."""
    x, v = vector([var('x0'), var('x1')]), vector([var('v0'), var('v1')])
    particle = ParticleElement('ball', x=[0, 1], m=2.0).check().prepare()
    (law,) = particle.run(SimpleNamespace(x=x, v=v))
    assert (law.target, law.value) == (x, v)
    assert particle.default_visual_ == VisualSpec('points', {'position': 'x'})
