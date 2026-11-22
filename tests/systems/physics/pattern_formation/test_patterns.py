"""Test the `_patterns` module of `skcomplex.systems.physics.pattern_formation`."""

import numpy as np

from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.solvers import ETDRK4Solver
from skcomplex.core.systems import System
from skcomplex.systems.physics.pattern_formation import (
    ComplexGinzburgLandauSystem,
    GinzburgLandauElement,
    GinzburgLandauInteraction,
)
from skcomplex.systems.physics.pattern_formation._ordering import PhaseFieldElement
from skcomplex.systems.physics.pattern_formation._patterns import SwiftHohenbergEnergyInteraction


def _amplitude(r):
    """Return the amplitude of a small mode of wavenumber 1 after a run of the Swift-Hohenberg equation."""
    sites = 64
    phi = 0.01 * np.cos(np.arange(sites) * 2 * np.pi / sites * 8)
    space = LatticeSpace((sites,), spacing=16 * np.pi / sites, discretization='spectral')
    field = PhaseFieldElement('cells', phi=phi, space=space)
    energy = SwiftHohenbergEnergyInteraction('energy', target='cells', state='phi', r=r)
    system = System(elements=[field], interactions=[energy], solver=ETDRK4Solver(dt=0.1))
    trajectory = system.prepare().simulate(until=60.0, dt=1.0)
    return np.abs(trajectory.get_values('cells.phi')[-1]).max()


def test_rolls_grow_above_onset_and_decay_below():
    """Test that the critical mode grows to the roll amplitude `sqrt(4 r / 3)` above onset and decays below it."""
    assert abs(_amplitude(0.3) - np.sqrt(4 * 0.3 / 3)) < 0.02
    assert _amplitude(-0.3) < 1e-4


def test_complex_ginzburg_landau_seeds_both_parts():
    """Test that the random state fixes both the real and the imaginary part of the initial amplitude."""
    first, second, other = (
        ComplexGinzburgLandauSystem((8, 8), random_state=seed).prepare().system_ for seed in (1, 1, 2)
    )
    for part in ('u', 'v'):
        values = [np.asarray(getattr(system.elements[0], part)) for system in (first, second, other)]
        np.testing.assert_array_equal(values[0], values[1])
        assert not np.array_equal(values[0], values[2])


def test_uniform_amplitude_turns_on_the_limit_cycle():
    """Test that a uniform amplitude of modulus 1 keeps its modulus and turns at the frequency `-c`."""
    space = LatticeSpace((4, 4), discretization='spectral')
    medium = GinzburgLandauElement('medium', u=np.ones(16), v=np.zeros(16), space=space)
    dynamics = GinzburgLandauInteraction('dynamics', target='medium', c=-1.5)
    system = System(elements=[medium], interactions=[dynamics], solver=ETDRK4Solver(dt=0.01))
    trajectory = system.prepare().simulate(until=1.0, dt=1.0)
    u, v = trajectory.get_values('medium.u')[-1], trajectory.get_values('medium.v')[-1]
    np.testing.assert_allclose(u**2 + v**2, 1.0, atol=1e-8)
    np.testing.assert_allclose(np.arctan2(v, u), 1.5, atol=1e-6)
