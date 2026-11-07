"""Test the `_patterns` module of `skcomplex.systems.physics.pattern_formation`."""

import numpy as np

from skcomplex.core.components.elements import FieldElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import ETDRK4Solver
from skcomplex.systems.physics.pattern_formation import ComplexGinzburgLandauSystem
from skcomplex.systems.physics.pattern_formation._ordering import PhaseFieldElement
from skcomplex.systems.physics.pattern_formation._patterns import SwiftHohenbergEnergyInteraction


def _amplitude(r):
    """Return the amplitude of a small mode of wavenumber 1 after a run of the Swift-Hohenberg equation."""
    sites = 64
    values = {'phi': 0.01 * np.cos(np.arange(sites) * 2 * np.pi / sites * 8)}
    space = LatticeSpace((sites,), spacing=16 * np.pi / sites)
    field = FieldElement('cells', element=PhaseFieldElement, space=space, values=values, discretization='spectral')
    energy = SwiftHohenbergEnergyInteraction('energy', target='cells', state='phi', r=r)
    simulation = Simulation(System(elements=[field], interactions=[energy]), dt=1.0, solver=ETDRK4Solver(dt=0.1))
    return np.abs(simulation.prepare().run(until=60.0).trajectory_.values('cells.phi')[-1]).max()


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
        values = [np.asarray(system.elements[0].values[part]) for system in (first, second, other)]
        np.testing.assert_array_equal(values[0], values[1])
        assert not np.array_equal(values[0], values[2])
