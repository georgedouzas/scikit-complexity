"""Test the `_substitutions` module of `skcomplex.models.biology.evolution`."""

import itertools

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.biology.evolution import origin_fixation_walk


def test_origin_fixation_walk_samples_fitness_to_the_power_n_minus_one():
    """Test that the time spent on each genome of four loci matches the stationary law Q ~ f^(N - 1)."""
    loci, size = 4, 10.0
    system = origin_fixation_walk(loci, N=size, mu=1.0, field=0.1, coupling=0.3, random_state=1)
    trajectory = (
        Simulation(system, dt=1.0, random_state=0, observables=describe('origin_fixation_walk').observables_of(system))
        .run(until=5000)
        .trajectory_
    )
    alleles = trajectory['genome.s'][100:]
    visited = np.bincount(((alleles > 0) * 2 ** np.arange(loci)).sum(axis=1).astype(int), minlength=2**loci)
    fields = system.elements[0].values['h']
    couplings = system.interactions[0].couplings
    genomes = np.array([[2 * bit - 1 for bit in reversed(bits)] for bits in itertools.product((0, 1), repeat=loci)])
    fitness = 1.0 + genomes @ fields + np.einsum('gi,ij,gj->g', genomes, couplings, genomes) / 2
    assert visited / visited.sum() == pytest.approx(fitness ** (size - 1) / np.sum(fitness ** (size - 1)), abs=0.02)


def test_origin_fixation_walk_climbs_the_landscape():
    """Test that a large population ends near the single peak of a landscape of random fields."""
    system = origin_fixation_walk(16, N=1000.0, mu=0.01, random_state=0)
    trajectory = (
        Simulation(system, dt=10.0, random_state=0, observables=describe('origin_fixation_walk').observables_of(system))
        .run(until=1000)
        .trajectory_
    )
    fields = system.elements[0].values['h']
    assert trajectory['fitness'][-1] == pytest.approx(1.0 + np.abs(fields).sum(), abs=0.02)
    assert trajectory['fitness'][-1] > trajectory['fitness'][0]
