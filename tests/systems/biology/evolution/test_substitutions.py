"""Test the `_substitutions` module of `skcomplex.systems.biology.evolution`."""

import itertools

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.evolution import OriginFixationWalkSystem


def test_origin_fixation_walk_samples_fitness_to_the_power_n_minus_one():
    """Test that the time spent on each genome of four loci matches the stationary law Q ~ f^(N - 1)."""
    loci, size = 4, 10.0
    system = OriginFixationWalkSystem(loci, N=size, mu=1.0, field=0.1, coupling=0.3, random_state=1)
    trajectory = simulate(system, until=5000, dt=1.0, random_state=0)
    alleles = trajectory.values('genome.s')[100:]
    visited = np.bincount(((alleles > 0) * 2 ** np.arange(loci)).sum(axis=1).astype(int), minlength=2**loci)
    fields = system.system_.elements[0].values['h']
    couplings = system.system_.interactions[0].couplings
    genomes = np.array([[2 * bit - 1 for bit in reversed(bits)] for bits in itertools.product((0, 1), repeat=loci)])
    fitness = 1.0 + genomes @ fields + np.einsum('gi,ij,gj->g', genomes, couplings, genomes) / 2
    assert visited / visited.sum() == pytest.approx(fitness ** (size - 1) / np.sum(fitness ** (size - 1)), abs=0.02)


def test_origin_fixation_walk_climbs_the_landscape():
    """Test that a large population ends near the single peak of a landscape of random fields."""
    system = OriginFixationWalkSystem(16, N=1000.0, mu=0.01, random_state=0)
    trajectory = simulate(system, until=1000, dt=10.0, random_state=0)
    fields = system.system_.elements[0].values['h']
    assert trajectory.values('fitness')[-1] == pytest.approx(1.0 + np.abs(fields).sum(), abs=0.02)
    assert trajectory.values('fitness')[-1] > trajectory.values('fitness')[0]
