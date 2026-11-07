"""Test the `_processes` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np
import pytest
from scipy import ndimage

from skcomplex.core.engine.runtime import Simulation, simulate
from skcomplex.systems.physics.statistical_mechanics import (
    ContactProcessSystem,
    DiffusionLimitedAggregationSystem,
    Q2rSystem,
    SitePercolationSystem,
    TASEPSystem,
)


def _ising_energy(spins):
    """Return the nearest-neighbour Ising energy of a periodic square lattice."""
    return -(spins * np.roll(spins, 1, axis=0)).sum() - (spins * np.roll(spins, 1, axis=1)).sum()


def test_q2r_conserves_energy():
    """Test that the Q2R automaton keeps the Ising energy while the magnetisation changes."""
    trajectory = simulate(Q2rSystem((32, 32)), until=50, random_state=0)
    spins = trajectory.lattice('spins.spin')
    assert {_ising_energy(frame) for frame in spins} == {_ising_energy(spins[0])}
    assert abs(spins[-1].mean()) < abs(spins[0].mean())


@pytest.mark.parametrize(('p', 'spans'), [(0.45, False), (0.75, True)])
def test_site_percolation_burns_the_clusters_of_the_top_row(p, spans):
    """Test that the fire burns the clusters touching the top row, which span the lattice only above threshold."""
    system = SitePercolationSystem((40, 40), p=p, random_state=1).prepare().system_
    occupied = np.reshape(system.elements[0].values['cover'], (40, 40)) > 0
    trajectory = Simulation(system, random_state=0).prepare().run(until=200).trajectory_
    burnt = trajectory.lattice('lattice.cover')[-1] == 3
    labels, _ = ndimage.label(occupied)
    lit = np.isin(labels, np.setdiff1d(labels[0], [0]))
    assert np.array_equal(burnt, lit)
    assert burnt[-1].any() == spans


@pytest.mark.parametrize(('rate', 'survives'), [(0.5, False), (4.0, True)])
def test_contact_process_has_an_absorbing_transition(rate, survives):
    """Test that the infection dies out below the critical rate and persists above it."""
    trajectory = simulate(ContactProcessSystem((32, 32), rate=rate), until=300, random_state=0)
    assert (trajectory.lattice('lattice.infected')[-1].mean() > 0.3) == survives


def test_tasep_ring_current():
    """Test that the parallel exclusion process on a ring keeps its particles and carries the exact current."""
    trajectory = simulate(TASEPSystem(400, ring=True, p=0.5), until=1000, random_state=0)
    occupied = trajectory.lattice('lane.occupied')
    current = ((occupied[:-1] == 1) & (occupied[1:] == 0)).sum(axis=1).mean() / 400
    assert set(occupied.sum(axis=1)) == {200}
    assert current == pytest.approx((1 - np.sqrt(0.5)) / 2, rel=0.03)


def test_tasep_open_line_stays_an_exclusion_process():
    """Test that the open exclusion process keeps one particle at most per site."""
    trajectory = simulate(TASEPSystem(100), until=200, random_state=0)
    assert set(np.unique(trajectory.lattice('lane.occupied'))) <= {0.0, 1.0}


def test_tasep_reports_the_hops_and_carries_the_exact_current_of_an_open_line():
    """Test that the reported hops on a ring and the exits of an open line give the exact parallel-update current."""
    exact = (1 - np.sqrt(0.5)) / 2
    ring = simulate(TASEPSystem(400, ring=True, p=0.5), random_state=0)
    assert np.asarray(ring.values('lane.hops'))[50:].sum(axis=1).mean() / 400 == pytest.approx(exact, rel=0.03)
    line = simulate(TASEPSystem(100, alpha=0.6, beta=0.6), until=3000, random_state=0)
    occupied = line.lattice('lane.occupied')[300:]
    exits = ((occupied[:-1, -1] == 1) & (occupied[1:, -1] == 0)).mean()
    assert exits == pytest.approx(exact, rel=0.05)


def test_diffusion_limited_aggregate_is_connected():
    """Test that the aggregate grows and every stuck walker touches another stuck walker."""
    trajectory = simulate(DiffusionLimitedAggregationSystem(200, size=31), until=100, random_state=0)
    agents = trajectory.agents('walkers', 100)
    cells = {tuple(location) for location, stuck in zip(agents.locations, agents.values['stuck'], strict=True) if stuck}
    assert len(cells) > 20
    steps = ((0, 1), (1, 0), (0, -1), (-1, 0))
    assert all(any(((x + dx) % 31, (y + dy) % 31) in cells for dx, dy in steps) for x, y in cells)
