"""Test entropy production of jump and Langevin dynamics and the fluctuation theorem."""

import math

import numpy as np
import pytest

from skcomplex.core.analysis import (
    count_entropy_production,
    entropy_production,
    fluctuation_theorem,
    jump_entropy_production,
    langevin_entropy_production,
    uncertainty_relation,
)
from skcomplex.core.base import AnalysisError
from skcomplex.core.components import Element
from skcomplex.core.laws import Jump
from skcomplex.core.runtime import Simulation
from skcomplex.core.system import System

RING = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)


def _driven_particles(rng: np.random.Generator, runs: int, duration: float, dt: float) -> np.ndarray:
    """Return unwrapped positions of overdamped particles on a ring driven by force 1 at unit friction and heat."""
    steps = rng.normal(dt, np.sqrt(2 * dt), (runs, round(duration / dt)))
    return np.concatenate([np.zeros((runs, 1)), np.cumsum(steps, axis=1)], axis=1)


def _ring_jumps(rng: np.random.Generator, forward: float, backward: float, jumps: int) -> tuple[np.ndarray, float]:
    """Return the states after every jump of a ring of 5 states and the duration of the path."""
    steps = np.where(rng.random(jumps) < forward / (forward + backward), 1, -1)
    states = np.concatenate([[0], np.cumsum(steps)]) % 5
    duration = float(rng.exponential(1 / (forward + backward), jumps).sum())
    return states, duration


def test_entropy_production_same_on_two_domains():
    """Test that a driven Brownian particle produces 1 and a driven jump ring (2 - 1) ln 2 per unit time."""
    rng = np.random.default_rng(0)
    particle = langevin_entropy_production(_driven_particles(rng, 500, 20.0, 0.01), 1.0, 1.0, 0.01)
    assert particle.rate == pytest.approx(1, rel=0.05)
    states, duration = _ring_jumps(rng, 2.0, 1.0, 300_000)
    jumps = jump_entropy_production(states, RING, duration)
    assert jumps.rate == pytest.approx(math.log(2), rel=0.05)
    assert entropy_production(RING) == pytest.approx(math.log(2), rel=1e-12)


def test_count_entropy_production_from_counts_alone():
    """Test that transition counts of the driven ring give ln 2 per unit time and a reversible chain almost 0."""
    rng = np.random.default_rng(1)
    states, duration = _ring_jumps(rng, 2.0, 1.0, 300_000)
    assert count_entropy_production(states, dt=duration / (states.size - 1)) == pytest.approx(math.log(2), rel=0.05)
    balanced, balanced_duration = _ring_jumps(rng, 1.0, 1.0, 300_000)
    assert count_entropy_production(balanced, dt=balanced_duration / (balanced.size - 1)) < 0.01


def test_fluctuation_theorem_of_a_driven_particle():
    """Test that 10,000 runs of the driven particle over time 1 give a slope of ln P(s) / P(-s) near 1."""
    paths = _driven_particles(np.random.default_rng(2), 10_000, 1.0, 0.01)
    entropies = langevin_entropy_production(paths, np.ones_like, 1.0, 0.01).entropies
    assert fluctuation_theorem(entropies, bins=30).slope == pytest.approx(1, abs=0.1)


def test_entropy_production_rejects_one_way_jumps():
    """Test that a jump without its reverse raises an error, and too few samples cannot check the theorem."""
    with pytest.raises(AnalysisError, match='no reverse jump'):
        entropy_production([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(AnalysisError, match='Fewer than 3 bins'):
        fluctuation_theorem([1.0, 2.0])


def _walker_currents(forward, backward, duration, paths, seed):
    """Return the net steps of a biased random walk on a ring over a duration, one per path."""
    Walker = Element.define(
        'Walker',
        states={'ahead': 0.0, 'behind': 0.0},
        laws=lambda s: [Jump(forward, {s.ahead: 1}), Jump(backward, {s.behind: 1})],
    )
    system = System(components=[Walker('walker')])
    currents = []
    for index in range(paths):
        trajectory = Simulation(system, dt=duration, random_state=seed + index).run(until=duration).trajectory_
        currents.append(trajectory['walker.ahead'][-1] - trajectory['walker.behind'][-1])
    return np.asarray(currents)


def test_uncertainty_relation_of_a_biased_random_walk():
    """Test that the current of a biased walk on a ring has a precision product of at least 2 within its error."""
    forward, backward, duration = 2.0, 1.0, 10.0
    currents = _walker_currents(forward, backward, duration, paths=600, seed=0)
    entropy = (forward - backward) * np.log(forward / backward) * duration
    relation = uncertainty_relation(currents, entropy, random_state=0)
    exact = (forward + backward) / (forward - backward) * np.log(forward / backward)
    assert relation.holds
    assert relation.product == pytest.approx(exact, abs=3 * relation.error)
    assert relation.product + 2 * relation.error >= 2


def test_uncertainty_relation_rejects_a_current_that_does_not_flow():
    """Test that a zero mean current and a single path raise errors."""
    with pytest.raises(AnalysisError, match='zero'):
        uncertainty_relation([-1.0, 1.0], 1.0)
    with pytest.raises(AnalysisError, match='at least 2 paths'):
        uncertainty_relation([1.0], 1.0)
