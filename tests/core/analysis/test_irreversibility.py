"""Test the irreversibility analysis of entropy production, detailed balance, fluctuation theorems, and responses."""

import math
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.analysis import correlation, irreversibility
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import StateSpec, VisualSpec

RING = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
RUNS = 20_000
STEP = 0.1


def _irreversibility(kind: str, source: object, **settings: object) -> irreversibility:
    """Return the irreversibility analysis of a source for one kind, run."""
    return irreversibility(source, kind=kind, **settings)


def _chain(matrix: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Return a path of a discrete-time chain with a transition matrix, started in state 0."""
    cumulative = np.cumsum(matrix, axis=1)
    draws = rng.random(steps)
    states = np.zeros(steps, dtype=int)
    for t in range(1, steps):
        states[t] = int(np.searchsorted(cumulative[states[t - 1]], draws[t]))
    return states


def _ornstein_uhlenbeck(rng: np.random.Generator, frames: int, field: float, switch: int) -> np.ndarray:
    """Return an equilibrium ensemble of `dx = (h - x) dt + sqrt(2) dW` updated exactly, the field on from a frame."""
    decay = np.exp(-STEP)
    spread = np.sqrt(1 - decay**2)
    states = rng.standard_normal(RUNS)
    frames_list = [states.copy()]
    noise_rng = np.random.default_rng(99)
    for frame in range(1, frames):
        forcing = field if frame > switch else 0.0
        states = states * decay + forcing * (1 - decay) + spread * noise_rng.standard_normal(RUNS)
        frames_list.append(states.copy())
    return np.asarray(frames_list).T


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
    paths = _driven_particles(rng, 500, 20.0, 0.01)
    particle = _irreversibility('langevin-entropy-production', paths, drive=1.0, intensity=1.0, dt=0.01)
    assert particle.entropy_production == pytest.approx(1, rel=0.05)
    states, duration = _ring_jumps(rng, 2.0, 1.0, 300_000)
    jumps = _irreversibility('jump-entropy-production', states, rates=RING, duration=duration)
    assert jumps.entropy_production == pytest.approx(math.log(2), rel=0.05)
    assert _irreversibility('entropy-production', RING).entropy_production == pytest.approx(math.log(2), rel=1e-12)


def test_count_entropy_production_from_counts_alone():
    """Test that transition counts of the driven ring give ln 2 per unit time and a reversible chain almost 0."""
    rng = np.random.default_rng(1)
    states, duration = _ring_jumps(rng, 2.0, 1.0, 300_000)
    driven = _irreversibility('count-entropy-production', states, dt=duration / (states.size - 1))
    assert driven.entropy_production == pytest.approx(math.log(2), rel=0.05)
    balanced, balanced_duration = _ring_jumps(rng, 1.0, 1.0, 300_000)
    reversible = _irreversibility('count-entropy-production', balanced, dt=balanced_duration / (balanced.size - 1))
    assert reversible.entropy_production < 0.01
    assert _irreversibility('count-entropy-production', [0, 1, 0, 1, 0]).entropy_production == 0.0


def test_fluctuation_theorem_of_a_driven_particle():
    """Test that 10,000 runs of the driven particle over time 1 give a slope of ln P(s) / P(-s) near 1."""
    paths = _driven_particles(np.random.default_rng(2), 10_000, 1.0, 0.01)
    entropies = _irreversibility('langevin-entropy-production', paths, drive=np.ones_like, dt=0.01).entropies
    assert _irreversibility('fluctuation-theorem', entropies, bins=30).slope == pytest.approx(1, abs=0.1)


def test_intensity_scales_the_entropy_and_the_response():
    """Test that the intensity multiplies the entropy of Langevin paths and divides the response in the ratio."""
    paths = [[0.0, 1.0, 3.0]]
    entropies = _irreversibility('langevin-entropy-production', paths, drive=2.0, intensity=0.5).entropies
    assert entropies.tolist() == [3.0]
    ratios = [
        _irreversibility(
            'fluctuation-dissipation',
            [[0.0, 1.0, 2.0]],
            unperturbed=[[0.0, 0.0, 0.0]],
            correlation=[2.0, 1.0, 0.0],
            intensity=intensity,
        ).fluctuation_dissipation_ratio
        for intensity in (1.0, 2.0)
    ]
    assert ratios == pytest.approx([1.0, 0.5])


def test_entropy_production_rejects_one_way_jumps():
    """Test that a jump without its reverse raises an error, and too few samples cannot check the theorem."""
    with pytest.raises(AnalysisError, match='no reverse jump'):
        _irreversibility('entropy-production', [[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(AnalysisError, match='Fewer than 3 bins'):
        _irreversibility('fluctuation-theorem', [1.0, 2.0])
    with pytest.raises(AnalysisError, match='needs rates'):
        irreversibility([0, 1, 0, 1], kind='jump-entropy-production')
    with pytest.raises(AnalysisError, match='does not read bins'):
        irreversibility([0, 1, 0, 1], kind='entropy-production', bins=10)
    with pytest.raises(AnalysisError, match='duration'):
        _irreversibility('jump-entropy-production', [0, 1, 0], rates=[[0, 2], [1, 0]])


def _walker_currents(forward, backward, duration, paths, seed):
    """Return the net steps of a biased random walk on a ring over a duration, one per path."""

    class WalkerElement(Element):
        """Walker element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            ahead: Annotated[object, StateSpec()] = 0.0,
            behind: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.ahead = ahead
            self.behind = behind

        def _get_laws(self, s):
            """Return the laws of the walker element."""
            return [JumpLaw(forward, {s.ahead: 1}), JumpLaw(backward, {s.behind: 1})]

    system = System(elements=[WalkerElement('walker')])
    currents = []
    for index in range(paths):
        trajectory = (
            Simulation(system, dt=duration, random_state=seed + index).prepare().run(until=duration).trajectory_
        )
        currents.append(trajectory.values('walker.ahead')[-1] - trajectory.values('walker.behind')[-1])
    return np.asarray(currents)


def test_uncertainty_relation_of_a_biased_random_walk():
    """Test that the current of a biased walk on a ring has a precision product of at least 2 within its error."""
    forward, backward, duration = 2.0, 1.0, 10.0
    currents = _walker_currents(forward, backward, duration, paths=600, seed=0)
    entropy = (forward - backward) * np.log(forward / backward) * duration
    relation = _irreversibility('uncertainty-relation', currents, entropy=entropy, random_state=0)
    exact = (forward + backward) / (forward - backward) * np.log(forward / backward)
    assert relation.holds
    assert relation.product == pytest.approx(exact, abs=3 * relation.error)
    assert relation.product + 2 * relation.error >= 2


def test_uncertainty_relation_rejects_a_current_that_does_not_flow():
    """Test that a zero mean current and a single path raise errors."""
    with pytest.raises(AnalysisError, match='zero'):
        _irreversibility('uncertainty-relation', [-1.0, 1.0], entropy=1.0)
    with pytest.raises(AnalysisError, match='at least 2 paths'):
        _irreversibility('uncertainty-relation', [1.0], entropy=1.0)


def test_detailed_balance_of_reversible_and_driven_chains():
    """Test that a birth-death chain passes detailed balance and a driven cycle fails with a nonzero current."""
    rng = np.random.default_rng(1)
    size = 5
    birth_death = np.zeros((size, size))
    for state in range(size):
        if state < size - 1:
            birth_death[state, state + 1] = 0.3
        if state > 0:
            birth_death[state, state - 1] = 0.4
        birth_death[state, state] = 1 - birth_death[state].sum()
    cycle = np.asarray([[0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.6, 0.2, 0.2]])
    reversible = _irreversibility('detailed-balance', _chain(birth_death, 50_000, rng))
    driven = _irreversibility('detailed-balance', _chain(cycle, 50_000, rng))
    assert reversible.balanced
    assert not driven.balanced
    assert driven.currents[0, 1] == pytest.approx(0.4 / 3, abs=0.01)


def test_fluctuation_dissipation_ratio_in_equilibrium():
    """Test that the response of an equilibrium Ornstein-Uhlenbeck ensemble gives a ratio of 1."""
    rng = np.random.default_rng(0)
    lags = np.arange(31)
    unperturbed = _ornstein_uhlenbeck(rng, 1031, 0.0, 1000)
    curves = correlation(unperturbed, kind='two-time', waiting=[100, 1000], lags=lags).correlation
    perturbed = _ornstein_uhlenbeck(np.random.default_rng(0), 1031, 0.1, 1000)
    ratio = _irreversibility(
        'fluctuation-dissipation',
        perturbed[:, 1000:],
        unperturbed=unperturbed[:, 1000:],
        field=0.1,
        correlation=curves[1],
        intensity=1.0,
    )
    assert ratio.fluctuation_dissipation_ratio == pytest.approx(1, abs=0.1)
    response = _irreversibility('response', [[1.0, 2.0]], unperturbed=[[1.0, 1.0]], field=0.5).response
    assert response.tolist() == [0.0, 2.0]
