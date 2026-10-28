"""Test the polar, milling, nematic, and phase order of agents."""

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree
from sklearn.base import clone

from skcomplex.analysis import OrderAnalysis
from skcomplex.core.errors import AnalysisError
from skcomplex.models import run


def _order(kind: str, source: object, **settings: object) -> np.ndarray:
    """Return the order of every frame of a source for one kind."""
    return OrderAnalysis(kind, **settings).check().prepare(source).run().order_


def _vicsek(rng: np.random.Generator, agents: int, noise: float, steps: int) -> np.ndarray:
    """Return the headings of Vicsek agents of speed 0.5 at density 1 after relaxation."""
    side = math.sqrt(agents)
    positions = rng.random((agents, 2)) * side
    headings = rng.uniform(-np.pi, np.pi, agents)
    for _ in range(steps):
        pairs = cKDTree(positions, boxsize=side).query_pairs(1.0, output_type='ndarray')
        sines, cosines = np.sin(headings), np.cos(headings)
        total_sines, total_cosines = sines.copy(), cosines.copy()
        for first, second in ((0, 1), (1, 0)):
            np.add.at(total_sines, pairs[:, first], sines[pairs[:, second]])
            np.add.at(total_cosines, pairs[:, first], cosines[pairs[:, second]])
        headings = np.arctan2(total_sines, total_cosines) + noise * rng.uniform(-0.5, 0.5, agents)
        positions = np.mod(positions + 0.5 * np.column_stack([np.cos(headings), np.sin(headings)]), side)
    return headings


def _kuramoto_phases(rng: np.random.Generator, oscillators: int, coupling: float) -> np.ndarray:
    """Return Kuramoto phases with Lorentzian frequencies of width 1 after relaxation."""
    frequencies = np.tan(np.pi * (np.arange(1, oscillators + 1) / (oscillators + 1) - 0.5))
    phases = rng.uniform(-np.pi, np.pi, oscillators)
    for _ in range(2000):
        order = np.mean(np.exp(1j * phases))
        phases = phases + 0.025 * (frequencies + coupling * np.abs(order) * np.sin(np.angle(order) - phases))
    return phases


def _dorsogna_mill(rng: np.random.Generator, agents: int) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and velocities of a D'Orsogna swarm started turning one way, which settles in a mill."""
    angles = rng.uniform(0, 2 * np.pi, agents)
    radii = rng.uniform(1, 4, agents)
    positions = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    velocities = 1.4 * np.column_stack([-np.sin(angles), np.cos(angles)])
    dt = 0.02
    for _ in range(5000):
        separation = positions[:, None, :] - positions[None, :, :]
        distance = np.linalg.norm(separation, axis=-1) + np.eye(agents)
        strength = (np.exp(-distance / 0.5) / 0.5 - 0.5 * np.exp(-distance / 2.0) / 2.0) / distance
        np.fill_diagonal(strength, 0.0)
        force = np.sum(strength[:, :, None] * separation, axis=1)
        speed_squared = np.sum(velocities**2, axis=1, keepdims=True)
        velocities = velocities + dt * ((1.0 - 0.5 * speed_squared) * velocities + force)
        positions = positions + dt * velocities
    return positions, velocities


def test_polar_order_vicsek_ordering():
    """Test that 1000 Vicsek agents order at noise 0.1 and stay disordered at noise 5."""
    rng = np.random.default_rng(0)
    assert _order('polar', _vicsek(rng, 1000, 0.1, 400))[0] > 0.9
    assert _order('polar', _vicsek(rng, 1000, 5.0, 100))[0] < 3 / math.sqrt(1000)


def test_phase_order_same_on_oscillators_and_headings():
    """Test that Kuramoto phases and Vicsek headings give the same order as unit vectors at those angles."""
    rng = np.random.default_rng(1)
    for angles in (_kuramoto_phases(rng, 2000, 4.0), _vicsek(rng, 1000, 0.1, 400)):
        order = _order('phase', angles)[0]
        assert abs(order) > 0.6
        vectors = np.column_stack([np.cos(angles), np.sin(angles)])
        assert abs(order) == pytest.approx(_order('polar', vectors, vectors=True)[0], abs=1e-12)
        assert _order('nematic', angles)[0] >= 0


def test_milling_order_of_a_mill():
    """Test that a D'Orsogna swarm in its mill state has milling order above 0.8 and polar order below 0.2."""
    positions, velocities = _dorsogna_mill(np.random.default_rng(2), 100)
    assert _order('milling', np.hstack([positions, velocities]))[0] > 0.8
    assert _order('polar', velocities, vectors=True)[0] < 0.2


def test_order_of_every_frame_of_a_trajectory():
    """Test that a flock gives one polar, milling, and nematic order per frame between 0 and 1."""
    trajectory = run('boids', settings={'until': 3}).trajectory_
    for kind in ('polar', 'milling', 'nematic'):
        order = _order(kind, trajectory)
        assert order.shape == (len(trajectory.times),)
        assert np.all((order >= 0) & (order <= 1 + 1e-12))


def test_order_checks_its_settings():
    """Test that an unknown kind and a setting the kind ignores raise errors, and that a clone has no results."""
    with pytest.raises(AnalysisError, match='polar, milling, nematic, phase'):
        OrderAnalysis('swirl').check()
    with pytest.raises(AnalysisError, match='does not read vectors'):
        OrderAnalysis('milling', vectors=True).check()
    analysis = OrderAnalysis().check().prepare([0.0, 0.0]).run()
    assert not hasattr(clone(analysis), 'order_')
