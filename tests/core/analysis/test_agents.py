"""Test order parameters, kinetic temperatures, neighbour statistics, coarse-graining, and agent displacement."""

import math
from typing import ClassVar

import numpy as np
import pytest
from scipy.spatial import cKDTree

from skcomplex.core.analysis import (
    clark_evans_ratio,
    coarse_grain,
    kinetic_temperature,
    maxwell_boltzmann_test,
    mean_squared_displacement,
    milling_order,
    nearest_neighbor_distances,
    nematic_order,
    phase_order,
    polar_order,
    radial_distribution,
    snapshots_of,
    structure_factor,
)
from skcomplex.core.base import AnalysisError, Domain, VisualSpec, state
from skcomplex.core.model import AgentType, ContinuousSpace, GridSpace, Law, Rule, Symbols, System
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import AgentView, ModelView
from skcomplex.models import run


class GridWalker(AgentType):
    """Walker that moves to a random neighbouring cell of a periodic grid at every step."""

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {})
    steps: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the stepping rule."""
        return [Rule(self.step)]

    def step(self, agent: AgentView, model: ModelView) -> None:
        """Move to a random neighbouring cell."""
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(model.rng.integers(len(cells)))])
        agent.steps += 1


class Diffuser(AgentType):
    """Particle that takes a Gaussian step of unit variance per coordinate in a periodic square at every step."""

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {})
    steps: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the stepping rule."""
        return [Rule(self.step)]

    def step(self, agent: AgentView, model: ModelView) -> None:
        """Take a Gaussian step and wrap into the square."""
        position = np.asarray(agent.location, dtype=float) + model.rng.standard_normal(2)
        agent.move_to(tuple(np.mod(position, 20.0)))
        agent.steps += 1


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


def _langevin_velocities(rng: np.random.Generator, particles: int, temperature: float) -> np.ndarray:
    """Return velocities of underdamped Langevin particles with friction 1 after relaxing from rest."""
    decay = math.exp(-0.01)
    velocities = np.zeros((particles, 2))
    for _ in range(1000):
        velocities = velocities * decay + math.sqrt(temperature * (1 - decay**2)) * rng.standard_normal(
            velocities.shape,
        )
    return velocities


def _active_brownian(rng: np.random.Generator, particles: int, speed: float, rotation: float) -> np.ndarray:
    """Return positions of two-dimensional active Brownian particles every 0.01 over 100 time units."""
    dt, steps = 0.01, 10_000
    angles = rng.uniform(-np.pi, np.pi, particles) + np.cumsum(
        math.sqrt(2 * rotation * dt) * rng.standard_normal((steps, particles)),
        axis=0,
    )
    moves = speed * dt * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return np.concatenate([np.zeros((1, particles, 2)), np.cumsum(moves, axis=0)])


def test_polar_order_vicsek_ordering():
    """Test that 1000 Vicsek agents order at noise 0.1 and stay disordered at noise 5."""
    rng = np.random.default_rng(0)
    assert polar_order(_vicsek(rng, 1000, 0.1, 400)) > 0.9
    assert polar_order(_vicsek(rng, 1000, 5.0, 100)) < 3 / math.sqrt(1000)


def test_phase_order_same_on_oscillators_and_headings():
    """Test that Kuramoto phases and Vicsek headings give the same order as unit vectors at those angles."""
    rng = np.random.default_rng(1)
    for angles in (_kuramoto_phases(rng, 2000, 4.0), _vicsek(rng, 1000, 0.1, 400)):
        order = phase_order(angles)
        assert abs(order) > 0.6
        vectors = np.column_stack([np.cos(angles), np.sin(angles)])
        assert abs(order) == pytest.approx(polar_order(vectors, vectors=True), abs=1e-12)
        assert nematic_order(angles) >= 0


def test_milling_order_of_a_mill():
    """Test that a D'Orsogna swarm in its mill state has milling order above 0.8 and polar order below 0.2."""
    positions, velocities = _dorsogna_mill(np.random.default_rng(2), 100)
    assert milling_order(positions, velocities) > 0.8
    assert polar_order(velocities, vectors=True) < 0.2


def test_maxwell_boltzmann_test_in_two_and_three_dimensions():
    """Test that ideal gases in two and three dimensions follow the law of their constant kinetic temperature."""
    for dimension in (2, 3):
        snapshots = snapshots_of(run('ideal_gas', n_dim=dimension, random_state=0, settings={'until': 1.0}).trajectory_)
        temperatures = kinetic_temperature(snapshots.velocities)
        assert np.ptp(temperatures) < 1e-10
        assert maxwell_boltzmann_test(snapshots.velocities[-1]).p_value > 0.05


def test_kinetic_temperature_of_langevin_particles():
    """Test that 2000 underdamped Langevin particles reach the bath temperature 1.5 and its speed law."""
    velocities = _langevin_velocities(np.random.default_rng(3), 2000, 1.5)
    assert kinetic_temperature(velocities) == pytest.approx(1.5, rel=0.03)
    assert maxwell_boltzmann_test(velocities).p_value > 0.05


def test_clark_evans_ratio_random_and_ordered_positions():
    """Test that random points give 1 and a triangular lattice gives above 2."""
    points = np.random.default_rng(4).random((2000, 2))
    assert clark_evans_ratio(points, [1.0, 1.0]) == pytest.approx(1, abs=0.05)
    rows, columns = np.meshgrid(np.arange(40), np.arange(46), indexing='ij')
    lattice = np.column_stack([(columns + 0.5 * (rows % 2)).ravel(), (rows * math.sqrt(3) / 2).ravel()])
    assert clark_evans_ratio(lattice, [46.0, 40 * math.sqrt(3) / 2]) > 2
    assert nearest_neighbor_distances(lattice[:3]).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_radial_distribution_of_a_gas_and_a_fluid():
    """Test that an ideal gas gives g = 1 and a dense Lennard-Jones fluid peaks at 2^(1/6) sigma."""
    rng = np.random.default_rng(5)
    gas = rng.random((20, 4000, 2)) * 20
    _, density = radial_distribution(gas, [20.0, 20.0], bins=20)
    assert np.all(np.abs(density - 1) < 0.05)
    snapshots = snapshots_of(run('lennard_jones_fluid').trajectory_)
    box = snapshots.box
    distances, fluid = radial_distribution(np.mod(snapshots.positions[-10:], box), box, bins=150, r_max=3.0)
    assert distances[fluid.argmax()] == pytest.approx(2 ** (1 / 6), rel=0.05)


def test_coarse_grain_density():
    """Test that 10,000 agents coarse-grained on a 32 by 32 grid sum to 10,000 and have a structure factor."""
    rng = np.random.default_rng(6)
    density = coarse_grain(rng.random((10_000, 2)) * 5, [5.0, 5.0], [32, 32])
    assert density.sum() == 10_000
    assert structure_factor(density).power.size > 0
    polarization = coarse_grain(rng.random((100, 2)), [1.0, 1.0], [4, 4], values=np.ones((100, 2)))
    assert polarization.shape == (4, 4, 2)
    with pytest.raises(AnalysisError, match='one entry per dimension'):
        coarse_grain(rng.random((5, 2)), [1.0, 1.0], [4])


def test_mean_squared_displacement_diffusion_on_two_kinds_of_space():
    """Test that Brownian particles in a periodic box and walkers on a periodic grid diffuse without a plateau."""
    continuous = System(components=[Diffuser.group('particles', 300)], space=ContinuousSpace(20.0, 20.0, torus=True))
    grid = System(
        components=[GridWalker.group('walkers', 300)],
        space=GridSpace(20, 20, torus=True, neighborhood='von_neumann'),
    )
    for system in (continuous, grid):
        trajectory = Simulation(system, dt=1, random_state=0).run(until=600).trajectory_
        displacement = mean_squared_displacement(snapshots_of(trajectory).positions, fit_range=(1, 150))
        assert displacement.exponent == pytest.approx(1, abs=0.05)
        assert displacement.msd[-1] > 20.0**2


def test_mean_squared_displacement_active_particles():
    """Test the ballistic and diffusive regimes of active Brownian particles and their long-time diffusion."""
    positions = _active_brownian(np.random.default_rng(7), 400, 1.0, 1.0)
    ballistic = mean_squared_displacement(positions, dt=0.01, fit_range=(0.01, 0.1))
    diffusive = mean_squared_displacement(positions, dt=0.01, fit_range=(10.0, 25.0))
    assert ballistic.exponent == pytest.approx(2, abs=0.1)
    assert diffusive.exponent == pytest.approx(1, abs=0.1)
    assert diffusive.diffusion == pytest.approx(0.5, rel=0.1)
