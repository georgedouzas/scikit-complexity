"""Test the alignment of agents, fields, and series: mean vector, mean rotation, axial order, and mean phase."""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.spatial import cKDTree

from skcomplex.core.analysis import AlignmentResult, alignment
from skcomplex.core.errors import AnalysisError
from skcomplex.systems import create_simulation


def _alignment(kind: str, source: object, **settings: object) -> AlignmentResult:
    """Return the alignment analysis of a source for one kind, run."""
    return alignment(source, kind=kind, **settings)


def _order(kind: str, source: object, **settings: object) -> np.ndarray:
    """Return the order of every frame of a source for one kind."""
    return _alignment(kind, source, **settings).order


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


def test_mean_vector_vicsek_ordering():
    """Test that 1000 Vicsek agents order at noise 0.1 and stay disordered at noise 5."""
    rng = np.random.default_rng(0)
    assert _order('mean-vector', _vicsek(rng, 1000, 0.1, 400))[0] > 0.9
    assert _order('mean-vector', _vicsek(rng, 1000, 5.0, 100))[0] < 3 / math.sqrt(1000)


def test_mean_phase_same_on_oscillators_and_headings():
    """Test that Kuramoto phases and Vicsek headings give the same order as unit vectors at those angles."""
    rng = np.random.default_rng(1)
    for angles in (_kuramoto_phases(rng, 2000, 4.0), _vicsek(rng, 1000, 0.1, 400)):
        phases = _alignment('mean-phase', angles)
        order = phases.order[0]
        assert order > 0.6
        assert order * np.exp(1j * phases.phase[0]) == pytest.approx(np.mean(np.exp(1j * angles)), abs=1e-12)
        vectors = np.column_stack([np.cos(angles), np.sin(angles)])
        assert order == pytest.approx(_order('mean-vector', vectors, vectors=True)[0], abs=1e-12)
        assert _order('axial', angles)[0] >= 0


def test_mean_rotation_of_a_mill():
    """Test that a D'Orsogna swarm in its mill state has mean rotation above 0.8 and mean vector below 0.2."""
    positions, velocities = _dorsogna_mill(np.random.default_rng(2), 100)
    assert _order('mean-rotation', np.hstack([positions, velocities]))[0] > 0.8
    assert _order('mean-vector', velocities, vectors=True)[0] < 0.2


def test_alignment_of_every_frame_of_a_trajectory():
    """Test that a flock gives one mean-vector, mean-rotation, and axial order per frame between 0 and 1."""
    trajectory = create_simulation('boids').prepare().run(until=3).trajectory_
    for kind in ('mean-vector', 'mean-rotation', 'axial'):
        order = _order(kind, trajectory)
        assert order.shape == (len(trajectory.times),)
        assert np.all((order >= 0) & (order <= 1 + 1e-12))


def test_alignment_checks_its_settings():
    """Test that an unknown kind and a setting the kind ignores raise errors, and that the result is frozen."""
    with pytest.raises(AnalysisError, match='mean-vector, mean-rotation, axial, mean-phase'):
        alignment(np.zeros(10), kind='swirl')
    with pytest.raises(AnalysisError, match='does not read vectors'):
        alignment(np.zeros(10), kind='mean-rotation', vectors=True)
    with pytest.raises(FrozenInstanceError):
        alignment([0.0, 0.0]).order = np.zeros(1)


def test_alignment_on_three_domains():
    """Test the mean vector of a Vicsek flock and of an XY lattice, and the mean phase of Kuramoto oscillators."""
    flock = create_simulation('vicsek').prepare().run(until=3).trajectory_
    spins = create_simulation('xy').prepare().run(until=3).trajectory_
    oscillators = create_simulation('kuramoto').prepare().run(until=3).trajectory_
    runs = (
        (_alignment('mean-vector', flock), flock),
        (_alignment('mean-vector', spins), spins),
        (_alignment('mean-phase', oscillators, variables=['oscillators.theta']), oscillators),
    )
    for analysis, trajectory in runs:
        assert analysis.order.shape == (len(trajectory.times),)
        assert np.all((analysis.order >= 0) & (analysis.order <= 1 + 1e-12))
    assert runs[0][0].order == pytest.approx(flock.values('polarization'), abs=1e-9)
    angles = spins.lattice('spins.theta')
    assert runs[1][0].order == pytest.approx(np.abs(np.exp(1j * angles).mean(axis=(1, 2))), abs=1e-12)
    phases = oscillators.values('oscillators.theta')
    assert runs[2][0].order == pytest.approx(np.abs(np.exp(1j * phases).mean(axis=1)), abs=1e-12)


def test_alignment_of_a_field_of_vectors():
    """Test that frames of a field of three-dimensional vectors give the length of their mean unit vector."""
    field = np.zeros((2, 4, 4, 3))
    field[0, ..., 2] = 2.0
    field[1, ..., 0] = np.where(np.arange(4) % 2, 1.0, -1.0)
    assert _order('mean-vector', field, vectors=True).tolist() == [1.0, 0.0]


def test_alignment_of_velocities_along_a_line():
    """Test that agents moving along a line align by the sign of their velocities, per frame and per member."""
    trajectory = create_simulation('random_walk_with_memory', n_walkers=200).prepare().run(until=5.0).trajectory_
    signs = np.sign(trajectory.values('walkers.v'))
    expected = np.abs(np.where(signs == 0, 0.0, signs).mean(axis=1))
    assert _alignment('mean-vector', trajectory).order == pytest.approx(expected, abs=1e-12)
    assert (
        _alignment('mean-phase', create_simulation('optimal_velocity').prepare().run(until=1.0).trajectory_).order[-1]
        == 1.0
    )


@pytest.mark.parametrize(('system', 'state'), [('swarmalators', 'units.theta'), ('kuramoto', 'oscillators.theta')])
def test_alignment_reads_phases(system, state):
    """Test that particles and series with the role phase give the synchrony of their phases, not their observables."""
    trajectory = create_simulation(system).prepare().run(until=1.0).trajectory_
    expected = np.abs(np.exp(1j * trajectory.values(state)).mean(axis=1))
    assert _alignment('mean-vector', trajectory).order == pytest.approx(expected, abs=1e-12)


def test_alignment_rejects_a_source_without_a_readable_kind():
    """Test that a trajectory with no agents, field, or series names the kinds read and the kinds held."""
    trajectory = create_simulation('poisson_process', n_processes=1).prepare().run(until=3.0).trajectory_
    with pytest.raises(AnalysisError, match='reads agents, field, series, but the trajectory holds path'):
        _alignment('mean-vector', trajectory)


def test_principal_axis_of_a_uniform_tensor_field():
    """A uniform tensor at an angle gives the axis at that angle and its scalar order at every site."""
    angle, order = 0.3, 0.8
    components = order / 2 * np.array([np.cos(2 * angle), np.sin(2 * angle)])
    field = np.tile(components, (1, 3, 4, 1))
    analysis = _alignment('principal-axis', field)
    assert analysis.order.shape == (1, 3, 4)
    assert np.allclose(analysis.order, order)
    assert np.allclose(np.abs(analysis.axes @ np.array([np.cos(angle), np.sin(angle)])), 1.0)


def test_principal_axis_of_a_three_dimensional_tensor():
    """A uniaxial tensor along an axis in 3 dimensions has the order and the axis of its construction."""
    axis = np.array([0.0, 0.0, 1.0])
    tensor = 0.6 * (np.outer(axis, axis) - np.eye(3) / 3)
    components = [tensor[0, 0], tensor[0, 1], tensor[0, 2], tensor[1, 1], tensor[1, 2]]
    analysis = _alignment('principal-axis', [components])
    assert np.allclose(analysis.order, 0.6)
    assert np.allclose(np.abs(analysis.axes[0, 0]), axis)


def test_principal_axis_rejects_a_number_of_components_of_no_symmetric_tensor():
    """A field whose number of components belongs to no symmetric tensor raises an error."""
    with pytest.raises(AnalysisError, match='independent components'):
        _alignment('principal-axis', np.zeros((3, 4)))
