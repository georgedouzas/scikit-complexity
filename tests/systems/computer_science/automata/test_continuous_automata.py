"""Test the `_continuous_automata` module of `skcomplex.systems.computer_science.automata`."""

import numpy as np
import pytest

from skcomplex.core.errors import ParameterError
from skcomplex.systems import create_simulation
from skcomplex.systems.computer_science.automata import lenia


def _bump(q):
    """Return the exponential bump of Lenia's kernels, zero outside the open unit interval."""
    inside = (q > 0) & (q < 1)
    return np.where(inside, np.exp(4 - 1 / np.where(inside, q * (1 - q), 1)), 0)


def _growth(world, radius, mu, sigma):
    """Return the growth map of the convolution of a world with a single-ring kernel, by fast Fourier transforms."""
    rows, columns = world.shape
    y, x = np.meshgrid(np.fft.fftfreq(rows) * rows, np.fft.fftfreq(columns) * columns, indexing='ij')
    kernel = _bump(np.hypot(x, y) / radius)
    kernel /= kernel.sum()
    potential = np.real(np.fft.ifft2(np.fft.fft2(kernel) * np.fft.fft2(world)))
    return 2 * np.exp(-((potential - mu) ** 2) / (2 * sigma**2)) - 1


def _centre(world):
    """Return the mass and the centre of mass of a world."""
    mass = world.sum()
    rows, columns = np.indices(world.shape)
    return mass, np.array([(rows * world).sum(), (columns * world).sum()]) / mass


def test_lenia_update_is_equation_1():
    """Test that one step adds the growth map of the ring-kernel convolution over `T` and clips to `[0, 1]`."""
    trajectory = create_simulation('lenia').prepare().run(until=1).trajectory_
    world = trajectory.lattice('world.A')
    expected = np.clip(world[0] + _growth(world[0], 13.0, 0.15, 0.015) / 10, 0, 1)
    np.testing.assert_allclose(world[1], expected, atol=1e-9)


def test_lenia_orbium_glides_with_a_steady_mass():
    """Test that Orbium keeps its mass near 71 while its centre of mass moves across the torus."""
    trajectory = create_simulation('lenia').prepare().run(until=60).trajectory_
    world = trajectory.lattice('world.A')
    masses, centres = zip(*(_centre(frame) for frame in world[10:]), strict=True)
    assert np.ptp(masses) < 5
    assert np.mean(masses) == pytest.approx(71, abs=2)
    assert np.linalg.norm(centres[-1] - centres[0]) > 5


def test_lenia_expanded_universe_with_several_channels_and_kernels():
    """Test that the kernels of two channels follow equation 2: weighted growths of the source channels, clipped."""
    rng = np.random.default_rng(0)
    initial = np.zeros((2, 32, 32))
    initial[:, 10:22, 10:22] = rng.random((2, 12, 12))
    kernels = [
        (0, 0, 1.0, 1.0, (1.0,), 0.15, 0.03),
        (1, 0, 0.5, 3.0, (1.0,), 0.2, 0.05),
        (0, 1, 1.0, 1.0, (1.0,), 0.1, 0.02),
    ]
    trajectory = (
        create_simulation('lenia', shape=(32, 32), initial=initial, R=6.0, channels=2, kernels=kernels)
        .prepare()
        .run(until=1)
    )
    first, second = trajectory.trajectory_.lattice('world.A'), trajectory.trajectory_.lattice('world.B')
    growth_a = (_growth(initial[0], 6.0, 0.15, 0.03) + 3 * _growth(initial[1], 3.0, 0.2, 0.05)) / 4
    np.testing.assert_allclose(first[1], np.clip(initial[0] + growth_a / 10, 0, 1), atol=1e-9)
    growth_b = _growth(initial[0], 6.0, 0.1, 0.02)
    np.testing.assert_allclose(second[1], np.clip(initial[1] + growth_b / 10, 0, 1), atol=1e-9)


def test_lenia_runs_in_three_dimensions():
    """Test that a world of three axes takes a kernel of three-dimensional displacements and stays in `[0, 1]`."""
    rng = np.random.default_rng(1)
    initial = np.zeros((10, 10, 10))
    initial[3:7, 3:7, 3:7] = rng.random((4, 4, 4))
    trajectory = create_simulation('lenia', shape=(10, 10, 10), initial=initial, R=3.0).prepare().run(until=2)
    world = trajectory.trajectory_.lattice('world.A')
    assert world.shape == (3, 10, 10, 10)
    assert world.min() >= 0
    assert world.max() <= 1


def test_lenia_rejects_a_kernel_of_a_missing_channel():
    """Test that a kernel that reads a channel the world does not have raises a parameter error."""
    with pytest.raises(ParameterError, match='channels'):
        lenia(kernels=[(1, 0, 1.0, 1.0, (1.0,), 0.15, 0.015)])
