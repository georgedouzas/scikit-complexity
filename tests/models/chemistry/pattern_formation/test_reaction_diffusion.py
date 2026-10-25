"""Test the `_reaction_diffusion` module of `skcomplex.models.chemistry.pattern_formation`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Frame, Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.engine.viz import RerunObserver, resolve_visual
from skcomplex.models.chemistry.pattern_formation import brusselator, gray_scott, schnakenberg
from skcomplex.models.physics.heat import heat_diffusion


class Images:
    """Recording stand-in that keeps the images logged to it."""

    def __init__(self):
        """Start with no images."""
        self.images = []

    def log(self, path, archetype):
        """Keep a logged image with its path."""
        self.images.append((path, type(archetype).__name__))


def _turing_wavelength(a, b, du, dv):
    """Return the wavelength of the fastest growing mode of the Schnakenberg model."""
    u, v = a + b, b / (a + b) ** 2
    jacobian = np.array([[-1 + 2 * u * v, u**2], [-2 * u * v, -(u**2)]])
    wavenumbers = np.linspace(0.01, 2, 4000)
    growth = [np.linalg.eigvals(jacobian - np.diag([du, dv]) * q**2).real.max() for q in wavenumbers]
    return 2 * np.pi / wavenumbers[np.argmax(growth)]


def test_turing_spots_have_predicted_wavelength():
    """Test that the Schnakenberg model forms a stationary pattern with the wavelength of the linear prediction."""
    simulation = Simulation(schnakenberg((256,)), dt=50, solver=RK4Solver(dt=0.01)).run(until=600)
    values = simulation.trajectory_.values('cells.u')
    wavelengths = [256 / np.argmax(np.abs(np.fft.rfft(row - row.mean()))) for row in values[-5:]]
    changes = np.abs(np.diff(values[-5:], axis=0)).max(axis=1)
    assert len(set(wavelengths)) == 1
    assert wavelengths[0] == pytest.approx(_turing_wavelength(0.1, 0.9, 1.0, 40.0), rel=0.1)
    assert np.all(np.diff(changes) < 0)
    assert changes[-1] < 0.05 * np.ptp(values[-1])


def test_brusselator_and_heat_stay_bounded():
    """Test that the Brusselator stays bounded and insulated heat keeps its total."""
    values = Simulation(brusselator((32, 32)), dt=1, solver=RK4Solver(dt=0.01)).run(until=20).state_
    assert np.all(np.isfinite(values))
    assert values.max() < 20
    trajectory = Simulation(heat_diffusion((32, 32)), dt=5).run(until=50).trajectory_
    assert trajectory.values('plate.u').sum(axis=1) == pytest.approx(
        np.full(len(trajectory.times), trajectory.values('plate.u')[0].sum()),
    )


def test_gray_scott_streams_images(tmp_path):
    """Test that both Gray-Scott fields are logged as images every frame."""
    simulation = Simulation(gray_scott((32, 32)), dt=20, solver=RK4Solver(dt=1.0)).run(until=40)
    element = simulation.problem_.elements[0]
    recording = Images()
    for step in range(len(simulation.trajectory_.times)):
        frame = Frame(step, 0.0, simulation.trajectory_.states[step], simulation.parameters_, simulation.problem_)
        resolve_visual(element.visual).prepare(element, frame.problem).run(frame, recording)
    assert recording.images == [('cells/u', 'Image'), ('cells/v', 'Image')] * 3
    observer = RerunObserver('gray-scott', save_path=tmp_path / 'gray-scott.rrd')
    Simulation(gray_scott((32, 32)), dt=20, solver=RK4Solver(dt=1.0), observers=[observer]).run(until=40)
    assert observer.logged_frames_ == 3
