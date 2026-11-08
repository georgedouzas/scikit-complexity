"""Test the `_reaction_diffusion` module of `skcomplex.systems.chemistry.pattern_formation`."""

import numpy as np
import pytest

from skcomplex.core.engine.problems import FrameSpec
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.chemistry.pattern_formation import BrusselatorSystem, GrayScottSystem, SchnakenbergSystem
from skcomplex.systems.physics.heat import HeatDiffusionSystem


class Images:
    """Recording stand-in that keeps the images logged to it."""

    def __init__(self):
        self.images = []

    def log(self, path, archetype):
        """Keep a logged image with its path."""
        self.images.append((path, type(archetype).__name__))


def _turing_wavelength(a, b, du, dv):
    """Return the wavelength of the fastest growing mode of the Schnakenberg system."""
    u, v = a + b, b / (a + b) ** 2
    jacobian = np.array([[-1 + 2 * u * v, u**2], [-2 * u * v, -(u**2)]])
    wavenumbers = np.linspace(0.01, 2, 4000)
    growth = [np.linalg.eigvals(jacobian - np.diag([du, dv]) * q**2).real.max() for q in wavenumbers]
    return 2 * np.pi / wavenumbers[np.argmax(growth)]


def test_turing_spots_have_predicted_wavelength():
    """Test that the Schnakenberg system forms a stationary pattern with the wavelength of the linear prediction."""
    values = simulate(SchnakenbergSystem((256,)), until=600, dt=50, solver=RK4Solver(dt=0.01)).values('cells.u')
    wavelengths = [256 / np.argmax(np.abs(np.fft.rfft(row - row.mean()))) for row in values[-5:]]
    changes = np.abs(np.diff(values[-5:], axis=0)).max(axis=1)
    assert len(set(wavelengths)) == 1
    assert wavelengths[0] == pytest.approx(_turing_wavelength(0.1, 0.9, 1.0, 40.0), rel=0.1)
    assert np.all(np.diff(changes) < 0)
    assert changes[-1] < 0.05 * np.ptp(values[-1])


def test_brusselator_and_heat_stay_bounded():
    """Test that the Brusselator stays bounded and insulated heat keeps its total."""
    values = simulate(BrusselatorSystem((32, 32)), until=20, dt=1).states[-1]
    assert np.all(np.isfinite(values))
    assert values.max() < 20
    trajectory = simulate(HeatDiffusionSystem((32, 32)).prepare().system_, until=50, dt=5)
    assert trajectory.values('plate.u').sum(axis=1) == pytest.approx(
        np.full(len(trajectory.times), trajectory.values('plate.u')[0].sum()),
    )


def test_gray_scott_streams_images(tmp_path):
    """Test that both Gray-Scott fields are logged as images every frame."""
    composite = GrayScottSystem((32, 32)).prepare().system_
    trajectory = simulate(composite, until=40, dt=20, solver=RK4Solver(dt=1.0))
    problem, recording = trajectory.problem, Images()
    visual = RerunObserver().prepare(problem).visuals_[0].set_params(recording=recording).prepare(problem)
    for step in range(len(trajectory.times)):
        visual.run(FrameSpec(step, 0.0, trajectory.states[step], trajectory.parameters[-1], problem))
    assert recording.images == [('cells/u', 'Image'), ('cells/v', 'Image')] * 3
    observer = RerunObserver('gray-scott', save_path=tmp_path / 'gray-scott.rrd')
    simulate(GrayScottSystem((32, 32)), until=40, observers=[observer])
    assert observer.logged_frames_ == 3
