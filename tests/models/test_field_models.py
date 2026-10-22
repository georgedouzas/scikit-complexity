"""Test the field models of the library and their images in Rerun."""

import numpy as np
import pytest

from skcomplex.core.analysis import dispersion_relation, fields_of
from skcomplex.core.runtime import Frame, Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver, resolve_visual
from skcomplex.models.biology.collective_behaviour import pheromone_foraging
from skcomplex.models.biology.ecology import fisher_kpp
from skcomplex.models.biology.physiology import fitzhugh_nagumo
from skcomplex.models.chemistry.pattern_formation import brusselator, gray_scott, schnakenberg
from skcomplex.models.physics.heat import heat_diffusion
from skcomplex.models.physics.pattern_formation import complex_ginzburg_landau, nonreciprocal_cahn_hilliard


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


def test_fisher_front_speed():
    """Test that the Fisher-KPP front moves at a speed within 5% of 2 sqrt(r D)."""
    for r, diffusion in [(1.0, 1.0), (0.5, 2.0)]:
        trajectory = Simulation(fisher_kpp(D=diffusion, r=r), dt=2.0).run(until=80).trajectory_
        positions = np.arange(800) * 0.25
        fronts = np.asarray([positions[np.argmax(values < 0.5)] for values in trajectory['line.u']])
        late = trajectory.times >= 40
        speed = np.polyfit(trajectory.times[late], fronts[late], 1)[0]
        assert speed == pytest.approx(2 * np.sqrt(r * diffusion), rel=0.05)


def test_turing_spots_have_predicted_wavelength():
    """Test that the Schnakenberg model forms a stationary pattern with the wavelength of the linear prediction."""
    simulation = Simulation(schnakenberg((256,)), dt=50, solver=RK4(dt=0.01)).run(until=600)
    values = simulation.trajectory_['cells.u']
    wavelengths = [256 / np.argmax(np.abs(np.fft.rfft(row - row.mean()))) for row in values[-5:]]
    changes = np.abs(np.diff(values[-5:], axis=0)).max(axis=1)
    assert len(set(wavelengths)) == 1
    assert wavelengths[0] == pytest.approx(_turing_wavelength(0.1, 0.9, 1.0, 40.0), rel=0.1)
    assert np.all(np.diff(changes) < 0)
    assert changes[-1] < 0.05 * np.ptp(values[-1])


def test_pheromone_trail_marks_shortest_path():
    """Test that the pheromone along the line between nest and food ends higher than elsewhere."""
    simulation = Simulation(pheromone_foraging(), dt=100, random_state=0).run(until=600)
    pheromone = simulation.trajectory_.lattice('ground/pheromone.level')[-1]
    elsewhere = np.ones(pheromone.shape, dtype=bool)
    elsewhere[5:35, 9:12] = False
    assert pheromone[5:35, 10].mean() > 2 * pheromone[elsewhere].mean()
    assert pheromone[5:35, 10].min() > 0


def test_spiral_keeps_medium_active():
    """Test that the broken wave of the FitzHugh-Nagumo medium becomes a spiral that keeps exciting a probe."""
    trajectory = Simulation(fitzhugh_nagumo((100, 100)), dt=5, solver=RK4(dt=0.05)).run(until=400).trajectory_
    potential = trajectory.lattice('medium.u')
    probe = potential[:, 25, 75]
    excitations = np.sum((probe[1:] > 0) & (probe[:-1] <= 0))
    assert potential[-1].max() > 1
    assert excitations >= 3


def test_brusselator_and_heat_stay_bounded():
    """Test that the Brusselator stays bounded and insulated heat keeps its total."""
    values = Simulation(brusselator((32, 32)), dt=1, solver=RK4(dt=0.01)).run(until=20).state_
    assert np.all(np.isfinite(values))
    assert values.max() < 20
    trajectory = Simulation(heat_diffusion((32, 32)), dt=5).run(until=50).trajectory_
    assert trajectory['plate.u'].sum(axis=1) == pytest.approx(np.full(len(trajectory), trajectory['plate.u'][0].sum()))


def test_gray_scott_streams_images(tmp_path):
    """Test that both Gray-Scott fields are logged as images every frame."""
    simulation = Simulation(gray_scott((32, 32)), dt=20, solver=RK4(dt=1.0)).run(until=40)
    element = simulation.problem_.elements[0]
    recording = Images()
    for step in range(len(simulation.trajectory_)):
        frame = Frame(step, 0.0, simulation.trajectory_.states[step], simulation.parameters_, simulation.problem_)
        resolve_visual(element.visual).log(recording, element, frame)
    assert recording.images == [('cells/u', 'Image'), ('cells/v', 'Image')] * 3
    observer = RerunObserver('gray-scott', save_path=tmp_path / 'gray-scott.rrd')
    Simulation(gray_scott((32, 32)), dt=20, solver=RK4(dt=1.0), observers=[observer]).run(until=40)
    assert observer.logged_frames_ == 3


def test_complex_ginzburg_landau_seeds_both_parts():
    """Test that the random state fixes both the real and the imaginary part of the initial amplitude."""
    first, second, other = (complex_ginzburg_landau((8, 8), random_state=seed) for seed in (1, 1, 2))
    for part in ('u', 'v'):
        values = [np.asarray(system.elements[0].values[part]) for system in (first, second, other)]
        np.testing.assert_array_equal(values[0], values[1])
        assert not np.array_equal(values[0], values[2])


def _domain_speed(**coupling):
    """Return the speed of the domains of the nonreciprocal Cahn-Hilliard model over the second half of a run."""
    length = 40.0
    system = nonreciprocal_cahn_hilliard((64,), length=length, **coupling)
    phi = fields_of(Simulation(system, dt=2.0).run(until=300).trajectory_, 'mixture.phi')
    modes = np.fft.rfft(phi, axis=1)
    mode = 1 + np.argmax(np.abs(modes[-1, 1:]))
    phases = np.unwrap(np.angle(modes[75:, mode]))
    drift = np.polyfit(2.0 * np.arange(75, len(phi)), phases, 1)[0]
    return abs(drift) / (2 * np.pi * mode / length)


def test_nonreciprocal_cahn_hilliard_domains_travel_above_threshold():
    """Test that the demixed domains travel when alpha^2 - chi^2 exceeds D^2, and stay put otherwise."""
    assert _domain_speed(alpha=0.3) > 0.05
    assert _domain_speed(alpha=0.05) < 0.005
    assert _domain_speed(chi=0.3, alpha=0.0) < 0.005


def test_nonreciprocal_cahn_hilliard_dispersion_relation():
    """Test that the unstable band propagates above threshold, on the band edge of Brauns and Marchetti."""
    for alpha, travels in [(0.3, True), (0.05, False)]:
        relation = dispersion_relation(nonreciprocal_cahn_hilliard((256,), length=200.0, alpha=alpha))
        growing = relation.growth_rates > 1e-9
        assert (np.abs(relation.frequencies[growing]).max() > 1e-6) == travels
        edge = np.sqrt(1 - 0.1) if travels else np.sqrt(1 - alpha**2 / 0.1)
        np.testing.assert_allclose(relation.magnitudes[growing].max(), edge, atol=2 * np.pi / 200.0)
