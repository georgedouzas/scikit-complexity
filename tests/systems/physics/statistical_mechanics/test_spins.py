"""Test the `_spins` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.core.engine.problems import FrameSpec
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.engine.viz import ImageVisual, RerunObserver
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import sweep
from skcomplex.systems.physics.statistical_mechanics import IsingSystem, PottsSystem, XYSystem
from skcomplex.systems.physics.statistical_mechanics._spins import PottsSamplingInteraction, XYSamplingInteraction

from ..common.spins.conftest import _run
from ..conftest import _run_default
from .conftest import magnetization

CRITICAL_TEMPERATURE = 2 / np.log(1 + np.sqrt(2))


def test_potts_spins_take_only_their_states_and_order_cold():
    """Test that Potts spins stay within their `q` states and order into one state at a low temperature."""
    colors = np.random.default_rng(1).integers(0, 3, (12, 12)).astype(float)
    sampling = PottsSamplingInteraction('potts', target='sites', q=3)
    solver = MetropolisSolver(
        moves={'sites.color': 'random-value'},
        rule='heat-bath',
        values={'sites.color': (0.0, 1.0, 2.0)},
    )
    values = _run(sampling, 'color', {'color': colors, 'temperature': 0.1}, solver)
    assert set(np.unique(values)) <= {0.0, 1.0, 2.0}
    assert np.bincount(values[-1].astype(int).ravel()).max() > 0.6 * values[-1].size


def test_aligned_xy_spins_stay_aligned_when_cold():
    """Test that aligned planar spins keep a large magnetization at a low temperature."""
    solver = MetropolisSolver(moves={'sites.theta': 'rotate'}, width=0.5)
    angles = _run(XYSamplingInteraction('xy', target='sites'), 'theta', {'temperature': 0.1}, solver)
    assert np.abs(np.exp(1j * angles[-1]).mean()) > 0.9


class Images:
    """Recording stand-in that keeps the images logged to it."""

    def __init__(self):
        self.images = []

    def log(self, path, archetype):
        """Keep a logged image with its path."""
        self.images.append((path, archetype))


@pytest.mark.slow
def test_ising_transition():
    """Test that the ensemble magnetisation of a 64 x 64 Ising lattice drops below one half between 2.2 and 2.4."""
    temperatures = [1.5, 2.0, 2.2, 2.4, 3.0, 3.5]
    result = sweep(
        lambda temperature: IsingSystem(64, temperature=temperature, initial='up'),
        {'temperature': temperatures},
        replicates=4,
        until=1000,
        dt=50,
        warmup=200,
        outputs=[magnetization],
        random_state=0,
    )
    means = {row['temperature']: row['magnetization.mean'] for row in result.get_summary()}
    assert means[2.2] > 0.5 > means[2.4]
    assert means[1.5] > 0.95
    assert means[3.5] < 0.1
    assert 2.2 < CRITICAL_TEMPERATURE < 2.4


@pytest.mark.parametrize('factory', [PottsSystem, XYSystem])
def test_cold_spin_systems_order(factory):
    """Test that the Potts and XY systems order at low temperature."""
    system = factory(16, temperature=0.2)
    trajectory = _run_default(system, until=300, dt=100, solver=system.prepare().system_.solver)
    values = trajectory.states[-1]
    if factory is PottsSystem:
        assert np.bincount(values.astype(int)).max() > 0.8 * values.size
    else:
        assert np.abs(np.exp(1j * values).mean()) > 0.8


def test_lattices_log_images():
    """Test that a lattice state is logged as an image of the lattice shape."""
    recording = Images()
    trajectory = IsingSystem(8, initial='up').prepare().simulate(until=1, dt=1)
    frame = FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], trajectory.problem)
    (
        ImageVisual(low=-1, high=1, path=trajectory.problem.elements[0].path, recording=recording)
        .prepare(trajectory.problem)
        .run(frame)
    )
    ((path, image),) = recording.images
    assert path == 'spins/spin'
    assert type(image).__name__ == 'Image'


def test_lattice_recording_is_saved(tmp_path):
    """Test that a lattice system streams to a recording with one image per frame."""
    observer = RerunObserver('ising', save_path=tmp_path / 'ising.rrd')
    IsingSystem(16).prepare().simulate(until=5, observers=[observer])
    assert observer.logged_frames_ == 6
    assert (tmp_path / 'ising.rrd').stat().st_size > 0


def test_ising_runs_on_an_odd_periodic_lattice():
    """Test that an Ising system of odd side samples its spins in a random order instead of a checkerboard."""
    trajectory = IsingSystem(9, temperature=2.0).prepare().simulate(until=2, random_state=0)
    assert trajectory.get_values('spins.spin').shape == (3, 81)


def test_ising_names_the_solver_setting_when_its_checkerboard_cannot_run():
    """Test that an explicit checkerboard on an odd periodic lattice raises an error naming the solver setting."""
    solver = MetropolisSolver(moves={'spins.spin': 'flip'}, schedule='checkerboard')
    with pytest.raises(ValidationError, match=r"solver=MetropolisSolver\(moves=\.\.\., schedule='random-sequential'\)"):
        IsingSystem(9, solver=solver).prepare()


def test_ising_records_the_energy_of_all_up_spins():
    """Test that the energy of all up Ising spins is `-2 J` per site minus the field term."""
    system = IsingSystem(8, J=1.0, h=0.3, initial='up', temperature=0.1)
    energy = system.prepare().simulate(until=2, random_state=0).get_values('energy')
    assert energy.shape == (3,)
    assert energy[0] == pytest.approx(-2 * 64 - 0.3 * 64)


def test_ising_energy_is_symmetric_under_a_global_flip():
    """Test that an Ising configuration and its global flip have the same energy without a field."""
    trajectory = IsingSystem(8, h=0.0).prepare().simulate(until=0)
    spins = trajectory.get_values('spins.spin')[0]
    flipped = IsingSystem(8, h=0.0).prepare().simulate(until=0, initial_state={'spins.spin': -spins})
    assert flipped.get_values('energy')[0] == pytest.approx(trajectory.get_values('energy')[0])


def test_ising_energy_matches_the_rolled_lattice():
    """Test that the Ising energy is the sum over the right and lower neighbours of every site."""
    trajectory = IsingSystem(6, J=0.5, h=0.2).prepare().simulate(until=0)
    spins = trajectory.get_values('spins.spin')[0].reshape(6, 6)
    bonds = (spins * np.roll(spins, 1, axis=0)).sum() + (spins * np.roll(spins, 1, axis=1)).sum()
    assert trajectory.get_values('energy')[0] == pytest.approx(-0.5 * bonds - 0.2 * spins.sum())


def test_potts_and_xy_record_their_energy():
    """Test that the Potts energy counts like neighbours and the XY energy sums the cosines of neighbours."""
    potts = PottsSystem(6, q=3, J=2.0).prepare().simulate(until=0)
    colors = potts.get_values('spins.color')[0].reshape(6, 6)
    like = sum((colors == np.roll(colors, 1, axis=axis)).sum() for axis in (0, 1))
    assert potts.get_values('energy')[0] == pytest.approx(-2.0 * like)
    xy = XYSystem(6).prepare().simulate(until=0)
    angles = xy.get_values('spins.theta')[0].reshape(6, 6)
    cosines = sum(np.cos(angles - np.roll(angles, 1, axis=axis)).sum() for axis in (0, 1))
    assert xy.get_values('energy')[0] == pytest.approx(-cosines)
