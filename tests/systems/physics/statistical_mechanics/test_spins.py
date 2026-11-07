"""Test the `_spins` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.core.engine.problems import FrameSpec
from skcomplex.core.engine.runtime import Simulation, simulate
from skcomplex.core.engine.viz import ImageVisual, RerunObserver
from skcomplex.core.experiments import sweep
from skcomplex.systems.physics.statistical_mechanics import IsingSystem, PottsSystem, XYSystem
from skcomplex.systems.physics.statistical_mechanics._spins import PottsSamplingInteraction, XYSamplingInteraction

from ..common.spins.conftest import _run
from .conftest import magnetization


def test_potts_spins_take_only_their_states_and_order_cold():
    """Test that Potts spins stay within their `q` states and order into one state at a low temperature."""
    colors = np.random.default_rng(1).integers(0, 3, (12, 12)).astype(float)
    sampling = PottsSamplingInteraction('potts', target='sites', q=3)
    values = _run(sampling, 'color', {'color': colors, 'temperature': 0.1})
    assert set(np.unique(values)) <= {0.0, 1.0, 2.0}
    assert np.bincount(values[-1].astype(int).ravel()).max() > 0.6 * values[-1].size


def test_aligned_xy_spins_stay_aligned_when_cold():
    """Test that aligned planar spins keep a large magnetization at a low temperature."""
    angles = _run(XYSamplingInteraction('xy', target='sites', width=0.5), 'theta', {'temperature': 0.1})
    assert np.abs(np.exp(1j * angles[-1]).mean()) > 0.9


CRITICAL_TEMPERATURE = 2 / np.log(1 + np.sqrt(2))


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
        lambda temperature: IsingSystem(64, temperature=temperature, initial='up').prepare().system_,
        {'temperature': temperatures},
        replicates=4,
        until=1000,
        dt=50,
        warmup=200,
        outputs=[magnetization],
        random_state=0,
    )
    means = {row['temperature']: row['magnetization.mean'] for row in result.summary()}
    assert means[2.2] > 0.5 > means[2.4]
    assert means[1.5] > 0.95
    assert means[3.5] < 0.1
    assert 2.2 < CRITICAL_TEMPERATURE < 2.4


@pytest.mark.parametrize('factory', [PottsSystem, XYSystem])
def test_cold_spin_systems_order(factory):
    """Test that the Potts and XY systems order at low temperature."""
    simulation = (
        Simulation(factory(16, temperature=0.2).prepare().system_, dt=100, random_state=0).prepare().run(until=300)
    )
    values = simulation.state_
    if factory is PottsSystem:
        assert np.bincount(values.astype(int)).max() > 0.8 * values.size
    else:
        assert np.abs(np.exp(1j * values).mean()) > 0.8


def test_lattices_log_images():
    """Test that a lattice state is logged as an image of the lattice shape."""
    recording = Images()
    simulation = Simulation(IsingSystem(8, initial='up').prepare().system_, dt=1).prepare().run(until=1)
    frame = FrameSpec(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    (
        ImageVisual(low=-1, high=1, path=simulation.problem_.elements[0].path, recording=recording)
        .prepare(simulation.problem_)
        .run(frame)
    )
    ((path, image),) = recording.images
    assert path == 'spins/spin'
    assert type(image).__name__ == 'Image'


def test_lattice_recording_is_saved(tmp_path):
    """Test that a lattice system streams to a recording with one image per frame."""
    observer = RerunObserver('ising', save_path=tmp_path / 'ising.rrd')
    simulate(IsingSystem(16), until=5, observers=[observer])
    assert observer.logged_frames_ == 6
    assert (tmp_path / 'ising.rrd').stat().st_size > 0
