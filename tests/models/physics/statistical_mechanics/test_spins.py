"""Test the `_spins` module of `skcomplex.models.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.engine.problems import FrameSpec
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import ImageVisual, RerunObserver
from skcomplex.experiments import Experiment, GridDesign
from skcomplex.models.physics.statistical_mechanics import ising, potts, xy

from .conftest import magnetization

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
    result = (
        Experiment(
            lambda temperature: ising(64, temperature=temperature, initial='up'),
            GridDesign(),
            {'temperature': temperatures},
            replicates=4,
            until=1000,
            dt=50,
            warmup=200,
            outputs=[magnetization],
            random_state=0,
        )
        .check()
        .prepare()
        .run()
        .result_
    )
    means = {row['temperature']: row['magnetization.mean'] for row in result.summary()}
    assert means[2.2] > 0.5 > means[2.4]
    assert means[1.5] > 0.95
    assert means[3.5] < 0.1
    assert 2.2 < CRITICAL_TEMPERATURE < 2.4


@pytest.mark.parametrize('factory', [potts, xy])
def test_cold_spin_models_order(factory):
    """Test that the Potts and XY models order at low temperature."""
    simulation = Simulation(factory(16, temperature=0.2), dt=100, random_state=0).check().prepare().run(until=300)
    values = simulation.state_
    if factory is potts:
        assert np.bincount(values.astype(int)).max() > 0.8 * values.size
    else:
        assert np.abs(np.exp(1j * values).mean()) > 0.8


def test_lattices_log_images():
    """Test that a lattice state is logged as an image of the lattice shape."""
    recording = Images()
    simulation = Simulation(ising(8, initial='up'), dt=1).check().prepare().run(until=1)
    frame = FrameSpec(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    (
        ImageVisual(low=-1, high=1, path=simulation.problem_.elements[0].path, recording=recording)
        .check()
        .prepare(simulation.problem_)
        .run(frame)
    )
    ((path, image),) = recording.images
    assert path == 'spins/spin'
    assert type(image).__name__ == 'Image'


def test_lattice_recording_is_saved(tmp_path):
    """Test that a lattice model streams to a recording with one image per frame."""
    observer = RerunObserver('ising', save_path=tmp_path / 'ising.rrd')
    Simulation(ising(16), dt=1, observers=[observer]).check().prepare().run(until=5)
    assert observer.logged_frames_ == 6
    assert (tmp_path / 'ising.rrd').stat().st_size > 0
