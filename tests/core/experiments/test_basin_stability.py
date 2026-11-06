"""Test the basin stability of the attractors of a system over random initial states."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import BasinStabilityExperiment
from skcomplex.systems.physics.nonlinear_oscillators import duffing

BOUNDS = {'duffing.x': (-2.0, 2.0), 'duffing.x_d1': (-2.0, 2.0)}


def side(trajectory):
    """Return the well an unforced double-well oscillator settles in."""
    return 'right' if trajectory.values('duffing.x')[-1] > 0 else 'left'


def test_double_well_basins_are_symmetric():
    """Test that the damped double-well oscillator's two basins hold about half of the initial states each."""
    experiment = BasinStabilityExperiment(duffing(gamma=0.0), BOUNDS, side, n=80, until=40, dt=0.5, random_state=0)
    experiment.prepare().run()
    assert set(experiment.fractions_) == {'left', 'right'}
    assert experiment.fractions_['left'] == pytest.approx(0.5, abs=3 * experiment.errors_['left'] + 0.05)
    assert len(experiment.labels_) == 80


@pytest.mark.slow
def test_vectorised_basins_match_the_serial_runs():
    """Test that basin stability over 200 initial states run as one ensemble equals the serial runs."""
    settings = {'n': 200, 'until': 40, 'dt': 0.5, 'random_state': 0}
    serial = BasinStabilityExperiment(duffing(gamma=0.0), BOUNDS, side, **settings).prepare().run()
    vectorised = BasinStabilityExperiment(duffing(gamma=0.0), BOUNDS, side, vectorized=True, **settings)
    assert vectorised.prepare().run().fractions_ == serial.fractions_


def test_invalid_number_of_initial_states():
    """Test that zero initial states raise an error naming n."""
    with pytest.raises(ValidationError, match='n = 0'):
        BasinStabilityExperiment(duffing(gamma=0.0), BOUNDS, side, n=0).prepare()
