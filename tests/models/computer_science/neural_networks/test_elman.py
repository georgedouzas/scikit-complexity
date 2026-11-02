"""Test the `_elman` module of `skcomplex.models.computer_science.neural_networks`."""

from skcomplex.engine.runtime import Simulation
from skcomplex.models import run
from skcomplex.models.computer_science.neural_networks import elman_network


def test_elman_network_learns_the_temporal_xor():
    """Test that 200 epochs of Adam predict the XOR bits and leave the random bits unpredictable."""
    trajectory = run('elman_network', settings={'until': 200, 'random_state': 0}).trajectory_
    assert trajectory.values('training/xor_error')[-1] < 0.05
    assert trajectory.values('training/random_error')[-1] > 0.2
    loss = trajectory.values('training/loss')
    assert loss[-1] < loss[0]


def test_elman_network_with_elman_original_rule():
    """Test that a truncation of one step, Elman's original rule, still predicts the XOR bits better."""
    simulation = Simulation(elman_network(truncation=1), random_state=0).check().prepare()
    trajectory = simulation.run(until=500).trajectory_
    assert trajectory.values('training/xor_error')[-1] < trajectory.values('training/random_error')[-1]


def test_elman_network_holds_one_window_per_third_bit():
    """Test that the dataset holds the 996 windows of 12 bits that start at every third bit of 3000 bits."""
    data = elman_network().interactions[0].data
    assert data.values['inputs.x'].shape == (996, 12, 1)
    assert data.values['outputs.target'].shape == (996, 12, 1)
