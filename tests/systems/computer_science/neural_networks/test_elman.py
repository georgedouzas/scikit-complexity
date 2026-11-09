"""Test the `_elman` module of `skcomplex.systems.computer_science.neural_networks`."""

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.computer_science.neural_networks import ElmanNetworkSystem


def test_elman_network_learns_the_temporal_xor():
    """Test that 200 epochs of Adam predict the XOR bits and leave the random bits unpredictable."""
    trajectory = simulate(ElmanNetworkSystem(), until=200, random_state=0)
    assert trajectory.get_values('training/xor_error')[-1] < 0.05
    assert trajectory.get_values('training/random_error')[-1] > 0.2
    loss = trajectory.get_values('training/loss')
    assert loss[-1] < loss[0]


def test_elman_network_with_elman_original_rule():
    """Test that a truncation of one step, Elman's original rule, still predicts the XOR bits better."""
    simulation = simulate(ElmanNetworkSystem(truncation=1).prepare().system_, until=500, random_state=0)
    trajectory = simulation
    assert trajectory.get_values('training/xor_error')[-1] < trajectory.get_values('training/random_error')[-1]


def test_elman_network_holds_one_window_per_third_bit():
    """Test that the dataset holds the 996 windows of 12 bits that start at every third bit of 3000 bits."""
    data = ElmanNetworkSystem().prepare().system_.interactions[0].data
    assert data.values['inputs.x'].shape == (996, 12, 1)
    assert data.values['outputs.target'].shape == (996, 12, 1)
