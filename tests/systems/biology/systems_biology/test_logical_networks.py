"""Test the `_logical_networks` module of `skcomplex.systems.biology.systems_biology`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.systems_biology import GeneralizedLogicalNetworkSystem, PetriNetSystem
from skcomplex.systems.biology.systems_biology._logical_networks import POSITIVE_CIRCUIT


def test_negative_circuit_cycles_and_positive_circuit_settles():
    """Test Thomas's rules: a negative circuit keeps cycling through its states, a positive one stops in a state."""
    cycle = simulate(GeneralizedLogicalNetworkSystem(), dt=0.5, random_state=0, until=100)
    levels = np.c_[cycle.values('x.level'), cycle.values('y.level')]
    assert levels.min() == 0
    assert levels[:, 0].max() == 2
    assert levels[:, 1].max() == 1
    assert len({tuple(state) for state in levels[100:]}) >= 3
    switch = simulate(
        GeneralizedLogicalNetworkSystem(POSITIVE_CIRCUIT).prepare().system_,
        until=50,
        dt=0.5,
        random_state=0,
    )
    final = switch
    assert (final.values('x.level')[-20:] == final.values('x.level')[-1]).all()
    assert (final.values('x.level')[-1], final.values('y.level')[-1]) in {(0.0, 1.0), (2.0, 0.0)}


def test_petri_net_keeps_its_place_invariant_and_integer_tokens():
    """Test that the gene of the Petri net is on or off, never both, and tokens stay nonnegative integers."""
    trajectory = simulate(PetriNetSystem(), dt=0.5, random_state=0, until=100)
    np.testing.assert_array_equal(trajectory.values('gene_on.n') + trajectory.values('gene_off.n'), 1.0)
    for place in ('mRNA', 'protein'):
        tokens = trajectory.values(f'{place}.n')
        assert np.all(tokens >= 0)
        np.testing.assert_array_equal(tokens, np.round(tokens))
