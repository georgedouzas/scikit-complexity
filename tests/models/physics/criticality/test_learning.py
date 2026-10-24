"""Test the `_learning` module of `skcomplex.models.physics.criticality`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.criticality import sk_game


def test_sk_game_freezes_at_low_temperature():
    """Test that SK game agents freeze into a rewarding state at low temperature and explore noisily at high."""
    outcomes = []
    for temperature in (0.05, 3.0):
        system = sk_game(128, temperature=temperature)
        couplings = system.elements[0].values['couplings']
        actions = np.asarray(Simulation(system, dt=1, random_state=0).run(150).trajectory_['agents.sigma'])[-50:]
        reward = np.einsum('ti,ij,tj->t', actions, couplings, actions).mean() / actions.shape[1]
        outcomes.append((reward, (actions[1:] != actions[:-1]).mean()))
    (frozen_reward, frozen_flips), (noisy_reward, noisy_flips) = outcomes
    assert frozen_flips < 0.02
    assert noisy_flips > 0.3
    assert frozen_reward > 1.0
    assert noisy_reward < 0.2
