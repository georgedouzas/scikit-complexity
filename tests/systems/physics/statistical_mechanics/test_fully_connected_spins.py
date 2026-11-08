"""Test the `_fully_connected_spins` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np
import pytest
from scipy.optimize import brentq

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.statistical_mechanics import CurieWeissSystem, SherringtonKirkpatrickSystem


def spontaneous_magnetization(temperature):
    """Return the positive root of `m = tanh(m / T)`, the Curie-Weiss magnetization for `J = 1`."""
    return brentq(lambda m: m - np.tanh(m / temperature), 1e-6, 1.0)


@pytest.mark.parametrize('temperature', [0.5, 2.0])
def test_curie_weiss_magnetization(temperature):
    """Test that 2000 spins order at `T = 0.5 J` as `m = tanh(m J / T)` predicts and stay disordered at `T = 2 J`."""
    system = CurieWeissSystem(2000, temperature=temperature, random_state=1).prepare().system_
    trajectory = simulate(system, until=30, dt=1.0, random_state=0, observables=CurieWeissSystem.observables)
    magnetization = np.abs(trajectory.values('magnetization')[10:]).mean()
    if temperature < 1:
        assert abs(magnetization - spontaneous_magnetization(temperature)) < 0.02
    else:
        assert magnetization < 0.05


def test_sherrington_kirkpatrick_energy_and_freezing():
    """Test that the energy above `T_c` is `-1 / (2 T)` and that below `T_c` the spins freeze with no magnetization."""
    frozen = []
    for temperature in (2.0, 0.3):
        system = SherringtonKirkpatrickSystem(200, temperature=temperature, random_state=0).prepare().system_
        couplings = system.elements[0].values['couplings']
        trajectory = simulate(
            system,
            until=100,
            dt=1.0,
            random_state=0,
            observables=SherringtonKirkpatrickSystem.observables,
        )
        spins = trajectory.values('spins.spin')[50:]
        energy = -0.5 * np.einsum('ti,ij,tj->t', spins, couplings, spins).mean() / 200
        frozen.append(np.mean(spins.mean(axis=0) ** 2))
        if temperature > 1:
            assert abs(energy + 1 / (2 * temperature)) < 0.03
        else:
            assert energy < -0.65
    assert frozen[0] < 0.1
    assert frozen[1] > 0.6
