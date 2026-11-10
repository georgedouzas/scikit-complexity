"""Test thermal sampling, Monte Carlo sweeps that read an energy at a temperature."""

from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec
from skcomplex.systems.physics.common.spins import ThermalSamplingInteraction


class SpinElement(Element):
    """Spin with one state."""

    def __init__(self, label, s: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.s = s


def _energy(namespace):
    """Return the local energy of a spin with its neighbours."""
    return -namespace.target.s * neighbors(namespace.target.other.s)


def _system(sampling):
    """Return a square lattice of spins sampled by an interaction."""
    spins = GroupElement('spins', element=SpinElement, space=LatticeSpace((16, 16)), on='sites')
    return CompositeSystem(elements=[spins], interactions=[sampling], time=DiscreteTime())


def test_thermal_sampling_gives_the_core_one_over_the_temperature():
    """A temperature gives the sweeps of the core at the intensity one over it, up to rounding."""
    thermal = ThermalSamplingInteraction('sampling', target='spins', state='s', energy=_energy, temperature=2.0)
    core = SamplingInteraction('sampling', target='spins', state='s', objective=_energy, intensity=0.5)
    first = simulate(_system(thermal), until=20, random_state=0).states
    second = simulate(_system(core), until=20, random_state=0).states
    np.testing.assert_array_equal(first, second)


def test_thermal_sampling_temperature_sweeps_live():
    """A schedule of the temperature of the label disorders an ordered lattice."""
    sampling = ThermalSamplingInteraction('sampling', target='spins', state='s', energy=_energy, temperature=0.5)
    states = simulate(
        _system(sampling),
        until=60,
        random_state=0,
        schedule={'sampling.temperature': lambda time: 0.5 if time < 30 else 20.0},
    ).states
    assert abs(states[29].mean()) > 0.9
    assert abs(states[-1].mean()) < 0.3
