"""Test thermal sampling, the Boltzmann weight of an energy at a temperature."""

from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.specs import StateSpec
from skcomplex.core.systems import System
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
    """Return a square lattice of spins sampled by an interaction, flipped by a Metropolis solver."""
    spins = Element('spins', values={'s': np.ones(256)}, space=LatticeSpace((16, 16)))
    return System(elements=[spins], interactions=[sampling], solver=MetropolisSolver(moves={'spins.s': 'flip'}))


def test_thermal_sampling_draws_as_the_core_sweeps_at_one_over_the_temperature():
    """An energy over a temperature of 2 gives the sweeps of the core at the intensity one half, draw for draw."""
    thermal = ThermalSamplingInteraction('sampling', target='spins', state='s', energy=_energy, temperature=2.0)
    first = _system(thermal).prepare().simulate(until=20, random_state=0).states
    core = SamplingInteraction('sampling', target='spins', state='s', objective=_energy, intensity=0.5)
    spins = GroupElement('spins', element=SpinElement, space=LatticeSpace((16, 16)), on='sites')
    composite = CompositeSystem(elements=[spins], interactions=[core], time=DiscreteTime())
    second = simulate(composite, until=20, random_state=0).states
    np.testing.assert_array_equal(first, second)


def test_thermal_sampling_temperature_sweeps_live():
    """A schedule of the temperature of the interaction disorders an ordered lattice."""
    sampling = ThermalSamplingInteraction('sampling', target='spins', state='s', energy=_energy, temperature=0.5)
    states = (
        _system(sampling)
        .prepare()
        .simulate(until=60, random_state=0, schedule={'sampling.temperature': lambda time: 0.5 if time < 30 else 20.0})
        .states
    )
    assert abs(states[29].mean()) > 0.9
    assert abs(states[-1].mean()) < 0.3
