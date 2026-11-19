"""Test thermal sampling, the Boltzmann weight of an energy at a temperature."""

from typing import Annotated

import numpy as np
import pytest

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

COLD = 1e-9


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


def _total_energy(spins):
    """Return the Ising energy of a periodic lattice."""
    return -(spins * np.roll(spins, 1, axis=-1)).sum(axis=(-1, -2)) - (spins * np.roll(spins, 1, axis=-2)).sum(
        axis=(-1, -2),
    )


def _spins_at(energy, rule='metropolis'):
    """Return the lattice frames of random spins whose energy is sampled at a temperature near 0."""
    values = {'s': np.random.default_rng(0).choice([-1.0, 1.0], size=(16, 16))}
    spins = Element('spins', values=values, space=LatticeSpace((16, 16)))
    sampling = ThermalSamplingInteraction('ising', target='spins', state='s', energy=energy, temperature=COLD)
    solver = MetropolisSolver(moves={'spins.s': 'flip'}, rule=rule, values={'spins.s': (-1.0, 1.0)})
    system = System(elements=[spins], interactions=[sampling], solver=solver).prepare()
    return system.simulate(until=60, random_state=0).get_lattice('spins.s')


@pytest.mark.parametrize('rule', ['metropolis', 'glauber', 'heat-bath'])
def test_thermal_sampling_near_zero_temperature_never_raises_the_energy(rule):
    """Test that the total energy of an Ising lattice never grows from one frame to the next near temperature 0."""
    energies = _total_energy(_spins_at(_energy, rule))
    assert np.all(np.diff(energies) <= 0)
    assert energies[-1] < energies[0]


def test_thermal_sampling_near_zero_temperature_accepts_moves_that_keep_the_energy():
    """Test that the Metropolis rule near temperature 0 accepts a move that keeps the energy."""
    frames = _spins_at(lambda n: 0 * n.target.s)
    assert np.array_equal(frames[1], -frames[0])
