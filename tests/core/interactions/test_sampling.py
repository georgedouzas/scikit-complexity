"""Test the `_sampling` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.interactions import SamplingInteraction
from skcomplex.core.laws.utils import neighbors
from skcomplex.core.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec

SIDE = 16


class SiteElement(Element):
    """Site with a value `s`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        s: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s


def _ising(namespace):
    """Return the local energy of a spin with its four neighbours."""
    return -namespace.target.s * neighbors(namespace.target.other.s)


def _final(sampling, until, initial, seed=0):
    """Return the values of the sites after some sweeps."""
    sites = LatticeElement('sites', element=SiteElement, space=LatticeSpace((SIDE, SIDE)), values={'s': initial})
    system = System(elements=[sites], interactions=[sampling], time=DiscreteTime())
    return Simulation(system, random_state=seed).check().prepare().run(until=until).state_


def _spins(seed=1):
    """Return random spins of the sites."""
    return np.random.default_rng(seed).choice([-1.0, 1.0], SIDE * SIDE)


def test_ising_magnetization_orders_well_below_the_critical_temperature():
    """Test that the Ising energy sampled at temperature 1 orders with absolute magnetization above 0.9."""
    sampling = SamplingInteraction('ising', target='sites', state='s', energy=_ising, temperature=1.0)
    assert abs(_final(sampling, 300, _spins()).mean()) > 0.9


def test_ising_magnetization_vanishes_well_above_the_critical_temperature():
    """Test that the Ising energy sampled at temperature 5 stays disordered."""
    sampling = SamplingInteraction('ising', target='sites', state='s', energy=_ising, temperature=5.0, rule='glauber')
    assert abs(_final(sampling, 100, np.ones(SIDE * SIDE)).mean()) < 0.3


def test_gaussian_moves_sample_the_boltzmann_distribution():
    """Test that Gaussian moves in a harmonic energy sample values of variance equal to the temperature."""
    sampling = SamplingInteraction(
        'harmonic',
        target='sites',
        state='s',
        energy=lambda n: n.target.s**2 / 2,
        move='gaussian',
        width=1.5,
        temperature=2.0,
    )
    values = np.concatenate([_final(sampling, 200, np.zeros(SIDE * SIDE), seed) for seed in range(4)])
    assert values.var() == pytest.approx(2.0, rel=0.1)


def test_function_moves_propose_the_values_it_returns():
    """Test that a function move that proposes random spins samples the ordered Ising phase too."""

    def redraw(values, random):
        return random.choice([-1.0, 1.0], values.size)

    sampling = SamplingInteraction('ising', target='sites', state='s', energy=_ising, move=redraw, temperature=1.0)
    final = _final(sampling, 300, _spins())
    assert set(np.unique(final)) <= {-1.0, 1.0}
    assert abs(final.mean()) > 0.9


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'energy': _ising, 'move': 'swap'}, 'move of sampling'),
        ({'energy': 1.0}, 'function of the namespace'),
        ({'energy': _ising, 'rule': 'greedy'}, 'Unknown sampler'),
        ({'energy': _ising, 'schedule': 'synchronous'}, 'Unknown Monte Carlo schedule'),
    ],
)
def test_check_rejects_moves_rules_and_energies_it_cannot_take(arguments, match):
    """Test that an unknown move, rule, or schedule, or an energy that is not a function, raise errors."""
    with pytest.raises(ValidationError, match=match):
        SamplingInteraction('sampling', target='sites', state='s', **arguments).check()
