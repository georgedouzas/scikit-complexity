"""Test the `_conservation` module of `skcomplex.core.engine.runtime`."""

from dataclasses import replace
from typing import Annotated

import pytest

from skcomplex.core.components.elements import Element, Quantity
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import step, where
from skcomplex.core.engine.runtime import ConservationObserver
from skcomplex.core.errors import SimulationError, ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.biology.common.populations import PopulationElement


class TubElement(Element):
    """Tub of water, a quantity that flows in and out."""

    def __init__(self, label: str, water: Annotated[float, Quantity('water')] = 50.0) -> None:
        super().__init__(label)
        self.water = water


def test_conservation_observer_keeps_the_totals():
    """Test that the conservation observer tracks the total of every quantity of a balanced run."""
    observer = ConservationObserver()
    faucet = Interaction(
        'faucet',
        roles={'target': 'tub'},
        laws=lambda s: [RateLaw({s.target.water: +1}, 5.0 * step(s.t, 5.0), label='faucet')],
    )
    drain = Interaction(
        'drain',
        roles={'source': 'tub'},
        laws=lambda s: [RateLaw({s.source.water: -1}, where(s.source.water > 0, 5.0, 0), label='drain')],
    )
    system = System(elements=[TubElement('tub')], interactions=[faucet, drain]).prepare()
    system.simulate(until=10, dt=1, observers=[observer])
    assert observer.totals_['water'] == pytest.approx(25.0)
    assert observer.initial_ is not None


def test_conservation_observer_needs_a_transfer_network():
    """Test that a problem without a transfer network, such as a partitioned one, cannot be monitored."""
    problem = System(elements=[PopulationElement('a')]).prepare().simulate(until=0).problem
    with pytest.raises(SimulationError, match='transfer network'):
        ConservationObserver().prepare(replace(problem, transfer_network=None))


@pytest.mark.parametrize('settings', [{'rtol': -1.0}, {'atol': 'small'}])
def test_conservation_observer_check_rejects_invalid_tolerances(settings):
    """Test that checking the observer rejects a tolerance that is not a nonnegative number."""
    with pytest.raises(ValidationError, match='nonnegative'):
        ConservationObserver(**settings).prepare()
