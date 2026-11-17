"""Test the `_conservation` module of `skcomplex.core.engine.runtime`."""

from dataclasses import replace
from typing import Annotated

import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import TransferInteraction
from skcomplex.core.components.laws import Sink, Source
from skcomplex.core.components.laws.utils import step, where
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import ConservationObserver, simulate
from skcomplex.core.errors import SimulationError, ValidationError
from skcomplex.core.specs import StateSpec
from skcomplex.systems.biology.common.populations import PopulationElement


class TubElement(Element):
    """Tub of water, a quantity that flows in and out."""

    def __init__(self, label: str, water: Annotated[float, StateSpec(quantity='water')] = 50.0) -> None:
        super().__init__(label)
        self.water = water


def test_conservation_observer_keeps_the_totals():
    """Test that the conservation observer tracks the total of every quantity of a balanced run."""
    observer = ConservationObserver()
    faucet = TransferInteraction(
        'faucet',
        source='tub',
        target='tub',
        state=Source(),
        into='water',
        amount=lambda s: 5.0 * step(s.t, 5.0),
        flow='faucet',
    )
    drain = TransferInteraction(
        'drain',
        source='tub',
        target='tub',
        state='water',
        into=Sink(),
        amount=lambda s: where(s.source.water > 0, 5.0, 0),
        flow='drain',
    )
    system = CompositeSystem(elements=[TubElement('tub')], interactions=[faucet, drain])
    simulate(system, until=10, dt=1, observers=[observer])
    assert observer.totals_['water'] == pytest.approx(25.0)
    assert observer.initial_ is not None


def test_conservation_observer_needs_a_transfer_network():
    """Test that a problem without a transfer network, such as a partitioned one, cannot be monitored."""
    problem = simulate(CompositeSystem(elements=[PopulationElement('a')]), until=0).problem
    with pytest.raises(SimulationError, match='transfer network'):
        ConservationObserver().prepare(replace(problem, transfer_network=None))


@pytest.mark.parametrize('settings', [{'rtol': -1.0}, {'atol': 'small'}])
def test_conservation_observer_check_rejects_invalid_tolerances(settings):
    """Test that checking the observer rejects a tolerance that is not a nonnegative number."""
    with pytest.raises(ValidationError, match='nonnegative'):
        ConservationObserver(**settings).prepare()
