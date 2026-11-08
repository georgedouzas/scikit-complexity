"""Test the `_conservation` module of `skcomplex.core.engine.runtime`."""

from dataclasses import replace

import pytest

from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import ConservationObserver, simulate
from skcomplex.core.errors import SimulationError, ValidationError
from skcomplex.systems.biology.common.populations import PopulationElement
from skcomplex.systems.engineering.processes import BathtubSystem


def test_conservation_observer_keeps_the_totals():
    """Test that the conservation observer tracks the total of every quantity of a balanced run."""
    observer = ConservationObserver()
    simulate(BathtubSystem().prepare().system_, until=10, dt=1, observers=[observer])
    assert observer.totals_['water'] == pytest.approx(25.0)
    assert observer.initial_ is not None


def test_conservation_observer_needs_a_transfer_network():
    """Test that a problem without a transfer network, such as a partitioned one, cannot be monitored."""
    problem = simulate(System(elements=[PopulationElement('a')]), until=0).problem
    with pytest.raises(SimulationError, match='transfer network'):
        ConservationObserver().prepare(replace(problem, transfer_network=None))


@pytest.mark.parametrize('settings', [{'rtol': -1.0}, {'atol': 'small'}])
def test_conservation_observer_check_rejects_invalid_tolerances(settings):
    """Test that checking the observer rejects a tolerance that is not a nonnegative number."""
    with pytest.raises(ValidationError, match='nonnegative'):
        ConservationObserver(**settings).prepare()
