"""Test the `_conservation` module of `skcomplex.engine.runtime`."""

from dataclasses import replace

import pytest

from skcomplex.core.errors import SimulationError, ValidationError
from skcomplex.core.system import System
from skcomplex.engine.runtime import ConservationObserver, Simulation
from skcomplex.models import create
from skcomplex.models.biology.ecology import PopulationElement


def test_conservation_observer_keeps_the_totals():
    """Test that the conservation observer tracks the total of every quantity of a balanced run."""
    observer = ConservationObserver()
    Simulation(create('bathtub'), dt=1, observers=[observer]).check().run(10)
    assert observer.totals_['water'] == pytest.approx(25.0)
    assert observer.initial_ is not None


def test_conservation_observer_needs_a_transfer_network():
    """Test that a problem without a transfer network, such as a partitioned one, cannot be monitored."""
    problem = Simulation(System(elements=[PopulationElement('a')])).check().prepare().problem_
    with pytest.raises(SimulationError, match='transfer network'):
        ConservationObserver().check().prepare(replace(problem, transfer_network=None))


@pytest.mark.parametrize('settings', [{'rtol': -1.0}, {'atol': 'small'}])
def test_conservation_observer_check_rejects_invalid_tolerances(settings):
    """Test that checking the observer rejects a tolerance that is not a nonnegative number."""
    with pytest.raises(ValidationError, match='nonnegative'):
        ConservationObserver(**settings).check()
