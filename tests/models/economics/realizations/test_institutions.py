"""Test the `_institutions` module of `skcomplex.models.economics.realizations`."""

import pytest

from skcomplex.core.base import SimulationError
from skcomplex.core.runtime import Simulation
from skcomplex.core.system import Observable, System
from skcomplex.models.economics.realizations import (
    Institution,
)

from .conftest import Trader


def _report_total(model):
    """Report an undeclared value."""
    model.report('total', 1.0)


def test_undeclared_report_raises():
    """Reporting a name the owner does not declare raises an error."""
    system = System(components=[Trader.group('traders', 1)], interactions=[Institution('reader', rule=_report_total)])
    with pytest.raises(SimulationError, match='declare'):
        Simulation(system).run(until=1)


def test_observable_and_set_parameter_by_view():
    """Institutions read observables and reject unknown parameters."""

    class Reader(Institution):
        reports = ('total',)

        def act(self, model):
            model.report('total', model.observable('wealth'))
            with pytest.raises(SimulationError, match='No parameter'):
                model.set_parameter('missing', 1.0)

    system = System(
        components=[Trader.group('traders', 4, money=2.0)],
        interactions=[Reader('reader')],
    )
    observables = [Observable('wealth', group='traders', variable='money', reduce='sum')]
    trajectory = Simulation(system, random_state=0, observables=observables).run(until=2).trajectory_
    assert trajectory['reader.total'][-1] == 8.0
