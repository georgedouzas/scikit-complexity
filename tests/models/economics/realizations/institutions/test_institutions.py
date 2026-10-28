"""Test the `_institutions` module of `skcomplex.models.economics.realizations.institutions`."""

import pytest

from skcomplex.core.elements import GroupElement
from skcomplex.core.errors import SimulationError
from skcomplex.core.specs import ObservableSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.economics.realizations.institutions import (
    InstitutionInteraction,
)

from ..conftest import TraderElement


def _report_total(model):
    """Report an undeclared value."""
    model.report('total', 1.0)


def test_undeclared_report_raises():
    """Reporting a name the owner does not declare raises an error."""
    system = System(
        elements=[GroupElement('traders', element=TraderElement, size=1)],
        interactions=[InstitutionInteraction('reader', rule=_report_total)],
    )
    with pytest.raises(SimulationError, match='declare'):
        Simulation(system).run(until=1)


def test_observable_and_set_parameter_by_view():
    """Institutions read observables and reject unknown parameters."""

    class ReaderInteraction(InstitutionInteraction):
        reports = ('total',)

        def act(self, model):
            model.report('total', model.observable('wealth'))
            with pytest.raises(SimulationError, match='No parameter'):
                model.set_parameter('missing', 1.0)

    system = System(
        elements=[GroupElement('traders', element=TraderElement, size=4, values={'money': 2.0})],
        interactions=[ReaderInteraction('reader')],
    )
    observables = [ObservableSpec('wealth', group='traders', variable='money', reduce='sum')]
    trajectory = Simulation(system, random_state=0, observables=observables).run(until=2).trajectory_
    assert trajectory.values('reader.total')[-1] == 8.0
