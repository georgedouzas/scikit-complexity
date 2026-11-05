"""Test the `_feedback` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import FeedbackInteraction
from skcomplex.core.laws import RuleLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import SimulationError, ValidationError
from skcomplex.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec

from .conftest import ParticipantElement


class CounterElement(Element):
    """Counter whose level rises by one every step."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level

    def _get_laws(self, s):
        """Return the counting rule."""
        return [RuleLaw(self.count)]

    def count(self, agent, model):
        """Raise the level by one."""
        agent.level += 1


class ControllerInteraction(FeedbackInteraction):
    """Feedback that raises its rate whenever the total level exceeds the target."""

    reports = ('total',)

    def __init__(
        self,
        label: str,
        *,
        rule=None,
        phase: str | None = None,
        rate: Annotated[float, ParameterSpec()] = 0.0,
        target: float = 5.0,
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.rate = rate
        self.target = target

    def act(self, model):
        """Report the total and raise the rate when it exceeds the target."""
        total = model.observable('total')
        model.report('total', total)
        if total > self.target:
            model.set_parameter('rate', model.parameter('rate') + 1.0)


def _report_total(model):
    """Report an undeclared value."""
    model.report('total', 1.0)


def test_act_changes_a_parameter_only_above_the_target():
    """Test that the rate changes only at the rounds where the observable exceeds the target."""
    system = System(
        elements=[GroupElement('counters', element=CounterElement, size=1)],
        interactions=[ControllerInteraction('controller')],
    )
    observables = [ObservableSpec('total', group='counters', variable='level', reduce='sum')]
    simulation = Simulation(system, random_state=0, observables=observables).prepare().run(until=10)
    totals = simulation.trajectory_.values('controller.total')[1:]
    changes = np.diff(simulation.trajectory_.values('controller.rate'))
    np.testing.assert_array_equal(changes, (totals > 5.0).astype(float))


def test_act_raises_for_an_undeclared_report():
    """Test that reporting a name the feedback does not declare raises an error."""
    system = System(
        elements=[GroupElement('participants', element=ParticipantElement, size=1)],
        interactions=[FeedbackInteraction('reader', rule=_report_total)],
    )
    with pytest.raises(SimulationError, match='declare'):
        Simulation(system).prepare().run(until=1)


def test_act_raises_without_a_rule():
    """Test that a feedback with neither a rule nor its own `act` raises an error."""
    system = System(
        elements=[GroupElement('participants', element=ParticipantElement, size=1)],
        interactions=[FeedbackInteraction('reader')],
    )
    with pytest.raises(ValidationError, match='no rule'):
        Simulation(system).prepare().run(until=1)
