"""Test the `_feedback` module of `skcomplex.core.components.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import FeedbackInteraction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import SimulationError, ValidationError
from skcomplex.core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec

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

    def count(self, agent, system):
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

    def act(self, system):
        """Report the total and raise the rate when it exceeds the target."""
        total = system.observable('total')
        system.report('total', total)
        if total > self.target:
            system.set_parameter('rate', system.parameter('rate') + 1.0)


def _report_total(system):
    """Report an undeclared value."""
    system.report('total', 1.0)


def test_act_changes_a_parameter_only_above_the_target():
    """Test that the rate changes only at the rounds where the observable exceeds the target."""
    system = CompositeSystem(
        elements=[GroupElement('counters', element=CounterElement, size=1)],
        interactions=[ControllerInteraction('controller')],
    )
    observables = [ObservableSpec('total', group='counters', variable='level', reduce='sum')]
    trajectory = simulate(system, until=10, random_state=0, observables=observables)
    totals = trajectory.values('controller.total')[1:]
    changes = np.diff(trajectory.values('controller.rate'))
    np.testing.assert_array_equal(changes, (totals > 5.0).astype(float))


def test_act_raises_for_an_undeclared_report():
    """Test that reporting a name the feedback does not declare raises an error."""
    system = CompositeSystem(
        elements=[GroupElement('participants', element=ParticipantElement, size=1)],
        interactions=[FeedbackInteraction('reader', rule=_report_total)],
    )
    with pytest.raises(SimulationError, match='declare'):
        simulate(system, until=1)


def test_act_raises_without_a_rule():
    """Test that a feedback with neither a rule nor its own `act` raises an error."""
    system = CompositeSystem(
        elements=[GroupElement('participants', element=ParticipantElement, size=1)],
        interactions=[FeedbackInteraction('reader')],
    )
    with pytest.raises(ValidationError, match='no rule'):
        simulate(system, until=1)
