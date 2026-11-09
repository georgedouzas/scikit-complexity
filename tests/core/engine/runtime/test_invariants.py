"""Test the `_invariants` module of `skcomplex.core.engine.runtime`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw, RuleLaw
from skcomplex.core.components.laws.utils import transfer
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import InvariantObserver, simulate
from skcomplex.core.engine.solvers import AgentView
from skcomplex.core.errors import ConservationError, ValidationError
from skcomplex.core.specs import ObservableSpec, StateSpec, VisualSpec

TOTAL_WEALTH = ObservableSpec('total_wealth', group='traders', variable='wealth', reduce='sum')


class TraderElement(Element):
    """Agent who holds wealth and hands one unit to the next agent at every step, which keeps the total."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec()] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth

    def _get_laws(self, s):
        """Return the gift rule."""
        return [RuleLaw(self.give)]

    def give(self, agent, system):
        """Hand one unit of wealth to the next agent of the group."""
        others = system.agents('traders', AgentView)
        receiver = others[(others.index(agent) + 1) % len(others)]
        if agent.wealth >= 1:
            transfer(agent, receiver, 1.0, 'wealth')


class PrinterElement(TraderElement):
    """Agent who creates one unit of wealth from nothing at every step."""

    def give(self, agent, system):
        """Add one unit of wealth."""
        del system
        agent.wealth += 1.0


def _traders(kind):
    """Return a system of four traders of a kind."""
    return CompositeSystem(elements=[GroupElement('traders', element=kind, size=4)])


def test_invariant_observer_passes_when_the_laws_conserve_the_invariant():
    """Test that a run whose laws only move wealth between traders keeps the total wealth invariant."""
    observer = InvariantObserver({'total_wealth': 1e-9})
    trajectory = simulate(_traders(TraderElement), until=5, observables=[TOTAL_WEALTH], observers=[observer])
    np.testing.assert_allclose(trajectory.values('total_wealth'), 40.0)
    assert observer.initial_ == {'total_wealth': 40.0}


def test_simulation_invariants_name_the_observable_and_the_first_breaking_frame():
    """Test that a law creating wealth from nothing raises a conservation error at the first frame that breaks it."""
    with pytest.raises(ConservationError, match=r'invariant total_wealth changed from 40 to 44 at frame 1,'):
        simulate(_traders(PrinterElement), until=5, observables=[TOTAL_WEALTH], invariants={'total_wealth': 1e-9})


def test_invariant_observer_rejects_an_invariant_without_its_observable():
    """Test that an invariant naming no observable of the run raises a validation error that names it."""
    with pytest.raises(ValidationError, match='total_wealth'):
        simulate(_traders(TraderElement), until=2, invariants={'total_wealth': 1e-9})


def test_invariant_observer_checks_continuous_runs():
    """Test that a rate law that changes an invariant state raises a conservation error at the first breaking frame."""

    class TankElement(Element):
        """Tank element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            a: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.a = a

        def _get_laws(self, s):
            """Return the laws of the tank element."""
            return [RateLaw(s.a, 1.0)]

    level = ObservableSpec('level', group='tank', variable='a', reduce='sum')
    system = CompositeSystem(elements=[TankElement('tank')])
    with pytest.raises(ConservationError, match=r'invariant level .* at frame 1'):
        simulate(system, until=2, dt=0.5, observables=[level], invariants={'level': 0.1})


@pytest.mark.parametrize('invariants', [['total'], {'total': -1.0}, {'total': 'tight'}])
def test_invariant_observer_check_rejects_invalid_invariants(invariants):
    """Test that checking the observer rejects invariants that are not paths with nonnegative tolerances."""
    with pytest.raises(ValidationError, match='invariant'):
        InvariantObserver(invariants).prepare()
