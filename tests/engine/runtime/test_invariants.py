"""Test the `_invariants` module of `skcomplex.engine.runtime`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.errors import ConservationError, ValidationError
from skcomplex.core.laws import RateLaw, RuleLaw
from skcomplex.core.laws.utils import transfer
from skcomplex.core.specs import ObservableSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import InvariantObserver, Simulation

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

    def give(self, agent, model):
        """Hand one unit of wealth to the next agent of the group."""
        others = model.agents('traders')
        receiver = others[(others.index(agent) + 1) % len(others)]
        if agent.wealth >= 1:
            transfer(agent, receiver, 1.0, 'wealth')


class PrinterElement(TraderElement):
    """Agent who creates one unit of wealth from nothing at every step."""

    def give(self, agent, model):
        """Add one unit of wealth."""
        del model
        agent.wealth += 1.0


def _traders(kind):
    """Return a system of four traders of a kind."""
    return System(elements=[GroupElement('traders', element=kind, size=4)])


def test_invariant_observer_passes_when_the_laws_conserve_the_invariant():
    """Test that a run whose laws only move wealth between traders keeps the total wealth invariant."""
    observer = InvariantObserver({'total_wealth': 1e-9})
    simulation = Simulation(_traders(TraderElement), observables=[TOTAL_WEALTH], observers=[observer])
    trajectory = simulation.run(5).trajectory_
    np.testing.assert_allclose(trajectory.values('total_wealth'), 40.0)
    assert observer.initial_ == {'total_wealth': 40.0}


def test_simulation_invariants_name_the_observable_and_the_first_breaking_frame():
    """Test that a law creating wealth from nothing raises a conservation error at the first frame that breaks it."""
    simulation = Simulation(
        _traders(PrinterElement),
        observables=[TOTAL_WEALTH],
        invariants={'total_wealth': 1e-9},
    )
    with pytest.raises(ConservationError, match=r'invariant total_wealth changed from 40 to 44 at frame 1,'):
        simulation.run(5)


def test_invariant_observer_rejects_an_invariant_without_its_observable():
    """Test that an invariant naming no observable of the run raises a validation error that names it."""
    with pytest.raises(ValidationError, match='total_wealth'):
        Simulation(_traders(TraderElement), invariants={'total_wealth': 1e-9}).run(2)


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
    system = System(elements=[TankElement('tank')])
    simulation = Simulation(system, dt=0.5, observables=[level], invariants={'level': 0.1})
    with pytest.raises(ConservationError, match=r'invariant level .* at frame 1'):
        simulation.run(2)


@pytest.mark.parametrize('invariants', [['total'], {'total': -1.0}, {'total': 'tight'}])
def test_invariant_observer_check_rejects_invalid_invariants(invariants):
    """Test that checking the observer rejects invariants that are not paths with nonnegative tolerances."""
    with pytest.raises(ValidationError, match='invariant'):
        InvariantObserver(invariants).check()
