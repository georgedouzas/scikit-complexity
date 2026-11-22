"""Test the `_invariants` module of `skcomplex.core.engine.runtime`."""

import numpy as np
import pytest

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import transfer
from skcomplex.core.engine.runtime import InvariantObserver
from skcomplex.core.errors import ConservationError, ValidationError
from skcomplex.core.systems import System

TOTAL_WEALTH = Observable('total_wealth', group='traders', variable='wealth', reduce='sum')


def _give(agent, system):
    """Hand one unit of wealth to the next agent of the group, which keeps the total."""
    others = system.get_agents('traders')
    receiver = others[(others.index(agent) + 1) % len(others)]
    if agent.wealth >= 1:
        transfer(agent, receiver, 1.0, 'wealth')


def _print(agent, system):
    """Add one unit of wealth from nothing."""
    del system
    agent.wealth += 1.0


def _traders(rule):
    """Return a prepared system of four traders who hold wealth and follow a rule at every step."""
    gifts = Interaction('gifts', roles={'target': 'traders'}, laws=lambda s: [UpdateLaw(s.target, rule)])
    return System(elements=[Element('traders', values={'wealth': 10.0}, size=4)], interactions=[gifts]).prepare()


def test_invariant_observer_passes_when_the_laws_conserve_the_invariant():
    """Test that a run whose laws only move wealth between traders keeps the total wealth invariant."""
    observer = InvariantObserver({'total_wealth': 1e-9})
    trajectory = _traders(_give).simulate(until=5, observables=[TOTAL_WEALTH], observers=[observer])
    np.testing.assert_allclose(trajectory.get_values('total_wealth'), 40.0)
    assert observer.initial_ == {'total_wealth': 40.0}


def test_simulation_invariants_name_the_observable_and_the_first_breaking_frame():
    """Test that a law creating wealth from nothing raises a conservation error at the first frame that breaks it."""
    with pytest.raises(ConservationError, match=r'invariant total_wealth changed from 40 to 44 at frame 1,'):
        _traders(_print).simulate(until=5, observables=[TOTAL_WEALTH], invariants={'total_wealth': 1e-9})


def test_invariant_observer_rejects_an_invariant_without_its_observable():
    """Test that an invariant naming no observable of the run raises a validation error that names it."""
    with pytest.raises(ValidationError, match='total_wealth'):
        _traders(_give).simulate(until=2, invariants={'total_wealth': 1e-9})


def test_invariant_observer_checks_continuous_runs():
    """Test that a rate law that changes an invariant state raises a conservation error at the first breaking frame."""
    filling = Interaction('filling', roles={'target': 'tank'}, laws=lambda s: [RateLaw(s.target.a, 1.0)])
    level = Observable('level', group='tank', variable='a', reduce='sum')
    system = System(elements=[Element('tank', values={'a': 1.0})], interactions=[filling]).prepare()
    with pytest.raises(ConservationError, match=r'invariant level .* at frame 1'):
        system.simulate(until=2, dt=0.5, observables=[level], invariants={'level': 0.1})


@pytest.mark.parametrize('invariants', [['total'], {'total': -1.0}, {'total': 'tight'}])
def test_invariant_observer_check_rejects_invalid_invariants(invariants):
    """Test that checking the observer rejects invariants that are not paths with nonnegative tolerances."""
    with pytest.raises(ValidationError, match='invariant'):
        InvariantObserver(invariants).prepare()
