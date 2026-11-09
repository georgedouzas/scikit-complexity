"""Test the `_spreading` module of `skcomplex.systems.biology.common.networks`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TransitionInteraction
from skcomplex.core.components.spaces import LatticeSpace, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.systems.biology.epidemiology import SpreadingInteraction

SIDE = 20
CRITICAL = 1.65


class SiteElement(Element):
    """Site whose state is inactive, active, or removed."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        state: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.state = state


def _states(space, spreading, initial, until, seed=0):
    """Return the states of the sites at every unit of time."""
    system = CompositeSystem(
        elements=[GroupElement('sites', element=SiteElement, space=space, values={'state': initial}, on='sites')],
        interactions=[spreading],
    )
    trajectory = simulate(system, until=until, dt=1.0, random_state=seed)
    return trajectory.get_values('sites.state')


def test_contact_process_dies_out_below_and_survives_above_the_critical_rate():
    """Test that the contact process on a square lattice dies out below its critical rate and survives above it."""
    space, active = LatticeSpace((SIDE, SIDE)), np.ones(SIDE * SIDE)
    below = SpreadingInteraction('contact', target='sites', beta=0.5 * CRITICAL, gamma=1.0, form='fraction')
    above = SpreadingInteraction('contact', target='sites', beta=2 * CRITICAL, gamma=1.0, form='fraction')
    assert _states(space, below, active, 30.0)[-1].mean() == 0.0
    assert _states(space, above, active, 30.0)[-10:].mean(axis=1).min() > 0.3


def test_recovery_runs_once_per_member():
    """Test that active members recover at the rate gamma once each, not once per edge."""
    spreading = SpreadingInteraction('recovery', target='sites', beta=0.0, gamma=1.0)
    states = _states(LatticeSpace((SIDE, SIDE)), spreading, np.ones(SIDE * SIDE), 1.0)
    assert states[-1].mean() == pytest.approx(np.exp(-1.0), abs=0.06)


def test_removed_members_return_to_the_inactive_value():
    """Test that with a removed value, active members recover to it and removed members return to inactive."""
    spreading = SpreadingInteraction('cycle', target='sites', beta=4.0, gamma=1.0, omega=1.0, removed=2.0)
    initial = np.zeros(30)
    initial[0] = 1.0
    states = _states(NetworkSpace(nx.cycle_graph(30)), spreading, initial, 20.0)
    assert set(np.unique(states)) == {0.0, 1.0, 2.0}
    removed = np.argmax(states == 2.0, axis=0)
    returned = [np.any(states[start:, site] == 0.0) for site, start in enumerate(removed) if states[start, site] == 2.0]
    assert any(returned)


def test_number_form_spreads_faster_than_the_fraction_form():
    """Test that spreading by the number of neighbours activates more members early than by their fraction."""
    initial = np.zeros(SIDE * SIDE)
    initial[0] = 1.0
    forms = [
        _states(LatticeSpace((SIDE, SIDE)), SpreadingInteraction('s', target='sites', form=form), initial, 3.0)[-1]
        for form in ('number', 'fraction')
    ]
    assert forms[0].sum() > forms[1].sum()


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [({'form': 'power'}, 'form of spreading'), ({'removed': 1.0}, 'distinct value')],
)
def test_check_rejects_unknown_forms_and_equal_values(arguments, match):
    """Test that an unknown form or values of the state that are not distinct raise an error."""
    with pytest.raises(ValidationError, match=match):
        SpreadingInteraction('spreading', target='sites', **arguments).prepare()


def test_spreading_is_a_transition():
    """Test that spreading is a transition of a discrete state between its values."""
    assert issubclass(SpreadingInteraction, TransitionInteraction)
