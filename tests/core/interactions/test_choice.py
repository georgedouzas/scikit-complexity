"""Test the `_choice` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import ChoiceInteraction
from skcomplex.core.laws.utils import mean
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec


class ChooserElement(Element):
    """Member with one discrete state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        option: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.option = option


def _run_choice(choice: ChoiceInteraction, initial: np.ndarray, until: int) -> Trajectory:
    """Return the trajectory of a group of choosers bound to a choice."""
    group = GroupElement('members', element=ChooserElement, size=len(initial), values={'option': initial})
    system = System(elements=[group], interactions=[choice], time=DiscreteTime())
    return Simulation(system, random_state=0).check().prepare().run(until=until).trajectory_


def _fixed(namespace: object, option: float) -> float:
    """Return a utility equal to the option."""
    return option


def test_choice_interaction_heat_bath_reaches_mean_field():
    """Test that fully connected spins under the heat-bath rule at intensity 2 reach the root of `m = tanh(2 m)`."""
    choice = ChoiceInteraction(
        'choice',
        members='members',
        state='option',
        options=(-1, 1),
        utility=lambda s, option: option * mean(s.members.option),
        rule='heat_bath',
        intensity=2.0,
    )
    initial = np.where(np.random.default_rng(0).random(1000) < 0.7, 1.0, -1.0)
    magnetization = _run_choice(choice, initial, 40).values('members.option').mean(axis=1)
    root = 0.9575040240772687
    assert abs(magnetization[-10:].mean() - root) < 0.02
    assert abs(np.tanh(2 * root) - root) < 1e-12


def test_choice_interaction_logit_draws_softmax_probabilities():
    """Test that the logit rule draws every option with probability `exp(beta u) / sum`."""
    choice = ChoiceInteraction('choice', members='members', state='option', options=(0, 1, 2), utility=_fixed)
    values = _run_choice(choice, np.zeros(20000), 10).values('members.option')[1:]
    frequencies = np.bincount(values.astype(int).ravel(), minlength=3) / values.size
    expected = np.exp([0.0, 1.0, 2.0]) / np.exp([0.0, 1.0, 2.0]).sum()
    assert np.allclose(frequencies, expected, atol=0.01)


@pytest.mark.parametrize('rule', ['metropolis', 'fermi'])
def test_choice_interaction_proposal_rules_reach_boltzmann_distribution(rule):
    """Test that the Metropolis and Fermi rules converge to the distribution `exp(beta u) / sum`."""
    choice = ChoiceInteraction(
        'choice',
        members='members',
        state='option',
        options=(0, 1, 2),
        utility=_fixed,
        rule=rule,
        intensity=0.5,
    )
    values = _run_choice(choice, np.zeros(5000), 60).values('members.option')[-20:]
    frequencies = np.bincount(values.astype(int).ravel(), minlength=3) / values.size
    expected = np.exp([0.0, 0.5, 1.0]) / np.exp([0.0, 0.5, 1.0]).sum()
    assert np.allclose(frequencies, expected, atol=0.01)


def test_choice_interaction_keeps_values_among_options():
    """Test that a proposal rule only ever sets the state to one of the options."""
    choice = ChoiceInteraction(
        'choice',
        members='members',
        state='option',
        options=(-1, 1),
        utility=_fixed,
        rule='metropolis',
    )
    values = _run_choice(choice, np.ones(200), 10).values('members.option')
    assert set(np.unique(values).tolist()) <= {-1.0, 1.0}


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'rule': 'glauber'}, 'Unknown rule'),
        ({'options': (1,)}, 'at least 2 options'),
        ({'utility': 1.0}, 'must be callable'),
    ],
)
def test_check_rejects_invalid_arguments(arguments, message):
    """Test that an unknown rule, a single option, or a utility that is not callable is rejected."""
    values = {'members': 'members', 'state': 'option', 'options': (0, 1), 'utility': _fixed, **arguments}
    with pytest.raises(ValidationError, match=message):
        ChoiceInteraction('choice', **values).check()
