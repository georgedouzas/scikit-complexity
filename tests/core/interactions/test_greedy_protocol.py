"""Test the `_greedy_protocol` module of `skcomplex.core.interactions`."""

import pytest

from skcomplex.core.elements import GroupElement
from skcomplex.core.interactions import GreedyProtocol, MatchingInteraction
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError

from .conftest import ParticipantElement


def test_match_takes_the_lowest_feasible_limit_by_default(clear):
    """Test that every requester takes the provider of the lowest limit it accepts, at the provider's limit."""
    matches = clear(GreedyProtocol(), {'limit': [10.0, 10.0]}, {'limit': [3.0, 5.0, 12.0]})
    assert sorted((provider, value) for _, provider, _, _, value in matches) == [(3.0, 3.0), (5.0, 5.0)]


def test_match_takes_the_provider_of_the_highest_score(clear):
    """Test that requesters take the provider of the highest score instead of the lowest limit."""
    protocol = GreedyProtocol(score=lambda requester, provider: provider.value, clearing='requester')
    matches = clear(protocol, {'limit': [10.0]}, {'limit': [3.0, 5.0], 'value': [1.0, 2.0]})
    assert [(provider, value) for _, provider, _, _, value in matches] == [(5.0, 10.0)]


def test_match_respects_the_capacity_of_providers(clear):
    """Test that a provider of capacity one takes one requester although it offers more."""
    protocol = GreedyProtocol(capacity='value')
    matches = clear(protocol, {'limit': [10.0, 10.0, 10.0]}, {'limit': [1.0, 2.0], 'quantity': 5.0, 'value': 1.0})
    assert sorted(provider for _, provider, _, _, _ in matches) == [1.0, 2.0]


def test_match_considers_neighbors_and_distances_only():
    """Test that requesters match only with providers in their cell or around it, or within a distance."""
    for candidates, expected in (('neighbors', 1.0), (1.5, 1.0), ('all', 2.0)):
        requesters = GroupElement(
            'requesters',
            element=ParticipantElement,
            size=2,
            values={'limit': 10.0},
            location=[(0, 0), (4, 4)],
        )
        providers = GroupElement(
            'providers',
            element=ParticipantElement,
            size=1,
            values={'limit': 1.0, 'quantity': 2.0},
            location=[(1, 1)],
        )
        matching = MatchingInteraction(
            'matching',
            requesters='requesters',
            providers='providers',
            protocol=GreedyProtocol(candidates=candidates),
        )
        space = LatticeSpace((5, 5), neighborhood='moore', boundary='fixed')
        system = System(elements=[requesters, providers], interactions=[matching], space=space)
        trajectory = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_
        assert trajectory.values('matching.volume')[-1] == pytest.approx(expected)


@pytest.mark.parametrize(
    'arguments',
    [{'candidates': 'friends'}, {'candidates': 0.0}, {'candidates': True}, {'clearing': 'bid'}, {'score': 2}],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that unknown candidates, a distance that is not positive, an unknown clearing, and a bad score fail."""
    with pytest.raises(ValidationError):
        GreedyProtocol(**arguments).check()
