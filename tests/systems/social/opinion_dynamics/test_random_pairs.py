"""Test the `_random_pairs` module of `skcomplex.systems.social.opinion_dynamics`."""

import networkx as nx
import pytest

from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.social.opinion_dynamics import (
    DiscussantElement,
    MeetingsInteraction,
    NeighborhoodsInteraction,
    RandomPairsInteraction,
)


class CountingPairsInteraction(RandomPairsInteraction):
    """Random pairs that count the encounters of every agent in its opinion."""

    def act_on(self, members, system):
        """Count one encounter for every member of a pair of distinct agents."""
        assert len({member.id for member in members}) == len(members) == 2
        for member in members:
            member.opinion += 1.0


def test_random_pairs_draw_one_pair_per_agent():
    """Test that a step draws as many pairs of distinct agents as there are agents."""
    people = DiscussantElement('people', opinion=0.0, size=10)
    system = System(elements=[people], interactions=[CountingPairsInteraction('meetings', target='people')])
    trajectory = system.prepare().simulate(until=3, random_state=0)
    assert trajectory.get_agents('people', 3).values['opinion'].sum() == 3 * 2 * 10


class CountingNeighborsInteraction(NeighborhoodsInteraction):
    """Neighbourhoods that record in the opinion of every agent the number of its neighbours."""

    def act_on(self, members, system):
        """Set the opinion of the first member to the number of the others."""
        members[0].opinion = float(len(members) - 1)


def test_neighborhoods_group_every_agent_with_its_neighbours():
    """Test that every agent of a path meets the agents on the nodes next to it."""
    people = DiscussantElement('people', opinion=0.0, size=4)
    system = System(
        elements=[people],
        interactions=[CountingNeighborsInteraction('meetings', target='people')],
        space=NetworkSpace(nx.path_graph(4)),
    )
    trajectory = system.prepare().simulate(until=1, random_state=0)
    assert sorted(trajectory.get_agents('people', 1).values['opinion'].tolist()) == [1.0, 1.0, 2.0, 2.0]


def test_meetings_need_an_action():
    """Test that meetings without their own `act_on` name the missing action."""
    people = DiscussantElement('people', opinion=0.0, size=2)
    system = System(elements=[people], interactions=[MeetingsInteraction('meetings', target='people')]).prepare()
    with pytest.raises(ValidationError, match='has no action'):
        system.simulate(until=1, random_state=0)
