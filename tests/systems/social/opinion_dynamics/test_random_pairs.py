"""Test the `_random_pairs` module of `skcomplex.systems.social.opinion_dynamics`."""

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.social.opinion_dynamics import DiscussantElement, RandomPairsInteraction


class CountingPairsInteraction(RandomPairsInteraction):
    """Random pairs that count the encounters of every agent in its opinion."""

    def act_on(self, members, system):
        """Count one encounter for every member of a pair of distinct agents."""
        assert len({member.id for member in members}) == len(members) == 2
        for member in members:
            member.opinion += 1.0


def test_random_pairs_draw_one_pair_per_agent():
    """Test that a step draws as many pairs of distinct agents as there are agents."""
    people = GroupElement('people', element=DiscussantElement, size=10, values={'opinion': 0.0})
    system = CompositeSystem(elements=[people], interactions=[CountingPairsInteraction('meetings', target='people')])
    trajectory = simulate(system, until=3, random_state=0)
    assert trajectory.get_agents('people', 3).values['opinion'].sum() == 3 * 2 * 10
