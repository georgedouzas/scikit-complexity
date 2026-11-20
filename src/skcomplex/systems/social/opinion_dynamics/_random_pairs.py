"""Define meetings, the groups of agents that act together at every step.

Random pairs draw as many pairs per step as there are agents, and neighbourhoods group every agent with the agents in
the cells or nodes around it.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from ....core.components.base import BaseComponent
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.solvers import SystemView
from ....core.errors import ValidationError
from ....core.types import Role

PAIR = 2


class MeetingsInteraction(Interaction):
    """Rule of the system that lets groups of agents act together at every step.

    Every step the groups that `_get_groups` returns act in random order through `act_on`, which a subclass overrides,
    such as the pairs of a naming game or the neighbourhoods of Axelrod's cultures.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group whose agents meet.

        phase:
            The phase of the step the meetings run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import MeetingsInteraction
        >>> MeetingsInteraction('meetings', target='people').target
        'people'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role, phase: str | None = None) -> None:
        super().__init__(label)
        self.target = target
        self.phase = phase

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the encounters of every step as the rule of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context, self._run_meetings, phase=self.phase)]

    def _run_meetings(self, system: SystemView) -> None:
        """Act on every group of one step in random order."""
        groups = self._get_groups(system)
        for index in system.rng.permutation(len(groups)):
            self.act_on(groups[int(index)], system)

    def _get_groups(self, system: SystemView) -> list[list[AgentViewProtocol]]:
        """Return the groups of one step, every agent alone unless a subclass says otherwise."""
        return [[agent] for agent in system.get_agents('target') if agent.alive]

    def act_on(self, members: list[AgentViewProtocol], system: SystemView) -> None:
        """Act on the members of one group, which a subclass does.

        Args:
            members:
                The agents of the group.

            system:
                The system view of the rule.

        Raises:
            ValidationError:
                The class does not override `act_on`.
        """
        del members, system
        error_msg = f'The interaction {self.label!r} has no action. Override `act_on` in a subclass.'
        raise ValidationError(error_msg)


class RandomPairsInteraction(MeetingsInteraction):
    """Rule of the system that draws random ordered pairs of distinct agents.

    It draws as many pairs per step as there are agents. The pairs are drawn independently, so an agent can meet several
    partners in one step, and every pair acts in random order through `act_on`, which a subclass overrides. One unit of
    time holds one encounter per agent, as in the random sequential encounters of mean-field opinion and convention
    systems.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group whose agents meet in pairs.

        phase:
            The phase of the step the encounters run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import RandomPairsInteraction
        >>> RandomPairsInteraction('meetings', target='people').target
        'people'
    """

    def _get_groups(self, system: SystemView) -> list[list[AgentViewProtocol]]:
        """Return one random ordered pair of distinct agents per agent."""
        agents = [agent for agent in system.get_agents('target') if agent.alive]
        if len(agents) < PAIR:
            return []
        draws = [system.rng.choice(len(agents), size=PAIR, replace=False) for _ in agents]
        return [[agents[int(first)], agents[int(second)]] for first, second in draws]


class NeighborhoodsInteraction(MeetingsInteraction):
    """Rule of the system that groups every agent with the agents in the cells or nodes around it.

    Every step the groups act in random order through `act_on`, which a subclass overrides, with the agent first and its
    neighbours after it, as in the cultural influence of Axelrod or the persuasion of social impact.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group whose agents meet their neighbours.

        phase:
            The phase of the step the encounters run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import NeighborhoodsInteraction
        >>> NeighborhoodsInteraction('meetings', target='villages').target
        'villages'
    """

    def _get_groups(self, system: SystemView) -> list[list[AgentViewProtocol]]:
        """Return every agent followed by the agents in the cells or nodes around it."""
        agents = [agent for agent in system.get_agents('target') if agent.alive]
        located: dict[object, list[AgentViewProtocol]] = {}
        for agent in agents:
            located.setdefault(agent.location_key, []).append(agent)
        return [
            [agent, *(other for cell in agent.get_neighbor_cells() for other in located.get(cell, []))]
            for agent in agents
        ]
