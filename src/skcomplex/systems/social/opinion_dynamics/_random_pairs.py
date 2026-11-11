"""Define random pairs, a grouping of random ordered pairs of agents.

It draws as many pairs per step as there are agents.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.interactions import GroupingInteraction
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.specs import RoleSpec

PAIR = 2


class RandomPairsInteraction(GroupingInteraction):
    """Grouping that draws random ordered pairs of distinct agents.

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
        >>> RandomPairsInteraction('meetings', target='people').groups
        'pairs'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, groups='pairs', phase=phase)

    def _get_groups(self, system: SystemViewProtocol) -> list[list[AgentViewProtocol]]:
        """Return one random ordered pair of distinct agents per agent."""
        agents = [agent for agent in system.get_agents('target') if agent.alive]
        if len(agents) < PAIR:
            return []
        draws = [system.rng.choice(len(agents), size=PAIR, replace=False) for _ in agents]
        return [[agents[int(first)], agents[int(second)]] for first, second in draws]
