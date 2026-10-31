"""Define random pairs, a grouping that draws as many random ordered pairs of agents per step as there are agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.interactions import GroupingInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import RoleSpec

PAIR = 2


class RandomPairsInteraction(GroupingInteraction):
    """Grouping that draws as many random ordered pairs of distinct agents per step as there are agents.

    The pairs are drawn independently, so an agent can meet several partners in one step, and every pair acts in random
    order through `act_on`, which a subclass overrides. One unit of time holds one encounter per agent, as in the
    random sequential encounters of mean-field opinion and convention models.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group whose agents meet in pairs.

        phase:
            The phase of the step the encounters run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import RandomPairsInteraction
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

    def _get_groups(self, model: ModelViewProtocol) -> list[list[AgentViewProtocol]]:
        """Return one random ordered pair of distinct agents per agent."""
        agents = [agent for agent in model.agents('target') if agent.alive]
        if len(agents) < PAIR:
            return []
        draws = [model.rng.choice(len(agents), size=PAIR, replace=False) for _ in agents]
        return [[agents[int(first)], agents[int(second)]] for first, second in draws]
