"""Random binary encounters of the Deffuant system, as many per step as there are agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ._discussant_view import DiscussantView
from ._random_pairs import RandomPairsInteraction


class EncountersInteraction(RandomPairsInteraction):
    """Random binary encounters of the Deffuant system, as many per step as there are agents.

    In every encounter two random agents whose opinions differ by less than the confidence bound move towards each
    other by the convergence fraction of their difference.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of discussants.

        phase:
            The phase of the step the encounters run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import EncountersInteraction
        >>> EncountersInteraction('meetings', target='people').target
        'people'
    """

    def act_on(self, members: list[AgentViewProtocol], system: SystemViewProtocol) -> None:
        """Move the opinions of a pair towards each other when they differ by less than the confidence bound.

        Args:
            members:
                The two discussants of the encounter.

            system:
                The system view of the rule.
        """
        first, second = [member for member in members if isinstance(member, DiscussantView)]
        difference = second.opinion - first.opinion
        if abs(difference) < first.confidence:
            first.opinion += first.convergence * difference
            second.opinion -= first.convergence * difference
