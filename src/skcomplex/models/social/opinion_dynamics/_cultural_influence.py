"""Cultural influence of Axelrod's model, where every village in turn may copy a trait of a random neighbour."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np

from ....core.interactions import GroupingInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import RoleSpec
from ....types import IntArray
from ._village_view import VillageView


def cultural_traits(culture: float, features: int, traits: int) -> IntArray:
    """Return the traits of a culture vector stored as one integer, its digits in base `q`.

    Args:
        culture:
            The integer code of the culture vector.

        features:
            The number of features `F`.

        traits:
            The number of traits `q` of every feature.

    Returns:
        traits:
            The trait of every feature, the first feature first.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import cultural_traits
        >>> cultural_traits(7, 3, 2).tolist()
        [1, 1, 1]
    """
    return (int(culture) // traits ** np.arange(features, dtype=np.int64)) % traits


class CulturalInfluenceInteraction(GroupingInteraction):
    """Cultural influence of Axelrod's model, where every village in turn may copy a trait of a random neighbour.

    Every step every village, in random order, picks a random neighbour and, with probability equal to the fraction of
    features they share, copies the trait of one feature on which they differ.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of villages.

        phase:
            The phase of the step the influence runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import CulturalInfluenceInteraction
        >>> CulturalInfluenceInteraction('influence', target='villages').groups
        'neighborhood'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, groups='neighborhood', phase=phase)

    def act_on(self, members: list[AgentViewProtocol], model: ModelViewProtocol) -> None:
        """Copy one differing trait of a random neighbour with probability equal to the cultural overlap.

        Args:
            members:
                The village, followed by its neighbours.

            model:
                The model view of the rule.
        """
        agent, *neighbors = [member for member in members if isinstance(member, VillageView)]
        if not neighbors:
            return
        neighbor = neighbors[int(model.rng.integers(len(neighbors)))]
        features, traits = int(agent.features), int(agent.traits)
        own = cultural_traits(agent.culture, features, traits)
        other = cultural_traits(neighbor.culture, features, traits)
        different = np.flatnonzero(own != other)
        if len(different) and model.rng.random() < 1 - len(different) / features:
            feature = int(different[int(model.rng.integers(len(different)))])
            agent.culture += float((int(other[feature]) - int(own[feature])) * traits**feature)
