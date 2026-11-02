"""Deaths of random individuals at unit rate, each replaced by the offspring of another, a new species at rate `nu`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.real_set import RealSet

from ....core.interactions import EvolutionInteraction
from ....specs import ParameterSpec, RoleSpec


class NeutralReplacementInteraction(EvolutionInteraction):
    """Deaths of random individuals at unit rate, each replaced by the offspring of another, a new species at rate `nu`.

    One unit of time holds `N` deaths. The offspring belongs to the species of a parent drawn among the individuals, or
    with the probability `mutation` to a new species, labelled one above the largest label present. It is the neutral
    Moran replacement of `EvolutionInteraction` with mutation to new labels.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of individuals.

        mutation:
            The probability `nu` that an offspring founds a new species.

    Examples:
        >>> from skcomplex.models.biology.ecology import NeutralReplacementInteraction
        >>> NeutralReplacementInteraction('replacement', target='community', mutation=0.01).check().change
        'new'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        mutation: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.01,
    ) -> None:
        super().__init__(
            label,
            target=target,
            strategy='species',
            fitness=None,
            selection='proportional',
            replacement='moran',
            mutation=mutation,
            alphabet=None,
            change='new',
        )
