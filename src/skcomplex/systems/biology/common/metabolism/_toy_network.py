"""Toy metabolic network of Raman.

It grows on A alone.

`TOY_REACTIONS` holds the reactions of Table 8.2 of Raman (2021), with the default bounds of its Section 8.6.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from .....core.specs import Role
from ._flux_balance import FluxBalanceInteraction
from ._reaction_spec import ReactionName, ReactionSpec

TOY_REACTIONS = {
    'v1': ReactionSpec({'A': -1.0, 'ATP': 1.0, 'B': 1.0}),
    'v2': ReactionSpec({'A': -1.0, 'C': 1.0}),
    'v3': ReactionSpec({'C': -1.0, 'ATP': 1.0, 'B': 1.0}),
    'v4': ReactionSpec({'B': -1.0, 'D': 2.0}, lower=-1000.0),
    'v5': ReactionSpec({'B': -1.0, 'F': 1.0}),
    'v6': ReactionSpec({'D': -1.0, 'E': 1.0}, lower=-1000.0),
    'v7': ReactionSpec({'D': -1.0, 'ATP': -1.0, 'G': 1.0}),
    'b1': ReactionSpec({'A': 1.0}),
    'b2': ReactionSpec({'E': -1.0}, lower=-1000.0),
    'vbio': ReactionSpec({'F': -1.0, 'G': -2.0, 'ATP': -10.0}),
}
TOY_OBJECTIVE = {'vbio': 1.0}


class ToyFluxBalanceInteraction(FluxBalanceInteraction):
    """Flux balance of the toy network of Table 8.2 of Raman (2021), which grows on A alone.

    Args:
        label:
            The label of the interaction.

        cell:
            The path of the cell whose values hold the fluxes, the bounds, and the switches.

        reactions:
            The reactions by name.

        objective:
            The coefficient of every reaction in the objective.

        parsimonious:
            Whether to minimise the total absolute flux after the objective.

        infeasible:
            What a run does when the balances and bounds cannot all hold.

        name:
            The name of the network.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import ToyFluxBalanceInteraction
        >>> ToyFluxBalanceInteraction('fba').get_values()['upper_v1']
        1000.0
    """

    def __init__(
        self,
        label: str = 'fba',
        *,
        cell: Role = 'cell',
        reactions: Mapping[ReactionName, ReactionSpec] = TOY_REACTIONS,
        objective: Mapping[ReactionName, float] = TOY_OBJECTIVE,
        parsimonious: bool = False,
        infeasible: str = 'raise',
        name: str | None = 'ToyNetwork',
    ) -> None:
        super().__init__(
            label,
            cell=cell,
            reactions=reactions,
            objective=objective,
            parsimonious=parsimonious,
            infeasible=infeasible,
            name=name,
        )
