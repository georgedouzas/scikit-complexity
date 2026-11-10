"""Toy metabolic network of Raman.

It grows on A alone.

`TOY_REACTIONS` holds the reactions of Table 8.2 of Raman (2021), with the default bounds of its Section 8.6.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from .....core.specs import DeclaredVariableSpec, VisualSpec
from ._metabolic_network import MetabolicNetworkElement
from ._reaction_spec import ReactionSpec

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


class ToyNetworkElement(MetabolicNetworkElement):
    """The toy network of Table 8.2 of Raman (2021).

    It grows on A alone.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for the bars of its program.

        objective:
            The coefficient of every reaction in the objective, or `None` for `default_objective`.

        parsimonious:
            Whether to minimise the total absolute flux after the objective.

        infeasible:
            The infeasibility policy of the program.

        name:
            The name of the network.

        reactions:
            The reactions by name.

        default_objective:
            The objective of the network when `objective` is `None`.

        variables:
            The declared states and parameters that replace the ones of the reactions, by name, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import ToyNetworkElement
        >>> ToyNetworkElement('cell').get_value('upper_v1')
        1000.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        objective: Mapping[str, float] | None = None,
        parsimonious: bool = False,
        infeasible: str = 'raise',
        name: str | None = 'ToyNetwork',
        reactions: Mapping[str, ReactionSpec] = TOY_REACTIONS,
        default_objective: Mapping[str, float] = TOY_OBJECTIVE,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
    ) -> None:
        super().__init__(
            label,
            visual,
            objective=objective,
            parsimonious=parsimonious,
            infeasible=infeasible,
            name=name,
            reactions=reactions,
            default_objective=default_objective,
            variables=variables,
        )
