"""Inhibition of every unit by the summed activity of all the other units, `-beta x_i sum_{j != i} x_j / tau_i`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import CouplingInteraction
from ....core.types import Target
from ....specs import ParameterSpec, RoleSpec


class MutualInhibitionInteraction(CouplingInteraction):
    """Inhibition of every unit by the summed activity of all the other units, `-beta x_i sum_{j != i} x_j / tau_i`.

    It is the coupling of the units to each other, as separate elements or as the members of a group, with the term
    `-beta x_i x_j / tau_i` of every other unit `j`.

    Args:
        label:
            The label of the interaction.

        units:
            The paths of the units, an element set, or the path of a group of units.

        beta:
            The strength of the inhibition.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import MutualInhibitionInteraction
        >>> MutualInhibitionInteraction('inhibition', units=['a', 'b'], beta=2.0).beta
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        units: Annotated[str | Sequence[str], RoleSpec()],
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, target=units, function=self._get_inhibition, mode='group')
        self.units = units
        self.beta = beta

    @staticmethod
    def _get_inhibition(namespace: SimpleNamespace) -> Target:
        """Return the inhibition of the unit by one other unit, `-beta x_i x_j / tau_i`."""
        member = namespace.member
        return -namespace.beta * member.x * namespace.other.x / member.tau
