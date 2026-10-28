"""Define pair interactions, whose laws run for every pair of members closer than a cutoff."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ...specs import ParameterSpec, RoleSpec
from ..base import BasePart
from ._base import BaseInteraction


class PairInteraction(BaseInteraction):
    """Base class of interactions whose laws run once for every pair of members closer than a cutoff.

    The laws are written for one pair and read `namespace.first` and `namespace.second`, the members of the pair, and
    the parameters. A subclass adds its own roles and parameters to the ones of every pair interaction.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        first:
            The path of the group whose members are the first of every pair.

        second:
            The path of the group whose members are the second of every pair, which may be the first group.

        cutoff:
            The largest distance between the members of a pair.

        position:
            The name of the vector state that holds the positions of the members.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.interactions import PairInteraction
        >>> from skcomplex.specs import ParameterSpec
        >>> class RepulsionInteraction(PairInteraction):
        ...     def __init__(
        ...         self, label, first, second, cutoff=1.0, position='x', k: Annotated[float, ParameterSpec()] = 1.0
        ...     ):
        ...         super().__init__(label, first, second, cutoff=cutoff, position=position)
        ...         self.k = k
        >>> RepulsionInteraction('repulsion', 'particles', 'particles').cutoff
        1.0
    """

    def __init__(
        self,
        label: str,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        cutoff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        position: str = 'x',
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.cutoff = cutoff
        self.position = position

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the laws of the interaction for one pair, none by default.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return []
