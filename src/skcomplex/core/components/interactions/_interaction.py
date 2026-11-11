"""Define the interaction whose laws run once for the elements its roles bind."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from ..base import BaseComponent
from ._base import BaseInteraction


class Interaction(BaseInteraction):
    """An interaction whose laws run once for the elements its roles bind.

    When a role binds a group, the laws run once for every member of the group, as arrays. A subclass takes its roles
    and parameters as arguments of its `__init__`, annotated with `RoleSpec` or `ParameterSpec` inside `Annotated`, and
    returns its laws from `_get_laws`, which returns no laws by default.

    Unlike `BaseInteraction`, it fixes how the laws run: once per bound element or member, with no edges or pairs.
    Choose it when a law reads the elements it binds directly, such as predation between two populations, and choose
    `NeighborInteraction` or `PairInteraction` when it runs along edges or between close members.

    It appears in many fields. Predation between two populations, a reaction between two chemical species, and a payment
    from a firm to its workers are each a subclass of it.

    Args:
        label:
            The name of the interaction, unique among the parts of its system, such as `'predation'`.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.specs import ParameterSpec, RoleSpec
        >>> from sage.all__sagemath_symbolics import oo
        >>> from sage.sets.real_set import RealSet
        >>> class PredationInteraction(Interaction):
        ...     def __init__(
        ...         self,
        ...         label,
        ...         predator: Annotated[str, RoleSpec()],
        ...         prey: Annotated[str, RoleSpec()],
        ...         a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        ...     ):
        ...         super().__init__(label)
        ...         self.predator = predator
        ...         self.prey = prey
        ...         self.a = a
        >>> PredationInteraction('predation', predator='foxes', prey='rabbits')
        PredationInteraction(label='predation', predator='foxes', prey='rabbits')
    """

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the laws of the interaction for a namespace of symbols, none by default.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return []
