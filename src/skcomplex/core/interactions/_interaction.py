"""Define interactions whose laws run for the elements or members their roles bind."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from ..base import BasePart
from ._base import BaseInteraction


class Interaction(BaseInteraction):
    """Base class of interactions whose laws run once for the elements, or for every member of the groups, they bind.

    A subclass takes its roles and parameters as arguments of its own `__init__`, annotating each with a `RoleSpec` or
    a `ParameterSpec` inside `Annotated`, and returns its laws from `_get_laws`, which returns no laws by default.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.interactions import Interaction
        >>> from skcomplex.specs import ParameterSpec, RoleSpec
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

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the laws of the interaction for a namespace of symbols, none by default.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return []
