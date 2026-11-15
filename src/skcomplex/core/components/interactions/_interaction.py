"""Define the interaction whose laws run once for the elements its roles bind."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from ._base import BaseInteraction


class Interaction(BaseInteraction):
    """An interaction whose laws run once for the elements its roles bind.

    When a role binds a group, the laws run once for every member of the group, as arrays. A subclass takes its roles
    and parameters as arguments of its `__init__`, annotated with `RoleSpec` or `ParameterSpec` inside `Annotated`, and
    returns its laws from `_get_laws`, which returns no laws by default.

    Unlike `BaseInteraction`, it fixes how the laws run: once per bound element or member, or once per edge with
    `along`. Choose it when a law reads the elements it binds, such as predation between two populations, or the two
    ends of a link, such as a contact that passes on an infection.

    It appears in many fields. Predation between two populations, a reaction between two chemical species, and a payment
    from a firm to its workers are each a subclass of it.

    With `among` and `along`, the laws run once for every edge of a network instead, and read `s.source` and
    `s.target`, the members at the two ends. A subclass lists both in its own `__init__` and passes them on, such as an
    infection along the contacts of a network, `Infection('infect', among='people', along=True, beta=0.3)`.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'predation'`.

        among:
            The group whose members sit at the ends of the edges, such as `'people'`, or `None` to bind elements by
            their roles only. A subclass annotates it as a role.

        along:
            `True` for the edges of the network the group lives on, the name of a layer of a network with several, such
            as `'contacts'`, or `None` for no edges.

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

    def __init__(self, label: str, *, among: str | None = None, along: bool | str | None = None) -> None:
        super().__init__(label)
        self.among = among
        self.along = along

    def _runs_on_edges(self) -> bool:
        """Return whether the interaction runs along edges."""
        return self.along is not None and self.along is not False

    def _get_layer(self) -> str | None:
        """Return the layer `along` names, or `None` for the graph of the network."""
        return self.along if isinstance(self.along, str) else None
