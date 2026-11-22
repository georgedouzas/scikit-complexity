"""Define the reactor of the Brusselator.

It holds the two intermediates and the supplies of its two buffered reactants.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.components.spaces import BaseSpace


class BrusselatorReactionElement(Element):
    """Reactor of the Brusselator, the concentrations `u` and `v` of its intermediates.

    The reactions `A -> U`, `B + U -> V + D`, `2U + V -> 3U`, and `U -> E` change the concentrations, with the buffered
    reactants `A` and `B` held at the supplies `a` and `b`, which the reactions read as modifiers. The well-mixed
    `BrusselatorODESystem` runs them in one reactor and the reaction-diffusion `BrusselatorSystem` at every site of a
    field.

    Args:
        label:
            The label of the reactor.

        u:
            The initial concentration of the autocatalytic intermediate.

        v:
            The initial concentration of the second intermediate.

        a:
            The constant supply of the first reactant.

        b:
            The constant supply of the second reactant.

        space:
            The lattice of the sites of a field, or `None` for one reactor.

    Examples:
        >>> from skcomplex.systems.chemistry.common.kinetics import BrusselatorReactionElement
        >>> BrusselatorReactionElement('reactor').b
        3.0
    """

    def __init__(
        self,
        label: str,
        *,
        u: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        v: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        a: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        b: Annotated[float, RealSet.closed_open(0, oo)] = 3.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.u = u
        self.v = v
        self.a = a
        self.b = b
