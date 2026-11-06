"""Define the reactor of the Brusselator, the two intermediates and the supplies of its two buffered reactants."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.specs import ParameterSpec, StateSpec, VisualSpec


class BrusselatorReactionElement(Element):
    """Reactor of the Brusselator, the concentrations `u` and `v` of its two intermediates.

    The reactions `A -> U`, `B + U -> V + D`, `2U + V -> 3U`, and `U -> E` change the concentrations, with the buffered
    reactants `A` and `B` held at the supplies `a` and `b`, which the reactions read as modifiers. The well-mixed
    `brusselator_ode` runs them in one reactor and the reaction-diffusion `brusselator` at every site of a field.

    Args:
        label:
            The label of the reactor.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial concentration of the autocatalytic intermediate.

        v:
            The initial concentration of the second intermediate.

        a:
            The constant supply of the first reactant.

        b:
            The constant supply of the second reactant.

    Examples:
        >>> from skcomplex.systems.chemistry.common.kinetics import BrusselatorReactionElement
        >>> BrusselatorReactionElement('reactor').b
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        v: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v
        self.a = a
        self.b = b
