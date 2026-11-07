"""Site of the HPP lattice gas, with one boolean channel per direction of the square lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class HPPSiteElement(Element):
    """Site of the HPP lattice gas, with one boolean channel per direction of the square lattice.

    The local rules of the gas turn every head-on pair of particles at a site by a quarter turn, and then move every
    particle to the neighbouring site along its direction. The state `density` holds the number of particles at the
    site.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n0:
            The initial occupation of the channel along the columns, towards higher column indices.

        n1:
            The initial occupation of the channel along the rows, towards higher row indices.

        n2:
            The initial occupation of the channel along the columns, towards lower column indices.

        n3:
            The initial occupation of the channel along the rows, towards lower row indices.

        density:
            The initial number of particles at the site.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.fluids import HPPSiteElement
        >>> HPPSiteElement('site').n0
        0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n0: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        n1: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        n2: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        n3: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        density: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.density = density
