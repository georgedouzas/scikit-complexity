"""Site of a lattice of particles, holding the number `n` of particles or quanta on it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class ParticleSiteElement(Element):
    """Site of a lattice of particles, holding the number `n` of particles or quanta on it.

    Args:
        label:
            The label of the site or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of particles.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ParticleSiteElement
        >>> ParticleSiteElement('site', n=3.0).n
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
