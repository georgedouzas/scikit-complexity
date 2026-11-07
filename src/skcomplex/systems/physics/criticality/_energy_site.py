"""Site of the Zhang sandpile, which stores a continuous energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class EnergySiteElement(Element):
    """Site of the Zhang sandpile, which stores a continuous energy.

    A cascade loads the energy in random packets and resets the sites above the threshold.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        energy:
            The initial energy.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.criticality import EnergySiteElement
        >>> EnergySiteElement('site').energy
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        energy: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.energy = energy
