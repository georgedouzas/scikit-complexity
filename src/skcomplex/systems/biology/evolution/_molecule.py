"""Self-replicating molecule of a prebiotic soup, with its concentration `x` and its own growth rate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class MoleculeElement(Element):
    """Self-replicating molecule of a prebiotic soup, with its concentration `x` and its own growth rate.

    Args:
        label:
            The label of the molecule or of the group of molecules.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial concentration.

        growth:
            The rate of replication without catalysis.

    Examples:
        >>> from skcomplex.systems.biology.evolution import MoleculeElement
        >>> MoleculeElement('molecule', growth=0.1).growth
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        growth: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.growth = growth
