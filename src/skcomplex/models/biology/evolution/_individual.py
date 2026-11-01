"""Individual of a finite population, with an integer trait or genome."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._individual_view import IndividualView


class IndividualElement(Element):
    """Individual of a finite population, with an integer trait or genome.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        trait:
            The trait, a type index or the bits of a genome.

        fitness:
            The fitness of the trait.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.evolution import IndividualElement
        >>> GroupElement('population', element=IndividualElement, size=10, values={'trait': 0.0}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'trait'}),
        *,
        trait: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.trait = trait
        self.fitness = fitness

    @classmethod
    def _get_view(cls) -> type[IndividualView]:
        """Return the view of an individual, `IndividualView`."""
        return IndividualView
