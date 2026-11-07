"""Species of the competitive Jain-Krishna model, a relative population and the signed effects of all species on it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._competing_species_view import CompetingSpeciesView


class CompetingSpeciesElement(Element):
    """Species of the competitive Jain-Krishna model, a relative population and the signed effects of all species on it.

    The links are the row of the interaction matrix of the species, `M_ij` for every species `j`, between -1 and 1, with
    the degradation `M_ii` between -1 and 0 on the diagonal.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The relative population.

        links:
            The effects of every species on this one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CompetingSpeciesElement
        >>> CompetingSpeciesElement('species', x=0.1).x
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        links: Annotated[object, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.links = links

    @classmethod
    def _get_view(cls) -> type[CompetingSpeciesView]:
        """Return the view of a species, `CompetingSpeciesView`."""
        return CompetingSpeciesView
