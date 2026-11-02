"""Individual of a neutral community, labelled by its species."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....core.laws import BaseLaw
from ....specs import StateSpec, VisualSpec
from ._organism_view import OrganismView


class OrganismElement(Element):
    """Individual of a neutral community, labelled by its species.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        species:
            The label of the species.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.ecology import OrganismElement
        >>> GroupElement('community', element=OrganismElement, size=10, values={'species': 0.0}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'species'}),
        *,
        species: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.species = species

    @classmethod
    def _get_view(cls) -> type[OrganismView]:
        """Return the view of an organism, `OrganismView`."""
        return OrganismView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the replacement moves every individual.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []
