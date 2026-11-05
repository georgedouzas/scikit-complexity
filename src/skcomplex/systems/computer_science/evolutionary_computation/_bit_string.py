"""Agent whose genome is a string of bits, stored as the integer it spells, with the fitness of that genome."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._bit_string_view import BitStringView


class BitStringElement(Element):
    """Agent whose genome is a string of bits, stored as the integer it spells, with the fitness of that genome.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        genome:
            The initial genome, an integer whose binary digits are the bits.

        fitness:
            The initial fitness.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.computer_science.evolutionary_computation import BitStringElement
        >>> group = GroupElement('genomes', element=BitStringElement, size=4, values={'genome': [0, 1, 2, 3]})
        >>> group.label, group.element.__name__, group.size
        ('genomes', 'BitStringElement', 4)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'fitness'}),
        *,
        genome: Annotated[float, StateSpec(domain=NN)] = 0.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.genome = genome
        self.fitness = fitness

    @classmethod
    def _get_view(cls) -> type[BitStringView]:
        """Return the view of the individuals, which types their states."""
        return BitStringView
