"""Tumour cell that consumes oxygen, divides when supplied, secretes VEGF when hypoxic, and dies without oxygen."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._tumour_cell_view import TumourCellView


class TumourCellElement(Element):
    """Tumour cell that consumes oxygen, divides when supplied, secretes VEGF when hypoxic, and dies without oxygen.

    Every step the cell takes up `consumption` oxygen at its site, or what is left. Above `proliferation` it divides
    with probability `division` into a free neighbouring cell. Below `hypoxia` it secretes `secretion` VEGF, and below
    `necrosis` it dies. The system adds the uptake, the hypoxia, the secretion, and the life cycle as member
    interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        hypoxic:
            Whether the cell is hypoxic, `1`, or not, `0`.

        consumption:
            The oxygen a cell takes up per step.

        proliferation:
            The oxygen above which a cell divides.

        division:
            The probability per step of a division.

        hypoxia:
            The oxygen below which a cell secretes VEGF.

        secretion:
            The VEGF a hypoxic cell secretes per step.

        necrosis:
            The oxygen below which a cell dies.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.oncology import TumourCellElement
        >>> GroupElement('tumour', element=TumourCellElement, size=3, location=(5, 5)).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'hypoxic', 'color_range': (0.0, 1.0)}),
        *,
        hypoxic: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        consumption: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        proliferation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        division: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        hypoxia: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        secretion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        necrosis: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.hypoxic = hypoxic
        self.consumption = consumption
        self.proliferation = proliferation
        self.division = division
        self.hypoxia = hypoxia
        self.secretion = secretion
        self.necrosis = necrosis

    @classmethod
    def _get_view(cls) -> type[TumourCellView]:
        """Return the view of a tumour cell, `TumourCellView`."""
        return TumourCellView
