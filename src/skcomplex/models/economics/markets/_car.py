"""Car at the wash, waiting for or receiving a wash or a valet."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....core.laws import BaseLaw
from ....specs import StateSpec, VisualSpec
from ._car_view import CarView

WASH = 0


class CarElement(Element):
    """Car at the wash, waiting for or receiving a wash or a valet.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        stage:
            0 for the wash and 1 for the valet.

        work:
            The person-seconds of work left at the current stage.

        crew:
            The number of workers on the car in the current second.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import CarElement
        >>> GroupElement('cars', element=CarElement, size=1).size
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        stage: Annotated[float, StateSpec(domain=NN)] = WASH,
        work: Annotated[float, StateSpec()] = 525.0,
        crew: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.stage = stage
        self.work = work
        self.crew = crew

    @classmethod
    def _get_view(cls) -> type[CarView]:
        """Return the view of a car, with its stage, work, and crew."""
        return CarView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the workers and the till act on the cars.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []
