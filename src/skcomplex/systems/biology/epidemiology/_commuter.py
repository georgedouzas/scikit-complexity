"""Person of a synthetic city who sleeps at home and spends the day at a workplace, with an SIR health state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._commuter_view import CommuterView


class CommuterElement(Element):
    """Person of a synthetic city who sleeps at home and spends the day at a workplace, with an SIR health state.

    The home and the workplace are the locations the person visits, the edges of the person in the bipartite graph of
    people and locations. Health is 0 susceptible, 1 infected, or 2 recovered.

    Args:
        label:
            The label of the person or of the group of people.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        home:
            The location of the home.

        work:
            The location of the workplace or school.

        health:
            The health state.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import CommuterElement
        >>> CommuterElement('person', home=3.0, work=12.0).work
        12.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 2.0)}),
        *,
        home: Annotated[float, StateSpec(domain=NN)] = 0.0,
        work: Annotated[float, StateSpec(domain=NN)] = 0.0,
        health: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.home = home
        self.work = work
        self.health = health

    @classmethod
    def _get_view(cls) -> type[CommuterView]:
        """Return the view of a commuter, `CommuterView`."""
        return CommuterView
