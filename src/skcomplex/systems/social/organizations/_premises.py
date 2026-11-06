"""House or brothel of the brothel movement system, whose energy bills grow with the people inside."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._premises_view import PremisesView


class PremisesElement(Element):
    """House or brothel of the brothel movement system, whose energy bills grow with the people inside.

    Every day the bills grow by `1 / (1 + exp(x0 - n))` for the `n` people in the premises' cell, and every quarter
    they are recorded and reset. In `trafficking`, maps of the bills read the counts of the women by cell every day,
    and maps in a phase of a period of 90 days record and reset them.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        brothel:
            Whether the premises are a brothel, 1, or a house, 0.

        bills:
            The bills of the current quarter.

        last_bills:
            The bills of the last full quarter.

        efficiency:
            The midpoint `x0` of the sigmoid of the bills.

        day:
            The number of days run.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.organizations import PremisesElement
        >>> group = GroupElement('premises', element=PremisesElement, size=3, values={'brothel': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('premises', 'PremisesElement', 3)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'brothel', 'color_range': (0.0, 1.0)}),
        *,
        brothel: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        bills: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        last_bills: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        efficiency: Annotated[float, StateSpec()] = 1.0,
        day: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.brothel = brothel
        self.bills = bills
        self.last_bills = last_bills
        self.efficiency = efficiency
        self.day = day

    @classmethod
    def _get_view(cls) -> type[PremisesView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return PremisesView
