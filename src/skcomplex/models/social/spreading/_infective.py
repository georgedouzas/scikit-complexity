"""Node of a contagion, susceptible at 0 and infected at 1, that recovers at a rate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, JumpLaw
from ....core.laws.utils import where
from ....specs import ParameterSpec, StateSpec, VisualSpec


class InfectiveElement(Element):
    """Node of a contagion, susceptible at 0 and infected at 1, that recovers at a rate.

    Args:
        label:
            The label of the node or the group.

        visual:
            How the node is drawn, or `None` for `default_visual_`.

        infected:
            The initial state, 0 or 1.

        recovery:
            The rate at which an infected node recovers.

    Examples:
        >>> from skcomplex.models.social.spreading import InfectiveElement
        >>> InfectiveElement('node').recovery
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        infected: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        recovery: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.infected = infected
        self.recovery = recovery

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the recovery of an infected node."""
        return [JumpLaw(s.recovery * where(s.infected == 1, 1, 0), {s.infected: -1})]
