"""Rats in one behavioural state, calm or nervous, counted by the state `n`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import CompartmentElement
from ....specs import StateSpec, VisualSpec


class RatsElement(CompartmentElement):
    """Rats in one behavioural state, calm or nervous, counted by the state `n`.

    Args:
        label:
            The label of the state.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of rats.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import RatsElement
        >>> RatsElement('calm', n=90.0).n
        90.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, n=n)
