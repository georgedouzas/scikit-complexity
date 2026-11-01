"""Density of security agents at one efficacy class, with the share of new recruits that join the class."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class SecurityAgentsElement(Element):
    """Density of security agents at one efficacy class, with the share of new recruits that join the class.

    Args:
        label:
            The label of the distribution.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        f:
            The initial density.

        share:
            The share of new recruits that join the efficacy class.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.crowds import SecurityAgentsElement
        >>> GroupElement('security', element=SecurityAgentsElement, size=3, values={'share': [0.5, 0.3, 0.2]}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        f: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        share: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.f = f
        self.share = share
