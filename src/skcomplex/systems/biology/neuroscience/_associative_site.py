"""Site of a dense homogeneous associative network, an activity coupled to a slow reservoir."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import where
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

UNIT = RealSet.closed(0, 1)


class AssociativeSiteElement(Element):
    """Site of a dense homogeneous associative network, an activity coupled to a slow reservoir.

    The reservoir `phi` of an active site, `x > x_c`, depletes as `dphi/dt = -gamma_minus phi`, and the reservoir of an
    inactive site recovers as `dphi/dt = gamma_plus (1 - phi) (1 - x / x_c)`. The network couples the activities.

    Args:
        label:
            The label of the site or of the group of sites.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial activity.

        phi:
            The initial reservoir.

        x_c:
            The activity above which a site counts as active.

        gamma_plus:
            The recovery rate of the reservoir.

        gamma_minus:
            The depletion rate of the reservoir.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import AssociativeSiteElement
        >>> AssociativeSiteElement('sites').x_c
        0.85
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=UNIT)] = 0.0,
        phi: Annotated[float, StateSpec(domain=UNIT)] = 1.0,
        x_c: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.85,
        gamma_plus: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        gamma_minus: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.phi = phi
        self.x_c = x_c
        self.gamma_plus = gamma_plus
        self.gamma_minus = gamma_minus

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the depletion and the recovery of the reservoir.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        recovery = s.gamma_plus * (1 - s.phi) * (1 - s.x / s.x_c)
        return [RateLaw(s.phi, where(s.x > s.x_c, -s.gamma_minus * s.phi, recovery))]
