"""Site of a neural field, the mean membrane potential of the neurons at one position."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class NeuralFieldSiteElement(Element):
    """Site of a neural field, the mean membrane potential of the neurons at one position.

    The site holds the potential `u`, its time constant `tau`, the resting level `h`, and the amplitude `A` and the
    range `sigma` of the lateral-inhibition kernel `w(x) = A (1 - |x| / sigma) exp(-|x| / sigma)`, which excites near
    neighbours and inhibits distant ones.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        u:
            The initial membrane potential.

        tau:
            The time constant of the potential.

        h:
            The resting level.

        A:
            The amplitude of the kernel.

        sigma:
            The range of the kernel.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import NeuralFieldSiteElement
        >>> NeuralFieldSiteElement('site').h
        -0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        h: Annotated[float, ParameterSpec()] = -0.2,
        A: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.tau = tau
        self.h = h
        self.A = A
        self.sigma = sigma
