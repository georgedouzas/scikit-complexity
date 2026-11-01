"""Brownian particle on the real line held by a restoring force of stiffness `k` that weakens beyond `ell`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class WeakeningParticleElement(Element):
    """Brownian particle on the real line held by a restoring force of stiffness `k` that weakens beyond `ell`.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        k:
            The stiffness of the force near the origin.

        ell:
            The distance where the force starts to weaken.

        D:
            The diffusion coefficient.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import WeakeningParticleElement
        >>> WeakeningParticleElement('particle', D=0.5).D
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        ell: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.25,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.k = k
        self.ell = ell
        self.D = D
