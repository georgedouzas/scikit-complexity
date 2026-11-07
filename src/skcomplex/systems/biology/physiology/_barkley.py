"""Excitable medium of Barkley, a fast excitation `u` and a slow recovery `w` with a polynomial nullcline."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class BarkleyElement(Element):
    """Excitable medium of Barkley, a fast excitation `u` and a slow recovery `w` with a polynomial nullcline.

    The excitation follows `u' = u (1 - u) (u - (w + b) / a)` and the recovery `w' = epsilon (u^gamma - w)`. A diffusion
    of the excitation couples the sites of the medium. The threshold `(w + b) / a` rises with the recovery, so an
    excited site recovers before it can fire again.

    Args:
        label:
            The label of the medium.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial excitation.

        w:
            The initial recovery.

        a:
            The inverse slope of the threshold.

        b:
            The offset of the threshold.

        epsilon:
            The ratio of the time scales of the recovery and the excitation.

        gamma:
            The exponent of the excitation in the recovery, 1 for Barkley's model and 3 for the variant of Bär and
            Eiswirth with spiral breakup.

    Examples:
        >>> from skcomplex.systems.biology.physiology import BarkleyElement
        >>> BarkleyElement('medium').a
        0.6
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        w: Annotated[float, StateSpec()] = 0.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.6,
        b: Annotated[float, ParameterSpec()] = 0.05,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.02,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.w = w
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the cubic excitation and its slow recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.u, s.u * (1 - s.u) * (s.u - (s.w + s.b) / s.a)),
            RateLaw(s.w, s.epsilon * (s.u**s.gamma - s.w)),
        ]
