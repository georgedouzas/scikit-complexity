"""Relaxation oscillator of Terman and Wang, the dynamics of one site of a LEGION lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, tanh
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

SILENT = -2.0


class TermanWangElement(Element):
    """Relaxation oscillator of Terman and Wang, the dynamics of one site of a LEGION lattice.

    The fast variable follows `x' = 3x - x³ + 2 - y + I` and the slow one `y' = epsilon (gamma (1 + tanh(x / beta)) -
    y)`. A positive input `I` makes the oscillator oscillate and a negative one makes it excitable. The parameters `m`,
    `W`, `theta_x`, and `kappa` belong to the coupling of a lattice of oscillators, a `CouplingInteraction` that adds
    `W m (sum_k m_k H(x_k - theta_x))` over the four nearest sites, where `H` is a smooth step of width `kappa` and `m`
    marks the stimulated sites, so only stimulated neighbours excite each other.

    Args:
        label:
            The label of the oscillator or the lattice field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial fast variable.

        y:
            The initial slow variable.

        I:
            The external input.

        m:
            1 for a stimulated site and 0 otherwise.

        W:
            The weight of each excitatory link into the site.

        epsilon:
            The time scale of the slow variable.

        gamma:
            The height of the slow nullcline.

        beta:
            The width of the slow nullcline's step.

        theta_x:
            The threshold of the active phase.

        kappa:
            The width of the smooth steps.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import TermanWangElement
        >>> TermanWangElement('oscillator').gamma
        6.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = SILENT,
        y: Annotated[float, StateSpec()] = 0.0,
        I: Annotated[float, ParameterSpec()] = 0.2,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        W: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 6.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.02,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 6.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        theta_x: Annotated[float, ParameterSpec()] = -0.5,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.I = I
        self.m = m
        self.W = W
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.theta_x = theta_x
        self.kappa = kappa

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the fast variable and the slow recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.x, 3 * s.x - s.x**3 + 2 - s.y + s.I),
            RateLaw(s.y, s.epsilon * (s.gamma * (1 + tanh(s.x / s.beta)) - s.y)),
        ]
