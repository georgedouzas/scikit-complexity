"""Following tipping elements: a fold and a Hopf normal form whose control a leading element shifts linearly."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class FoldFollowerElement(Element):
    """Stochastic fold `dy = (-y³ + y + c) dt + sigma dW`, whose control a leading element shifts.

    The fold has a lower and an upper stable branch while `|c| < 2 / (3 √3)`, and a coupling adds to its control.

    Args:
        label:
            The label of the follower.

        visual:
            How the follower is drawn, or `None` for `default_visual_`.

        y:
            The initial state.

        c:
            The control of the fold, without the coupling.

        sigma:
            The amplitude of the noise.

    Attributes:
        default_visual_:
            The visual of the follower when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.climate import FoldFollowerElement
        >>> FoldFollowerElement('follower').c
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[float, StateSpec()] = -1.0,
        c: Annotated[float, ParameterSpec()] = 0.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.c = c
        self.sigma = sigma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the fold and its noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law and the noise law of the state.
        """
        return [RateLaw(s.y, -(s.y**3) + s.y + s.c), NoiseLaw(s.y, s.sigma)]


class HopfFollowerElement(Element):
    """Stochastic Hopf normal form `dz = ((mu + i omega) z - |z|² z) dt + sigma dW`, whose `mu` a leader shifts.

    The complex amplitude `z = y + i z_im` is held as two real states. For `mu < 0` it decays to a focus at the origin,
    and for `mu > 0` it settles on a limit cycle of radius `sqrt(mu)`, and a coupling adds to `mu`.

    Args:
        label:
            The label of the follower.

        visual:
            How the follower is drawn, or `None` for `default_visual_`.

        y:
            The initial real part of the amplitude.

        z:
            The initial imaginary part of the amplitude.

        mu:
            The distance from the Hopf bifurcation, without the coupling.

        omega:
            The frequency of the oscillation.

        sigma:
            The amplitude of the noise.

    Attributes:
        default_visual_:
            The visual of the follower when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.climate import HopfFollowerElement
        >>> HopfFollowerElement('follower').omega
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[float, StateSpec()] = 0.1,
        z: Annotated[float, StateSpec()] = 0.0,
        mu: Annotated[float, ParameterSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.z = z
        self.mu = mu
        self.omega = omega
        self.sigma = sigma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Hopf normal form and its noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws and the noise laws of the two parts of the amplitude.
        """
        radius = s.y**2 + s.z**2
        return [
            RateLaw(s.y, s.mu * s.y - s.omega * s.z - radius * s.y),
            RateLaw(s.z, s.omega * s.y + s.mu * s.z - radius * s.z),
            NoiseLaw(s.y, s.sigma),
            NoiseLaw(s.z, s.sigma),
        ]
