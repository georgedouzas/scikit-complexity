"""Active Ornstein-Uhlenbeck particle, whose propulsion velocity `u` is an Ornstein-Uhlenbeck process."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._active_particle import POINTS


class OrnsteinUhlenbeckParticleElement(Element):
    """Active Ornstein-Uhlenbeck particle, whose propulsion velocity `u` is an Ornstein-Uhlenbeck process.

    The laws are `dx = u dt + sqrt(2 D_t) dW` and `tau du = -u dt + sqrt(2 D) dW`, so the propulsion keeps its
    direction for the persistence time `tau` and the particle diffuses with `D + D_t` at long times.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        u:
            The initial propulsion velocity.

        tau:
            The persistence time.

        D:
            The active diffusion coefficient.

        D_t:
            The translational diffusion coefficient.

    Examples:
        >>> from skcomplex.models.physics.active_matter import OrnsteinUhlenbeckParticleElement
        >>> OrnsteinUhlenbeckParticleElement('swimmer').tau
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        u: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        D_t: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.u = u
        self.tau = tau
        self.D = D
        self.D_t = D_t

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the propulsion, its relaxation and noise, and the translational noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.x, s.u),
            NoiseLaw(s.x, sqrt(2 * s.D_t)),
            RateLaw(s.u, -s.u / s.tau),
            NoiseLaw(s.u, sqrt(2 * s.D) / s.tau),
        ]
