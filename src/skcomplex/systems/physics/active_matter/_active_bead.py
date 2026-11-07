"""Bead of an active polymer, an active Brownian particle bonded to its neighbours along a chain by springs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo, pi, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.components.laws.utils import laplacian
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ActiveBeadElement(Element):
    """Bead of an active polymer, an active Brownian particle bonded to its neighbours along a chain by springs.

    The bead at `(x, y)` swims at the speed `v0` along its orientation `theta`, which diffuses with `D_r`, its position
    diffuses with `D_t`, and harmonic springs of stiffness `k` and zero rest length pull it towards its neighbours
    along the chain, `k (r_{i+1} + r_{i-1} - 2 r_i)`, the Rouse chain. The beads at the ends have one neighbour.

    Args:
        label:
            The label of the bead or the chain.

        visual:
            How the bead is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        theta:
            The initial orientation.

        v0:
            The self-propulsion speed.

        k:
            The stiffness of the bonds.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

    Attributes:
        default_visual_:
            The visual of the bead when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveBeadElement
        >>> ActiveBeadElement('bead').k
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        v0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        D_t: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        D_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.theta = theta
        self.v0 = v0
        self.k = k
        self.D_t = D_t
        self.D_r = D_r

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the propulsion and the bonds of the bead, and the noises of its position and orientation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.x, s.v0 * cos(s.theta) + s.k * laplacian(s.x)),
            RateLaw(s.y, s.v0 * sin(s.theta) + s.k * laplacian(s.y)),
            NoiseLaw(s.x, sqrt(2 * s.D_t)),
            NoiseLaw(s.y, sqrt(2 * s.D_t)),
            NoiseLaw(s.theta, sqrt(2 * s.D_r)),
        ]
