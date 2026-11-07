"""Inverse radius of a planet along its orbit angle under the relativistic correction of Newtonian gravity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class RelativisticOrbitElement(Element):
    """Inverse radius of a planet along its orbit angle under the relativistic correction of Newtonian gravity.

    The inverse radius `u = 1 / r` follows Einstein's orbit equation `u'' + u = alpha + epsilon u²`, whose independent
    variable is the polar angle of the planet, which the simulation time stands for. For `epsilon = 0` it is the
    Newtonian conic `u = alpha (1 + e cos θ)`, and the quadratic term makes the perihelion advance every revolution.

    Args:
        label:
            The label of the planet.

        visual:
            How the planet is drawn, or `None` for `default_visual_`.

        u:
            The initial inverse radius.

        u_d1:
            The initial derivative of the inverse radius along the angle.

        alpha:
            The Newtonian inverse semi-latus rectum `G M / h²`, with `h` the angular momentum per unit mass.

        epsilon:
            The strength of the relativistic correction, `3 G M / c²`.

    Attributes:
        default_visual_:
            The visual of the planet when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import RelativisticOrbitElement
        >>> RelativisticOrbitElement('planet').epsilon
        0.05
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec(order=2)] = 1.5,
        u_d1: float = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.u_d1 = u_d1
        self.alpha = alpha
        self.epsilon = epsilon

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return Einstein's orbit equation.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of the second derivative of `u` along the angle.
        """
        s = namespace
        return [RateLaw(s.u, s.alpha + s.epsilon * s.u**2 - s.u)]
