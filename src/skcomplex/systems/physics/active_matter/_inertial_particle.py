"""Active Brownian particle with mass and moment of inertia, whose velocity and spin relax with friction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, cos, oo, sin, sqrt, vector
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._active_particle import POINTS


class InertialActiveParticleElement(Element):
    """Active Brownian particle with mass and moment of inertia, whose velocity and spin relax with friction.

    With unit friction, the position follows `m x'' = -x' + v0 n + sqrt(2 D_t) xi` along the orientation
    `n = (cos theta, sin theta)`, and the orientation follows `J theta'' = -theta' + sqrt(2 D_r) xi_r`. The velocity
    relaxes over the time `m` and the angular velocity over the time `J`, and both vanish to give the overdamped
    active Brownian particle.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the particle is drawn.

        x:
            The initial position.

        x_d1:
            The initial velocity.

        theta:
            The initial orientation.

        theta_d1:
            The initial angular velocity.

        speed:
            The self-propulsion speed `v0`.

        mass:
            The mass `m`, in units of the friction.

        moment:
            The moment of inertia `J`, in units of the rotational friction.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import InertialActiveParticleElement
        >>> InertialActiveParticleElement('swimmer').mass
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        theta: Annotated[float, StateSpec(order=2, role='direction')] = 0.0,
        theta_d1: float = 0.0,
        speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        mass: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        moment: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        D_t: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        D_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.speed = speed
        self.mass = mass
        self.moment = moment
        self.D_t = D_t
        self.D_r = D_r

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the accelerations of the position and the orientation, and their noises.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        heading = vector(SR, [cos(s.theta), sin(s.theta)])
        return [
            RateLaw(s.x, (s.speed * heading - s.x_d1) / s.mass),
            NoiseLaw(s.x, sqrt(2 * s.D_t) / s.mass),
            RateLaw(s.theta, -s.theta_d1 / s.moment),
            NoiseLaw(s.theta, sqrt(2 * s.D_r) / s.moment),
        ]
