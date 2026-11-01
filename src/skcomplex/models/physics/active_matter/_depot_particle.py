"""Brownian particle pumped by an internal energy depot, as in the model of Schweitzer, Ebeling, and Tilch."""

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


class DepotParticleElement(Element):
    """Brownian particle pumped by an internal energy depot, as in the model of Schweitzer, Ebeling, and Tilch.

    The depot takes energy at the rate `q`, loses it at the rate `c e`, and converts it into motion at the rate
    `d e |v|^2`, so `de/dt = q - c e - d e |v|^2` and `dv = (-gamma v + d e v) dt + sqrt(2 D) dW`. Above
    `q d = c gamma` the particle moves at the speed `sqrt(q / gamma - c / d)`.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        v:
            The initial velocity.

        e:
            The initial energy of the depot.

        gamma:
            The friction coefficient.

        q:
            The rate of energy intake.

        c:
            The rate of internal energy loss.

        d:
            The rate of conversion of depot energy into motion.

        D:
            The intensity of the velocity noise.

    Examples:
        >>> from skcomplex.models.physics.active_matter import DepotParticleElement
        >>> DepotParticleElement('walker').q
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        v: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        e: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        q: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.e = e
        self.gamma = gamma
        self.q = q
        self.c = c
        self.d = d
        self.D = D

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the motion, the friction and pumping of the velocity, its noise, and the depot balance.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.x, s.v),
            RateLaw(s.v, (-s.gamma + s.d * s.e) * s.v),
            NoiseLaw(s.v, sqrt(2 * s.D)),
            RateLaw(s.e, s.q - s.c * s.e - s.d * s.e * s.v.dot_product(s.v)),
        ]
