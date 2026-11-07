"""Deformable self-propelled particle, whose velocity and traceless deformation tensor drive each other."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, identity_matrix, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._active_particle import POINTS


class DeformableParticleElement(Element):
    """Deformable self-propelled particle, whose velocity and traceless deformation tensor drive each other.

    The particle moves with its velocity `v`, which follows `v' = gamma v - |v|² v - a S v`, and its deformation, the
    traceless symmetric tensor `S`, follows `S' = -kappa S + b (v v - |v|² I / 2)`, the model of Ohta and Ohkuma. The
    motion stretches the particle along the velocity for `b > 0`, and the deformation turns the velocity back. Noise of
    amplitude `noise` may kick the velocity.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the particle is drawn.

        x:
            The initial position.

        v:
            The initial velocity.

        S:
            The initial deformation, given by its two independent components.

        gamma:
            The distance from the onset of self-propulsion.

        a:
            The coupling of the velocity to the deformation.

        b:
            The coupling of the deformation to the velocity.

        kappa:
            The relaxation rate of the deformation.

        noise:
            The amplitude of the noise of the velocity.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import DeformableParticleElement
        >>> DeformableParticleElement('cell').kappa
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        v: Annotated[npt.ArrayLike, StateSpec(vector=True, role='velocity')] = 0.0,
        S: Annotated[npt.ArrayLike, StateSpec(tensor='traceless')] = 0.0,
        gamma: Annotated[float, ParameterSpec()] = 1.0,
        a: Annotated[float, ParameterSpec()] = 1.0,
        b: Annotated[float, ParameterSpec()] = 1.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.S = S
        self.gamma = gamma
        self.a = a
        self.b = b
        self.kappa = kappa
        self.noise = noise

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the motion, the velocity and its noise, and the deformation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        speed = s.v.dot_product(s.v)
        stretch = s.v.outer_product(s.v) - speed / 2 * identity_matrix(SR, 2)
        return [
            RateLaw(s.x, s.v),
            RateLaw(s.v, s.gamma * s.v - speed * s.v - s.a * (s.S * s.v)),
            NoiseLaw(s.v, sqrt(2) * s.noise),
            RateLaw(s.S, -s.kappa * s.S + s.b * stretch),
        ]
