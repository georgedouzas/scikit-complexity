"""Soft matter systems.

The module covers Brownian particles and noise-driven transitions.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, GradientInteraction, RateInteraction
from ....core.components.laws import BaseLaw, BoundaryLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.particles import DoubleWellElement


class WallParticleElement(Element):
    """Brownian particle at the distance `z` from a wall.

    The wall reflects the particle, and its diffusion slows near the wall.

    Args:
        label:
            The label of the particle.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        z:
            The distance from the wall.

        D0:
            The diffusion coefficient far from the wall.

        a:
            The distance over which the wall slows diffusion.

        drift:
            The settling velocity far from the wall.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import WallParticleElement
        >>> WallParticleElement('bead').a
        1.0
    """

    law_classes = (BoundaryLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        D0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        drift: Annotated[float, ParameterSpec()] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.D0 = D0
        self.a = a
        self.drift = drift

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the reflection at the wall.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [BoundaryLaw(s.z, abs(s.z))]


def _get_double_well_drive(s: SimpleNamespace) -> Expression:
    """Return the periodic forcing `A cos(2 pi f t)` of a particle in a double well."""
    return s.target.amplitude * cos(2 * math.pi * s.target.frequency * s.t)


class StochasticResonanceSystem(BaseSystem):
    """A particle in a double well.

    The particle follows `dx = (x - x^3 + A cos(2 pi f t)) dt + sqrt(2 D) dW`.

    The forcing alone is too weak to move the particle between the wells, but noise of the right intensity makes the
    hops follow the forcing. The system has the position `particle.x`.

    Args:
        x:
            The initial position.

        amplitude:
            The amplitude of the periodic forcing.

        frequency:
            The frequency of the forcing.

        D:
            The noise intensity.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import StochasticResonanceSystem
        >>> StochasticResonanceSystem(D=0.2).prepare().system_.elements[0].D
        0.2
    """

    field = 'physics/soft_matter'
    topics = ('noise', 'sde', 'forcing', 'resonance')
    references = ('Benzi, Sutera, and Vulpiani 1981', 'Gammaitoni et al. 1998')
    until = 500.0
    dt = 0.5

    def __init__(
        self,
        x: float = -1.0,
        *,
        amplitude: float = 0.1,
        frequency: float = 0.01,
        D: float = 0.12,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.amplitude = amplitude
        self.frequency = frequency
        self.D = D

    @staticmethod
    def _get_double_well_interactions(target: str) -> list[BaseInteraction]:
        """Return the interactions of particles in a double well.

        They are the gradient of the well and the noisy forcing.
        """
        return [
            GradientInteraction('well', target=target, potential=lambda s: s.target.x**4 / 4 - s.target.x**2 / 2),
            RateInteraction(
                'drive',
                target=target,
                rate=_get_double_well_drive,
                noise=lambda s: sqrt(2 * s.target.D),
            ),
        ]

    def _build_system(self) -> CompositeSystem:
        """Return a particle in a double well.

        The particle follows `dx = (x - x^3 + A cos(2 pi f t)) dt + sqrt(2 D) dW`.
        """
        return CompositeSystem(
            self.label,
            elements=[
                DoubleWellElement('particle', x=self.x, amplitude=self.amplitude, frequency=self.frequency, D=self.D),
            ],
            interactions=self._get_double_well_interactions('particle'),
        )


class BrownianNearWallSystem(BaseSystem):
    """A Brownian particle that settles on a wall.

    Its diffusion slows near the wall, `D(z) = D0 z² / (z² + a²)`.

    The diffusion is a smooth form of the hydrodynamic hindrance that vanishes at the wall, and the drift is the
    settling velocity scaled by the same mobility. Diffusion that depends on the position makes the noise
    multiplicative, and the anti-Itô convention gives the equilibrium of physical particles. The laws are symmetric
    about the wall at `z = 0`, which reflects the particle at every frame. The system has the distance `particle.z`.

    Args:
        z:
            The initial distance from the wall.

        D0:
            The diffusion coefficient far from the wall.

        a:
            The distance over which the wall slows diffusion, such as the particle radius.

        drift:
            The settling velocity far from the wall.

        convention:
            The convention of the noise, `'ito'`, `'stratonovich'`, or `'anti-ito'`.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import BrownianNearWallSystem
        >>> BrownianNearWallSystem(convention='ito').prepare().system_.interactions[0].convention
        'ito'
    """

    field = 'physics/soft_matter'
    topics = ('brownian motion', 'sde', 'multiplicative noise', 'hydrodynamics')
    references = ('Brenner 1961', 'Volpe and Wehr 2016')
    until = 10.0
    dt = 0.1

    def __init__(
        self,
        z: float = 1.0,
        *,
        D0: float = 1.0,
        a: float = 1.0,
        drift: float = 0.2,
        convention: str = 'anti-ito',
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.z = z
        self.D0 = D0
        self.a = a
        self.drift = drift
        self.convention = convention

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        mobility = RateInteraction(
            'mobility',
            target='particle',
            state='z',
            rate=lambda s: -s.target.drift * s.target.z * abs(s.target.z) / (s.target.z**2 + s.target.a**2),
            noise=lambda s: sqrt(2 * s.target.D0) * s.target.z / sqrt(s.target.z**2 + s.target.a**2),
            convention=self.convention,
        )
        particle = WallParticleElement('particle', z=self.z, D0=self.D0, a=self.a, drift=self.drift)
        return CompositeSystem(self.label, elements=[particle], interactions=[mobility])
