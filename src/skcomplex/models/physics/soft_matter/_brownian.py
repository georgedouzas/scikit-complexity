"""Soft matter models: Brownian particles and noise-driven transitions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, BoundaryLaw, NoiseLaw, RateLaw
from ....core.system import System
from ....engine.runtime import register_model
from ....specs import ParameterSpec, StateSpec, VisualSpec


class DoubleWellElement(Element):
    """Double well element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = -1.0,
        amplitude: Annotated[object, ParameterSpec()] = 0.1,
        frequency: Annotated[object, ParameterSpec()] = 0.01,
        D: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.12,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.amplitude = amplitude
        self.frequency = frequency
        self.D = D

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the double well element."""
        return [
            RateLaw(s.x, s.x - s.x**3 + s.amplitude * cos(2 * math.pi * s.frequency * s.t)),
            NoiseLaw(s.x, sqrt(2 * s.D)),
        ]


@register_model(
    field='physics/soft_matter',
    tags=('noise', 'sde', 'forcing', 'resonance'),
    references=('Benzi, Sutera, and Vulpiani 1981', 'Gammaitoni et al. 1998'),
    defaults={'until': 500.0, 'dt': 0.5},
)
def stochastic_resonance(
    x: float = -1.0,
    *,
    amplitude: float = 0.1,
    frequency: float = 0.01,
    D: float = 0.12,
) -> System:
    """Return a particle in a double well, `dx = (x - x^3 + A cos(2 pi f t)) dt + sqrt(2 D) dW`.

    The forcing alone is too weak to move the particle between the wells, but noise of the right intensity makes the
    hops follow the forcing.

    Args:
        x:
            The initial position.

        amplitude:
            The amplitude of the periodic forcing.

        frequency:
            The frequency of the forcing.

        D:
            The noise intensity.

    Returns:
        system:
            The model, with the position `particle.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import stochastic_resonance
        >>> stochastic_resonance(D=0.2).elements[0].D
        0.2
    """
    return System(elements=[DoubleWellElement('particle', x=x, amplitude=amplitude, frequency=frequency, D=D)])


class WallParticleElement(Element):
    """Brownian particle whose diffusion slows near a wall, with multiplicative noise of a chosen convention.

    Args:
        label:
            The label of the particle.

        convention:
            The convention of the noise, `'ito'`, `'stratonovich'`, or `'anti-ito'`.

        **values:
            The values of the declared variables, by name:

            z:
                The distance from the wall.

            D0:
                The diffusion coefficient far from the wall.

            a:
                The distance over which the wall slows diffusion.

            drift:
                The settling velocity far from the wall.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import WallParticleElement
        >>> WallParticleElement('bead').convention
        'anti-ito'
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        D0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        drift: Annotated[float, ParameterSpec()] = 0.2,
        convention: str = 'anti-ito',
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.D0 = D0
        self.a = a
        self.drift = drift
        self.convention = convention

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the settling drift, the hindered noise, and the reflection at the wall.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.z, -s.drift * s.z * abs(s.z) / (s.z**2 + s.a**2)),
            NoiseLaw(s.z, sqrt(2 * s.D0) * s.z / sqrt(s.z**2 + s.a**2), convention=self.convention),
            BoundaryLaw(s.z, abs(s.z)),
        ]


@register_model(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'multiplicative noise', 'hydrodynamics'),
    references=('Brenner 1961', 'Volpe and Wehr 2016'),
    defaults={'until': 10.0, 'dt': 0.1},
)
def brownian_near_wall(
    z: float = 1.0,
    *,
    D0: float = 1.0,
    a: float = 1.0,
    drift: float = 0.2,
    convention: str = 'anti-ito',
) -> System:
    """Return a Brownian particle whose diffusion slows near a wall, `D(z) = D0 z² / (z² + a²)`, as it settles on it.

    The diffusion is a smooth form of the hydrodynamic hindrance that vanishes at the wall, and the drift is the
    settling velocity scaled by the same mobility. Diffusion that depends on the position makes the noise
    multiplicative, and the anti-Itô convention gives the equilibrium of physical particles. The laws are symmetric
    about the wall at `z = 0`, which reflects the particle at every frame.

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

    Returns:
        system:
            The model, with the distance `particle.z`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import brownian_near_wall
        >>> brownian_near_wall(convention='ito').elements[0].convention
        'ito'
    """
    particle = WallParticleElement('particle', z=z, D0=D0, a=a, drift=drift, convention=convention)
    return System(elements=[particle])
