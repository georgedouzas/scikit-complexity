"""Soft matter systems.

The module covers Brownian particles and noise-driven transitions.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw, UpdateLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.particles import DoubleWellElement, DoubleWellInteraction


class WallParticleElement(Element):
    """Brownian particle at the distance `z` from a wall.

    Args:
        label:
            The label of the particle.

        z:
            The distance from the wall.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import WallParticleElement
        >>> WallParticleElement('bead', z=2.0).z
        2.0
    """

    def __init__(self, label: str, *, z: Annotated[float, RealSet.closed_open(0, oo)] = 1.0) -> None:
        super().__init__(label)
        self.z = z


class WallMobilityInteraction(Interaction):
    """Settling of a Brownian particle on a wall that reflects it and slows its diffusion.

    The diffusion is `D(z) = D0 z² / (z² + a²)` at the distance `z` from the wall, and the drift is the settling
    velocity scaled by the same mobility, so `dz = -drift z |z| / (z² + a²) dt + sqrt(2 D(z)) dW`. The laws are
    symmetric about the wall at `z = 0`, which reflects the particle at every frame. It is a colloid settling on a
    microscope slide, or a protein near a membrane.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        D0:
            The diffusion coefficient far from the wall.

        a:
            The distance over which the wall slows diffusion.

        drift:
            The settling velocity far from the wall.

        convention:
            The convention of the noise, `'ito'`, `'stratonovich'`, or `'anti-ito'`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import WallMobilityInteraction
        >>> WallMobilityInteraction('mobility', target='bead').a
        1.0
    """

    law_classes = (RateLaw, NoiseLaw, UpdateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        D0: Annotated[float, RealSet.open(0, oo)] = 1.0,
        a: Annotated[float, RealSet.open(0, oo)] = 1.0,
        drift: float = 0.2,
        convention: str = 'anti-ito',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.D0 = D0
        self.a = a
        self.drift = drift
        self.convention = convention

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the drift, the noise, and the reflection of the distance of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s, t = context, context.t
        z = s.target.z
        self.laws_ = [
            RateLaw(z.diff(t), -s.drift * z * abs(z) / (z**2 + s.a**2)),
            NoiseLaw(z, sqrt(2 * s.D0) * z / sqrt(z**2 + s.a**2), convention=self.convention),
            UpdateLaw(z, abs(z)),
        ]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import StochasticResonanceSystem
        >>> StochasticResonanceSystem(D=0.2).prepare().system_.interactions[0].D
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.amplitude = amplitude
        self.frequency = frequency
        self.D = D

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a particle in a double well.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        well = DoubleWellInteraction(
            'well',
            target='particle',
            amplitude=self.amplitude,
            frequency=self.frequency,
            D=self.D,
        )
        self.system_ = System(self.label, elements=[DoubleWellElement('particle', x=self.x)], interactions=[well])


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.z = z
        self.D0 = D0
        self.a = a
        self.drift = drift
        self.convention = convention

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a Brownian particle that settles on a wall.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        mobility = WallMobilityInteraction(
            'mobility',
            target='particle',
            D0=self.D0,
            a=self.a,
            drift=self.drift,
            convention=self.convention,
        )
        self.system_ = System(
            self.label,
            elements=[WallParticleElement('particle', z=self.z)],
            interactions=[mobility],
        )
