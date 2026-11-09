"""Diffusion systems.

The module covers free, trapped, escaping, scaled, and speckle-driven Brownian particles, and noise conventions.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, cos, exp, oo, sin, sqrt, vector
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import BaseInteraction, GradientInteraction, RateInteraction
from ....core.components.laws import BaseLaw, BoundaryLaw
from ....core.components.laws.utils import pulse_train, where
from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.particles import (
    POINTS,
    BrownianParticleElement,
    DoubleWellElement,
    LangevinThermostatInteraction,
    MassPointElement,
    ThermalNoiseInteraction,
)


class ModulatedParticleElement(Element):
    """Particle at the position `x` in a box `[0, L]` whose walls reflect it.

    The noise amplitude `sigma = s0 + ds (x / L - 1/2)` grows across the box. The walls reflect the particle at every
    frame, so frames must be short against the time to cross the box.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        s0:
            The noise amplitude at the middle of the box.

        ds:
            The change of the noise amplitude across the box.

        L:
            The length of the box.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ModulatedParticleElement
        >>> ModulatedParticleElement('bead').ds
        1.8
    """

    law_classes = (BoundaryLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(role='position')] = 50.0,
        s0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        ds: Annotated[float, ParameterSpec()] = 1.8,
        L: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.s0 = s0
        self.ds = ds
        self.L = L

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the reflection of the particle at the walls of the box.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The boundary law of the position.
        """
        return [BoundaryLaw(s.x, where(s.x < 0, -s.x, where(s.x > s.L, 2 * s.L - s.x, s.x)))]


class ScaledWalkerElement(Element):
    """Walker at the position `x` whose diffusion coefficient changes in time.

    The coefficient is `K alpha (t + t0)^(alpha - 1)`.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the walker is drawn, or `None` for none.

        x:
            The initial position.

        K:
            The generalized diffusion coefficient.

        alpha:
            The anomalous diffusion exponent.

        t0:
            The time offset of the diffusion coefficient.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ScaledWalkerElement
        >>> ScaledWalkerElement('walker', alpha=1.5).alpha
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(role='position')] = 0.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        t0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.K = K
        self.alpha = alpha
        self.t0 = t0


class OpticalTrapInteraction(GradientInteraction):
    """Harmonic optical trap on overdamped particles.

    The potential is `sum_d k_d (x_d - c_d(t))^2 / 2` over every axis.

    The center oscillates as `c(t) = amplitude sin(2 pi frequency t)`, and a blinking trap is on for the first
    `duty` fraction of every `blink_period` and off for the rest. The force divides by the friction `gamma` of the
    target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group.

        k:
            The stiffness along every axis.

        amplitude:
            The amplitude of the oscillation of the center along every axis.

        frequency:
            The frequency of the oscillation of the center.

        blink_period:
            The period of a blinking trap, or `None` for a trap that is always on.

        duty:
            The fraction of every blinking period during which the trap is on.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import OpticalTrapInteraction
        >>> OpticalTrapInteraction('tweezers', target='bead', k=[1.0, 0.25]).k
        [1.0, 0.25]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        k: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 1.0,
        amplitude: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        frequency: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        blink_period: float | None = None,
        duty: float = 0.5,
    ) -> None:
        super().__init__(label, target=target, states='x', divisor='gamma')
        self.k = k
        self.amplitude = amplitude
        self.frequency = frequency
        self.blink_period = blink_period
        self.duty = duty

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the potential of the trap, switched by the blinking.

        Args:
            namespace:
                The namespace of the symbols the potential can use.

        Returns:
            potential:
                The potential.
        """
        s = namespace
        center = s.amplitude * sin(2 * math.pi * s.frequency * s.t)
        on = 1 if self.blink_period is None else pulse_train(s.t, self.blink_period, self.duty * self.blink_period)
        energy = sum(
            (
                stiffness * (item - middle) ** 2 / 2
                for stiffness, item, middle in zip(s.k, s.target.x, center, strict=True)
            ),
            SR(0),
        )
        return on * energy


class SpeckleFieldInteraction(GradientInteraction):
    """Random optical speckle field of Gaussian hotspots.

    The field drags particles up its gradient at `u grad(I)`.

    The potential is `-u I(x)`, with the intensity `I(x) = sum_i I_i exp(-|x - c_i|^2 / d_i^2)` of hotspots whose
    centers, intensities relative to `strength`, and decay lengths are fixed when the field is created.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group.

        strength:
            The product `u I0` of the polarizability constant and the reference intensity.

        centers:
            The centers of the hotspots.

        intensities:
            The intensities of the hotspots in units of the reference intensity.

        lengths:
            The decay lengths of the hotspots.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import SpeckleFieldInteraction
        >>> SpeckleFieldInteraction(
        ...     'speckle', target='bead', centers=((0.0, 0.0),), intensities=(1.0,), lengths=(1.0,)
        ... ).strength
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        strength: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        centers: tuple[tuple[float, float], ...] = (),
        intensities: tuple[float, ...] = (),
        lengths: tuple[float, ...] = (),
    ) -> None:
        super().__init__(label, target=target, states='x')
        self.strength = strength
        self.centers = centers
        self.intensities = intensities
        self.lengths = lengths

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the potential `-u I(x)` of the speckle field.

        Args:
            namespace:
                The namespace of the symbols the potential can use.

        Returns:
            potential:
                The potential.
        """
        position = namespace.target.x
        intensity = SR(0)
        for center, height, length in zip(self.centers, self.intensities, self.lengths, strict=True):
            offset = position - vector(SR, center)
            intensity += height * exp(-offset.dot_product(offset) / length**2)
        return -namespace.strength * intensity


def _get_double_well_drive(s: SimpleNamespace) -> Expression:
    """Return the periodic forcing `A cos(2 pi f t)` of a particle in a double well."""
    return s.target.amplitude * cos(2 * math.pi * s.target.frequency * s.t)


class BrownianMotionSystem(BaseSystem):
    """Free Brownian particles with or without inertia.

    With inertia they follow `m x'' = -gamma x' + sqrt(2 kT gamma) W`.

    With inertia the motion is ballistic, `<x^2> = kT t^2 / m`, below the momentum relaxation time `m / gamma`, and
    diffusive, `<x^2> = 2 kT t / gamma` per axis, above it. Without inertia it is diffusive at every time. The system
    has the positions `beads.x`.

    Args:
        n_particles:
            The number of particles.

        inertial:
            Whether the particles have inertia.

        m:
            The mass, used with inertia.

        gamma:
            The friction coefficient.

        kT:
            The thermal energy.

        n_dim:
            The number of dimensions.

        random_state:
            The seed or generator of the thermal initial velocities.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import BrownianMotionSystem
        >>> BrownianMotionSystem(10, inertial=False).prepare().system_.elements[0].element.__name__
        'BrownianParticleElement'
    """

    field = 'physics/soft_matter'
    topics = ('brownian motion', 'sde', 'diffusion', 'inertia')
    references = ('Argun et al. 2021', 'Li et al. 2010', 'Volpe and Volpe 2013')
    until = 10.0
    dt = 0.1
    solver = EulerMaruyamaSolver(dt=0.005)

    def __init__(
        self,
        n_particles: int = 500,
        *,
        inertial: bool = True,
        m: float = 1.0,
        gamma: float = 1.0,
        kT: float = 1.0,
        n_dim: int = 1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.inertial = inertial
        self.m = m
        self.gamma = gamma
        self.kT = kT
        self.n_dim = n_dim
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return free Brownian particles with or without inertia.

        With inertia they follow `m x'' = -gamma x' + sqrt(2 kT gamma) W`.
        """
        space = ContinuousSpace(n_dim=self.n_dim)
        if not self.inertial:
            beads = GroupElement(
                'beads',
                element=BrownianParticleElement,
                size=self.n_particles,
                values={'x': np.zeros((self.n_particles, self.n_dim)), 'gamma': self.gamma, 'temperature': self.kT},
            )
            return CompositeSystem(
                self.label,
                elements=[beads],
                interactions=[ThermalNoiseInteraction('bath', target='beads')],
                space=space,
            )
        velocities = np.random.default_rng(self.random_state).normal(
            0.0,
            math.sqrt(self.kT / self.m),
            (self.n_particles, self.n_dim),
        )
        beads = GroupElement(
            'beads',
            element=MassPointElement,
            size=self.n_particles,
            values={'x': np.zeros((self.n_particles, self.n_dim)), 'x_d1': velocities, 'm': self.m},
            visual=POINTS,
        )
        bath = LangevinThermostatInteraction('bath', target='beads', temperature=self.kT, gamma=self.gamma)
        return CompositeSystem(self.label, elements=[beads], interactions=[bath], space=space)


class OpticalTrapSystem(BaseSystem):
    """Overdamped particles in a harmonic optical trap.

    The trap is still, oscillating, or blinking.

    In a still trap every coordinate is normal with variance `kT / k` and relaxes in the time `gamma / k`. The system
    has the positions `beads.x`.

    Args:
        n_particles:
            The number of independent particles.

        k:
            The stiffness along each axis.

        gamma:
            The friction coefficient.

        kT:
            The thermal energy.

        amplitude:
            The amplitude of the oscillation of the center along each axis.

        frequency:
            The frequency of the oscillation of the center.

        blink_period:
            The period of a blinking trap, or `None` for a trap that is always on.

        duty:
            The fraction of every blinking period during which the trap is on.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import OpticalTrapSystem
        >>> OpticalTrapSystem(10, blink_period=2.0).prepare().system_.interactions[0].blink_period
        2.0
    """

    field = 'physics/soft_matter'
    topics = ('brownian motion', 'sde', 'optical tweezers', 'forcing')
    references = ('Argun et al. 2021', 'Ashkin et al. 1986', 'Jones, Maragò, and Volpe 2015')
    until = 20.0
    dt = 0.1
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        n_particles: int = 200,
        *,
        k: Sequence[float] = (1.0, 0.25),
        gamma: float = 1.0,
        kT: float = 1.0,
        amplitude: Sequence[float] = (0.0, 0.0),
        frequency: float = 0.0,
        blink_period: float | None = None,
        duty: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.k = k
        self.gamma = gamma
        self.kT = kT
        self.amplitude = amplitude
        self.frequency = frequency
        self.blink_period = blink_period
        self.duty = duty

    def _build_system(self) -> CompositeSystem:
        """Return overdamped particles in a harmonic optical trap.

        The trap is still, oscillating, or blinking.
        """
        n_dim = len(self.k)
        beads = GroupElement(
            'beads',
            element=BrownianParticleElement,
            size=self.n_particles,
            values={'x': np.zeros((self.n_particles, n_dim)), 'gamma': self.gamma, 'temperature': self.kT},
        )
        trap = OpticalTrapInteraction(
            'tweezers',
            target='beads',
            k=list(self.k),
            amplitude=list(self.amplitude),
            frequency=self.frequency,
            blink_period=self.blink_period,
            duty=self.duty,
        )
        bath = ThermalNoiseInteraction('bath', target='beads')
        return CompositeSystem(
            self.label,
            elements=[beads],
            interactions=[trap, bath],
            space=ContinuousSpace(n_dim=n_dim),
        )


class KramersEscapeSystem(BaseSystem):
    """Particles that escape over the barrier of `U = x^4 / 4 - x^2 / 2` by noise.

    The noise is thermal with the intensity `D`.

    The particles start in the left well, and they hop between the wells at the Kramers rate `sqrt(U''(-1) |U''(0)|) /
    (2 pi) exp(-1 / (4 D))`, which is `sqrt(2) / (2 pi) exp(-1 / (4 D))`. The system has the positions `particles.x`.

    Args:
        n_particles:
            The number of independent particles.

        D:
            The noise intensity.

        random_state:
            The seed or generator of the small initial spread in the left well.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import KramersEscapeSystem
        >>> KramersEscapeSystem(10, D=0.2).prepare().system_.elements[0].values['D']
        0.2
    """

    field = 'physics/soft_matter'
    topics = ('noise', 'sde', 'escape', 'double well', 'rates')
    references = ('Kramers 1940', 'Hänggi, Talkner, and Borkovec 1990', 'Gros 2015')
    until = 100.0
    dt = 1.0
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        n_particles: int = 200,
        *,
        D: float = 0.1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.D = D
        self.random_state = random_state

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
        """Return the parts of the system, built from its model parameters."""
        positions = -1.0 + np.random.default_rng(self.random_state).normal(0.0, 0.1, self.n_particles)
        group = GroupElement(
            'particles',
            element=DoubleWellElement,
            size=self.n_particles,
            values={'x': positions, 'amplitude': 0.0, 'D': self.D},
        )
        return CompositeSystem(
            self.label,
            elements=[group],
            interactions=self._get_double_well_interactions('particles'),
        )


class MultiplicativeNoiseSystem(BaseSystem):
    """Free particles in a box with position-dependent noise.

    The noise is read in the Itô, Stratonovich, or anti-Itô sense.

    Only the anti-Itô convention gives the flat stationary density of particles at thermal equilibrium without forces.
    The particles start at the middle of the box. The system has the positions `particles.x`.

    Args:
        n_particles:
            The number of independent particles.

        s0:
            The noise amplitude at the middle of the box.

        ds:
            The change of the noise amplitude across the box.

        L:
            The length of the box.

        convention:
            The convention of the noise, `'ito'`, `'stratonovich'`, or `'anti-ito'`.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import MultiplicativeNoiseSystem
        >>> MultiplicativeNoiseSystem(10, convention='ito').prepare().system_.interactions[0].convention
        'ito'
    """

    field = 'physics/soft_matter'
    topics = ('noise', 'sde', 'multiplicative noise', 'spurious drift')
    references = ('Argun et al. 2021', 'Volpe and Wehr 2016')
    until = 100.0
    dt = 0.05
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        n_particles: int = 500,
        *,
        s0: float = 1.0,
        ds: float = 1.8,
        L: float = 100.0,
        convention: str = 'anti-ito',
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.s0 = s0
        self.ds = ds
        self.L = L
        self.convention = convention

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        group = GroupElement(
            'particles',
            element=ModulatedParticleElement,
            size=self.n_particles,
            values={'x': self.L / 2, 's0': self.s0, 'ds': self.ds, 'L': self.L},
        )
        noise = RateInteraction(
            'noise',
            target='particles',
            noise=lambda s: s.target.s0 + s.target.ds * (s.target.x / s.target.L - 0.5),
            convention=self.convention,
        )
        return CompositeSystem(self.label, elements=[group], interactions=[noise])


class ScaledBrownianMotionSystem(BaseSystem):
    """Scaled Brownian motion, `dx = sqrt(2 K alpha (t + t0)^(alpha - 1)) dW`.

    Its mean squared displacement is `<x^2> ~ 2 K t^alpha`.

    The diffusion coefficient changes deterministically in time, which is the same as running Brownian motion on the
    rescaled clock `t^alpha`. The offset `t0` keeps the coefficient finite at the start. The system has the positions
    `walkers.x`.

    Args:
        n_particles:
            The number of independent walkers.

        alpha:
            The anomalous diffusion exponent, below 1 for subdiffusion and above 1 for superdiffusion.

        K:
            The generalized diffusion coefficient.

        t0:
            The time offset of the diffusion coefficient.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ScaledBrownianMotionSystem
        >>> ScaledBrownianMotionSystem(10, alpha=1.5).prepare().system_.elements[0].values['alpha']
        1.5
    """

    field = 'physics/soft_matter'
    topics = ('anomalous diffusion', 'sde', 'non-ergodic')
    references = ('Argun et al. 2021', 'Lim and Muniandy 2002', 'Jeon et al. 2014')
    until = 10.0
    dt = 0.1
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        n_particles: int = 500,
        *,
        alpha: float = 0.5,
        K: float = 1.0,
        t0: float = 0.01,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.alpha = alpha
        self.K = K
        self.t0 = t0

    def _build_system(self) -> CompositeSystem:
        """Return scaled Brownian motion, `dx = sqrt(2 K alpha (t + t0)^(alpha - 1)) dW`.

        Its mean squared displacement is `<x^2> ~ 2 K t^alpha`.
        """
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'walkers',
                    element=ScaledWalkerElement,
                    size=self.n_particles,
                    values={'K': self.K, 'alpha': self.alpha, 't0': self.t0},
                ),
            ],
            interactions=[
                RateInteraction(
                    'noise',
                    target='walkers',
                    noise=lambda s: sqrt(2 * s.target.K * s.target.alpha * (s.t + s.target.t0) ** (s.target.alpha - 1)),
                ),
            ],
        )


class DiffusionInASpeckleFieldSystem(BaseSystem):
    """Brownian particles in a random speckle field.

    The particles are trapped at hotspots for longer as the field grows stronger.

    The hotspots sit at random places in a square, with intensities `r^(-1/2)` and decay lengths `length r^(-1/2)` for
    independent uniform numbers `r`. The particles start at random places in the middle half of the square.

    The default strength 30 draws the particles into the hotspots over the first ten time units and lets them escape now
    and then, so from time 10 to 60 their mean squared displacement grows as `t^0.36`, a tenth of free diffusion. A
    strength of 100 or more traps every particle within the first frames and leaves only the rattling inside one
    hotspot, with a flat mean squared displacement. The system has the positions `beads.x`.

    Args:
        n_particles:
            The number of independent particles.

        n_hotspots:
            The number of hotspots.

        size:
            The side of the square of the hotspots.

        strength:
            The product `u I0` of the polarizability constant and the reference intensity.

        length:
            The reference decay length of the hotspots.

        D:
            The diffusion coefficient.

        random_state:
            The seed or generator of the hotspots and the initial positions.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import DiffusionInASpeckleFieldSystem
        >>> len(DiffusionInASpeckleFieldSystem(5, n_hotspots=3).prepare().system_.interactions[0].centers)
        3
    """

    field = 'physics/soft_matter'
    topics = ('anomalous diffusion', 'sde', 'optical forces', 'disorder')
    references = ('Argun et al. 2021', 'Volpe, Volpe, and Gigan 2014')
    until = 60.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        n_particles: int = 100,
        *,
        n_hotspots: int = 50,
        size: float = 100.0,
        strength: float = 30.0,
        length: float = 10.0,
        D: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.n_hotspots = n_hotspots
        self.size = size
        self.strength = strength
        self.length = length
        self.D = D
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        centers = rng.uniform(0.0, self.size, (self.n_hotspots, 2))
        intensities = (1 - rng.random(self.n_hotspots)) ** -0.5
        lengths = self.length * (1 - rng.random(self.n_hotspots)) ** -0.5
        speckle = SpeckleFieldInteraction(
            'speckle',
            target='beads',
            strength=self.strength,
            centers=tuple((float(x), float(y)) for x, y in centers),
            intensities=tuple(float(item) for item in intensities),
            lengths=tuple(float(item) for item in lengths),
        )
        positions = rng.uniform(self.size / 4, 3 * self.size / 4, (self.n_particles, 2))
        beads = GroupElement(
            'beads',
            element=BrownianParticleElement,
            size=self.n_particles,
            values={'x': positions, 'gamma': 1.0, 'temperature': self.D},
        )
        bath = ThermalNoiseInteraction('bath', target='beads')
        return CompositeSystem(
            self.label,
            elements=[beads],
            interactions=[speckle, bath],
            space=ContinuousSpace(n_dim=2),
        )
