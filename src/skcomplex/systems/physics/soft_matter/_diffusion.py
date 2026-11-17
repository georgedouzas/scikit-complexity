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
from sage.all__sagemath_symbolics import SR, exp, oo, sin, sqrt, vector
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BoundaryLaw, NoiseLaw, RateLaw
from ....core.components.laws.utils import pulse_train, where
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.components.systems import BaseSystem as OldBaseSystem
from ....core.components.systems import CompositeSystem
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ..common.particles import (
    POINTS,
    BrownianParticleElement,
    DoubleWellElement,
    DoubleWellInteraction,
    FrictionNoiseInteraction,
    LangevinThermostatInteraction,
    MassPointElement,
    ThermalNoiseInteraction,
)


class ModulatedParticleElement(Element):
    """Particles at the positions `x` in a box.

    Args:
        label:
            The label of the particles.

        x:
            The initial position, one per particle or shared.

        size:
            The number of particles, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ModulatedParticleElement
        >>> ModulatedParticleElement('beads', x=5.0, size=10).x
        5.0
    """

    def __init__(self, label: str, *, x: object = 50.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x


class ModulatedNoiseInteraction(Interaction):
    """Noise whose amplitude grows across a box `[0, L]` whose walls reflect the particles.

    The noise amplitude is `sigma = s0 + ds (x / L - 1/2)`, read in the Itô, Stratonovich, or anti-Itô sense. The walls
    reflect the particles at every frame, so frames must be short against the time to cross the box. It is a colloid
    in a temperature gradient, or a particle whose mobility changes with its place.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles.

        s0:
            The noise amplitude at the middle of the box.

        ds:
            The change of the noise amplitude across the box.

        L:
            The length of the box.

        convention:
            The convention of the noise, `'ito'`, `'stratonovich'`, or `'anti-ito'`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ModulatedNoiseInteraction
        >>> ModulatedNoiseInteraction('noise', target='beads').ds
        1.8
    """

    law_classes = (NoiseLaw, BoundaryLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        s0: Annotated[float, RealSet.open(0, oo)] = 1.0,
        ds: float = 1.8,
        L: Annotated[float, RealSet.open(0, oo)] = 100.0,
        convention: str = 'anti-ito',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.s0 = s0
        self.ds = ds
        self.L = L
        self.convention = convention

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the noise of the position of the target and its reflection at the walls.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            x = s.target.x
            self.laws_ = [
                NoiseLaw(x, s.s0 + s.ds * (x / s.L - 0.5), convention=self.convention),
                BoundaryLaw(x, where(x < 0, -x, where(x > s.L, 2 * s.L - x, x))),
            ]


class ScaledWalkerElement(Element):
    """Walkers at the positions `x`.

    Args:
        label:
            The label of the walkers.

        x:
            The initial position, one per walker or shared.

        size:
            The number of walkers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ScaledWalkerElement
        >>> ScaledWalkerElement('walkers', x=1.0, size=10).x
        1.0
    """

    def __init__(self, label: str, *, x: object = 0.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x


class ScaledNoiseInteraction(Interaction):
    """Noise whose diffusion coefficient `K alpha (t + t0)^(alpha - 1)` changes in time.

    The position follows `dx = sqrt(2 K alpha (t + t0)^(alpha - 1)) dW`, Brownian motion on the rescaled clock
    `t^alpha`. It is a tracer in a cooling or ageing medium, or a walker whose steps shrink or grow in time.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the walkers.

        K:
            The generalized diffusion coefficient.

        alpha:
            The anomalous diffusion exponent.

        t0:
            The time offset of the diffusion coefficient.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ScaledNoiseInteraction
        >>> ScaledNoiseInteraction('noise', target='walkers', alpha=1.5).alpha
        1.5
    """

    law_classes = (NoiseLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        K: Annotated[float, RealSet.open(0, oo)] = 1.0,
        alpha: Annotated[float, RealSet.open(0, oo)] = 0.5,
        t0: Annotated[float, RealSet.open(0, oo)] = 0.01,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.K = K
        self.alpha = alpha
        self.t0 = t0

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the noise of the position of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [NoiseLaw(s.target.x, sqrt(2 * s.K * s.alpha * (s.t + s.t0) ** (s.alpha - 1)))]


class OpticalTrapInteraction(Interaction):
    """Harmonic optical trap on overdamped particles.

    The potential is `sum_d k_d (x_d - c_d(t))^2 / 2` over every axis.

    The center oscillates as `c(t) = amplitude sin(2 pi frequency t)`, and a blinking trap is on for the first
    `duty` fraction of every `blink_period` and off for the rest. The force divides by the friction `gamma` of the
    target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles, whose positions are `x` and friction `gamma`.

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

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        k: Sequence[float] = (1.0,),
        amplitude: Sequence[float] = (0.0,),
        frequency: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        blink_period: float | None = None,
        duty: float = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.k = k
        self.amplitude = amplitude
        self.frequency = frequency
        self.blink_period = blink_period
        self.duty = duty

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the drift of the positions of the target towards the center of the trap.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x = s.target.x
            wave = sin(2 * math.pi * s.frequency * t)
            on = 1 if self.blink_period is None else pulse_train(t, self.blink_period, self.duty * self.blink_period)
            rates = [
                -on * stiffness * (item - shift * wave) / s.target.gamma
                for stiffness, item, shift in zip(self.k, x, self.amplitude, strict=True)
            ]
            self.laws_ = [RateLaw(x, vector(SR, rates))]


class SpeckleFieldInteraction(Interaction):
    """Random optical speckle field of Gaussian hotspots.

    The field drags particles up its gradient at `u grad(I)`.

    The potential is `-u I(x)`, with the intensity `I(x) = sum_i I_i exp(-|x - c_i|^2 / d_i^2)` of hotspots whose
    centers, intensities relative to `strength`, and decay lengths are fixed when the field is created.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles, whose positions are `x`.

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

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        strength: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        centers: tuple[tuple[float, float], ...] = (),
        intensities: tuple[float, ...] = (),
        lengths: tuple[float, ...] = (),
    ) -> None:
        super().__init__(label)
        self.target = target
        self.strength = strength
        self.centers = centers
        self.intensities = intensities
        self.lengths = lengths

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the drift of the positions of the target up the gradient of the intensity.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            gradient = vector(SR, [0] * len(x))
            for center, height, length in zip(self.centers, self.intensities, self.lengths, strict=True):
                offset = x - vector(SR, center)
                gradient += -2 * height * exp(-offset.dot_product(offset) / length**2) * offset / length**2
            self.laws_ = [RateLaw(x, context.strength * gradient)]


class BeadElement(MassPointElement):
    """Point mass drawn as a point at its position."""

    visual = POINTS


class BrownianMotionSystem(OldBaseSystem):
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
            element=BeadElement,
            size=self.n_particles,
            values={'x': np.zeros((self.n_particles, self.n_dim)), 'x_d1': velocities, 'm': self.m},
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.k = k
        self.gamma = gamma
        self.kT = kT
        self.amplitude = amplitude
        self.frequency = frequency
        self.blink_period = blink_period
        self.duty = duty

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of overdamped particles in a harmonic optical trap.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        n_dim = len(self.k)
        beads = Element(
            'beads',
            values={'x': np.zeros((self.n_particles, n_dim)), 'gamma': self.gamma},
            size=self.n_particles,
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
        bath = FrictionNoiseInteraction('bath', target='beads', temperature=self.kT)
        self.system_ = System(
            self.label,
            elements=[beads],
            interactions=[trap, bath],
            space=ContinuousSpace(n_dim=n_dim),
            solver=EulerMaruyamaSolver(dt=0.01),
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import KramersEscapeSystem
        >>> KramersEscapeSystem(10, D=0.2).prepare().system_.interactions[0].D
        0.2
    """

    field = 'physics/soft_matter'
    topics = ('noise', 'sde', 'escape', 'double well', 'rates')
    references = ('Kramers 1940', 'Hänggi, Talkner, and Borkovec 1990', 'Gros 2015')
    until = 100.0
    dt = 1.0

    def __init__(
        self,
        n_particles: int = 200,
        *,
        D: float = 0.1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.D = D
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of particles in a double well that start in the left well.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        positions = -1.0 + np.random.default_rng(self.random_state).normal(0.0, 0.1, self.n_particles)
        self.system_ = System(
            self.label,
            elements=[DoubleWellElement('particles', x=positions, size=self.n_particles)],
            interactions=[DoubleWellInteraction('well', target='particles', amplitude=0.0, D=self.D)],
            solver=EulerMaruyamaSolver(dt=0.01),
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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

    def __init__(
        self,
        n_particles: int = 500,
        *,
        s0: float = 1.0,
        ds: float = 1.8,
        L: float = 100.0,
        convention: str = 'anti-ito',
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.s0 = s0
        self.ds = ds
        self.L = L
        self.convention = convention

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of free particles in a box with position-dependent noise.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        noise = ModulatedNoiseInteraction(
            'noise',
            target='particles',
            s0=self.s0,
            ds=self.ds,
            L=self.L,
            convention=self.convention,
        )
        self.system_ = System(
            self.label,
            elements=[ModulatedParticleElement('particles', x=self.L / 2, size=self.n_particles)],
            interactions=[noise],
            solver=EulerMaruyamaSolver(dt=0.01),
        )


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ScaledBrownianMotionSystem
        >>> ScaledBrownianMotionSystem(10, alpha=1.5).prepare().system_.interactions[0].alpha
        1.5
    """

    field = 'physics/soft_matter'
    topics = ('anomalous diffusion', 'sde', 'non-ergodic')
    references = ('Argun et al. 2021', 'Lim and Muniandy 2002', 'Jeon et al. 2014')
    until = 10.0
    dt = 0.1

    def __init__(
        self,
        n_particles: int = 500,
        *,
        alpha: float = 0.5,
        K: float = 1.0,
        t0: float = 0.01,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.alpha = alpha
        self.K = K
        self.t0 = t0

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of walkers whose diffusion coefficient changes in time.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[ScaledWalkerElement('walkers', size=self.n_particles)],
            interactions=[ScaledNoiseInteraction('noise', target='walkers', K=self.K, alpha=self.alpha, t0=self.t0)],
            solver=EulerMaruyamaSolver(dt=0.01),
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.n_hotspots = n_hotspots
        self.size = size
        self.strength = strength
        self.length = length
        self.D = D
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of Brownian particles in a random speckle field.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
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
        beads = Element('beads', values={'x': positions, 'gamma': 1.0}, size=self.n_particles)
        self.system_ = System(
            self.label,
            elements=[beads],
            interactions=[speckle, FrictionNoiseInteraction('bath', target='beads', temperature=self.D)],
            space=ContinuousSpace(n_dim=2),
            solver=EulerMaruyamaSolver(dt=0.01),
        )
