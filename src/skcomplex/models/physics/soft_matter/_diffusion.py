"""Diffusion models: free, trapped, escaping, scaled, and speckle-driven Brownian particles, and noise conventions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, exp, sin, sqrt

from ....core.base import Domain, VisualSpec, parameter, role, state, vector
from ....core.model import (
    Boundary,
    Element,
    EuclideanSpace,
    Interaction,
    LangevinThermostat,
    Law,
    Noise,
    Particle,
    Rate,
    Symbols,
    System,
    pulse_train,
    where,
)
from ....core.runtime import register_model
from ....core.solvers import EulerMaruyama
from ._brownian import DoubleWell

type Array = npt.NDArray[np.float64]

POINTS = VisualSpec('points', {'position': 'x', 'radius': 0.05})


class BrownianParticle(Element):
    """Overdamped Brownian particle, `dx = F / gamma dt + sqrt(2 temperature / gamma) dW`, pushed by interactions.

    Args:
        label:
            The label of the particle or the group.

        x:
            The initial position.

        gamma:
            The friction coefficient.

        temperature:
            The thermal energy.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import BrownianParticle
        >>> BrownianParticle('bead', temperature=2.0).temperature
        2.0
    """

    visual: ClassVar[VisualSpec] = POINTS
    x: npt.ArrayLike = state(default=0.0, vector=True)
    gamma: float = parameter(default=1.0, domain=Domain.POSITIVE)
    temperature: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the thermal noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Noise(s.x, sqrt(2 * s.temperature / s.gamma))]


class OpticalTrap(Interaction):
    """Harmonic optical trap on overdamped particles, `F = -k (x - c(t))` with one stiffness per axis.

    The center oscillates as `c(t) = amplitude sin(2 pi frequency t)`, and a blinking trap is on for the first
    `duty` fraction of every `blink_period` and off for the rest.

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
        >>> from skcomplex.models.physics.soft_matter import OpticalTrap
        >>> OpticalTrap('tweezers', target='bead', k=[1.0, 0.25]).k
        [1.0, 0.25]
    """

    target: str = role()
    k: npt.ArrayLike = parameter(default=1.0, vector=True)
    amplitude: npt.ArrayLike = parameter(default=0.0, vector=True)
    frequency: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    blink_period: float | None = None
    duty: float = 0.5

    def laws(self, s: Symbols) -> list[Law]:
        """Return the drift towards the center of the trap.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        center = s.amplitude * sin(2 * math.pi * s.frequency * s.t)
        on = 1 if self.blink_period is None else pulse_train(s.t, self.blink_period, self.duty * self.blink_period)
        position = s.target.x
        force = vector(
            [-stiffness * (item - middle) for stiffness, item, middle in zip(s.k, position, center, strict=True)],
        )
        return [Rate(position, on * force / s.target.gamma)]


class SpeckleField(Interaction):
    """Random optical speckle field of Gaussian hotspots, which drags particles up its gradient at `u grad(I)`.

    The intensity is `I(x) = sum_i I_i exp(-|x - c_i|^2 / d_i^2)`, with hotspot centers, intensities relative to
    `strength`, and decay lengths fixed when the field is created.

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
        >>> from skcomplex.models.physics.soft_matter import SpeckleField
        >>> SpeckleField('speckle', target='bead', centers=((0.0, 0.0),), intensities=(1.0,), lengths=(1.0,)).strength
        1.0
    """

    target: str = role()
    strength: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    centers: tuple[tuple[float, float], ...] = ()
    intensities: tuple[float, ...] = ()
    lengths: tuple[float, ...] = ()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the drift up the intensity gradient.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        position = s.target.x
        intensity = SR(0)
        for center, height, length in zip(self.centers, self.intensities, self.lengths, strict=True):
            offset = position - vector(center)
            intensity += height * exp(-offset.dot_product(offset) / length**2)
        return [Rate(position, s.strength * vector([intensity.diff(item) for item in position]))]


ScaledWalker = Element.define(
    'ScaledWalker',
    states={'x': 0.0},
    parameters={
        'K': parameter(default=1.0, domain=Domain.POSITIVE),
        'alpha': parameter(default=0.5, domain=Domain.POSITIVE),
        't0': parameter(default=0.01, domain=Domain.POSITIVE),
    },
    laws=lambda s: [Noise(s.x, sqrt(2 * s.K * s.alpha * (s.t + s.t0) ** (s.alpha - 1)))],
)


class ModulatedParticle(Element):
    """Particle in a box `[0, L]` whose noise grows across it, `dx = sigma(x) dW` with `sigma = s0 + ds (x / L - 1/2)`.

    The walls reflect the particle at every frame, so frames must be short against the time to cross the box. The
    stationary density is flat in the anti-Itô convention, as for a particle at
    thermal equilibrium, `1 / sigma` in the Stratonovich convention, and `1 / sigma^2` in the Itô convention.

    Args:
        label:
            The label of the particle or the group.

        x:
            The initial position.

        s0:
            The noise amplitude at the middle of the box.

        ds:
            The change of the noise amplitude across the box.

        L:
            The length of the box.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import ModulatedParticle
        >>> ModulatedParticle('bead').ds
        1.8
    """

    x: float = state(default=50.0)
    s0: float = parameter(default=1.0, domain=Domain.POSITIVE)
    ds: float = parameter(default=1.8)
    L: float = parameter(default=100.0, domain=Domain.POSITIVE)
    convention: str = 'anti-ito'

    def laws(self, s: Symbols) -> list[Law]:
        """Return the multiplicative noise and the reflecting walls.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        reflected = where(s.x < 0, -s.x, where(s.x > s.L, 2 * s.L - s.x, s.x))
        noise = Noise(s.x, s.s0 + s.ds * (s.x / s.L - 0.5), convention=self.convention)
        return [noise, Boundary(s.x, reflected)]


@register_model(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'diffusion', 'inertia'),
    references=('Argun et al. 2021', 'Li et al. 2010', 'Volpe and Volpe 2013'),
    defaults={'until': 10.0, 'dt': 0.1, 'solver': EulerMaruyama(dt=0.005)},
)
def brownian_motion(
    n_particles: int = 500,
    *,
    inertial: bool = True,
    m: float = 1.0,
    gamma: float = 1.0,
    kT: float = 1.0,
    n_dim: int = 1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return free Brownian particles with inertia, `m x'' = -gamma x' + sqrt(2 kT gamma) W`, or without it.

    With inertia the motion is ballistic, `<x^2> = kT t^2 / m`, below the momentum relaxation time `m / gamma`, and
    diffusive, `<x^2> = 2 kT t / gamma` per axis, above it. Without inertia it is diffusive at every time.

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

    Returns:
        system:
            The model, with the positions `beads.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import brownian_motion
        >>> brownian_motion(10, inertial=False).elements[0].element.__name__
        'BrownianParticle'
    """
    space = EuclideanSpace(n_dim)
    if not inertial:
        beads = BrownianParticle.group(
            'beads',
            n_particles,
            x=np.zeros((n_particles, n_dim)),
            gamma=gamma,
            temperature=kT,
        )
        return System(components=[beads], space=space)
    velocities = np.random.default_rng(random_state).normal(0.0, math.sqrt(kT / m), (n_particles, n_dim))
    beads = Particle.group('beads', n_particles, x=np.zeros((n_particles, n_dim)), v=velocities, m=m, visual=POINTS)
    bath = LangevinThermostat('bath', target='beads', temperature=kT, gamma=gamma)
    return System(components=[beads], interactions=[bath], space=space)


@register_model(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'optical tweezers', 'forcing'),
    references=('Argun et al. 2021', 'Ashkin et al. 1986', 'Jones, Maragò, and Volpe 2015'),
    defaults={'until': 20.0, 'dt': 0.1, 'solver': EulerMaruyama(dt=0.01)},
)
def optical_trap(
    n_particles: int = 200,
    *,
    k: Sequence[float] = (1.0, 0.25),
    gamma: float = 1.0,
    kT: float = 1.0,
    amplitude: Sequence[float] = (0.0, 0.0),
    frequency: float = 0.0,
    blink_period: float | None = None,
    duty: float = 0.5,
) -> System:
    """Return overdamped particles in a harmonic optical trap, still, oscillating, or blinking.

    In a still trap every coordinate is normal with variance `kT / k` and relaxes in the time `gamma / k`.

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

    Returns:
        system:
            The model, with the positions `beads.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import optical_trap
        >>> optical_trap(10, blink_period=2.0).interactions[0].blink_period
        2.0
    """
    n_dim = len(k)
    beads = BrownianParticle.group('beads', n_particles, x=np.zeros((n_particles, n_dim)), gamma=gamma, temperature=kT)
    trap = OpticalTrap(
        'tweezers',
        target='beads',
        k=list(k),
        amplitude=list(amplitude),
        frequency=frequency,
        blink_period=blink_period,
        duty=duty,
    )
    return System(components=[beads], interactions=[trap], space=EuclideanSpace(n_dim))


@register_model(
    field='physics/soft_matter',
    tags=('noise', 'sde', 'escape', 'double well', 'rates'),
    references=('Kramers 1940', 'Hänggi, Talkner, and Borkovec 1990', 'Gros 2015'),
    defaults={'until': 100.0, 'dt': 1.0, 'solver': EulerMaruyama(dt=0.01)},
)
def kramers_escape(
    n_particles: int = 200,
    *,
    D: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return particles that escape over the barrier of `U = x^4 / 4 - x^2 / 2` by thermal noise of intensity `D`.

    The particles start in the left well, and they hop between the wells at the Kramers rate
    `sqrt(U''(-1) |U''(0)|) / (2 pi) exp(-1 / (4 D))`, which is `sqrt(2) / (2 pi) exp(-1 / (4 D))`.

    Args:
        n_particles:
            The number of independent particles.

        D:
            The noise intensity.

        random_state:
            The seed or generator of the small initial spread in the left well.

    Returns:
        system:
            The model, with the positions `particles.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import kramers_escape
        >>> kramers_escape(10, D=0.2).elements[0].values['D']
        0.2
    """
    positions = -1.0 + np.random.default_rng(random_state).normal(0.0, 0.1, n_particles)
    group = DoubleWell.group('particles', n_particles, x=positions, amplitude=0.0, D=D)
    return System(components=[group])


@register_model(
    field='physics/soft_matter',
    tags=('noise', 'sde', 'multiplicative noise', 'spurious drift'),
    references=('Argun et al. 2021', 'Volpe and Wehr 2016'),
    defaults={'until': 100.0, 'dt': 0.05, 'solver': EulerMaruyama(dt=0.01)},
)
def multiplicative_noise(
    n_particles: int = 500,
    *,
    s0: float = 1.0,
    ds: float = 1.8,
    L: float = 100.0,
    convention: str = 'anti-ito',
) -> System:
    """Return free particles in a box with position-dependent noise, read in the Itô, Stratonovich, or anti-Itô sense.

    Only the anti-Itô convention gives the flat stationary density of particles at thermal equilibrium without
    forces. The particles start at the middle of the box.

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

    Returns:
        system:
            The model, with the positions `particles.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import multiplicative_noise
        >>> multiplicative_noise(10, convention='ito').components[0].values['convention']
        'ito'
    """
    group = ModulatedParticle.group('particles', n_particles, x=L / 2, s0=s0, ds=ds, L=L, convention=convention)
    return System(components=[group])


@register_model(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'sde', 'non-ergodic'),
    references=('Argun et al. 2021', 'Lim and Muniandy 2002', 'Jeon et al. 2014'),
    defaults={'until': 10.0, 'dt': 0.1, 'solver': EulerMaruyama(dt=0.01)},
)
def scaled_brownian_motion(
    n_particles: int = 500,
    *,
    alpha: float = 0.5,
    K: float = 1.0,
    t0: float = 0.01,
) -> System:
    """Return scaled Brownian motion, `dx = sqrt(2 K alpha (t + t0)^(alpha - 1)) dW`, with `<x^2> ~ 2 K t^alpha`.

    The diffusion coefficient changes deterministically in time, which is the same as running Brownian motion on the
    rescaled clock `t^alpha`. The offset `t0` keeps the coefficient finite at the start.

    Args:
        n_particles:
            The number of independent walkers.

        alpha:
            The anomalous diffusion exponent, below 1 for subdiffusion and above 1 for superdiffusion.

        K:
            The generalized diffusion coefficient.

        t0:
            The time offset of the diffusion coefficient.

    Returns:
        system:
            The model, with the positions `walkers.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import scaled_brownian_motion
        >>> scaled_brownian_motion(10, alpha=1.5).elements[0].values['alpha']
        1.5
    """
    return System(components=[ScaledWalker.group('walkers', n_particles, K=K, alpha=alpha, t0=t0)])


@register_model(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'sde', 'optical forces', 'disorder'),
    references=('Argun et al. 2021', 'Volpe, Volpe, and Gigan 2014'),
    defaults={'until': 60.0, 'dt': 0.5, 'solver': EulerMaruyama(dt=0.01)},
)
def diffusion_in_a_speckle_field(
    n_particles: int = 100,
    *,
    n_hotspots: int = 50,
    size: float = 100.0,
    strength: float = 30.0,
    length: float = 10.0,
    D: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Brownian particles in a random speckle field, trapped at hotspots for longer as the field grows stronger.

    The hotspots sit at random places in a square, with intensities `r^(-1/2)` and decay lengths `length r^(-1/2)` for
    independent uniform numbers `r`. The particles start at random places in the middle half of the square.

    The default strength 30 draws the particles into the hotspots over the first ten time units and lets them escape
    now and then, so from time 10 to 60 their mean squared displacement grows as `t^0.36`, a tenth of free diffusion.
    A strength of 100 or more traps every particle within the first frames and leaves only the rattling inside one
    hotspot, with a flat mean squared displacement.

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

    Returns:
        system:
            The model, with the positions `beads.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import diffusion_in_a_speckle_field
        >>> len(diffusion_in_a_speckle_field(5, n_hotspots=3).interactions[0].centers)
        3
    """
    rng = np.random.default_rng(random_state)
    centers = rng.uniform(0.0, size, (n_hotspots, 2))
    intensities = (1 - rng.random(n_hotspots)) ** -0.5
    lengths = length * (1 - rng.random(n_hotspots)) ** -0.5
    speckle = SpeckleField(
        'speckle',
        target='beads',
        strength=strength,
        centers=tuple((float(x), float(y)) for x, y in centers),
        intensities=tuple(float(item) for item in intensities),
        lengths=tuple(float(item) for item in lengths),
    )
    positions = rng.uniform(size / 4, 3 * size / 4, (n_particles, 2))
    beads = BrownianParticle.group('beads', n_particles, x=positions, gamma=1.0, temperature=D)
    return System(components=[beads], interactions=[speckle], space=EuclideanSpace(2))
