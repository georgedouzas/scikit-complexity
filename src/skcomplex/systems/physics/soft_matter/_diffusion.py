"""Diffusion systems: free, trapped, escaping, scaled, and speckle-driven Brownian particles, and noise conventions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np
from sage.all__sagemath_symbolics import sqrt

from ....core.components.elements import GroupElement
from ....core.components.interactions import RateInteraction
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver
from ..._catalog import register_system
from ..common.particles import LangevinThermostatInteraction, MassPointElement
from ._brownian import _get_double_well_interactions
from ._brownian_particle import POINTS, BrownianParticleElement
from ._double_well import DoubleWellElement
from ._modulated_particle import ModulatedParticleElement
from ._optical_trap import OpticalTrapInteraction
from ._scaled_walker import ScaledWalkerElement
from ._speckle_field import SpeckleFieldInteraction
from ._thermal_noise import ThermalNoiseInteraction


@register_system(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'diffusion', 'inertia'),
    references=('Argun et al. 2021', 'Li et al. 2010', 'Volpe and Volpe 2013'),
    defaults={'until': 10.0, 'dt': 0.1, 'solver': EulerMaruyamaSolver(dt=0.005)},
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
            The system, with the positions `beads.x`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import brownian_motion
        >>> brownian_motion(10, inertial=False).elements[0].element.__name__
        'BrownianParticleElement'
    """
    space = ContinuousSpace(n_dim=n_dim)
    if not inertial:
        beads = GroupElement(
            'beads',
            element=BrownianParticleElement,
            size=n_particles,
            values={'x': np.zeros((n_particles, n_dim)), 'gamma': gamma, 'temperature': kT},
        )
        return System(elements=[beads], interactions=[ThermalNoiseInteraction('bath', target='beads')], space=space)
    velocities = np.random.default_rng(random_state).normal(0.0, math.sqrt(kT / m), (n_particles, n_dim))
    beads = GroupElement(
        'beads',
        element=MassPointElement,
        size=n_particles,
        values={'x': np.zeros((n_particles, n_dim)), 'x_d1': velocities, 'm': m},
        visual=POINTS,
    )
    bath = LangevinThermostatInteraction('bath', target='beads', temperature=kT, gamma=gamma)
    return System(elements=[beads], interactions=[bath], space=space)


@register_system(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'optical tweezers', 'forcing'),
    references=('Argun et al. 2021', 'Ashkin et al. 1986', 'Jones, Maragò, and Volpe 2015'),
    defaults={'until': 20.0, 'dt': 0.1, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
            The system, with the positions `beads.x`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import optical_trap
        >>> optical_trap(10, blink_period=2.0).interactions[0].blink_period
        2.0
    """
    n_dim = len(k)
    beads = GroupElement(
        'beads',
        element=BrownianParticleElement,
        size=n_particles,
        values={'x': np.zeros((n_particles, n_dim)), 'gamma': gamma, 'temperature': kT},
    )
    trap = OpticalTrapInteraction(
        'tweezers',
        target='beads',
        k=list(k),
        amplitude=list(amplitude),
        frequency=frequency,
        blink_period=blink_period,
        duty=duty,
    )
    bath = ThermalNoiseInteraction('bath', target='beads')
    return System(elements=[beads], interactions=[trap, bath], space=ContinuousSpace(n_dim=n_dim))


@register_system(
    field='physics/soft_matter',
    tags=('noise', 'sde', 'escape', 'double well', 'rates'),
    references=('Kramers 1940', 'Hänggi, Talkner, and Borkovec 1990', 'Gros 2015'),
    defaults={'until': 100.0, 'dt': 1.0, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
            The system, with the positions `particles.x`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import kramers_escape
        >>> kramers_escape(10, D=0.2).elements[0].values['D']
        0.2
    """
    positions = -1.0 + np.random.default_rng(random_state).normal(0.0, 0.1, n_particles)
    group = GroupElement(
        'particles',
        element=DoubleWellElement,
        size=n_particles,
        values={'x': positions, 'amplitude': 0.0, 'D': D},
    )
    return System(elements=[group], interactions=_get_double_well_interactions('particles'))


@register_system(
    field='physics/soft_matter',
    tags=('noise', 'sde', 'multiplicative noise', 'spurious drift'),
    references=('Argun et al. 2021', 'Volpe and Wehr 2016'),
    defaults={'until': 100.0, 'dt': 0.05, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
            The system, with the positions `particles.x`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import multiplicative_noise
        >>> multiplicative_noise(10, convention='ito').interactions[0].convention
        'ito'
    """
    group = GroupElement(
        'particles',
        element=ModulatedParticleElement,
        size=n_particles,
        values={'x': L / 2, 's0': s0, 'ds': ds, 'L': L},
    )
    noise = RateInteraction(
        'noise',
        target='particles',
        noise=lambda s: s.target.s0 + s.target.ds * (s.target.x / s.target.L - 0.5),
        convention=convention,
    )
    return System(elements=[group], interactions=[noise])


@register_system(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'sde', 'non-ergodic'),
    references=('Argun et al. 2021', 'Lim and Muniandy 2002', 'Jeon et al. 2014'),
    defaults={'until': 10.0, 'dt': 0.1, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
            The system, with the positions `walkers.x`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import scaled_brownian_motion
        >>> scaled_brownian_motion(10, alpha=1.5).elements[0].values['alpha']
        1.5
    """
    return System(
        elements=[
            GroupElement(
                'walkers',
                element=ScaledWalkerElement,
                size=n_particles,
                values={'K': K, 'alpha': alpha, 't0': t0},
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


@register_system(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'sde', 'optical forces', 'disorder'),
    references=('Argun et al. 2021', 'Volpe, Volpe, and Gigan 2014'),
    defaults={'until': 60.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
            The system, with the positions `beads.x`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import diffusion_in_a_speckle_field
        >>> len(diffusion_in_a_speckle_field(5, n_hotspots=3).interactions[0].centers)
        3
    """
    rng = np.random.default_rng(random_state)
    centers = rng.uniform(0.0, size, (n_hotspots, 2))
    intensities = (1 - rng.random(n_hotspots)) ** -0.5
    lengths = length * (1 - rng.random(n_hotspots)) ** -0.5
    speckle = SpeckleFieldInteraction(
        'speckle',
        target='beads',
        strength=strength,
        centers=tuple((float(x), float(y)) for x, y in centers),
        intensities=tuple(float(item) for item in intensities),
        lengths=tuple(float(item) for item in lengths),
    )
    positions = rng.uniform(size / 4, 3 * size / 4, (n_particles, 2))
    beads = GroupElement(
        'beads',
        element=BrownianParticleElement,
        size=n_particles,
        values={'x': positions, 'gamma': 1.0, 'temperature': D},
    )
    bath = ThermalNoiseInteraction('bath', target='beads')
    return System(elements=[beads], interactions=[speckle, bath], space=ContinuousSpace(n_dim=2))
