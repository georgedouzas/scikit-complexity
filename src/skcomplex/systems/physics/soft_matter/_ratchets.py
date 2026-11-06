"""Driven Brownian particles in periodic potentials: tilted washboards and flashing ratchets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver
from ..._catalog import register_system
from ._brownian_particle import BrownianParticleElement
from ._flashing_sawtooth import FlashingSawtoothInteraction
from ._thermal_noise import ThermalNoiseInteraction
from ._tilted_washboard import TiltedWashboardInteraction


@register_system(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'nonequilibrium steady state', 'current'),
    references=('Bertin 2021', 'Risken 1996', 'Reimann 2002'),
    defaults={'until': 50.0, 'dt': 0.25, 'solver': EulerMaruyamaSolver(dt=0.005)},
)
def tilted_periodic_potential(
    n_particles: int = 500,
    *,
    f: float = 1.0,
    U0: float = 0.5,
    L: float = 1.0,
    gamma: float = 1.0,
    kT: float = 0.5,
) -> System:
    """Return overdamped particles in the tilted periodic potential `V(x) = U0 cos(2 pi x / L) - f x`.

    The force `f` breaks the left and right symmetry, so the particles reach a nonequilibrium steady state with a
    uniform probability current `J`. Their mean velocity is
    `J L = L (1 - exp(-f L / kT)) / int_0^L dx int_0^L dy exp((V(x) - V(x - y)) / kT) kT / gamma`, which is
    `f / gamma` without the potential and exponentially small in the barrier for a weak force. The particles start
    at the origin.

    Args:
        n_particles:
            The number of independent particles.

        f:
            The constant force that tilts the potential.

        U0:
            The amplitude of the periodic potential.

        L:
            The period of the potential.

        gamma:
            The friction coefficient.

        kT:
            The thermal energy.

    Returns:
        system:
            The system, with the positions `particles.x`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import tilted_periodic_potential
        >>> tilted_periodic_potential(10, f=2.0).interactions[0].f
        2.0
    """
    particles = GroupElement(
        'particles',
        element=BrownianParticleElement,
        size=n_particles,
        values={'x': np.zeros((n_particles, 1)), 'gamma': gamma, 'temperature': kT},
    )
    washboard = TiltedWashboardInteraction('washboard', target='particles', U0=U0, L=L, f=f)
    bath = ThermalNoiseInteraction('bath', target='particles')
    return System(elements=[particles], interactions=[washboard, bath], space=ContinuousSpace(n_dim=1))


@register_system(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'ratchet', 'nonequilibrium steady state', 'current'),
    references=('Bertin 2021', 'Ajdari and Prost 1992', 'Reimann 2002'),
    defaults={'until': 100.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.002)},
)
def flashing_ratchet(
    n_particles: int = 500,
    *,
    barrier: float = 1.0,
    asymmetry: float = 0.1,
    L: float = 1.0,
    rate: float = 1.0,
    duration: float = 100.0,
    gamma: float = 1.0,
    kT: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the on-off flashing ratchet, Brownian particles in a sawtooth potential switched on and off at random.

    The potential switches at the rate `rate`, so on and off phases last `1 / rate` on average. While it is off the
    particles diffuse freely, and while it is on they slide to the minima of the sawtooth. A potential that rises over
    a short fraction `asymmetry` of its period sends more particles over its near maximum than over its far one, so
    the particles drift with a mean velocity of the sign of `0.5 - asymmetry`, and a symmetric potential gives no
    current. The switching times are drawn once, as a Poisson process over `duration` shared by every particle. The
    particles start at the minimum at the origin.

    Args:
        n_particles:
            The number of independent particles.

        barrier:
            The height of the sawtooth.

        asymmetry:
            The fraction of the period over which the potential rises, 0.5 for a symmetric potential.

        L:
            The period of the sawtooth.

        rate:
            The rate at which the potential switches.

        duration:
            The time span of the drawn switching times.

        gamma:
            The friction coefficient.

        kT:
            The thermal energy.

        random_state:
            The seed or generator of the switching times.

    Returns:
        system:
            The system, with the positions `particles.x`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import flashing_ratchet
        >>> len(flashing_ratchet(10, rate=2.0, duration=5.0).interactions[0].switches) > 0
        True
    """
    rng = np.random.default_rng(random_state)
    times = np.cumsum(rng.exponential(1 / rate, int(2 * rate * duration) + 10))
    particles = GroupElement(
        'particles',
        element=BrownianParticleElement,
        size=n_particles,
        values={'x': np.zeros((n_particles, 1)), 'gamma': gamma, 'temperature': kT},
    )
    sawtooth = FlashingSawtoothInteraction(
        'sawtooth',
        target='particles',
        barrier=barrier,
        L=L,
        asymmetry=asymmetry,
        switches=tuple(float(time) for time in times[times < duration]),
    )
    bath = ThermalNoiseInteraction('bath', target='particles')
    return System(elements=[particles], interactions=[sawtooth, bath], space=ContinuousSpace(n_dim=1))
