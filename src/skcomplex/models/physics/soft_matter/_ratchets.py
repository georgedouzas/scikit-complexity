"""Driven Brownian particles in periodic potentials: tilted washboards and flashing ratchets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from sage.all__sagemath_symbolics import sin

from ....core.base import Domain, parameter, role, vector
from ....core.model import ContinuousSpace, Interaction, Law, Rate, Symbols, System, mod, series, where
from ....core.runtime import register_model
from ....core.solvers import EulerMaruyama
from ._diffusion import BrownianParticle


class TiltedWashboard(Interaction):
    """Tilted periodic potential `U(x) - f x`, with `U(x) = U0 cos(2 pi x / L)`, on overdamped particles on a line.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group.

        U0:
            The amplitude of the periodic potential, half its barrier height.

        L:
            The period of the potential.

        f:
            The constant non-conservative force that tilts the potential.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import TiltedWashboard
        >>> TiltedWashboard('washboard', target='beads', f=0.5).f
        0.5
    """

    target: str = role()
    U0: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    L: float = parameter(default=1.0, domain=Domain.POSITIVE)
    f: float = parameter(default=0.0)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the drift down the tilted potential.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        position = s.target.x
        force = 2 * math.pi * s.U0 / s.L * sin(2 * math.pi * position[0] / s.L) + s.f
        return [Rate(position, vector([force / s.target.gamma]))]


class FlashingSawtooth(Interaction):
    """Sawtooth potential on overdamped particles on a line, turned on and off at given times.

    The potential of period `L` has a minimum at every multiple of `L`, rises linearly by `barrier` over the fraction
    `asymmetry` of the period, and falls back over the rest. It is on from time 0 to the first of the `switches`, off
    until the next one, and so on, and it keeps its state after the last one.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group.

        barrier:
            The height of the sawtooth.

        L:
            The period of the sawtooth.

        asymmetry:
            The fraction of the period over which the potential rises, 0.5 for a symmetric potential.

        switches:
            The increasing times at which the potential turns off and on in turn.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import FlashingSawtooth
        >>> FlashingSawtooth('ratchet', target='beads', switches=(1.0, 2.5)).switches
        (1.0, 2.5)
    """

    target: str = role()
    barrier: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    L: float = parameter(default=1.0, domain=Domain.POSITIVE)
    asymmetry: float = parameter(default=0.1, domain=Domain.POSITIVE)
    switches: tuple[float, ...] = ()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the drift down the sawtooth while it is on.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        times = (0.0, *self.switches)
        on = series(s.t, times, [float(index % 2 == 0) for index in range(len(times))], 'previous')
        position = s.target.x
        rise = s.asymmetry * s.L
        force = where(mod(position[0], s.L) < rise, -s.barrier / rise, s.barrier / (s.L - rise))
        return [Rate(position, vector([on * force / s.target.gamma]))]


@register_model(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'nonequilibrium steady state', 'current'),
    references=('Bertin 2021', 'Risken 1996', 'Reimann 2002'),
    defaults={'until': 50.0, 'dt': 0.25, 'solver': EulerMaruyama(dt=0.005)},
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
            The model, with the positions `particles.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import tilted_periodic_potential
        >>> tilted_periodic_potential(10, f=2.0).interactions[0].f
        2.0
    """
    particles = BrownianParticle.group(
        'particles',
        n_particles,
        x=np.zeros((n_particles, 1)),
        gamma=gamma,
        temperature=kT,
    )
    washboard = TiltedWashboard('washboard', target='particles', U0=U0, L=L, f=f)
    return System(components=[particles], interactions=[washboard], space=ContinuousSpace(n_dim=1))


@register_model(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'ratchet', 'nonequilibrium steady state', 'current'),
    references=('Bertin 2021', 'Ajdari and Prost 1992', 'Reimann 2002'),
    defaults={'until': 100.0, 'dt': 0.5, 'solver': EulerMaruyama(dt=0.002)},
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
            The model, with the positions `particles.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import flashing_ratchet
        >>> len(flashing_ratchet(10, rate=2.0, duration=5.0).interactions[0].switches) > 0
        True
    """
    rng = np.random.default_rng(random_state)
    times = np.cumsum(rng.exponential(1 / rate, int(2 * rate * duration) + 10))
    particles = BrownianParticle.group(
        'particles',
        n_particles,
        x=np.zeros((n_particles, 1)),
        gamma=gamma,
        temperature=kT,
    )
    sawtooth = FlashingSawtooth(
        'sawtooth',
        target='particles',
        barrier=barrier,
        L=L,
        asymmetry=asymmetry,
        switches=tuple(float(time) for time in times[times < duration]),
    )
    return System(components=[particles], interactions=[sawtooth], space=ContinuousSpace(n_dim=1))
