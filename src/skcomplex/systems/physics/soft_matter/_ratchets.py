"""Driven Brownian particles in periodic potentials.

The module covers tilted washboards and flashing ratchets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import GroupElement
from ....core.components.interactions import GradientInteraction, RateInteraction
from ....core.components.laws.utils import mod, series, where
from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ParameterSpec, RoleSpec
from ..common.particles import BrownianParticleElement, ThermalNoiseInteraction


class FlashingSawtoothInteraction(RateInteraction):
    """Sawtooth potential on overdamped particles on a line.

    The potential is turned on and off at given times.

    Its force is the rate of the position, divided by the friction `gamma` of the target.

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
        >>> from skcomplex.systems.physics.soft_matter import FlashingSawtoothInteraction
        >>> FlashingSawtoothInteraction('ratchet', target='beads', switches=(1.0, 2.5)).switches
        (1.0, 2.5)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        barrier: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        L: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        asymmetry: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        switches: tuple[float, ...] = (),
    ) -> None:
        super().__init__(label, target=target, state="x", divisor="gamma")
        self.barrier = barrier
        self.L = L
        self.asymmetry = asymmetry
        self.switches = switches

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the force of the sawtooth while it is on.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The force, before the division by the friction.
        """
        s = namespace
        times = (0.0, *self.switches)
        on = series(s.t, times, [float(index % 2 == 0) for index in range(len(times))], "previous")
        rise = s.asymmetry * s.L
        return on * where(mod(s.target.x[0], s.L) < rise, -s.barrier / rise, s.barrier / (s.L - rise))


class TiltedWashboardInteraction(GradientInteraction):
    """Tilted periodic potential `U(x) - f x` on overdamped particles on a line.

    The periodic part is `U(x) = U0 cos(2 pi x / L)`.

    The force divides by the friction `gamma` of the target.

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
        >>> from skcomplex.systems.physics.soft_matter import TiltedWashboardInteraction
        >>> TiltedWashboardInteraction('washboard', target='beads', f=0.5).f
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        U0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        L: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        f: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, target=target, states="x", divisor="gamma")
        self.U0 = U0
        self.L = L
        self.f = f

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return the tilted potential `U0 cos(2 pi x / L) - f x`.

        Args:
            s:
                The namespace of the symbols the potential can use.

        Returns:
            potential:
                The potential.
        """
        position = s.target.x[0]
        return s.U0 * cos(2 * math.pi * position / s.L) - s.f * position


class TiltedPeriodicPotentialSystem(BaseSystem):
    """Overdamped particles in a tilted periodic potential.

    The potential is `V(x) = U0 cos(2 pi x / L) - f x`.

    The force `f` breaks the left and right symmetry, so the particles reach a nonequilibrium steady state with a
    uniform probability current `J`. Their mean velocity is `J L = L (1 - exp(-f L / kT)) / int_0^L dx int_0^L dy
    exp((V(x) - V(x - y)) / kT) kT / gamma`, which is `f / gamma` without the potential and exponentially small in the
    barrier for a weak force. The particles start at the origin. The system has the positions `particles.x`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import TiltedPeriodicPotentialSystem
        >>> TiltedPeriodicPotentialSystem(10, f=2.0).prepare().system_.interactions[0].f
        2.0
    """

    field = "physics/soft_matter"
    topics = ("brownian motion", "sde", "nonequilibrium steady state", "current")
    references = ("Bertin 2021", "Risken 1996", "Reimann 2002")
    until = 50.0
    dt = 0.25
    solver = EulerMaruyamaSolver(dt=0.005)

    def __init__(
        self,
        n_particles: int = 500,
        *,
        f: float = 1.0,
        U0: float = 0.5,
        L: float = 1.0,
        gamma: float = 1.0,
        kT: float = 0.5,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.f = f
        self.U0 = U0
        self.L = L
        self.gamma = gamma
        self.kT = kT

    def _build_system(self) -> System:
        """Return overdamped particles in a tilted periodic potential.

        The potential is `V(x) = U0 cos(2 pi x / L) - f x`.
        """
        particles = GroupElement(
            "particles",
            element=BrownianParticleElement,
            size=self.n_particles,
            values={"x": np.zeros((self.n_particles, 1)), "gamma": self.gamma, "temperature": self.kT},
        )
        washboard = TiltedWashboardInteraction("washboard", target="particles", U0=self.U0, L=self.L, f=self.f)
        bath = ThermalNoiseInteraction("bath", target="particles")
        return System(self.label, elements=[particles], interactions=[washboard, bath], space=ContinuousSpace(n_dim=1))


class FlashingRatchetSystem(BaseSystem):
    """The on-off flashing ratchet.

    Brownian particles move in a sawtooth potential switched on and off at random.

    The potential switches at the rate `rate`, so on and off phases last `1 / rate` on average. While it is off the
    particles diffuse freely, and while it is on they slide to the minima of the sawtooth. A potential that rises over a
    short fraction `asymmetry` of its period sends more particles over its near maximum than over its far one, so the
    particles drift with a mean velocity of the sign of `0.5 - asymmetry`, and a symmetric potential gives no current.
    The switching times are drawn once, as a Poisson process over `duration` shared by every particle. The particles
    start at the minimum at the origin. The system has the positions `particles.x`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import FlashingRatchetSystem
        >>> len(FlashingRatchetSystem(10, rate=2.0, duration=5.0).prepare().system_.interactions[0].switches) > 0
        True
    """

    field = "physics/soft_matter"
    topics = ("brownian motion", "sde", "ratchet", "nonequilibrium steady state", "current")
    references = ("Bertin 2021", "Ajdari and Prost 1992", "Reimann 2002")
    until = 100.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=0.002)

    def __init__(
        self,
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
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.barrier = barrier
        self.asymmetry = asymmetry
        self.L = L
        self.rate = rate
        self.duration = duration
        self.gamma = gamma
        self.kT = kT
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        times = np.cumsum(rng.exponential(1 / self.rate, int(2 * self.rate * self.duration) + 10))
        particles = GroupElement(
            "particles",
            element=BrownianParticleElement,
            size=self.n_particles,
            values={"x": np.zeros((self.n_particles, 1)), "gamma": self.gamma, "temperature": self.kT},
        )
        sawtooth = FlashingSawtoothInteraction(
            "sawtooth",
            target="particles",
            barrier=self.barrier,
            L=self.L,
            asymmetry=self.asymmetry,
            switches=tuple(float(time) for time in times[times < self.duration]),
        )
        bath = ThermalNoiseInteraction("bath", target="particles")
        return System(self.label, elements=[particles], interactions=[sawtooth, bath], space=ContinuousSpace(n_dim=1))
