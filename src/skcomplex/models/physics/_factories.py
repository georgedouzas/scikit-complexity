"""Build the ready-made physics models from particles, forces, potentials, and boxes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from sage.all__sagemath_symbolics import cos, sin

from ...core.base import Domain, parameter, state
from ...core.model import Element, Equation, EuclideanSpace, Interaction, Rate, System, derivative
from ...core.runtime import register_model
from ._boxes import PeriodicBox
from ._mechanics import ConstantForce, Gravity, HarmonicPotential, LinearDrag, Particle
from ._potentials import LangevinThermostat, LennardJones

GRAVITY = 9.81


@register_model(
    field='physics',
    tags=('mechanics', 'ode'),
    references=('Newton 1687',),
    defaults={'until': 2.0, 'dt': 0.01},
)
def projectile(speed: float = 10.0, angle: float = 45.0, *, mass: float = 1.0, drag: float = 0.0) -> System:
    """Return a particle launched under uniform gravity, optionally slowed by linear drag.

    Args:
        speed:
            The launch speed.

        angle:
            The launch angle above the horizon, in degrees.

        mass:
            The mass.

        drag:
            The linear drag coefficient.

    Returns:
        system:
            The model, with the particle `ball` in two dimensions.

    Examples:
        >>> from skcomplex.models.physics import projectile
        >>> projectile(speed=5.0, angle=90.0).elements[0].v[1]
        5.0
    """
    radians = np.radians(angle)
    velocity = [round(float(speed * np.cos(radians)), 12), round(float(speed * np.sin(radians)), 12)]
    interactions: list[Interaction] = [ConstantForce('gravity', target='ball', F=[0.0, -GRAVITY * mass])]
    if drag:
        interactions.append(LinearDrag('drag', target='ball', gamma=drag))
    return System(
        elements=[Particle('ball', x=[0.0, 0.0], v=velocity, m=mass)],
        interactions=interactions,
        space=EuclideanSpace(2),
    )


@register_model(
    field='physics',
    tags=('mechanics', 'ode', 'oscillation'),
    references=('Hooke 1678',),
    defaults={'until': 20.0, 'dt': 0.05},
)
def harmonic_oscillator(
    x: float = 1.0,
    v: float = 0.0,
    *,
    k: float = 1.0,
    mass: float = 1.0,
    damping: float = 0.0,
) -> System:
    """Return a particle on a spring, optionally damped.

    Args:
        x:
            The initial displacement.

        v:
            The initial velocity.

        k:
            The stiffness.

        mass:
            The mass.

        damping:
            The linear damping coefficient.

    Returns:
        system:
            The model, with the particle `mass` in one dimension.

    Examples:
        >>> from skcomplex.models.physics import harmonic_oscillator
        >>> harmonic_oscillator(k=4.0).interactions[0].k
        4.0
    """
    interactions: list[Interaction] = [HarmonicPotential('spring', target='mass', k=k)]
    if damping:
        interactions.append(LinearDrag('damping', target='mass', gamma=damping))
    return System(
        elements=[Particle('mass', x=x, v=v, m=mass)],
        interactions=interactions,
        space=EuclideanSpace(1),
    )


@register_model(
    field='physics',
    tags=('mechanics', 'ode', 'gravity'),
    references=('Kepler 1609', 'Newton 1687'),
    defaults={'until': 6.0, 'dt': 0.01, 'solver': 'rk4'},
)
def two_body_orbit(
    eccentricity: float = 0.0,
    *,
    G: float = 1.0,
    mass: float = 1000.0,
    radius: float = 10.0,
) -> System:
    """Return a light planet orbiting a heavy star under Newtonian gravity.

    Args:
        eccentricity:
            The eccentricity of the orbit, between 0 for a circle and 1 for an escape.

        G:
            The gravitational constant.

        mass:
            The mass of the star.

        radius:
            The distance of the planet from the star at its closest approach.

    Returns:
        system:
            The model, with the particles `star` and `planet` in two dimensions.

    Examples:
        >>> from skcomplex.models.physics import two_body_orbit
        >>> [element.label for element in two_body_orbit().elements]
        ['star', 'planet']
    """
    speed = float(np.sqrt(G * mass * (1 + eccentricity) / radius))
    return System(
        elements=[Particle('star', m=mass), Particle('planet', x=[radius, 0.0], v=[0.0, speed], m=1e-6)],
        interactions=[Gravity('gravity', first='star', second='planet', G=G)],
        space=EuclideanSpace(2),
    )


@register_model(
    field='physics',
    tags=('molecular dynamics', 'pair potential', 'ode'),
    references=('Lennard-Jones 1924', 'Verlet 1967'),
    defaults={'until': 0.5, 'dt': 0.005, 'solver': 'velocity-verlet'},
)
def lennard_jones_fluid(
    n_particles: int = 400,
    *,
    density: float = 0.5,
    temperature: float | None = None,
    cutoff: float = 2.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Lennard-Jones particles in a periodic square box, optionally held at a temperature by a thermostat.

    Args:
        n_particles:
            The number of particles.

        density:
            The number of particles per unit area.

        temperature:
            The temperature of the Langevin thermostat, or `None` for constant energy. A thermostat makes the model
            stochastic, so it runs with the Euler-Maruyama solver instead of the default velocity Verlet.

        cutoff:
            The distance beyond which pairs do not interact.

        random_state:
            The seed or generator of the initial jitter and velocities.

    Returns:
        system:
            The model, with the particle group `fluid` in two dimensions.

    Examples:
        >>> from skcomplex.models.physics import lennard_jones_fluid
        >>> lennard_jones_fluid(100).elements[0].size
        100
    """
    rng = np.random.default_rng(random_state)
    size = float(np.sqrt(n_particles / density))
    side = int(np.ceil(np.sqrt(n_particles)))
    spacing = size / side
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n_particles]
    positions = np.mod((grid + 0.5) * spacing + rng.normal(0, 0.02 * spacing, (n_particles, 2)), size)
    velocities = rng.normal(0, np.sqrt(temperature or 1.0), (n_particles, 2))
    velocities -= velocities.mean(axis=0)
    interactions: list[Interaction] = [
        LennardJones('lj', first='fluid', second='fluid', cutoff=cutoff, period=size),
        PeriodicBox('box', target='fluid', size=size),
    ]
    if temperature is not None:
        interactions.append(LangevinThermostat('thermostat', target='fluid', temperature=temperature))
    return System(
        elements=[Particle.group('fluid', n_particles, x=positions, v=velocities)],
        interactions=interactions,
        space=EuclideanSpace(2),
    )


Pendulum = Element.define(
    'Pendulum',
    states={'theta': state(default=0.5, domain=Domain.ANGLE, order=2)},
    parameters={
        'm': parameter(default=1.0, domain=Domain.POSITIVE),
        'g': parameter(default=9.81, domain=Domain.POSITIVE),
        'l': parameter(default=1.0, domain=Domain.POSITIVE),
    },
    laws=lambda s: [Equation(s.m * s.l * derivative(s.theta, 2) == -s.m * s.g * sin(s.theta))],
)

Oscillator = Element.define(
    'Oscillator',
    states={'x': state(default=1.0, order=2)},
    parameters={
        'm': parameter(default=1.0, domain=Domain.POSITIVE),
        'k': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'c': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'F': parameter(default=0.0),
        'omega': parameter(default=1.0, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [Rate(s.x, (-s.k * s.x - s.c * derivative(s.x, 1) + s.F * cos(s.omega * s.t)) / s.m)],
)

Circuit = Element.define(
    'Circuit',
    states={'q': state(default=1.0, order=2)},
    parameters={
        'L': parameter(default=1.0, domain=Domain.POSITIVE),
        'R': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'C': parameter(default=1.0, domain=Domain.POSITIVE),
        'V': parameter(default=0.0),
        'omega': parameter(default=1.0, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Equation(s.L * derivative(s.q, 2) + s.R * derivative(s.q, 1) + s.q / s.C == s.V * cos(s.omega * s.t)),
    ],
)


@register_model(
    field='physics',
    tags=('mechanics', 'ode', 'second order', 'oscillation'),
    references=('Newton 1687',),
    defaults={'until': 10.0, 'dt': 0.01},
)
def pendulum(
    theta: float = 0.5,
    omega: float = 0.0,
    *,
    m: float = 1.0,
    g: float = 9.81,
    l: float = 1.0,
) -> System:
    """Return a pendulum written as Newton's second law, `m l θ'' = -m g sin θ`.

    Args:
        theta:
            The initial angle, in radians.

        omega:
            The initial angular velocity.

        m:
            The mass of the bob.

        g:
            The gravitational acceleration.

        l:
            The length of the rod.

    Returns:
        system:
            The model, with the pendulum `bob`, whose angle has order 2 and angular velocity `bob.theta_d1`.

    Examples:
        >>> from skcomplex.models.physics import pendulum
        >>> pendulum(theta=0.1).elements[0].theta_d1
        0.0
    """
    return System(elements=[Pendulum('bob', theta=theta, theta_d1=omega, m=m, g=g, l=l)])


@register_model(
    field='physics',
    tags=('mechanics', 'ode', 'second order', 'oscillation', 'forcing'),
    references=('Hooke 1678',),
    defaults={'until': 50.0, 'dt': 0.05},
)
def driven_damped_oscillator(
    x: float = 1.0,
    v: float = 0.0,
    *,
    m: float = 1.0,
    k: float = 1.0,
    c: float = 0.1,
    F: float = 0.0,
    omega: float = 1.0,
) -> System:
    """Return a mass on a spring with damping and a periodic force, `m x'' = -k x - c x' + F cos(ω t)`.

    Args:
        x:
            The initial displacement.

        v:
            The initial velocity.

        m:
            The mass.

        k:
            The stiffness.

        c:
            The damping coefficient.

        F:
            The amplitude of the driving force.

        omega:
            The angular frequency of the driving force.

    Returns:
        system:
            The model, with the oscillator `mass`, whose displacement has order 2.

    Examples:
        >>> from skcomplex.models.physics import driven_damped_oscillator
        >>> driven_damped_oscillator(F=0.5).elements[0].F
        0.5
    """
    return System(elements=[Oscillator('mass', x=x, x_d1=v, m=m, k=k, c=c, F=F, omega=omega)])


@register_model(
    field='physics',
    tags=('electromagnetism', 'circuit', 'ode', 'second order'),
    references=('Kirchhoff 1845',),
    defaults={'until': 50.0, 'dt': 0.05},
)
def rlc_circuit(
    q: float = 1.0,
    current: float = 0.0,
    *,
    L: float = 1.0,
    R: float = 0.1,
    C: float = 1.0,
    V: float = 0.0,
    omega: float = 1.0,
) -> System:
    """Return a series RLC circuit written as Kirchhoff's voltage law, `L q'' + R q' + q / C = V cos(ω t)`.

    Args:
        q:
            The initial charge of the capacitor.

        current:
            The initial current.

        L:
            The inductance.

        R:
            The resistance.

        C:
            The capacitance.

        V:
            The amplitude of the source voltage.

        omega:
            The angular frequency of the source.

    Returns:
        system:
            The model, with the circuit `circuit`, whose charge has order 2 and current `circuit.q_d1`.

    Examples:
        >>> from skcomplex.models.physics import rlc_circuit
        >>> rlc_circuit(R=0.5).elements[0].R
        0.5
    """
    return System(elements=[Circuit('circuit', q=q, q_d1=current, L=L, R=R, C=C, V=V, omega=omega)])
