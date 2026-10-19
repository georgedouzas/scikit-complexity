"""Model mechanics with impacts and walls: bouncing balls, Newton's cradle, billiards, and Brownian particles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import sqrt

from ...core.base import Domain, parameter, state
from ...core.model import Boundary, Element, Event, Interaction, Noise, Rate, System, derivative
from ...core.runtime import register_model

BouncingBall = Element.define(
    'BouncingBall',
    states={'h': state(default=1.0, order=2)},
    parameters={'g': 9.81, 'e': parameter(default=0.8, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [
        Rate(s.h, -s.g),
        Event(s.h, {derivative(s.h, 1): -s.e * derivative(s.h, 1)}, direction=-1, label='impact'),
    ],
)

HangingBall = Element.define(
    'HangingBall',
    states={'x': state(default=0.0, order=2)},
    parameters={'rest': 0.0, 'g': 9.81, 'length': parameter(default=1.0, domain=Domain.POSITIVE)},
    laws=lambda s: [Rate(s.x, -s.g / s.length * (s.x - s.rest))],
)

ElasticCollision = Interaction.define(
    'ElasticCollision',
    roles=['left', 'right'],
    parameters={'diameter': parameter(default=0.1, domain=Domain.POSITIVE)},
    laws=lambda s: [
        Event(
            s.right.x - s.left.x - s.diameter,
            {
                derivative(s.left.x, 1): derivative(s.right.x, 1),
                derivative(s.right.x, 1): derivative(s.left.x, 1),
            },
            direction=-1,
        ),
    ],
)

DiscBilliard = Element.define(
    'DiscBilliard',
    states={'x': 0.3, 'y': 0.0, 'vx': 0.4, 'vy': 1.0},
    parameters={'radius': parameter(default=1.0, domain=Domain.POSITIVE)},
    laws=lambda s: [
        Rate(s.x, s.vx),
        Rate(s.y, s.vy),
        Event(
            s.x**2 + s.y**2 - s.radius**2,
            {
                s.vx: s.vx - 2 * (s.vx * s.x + s.vy * s.y) * s.x / (s.x**2 + s.y**2),
                s.vy: s.vy - 2 * (s.vx * s.x + s.vy * s.y) * s.y / (s.x**2 + s.y**2),
            },
            direction=1,
            label='bounce',
        ),
    ],
)

WallParticle = Element.define(
    'WallParticle',
    states={'z': state(default=1.0, domain=Domain.NONNEGATIVE)},
    parameters={
        'D0': parameter(default=1.0, domain=Domain.POSITIVE),
        'a': parameter(default=1.0, domain=Domain.POSITIVE),
        'drift': 0.2,
    },
    laws=lambda s: [
        Rate(s.z, -s.drift * s.z * abs(s.z) / (s.z**2 + s.a**2)),
        Noise(s.z, sqrt(2 * s.D0) * s.z / sqrt(s.z**2 + s.a**2)),
        Boundary(s.z, abs(s.z)),
    ],
)


@register_model(
    field='physics',
    tags=('mechanics', 'events', 'impacts'),
    references=('Strogatz 2015', 'Hairer, Nørsett, and Wanner 1993'),
    defaults={'until': 3.0, 'dt': 0.01},
)
def bouncing_ball(h: float = 1.0, v: float = 0.0, *, g: float = 9.81, e: float = 0.8) -> System:
    """Return a ball that falls under gravity and bounces off the floor, losing speed by a coefficient of restitution.

    Every impact is an event that reverses the velocity and scales it by `e`. The impacts accumulate at the Zeno time
    `sqrt(2 h / g) (1 + e) / (1 - e)` for a ball dropped from rest, so runs should end before it.

    Args:
        h:
            The initial height.

        v:
            The initial vertical velocity.

        g:
            The gravitational acceleration.

        e:
            The coefficient of restitution.

    Returns:
        system:
            The model, with the height `ball.h` of order 2 and its velocity `ball.h_d1`.

    Examples:
        >>> from skcomplex.models.physics import bouncing_ball
        >>> from skcomplex.core.runtime import Simulation
        >>> trajectory = Simulation(bouncing_ball(), dt=0.01).run(until=1.0).trajectory_
        >>> round(trajectory.events[0].time, 6)
        0.451524
    """
    return System(elements=[BouncingBall('ball', h=h, h_d1=v, g=g, e=e)])


@register_model(
    field='physics',
    tags=('mechanics', 'events', 'impacts', 'momentum'),
    references=('Hinch and Saint-Jean 1999',),
    defaults={'until': 5.0, 'dt': 0.01},
)
def newtons_cradle(
    n_balls: int = 5,
    *,
    speed: float = 0.5,
    diameter: float = 0.1,
    gap: float = 0.001,
    length: float = 1.0,
    g: float = 9.81,
) -> System:
    """Return Newton's cradle: pendulum balls in a row that pass momentum along by elastic collisions.

    The balls swing with small amplitude about rest positions that leave a small gap between neighbours, so collisions
    are separate events. The first ball starts with a speed towards the others, and equal masses exchange velocities
    at every collision.

    Args:
        n_balls:
            The number of balls.

        speed:
            The initial speed of the first ball.

        diameter:
            The diameter of the balls.

        gap:
            The gap between neighbouring balls at rest.

        length:
            The length of the pendulums.

        g:
            The gravitational acceleration.

    Returns:
        system:
            The model, with the horizontal positions `ball_<i>.x` and velocities `ball_<i>.x_d1`.

    Examples:
        >>> from skcomplex.models.physics import newtons_cradle
        >>> len(newtons_cradle(n_balls=3).interactions)
        2
    """
    spacing = diameter + gap
    elements = [
        HangingBall(
            f'ball_{index}',
            x=index * spacing,
            x_d1=speed if index == 0 else 0.0,
            rest=index * spacing,
            g=g,
            length=length,
        )
        for index in range(n_balls)
    ]
    interactions = [
        ElasticCollision(f'collision_{index}', left=f'ball_{index}', right=f'ball_{index + 1}', diameter=diameter)
        for index in range(n_balls - 1)
    ]
    return System(elements=elements, interactions=interactions)


@register_model(
    field='physics',
    tags=('mechanics', 'events', 'billiard', 'integrable'),
    references=('Tabachnikov 2005', 'Datseris and Parlitz 2022'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def circular_billiard(
    x: float = 0.3,
    y: float = 0.0,
    vx: float = 0.4,
    vy: float = 1.0,
    *,
    radius: float = 1.0,
) -> System:
    """Return a particle moving freely inside a circle and reflecting specularly off its wall.

    The angle of incidence is the same at every bounce, so the circular billiard is integrable.

    Args:
        x:
            The initial horizontal position.

        y:
            The initial vertical position.

        vx:
            The initial horizontal velocity.

        vy:
            The initial vertical velocity.

        radius:
            The radius of the table.

    Returns:
        system:
            The model, with the position `particle.x`, `particle.y` and the velocity `particle.vx`, `particle.vy`.

    Examples:
        >>> from skcomplex.models.physics import circular_billiard
        >>> circular_billiard(radius=2.0).elements[0].radius
        2.0
    """
    return System(elements=[DiscBilliard('particle', x=x, y=y, vx=vx, vy=vy, radius=radius)])


@register_model(
    field='physics',
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
    noise: str = 'anti-ito',
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

        noise:
            The convention of the noise, `'ito'`, `'stratonovich'`, or `'anti-ito'`.

    Returns:
        system:
            The model, with the distance `particle.z`.

    Examples:
        >>> from skcomplex.models.physics import brownian_near_wall
        >>> brownian_near_wall(noise='ito').noise
        'ito'
    """
    return System(elements=[WallParticle('particle', z=z, D0=D0, a=a, drift=drift)], noise=noise)
