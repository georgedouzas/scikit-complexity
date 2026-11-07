"""Mechanics systems: the beam-beam map of colliding beams and a mass that sticks and slips under friction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, cos, exp, pi, sin

from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import where
from ....core.components.spaces import DiscreteTime
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ..._catalog import register_system
from ._beam import BeamElement
from ._friction_oscillator import FrictionOscillatorElement


def _rotate(s: SimpleNamespace) -> tuple[Expression, Expression]:
    """Return the position and momentum of the target turned by the angle `pi nu`."""
    a, b = cos(pi * s.target.nu), sin(pi * s.target.nu)
    return a * s.target.x + b * s.target.y, -b * s.target.x + a * s.target.y


def _kick(s: SimpleNamespace) -> Expression:
    """Return the momentum of the target after its rotation and the kick `-8 pi c (1 - exp(-u² / 2)) / u`."""
    position, momentum = _rotate(s)
    safe = where(position == 0, 1, position)
    return momentum - 8 * pi * s.target.c * where(position == 0, 0, (1 - exp(-(safe**2) / 2)) / safe)


@register_system(
    field='physics/mechanics',
    tags=('discrete map', 'hamiltonian', 'area preserving', 'accelerator', 'chaos'),
    references=('Dilão and Alves-Pires 1996', 'Dilão 2023'),
    defaults={'until': 2000, 'dt': 1},
)
def beam_beam_map(x: float = 0.5, y: float = 0.0, *, nu: float = 0.3, c: float = 0.05) -> System:
    """Return the beam-beam map of a particle of an accelerator beam that meets the opposite beam once every turn.

    Every turn the transverse position and momentum rotate by the angle `pi nu` of the tune, `u = a x + b y` with
    `a = cos(pi nu)` and `b = sin(pi nu)`, and the round Gaussian charge of the opposite beam kicks the momentum:
    `x' = u`, `y' = -b x + a y - 8 pi c (1 - exp(-u² / 2)) / u`. The map preserves area, as the composition of a
    rotation and a kick. The origin is a fixed point, whose linearisation has the trace `2 a - 4 pi c b` and is
    elliptic when the trace lies in `(-2, 2)`, and the phase space mixes invariant curves, island chains, and chaotic
    layers.

    Args:
        x:
            The initial transverse position.

        y:
            The initial transverse momentum.

        nu:
            The tune, in `[0, 2]`.

        c:
            The strength of the beam-beam kick.

    Returns:
        system:
            The system, with the states `particle.x` and `particle.y`.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import beam_beam_map
        >>> beam_beam_map(nu=0.5).elements[0].nu
        0.5
    """
    rotation = MapInteraction('x_map', target='particle', state='x', function=lambda s: _rotate(s)[0])
    kick = MapInteraction('y_map', target='particle', state='y', function=_kick)
    return System(
        elements=[BeamElement('particle', x=x, y=y, nu=nu, c=c)],
        interactions=[rotation, kick],
        time=DiscreteTime(),
    )


@register_system(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'friction', 'stick-slip', 'piecewise linear'),
    references=('Khalil 2002', 'Trzaska 2023'),
    defaults={'until': 40.0, 'dt': 0.01, 'solver': RK4Solver(dt=0.005)},
)
def friction_oscillator(
    x: float = 1.03,
    v: float = 0.0,
    *,
    m: float = 1.0,
    k: float = 1.0,
    c: float = 0.0,
    mu_s: float = 0.06,
    mu_k: float = 0.05,
    g: float = 1.0,
    v_stick: float = 0.01,
) -> System:
    """Return a mass on a linear spring with static, Coulomb, and viscous friction, `m x'' + c x' + k x + f = 0`.

    While the mass slides, the friction `f = mu_k m g sign(x')` opposes its motion, so in each half of the phase plane
    the motion is linear, around the shifted centres `x = ∓ mu_k m g / k`. At rest the static friction holds the mass
    while `|k x| <= mu_s m g`, so the system has a whole segment of equilibria rather than isolated ones. Without
    viscous friction, every half swing ends `2 mu_k m g / k` closer to the origin, and the mass sticks at the first
    turning point inside the segment, after finitely many swings. The friction is discontinuous, so the mass counts
    as at rest below the speed `v_stick`.

    Args:
        x:
            The initial position.

        v:
            The initial velocity.

        m:
            The mass.

        k:
            The stiffness of the spring.

        c:
            The viscous friction coefficient.

        mu_s:
            The static friction coefficient.

        mu_k:
            The kinetic friction coefficient.

        g:
            The gravitational acceleration.

        v_stick:
            The speed below which the mass counts as at rest.

    Returns:
        system:
            The system, with the position `mass.x` and the velocity `mass.v`.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import friction_oscillator
        >>> friction_oscillator(mu_k=0.1).elements[0].mu_k
        0.1
    """
    mass = FrictionOscillatorElement('mass', x=x, v=v, m=m, k=k, c=c, mu_s=mu_s, mu_k=mu_k, g=g, v_stick=v_stick)
    return System(elements=[mass])
