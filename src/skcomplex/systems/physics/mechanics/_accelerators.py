"""Mechanics systems of colliding beams and friction.

The module holds the beam-beam map of colliding beams and a mass that sticks and slips under friction.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, cos, exp, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components.laws.utils import both, where
from ....core.engine.solvers import RK4Solver
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class BeamBeamMapSystem(DynamicalSystem):
    """The beam-beam map of a particle of an accelerator beam.

    The particle meets the opposite beam once every turn.

    Every turn the transverse position and momentum rotate by the angle `pi nu` of the tune, `u = a x + b y` with `a =
    cos(pi nu)` and `b = sin(pi nu)`, and the round Gaussian charge of the opposite beam kicks the momentum: `x' = u`,
    `y' = -b x + a y - 8 pi c (1 - exp(-u² / 2)) / u`. The map preserves area, as the composition of a rotation and a
    kick. The origin is a fixed point, whose linearisation has the trace `2 a - 4 pi c b` and is elliptic when the trace
    lies in `(-2, 2)`, and the phase space mixes invariant curves, island chains, and chaotic layers. The system has the
    states `particle.x` and `particle.y`.

    Args:
        x:
            The initial transverse position.

        y:
            The initial transverse momentum.

        nu:
            The tune, in `[0, 2]`.

        c:
            The strength of the beam-beam kick.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BeamBeamMapSystem
        >>> BeamBeamMapSystem(nu=0.5).prepare().system_.elements[0].get_value('nu')
        0.5
    """

    field = 'physics/mechanics'
    topics = ('discrete map', 'hamiltonian', 'area preserving', 'accelerator', 'chaos')
    references = ('Dilão and Alves-Pires 1996', 'Dilão 2023')
    until = 2000
    dt = 1

    @staticmethod
    def _rotate(s: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the position and momentum of the target turned by the angle `pi nu`."""
        a, b = cos(pi * s.target.nu), sin(pi * s.target.nu)
        return a * s.target.x + b * s.target.y, -b * s.target.x + a * s.target.y

    def __init__(
        self,
        x: float = 0.5,
        y: float = 0.0,
        *,
        nu: float = 0.3,
        c: float = 0.05,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.nu = nu
        self.c = c

    @staticmethod
    def _kick(s: SimpleNamespace) -> Expression:
        """Return the momentum of the target after its rotation and kick.

        The kick is `-8 pi c (1 - exp(-u² / 2)) / u`.
        """
        position, momentum = BeamBeamMapSystem._rotate(s)
        safe = where(position == 0, 1, position)
        return momentum - 8 * pi * s.target.c * where(position == 0, 0, (1 - exp(-(safe**2) / 2)) / safe)

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'particle'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'x': DeclaredVariableSpec(self.x, StateSpec()),
            'y': DeclaredVariableSpec(self.y, StateSpec()),
            'nu': DeclaredVariableSpec(self.nu, ParameterSpec(domain=RealSet.closed(0, 2))),
            'c': DeclaredVariableSpec(self.c, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the parts of the system, built from its model parameters."""
        return {
            'x': lambda s: BeamBeamMapSystem._rotate(s)[0],
            'y': self._kick,
        }


class FrictionOscillatorSystem(DynamicalSystem):
    """A mass on a linear spring with static, Coulomb, and viscous friction.

    The mass follows `m x'' + c x' + k x + f = 0`.

    While the mass slides, the friction `f = mu_k m g sign(x')` opposes its motion, so in each half of the phase plane
    the motion is linear, around the shifted centres `x = ∓ mu_k m g / k`. At rest the static friction holds the mass
    while `|k x| <= mu_s m g`, so the system has a whole segment of equilibria rather than isolated ones. Without
    viscous friction, every half swing ends `2 mu_k m g / k` closer to the origin, and the mass sticks at the first
    turning point inside the segment, after finitely many swings. The friction is discontinuous, so the mass counts as
    at rest below the speed `v_stick`. The system has the position `mass.x` and the velocity `mass.v`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import FrictionOscillatorSystem
        >>> FrictionOscillatorSystem(mu_k=0.1).prepare().system_.elements[0].get_value('mu_k')
        0.1
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'friction', 'stick-slip', 'piecewise linear')
    references = ('Khalil 2002', 'Trzaska 2023')
    until = 40.0
    dt = 0.01
    solver = RK4Solver(dt=0.005)

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.v = v
        self.m = m
        self.k = k
        self.c = c
        self.mu_s = mu_s
        self.mu_k = mu_k
        self.g = g
        self.v_stick = v_stick

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'mass'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        nonnegative = RealSet.closed_open(0, oo)
        return {
            'x': DeclaredVariableSpec(self.x, StateSpec()),
            'v': DeclaredVariableSpec(self.v, StateSpec()),
            'm': DeclaredVariableSpec(self.m, ParameterSpec(domain=RealSet.open(0, oo))),
            'k': DeclaredVariableSpec(self.k, ParameterSpec(domain=nonnegative)),
            'c': DeclaredVariableSpec(self.c, ParameterSpec(domain=nonnegative)),
            'mu_s': DeclaredVariableSpec(self.mu_s, ParameterSpec(domain=nonnegative)),
            'mu_k': DeclaredVariableSpec(self.mu_k, ParameterSpec(domain=nonnegative)),
            'g': DeclaredVariableSpec(self.g, ParameterSpec(domain=RealSet.open(0, oo))),
            'v_stick': DeclaredVariableSpec(self.v_stick, ParameterSpec(domain=RealSet.open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the motion of the mass under the spring and the friction, which makes it stick and slip."""
        return {
            'x': lambda s: where(abs(s.target.v) > s.target.v_stick, s.target.v, 0),
            'v': self._get_acceleration,
        }

    @staticmethod
    def _get_acceleration(s: SimpleNamespace) -> Expression:
        """Return the acceleration of the mass, zero while the static friction holds it."""
        mass = s.target
        force = -mass.k * mass.x - mass.c * mass.v
        moving = abs(mass.v) > mass.v_stick
        direction = where(moving, where(mass.v > 0, 1, -1), where(force > 0, 1, -1))
        stuck = both(abs(mass.v) <= mass.v_stick, abs(mass.k * mass.x) <= mass.mu_s * mass.m * mass.g)
        return where(stuck, 0, (force - mass.mu_k * mass.m * mass.g * direction) / mass.m)
