"""Mechanics systems of colliding beams and friction.

The module holds the beam-beam map of colliding beams and a mass that sticks and slips under friction.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, exp, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import both, where
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import RK4Solver
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class FrictionOscillatorElement(Element):
    """Mass on a spring with static, Coulomb, and viscous friction.

    The mass sticks and slips.

    The mass `m` at the position `x` with the velocity `v` feels the spring `-k x` and the viscous friction `-c v`.
    While it slides it feels the Coulomb friction `-mu_k m g sign(v)`, and at rest it stays at rest while the spring
    force stays within the static friction `mu_s m g`. The friction is discontinuous in the velocity, so the mass
    counts as at rest below the speed `v_stick`, which must exceed the change of speed of one solver step, and its
    position stays still while it is at rest.

    Args:
        label:
            The label of the mass.

        visual:
            How the mass is drawn, or `None` for `default_visual_`.

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

    Attributes:
        default_visual_:
            The visual of the mass when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import FrictionOscillatorElement
        >>> FrictionOscillatorElement('mass').mu_s
        0.06
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.03,
        v: Annotated[float, StateSpec()] = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        mu_s: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.06,
        mu_k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        v_stick: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.m = m
        self.k = k
        self.c = c
        self.mu_s = mu_s
        self.mu_k = mu_k
        self.g = g
        self.v_stick = v_stick

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the motion of the mass under the spring and the friction.

        The friction makes the mass stick and slip.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the position and the velocity.
        """
        s = namespace
        force = -s.k * s.x - s.c * s.v
        moving = abs(s.v) > s.v_stick
        direction = where(moving, where(s.v > 0, 1, -1), where(force > 0, 1, -1))
        stuck = both(abs(s.v) <= s.v_stick, abs(s.k * s.x) <= s.mu_s * s.m * s.g)
        return [
            RateLaw(s.x, where(moving, s.v, 0)),
            RateLaw(s.v, where(stuck, 0, (force - s.mu_k * s.m * s.g * direction) / s.m)),
        ]


def _rotate(s: SimpleNamespace) -> tuple[Expression, Expression]:
    """Return the position and momentum of the target turned by the angle `pi nu`."""
    a, b = cos(pi * s.target.nu), sin(pi * s.target.nu)
    return a * s.target.x + b * s.target.y, -b * s.target.x + a * s.target.y


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
        position, momentum = _rotate(s)
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
            'x': lambda s: _rotate(s)[0],
            'y': self._kick,
        }


class FrictionOscillatorSystem(BaseSystem):
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
        >>> FrictionOscillatorSystem(mu_k=0.1).prepare().system_.elements[0].mu_k
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
        super().__init__(label)
        self.x = x
        self.v = v
        self.m = m
        self.k = k
        self.c = c
        self.mu_s = mu_s
        self.mu_k = mu_k
        self.g = g
        self.v_stick = v_stick

    def _build_system(self) -> CompositeSystem:
        """Return a mass on a linear spring with static, Coulomb, and viscous friction.

        The mass follows `m x'' + c x' + k x + f = 0`.
        """
        mass = FrictionOscillatorElement(
            'mass',
            x=self.x,
            v=self.v,
            m=self.m,
            k=self.k,
            c=self.c,
            mu_s=self.mu_s,
            mu_k=self.mu_k,
            g=self.g,
            v_stick=self.v_stick,
        )
        return CompositeSystem(self.label, elements=[mass])
