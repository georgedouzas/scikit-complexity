"""Mechanics systems of colliding beams and friction.

The module holds the beam-beam map of colliding beams and a mass that sticks and slips under friction.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, exp, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw, UpdateLaw
from ....core.components.laws.utils import both, where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, RK4Solver
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class BeamBeamInteraction(Interaction):
    """One turn of the beam-beam map, a rotation by the angle `pi nu` followed by a kick of the opposite beam.

    With `a = cos(pi nu)`, `b = sin(pi nu)`, and `u = a x + b y`, the position and momentum of the target become
    `x = u` and `y = -b x + a y - 8 pi c (1 - exp(-u² / 2)) / u`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        nu:
            The tune, in `[0, 2]`.

        c:
            The strength of the beam-beam kick.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BeamBeamInteraction
        >>> BeamBeamInteraction('turn', target='particle', nu=0.5).nu
        0.5
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        nu: Annotated[float, RealSet.closed(0, 2)] = 0.3,
        c: Annotated[float, RealSet.closed_open(0, oo)] = 0.05,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.nu = nu
        self.c = c

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the position and the momentum of the particle after one turn.

        Args:
            context:
                The namespace of the particle, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, y = context.target.x, context.target.y
            a, b = cos(pi * context.nu), sin(pi * context.nu)
            position, momentum = a * x + b * y, -b * x + a * y
            safe = where(position == 0, 1, position)
            kick = 8 * pi * context.c * where(position == 0, 0, (1 - exp(-(safe**2) / 2)) / safe)
            self.laws_ = [UpdateLaw(x, position), UpdateLaw(y, momentum - kick)]


class StickSlipInteraction(Interaction):
    """A linear spring with static, Coulomb, and viscous friction on a mass.

    The mass follows `m x'' + c x' + k x + f = 0`, with the sliding friction `f = mu_k m g sign(x')`. While
    `|x'| <= v_stick` and `|k x| <= mu_s m g`, the static friction holds the mass at rest. The position `x` and the
    velocity `v` of the target are both states, since the position stops while the static friction holds it.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the mass.

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

    Examples:
        >>> from skcomplex.systems.physics.mechanics import StickSlipInteraction
        >>> StickSlipInteraction('friction', target='mass', mu_k=0.1).mu_k
        0.1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        k: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        c: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        mu_s: Annotated[float, RealSet.closed_open(0, oo)] = 0.06,
        mu_k: Annotated[float, RealSet.closed_open(0, oo)] = 0.05,
        g: Annotated[float, RealSet.open(0, oo)] = 1.0,
        v_stick: Annotated[float, RealSet.open(0, oo)] = 0.01,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.k = k
        self.c = c
        self.mu_s = mu_s
        self.mu_k = mu_k
        self.g = g
        self.v_stick = v_stick

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the position and the velocity, both zero while the static friction holds the mass.

        Args:
            context:
                The namespace of the mass, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            t, x, v, m = context.t, context.target.x, context.target.v, context.target.m
            force = -context.k * x - context.c * v
            moving = abs(v) > context.v_stick
            direction = where(moving, where(v > 0, 1, -1), where(force > 0, 1, -1))
            stuck = both(abs(v) <= context.v_stick, abs(context.k * x) <= context.mu_s * m * context.g)
            acceleration = where(stuck, 0, (force - context.mu_k * m * context.g * direction) / m)
            self.laws_ = [RateLaw(x.diff(t), where(moving, v, 0)), RateLaw(v.diff(t), acceleration)]


class BeamBeamMapSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BeamBeamMapSystem
        >>> BeamBeamMapSystem(nu=0.5).prepare().system_.interactions[0].nu
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.nu = nu
        self.c = c

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a particle that meets the opposite beam once every turn.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('particle', values={'x': self.x, 'y': self.y})],
            interactions=[BeamBeamInteraction('turn', target='particle', nu=self.nu, c=self.c)],
        )


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import FrictionOscillatorSystem
        >>> FrictionOscillatorSystem(mu_k=0.1).prepare().system_.interactions[0].mu_k
        0.1
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'friction', 'stick-slip', 'piecewise linear')
    references = ('Khalil 2002', 'Trzaska 2023')
    until = 40.0
    dt = 0.01

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.v = v
        self.m = m
        self.k = k
        self.c = c
        self.mu_s = mu_s
        self.mu_k = mu_k
        self.g = g
        self.v_stick = v_stick

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a mass on a spring that sticks and slips under friction.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        friction = StickSlipInteraction(
            'friction',
            target='mass',
            k=self.k,
            c=self.c,
            mu_s=self.mu_s,
            mu_k=self.mu_k,
            g=self.g,
            v_stick=self.v_stick,
        )
        self.system_ = System(
            self.label,
            elements=[Element('mass', values={'x': self.x, 'v': self.v, 'm': self.m})],
            interactions=[friction],
            solver=RK4Solver(dt=0.005),
        )
