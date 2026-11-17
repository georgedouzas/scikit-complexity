"""Nonlinear oscillator systems.

The module covers Duffing, Toda, Van der Pol, Stuart-Landau, Adler, impulses, and Hamiltonian chaos.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR, Expression, cos, exp, oo, sin, sqrt, tan
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, EventLaw, RateLaw
from ....core.components.laws.utils import derivative, periodic_guard
from ....core.components.spaces import BaseSpace
from ....core.components.systems import BaseSystem as OldBaseSystem
from ....core.components.systems import CompositeSystem
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, ScipyODESolver, Yoshida4Solver
from ....core.errors import ParameterError
from ....core.specs import ParameterSpec, Role, StateSpec, VisualSpec
from ....core.systems import BaseSystem, System
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ...physics.common.mechanics import HamiltonianInteraction

SPRINGS = ('tangent', 'linear')


class KickInteraction(Interaction):
    """Periodic kicks that change the velocity of an oscillator by a fixed amount.

    Physics names them kicks, instantaneous changes of momentum at `start + j * period` for `j` from 0. Every kick adds
    `kick` to the velocity `x'` of the state `x` of the target. A kick at the initial time of a run does not fire.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the oscillator whose velocity the kicks change.

        state:
            The name of the state whose velocity the kicks change.

        kick:
            The change of the velocity at every kick.

        period:
            The period of the kicks.

        start:
            The time of the first kick.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import KickInteraction
        >>> KickInteraction('kick', target='oscillator', kick=-1.0, period=2.0).start
        0.0
    """

    law_classes = (EventLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'x',
        kick: float = 1.0,
        period: float = 1.0,
        start: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.kick = kick
        self.period = period
        self.start = start

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the event that kicks the velocity whenever the periodic guard crosses zero.

        Args:
            context:
                The namespace of the oscillator, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            velocity = getattr(context.target, self.state).diff(context.t)
            guard = periodic_guard(context.t, self.period, self.start)
            self.laws_ = [EventLaw(guard, {velocity: velocity + context.kick}, label=self.label)]


class HenonHeilesElement(Element):
    """Star in the Hénon-Heiles potential `V = (x^2 + y^2) / 2 + x^2 y - y^3 / 3`.

    The star has the canonical momenta `px, py`.

    The coordinates and the momenta are states of order 1, which a Hamiltonian interaction moves. Below the escape
    energy `1 / 6` the motion is regular for small energies and mostly chaotic near the escape energy.

    Args:
        label:
            The label of the star.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        px:
            The initial first momentum.

        py:
            The initial second momentum.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import HenonHeilesElement
        >>> HenonHeilesElement('star').px
        0.35
    """

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None})

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.1,
        px: Annotated[float, StateSpec()] = 0.35,
        py: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.px = px
        self.py = py


class ImpulsiveOscillatorElement(Element):
    """Displacement of an oscillator that rings freely between periodic impulses.

    The impulses kick its velocity.

    A spring interaction gives the restoring force, and kick interactions change the velocity `x'` by multiples of the
    amplitude `H` at their times.

    Args:
        label:
            The label of the oscillator.

        x:
            The initial displacement.

        H:
            The amplitude of the impulses, half the jump of the velocity that one impulse of the symmetric train
            causes.

        values:
            Further values by name, such as the initial velocity `x'`, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ImpulsiveOscillatorElement
        >>> ImpulsiveOscillatorElement('oscillator').H
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: float = 0.0,
        H: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.x = x
        self.H = H


class QuadrupoleElement(Element):
    """Surface of a nucleus in the quartic quadrupole potential.

    The surface has the coordinates `q1, q2` and momenta `p1, p2`.

    The quadrupole deformation of the nuclear surface moves under the Hamiltonian
    `H = (q² + p²) / 2 + B q1 (3 q2² - q1²) / √2 + q⁴ / 8`, with `q² = q1² + q2²`, which `QuadrupoleMotionInteraction`
    turns into the equations of motion. The cubic term of strength `B` breaks the rotational symmetry and makes the
    motion chaotic at intermediate energies, while the quartic term keeps it bounded at every energy.

    Args:
        label:
            The label of the nucleus.

        q1:
            The initial first coordinate.

        q2:
            The initial second coordinate.

        p1:
            The initial first momentum.

        p2:
            The initial second momentum.

        B:
            The strength of the cubic term.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import QuadrupoleElement
        >>> QuadrupoleElement('nucleus').B
        0.5
    """

    visual = VisualSpec('bodies', {'positions': ('q1', 'q2'), 'trail': None})

    def __init__(
        self,
        label: str,
        *,
        q1: float = 0.0,
        q2: float = 1.0,
        p1: float = 0.5,
        p2: float = 0.0,
        B: float = 0.5,
    ) -> None:
        super().__init__(label)
        self.q1 = q1
        self.q2 = q2
        self.p1 = p1
        self.p2 = p2
        self.B = B


class VanDerPolElement(Element):
    """Van der Pol oscillator `x'' = mu (1 - x^2) x' - x + A cos(omega t)`.

    The periodic forcing is optional.

    The damping is negative for small amplitudes and positive for large ones, so the unforced oscillator, `A = 0`,
    settles on a limit cycle of amplitude near 2. A forcing of amplitude `A` and frequency `omega` locks it or drives it
    quasiperiodically or chaotically.

    Args:
        label:
            The label of the oscillator.

        x:
            The initial displacement.

        x_d1:
            The initial velocity.

        mu:
            The strength of the nonlinear damping.

        A:
            The amplitude of the forcing.

        omega:
            The frequency of the forcing.

    Attributes:
        default_visual_:
            How the oscillator is drawn by default, the `visual` of its class.

    Examples:
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.physics.nonlinear_oscillators import VanDerPolElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> system = CompositeSystem(elements=[VanDerPolElement('oscillator', x=0.1)])
        >>> trajectory = simulate(system, until=50.0, dt=0.1)
        >>> round(float(abs(trajectory.get_values('oscillator.x')[-300:]).max()), 1)
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, StateSpec(order=2)] = 2.0,
        x_d1: float = 0.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        A: Annotated[float, ParameterSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1
        self.mu = mu
        self.A = A
        self.omega = omega

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the acceleration of the oscillator.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of the second derivative of `x`.
        """
        x = namespace.x
        damping = namespace.mu * (1 - x**2) * derivative(x, 1)
        return [RateLaw(x, damping - x + namespace.A * cos(namespace.omega * namespace.t))]


class DuffingSystem(DynamicalSystem):
    """The forced Duffing oscillator, chaotic here.

    The oscillator is `x'' = -delta x' - alpha x - beta x^3 + gamma cos(omega t)`.

    The system has the displacement `duffing.x` of order 2 and its velocity `duffing.x'`.

    Args:
        x:
            The initial displacement.

        v:
            The initial velocity.

        delta:
            The damping.

        alpha:
            The linear stiffness, negative for a double well.

        beta:
            The cubic stiffness.

        gamma:
            The amplitude of the forcing.

        omega:
            The angular frequency of the forcing.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import DuffingSystem
        >>> DuffingSystem(gamma=0.3).prepare().system_.elements[0].values['gamma']
        0.3
    """

    field = 'physics/nonlinear_oscillators'
    topics = ('chaos', 'ode', 'second order', 'forcing')
    references = ('Duffing 1918', 'Trzaska 2023')
    until = 200.0
    dt = 0.05

    def __init__(
        self,
        x: float = 1.0,
        v: float = 0.0,
        *,
        delta: float = 0.3,
        alpha: float = -1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        omega: float = 1.2,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.x = x
        self.v = v
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'duffing'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'x': self.x,
            "x'": self.v,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'delta': self.delta,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'omega': self.omega,
        }

    def _get_equations(self) -> Equations:
        """Return the components of the system, built from its model parameters."""
        return {
            "x''": lambda s: (
                -s.target.delta * s.target.x.diff(s.t)
                - s.target.alpha * s.target.x
                - s.target.beta * s.target.x**3
                + s.target.gamma * cos(s.target.omega * s.t)
            ),
        }


class VanDerPolSystem(OldBaseSystem):
    """The Van der Pol oscillator `x'' = μ (1 - x²) x' - x`.

    The oscillator settles on a limit cycle for `μ > 0`.

    The system has the displacement `oscillator.x` of order 2 and its velocity `oscillator.x_d1`.

    Args:
        x:
            The initial displacement.

        v:
            The initial velocity.

        mu:
            The strength of the nonlinear damping.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import VanDerPolSystem
        >>> VanDerPolSystem(mu=3.0).prepare().system_.elements[0].mu
        3.0
    """

    field = 'physics/nonlinear_oscillators'
    topics = ('limit cycle', 'ode', 'second order', 'oscillation')
    references = ('van der Pol 1926', 'Gros 2015')
    until = 50.0
    dt = 0.05

    def __init__(
        self,
        x: float = 2.0,
        v: float = 0.0,
        *,
        mu: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.mu = mu

    def _build_system(self) -> CompositeSystem:
        """Return the Van der Pol oscillator `x'' = μ (1 - x²) x' - x`.

        The oscillator settles on a limit cycle for `μ > 0`.
        """
        return CompositeSystem(self.label, elements=[VanDerPolElement('oscillator', x=self.x, x_d1=self.v, mu=self.mu)])


def _get_henon_heiles_energy(star: SimpleNamespace) -> Expression:
    """Return the Hénon-Heiles energy of a star.

    The energy is `(px² + py²) / 2 + (x² + y²) / 2 + x² y - y³ / 3`.
    """
    kinetic = (star.px**2 + star.py**2) / 2
    return kinetic + (star.x**2 + star.y**2) / 2 + star.x**2 * star.y - star.y**3 / 3


class HenonHeilesSystem(OldBaseSystem):
    """The Hénon-Heiles system of a star in a galactic potential.

    The motion is chaotic at higher energies.

    The star moves under the Hamiltonian `H = (px² + py²) / 2 + (x² + y²) / 2 + x² y - y³ / 3`, which is conserved. The
    system has the coordinates `star.x` and `star.y` and their momenta `star.px` and `star.py`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        px:
            The initial first momentum.

        py:
            The initial second momentum.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import HenonHeilesSystem
        >>> HenonHeilesSystem(px=0.3).prepare().system_.elements[0].px
        0.3
    """

    field = 'physics/nonlinear_oscillators'
    topics = ('chaos', 'ode', 'hamiltonian')
    references = ('Hénon and Heiles 1964', 'Datseris and Parlitz 2022')
    until = 200.0
    dt = 0.05
    solver = Yoshida4Solver(dt=0.01)
    invariants: ClassVar[Mapping[str, float]] = {'star.energy': 1e-6}
    observables = (Observable('energy', _get_henon_heiles_energy, element='star'),)

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.1,
        px: float = 0.35,
        py: float = 0.0,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.px = px
        self.py = py

    @staticmethod
    def _get_henon_heiles_hamiltonian(s: SimpleNamespace) -> Expression:
        """Return the Hénon-Heiles Hamiltonian of the target."""
        return _get_henon_heiles_energy(s.target)

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        return CompositeSystem(
            self.label,
            elements=[HenonHeilesElement('star', x=self.x, y=self.y, px=self.px, py=self.py)],
            interactions=[
                HamiltonianInteraction(
                    'motion',
                    target='star',
                    coordinates=('x', 'y'),
                    momenta=('px', 'py'),
                    hamiltonian=self._get_henon_heiles_hamiltonian,
                ),
            ],
        )


class StuartLandauInteraction(Interaction):
    """Stuart-Landau dynamics `z' = (mu + i omega) z - (1 + i beta) |z|² z` of a complex amplitude.

    The amplitude `z = x + i y` of the target is written as two real values `x` and `y`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the oscillator.

        mu:
            The distance from the Hopf bifurcation.

        omega:
            The linear frequency.

        beta:
            The nonlinear frequency shift.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import StuartLandauInteraction
        >>> StuartLandauInteraction('hopf', target='oscillator', beta=0.5).beta
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        mu: float = 1.0,
        omega: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu
        self.omega = omega
        self.beta = beta

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the real and imaginary parts of the amplitude.

        Args:
            context:
                The namespace of the oscillator, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, y, t = context.target.x, context.target.y, context.t
            mu, omega, beta = context.mu, context.omega, context.beta
            saturation = x**2 + y**2
            self.laws_ = [
                RateLaw(x.diff(t), mu * x - omega * y - saturation * (x - beta * y)),
                RateLaw(y.diff(t), mu * y + omega * x - saturation * (y + beta * x)),
            ]


class StuartLandauSystem(BaseSystem):
    """The Stuart-Landau oscillator `z' = (μ + i ω) z - (1 + i β) |z|² z`.

    It is the normal form of a Hopf bifurcation.

    The complex amplitude `z = x + i y` is written as two real states. For `μ > 0` every orbit except the origin settles
    on the circle of radius `sqrt(μ)`, turning at the frequency `ω - β μ`. The system has the real part `oscillator.x`
    and the imaginary part `oscillator.y` of the amplitude.

    Args:
        x:
            The initial real part of the amplitude.

        y:
            The initial imaginary part of the amplitude.

        mu:
            The distance from the Hopf bifurcation.

        omega:
            The linear frequency.

        beta:
            The nonlinear frequency shift.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import StuartLandauSystem
        >>> StuartLandauSystem(mu=0.5).prepare().system_.interactions[0].mu
        0.5
    """

    field = 'physics/nonlinear_oscillators'
    topics = ('limit cycle', 'ode', 'hopf bifurcation', 'oscillation')
    references = ('Gros 2015', 'Dilão 2023')
    until = 30.0
    dt = 0.05

    def __init__(
        self,
        x: float = 0.1,
        y: float = 0.0,
        *,
        mu: float = 1.0,
        omega: float = 1.0,
        beta: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.mu = mu
        self.omega = omega
        self.beta = beta

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the normal form of a Hopf bifurcation.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        oscillator = Element('oscillator', values={'x': self.x, 'y': self.y})
        hopf = StuartLandauInteraction('hopf', target='oscillator', mu=self.mu, omega=self.omega, beta=self.beta)
        self.system_ = System(self.label, elements=[oscillator], interactions=[hopf])


class ForcedVanDerPolInteraction(Interaction):
    """Van der Pol dynamics `x'' = mu (1 - x²) x' - x + A cos(omega t)` of an oscillator driven by a periodic force.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the oscillator.

        mu:
            The strength of the nonlinear damping.

        A:
            The amplitude of the force.

        omega:
            The angular frequency of the force.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ForcedVanDerPolInteraction
        >>> ForcedVanDerPolInteraction('drive', target='oscillator', A=0.5).A
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        mu: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        A: float = 1.0,
        omega: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu
        self.A = A
        self.omega = omega

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the oscillator.

        Args:
            context:
                The namespace of the oscillator, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            damping = context.mu * (1 - x**2) * x.diff(t)
            self.laws_ = [RateLaw(x.diff(t, 2), damping - x + context.A * cos(context.omega * t))]


class ForcedVanDerPolSystem(BaseSystem):
    """The Van der Pol oscillator driven by a periodic force.

    The oscillator follows `x'' = μ (1 - x²) x' - x + A cos(ω t)`.

    A strong enough force entrains the limit cycle, so the oscillator locks to the driving frequency inside Arnold
    tongues, and the motion can be quasiperiodic or chaotic outside them. The system has the displacement `oscillator.x`
    and its velocity `oscillator.x'`.

    Args:
        x:
            The initial displacement.

        v:
            The initial velocity.

        mu:
            The strength of the nonlinear damping.

        A:
            The amplitude of the force.

        omega:
            The angular frequency of the force.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ForcedVanDerPolSystem
        >>> ForcedVanDerPolSystem(A=0.5).prepare().system_.interactions[0].A
        0.5
    """

    field = 'physics/nonlinear_oscillators'
    topics = ('limit cycle', 'ode', 'second order', 'forcing', 'synchronization')
    references = ('van der Pol and van der Mark 1927', 'Trzaska 2023')
    until = 100.0
    dt = 0.05

    def __init__(
        self,
        x: float = 2.0,
        v: float = 0.0,
        *,
        mu: float = 1.0,
        A: float = 1.0,
        omega: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.v = v
        self.mu = mu
        self.A = A
        self.omega = omega

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Van der Pol oscillator driven by a periodic force.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        oscillator = Element('oscillator', values={'x': self.x, "x'": self.v})
        drive = ForcedVanDerPolInteraction('drive', target='oscillator', mu=self.mu, A=self.A, omega=self.omega)
        self.system_ = System(self.label, elements=[oscillator], interactions=[drive])


class AdlerInteraction(Interaction):
    """Adler dynamics `phi' = detuning + epsilon sin(phi)` of the phase difference of an oscillator and its drive.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the phase difference, whose value is `phi`.

        detuning:
            The frequency difference `Δω`.

        epsilon:
            The strength of the coupling.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import AdlerInteraction
        >>> AdlerInteraction('locking', target='difference', detuning=2.0).detuning
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, detuning: float = 0.5, epsilon: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.detuning = detuning
        self.epsilon = epsilon

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the phase difference.

        Args:
            context:
                The namespace of the phase difference, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            phi = context.target.phi
            self.laws_ = [RateLaw(phi.diff(context.t), context.detuning + context.epsilon * sin(phi))]


class AdlerEquationSystem(BaseSystem):
    """The Adler equation `Δφ' = Δω + ε sin Δφ`.

    The equation is that of the phase difference of an oscillator and its drive.

    The phase difference locks at a fixed point when `|Δω| < ε`, inside the Arnold tongue. Outside it, the phase slips
    by `2π` every `2π / sqrt(Δω² - ε²)`. The system has the phase difference `difference.phi`.

    Args:
        phi:
            The initial phase difference.

        detuning:
            The frequency difference `Δω`.

        epsilon:
            The strength of the coupling.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import AdlerEquationSystem
        >>> AdlerEquationSystem(detuning=2.0).prepare().system_.interactions[0].detuning
        2.0
    """

    field = 'physics/nonlinear_oscillators'
    topics = ('synchronization', 'ode', 'phase', 'oscillation')
    references = ('Adler 1946', 'Datseris and Parlitz 2022')
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        phi: float = 0.0,
        *,
        detuning: float = 0.5,
        epsilon: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.phi = phi
        self.detuning = detuning
        self.epsilon = epsilon

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the phase difference of an oscillator and its drive.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        difference = Element('difference', values={'phi': self.phi})
        locking = AdlerInteraction('locking', target='difference', detuning=self.detuning, epsilon=self.epsilon)
        self.system_ = System(self.label, elements=[difference], interactions=[locking])


def _get_quadrupole_energy(nucleus: SimpleNamespace) -> Expression:
    """Return the quadrupole energy of a nucleus.

    The energy is `(q² + p²) / 2 + B q1 (3 q2² - q1²) / √2 + q⁴ / 8`.
    """
    q2 = nucleus.q1**2 + nucleus.q2**2
    kinetic = (nucleus.p1**2 + nucleus.p2**2) / 2
    cubic = nucleus.B * nucleus.q1 * (3 * nucleus.q2**2 - nucleus.q1**2) / sqrt(2)
    return kinetic + q2 / 2 + cubic + q2**2 / 8


class QuadrupoleMotionInteraction(Interaction):
    """Motion of the surface of a nucleus in the quartic quadrupole potential.

    The coordinates `(q1, q2)` and the momenta `(p1, p2)` of the target follow Hamilton's equations `q' = dH/dp` and
    `p' = -dH/dq` of `H = (q² + p²) / 2 + B q1 (3 q2² - q1²) / √2 + q⁴ / 8`, with the strength `B` of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the nucleus.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import QuadrupoleMotionInteraction
        >>> QuadrupoleMotionInteraction('motion', target='nucleus').target
        'nucleus'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the coordinates and the momenta of the nucleus.

        Args:
            context:
                The namespace of the nucleus and the time `t`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            nucleus = context.target
            names = ('q1', 'q2', 'p1', 'p2')
            values = [getattr(nucleus, name) for name in names]
            symbols = [SR.symbol(f'_{name}') for name in names]
            q1, q2, p1, p2 = symbols
            hamiltonian = _get_quadrupole_energy(SimpleNamespace(q1=q1, q2=q2, p1=p1, p2=p2, B=nucleus.B))
            back = dict(zip(symbols, values, strict=True))
            rates = [hamiltonian.diff(p1), hamiltonian.diff(p2), -hamiltonian.diff(q1), -hamiltonian.diff(q2)]
            self.laws_ = [
                RateLaw(value.diff(context.t), rate.subs(back)) for value, rate in zip(values, rates, strict=True)
            ]


class NuclearQuadrupoleSystem(BaseSystem):
    """The quadrupole vibrations of a nuclear surface.

    The vibrations are a Hamiltonian system of two degrees of freedom.

    The deformation coordinates `q = (q1, q2)` and their momenta `p` move under the Hamiltonian `H = (q² + p²) / 2 + B
    q1 (3 q2² - q1²) / √2 + q⁴ / 8`, which is conserved. The cubic term has the threefold symmetry of the Hénon-Heiles
    potential, and the quartic term keeps every orbit bounded, so the motion is regular at low and high energies and
    largely chaotic in between. The system has the coordinates `nucleus.q1` and `nucleus.q2`, their momenta `nucleus.p1`
    and `nucleus.p2`, and the observable `nucleus.energy`.

    Args:
        q1:
            The initial first coordinate.

        q2:
            The initial second coordinate.

        p1:
            The initial first momentum.

        p2:
            The initial second momentum.

        B:
            The strength of the cubic term.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import NuclearQuadrupoleSystem
        >>> NuclearQuadrupoleSystem(B=0.3).prepare().system_.elements[0].B
        0.3
    """

    field = 'physics/nonlinear_oscillators'
    topics = ('chaos', 'ode', 'hamiltonian', 'nuclear physics')
    references = ('Baran and Raduta 1998', 'Datseris and Parlitz 2022')
    until = 200.0
    dt = 0.05
    invariants: ClassVar[Mapping[str, float]] = {'nucleus.energy': 1e-6}
    observables = (Observable('energy', _get_quadrupole_energy, element='nucleus'),)

    def __init__(
        self,
        q1: float = 0.0,
        q2: float = 1.0,
        p1: float = 0.5,
        p2: float = 0.0,
        *,
        B: float = 0.5,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.q1 = q1
        self.q2 = q2
        self.p1 = p1
        self.p2 = p2
        self.B = B

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the quadrupole vibrations of a nuclear surface.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        nucleus = QuadrupoleElement('nucleus', q1=self.q1, q2=self.q2, p1=self.p1, p2=self.p2, B=self.B)
        motion = QuadrupoleMotionInteraction('motion', target='nucleus')
        self.system_ = System(self.label, elements=[nucleus], interactions=[motion], solver=Yoshida4Solver(dt=0.01))


class TodaInteraction(Interaction):
    """Forced Toda dynamics `x'' = -d x' - (e^x - 1) + a sin(omega t)` of a passive asymmetric oscillator.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the oscillator.

        d:
            The damping.

        a:
            The amplitude of the forcing.

        omega:
            The angular frequency of the forcing.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import TodaInteraction
        >>> TodaInteraction('spring', target='toda', a=1.2).a
        1.2
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        d: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        a: float = 0.88,
        omega: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.d = d
        self.a = a
        self.omega = omega

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the oscillator.

        Args:
            context:
                The namespace of the oscillator, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            rate = -context.d * x.diff(t) - (exp(x) - 1) + context.a * sin(context.omega * t)
            self.laws_ = [RateLaw(x.diff(t, 2), rate)]


class ForcedTodaSystem(BaseSystem):
    """The forced Toda oscillator `x'' + d x' + e^x - 1 = a sin(omega t)`.

    It is a passive asymmetric oscillator.

    The exponential spring is stiff for `x > 0` and soft for `x < 0`, unlike the symmetric spring of the Duffing
    oscillator. Its free frequency falls from 1 as the amplitude grows, so the resonance curve bends to low driving
    frequencies and the response swings far to the soft side. Stronger driving gives coexisting responses and period
    doubling cascades to chaos as the driving period `2π / ω` changes. The system has the displacement `toda.x` and its
    velocity `toda.x'`.

    Args:
        x:
            The initial displacement.

        v:
            The initial velocity.

        d:
            The damping.

        a:
            The amplitude of the forcing.

        omega:
            The angular frequency of the forcing.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ForcedTodaSystem
        >>> ForcedTodaSystem(a=1.2).prepare().system_.interactions[0].a
        1.2
    """

    field = 'physics/nonlinear_oscillators'
    topics = ('chaos', 'ode', 'second order', 'forcing', 'resonance')
    references = ('Toda 1967', 'Kurz and Lauterborn 1988', 'Datseris and Parlitz 2022')
    until = 300.0
    dt = 0.05

    def __init__(
        self,
        x: float = 0.0,
        v: float = 0.0,
        *,
        d: float = 0.1,
        a: float = 0.88,
        omega: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.v = v
        self.d = d
        self.a = a
        self.omega = omega

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the forced Toda oscillator.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        toda = Element('toda', values={'x': self.x, "x'": self.v})
        spring = TodaInteraction('spring', target='toda', d=self.d, a=self.a, omega=self.omega)
        self.system_ = System(self.label, elements=[toda], interactions=[spring])


class ImpulsiveSpringInteraction(Interaction):
    """Restoring force of an impulsive oscillator, the stiff tangent spring or a linear one.

    The tangent spring gives `x'' = -tan(x) / cos²(x)` and the linear spring `x'' = -omega0² x`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the oscillator.

        spring:
            The restoring force, `'tangent'` for `tan(x) / cos²(x)` or `'linear'` for `omega0² x`.

        omega0:
            The natural frequency of the linear spring.

    Raises:
        ParameterError:
            The spring is unknown.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ImpulsiveSpringInteraction
        >>> ImpulsiveSpringInteraction('spring', target='oscillator', spring='linear').spring
        'linear'
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        spring: str = 'tangent',
        omega0: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.spring = spring
        self.omega0 = omega0

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the oscillator.

        Args:
            context:
                The namespace of the oscillator, the time `t`, and the parameters.

        Raises:
            ParameterError:
                The spring is unknown.
        """
        super()._prepare(context)
        if self.spring not in SPRINGS:
            error_msg = f'Unknown spring {self.spring!r}. Use one of {", ".join(SPRINGS)}.'
            raise ParameterError(error_msg)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            rate = -tan(x) / cos(x) ** 2 if self.spring == 'tangent' else -(context.omega0**2) * x
            self.laws_ = [RateLaw(x.diff(context.t, 2), rate)]


class ImpulsiveOscillatorSystem(BaseSystem):
    """An oscillator kicked by a periodic train of impulses of alternating sign.

    The stiff spring gives `x'' + tan(x) / cos²(x) = 2 H Σ_k (-1)^k δ(t - (2 k - 1) a)`, a strongly nonlinear
    oscillator whose free motion `sin x = sin A sin(t / cos A)` of amplitude `A < π / 2` is known exactly. Between the
    impulses the oscillator rings freely, and every impulse is an event that kicks its velocity, by `-2 H` at the times
    `(4 k + 1) a` and by `2 H` at the times `(4 k + 3) a`. A motion of amplitude `A` is periodic with the period `4 a`
    of the impulses when an impulse reverses its velocity, which gives
    `a = q arccos(H q² / sqrt((1 - q²) (1 - q² H²)))` with `q = cos A`, for `arctan H < A < π / 2`. The defaults start
    the orbit of amplitude `A = 1.2` for `H = 2`, from `x = 0` with the speed `tan A`.

    The temporal dipole shift `gamma` in `(-1, 1)` replaces the rectangular wave whose derivative gives the impulses
    with the derivative of an asymmetric saw-tooth wave: the negative impulses move to `(4 k + 1 - gamma) a`, the
    positive ones to `(4 k + 3 + gamma) a`, and both grow to `2 H / (1 - gamma²)`. The linear spring gives the harmonic
    oscillator `x'' + omega0² x` under the same impulses. The system has the displacement `oscillator.x` and its
    velocity `oscillator.x'`, kicked by the impulses `pull` and `push`.

    Args:
        x:
            The initial displacement.

        v:
            The initial velocity.

        H:
            The amplitude of the impulses.

        a:
            A quarter of the period of the impulses.

        gamma:
            The temporal dipole shift of the impulses, in `(-1, 1)`.

        spring:
            The restoring force, `'tangent'` for `tan(x) / cos²(x)` or `'linear'` for `omega0² x`.

        omega0:
            The natural frequency of the linear spring.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The spring is unknown, the quarter period is not positive, or the shift is outside `(-1, 1)`.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ImpulsiveOscillatorSystem
        >>> ImpulsiveOscillatorSystem(a=0.5, gamma=0.5).prepare().system_.interactions[1].start
        0.25
    """

    field = 'physics/nonlinear_oscillators'
    topics = ('ode', 'second order', 'impulses', 'events', 'forcing')
    references = ('Pilipchuk 2010', 'Trzaska 2023')
    until = 10.0
    dt = 0.01

    def __init__(
        self,
        x: float = 0.0,
        v: float = 2.572,
        *,
        H: float = 2.0,
        a: float = 0.4165,
        gamma: float = 0.0,
        spring: str = 'tangent',
        omega0: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.v = v
        self.H = H
        self.a = a
        self.gamma = gamma
        self.spring = spring
        self.omega0 = omega0

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of an oscillator kicked by a periodic train of impulses of alternating sign.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ParameterError:
                The spring is unknown, the quarter period is not positive, or the shift is outside `(-1, 1)`.
        """
        super()._prepare(context)
        if self.spring not in SPRINGS:
            error_msg = f'Unknown spring {self.spring!r}. Use one of {", ".join(SPRINGS)}.'
            raise ParameterError(error_msg)
        if not self.a > 0 or not -1 < self.gamma < 1:
            error_msg = (
                f'The impulses need a positive quarter period and a shift in (-1, 1), but got {self.a!r} and '
                f'{self.gamma!r}.'
            )
            raise ParameterError(error_msg)
        kick = 2 / (1 - self.gamma**2) * self.H
        oscillator = ImpulsiveOscillatorElement('oscillator', x=self.x, H=self.H, values={"x'": self.v})
        restoring = ImpulsiveSpringInteraction('spring', target='oscillator', spring=self.spring, omega0=self.omega0)
        pull = KickInteraction(
            'pull',
            target='oscillator',
            kick=-kick,
            period=4 * self.a,
            start=(1 - self.gamma) * self.a,
        )
        push = KickInteraction(
            'push',
            target='oscillator',
            kick=kick,
            period=4 * self.a,
            start=(3 + self.gamma) * self.a,
        )
        solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)
        self.system_ = System(
            self.label,
            elements=[oscillator],
            interactions=[restoring, pull, push],
            solver=solver,
        )
