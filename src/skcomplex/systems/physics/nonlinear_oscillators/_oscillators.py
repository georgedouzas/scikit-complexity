"""Nonlinear oscillator systems.

The module covers Duffing, Toda, Van der Pol, Stuart-Landau, Adler, impulses, and Hamiltonian chaos.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import Expression, cos, exp, oo, sin, sqrt, tan
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import HamiltonianInteraction, ImpulseInteraction, RateInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import derivative
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import ScipyODESolver, Yoshida4Solver
from ....core.errors import ParameterError
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..common.oscillators import StuartLandauElement


class AdlerPhaseElement(Element):
    """Phase difference of a forced oscillator, the Adler equation.

    The phase difference follows `phi' = detuning + epsilon sin(phi)`.

    The phase locks at a fixed point when `|detuning| <= epsilon`, and slips with the period
    `2 pi / sqrt(detuning^2 - epsilon^2)` otherwise.

    Args:
        label:
            The label of the phase difference.

        visual:
            How the phase difference is drawn, or `None` for `default_visual_`.

        phi:
            The initial phase difference.

        detuning:
            The detuning of the natural frequency from the forcing.

        epsilon:
            The strength of the forcing.

    Attributes:
        default_visual_:
            The visual of the phase difference when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import AdlerPhaseElement
        >>> AdlerPhaseElement('difference').detuning
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec()] = 0.0,
        detuning: Annotated[float, ParameterSpec()] = 0.5,
        epsilon: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.detuning = detuning
        self.epsilon = epsilon

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the phase difference.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of `phi`.
        """
        return [RateLaw(namespace.phi, namespace.detuning + namespace.epsilon * sin(namespace.phi))]


class DuffingElement(Element):
    """Displacement and parameters of the forced Duffing oscillator.

    A `RateInteraction` sets its acceleration.

    The acceleration is `x'' = -delta x' - alpha x - beta x^3 + gamma cos(omega t)`.

    For `alpha < 0 < beta` the potential has two wells, and a strong enough forcing drives the oscillator chaotically
    between them.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        x_d1:
            The initial velocity.

        delta:
            The damping.

        alpha:
            The linear stiffness.

        beta:
            The cubic stiffness.

        gamma:
            The amplitude of the forcing.

        omega:
            The frequency of the forcing.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import DuffingElement
        >>> DuffingElement('duffing').alpha
        -1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 1.0,
        x_d1: float = 0.0,
        delta: Annotated[float, ParameterSpec()] = 0.3,
        alpha: Annotated[float, ParameterSpec()] = -1.0,
        beta: Annotated[float, ParameterSpec()] = 1.0,
        gamma: Annotated[float, ParameterSpec()] = 0.5,
        omega: Annotated[float, ParameterSpec()] = 1.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega


class HenonHeilesElement(Element):
    """Star in the Hénon-Heiles potential `V = (x^2 + y^2) / 2 + x^2 y - y^3 / 3`.

    The star has the canonical momenta `px, py`.

    The coordinates and the momenta are states of order 1, which a Hamiltonian interaction moves. Below the escape
    energy `1 / 6` the motion is regular for small energies and mostly chaotic near the escape energy.

    Args:
        label:
            The label of the star.

        visual:
            How the star is drawn.

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

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec("bodies", {"positions": ("x", "y"), "trail": None}),
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.1,
        px: Annotated[float, StateSpec()] = 0.35,
        py: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.px = px
        self.py = py


class ImpulsiveOscillatorElement(Element):
    """Displacement of an oscillator that rings freely between periodic impulses.

    The impulses kick its velocity.

    A rate interaction gives the restoring force, and impulse interactions kick the velocity `x_d1` by multiples of the
    amplitude `H` at their times, so the element holds the states and the parameters they read.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        x_d1:
            The initial velocity.

        H:
            The amplitude of the impulses, half the jump of the velocity that one impulse of the symmetric train
            causes.

        omega0:
            The natural frequency of a linear spring.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ImpulsiveOscillatorElement
        >>> ImpulsiveOscillatorElement('oscillator').H
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        H: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        omega0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.H = H
        self.omega0 = omega0


class QuadrupoleElement(Element):
    """Surface of a nucleus in the quartic quadrupole potential.

    The surface has the coordinates `q1, q2` and momenta `p1, p2`.

    The quadrupole deformation of the nuclear surface moves under the Hamiltonian
    `H = (q² + p²) / 2 + B q1 (3 q2² - q1²) / √2 + q⁴ / 8`, with `q² = q1² + q2²`, which a Hamiltonian interaction
    turns into the equations of motion. The cubic term of strength `B` breaks the rotational symmetry and makes the
    motion chaotic at intermediate energies, while the quartic term keeps it bounded at every energy.

    Args:
        label:
            The label of the nucleus.

        visual:
            How the nucleus is drawn.

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

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec("bodies", {"positions": ("q1", "q2"), "trail": None}),
        *,
        q1: Annotated[float, StateSpec()] = 0.0,
        q2: Annotated[float, StateSpec()] = 1.0,
        p1: Annotated[float, StateSpec()] = 0.5,
        p2: Annotated[float, StateSpec()] = 0.0,
        B: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.q1 = q1
        self.q2 = q2
        self.p1 = p1
        self.p2 = p2
        self.B = B


class TodaElement(Element):
    """Displacement and parameters of the forced Toda oscillator.

    Its exponential spring is stiff on one side only.

    The acceleration is `x'' = -d x' - (e^x - 1) + a sin(omega t)`. The restoring force `e^x - 1` grows exponentially
    for `x > 0` and saturates at `-1` for `x < 0`, so the oscillator is passive and asymmetric, and its resonance curve
    bends towards low frequencies.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        x_d1:
            The initial velocity.

        d:
            The damping.

        a:
            The amplitude of the forcing.

        omega:
            The frequency of the forcing.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import TodaElement
        >>> TodaElement('toda').d
        0.1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        a: Annotated[float, ParameterSpec()] = 0.88,
        omega: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.d = d
        self.a = a
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
        s = namespace
        return [RateLaw(s.x, -s.d * derivative(s.x, 1) - (exp(s.x) - 1) + s.a * sin(s.omega * s.t))]


class VanDerPolElement(Element):
    """Van der Pol oscillator `x'' = mu (1 - x^2) x' - x + A cos(omega t)`.

    The periodic forcing is optional.

    The damping is negative for small amplitudes and positive for large ones, so the unforced oscillator, `A = 0`,
    settles on a limit cycle of amplitude near 2. A forcing of amplitude `A` and frequency `omega` locks it or drives it
    quasiperiodically or chaotically.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

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
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.core.components.systems import System
        >>> from skcomplex.systems.physics.nonlinear_oscillators import VanDerPolElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> system = System(elements=[VanDerPolElement('oscillator', x=0.1)])
        >>> trajectory = simulate(system, until=50.0, dt=0.1)
        >>> round(float(abs(trajectory.values('oscillator.x')[-300:]).max()), 1)
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 2.0,
        x_d1: float = 0.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        A: Annotated[float, ParameterSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
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


def _get_henon_heiles_energy(star: SimpleNamespace) -> Expression:
    """Return the Hénon-Heiles energy of a star.

    The energy is `(px² + py²) / 2 + (x² + y²) / 2 + x² y - y³ / 3`.
    """
    kinetic = (star.px**2 + star.py**2) / 2
    return kinetic + (star.x**2 + star.y**2) / 2 + star.x**2 * star.y - star.y**3 / 3


SPRINGS = ("tangent", "linear")


def _get_quadrupole_energy(nucleus: SimpleNamespace) -> Expression:
    """Return the quadrupole energy of a nucleus.

    The energy is `(q² + p²) / 2 + B q1 (3 q2² - q1²) / √2 + q⁴ / 8`.
    """
    q2 = nucleus.q1**2 + nucleus.q2**2
    kinetic = (nucleus.p1**2 + nucleus.p2**2) / 2
    cubic = nucleus.B * nucleus.q1 * (3 * nucleus.q2**2 - nucleus.q1**2) / sqrt(2)
    return kinetic + q2 / 2 + cubic + q2**2 / 8


class DuffingSystem(BaseSystem):
    """The forced Duffing oscillator, chaotic here.

    The oscillator is `x'' = -delta x' - alpha x - beta x^3 + gamma cos(omega t)`.

    The system has the displacement `duffing.x` of order 2 and its velocity `duffing.x_d1`.

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
        >>> DuffingSystem(gamma=0.3).prepare().system_.elements[0].gamma
        0.3
    """

    field = "physics/nonlinear_oscillators"
    topics = ("chaos", "ode", "second order", "forcing")
    references = ("Duffing 1918", "Trzaska 2023")
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
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        acceleration = RateInteraction(
            "acceleration",
            target="duffing",
            state="x",
            rate=lambda s: (
                -s.target.delta * derivative(s.target.x, 1)
                - s.target.alpha * s.target.x
                - s.target.beta * s.target.x**3
                + s.target.gamma * cos(s.target.omega * s.t)
            ),
        )
        return System(
            self.label,
            elements=[
                DuffingElement(
                    "duffing",
                    x=self.x,
                    x_d1=self.v,
                    delta=self.delta,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                    omega=self.omega,
                ),
            ],
            interactions=[acceleration],
        )


class VanDerPolSystem(BaseSystem):
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

    field = "physics/nonlinear_oscillators"
    topics = ("limit cycle", "ode", "second order", "oscillation")
    references = ("van der Pol 1926", "Gros 2015")
    until = 50.0
    dt = 0.05

    def __init__(
        self,
        x: float = 2.0,
        v: float = 0.0,
        *,
        mu: float = 1.0,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.mu = mu

    def _build_system(self) -> System:
        """Return the Van der Pol oscillator `x'' = μ (1 - x²) x' - x`.

        The oscillator settles on a limit cycle for `μ > 0`.
        """
        return System(self.label, elements=[VanDerPolElement("oscillator", x=self.x, x_d1=self.v, mu=self.mu)])


class HenonHeilesSystem(BaseSystem):
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

    field = "physics/nonlinear_oscillators"
    topics = ("chaos", "ode", "hamiltonian")
    references = ("Hénon and Heiles 1964", "Datseris and Parlitz 2022")
    until = 200.0
    dt = 0.05
    solver = Yoshida4Solver(dt=0.01)
    invariants: ClassVar[Mapping[str, float]] = {"star.energy": 1e-6}
    observables = (ObservableSpec("energy", _get_henon_heiles_energy, element="star"),)

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.1,
        px: float = 0.35,
        py: float = 0.0,
        *,
        label: str = "system",
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

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        return System(
            self.label,
            elements=[HenonHeilesElement("star", x=self.x, y=self.y, px=self.px, py=self.py)],
            interactions=[
                HamiltonianInteraction(
                    "motion",
                    target="star",
                    coordinates=("x", "y"),
                    momenta=("px", "py"),
                    hamiltonian=self._get_henon_heiles_hamiltonian,
                ),
            ],
        )


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

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import StuartLandauSystem
        >>> StuartLandauSystem(mu=0.5).prepare().system_.elements[0].mu
        0.5
    """

    field = "physics/nonlinear_oscillators"
    topics = ("limit cycle", "ode", "hopf bifurcation", "oscillation")
    references = ("Gros 2015", "Dilão 2023")
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
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.mu = mu
        self.omega = omega
        self.beta = beta

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        return System(
            self.label,
            elements=[
                StuartLandauElement("oscillator", x=self.x, y=self.y, mu=self.mu, omega=self.omega, beta=self.beta),
            ],
        )


class ForcedVanDerPolSystem(BaseSystem):
    """The Van der Pol oscillator driven by a periodic force.

    The oscillator follows `x'' = μ (1 - x²) x' - x + A cos(ω t)`.

    A strong enough force entrains the limit cycle, so the oscillator locks to the driving frequency inside Arnold
    tongues, and the motion can be quasiperiodic or chaotic outside them. The system has the displacement `oscillator.x`
    of order 2 and its velocity `oscillator.x_d1`.

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

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ForcedVanDerPolSystem
        >>> ForcedVanDerPolSystem(A=0.5).prepare().system_.elements[0].A
        0.5
    """

    field = "physics/nonlinear_oscillators"
    topics = ("limit cycle", "ode", "second order", "forcing", "synchronization")
    references = ("van der Pol and van der Mark 1927", "Trzaska 2023")
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
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.mu = mu
        self.A = A
        self.omega = omega

    def _build_system(self) -> System:
        """Return the Van der Pol oscillator driven by a periodic force.

        The oscillator follows `x'' = μ (1 - x²) x' - x + A cos(ω t)`.
        """
        return System(
            self.label,
            elements=[VanDerPolElement("oscillator", x=self.x, x_d1=self.v, mu=self.mu, A=self.A, omega=self.omega)],
        )


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

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import AdlerEquationSystem
        >>> AdlerEquationSystem(detuning=2.0).prepare().system_.elements[0].detuning
        2.0
    """

    field = "physics/nonlinear_oscillators"
    topics = ("synchronization", "ode", "phase", "oscillation")
    references = ("Adler 1946", "Datseris and Parlitz 2022")
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        phi: float = 0.0,
        *,
        detuning: float = 0.5,
        epsilon: float = 1.0,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.phi = phi
        self.detuning = detuning
        self.epsilon = epsilon

    def _build_system(self) -> System:
        """Return the Adler equation `Δφ' = Δω + ε sin Δφ`.

        The equation is that of the phase difference of an oscillator and its drive.
        """
        return System(
            self.label,
            elements=[AdlerPhaseElement("difference", phi=self.phi, detuning=self.detuning, epsilon=self.epsilon)],
        )


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

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import NuclearQuadrupoleSystem
        >>> NuclearQuadrupoleSystem(B=0.3).prepare().system_.elements[0].B
        0.3
    """

    field = "physics/nonlinear_oscillators"
    topics = ("chaos", "ode", "hamiltonian", "nuclear physics")
    references = ("Baran and Raduta 1998", "Datseris and Parlitz 2022")
    until = 200.0
    dt = 0.05
    solver = Yoshida4Solver(dt=0.01)
    invariants: ClassVar[Mapping[str, float]] = {"nucleus.energy": 1e-6}
    observables = (ObservableSpec("energy", _get_quadrupole_energy, element="nucleus"),)

    def __init__(
        self,
        q1: float = 0.0,
        q2: float = 1.0,
        p1: float = 0.5,
        p2: float = 0.0,
        *,
        B: float = 0.5,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.q1 = q1
        self.q2 = q2
        self.p1 = p1
        self.p2 = p2
        self.B = B

    @staticmethod
    def _get_quadrupole_hamiltonian(s: SimpleNamespace) -> Expression:
        """Return the quadrupole Hamiltonian of the target."""
        return _get_quadrupole_energy(s.target)

    def _build_system(self) -> System:
        """Return the quadrupole vibrations of a nuclear surface.

        The vibrations are a Hamiltonian system of two degrees of freedom.
        """
        return System(
            self.label,
            elements=[QuadrupoleElement("nucleus", q1=self.q1, q2=self.q2, p1=self.p1, p2=self.p2, B=self.B)],
            interactions=[
                HamiltonianInteraction(
                    "motion",
                    target="nucleus",
                    coordinates=("q1", "q2"),
                    momenta=("p1", "p2"),
                    hamiltonian=self._get_quadrupole_hamiltonian,
                ),
            ],
        )


class ForcedTodaSystem(BaseSystem):
    """The forced Toda oscillator `x'' + d x' + e^x - 1 = a sin(omega t)`.

    It is a passive asymmetric oscillator.

    The exponential spring is stiff for `x > 0` and soft for `x < 0`, unlike the symmetric spring of the Duffing
    oscillator. Its free frequency falls from 1 as the amplitude grows, so the resonance curve bends to low driving
    frequencies and the response swings far to the soft side. Stronger driving gives coexisting responses and period
    doubling cascades to chaos as the driving period `2π / ω` changes. The system has the displacement `toda.x` of order
    2 and its velocity `toda.x_d1`.

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

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ForcedTodaSystem
        >>> ForcedTodaSystem(a=1.2).prepare().system_.elements[0].a
        1.2
    """

    field = "physics/nonlinear_oscillators"
    topics = ("chaos", "ode", "second order", "forcing", "resonance")
    references = ("Toda 1967", "Kurz and Lauterborn 1988", "Datseris and Parlitz 2022")
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
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.d = d
        self.a = a
        self.omega = omega

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        return System(
            self.label,
            elements=[TodaElement("toda", x=self.x, x_d1=self.v, d=self.d, a=self.a, omega=self.omega)],
        )


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
    oscillator `x'' + omega0² x` under the same impulses. The system has the displacement `oscillator.x` of order 2 and
    its velocity `oscillator.x_d1`, kicked by the impulses `pull` and `push`.

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

    Raises:
        ParameterError:
            The spring is unknown, the quarter period is not positive, or the shift is outside `(-1, 1)`.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ImpulsiveOscillatorSystem
        >>> ImpulsiveOscillatorSystem(a=0.5, gamma=0.5).prepare().system_.interactions[1].start
        0.25
    """

    field = "physics/nonlinear_oscillators"
    topics = ("ode", "second order", "impulses", "events", "forcing")
    references = ("Pilipchuk 2010", "Trzaska 2023")
    until = 10.0
    dt = 0.01
    solver = ScipyODESolver(method="DOP853", rtol=1e-10, atol=1e-12)

    def __init__(
        self,
        x: float = 0.0,
        v: float = 2.572,
        *,
        H: float = 2.0,
        a: float = 0.4165,
        gamma: float = 0.0,
        spring: str = "tangent",
        omega0: float = 1.0,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.H = H
        self.a = a
        self.gamma = gamma
        self.spring = spring
        self.omega0 = omega0

    def _build_system(self) -> System:
        """Return an oscillator kicked by a periodic train of impulses of alternating sign."""
        if self.spring not in SPRINGS:
            error_msg = f"Unknown spring {self.spring!r}. Use one of {', '.join(SPRINGS)}."
            raise ParameterError(error_msg)
        if not self.a > 0 or not -1 < self.gamma < 1:
            error_msg = (
                f"The impulses need a positive quarter period and a shift in (-1, 1), but got {self.a!r} and "
                f"{self.gamma!r}."
            )
            raise ParameterError(error_msg)
        if self.spring == "tangent":
            restoring = RateInteraction(
                "spring",
                target="oscillator",
                state="x",
                rate=lambda s: -tan(s.target.x) / cos(s.target.x) ** 2,
            )
        else:
            restoring = RateInteraction(
                "spring",
                target="oscillator",
                state="x",
                rate=lambda s: -(s.target.omega0**2) * s.target.x,
            )
        jump = 2 / (1 - self.gamma**2)
        pull = {"x_d1": lambda s: s.target.x_d1 - jump * s.target.H}
        push = {"x_d1": lambda s: s.target.x_d1 + jump * s.target.H}
        return System(
            self.label,
            elements=[ImpulsiveOscillatorElement("oscillator", x=self.x, x_d1=self.v, H=self.H, omega0=self.omega0)],
            interactions=[
                restoring,
                ImpulseInteraction(
                    "pull",
                    target="oscillator",
                    reset=pull,
                    period=4 * self.a,
                    start=(1 - self.gamma) * self.a,
                ),
                ImpulseInteraction(
                    "push",
                    target="oscillator",
                    reset=push,
                    period=4 * self.a,
                    start=(3 + self.gamma) * self.a,
                ),
            ],
        )
