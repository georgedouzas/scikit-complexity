"""Nonlinear oscillators models: Duffing, Van der Pol, Stuart-Landau, Adler, and Hamiltonian chaos."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo, sin
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ..._catalog import register_model


class DuffingElement(Element):
    """Duffing element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 1.0,
        x_d1: float = 0.0,
        delta: Annotated[object, ParameterSpec()] = 0.3,
        alpha: Annotated[object, ParameterSpec()] = -1.0,
        beta: Annotated[object, ParameterSpec()] = 1.0,
        gamma: Annotated[object, ParameterSpec()] = 0.5,
        omega: Annotated[object, ParameterSpec()] = 1.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the duffing element."""
        return [
            RateLaw(
                s.x,
                -s.delta * derivative(s.x, 1) - s.alpha * s.x - s.beta * s.x**3 + s.gamma * cos(s.omega * s.t),
            ),
        ]


class VanDerPolElement(Element):
    """Van der pol element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 2.0,
        x_d1: float = 0.0,
        mu: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.mu = mu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the van der pol element."""
        return [RateLaw(s.x, s.mu * (1 - s.x**2) * derivative(s.x, 1) - s.x)]


class HenonHeilesElement(Element):
    """Henon heiles element."""

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2, initial=(0.35,))] = 0.0,
        x_d1: float = 0.35,
        y: Annotated[object, StateSpec(order=2)] = 0.1,
        y_d1: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.y = y
        self.y_d1 = y_d1

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the henon heiles element."""
        return [RateLaw(s.x, -s.x - 2 * s.x * s.y), RateLaw(s.y, -s.y - s.x**2 + s.y**2)]


class StuartLandauElement(Element):
    """Stuart landau element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        y: Annotated[object, StateSpec()] = 0.0,
        mu: Annotated[object, ParameterSpec()] = 1.0,
        omega: Annotated[object, ParameterSpec()] = 1.0,
        beta: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.mu = mu
        self.omega = omega
        self.beta = beta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the stuart landau element."""
        return [
            RateLaw(s.x, s.mu * s.x - s.omega * s.y - (s.x**2 + s.y**2) * (s.x - s.beta * s.y)),
            RateLaw(s.y, s.mu * s.y + s.omega * s.x - (s.x**2 + s.y**2) * (s.y + s.beta * s.x)),
        ]


class ForcedVanDerPolElement(Element):
    """Forced van der pol element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 2.0,
        x_d1: float = 0.0,
        mu: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        A: Annotated[object, ParameterSpec()] = 1.0,
        omega: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.mu = mu
        self.A = A
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the forced van der pol element."""
        return [RateLaw(s.x, s.mu * (1 - s.x**2) * derivative(s.x, 1) - s.x + s.A * cos(s.omega * s.t))]


class AdlerPhaseElement(Element):
    """Adler phase element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[object, StateSpec()] = 0.0,
        detuning: Annotated[object, ParameterSpec()] = 0.5,
        epsilon: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.detuning = detuning
        self.epsilon = epsilon

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the adler phase element."""
        return [RateLaw(s.phi, s.detuning + s.epsilon * sin(s.phi))]


@register_model(
    field='physics/nonlinear_oscillators',
    tags=('chaos', 'ode', 'second order', 'forcing'),
    references=('Duffing 1918', 'Trzaska 2023'),
    defaults={'until': 200.0, 'dt': 0.05},
)
def duffing(
    x: float = 1.0,
    v: float = 0.0,
    *,
    delta: float = 0.3,
    alpha: float = -1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    omega: float = 1.2,
) -> System:
    """Return the forced Duffing oscillator `x'' = -delta x' - alpha x - beta x^3 + gamma cos(omega t)`, chaotic here.

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

    Returns:
        system:
            The model, with the displacement `duffing.x` of order 2 and its velocity `duffing.x_d1`.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import duffing
        >>> duffing(gamma=0.3).elements[0].gamma
        0.3
    """
    return System(
        elements=[
            DuffingElement('duffing', x=x, x_d1=v, delta=delta, alpha=alpha, beta=beta, gamma=gamma, omega=omega),
        ],
    )


@register_model(
    field='physics/nonlinear_oscillators',
    tags=('limit cycle', 'ode', 'second order', 'oscillation'),
    references=('van der Pol 1926', 'Gros 2015'),
    defaults={'until': 50.0, 'dt': 0.05},
)
def van_der_pol(x: float = 2.0, v: float = 0.0, *, mu: float = 1.0) -> System:
    """Return the Van der Pol oscillator `x'' = μ (1 - x²) x' - x`, which settles on a limit cycle for `μ > 0`.

    Args:
        x:
            The initial displacement.

        v:
            The initial velocity.

        mu:
            The strength of the nonlinear damping.

    Returns:
        system:
            The model, with the displacement `oscillator.x` of order 2 and its velocity `oscillator.x_d1`.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import van_der_pol
        >>> van_der_pol(mu=3.0).elements[0].mu
        3.0
    """
    return System(elements=[VanDerPolElement('oscillator', x=x, x_d1=v, mu=mu)])


@register_model(
    field='physics/nonlinear_oscillators',
    tags=('chaos', 'ode', 'second order', 'hamiltonian'),
    references=('Hénon and Heiles 1964', 'Datseris and Parlitz 2022'),
    defaults={'until': 200.0, 'dt': 0.05},
)
def henon_heiles(x: float = 0.0, y: float = 0.1, px: float = 0.35, py: float = 0.0) -> System:
    """Return the Hénon-Heiles model of a star in a galactic potential, whose motion is chaotic at higher energies.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        px:
            The initial first momentum.

        py:
            The initial second momentum.

    Returns:
        system:
            The model, with the coordinates `star.x` and `star.y` of order 2 and their momenta `star.x_d1` and
            `star.y_d1`.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import henon_heiles
        >>> henon_heiles(px=0.3).elements[0].x_d1
        0.3
    """
    return System(elements=[HenonHeilesElement('star', x=x, y=y, x_d1=px, y_d1=py)])


@register_model(
    field='physics/nonlinear_oscillators',
    tags=('limit cycle', 'ode', 'hopf bifurcation', 'oscillation'),
    references=('Gros 2015', 'Dilão 2023'),
    defaults={'until': 30.0, 'dt': 0.05},
)
def stuart_landau(
    x: float = 0.1,
    y: float = 0.0,
    *,
    mu: float = 1.0,
    omega: float = 1.0,
    beta: float = 0.0,
) -> System:
    """Return the Stuart-Landau oscillator `z' = (μ + i ω) z - (1 + i β) |z|² z`, the normal form of a Hopf bifurcation.

    The complex amplitude `z = x + i y` is written as two real states. For `μ > 0` every orbit except the origin
    settles on the circle of radius `sqrt(μ)`, turning at the frequency `ω - β μ`.

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

    Returns:
        system:
            The model, with the real part `oscillator.x` and the imaginary part `oscillator.y` of the amplitude.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import stuart_landau
        >>> stuart_landau(mu=0.5).elements[0].mu
        0.5
    """
    return System(elements=[StuartLandauElement('oscillator', x=x, y=y, mu=mu, omega=omega, beta=beta)])


@register_model(
    field='physics/nonlinear_oscillators',
    tags=('limit cycle', 'ode', 'second order', 'forcing', 'synchronization'),
    references=('van der Pol and van der Mark 1927', 'Trzaska 2023'),
    defaults={'until': 100.0, 'dt': 0.05},
)
def forced_van_der_pol(
    x: float = 2.0,
    v: float = 0.0,
    *,
    mu: float = 1.0,
    A: float = 1.0,
    omega: float = 1.0,
) -> System:
    """Return the Van der Pol oscillator driven by a periodic force, `x'' = μ (1 - x²) x' - x + A cos(ω t)`.

    A strong enough force entrains the limit cycle, so the oscillator locks to the driving frequency inside Arnold
    tongues, and the motion can be quasiperiodic or chaotic outside them.

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

    Returns:
        system:
            The model, with the displacement `oscillator.x` of order 2 and its velocity `oscillator.x_d1`.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import forced_van_der_pol
        >>> forced_van_der_pol(A=0.5).elements[0].A
        0.5
    """
    return System(elements=[ForcedVanDerPolElement('oscillator', x=x, x_d1=v, mu=mu, A=A, omega=omega)])


@register_model(
    field='physics/nonlinear_oscillators',
    tags=('synchronization', 'ode', 'phase', 'oscillation'),
    references=('Adler 1946', 'Datseris and Parlitz 2022'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def adler_equation(phi: float = 0.0, *, detuning: float = 0.5, epsilon: float = 1.0) -> System:
    """Return the Adler equation `Δφ' = Δω + ε sin Δφ` for the phase difference of an oscillator and its drive.

    The phase difference locks at a fixed point when `|Δω| < ε`, inside the Arnold tongue. Outside it, the phase slips
    by `2π` every `2π / sqrt(Δω² - ε²)`.

    Args:
        phi:
            The initial phase difference.

        detuning:
            The frequency difference `Δω`.

        epsilon:
            The strength of the coupling.

    Returns:
        system:
            The model, with the phase difference `difference.phi`.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import adler_equation
        >>> adler_equation(detuning=2.0).elements[0].detuning
        2.0
    """
    return System(elements=[AdlerPhaseElement('difference', phi=phi, detuning=detuning, epsilon=epsilon)])
