"""Nonlinear oscillators models: Duffing, Van der Pol, Stuart-Landau, Adler, and Hamiltonian chaos."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, sin

from ....core.elements import Element
from ....core.interactions import RateInteraction
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ..common.oscillators import StuartLandauElement
from ._van_der_pol import VanDerPolElement


class DuffingElement(Element):
    """Displacement and parameters of the forced Duffing oscillator, whose acceleration a `RateInteraction` sets.

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
        >>> from skcomplex.models.physics.nonlinear_oscillators import DuffingElement
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
    """Star in the Hénon-Heiles potential `V = (x^2 + y^2) / 2 + x^2 y - y^3 / 3`, with `x'' = -dV/dx`, `y'' = -dV/dy`.

    Below the escape energy `1 / 6` the motion is regular for small energies and mostly chaotic near the escape energy.

    Args:
        label:
            The label of the star.

        visual:
            How the star is drawn.

        x:
            The initial first coordinate.

        x_d1:
            The initial first momentum.

        y:
            The initial second coordinate.

        y_d1:
            The initial second momentum.

    Attributes:
        default_visual_:
            The visual of the star when it sets none.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import HenonHeilesElement
        >>> HenonHeilesElement('star').x_d1
        0.35
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None}),
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.35,
        y: Annotated[float, StateSpec(order=2)] = 0.1,
        y_d1: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.y = y
        self.y_d1 = y_d1

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the accelerations of the two coordinates.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the second derivatives of `x` and `y`.
        """
        s = namespace
        return [RateLaw(s.x, -s.x - 2 * s.x * s.y), RateLaw(s.y, -s.y - s.x**2 + s.y**2)]


class AdlerPhaseElement(Element):
    """Phase difference of a forced oscillator, `phi' = detuning + epsilon sin(phi)`, the Adler equation.

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
        >>> from skcomplex.models.physics.nonlinear_oscillators import AdlerPhaseElement
        >>> AdlerPhaseElement('difference').detuning
        0.5
    """

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
    acceleration = RateInteraction(
        'acceleration',
        target='duffing',
        state='x',
        rate=lambda s: (
            -s.target.delta * derivative(s.target.x, 1)
            - s.target.alpha * s.target.x
            - s.target.beta * s.target.x**3
            + s.target.gamma * cos(s.target.omega * s.t)
        ),
    )
    return System(
        elements=[
            DuffingElement('duffing', x=x, x_d1=v, delta=delta, alpha=alpha, beta=beta, gamma=gamma, omega=omega),
        ],
        interactions=[acceleration],
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
    return System(elements=[VanDerPolElement('oscillator', x=x, x_d1=v, mu=mu, A=A, omega=omega)])


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
