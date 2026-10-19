"""Nonlinear oscillators models: Duffing, Van der Pol, and Hamiltonian chaos."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import cos

from ....core.base import Domain, parameter, state
from ....core.model import Element, Rate, System, derivative
from ....core.runtime import register_model

Duffing = Element.define(
    'Duffing',
    states={'x': state(default=1.0, order=2)},
    parameters={'delta': 0.3, 'alpha': -1.0, 'beta': 1.0, 'gamma': 0.5, 'omega': 1.2},
    laws=lambda s: [
        Rate(s.x, -s.delta * derivative(s.x, 1) - s.alpha * s.x - s.beta * s.x**3 + s.gamma * cos(s.omega * s.t)),
    ],
)

VanDerPol = Element.define(
    'VanDerPol',
    states={'x': state(default=2.0, order=2)},
    parameters={'mu': parameter(default=1.0, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [Rate(s.x, s.mu * (1 - s.x**2) * derivative(s.x, 1) - s.x)],
)

HenonHeiles = Element.define(
    'HenonHeiles',
    states={'x': state(default=0.0, order=2, initial=(0.35,)), 'y': state(default=0.1, order=2)},
    laws=lambda s: [Rate(s.x, -s.x - 2 * s.x * s.y), Rate(s.y, -s.y - s.x**2 + s.y**2)],
)


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
        elements=[Duffing('duffing', x=x, x_d1=v, delta=delta, alpha=alpha, beta=beta, gamma=gamma, omega=omega)],
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
    return System(elements=[VanDerPol('oscillator', x=x, x_d1=v, mu=mu)])


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
    return System(elements=[HenonHeiles('star', x=x, y=y, x_d1=px, y_d1=py)])
