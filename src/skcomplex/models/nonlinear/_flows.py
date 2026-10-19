"""Model canonical chaotic flows: Lorenz, Rössler, Duffing, Van der Pol, and Hénon-Heiles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import cos

from ...core import Domain, parameter, state
from ...model import Element, Rate, System, derivative
from ...runtime import register_model

Lorenz = Element.define(
    'Lorenz',
    states={'x': 1.0, 'y': 1.0, 'z': 1.0},
    parameters={'sigma': 10.0, 'rho': 28.0, 'beta': 8 / 3},
    laws=lambda s: [
        Rate(s.x, s.sigma * (s.y - s.x)),
        Rate(s.y, s.x * (s.rho - s.z) - s.y),
        Rate(s.z, s.x * s.y - s.beta * s.z),
    ],
)

Rossler = Element.define(
    'Rossler',
    states={'x': 1.0, 'y': 1.0, 'z': 1.0},
    parameters={'a': 0.2, 'b': 0.2, 'c': 5.7},
    laws=lambda s: [
        Rate(s.x, -s.y - s.z),
        Rate(s.y, s.x + s.a * s.y),
        Rate(s.z, s.b + s.z * (s.x - s.c)),
    ],
)

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
    field='nonlinear dynamics',
    tags=('chaos', 'ode', 'strange attractor'),
    references=('Lorenz 1963', 'Datseris and Parlitz 2022'),
    defaults={'until': 50.0, 'dt': 0.01},
)
def lorenz(
    x: float = 1.0,
    y: float = 1.0,
    z: float = 1.0,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8 / 3,
) -> System:
    """Return the Lorenz-63 model of convection, chaotic on a butterfly-shaped attractor for its standard parameters.

    Args:
        x:
            The initial convection intensity.

        y:
            The initial temperature difference between rising and falling currents.

        z:
            The initial deviation of the vertical temperature profile.

        sigma:
            The Prandtl number.

        rho:
            The Rayleigh number relative to its critical value.

        beta:
            The geometric factor.

    Returns:
        system:
            The model, with the states `lorenz.x`, `lorenz.y`, and `lorenz.z`.

    Examples:
        >>> from skcomplex.models.nonlinear import lorenz
        >>> lorenz(rho=20.0).elements[0].rho
        20.0
    """
    return System(elements=[Lorenz('lorenz', x=x, y=y, z=z, sigma=sigma, rho=rho, beta=beta)])


@register_model(
    field='nonlinear dynamics',
    tags=('chaos', 'ode', 'strange attractor'),
    references=('Rössler 1976',),
    defaults={'until': 200.0, 'dt': 0.05},
)
def rossler(
    x: float = 1.0,
    y: float = 1.0,
    z: float = 1.0,
    *,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
) -> System:
    """Return the Rössler model, a chaotic flow with one folded band for its standard parameters.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        a:
            The feedback of the second coordinate on itself.

        b:
            The constant drive of the third coordinate.

        c:
            The threshold of the third coordinate's growth.

    Returns:
        system:
            The model, with the states `rossler.x`, `rossler.y`, and `rossler.z`.

    Examples:
        >>> from skcomplex.models.nonlinear import rossler
        >>> rossler(c=4.0).elements[0].c
        4.0
    """
    return System(elements=[Rossler('rossler', x=x, y=y, z=z, a=a, b=b, c=c)])


@register_model(
    field='nonlinear dynamics',
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
        >>> from skcomplex.models.nonlinear import duffing
        >>> duffing(gamma=0.3).elements[0].gamma
        0.3
    """
    return System(
        elements=[Duffing('duffing', x=x, x_d1=v, delta=delta, alpha=alpha, beta=beta, gamma=gamma, omega=omega)],
    )


@register_model(
    field='nonlinear dynamics',
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
        >>> from skcomplex.models.nonlinear import van_der_pol
        >>> van_der_pol(mu=3.0).elements[0].mu
        3.0
    """
    return System(elements=[VanDerPol('oscillator', x=x, x_d1=v, mu=mu)])


@register_model(
    field='nonlinear dynamics',
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
        >>> from skcomplex.models.nonlinear import henon_heiles
        >>> henon_heiles(px=0.3).elements[0].x_d1
        0.3
    """
    return System(elements=[HenonHeiles('star', x=x, y=y, x_d1=px, y_d1=py)])
