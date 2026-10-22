"""Fractional dynamics models: relaxation, oscillators, and chaotic flows with Caputo derivatives of states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Callable
from functools import cache

from ....core.base import state
from ....core.components import Element
from ....core.laws import Rate
from ....core.runtime import register_model
from ....core.system import System

ORDER_CODE = re.compile(r'[0-9]+p[0-9]+(e-?[0-9]+)?')


def _type_name(prefix: str, order: float) -> str:
    """Return the name of the element type of an order, which holds the order with `p` for the decimal point."""
    return f'{prefix}_{repr(order).replace(".", "p")}'


@cache
def relaxation_type(order: float) -> type[Element]:
    """Return the element type of fractional relaxation `D^alpha x = -lambda x` of an order.

    Args:
        order:
            The order `alpha` of the Caputo derivative of `x`.

    Returns:
        element_type:
            The element type, the same for equal orders.

    Examples:
        >>> from skcomplex.models.mathematics.fractional_dynamics import relaxation_type
        >>> relaxation_type(0.5).__name__
        'FractionalRelaxation_0p5'
    """
    return Element.define(
        _type_name('FractionalRelaxation', order),
        states={'x': state(1.0, order=order)},
        parameters={'lam': 1.0},
        laws=lambda s: [Rate(s.x, -s.lam * s.x)],
    )


@cache
def oscillator_type(order: float) -> type[Element]:
    """Return the element type of the fractional oscillator `D^alpha x = v`, `D^alpha v = -omega^2 x - 2 gamma v`.

    Args:
        order:
            The order `alpha` of the Caputo derivatives of `x` and `v`.

    Returns:
        element_type:
            The element type, the same for equal orders.

    Examples:
        >>> from skcomplex.models.mathematics.fractional_dynamics import oscillator_type
        >>> oscillator_type(0.8).__name__
        'FractionalOscillator_0p8'
    """
    return Element.define(
        _type_name('FractionalOscillator', order),
        states={'x': state(1.0, order=order), 'v': state(0.0, order=order)},
        parameters={'omega': 1.0, 'gamma': 0.0},
        laws=lambda s: [Rate(s.x, s.v), Rate(s.v, -(s.omega**2) * s.x - 2 * s.gamma * s.v)],
    )


@cache
def van_der_pol_type(order: float) -> type[Element]:
    """Return the element type of the fractional Van der Pol system `D^alpha x = y`, `D^alpha y = mu (1 - x^2) y - x`.

    Args:
        order:
            The order `alpha` of the Caputo derivatives of `x` and `y`.

    Returns:
        element_type:
            The element type, the same for equal orders.

    Examples:
        >>> from skcomplex.models.mathematics.fractional_dynamics import van_der_pol_type
        >>> van_der_pol_type(0.9).__name__
        'FractionalVanDerPol_0p9'
    """
    return Element.define(
        _type_name('FractionalVanDerPol', order),
        states={'x': state(2.0, order=order), 'y': state(0.0, order=order)},
        parameters={'mu': 1.0},
        laws=lambda s: [Rate(s.x, s.y), Rate(s.y, s.mu * (1 - s.x**2) * s.y - s.x)],
    )


@cache
def lorenz_type(order: float) -> type[Element]:
    """Return the element type of the fractional Lorenz flow, the Lorenz equations with Caputo derivatives of an order.

    Args:
        order:
            The order `q` of the Caputo derivatives of `x`, `y`, and `z`.

    Returns:
        element_type:
            The element type, the same for equal orders.

    Examples:
        >>> from skcomplex.models.mathematics.fractional_dynamics import lorenz_type
        >>> lorenz_type(0.99).__name__
        'FractionalLorenz_0p99'
    """
    return Element.define(
        _type_name('FractionalLorenz', order),
        states={'x': state(1.0, order=order), 'y': state(1.0, order=order), 'z': state(1.0, order=order)},
        parameters={'sigma': 10.0, 'rho': 28.0, 'beta': 8 / 3},
        laws=lambda s: [
            Rate(s.x, s.sigma * (s.y - s.x)),
            Rate(s.y, s.x * (s.rho - s.z) - s.y),
            Rate(s.z, s.x * s.y - s.beta * s.z),
        ],
    )


TYPES: dict[str, Callable[[float], type[Element]]] = {
    'FractionalRelaxation': relaxation_type,
    'FractionalOscillator': oscillator_type,
    'FractionalVanDerPol': van_der_pol_type,
    'FractionalLorenz': lorenz_type,
}


def __getattr__(name: str) -> type[Element]:
    """Return the element type of an order by its name, so the types of every order pickle by reference."""
    prefix, _, code = name.rpartition('_')
    if prefix not in TYPES or ORDER_CODE.fullmatch(code) is None:
        error_msg = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(error_msg)
    return TYPES[prefix](float(code.replace('p', '.')))


@register_model(
    field='mathematics/fractional_dynamics',
    tags=('fractional', 'memory', 'anomalous relaxation'),
    references=('Podlubny 1999', 'Trzaska 2023'),
    defaults={'until': 20.0, 'dt': 0.1},
)
def fractional_relaxation(x: float = 1.0, *, order: float = 0.6, lam: float = 1.0) -> System:
    """Return fractional relaxation `D^alpha x = -lambda x`, which decays as the Mittag-Leffler function `E_alpha`.

    The solution `x(0) E_alpha(-lambda t^alpha)` decays slower than any exponential, as the power law `t^(-alpha)` at
    long times, since the Caputo derivative keeps the memory of the whole past. Order 1 gives exponential decay.

    Args:
        x:
            The initial value.

        order:
            The order `alpha` of the Caputo derivative, in `(0, 1]`.

        lam:
            The relaxation rate.

    Returns:
        system:
            The model, with the state `relaxation.x`.

    Examples:
        >>> from skcomplex.models.mathematics.fractional_dynamics import fractional_relaxation
        >>> fractional_relaxation(order=0.5).elements[0].lam
        1.0
    """
    return System(components=[relaxation_type(float(order))('relaxation', x=x, lam=lam)])


@register_model(
    field='mathematics/fractional_dynamics',
    tags=('fractional', 'memory', 'oscillator'),
    references=('Podlubny 1999', 'Trzaska 2023'),
    defaults={'until': 30.0, 'dt': 0.1},
)
def fractional_oscillator(
    x: float = 1.0,
    v: float = 0.0,
    *,
    order: float = 0.8,
    omega: float = 1.0,
    gamma: float = 0.0,
) -> System:
    """Return the fractional damped oscillator `D^alpha x = v`, `D^alpha v = -omega^2 x - 2 gamma v`.

    With `gamma = 0` the system is the fractional oscillation equation of order `2 alpha`, whose memory damps the
    oscillations for `1/2 < alpha < 1` without any friction. Order 1 gives the harmonic oscillator.

    Args:
        x:
            The initial displacement.

        v:
            The initial value of the first state's derivative of order `alpha`.

        order:
            The order `alpha` of the Caputo derivatives, in `(0, 1]`.

        omega:
            The angular frequency.

        gamma:
            The damping rate.

    Returns:
        system:
            The model, with the states `oscillator.x` and `oscillator.v`.

    Examples:
        >>> from skcomplex.models.mathematics.fractional_dynamics import fractional_oscillator
        >>> fractional_oscillator(order=0.9).elements[0].omega
        1.0
    """
    return System(components=[oscillator_type(float(order))('oscillator', x=x, v=v, omega=omega, gamma=gamma)])


@register_model(
    field='mathematics/fractional_dynamics',
    tags=('fractional', 'memory', 'limit cycle'),
    references=('Podlubny 1999', 'Trzaska 2023'),
    defaults={'until': 40.0, 'dt': 0.1},
)
def fractional_van_der_pol(x: float = 2.0, y: float = 0.0, *, order: float = 0.9, mu: float = 1.0) -> System:
    """Return the fractional Van der Pol oscillator `D^alpha x = y`, `D^alpha y = mu (1 - x^2) y - x`.

    For orders close to 1 the trajectory approaches a closed cycle, and the memory of lower orders shrinks it toward
    the fixed point at the origin.

    Args:
        x:
            The initial displacement.

        y:
            The initial value of the displacement's derivative of order `alpha`.

        order:
            The order `alpha` of the Caputo derivatives, in `(0, 1]`.

        mu:
            The strength of the nonlinear damping.

    Returns:
        system:
            The model, with the states `oscillator.x` and `oscillator.y`.

    Examples:
        >>> from skcomplex.models.mathematics.fractional_dynamics import fractional_van_der_pol
        >>> fractional_van_der_pol(mu=2.0).elements[0].mu
        2.0
    """
    return System(components=[van_der_pol_type(float(order))('oscillator', x=x, y=y, mu=mu)])


@register_model(
    field='mathematics/fractional_dynamics',
    tags=('fractional', 'memory', 'chaos', 'strange attractor'),
    references=('Grigorenko and Grigorenko 2003', 'Trzaska 2023'),
    defaults={'until': 30.0, 'dt': 0.05},
)
def fractional_lorenz(
    x: float = 1.0,
    y: float = 1.0,
    z: float = 1.0,
    *,
    order: float = 0.99,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8 / 3,
) -> System:
    """Return the fractional Lorenz flow of Grigorenko and Grigorenko, chaotic for a total order of about 2.97.

    Args:
        x:
            The initial convection intensity.

        y:
            The initial temperature difference between rising and falling currents.

        z:
            The initial deviation of the vertical temperature profile.

        order:
            The order `q` of the Caputo derivatives of all three states, in `(0, 1]`.

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
        >>> from skcomplex.models.mathematics.fractional_dynamics import fractional_lorenz
        >>> fractional_lorenz(rho=20.0).elements[0].rho
        20.0
    """
    flow = lorenz_type(float(order))('lorenz', x=x, y=y, z=z, sigma=sigma, rho=rho, beta=beta)
    return System(components=[flow])
