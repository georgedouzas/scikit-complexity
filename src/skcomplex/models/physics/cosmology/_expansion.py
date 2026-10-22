"""Expansion models: the scale factor of a homogeneous universe under matter and dark energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from ....core.base import Domain, ValidationError, parameter, state
from ....core.components import Element
from ....core.laws import Equation, derivative
from ....core.runtime import register_model
from ....core.system import System

ScaleFactor = Element.define(
    'ScaleFactor',
    states={'a': state(default=1.0, domain=Domain.POSITIVE, order=2)},
    parameters={
        'H0': parameter(default=1.0, domain=Domain.POSITIVE),
        'omega_matter': parameter(default=0.3, domain=Domain.NONNEGATIVE),
        'omega_lambda': 0.7,
    },
    laws=lambda s: [
        Equation(
            s.a**2 * derivative(s.a, 2) == -s.omega_matter * s.H0**2 / 2 + s.omega_lambda * s.H0**2 * s.a**3,
        ),
    ],
)


@register_model(
    field='physics/cosmology',
    tags=('cosmology', 'ode', 'second order', 'gravity', 'dark energy'),
    references=('Friedmann 1922', 'Lemaître 1927', 'Satz 2022'),
    defaults={'until': 2.0, 'dt': 0.02},
)
def friedmann_expansion(
    a: float = 1.0,
    *,
    H0: float = 1.0,
    omega_matter: float = 0.3,
    omega_lambda: float = 0.7,
) -> System:
    """Return the scale factor of a homogeneous universe of matter and dark energy, after Friedmann and Lemaître.

    The acceleration equation `a² a'' = -4 pi G M / 3 + Lambda a³ / 3` is written with the density parameters of
    matter `Ω_m = 8 pi G M / (3 H0²)` and of the cosmological constant `Ω_Λ = Lambda / (3 H0²)` in units of the Hubble
    constant `H0` at `a = 1`, so `a² a'' = -Ω_m H0² / 2 + Ω_Λ H0² a³`. The initial expansion rate follows from the
    Friedmann equation `a'² = H0² (Ω_m / a + Ω_Λ a² + 1 - Ω_m - Ω_Λ)`, which stays a first integral of the motion.
    Matter decelerates the expansion and dark energy accelerates it once `a` exceeds `(Ω_m / (2 Ω_Λ))^(1/3)`.

    Args:
        a:
            The initial scale factor.

        H0:
            The Hubble constant, the expansion rate at `a = 1`.

        omega_matter:
            The density parameter of matter.

        omega_lambda:
            The density parameter of the cosmological constant.

    Returns:
        system:
            The model, with the scale factor `universe.a` of order 2 and its rate `universe.a_d1`.

    Raises:
        ValidationError:
            The universe does not expand at the initial scale factor.

    Examples:
        >>> from skcomplex.models.physics.cosmology import friedmann_expansion
        >>> friedmann_expansion(omega_matter=1.0, omega_lambda=0.0).elements[0].a_d1
        1.0
    """
    square = H0**2 * (omega_matter / a + omega_lambda * a**2 + 1 - omega_matter - omega_lambda)
    if square < 0:
        error_msg = (
            f'The Friedmann equation gives a negative squared expansion rate {square} at a = {a}. Choose a, '
            'omega_matter, and omega_lambda for which it is positive.'
        )
        raise ValidationError(error_msg)
    universe = ScaleFactor(
        'universe',
        a=a,
        a_d1=math.sqrt(square),
        H0=H0,
        omega_matter=omega_matter,
        omega_lambda=omega_lambda,
    )
    return System(components=[universe])
