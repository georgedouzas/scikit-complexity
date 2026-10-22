"""Dynamical systems models: the subcritical Hopf normal form, with bistability and hysteresis."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, Expression, parameter
from ....core.model import Element, Rate, Symbols, System
from ....core.runtime import register_model


def _subcritical_hopf_rates(s: Symbols) -> tuple[Expression, Expression]:
    """Return the real and imaginary parts of the right-hand side of the subcritical Hopf normal form."""
    radius = s.x**2 + s.y**2
    real = s.alpha + radius - s.gamma * radius**2
    imaginary = s.omega + s.q * radius - s.s * radius**2
    return real * s.x - imaginary * s.y, imaginary * s.x + real * s.y


SubcriticalHopf = Element.define(
    'SubcriticalHopf',
    states={'x': 0.1, 'y': 0.0},
    parameters={
        'alpha': -0.1,
        'omega': 1.0,
        'q': 0.0,
        'gamma': parameter(default=1.0, domain=Domain.POSITIVE),
        's': 0.0,
    },
    laws=lambda s: [Rate(s.x, _subcritical_hopf_rates(s)[0]), Rate(s.y, _subcritical_hopf_rates(s)[1])],
)


@register_model(
    field='mathematics/dynamical_systems',
    tags=('bifurcation', 'normal form', 'ode', 'limit cycle', 'bistability', 'hysteresis'),
    references=('Bertin 2021',),
    defaults={'until': 50.0, 'dt': 0.1},
)
def subcritical_hopf_form(
    x: float = 0.1,
    y: float = 0.0,
    *,
    alpha: float = -0.1,
    omega: float = 1.0,
    q: float = 0.0,
    gamma: float = 1.0,
    s: float = 0.0,
) -> System:
    """Return the subcritical Hopf normal form, where a stable focus coexists with a stable limit cycle.

    The complex amplitude `z = x + iy` follows `z' = (alpha + iω) z + (1 + iq) |z|² z - (gamma + is) |z|⁴ z`, so its
    modulus follows `r' = alpha r + r³ - gamma r⁵`. The focus `z = 0` is stable for `alpha < 0`. The stable limit cycle
    of radius `r² = (1 + √(1 + 4 alpha gamma)) / (2 gamma)` exists for `alpha > -1 / (4 gamma)`, and an unstable cycle
    separates the two attractors while `alpha < 0`. A slow sweep of `alpha` up and down therefore jumps to the cycle at
    `alpha = 0` and falls back to the focus at `alpha = -1 / (4 gamma)`, a hysteresis loop.

    Args:
        x:
            The initial real part of `z`.

        y:
            The initial imaginary part of `z`.

        alpha:
            The bifurcation parameter `alpha`, the linear growth rate.

        omega:
            The angular frequency `ω` at small amplitude.

        q:
            The nonlinear frequency shift `q` of the cubic term.

        gamma:
            The stabilizing quintic coefficient `gamma`, positive.

        s:
            The nonlinear frequency shift `s` of the quintic term.

    Returns:
        system:
            The model, with the states `form.x` and `form.y`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import subcritical_hopf_form
        >>> subcritical_hopf_form(alpha=0.1).elements[0].alpha
        0.1
    """
    return System(components=[SubcriticalHopf('form', x=x, y=y, alpha=alpha, omega=omega, q=q, gamma=gamma, s=s)])
