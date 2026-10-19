"""Paleoclimate oscillators: glacial cycles and the excitable carbon cycle."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

from sage.all__sagemath_symbolics import cos, pi

from ....core.base import Domain, Expression, state
from ....core.model import Element, Law, Rate, Symbols, System
from ....core.runtime import register_model


def _saltzman_laws(s: Symbols) -> list[Law]:
    """Return the feedbacks of ice volume, carbon dioxide, and deep ocean temperature, with orbital forcing."""
    orbital = cos(2 * pi * s.t0 * s.t / 22.1) - cos(2 * pi * s.t0 * s.t / 44.1)
    return [
        Rate(s.x, -s.x - s.y - s.u * s.z - s.nu * orbital),
        Rate(s.y, -s.p * s.z + s.r * s.y - s.s * s.y**2 - s.y**3),
        Rate(s.z, -s.q * (s.x + s.z)),
    ]


def _saturation(c: Any, crossover: Any, exponent: Any) -> Expression:  # noqa: ANN401
    """Return the sigmoid `c^n / (c^n + crossover^n)`."""
    return c**exponent / (c**exponent + crossover**exponent)


def _carbon_laws(s: Symbols) -> list[Law]:
    """Return Rothman's balance of carbonate ion and dissolved inorganic carbon."""
    burial = s.b * _saturation(s.c, s.c_p, s.gamma)
    respiration = s.theta * (1 - _saturation(s.c, s.c_x, s.gamma))
    buffer = s.f0 * _saturation(s.c, s.c_f, s.beta)
    return [
        Rate(s.c, buffer * (s.mu * (1 - burial - respiration - s.nu) + s.w - s.w0)),
        Rate(s.w, s.mu * (1 - burial + respiration + s.nu) - s.w + s.w0),
    ]


SaltzmanMaasch = Element.define(
    'SaltzmanMaasch',
    states={'x': -0.2, 'y': 1.0, 'z': 0.0},
    parameters={'p': 1.0, 'q': 2.5, 'r': 1.3, 's': 0.6, 'u': 0.2, 'nu': 0.0, 't0': 10.0},
    laws=_saltzman_laws,
)


def _ice_age_laws(s: Symbols) -> list[Law]:
    """Return the slow ice volume and the fast climate of a Van der Pol relaxation oscillator under orbital forcing."""
    forcing = cos(2 * pi * s.t / 23) + cos(2 * pi * s.t / 41)
    return [
        Rate(s.x, -(s.y + s.beta + s.gamma * forcing) / s.tau),
        Rate(s.y, -s.alpha * (s.y**3 / 3 - s.y - s.x) / s.tau),
    ]


VanDerPolIceAges = Element.define(
    'VanDerPolIceAges',
    states={'x': 0.0, 'y': 1.0},
    parameters={'alpha': 30.0, 'beta': 0.75, 'gamma': 0.0, 'tau': 35.09},
    laws=_ice_age_laws,
)

CarbonCycle = Element.define(
    'CarbonCycle',
    states={'c': state(default=83.58, domain=Domain.POSITIVE), 'w': state(default=2050.0, domain=Domain.POSITIVE)},
    parameters={
        'mu': 250.0,
        'b': 4.0,
        'theta': 5.0,
        'c_x': 55.0,
        'c_p': 110.0,
        'nu': 0.0,
        'w0': 2000.0,
        'gamma': 4.0,
        'f0': 0.694,
        'c_f': 43.9,
        'beta': 1.7,
    },
    laws=_carbon_laws,
)


@register_model(
    field='earth/climate',
    tags=('ode', 'limit cycle', 'ice ages', 'paleoclimate'),
    references=('Saltzman and Maasch 1991', 'Datseris and Parlitz 2022'),
    defaults={'until': 50.0, 'dt': 0.05},
)
def saltzman_glaciation(
    x: float = -0.2,
    y: float = 1.0,
    z: float = 0.0,
    *,
    p: float = 1.0,
    q: float = 2.5,
    r: float = 1.3,
    s: float = 0.6,
    u: float = 0.2,
    nu: float = 0.0,
) -> System:
    """Return the glacial cycles of Saltzman and Maasch, a self-sustained oscillation of ice, carbon, and ocean.

    The anomalies of global ice volume `x`, atmospheric carbon dioxide `y`, and deep ocean temperature `z` follow
    `dx/dt = -x - y - u z - nu F(t)`, `dy/dt = -p z + r y - s y^2 - y^3`, and `dz/dt = -q (x + z)`. The time unit is
    10 kyr, and `F(t) = cos(2π t / 2.21) - cos(2π t / 4.41)` is a quasiperiodic orbital forcing with the periods of
    precession and obliquity. Without forcing the default parameters give a limit cycle of about 100 kyr.

    Args:
        x:
            The initial ice volume anomaly.

        y:
            The initial carbon dioxide anomaly.

        z:
            The initial deep ocean temperature anomaly.

        p:
            The effect of the deep ocean temperature on carbon dioxide.

        q:
            The relaxation rate of the deep ocean temperature to the ice volume.

        r:
            The growth rate of carbon dioxide by its own feedback.

        s:
            The quadratic saturation of carbon dioxide.

        u:
            The effect of the deep ocean temperature on ice volume.

        nu:
            The amplitude of the orbital forcing.

    Returns:
        system:
            The model, with the anomalies `climate.x`, `climate.y`, and `climate.z`.

    Examples:
        >>> from skcomplex.models.earth.climate import saltzman_glaciation
        >>> saltzman_glaciation(nu=0.5).elements[0].nu
        0.5
    """
    return System(elements=[SaltzmanMaasch('climate', x=x, y=y, z=z, p=p, q=q, r=r, s=s, u=u, nu=nu)])


@register_model(
    field='earth/climate',
    tags=('ode', 'excitability', 'carbon cycle', 'mass extinctions'),
    references=('Rothman 2019', 'Datseris and Parlitz 2022'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def carbon_cycle(
    c: float = 83.58,
    w: float = 2050.0,
    *,
    c_x: float = 55.0,
    nu: float = 0.0,
) -> System:
    """Return Rothman's excitable carbon cycle, where a large enough perturbation of the ocean sets off an excursion.

    The carbonate ion concentration `c` and the dissolved inorganic carbon `w`, in μmol kg⁻¹, follow
    `dc/dt = f(c) (mu (1 - b s(c, c_p) - theta (1 - s(c, c_x)) - nu) + w - w0)` and
    `dw/dt = mu (1 - b s(c, c_p) + theta (1 - s(c, c_x)) + nu) - w + w0`, with the sigmoids
    `s(c, c*) = c^gamma / (c^gamma + c*^gamma)` of burial and respiration and the buffer
    `f(c) = f0 c^beta / (c^beta + c_f^beta)`. The steady state near `c = 83.6` and `w = 2197` is stable, and from a
    dissolved carbon far enough below it the ocean returns only after a large excursion of `c`.

    Args:
        c:
            The initial carbonate ion concentration.

        w:
            The initial dissolved inorganic carbon.

        c_x:
            The crossover carbonate concentration of respiration, which sets the excitability.

        nu:
            The rate of external carbon injection.

    Returns:
        system:
            The model, with the concentrations `ocean.c` and `ocean.w`.

    Examples:
        >>> from skcomplex.models.earth.climate import carbon_cycle
        >>> carbon_cycle(nu=0.1).elements[0].nu
        0.1
    """
    return System(elements=[CarbonCycle('ocean', c=c, w=w, c_x=c_x, nu=nu)])


@register_model(
    field='earth/climate',
    tags=('ode', 'limit cycle', 'relaxation oscillator', 'ice ages', 'paleoclimate'),
    references=('De Saedeleer, Crucifix, and Wieczorek 2013', 'Datseris and Parlitz 2022'),
    defaults={'until': 400.0, 'dt': 0.5},
)
def van_der_pol_ice_ages(
    x: float = 0.0,
    y: float = 1.0,
    *,
    alpha: float = 30.0,
    beta: float = 0.75,
    gamma: float = 0.0,
    tau: float = 35.09,
) -> System:
    """Return glacial cycles as a Van der Pol relaxation oscillator, optionally paced by orbital forcing.

    Time is in thousands of years. The slow variable `x` stands for ice volume and the fast variable `y` for the climate
    state, with `tau x' = -(y + beta + gamma F(t))` and `tau y' = -alpha (y^3/3 - y - x)`, where
    `F(t) = cos(2 pi t/23) + cos(2 pi t/41)` stands for precession and obliquity. For `|beta| < 1` the unforced
    oscillator has a relaxation limit cycle of slow ice build-up and fast terminations, and `gamma` sets how strongly
    the orbit paces it.

    Args:
        x:
            The initial ice volume.

        y:
            The initial climate state.

        alpha:
            The time-scale separation of the fast climate state.

        beta:
            The asymmetry of the cycle, between build-up and termination.

        gamma:
            The strength of the orbital forcing.

        tau:
            The time scale, in thousands of years.

    Returns:
        system:
            The model, with the states `climate.x` and `climate.y`.

    Examples:
        >>> from skcomplex.models.earth.climate import van_der_pol_ice_ages
        >>> van_der_pol_ice_ages(gamma=0.4).elements[0].gamma
        0.4
    """
    return System(
        elements=[VanDerPolIceAges('climate', x=x, y=y, alpha=alpha, beta=beta, gamma=gamma, tau=tau)],
    )
