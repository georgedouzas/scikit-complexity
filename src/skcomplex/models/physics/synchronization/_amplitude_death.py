"""Amplitude death: globally coupled limit-cycle oscillators whose common fixed point the coupling restabilizes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from sage.all__sagemath_symbolics import sqrt

from ....core.base import Domain, Expression, ValidationError, VisualSpec, parameter
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Rate, Symbols, mean
from ....core.runtime import register_model
from ....core.system import Observable, System

QUANTILE_GRID_SIZE = 20001
DISTRIBUTIONS = ('lorentzian', 'quartic')


def _stuart_landau_rates(s: Symbols) -> tuple[Expression, Expression]:
    """Return the real and imaginary parts of `τ ((β + iω) z - |z|² z)` for `z = x + iy`."""
    growth = s.beta - s.x**2 - s.y**2
    return s.tau * (growth * s.x - s.omega * s.y), s.tau * (growth * s.y + s.omega * s.x)


StuartLandau = Element.define(
    'StuartLandau',
    states={'x': 1.0, 'y': 0.0},
    parameters={'beta': 1.0, 'omega': 0.0, 'tau': parameter(default=1.0, domain=Domain.POSITIVE)},
    laws=lambda s: [Rate(s.x, _stuart_landau_rates(s)[0]), Rate(s.y, _stuart_landau_rates(s)[1])],
    visual=VisualSpec('points', {'position': 'x', 'states': ()}),
)

GlobalCoupling = Interaction.define(
    'GlobalCoupling',
    roles=['target', 'source'],
    parameters={'K': parameter(default=1.0, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [
        Rate(s.target.x, s.K * (mean(s.source.x) - s.target.x)),
        Rate(s.target.y, s.K * (mean(s.source.y) - s.target.y)),
    ],
)


def _amplitude(member: Symbols) -> Expression:
    """Return the amplitude `|z|` of an oscillator."""
    return sqrt(member.x**2 + member.y**2)


def _order(s: Symbols) -> Expression:
    """Return the modulus of the mean field `X + iY`."""
    return sqrt(s.X**2 + s.Y**2)


def _time_scale_deviation_x(member: Symbols) -> Expression:
    """Return the product of the deviation of the time scale from its mean 1 and the first variable."""
    return (member.tau - 1) * member.x


def _time_scale_deviation_y(member: Symbols) -> Expression:
    """Return the product of the deviation of the time scale from its mean 1 and the second variable."""
    return (member.tau - 1) * member.y


def _quartic_cumulative(t: np.ndarray) -> np.ndarray:
    """Return the cumulative distribution of the density proportional to `1 / (t⁴ + 1)`."""
    root = math.sqrt(2)
    antiderivative = np.log((t**2 + root * t + 1) / (t**2 - root * t + 1)) + 2 * (
        np.arctan(root * t + 1) + np.arctan(root * t - 1)
    )
    return 1 / 2 + antiderivative / (4 * math.pi)


def _frequency_quantiles(size: int, width: float, distribution: str) -> np.ndarray:
    """Return the midpoint quantiles of a Lorentzian or quartic density of frequencies of a given width."""
    levels = (np.arange(size) + 1 / 2) / size
    if distribution == 'lorentzian':
        return width * np.tan(math.pi * (levels - 1 / 2))
    if distribution == 'quartic':
        grid = np.tan(np.linspace(-math.pi / 2, math.pi / 2, QUANTILE_GRID_SIZE)[1:-1])
        return width * np.interp(levels, _quartic_cumulative(grid), grid)
    message = f'The distribution must be one of {DISTRIBUTIONS}, got {distribution!r}.'
    raise ValidationError(message)


def _spread_on_circle(size: int, random_state: int | np.random.Generator | None) -> tuple[np.ndarray, np.ndarray]:
    """Return points of the unit circle at phases drawn uniformly."""
    phases = np.random.default_rng(random_state).uniform(0, 2 * math.pi, size)
    return np.cos(phases), np.sin(phases)


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'oscillators', 'amplitude death', 'mean field', 'ode'),
    references=('Ermentrout 1990', 'Mirollo and Strogatz 1990', 'Bertin 2021'),
    defaults={'until': 20.0, 'dt': 0.2},
    observables=lambda system: [
        Observable('amplitude', _amplitude, group='oscillators'),
        Observable('X', group='oscillators', variable='x'),
        Observable('Y', group='oscillators', variable='y'),
        Observable('order', _order),
    ],
)
def oscillator_death(
    n_oscillators: int = 100,
    *,
    K: float = 3.0,
    width: float = 1.5,
    distribution: str = 'lorentzian',
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return globally coupled Stuart-Landau oscillators, whose oscillations all die for strong coupling and spread.

    Oscillator `j` follows `z_j' = (1 + iω_j) z_j - |z_j|² z_j + (K / N) Σ_k (z_k - z_j)` for `z_j = x_j + i y_j`.
    Alone it settles on the unit circle. The frequencies are the midpoint quantiles of a density of width `ω0`, the
    Lorentzian `ω0 / (π (ω² + ω0²))` or the quartic `g0 / (ω⁴ + ω0⁴)`. For many oscillators, the common fixed point
    `z = 0` is stable for `K > 1` and `ω0` above a critical width, 1 for the Lorentzian and
    `(1 + √(2K - 1)) / √2` for the quartic density, and every amplitude then decays to zero. For the Lorentzian, the
    slowest perturbation decays at the rate `ω0 - 1`.

    Args:
        n_oscillators:
            The number of oscillators.

        K:
            The coupling strength.

        width:
            The width `ω0` of the density of frequencies.

        distribution:
            The density of frequencies, `'lorentzian'` or `'quartic'`.

        random_state:
            The seed or generator of the initial phases.

    Returns:
        system:
            The model, with the states `oscillators.x` and `oscillators.y` and the observables `amplitude`, the mean
            amplitude, `X` and `Y`, the mean field, and `order`, its modulus.

    Raises:
        ValidationError:
            The distribution is neither `'lorentzian'` nor `'quartic'`.

    Examples:
        >>> from skcomplex.models.physics.synchronization import oscillator_death
        >>> oscillator_death(4, distribution='quartic').elements[0].size
        4
    """
    omegas = _frequency_quantiles(n_oscillators, width, distribution)
    x, y = _spread_on_circle(n_oscillators, random_state)
    oscillators = StuartLandau.group('oscillators', n_oscillators, x=x, y=y, beta=1.0, omega=omegas, tau=1.0)
    return System(
        components=[oscillators],
        interactions=[GlobalCoupling('coupling', target='oscillators', source='oscillators', K=K)],
    )


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'oscillators', 'amplitude death', 'mean field', 'heterogeneity', 'ode'),
    references=("De Monte, d'Ovidio, and Mosekilde 2003", 'Bertin 2021'),
    defaults={'until': 40.0, 'dt': 0.2},
    observables=lambda system: [
        Observable('amplitude', _amplitude, group='systems'),
        Observable('X', group='systems', variable='x'),
        Observable('Y', group='systems', variable='y'),
        Observable('V', _time_scale_deviation_x, group='systems'),
        Observable('W', _time_scale_deviation_y, group='systems'),
    ],
)
def heterogeneous_global_coupling(
    n_systems: int = 100,
    *,
    k: float = 2.0,
    beta: float = 0.1,
    gamma: float = 2.0,
    heterogeneity: float = 0.8,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return globally coupled oscillators with heterogeneous time scales, whose fixed point the coupling restabilizes.

    System `j` follows `x_j' = τ_j g(x_j, y_j) + k (X - x_j)` and `y_j' = τ_j h(x_j, y_j) + k (Y - y_j)`, where `X`
    and `Y` are the population means and `g + ih = (beta + i gamma) z - |z|² z` is the Hopf normal form of
    `z = x + iy`. Its fixed point 0 is an unstable focus with eigenvalues `beta ± i gamma`. The time scales `τ_j` are
    spread evenly over `[1 - heterogeneity, 1 + heterogeneity]`, with mean 1 and variance `sigma²`. The order
    parameters `X`, `Y`, `V = <δτ δx>`, and `W = <δτ δy>` close on four equations. The global fixed point is stable
    when their eigenvalues `nu - k / 2 ± √(k² / 4 + sigma² nu²)`, for `nu = beta ± i gamma`, have negative real parts,
    which holds roughly for `2 beta (1 + sigma) < k < (gamma² - beta²) sigma² / beta`.

    Args:
        n_systems:
            The number of coupled systems.

        k:
            The coupling constant.

        beta:
            The growth rate `β` of the unstable focus.

        gamma:
            The rotation rate `gamma` of the unstable focus.

        heterogeneity:
            The half width of the interval of time scales, below 1.

        random_state:
            The seed or generator of the initial phases.

    Returns:
        system:
            The model, with the states `systems.x` and `systems.y` and the observables `amplitude`, the mean amplitude,
            and the order parameters `X`, `Y`, `V`, and `W`.

    Examples:
        >>> from skcomplex.models.physics.synchronization import heterogeneous_global_coupling
        >>> heterogeneous_global_coupling(4).elements[0].values['tau']
        array([0.4, 0.8, 1.2, 1.6])
    """
    taus = 1 + heterogeneity * (2 * (np.arange(n_systems) + 1 / 2) / n_systems - 1)
    x, y = _spread_on_circle(n_systems, random_state)
    radius = math.sqrt(abs(beta))
    systems = StuartLandau.group('systems', n_systems, x=radius * x, y=radius * y, beta=beta, omega=gamma, tau=taus)
    return System(
        components=[systems],
        interactions=[GlobalCoupling('coupling', target='systems', source='systems', K=k)],
    )
