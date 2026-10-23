"""Tipping types: bifurcation-, noise-, and rate-induced tipping of a stochastic energy balance and a ramped fold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ....core.base import Domain, ValidationError, parameter, state
from ....core.components import Element
from ....core.laws import Law, Noise, Rate, Symbols, ramp
from ....core.runtime import register_model
from ....core.solvers import EulerMaruyama
from ....core.system import System

C0 = 1.10869e-8
C1 = 9.7137e9
REAL_TOLERANCE = 1e-9
TIPPING_TYPES = {
    'bifurcation': {'drift': 4e-4, 'b_ini': 1.04e-5, 'b_fin': 1.04e-5, 'a_ini': 0.2, 'nu': 0.0, 'lam': 0.0},
    'noise': {'drift': 0.0, 'b_ini': 1.69e-5, 'b_fin': 1.69e-5, 'a_ini': 0.6927, 'nu': 1.0, 'lam': 0.0},
    'rate': {'drift': 0.0, 'b_ini': 1.69e-5, 'b_fin': 1.835e-5, 'a_ini': 0.6927, 'nu': 0.0, 'lam': 1e-6},
}


def _warm_state(mu: float, b2: float, a2: float) -> float:
    """Return the warm stable equilibrium of the energy balance, the largest root of `T^4 = c1 mu (b2 T^2 - a2)`."""
    root = C1 * mu * b2 + math.sqrt((C1 * mu * b2) ** 2 - 4 * C1 * mu * a2)
    return math.sqrt(root / 2)


def _lower_branch(a: float, b: float, phi: float) -> float:
    """Return the lowest real root of `a x^3 + b x + phi`, the lower branch of the fold."""
    roots = np.roots([a, 0.0, b, phi])
    return float(min(root.real for root in roots if abs(root.imag) < REAL_TOLERANCE))


class StochasticEnergyBalance(Element):
    """Global mean surface temperature of the Sutera-Fraedrich energy balance, with drifting and ramped controls.

    The temperature follows `dT = c0 [-T^4 + c1 mu (b2 T^2 - a2)] dt + sqrt(nu) dW`, in kelvin and years. The control
    `mu` drifts at the rate `-drift`, and `b2 = b_ini + lam (b_fin - b_ini)` follows the logistic ramp
    `dlam/dt = rho lam (1 - lam)`. The control `a2 = a_ini + c1 mu (b2^2 - b_ini^2) / 4` moves with `b2` so that the
    discriminant `(c1 mu b2)^2 - 4 c1 mu a2` of the equilibria stays constant during the ramp.

    Args:
        label:
            The label of the climate.

        T:
            The initial temperature in kelvin.

        mu:
            The initial relative solar input.

        lam:
            The initial progress of the ramp, between 0 and 1.

        drift:
            The rate at which `mu` decreases, per year.

        rho:
            The rate of the logistic ramp of `b2`, per year.

        b_ini:
            The initial value of `b2`, per square kelvin.

        b_fin:
            The final value of `b2`, per square kelvin.

        a_ini:
            The initial value of `a2`.

        nu:
            The variance of the noise per year, in square kelvin.

        c0:
            The scale of the energy balance, per cubic kelvin and year.

        c1:
            The scale of the absorbed radiation, in kelvin to the fourth.

    Examples:
        >>> from skcomplex.models.earth.climate import StochasticEnergyBalance
        >>> StochasticEnergyBalance('earth').nu
        1.0
    """

    T: float = state(default=290.0)
    mu: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    lam: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    drift: float = parameter(default=0.0)
    rho: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    b_ini: float = parameter(default=1.69e-5, domain=Domain.NONNEGATIVE)
    b_fin: float = parameter(default=1.69e-5, domain=Domain.NONNEGATIVE)
    a_ini: float = parameter(default=0.6927, domain=Domain.NONNEGATIVE)
    nu: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    c0: float = parameter(default=C0, domain=Domain.POSITIVE)
    c1: float = parameter(default=C1, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the energy balance with its noise, the drift of `mu`, and the logistic ramp of `b2` and `a2`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        b2 = s.b_ini + s.lam * (s.b_fin - s.b_ini)
        a2 = s.a_ini + s.c1 * s.mu * (b2**2 - s.b_ini**2) / 4
        return [
            Rate(s.T, s.c0 * (-(s.T**4) + s.c1 * s.mu * (b2 * s.T**2 - a2))),
            Noise(s.T, s.nu**0.5),
            Rate(s.mu, -s.drift),
            Rate(s.lam, s.rho * s.lam * (1 - s.lam)),
        ]


class RampedFold(Element):
    """Stochastic fold `dx = (a x^3 + b x + phi(t)) dt + sigma dW` whose control rises linearly from `phi0` to `phi1`.

    For `a < 0 < b` the fold has two stable branches for `|phi| < phi_c = sqrt(-4 b^3 / (27 a))`, and the lower
    branch vanishes at `phi = phi_c`. The control rises at the rate `rate` and stays at `phi1` once it reaches it.

    Args:
        label:
            The label of the system.

        x:
            The initial state.

        a:
            The cubic coefficient, negative.

        b:
            The linear coefficient, positive.

        sigma:
            The amplitude of the noise.

        phi0:
            The initial control.

        phi1:
            The final control.

        rate:
            The rate at which the control rises.

    Examples:
        >>> from skcomplex.models.earth.climate import RampedFold
        >>> RampedFold('system').rate
        0.004
    """

    x: float = state(default=-1.3247)
    a: float = parameter(default=-1.0)
    b: float = parameter(default=1.0)
    sigma: float = parameter(default=0.02, domain=Domain.NONNEGATIVE)
    phi0: float = -1.0
    phi1: float = 1.0
    rate: float = 0.004

    def laws(self, s: Symbols) -> list[Law]:
        """Return the fold with its ramped control and its noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        control = self.phi0 + ramp(s.t, 0.0, (self.phi1 - self.phi0) / self.rate, self.phi1 - self.phi0)
        return [Rate(s.x, s.a * s.x**3 + s.b * s.x + control), Noise(s.x, s.sigma)]


@register_model(
    field='earth/climate',
    tags=('tipping points', 'energy balance', 'sde', 'B-tipping', 'N-tipping', 'R-tipping'),
    references=('Fang, Wang, Kurths, and Fan 2025', 'Sutera 1981', 'Fraedrich 1978', 'Ashwin et al. 2012'),
    defaults={'until': 300.0, 'dt': 0.5, 'solver': EulerMaruyama(dt=0.05)},
)
def sutera_fraedrich_tipping(tipping: str = 'noise', *, rho: float = 0.18, nu: float | None = None) -> System:
    """Return the Sutera-Fraedrich stochastic energy balance set up for bifurcation-, noise-, or rate-induced tipping.

    The temperature follows `dT = c0 [-T^4 + c1 mu (b2 T^2 - a2)] dt + sqrt(nu) dW` with the parameters of Table 1 of
    Fang et al. and starts at its warm stable equilibrium. In bifurcation-induced tipping, `mu` decreases at
    `4e-4` per year until the warm equilibrium meets the unstable one in a fold near `mu = 0.76`. In noise-induced
    tipping, the controls are fixed and noise of variance `nu = 1` carries the state over the unstable equilibrium. In
    rate-induced tipping, `b2` and `a2` ramp logistically at the rate `rho` with the equilibria preserved, and the state
    fails to track the moving warm equilibrium when `rho` exceeds about 0.175. A tipped state leaves the warm branch and
    falls to the other stable root of the quartic.

    Args:
        tipping:
            The type of tipping, `'bifurcation'`, `'noise'`, or `'rate'`.

        rho:
            The rate of the logistic ramp of rate-induced tipping, per year.

        nu:
            The variance of the noise per year, or `None` for the value of the type, 1 for noise-induced tipping and
            0 otherwise.

    Returns:
        system:
            The model, with the temperature `earth.T`, the control `earth.mu`, and the progress `earth.lam` of the ramp.

    Raises:
        ValidationError:
            The type of tipping is unknown.

    Examples:
        >>> from skcomplex.models.earth.climate import sutera_fraedrich_tipping
        >>> round(sutera_fraedrich_tipping('rate').elements[0].T, 1)
        291.6
    """
    if tipping not in TIPPING_TYPES:
        error_msg = f'Unknown tipping type {tipping!r}. Use one of {", ".join(TIPPING_TYPES)}.'
        raise ValidationError(error_msg)
    values = TIPPING_TYPES[tipping]
    earth = StochasticEnergyBalance(
        'earth',
        T=_warm_state(1.0, values['b_ini'], values['a_ini']),
        lam=values['lam'],
        drift=values['drift'],
        rho=rho if tipping == 'rate' else 0.0,
        b_ini=values['b_ini'],
        b_fin=values['b_fin'],
        a_ini=values['a_ini'],
        nu=values['nu'] if nu is None else nu,
    )
    return System(components=[earth])


@register_model(
    field='earth/climate',
    tags=('tipping points', 'fold', 'ramp', 'sde', 'early warning signals', 'critical slowing down'),
    references=('Fang, Wang, Kurths, and Fan 2025', 'Scheffer et al. 2009'),
    defaults={'until': 500.0, 'dt': 0.5, 'solver': EulerMaruyama(dt=0.01)},
)
def ramped_stochastic_fold(
    *,
    rate: float = 0.004,
    sigma: float = 0.02,
    a: float = -1.0,
    b: float = 1.0,
    phi0: float = -1.0,
    phi1: float = 1.0,
) -> System:
    """Return a stochastic fold whose control is ramped through the tipping point of its lower branch.

    The state follows `dx = (a x^3 + b x + phi(t)) dt + sigma dW` with `phi = phi0 + rate t` until it reaches `phi1`,
    Eq. 4 of Fang et al. with `phi = -1 + 0.004 t`. It starts on the lower branch, which vanishes at the static fold
    `phi_c = sqrt(-4 b^3 / (27 a))`, `2 / (3 sqrt(3))` for `a = -1` and `b = 1`. Near the fold, recovery slows down,
    so the variance and the lag-1 autocorrelation of the state rise. A ramp faster than the relaxation carries the
    control past `phi_c` before the state tips, by a lag that grows with the rate.

    Args:
        rate:
            The rate at which the control rises.

        sigma:
            The amplitude of the noise.

        a:
            The cubic coefficient, negative.

        b:
            The linear coefficient, positive.

        phi0:
            The initial control.

        phi1:
            The final control.

    Returns:
        system:
            The model, with the state `system.x`, starting on the lower branch at `phi0`.

    Examples:
        >>> from skcomplex.models.earth.climate import ramped_stochastic_fold
        >>> ramped_stochastic_fold(rate=0.01).elements[0].rate
        0.01
    """
    fold = RampedFold('system', x=_lower_branch(a, b, phi0), a=a, b=b, sigma=sigma, phi0=phi0, phi1=phi1, rate=rate)
    return System(components=[fold])
