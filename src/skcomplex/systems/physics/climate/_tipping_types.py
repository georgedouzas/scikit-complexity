"""Tipping types: bifurcation-, noise-, and rate-induced tipping of a stochastic energy balance and a ramped fold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.errors import ValidationError
from ..._catalog import register_system
from ._ramped_fold import RampedFoldElement
from ._stochastic_energy_balance import C1, StochasticEnergyBalanceElement

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


@register_system(
    field='physics/climate',
    tags=('tipping points', 'energy balance', 'sde', 'B-tipping', 'N-tipping', 'R-tipping'),
    references=('Fang, Wang, Kurths, and Fan 2025', 'Sutera 1981', 'Fraedrich 1978', 'Ashwin et al. 2012'),
    defaults={'until': 300.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.05)},
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
            The system, with the temperature `earth.T`, the control `earth.mu`, and the ramp progress `earth.lam`.

    Raises:
        ValidationError:
            The type of tipping is unknown.

    Examples:
        >>> from skcomplex.systems.physics.climate import sutera_fraedrich_tipping
        >>> round(sutera_fraedrich_tipping('rate').elements[0].T, 1)
        291.6
    """
    if tipping not in TIPPING_TYPES:
        error_msg = f'Unknown tipping type {tipping!r}. Use one of {", ".join(TIPPING_TYPES)}.'
        raise ValidationError(error_msg)
    values = TIPPING_TYPES[tipping]
    earth = StochasticEnergyBalanceElement(
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
    return System(elements=[earth])


@register_system(
    field='physics/climate',
    tags=('tipping points', 'fold', 'ramp', 'sde', 'early warning signals', 'critical slowing down'),
    references=('Fang, Wang, Kurths, and Fan 2025', 'Scheffer et al. 2009'),
    defaults={'until': 500.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
            The system, with the state `system.x`, starting on the lower branch at `phi0`.

    Examples:
        >>> from skcomplex.systems.physics.climate import ramped_stochastic_fold
        >>> ramped_stochastic_fold(rate=0.01).elements[0].rate
        0.01
    """
    fold = RampedFoldElement(
        'system',
        x=_lower_branch(a, b, phi0),
        a=a,
        b=b,
        sigma=sigma,
        phi0=phi0,
        phi1=phi1,
        rate=rate,
    )
    return System(elements=[fold])
