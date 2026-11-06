"""Resources systems: a logistic fish stock harvested by a quota set from a noisy measurement of the stock."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import both, normal, where
from ....core.components.system import System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


def _extinct_share(stocks: object) -> float:
    """Return the share of the fisheries whose stock is extinct."""
    return float(np.mean(np.asarray(stocks, dtype=float) <= 0))


class MeasuredFisheryElement(Element):
    """Logistic fish stock with a noisy growth rate, harvested by a quota set from a measurement with an error.

    Every step of length `step` is an Euler-Maruyama step of `dx = (R0 + sigma_R W) x (1 - x / K) dt - gamma dt`, where
    `W` is white noise. The fishermen measure `x~ = x + e`, with a Gaussian error `e` of standard deviation `error`, and
    set the quota `gamma = x~ - K / 2`, which is never negative unless `restocking` lets a negative quota release fish.
    A stock that reaches zero is extinct and stays at zero.

    Args:
        label:
            The label of the fishery.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The fish stock.

        catch:
            The quota of the last step, per unit of time.

        r0:
            The mean growth rate `R0`.

        sigma_r:
            The standard deviation `sigma_R` of the noise of the growth rate.

        K:
            The carrying capacity.

        error:
            The standard deviation of the measurement error.

        step:
            The length of a step in units of time.

        restocking:
            1 to let a negative quota release fish into the stock, 0 to floor the quota at zero.

    Examples:
        >>> from skcomplex.systems.economics.resources import MeasuredFisheryElement
        >>> MeasuredFisheryElement('fishery', error=0.1).error
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        catch: Annotated[float, StateSpec()] = 0.0,
        r0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        sigma_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        error: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        step: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        restocking: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.catch = catch
        self.r0 = r0
        self.sigma_r = sigma_r
        self.K = K
        self.error = error
        self.step = step
        self.restocking = restocking

    @staticmethod
    def _quota(s: SimpleNamespace) -> Expression:
        """Return the quota `x~ - K / 2` from the measured stock, floored at zero without restocking."""
        quota = s.x + normal('error', 0, 1) * s.error - s.K / 2
        return where(quota > 0, quota, s.restocking * quota)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Euler-Maruyama step of the stock and the quota of the step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        quota = self._quota(s)
        growth = s.x * (1 - s.x / s.K)
        noisy = s.x + (s.r0 * growth - quota) * s.step + s.sigma_r * growth * sqrt(s.step) * normal('growth', 0, 1)
        return [UpdateLaw(s.x, where(both(s.x > 0, noisy > 0), noisy, 0)), UpdateLaw(s.catch, quota)]


@register_system(
    field='economics/resources',
    tags=('fishery', 'harvesting', 'measurement error', 'sde', 'extinction', 'feedback control'),
    references=('Argun et al. 2021', 'Argun et al. 2016'),
    defaults={'until': 5000},
    observables=lambda system: [
        ObservableSpec('extinct', group='fisheries', variable='x', reduce=_extinct_share),
        ObservableSpec('mean_stock', group='fisheries', variable='x', reduce='mean'),
        ObservableSpec('mean_catch', group='fisheries', variable='catch', reduce='mean'),
    ],
)
def fishery_measurement_errors(
    n_fisheries: int = 200,
    *,
    x: float = 0.5,
    r0: float = 1.0,
    sigma_r: float = 1.0,
    K: float = 1.0,
    error: float = 0.0,
    step: float = 0.01,
    restocking: bool = False,
) -> System:
    """Return independent fisheries whose quota is set from a measurement of the stock with a Gaussian error.

    The fish stock grows logistically with a growth rate `R0 + sigma_R W(t)` that white noise `W` makes random, and the
    fishermen harvest the quota `gamma(x~) = x~ - K / 2` from their measurement `x~ = x + e(t)`, Problem 14.8 of
    Argun et al. (2021). The harvest of the deterministic system peaks at `x = K / 2`, at `R0 K / 4`, and the quota
    rule holds the stock at `K / sqrt(2)`, where the growth `R0 x (1 - x / K)` equals the quota. The problem asks
    whether the extinction probability can fall as the measurement error grows, after Argun et al. (2016). A quota is
    a catch, so it is floored at zero, and `restocking` lets a negative quota release fish as the formula of the source
    does. Every member of the group `fisheries` is an independent replica, so the share of extinct stocks estimates
    the extinction probability, and a step is an Euler-Maruyama step of length `step`.

    Args:
        n_fisheries:
            The number of independent fisheries.

        x:
            The initial stock.

        r0:
            The mean growth rate.

        sigma_r:
            The standard deviation of the noise of the growth rate.

        K:
            The carrying capacity.

        error:
            The standard deviation of the measurement error.

        step:
            The length of a step in units of time.

        restocking:
            Whether a negative quota releases fish.

    Returns:
        system:
            The system, with the group `fisheries` of the stocks `fisheries.x` and the quotas `fisheries.catch`. The
            observables of the system are the share of extinct stocks `extinct`, the mean stock `mean_stock`, and the
            mean quota `mean_catch`.

    Examples:
        >>> from skcomplex.systems.economics.resources import fishery_measurement_errors
        >>> fishery_measurement_errors(10).elements[0].prepare().size_
        10
    """
    fisheries = GroupElement(
        'fisheries',
        element=MeasuredFisheryElement,
        size=n_fisheries,
        values={
            'x': x,
            'r0': r0,
            'sigma_r': sigma_r,
            'K': K,
            'error': error,
            'step': step,
            'restocking': float(restocking),
        },
    )
    return System(elements=[fisheries])
