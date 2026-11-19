"""Resources systems: a logistic fish stock harvested by a quota.

The quota is set from a noisy measurement of the stock.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import both, normal, where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class MeasuredFisheryElement(Element):
    """Logistic fish stock with a noisy growth rate, harvested by a quota by `QuotaHarvestInteraction`.

    The quota is set from a measurement with an error. Every step of length `step` is an Euler-Maruyama step of
    `dx = (R0 + sigma_R W) x (1 - x / K) dt - gamma dt`, where `W` is white noise. The fishermen measure `x~ = x + e`,
    with a Gaussian error `e` of standard deviation `error`, and set the quota `gamma = x~ - K / 2`, which is never
    negative unless `restocking` lets a negative quota release fish. A stock that reaches zero is extinct and stays at
    zero.

    Args:
        label:
            The label of the fishery.

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

        size:
            The number of fisheries, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.resources import MeasuredFisheryElement
        >>> MeasuredFisheryElement('fishery', error=0.1).error
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        catch: object = 0.0,
        r0: Annotated[object, RealSet.open(0, oo)] = 1.0,
        sigma_r: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        K: Annotated[object, RealSet.open(0, oo)] = 1.0,
        error: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        step: Annotated[object, RealSet.open(0, oo)] = 0.01,
        restocking: Annotated[object, RealSet.closed(0, 1)] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.catch = catch
        self.r0 = r0
        self.sigma_r = sigma_r
        self.K = K
        self.error = error
        self.step = step
        self.restocking = restocking


class QuotaHarvestInteraction(Interaction):
    """Noisy logistic growth of a fish stock harvested by a quota set from a noisy measurement.

    Every step of length `step` is an Euler-Maruyama step of `dx = (R0 + sigma_R W) x (1 - x / K) dt - gamma dt`, with
    the quota `gamma = x~ - K / 2` of the measured stock `x~ = x + e`. All the values are those of the fishery.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the fishery or of the group of fisheries.

    Examples:
        >>> from skcomplex.systems.economics.resources import QuotaHarvestInteraction
        >>> QuotaHarvestInteraction('harvest', target='fisheries').target
        'fisheries'
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the Euler-Maruyama step of the stock and the quota of the step.

        Args:
            context:
                The namespace of the fishery.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            fishery = context.target
            quota = self._get_quota(fishery)
            growth = fishery.x * (1 - fishery.x / fishery.K)
            noisy = (
                fishery.x
                + (fishery.r0 * growth - quota) * fishery.step
                + fishery.sigma_r * growth * sqrt(fishery.step) * normal('growth', 0, 1)
            )
            self.laws_: list[BaseLaw] = [
                UpdateLaw(fishery.x, where(both(fishery.x > 0, noisy > 0), noisy, 0)),
                UpdateLaw(fishery.catch, quota),
            ]

    @staticmethod
    def _get_quota(fishery: SimpleNamespace) -> Expression:
        """Return the quota `x~ - K / 2` from the measured stock.

        It is floored at zero without restocking.
        """
        quota = fishery.x + normal('error', 0, 1) * fishery.error - fishery.K / 2
        return where(quota > 0, quota, fishery.restocking * quota)


def _get_extinct_share(stocks: object) -> float:
    """Return the share of the fisheries whose stock is extinct."""
    return float(np.mean(np.asarray(stocks, dtype=float) <= 0))


class FisheryMeasurementErrorsSystem(BaseSystem):
    """Independent fisheries whose quota is set from a measurement of the stock.

    The measurement has a Gaussian error. The fish stock grows logistically with a growth rate `R0 + sigma_R W(t)` that
    white noise `W` makes random, and the fishermen harvest the quota `gamma(x~) = x~ - K / 2` from their measurement
    `x~ = x + e(t)`, Problem 14.8 of Argun et al. (2021). The harvest of the deterministic system peaks at `x = K / 2`,
    at `R0 K / 4`, and the quota rule holds the stock at `K / sqrt(2)`, where the growth `R0 x (1 - x / K)` equals the
    quota. The problem asks whether the extinction probability can fall as the measurement error grows, after Argun et
    al. (2016). A quota is a catch, so it is floored at zero, and `restocking` lets a negative quota release fish as the
    formula of the source does. Every member of the group `fisheries` is an independent replica, so the share of extinct
    stocks estimates the extinction probability, and a step is an Euler-Maruyama step of length `step`.

    The system has the group `fisheries` of the stocks `fisheries.x` and the quotas `fisheries.catch`. The observables
    of the system are the share of extinct stocks `extinct`, the mean stock `mean_stock`, and the mean quota
    `mean_catch`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.resources import FisheryMeasurementErrorsSystem
        >>> FisheryMeasurementErrorsSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'economics/resources'
    topics = ('fishery', 'harvesting', 'measurement error', 'sde', 'extinction', 'feedback control')
    references = ('Argun et al. 2021', 'Argun et al. 2016')
    until = 5000
    observables = (
        Observable('extinct', group='fisheries', variable='x', reduce=_get_extinct_share),
        Observable('mean_stock', group='fisheries', variable='x', reduce='mean'),
        Observable('mean_catch', group='fisheries', variable='catch', reduce='mean'),
    )

    def __init__(
        self,
        n_fisheries: int = 200,
        *,
        x: float = 0.5,
        r0: float = 1.0,
        sigma_r: float = 1.0,
        K: float = 1.0,
        error: float = 0.0,
        step: float = 0.01,
        restocking: bool = False,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_fisheries = n_fisheries
        self.x = x
        self.r0 = r0
        self.sigma_r = sigma_r
        self.K = K
        self.error = error
        self.step = step
        self.restocking = restocking

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        fisheries = MeasuredFisheryElement(
            'fisheries',
            x=self.x,
            r0=self.r0,
            sigma_r=self.sigma_r,
            K=self.K,
            error=self.error,
            step=self.step,
            restocking=float(self.restocking),
            size=self.n_fisheries,
        )
        harvest = QuotaHarvestInteraction('harvest', target='fisheries')
        self.system_ = System(self.label, elements=[fisheries], interactions=[harvest])
