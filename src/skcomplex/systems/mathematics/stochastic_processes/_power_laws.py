"""Stochastic process systems of power laws.

The power laws come from exponential growth observed at random times.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import JumpLaw, RateLaw
from ....core.components.laws.utils import draw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, RK4Solver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ObservedGrowthElement(Element):
    """Quantities `x` that grow, one per sample.

    Args:
        label:
            The label of the samples.

        x:
            The initial value, positive, one per sample or shared.

        size:
            The number of samples, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import ObservedGrowthElement
        >>> ObservedGrowthElement('growth', x=2.0).x
        2.0
    """

    def __init__(self, label: str, *, x: Annotated[float, RealSet.open(0, oo)] = 1.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x


class ObservedGrowthInteraction(Interaction):
    """The exponential growth of a quantity that is observed and restarted at exponentially distributed times.

    The quantity follows `x' = mu x` from `x0`, and after a waiting time drawn from the exponential distribution of rate
    `lam` it is observed and starts again from `x0`.

    Args:
        label:
            The label of the interaction.

        target:
            The samples that grow.

        x0:
            The value every growth starts from.

        mu:
            The growth rate.

        lam:
            The rate of the observation times.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import ObservedGrowthInteraction
        >>> ObservedGrowthInteraction('growth', target='samples', lam=2.0).lam
        2.0
    """

    law_classes = (JumpLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'samples',
        x0: float = 1.0,
        mu: float = 1.0,
        lam: float = 1.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.x0 = x0
        self.mu = mu
        self.lam = lam

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the growth and the restart at the observation times as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            waiting = draw('wait', 'expon', scale=1 / context.lam)
            self.laws_: list[BaseComponent] = [
                RateLaw(x, context.mu * x),
                JumpLaw(waiting=waiting, reset={x: context.x0}, label='observations'),
            ]


class ExponentialGrowthAtRandomTimesSystem(BaseSystem):
    """Exponential growth observed at exponentially distributed times.

    The observed values follow a power law.

    Every sample grows as `x0 exp(mu t)` and is observed and restarted after a waiting time drawn from the exponential
    distribution of rate `lam`. The observed values, and the values of the samples at any later time, follow the power
    law `p(x) ~ x^(-lam / mu - 1)` (Thurner et al. 2018, section 3.3.6.1, after Reed and Hughes 2002). The system has
    the group `samples` of state `x` and the observations `samples.observations`.

    Args:
        n_samples:
            The number of independent samples.

        mu:
            The growth rate.

        lam:
            The rate of the observation times.

        x0:
            The value every growth starts from.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import ExponentialGrowthAtRandomTimesSystem
        >>> ExponentialGrowthAtRandomTimesSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'mathematics/stochastic_processes'
    topics = ('power law', 'renewal', 'exponential growth', 'random observation')
    references = ('Reed and Hughes 2002', 'Thurner et al. 2018')
    until = 20.0
    dt = 1.0

    def __init__(
        self,
        n_samples: int = 2000,
        *,
        mu: float = 1.0,
        lam: float = 1.5,
        x0: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_samples = n_samples
        self.mu = mu
        self.lam = lam
        self.x0 = x0

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the samples and their growth."""
        super()._prepare(context)
        growth = ObservedGrowthInteraction('growth', target='samples', x0=self.x0, mu=self.mu, lam=self.lam)
        self.system_ = System(
            self.label,
            elements=[ObservedGrowthElement('samples', x=self.x0, size=self.n_samples)],
            interactions=[growth],
            solver=RK4Solver(dt=0.01),
        )
