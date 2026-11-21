"""Finance systems: random multiplicative growth of the sizes of firms or fortunes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role


class GibratFirmView(AgentViewProtocol, Protocol):
    """Firm of Gibrat's process as the rules see it, with its size and growth factor.

    It reads its size `x`, its `previous` size, and the mean `mu` and the deviation `sigma` of its growth factor.
    """

    x: float
    previous: float
    mu: float
    sigma: float


class GibratFirmElement(Element):
    """Firm whose size is multiplied every step by a normal growth factor.

    The factor is independent, `x = lambda x`.

    Args:
        label:
            The label of the agent or group.

        x:
            The initial size.

        previous:
            The size before the last step.

        mu:
            The mean of the growth factor.

        sigma:
            The standard deviation of the growth factor.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.finance import GibratFirmElement
        >>> GibratFirmElement('firms', mu=1.0, size=10).size
        10
    """

    draws = True

    def __init__(
        self,
        label: str,
        *,
        x: object = 100.0,
        previous: object = 100.0,
        mu: object = 1.02,
        sigma: Annotated[object, RealSet.closed_open(0, oo)] = 0.05,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.previous = previous
        self.mu = mu
        self.sigma = sigma


class KestenFirmView(AgentViewProtocol, Protocol):
    """Firm of Kesten's process as its rule sees it.

    It holds its size `x` and the bounds of its random factor and term.
    """

    x: float
    lambda_low: float
    lambda_high: float
    b_low: float
    b_high: float


class KestenFirmElement(Element):
    """Firm whose size follows Kesten's process, `x = lambda x + b`.

    The `lambda` and `b` are uniform random.

    Args:
        label:
            The label of the agent or group.

        x:
            The initial size.

        lambda_low:
            The lower bound of the multiplicative factor.

        lambda_high:
            The upper bound of the multiplicative factor.

        b_low:
            The lower bound of the additive term.

        b_high:
            The upper bound of the additive term.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.finance import KestenFirmElement
        >>> KestenFirmElement('firms', b_high=2.0, size=10).size
        10
    """

    draws = True

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, RealSet.closed_open(0, oo)] = 100.0,
        lambda_low: Annotated[object, RealSet.closed_open(0, oo)] = 0.48,
        lambda_high: Annotated[object, RealSet.closed_open(0, oo)] = 1.48,
        b_low: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        b_high: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.b_low = b_low
        self.b_high = b_high


class GibratGrowthInteraction(Interaction):
    """Growth of the sizes of firms by independent normal factors, `x = lambda x`.

    Every step every firm first keeps its size as `previous`, and then multiplies it by a factor drawn from a normal
    distribution of mean `mu` and deviation `sigma`, both values of the firm.

    Args:
        label:
            The label of the interaction.

        target:
            The firms, with their sizes `x` and `previous` and their `mu` and `sigma`.

    Examples:
        >>> from skcomplex.systems.economics.finance import GibratGrowthInteraction
        >>> GibratGrowthInteraction('growth').target
        'firms'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'firms') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the memory of the sizes and their growth as the rules of the system.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [UpdateLaw(context, self._remember), UpdateLaw(context, self._grow)]

    def _remember(self, system: SystemView) -> None:
        """Keep the size of every firm as its previous size."""
        for firm in cast(list[GibratFirmView], system.get_agents('target')):
            firm.previous = firm.x

    def _grow(self, system: SystemView) -> None:
        """Multiply the size of every firm by a normal growth factor."""
        firms = cast(list[GibratFirmView], system.get_agents('target'))
        sizes = [firm.x * float(system.rng.normal(firm.mu, firm.sigma)) for firm in firms]
        for firm, size in zip(firms, sizes, strict=True):
            firm.x = size


class KestenGrowthInteraction(Interaction):
    """Growth of the sizes of firms by Kesten's process, `x = lambda x + b`.

    Every step every firm multiplies its size by a factor drawn uniformly between `lambda_low` and `lambda_high` and
    adds a term drawn uniformly between `b_low` and `b_high`, all values of the firm.

    Args:
        label:
            The label of the interaction.

        target:
            The firms, with their sizes `x` and the bounds of their factors and terms.

    Examples:
        >>> from skcomplex.systems.economics.finance import KestenGrowthInteraction
        >>> KestenGrowthInteraction('growth').target
        'firms'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'firms') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the growth of the sizes as the rule of the system.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [UpdateLaw(context, self._grow)]

    def _grow(self, system: SystemView) -> None:
        """Multiply the size of every firm by a random factor and add a random term."""
        firms = cast(list[KestenFirmView], system.get_agents('target'))
        sizes = []
        for firm in firms:
            factor = float(system.rng.uniform(firm.lambda_low, firm.lambda_high))
            sizes.append(factor * firm.x + float(system.rng.uniform(firm.b_low, firm.b_high)))
        for firm, size in zip(firms, sizes, strict=True):
            firm.x = size


class ReflectingFloorInteraction(Interaction):
    """Lower bound on the sizes of a group at a fraction of their mean.

    The mean is taken before the step, `x = max(x, c <x_{t-1}>)`. It is a rule of the system that raises every size
    below the floor to the floor, the mean of the values `previous` of the group times `c`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of firms.

        c:
            The fraction of the previous mean size below which no size falls.

        phase:
            The phase of the step the floor acts in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.finance import ReflectingFloorInteraction
        >>> ReflectingFloorInteraction('floor', target='firms', c=0.5).c
        0.5
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        c: Annotated[float, RealSet.closed_open(0, oo)] = 0.9,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.c = c
        self.phase = phase

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the floor of the sizes as the rule of the system, in its phase.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [UpdateLaw(context, self._raise, phase=self.phase)]

    def _raise(self, system: SystemView) -> None:
        """Raise every size below `c` times the mean size of the group before the step to that floor."""
        firms = cast(list[GibratFirmView], system.get_agents('target'))
        floor = system.get_parameter('c') * float(np.mean([firm.previous for firm in firms]))
        for firm in firms:
            firm.x = max(firm.x, floor)


def _get_size_observables() -> list[Observable]:
    """Return the mean and the largest size of the firms."""
    return [
        Observable('mean_size', group='firms', variable='x'),
        Observable('max_size', group='firms', reduce='max', variable='x'),
    ]


class GibratProcessSystem(BaseSystem):
    """Firms that grow by Gibrat's law of proportionate effect.

    The law makes their sizes lognormal. The system has the group `firms`, the mean size `mean_size`, and the largest
    size `max_size`.

    Args:
        n_firms:
            The number of firms.

        x:
            The initial size of every firm.

        mu:
            The mean of the growth factor.

        sigma:
            The standard deviation of the growth factor.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.finance import GibratProcessSystem
        >>> GibratProcessSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'economics/finance'
    topics = ('multiplicative process', 'lognormal', 'agents')
    references = ('Caiani et al. 2016', 'Thurner et al. 2018', 'Gibrat 1931')
    until = 100
    observables = tuple(_get_size_observables())

    def __init__(
        self,
        n_firms: int = 200,
        *,
        x: float = 100.0,
        mu: float = 1.02,
        sigma: float = 0.05,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_firms = n_firms
        self.x = x
        self.mu = mu
        self.sigma = sigma

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        firms = GibratFirmElement('firms', x=self.x, previous=self.x, mu=self.mu, sigma=self.sigma, size=self.n_firms)
        self.system_ = System(
            self.label,
            elements=[firms],
            interactions=[GibratGrowthInteraction('growth', target='firms')],
        )


class MultiplicativeProcessSystem(BaseSystem):
    """Levy and Solomon's multiplicative process.

    Its floor at a fraction of the mean makes a power-law tail. Every size is multiplied by a normal growth factor and
    then raised to `c` times the mean size of the step before if it fell below it. With `c = 0` the process is Gibrat's.
    The system has the group `firms`, the mean size `mean_size`, and the largest size `max_size`.

    Args:
        n_firms:
            The number of firms.

        x:
            The initial size of every firm.

        mu:
            The mean of the growth factor.

        sigma:
            The standard deviation of the growth factor.

        c:
            The fraction of the previous mean size below which no size falls.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.finance import MultiplicativeProcessSystem
        >>> MultiplicativeProcessSystem(c=0.5).prepare().system_.interactions[-1].c
        0.5
    """

    field = 'economics/finance'
    topics = ('multiplicative process', 'power law', 'agents')
    references = ('Caiani et al. 2016', 'Thurner et al. 2018', 'Levy and Solomon 1996')
    until = 100
    observables = tuple(_get_size_observables())

    def __init__(
        self,
        n_firms: int = 200,
        *,
        x: float = 100.0,
        mu: float = 1.02,
        sigma: float = 0.05,
        c: float = 0.9,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_firms = n_firms
        self.x = x
        self.mu = mu
        self.sigma = sigma
        self.c = c

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        firms = GibratFirmElement('firms', x=self.x, previous=self.x, mu=self.mu, sigma=self.sigma, size=self.n_firms)
        self.system_ = System(
            self.label,
            elements=[firms],
            interactions=[
                GibratGrowthInteraction('growth', target='firms'),
                ReflectingFloorInteraction('floor', target='firms', c=self.c),
            ],
        )


class KestenProcessSystem(BaseSystem):
    """Firms whose sizes follow Kesten's process.

    It is a random multiplicative factor plus a random positive term. When the factor shrinks sizes on average, the
    additive term reinjects them, and the stationary distribution has a power-law tail. The system has the group
    `firms`, the mean size `mean_size`, and the largest size `max_size`.

    Args:
        n_firms:
            The number of firms.

        x:
            The initial size of every firm.

        lambda_low:
            The lower bound of the multiplicative factor.

        lambda_high:
            The upper bound of the multiplicative factor.

        b_low:
            The lower bound of the additive term.

        b_high:
            The upper bound of the additive term.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.finance import KestenProcessSystem
        >>> KestenProcessSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'economics/finance'
    topics = ('multiplicative process', 'power law', 'agents')
    references = ('Caiani et al. 2016', 'Thurner et al. 2018', 'Kesten 1973')
    until = 100
    observables = tuple(_get_size_observables())

    def __init__(
        self,
        n_firms: int = 200,
        *,
        x: float = 100.0,
        lambda_low: float = 0.48,
        lambda_high: float = 1.48,
        b_low: float = 0.0,
        b_high: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_firms = n_firms
        self.x = x
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.b_low = b_low
        self.b_high = b_high

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        firms = KestenFirmElement(
            'firms',
            x=self.x,
            lambda_low=self.lambda_low,
            lambda_high=self.lambda_high,
            b_low=self.b_low,
            b_high=self.b_high,
            size=self.n_firms,
        )
        growth = KestenGrowthInteraction('growth', target='firms')
        self.system_ = System(self.label, elements=[firms], interactions=[growth])
