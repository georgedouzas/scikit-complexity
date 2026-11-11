"""Finance systems: random multiplicative growth of the sizes of firms or fortunes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.spaces import DiscreteTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec


class GibratFirmView(AgentView):
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

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial size.

        previous:
            The size before the last step.

        mu:
            The mean of the growth factor.

        sigma:
            The standard deviation of the growth factor.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.finance import GibratFirmElement
        >>> GroupElement('firms', element=GibratFirmElement, size=10, values={'mu': 1.0}).size
        10
    """

    draws = True

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 100.0,
        previous: Annotated[float, StateSpec()] = 100.0,
        mu: Annotated[float, ParameterSpec()] = 1.02,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.previous = previous
        self.mu = mu
        self.sigma = sigma

    @classmethod
    def _get_view(cls) -> type[GibratFirmView]:
        """Return the view of the agents of the element."""
        return GibratFirmView

    @staticmethod
    def get_growth(agent: GibratFirmView, system: SystemView) -> float:
        """Return the size multiplied by a growth factor drawn from a normal distribution.

        Args:
            agent:
                The firm.

            system:
                The system view of the growth.

        Returns:
            size:
                The size after the step.
        """
        return agent.x * float(system.rng.normal(agent.mu, agent.sigma))


class KestenFirmView(AgentView):
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

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.finance import KestenFirmElement
        >>> GroupElement('firms', element=KestenFirmElement, size=10, values={'b_high': 2.0}).size
        10
    """

    draws = True

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        lambda_low: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.48,
        lambda_high: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.48,
        b_low: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        b_high: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.b_low = b_low
        self.b_high = b_high

    @classmethod
    def _get_view(cls) -> type[KestenFirmView]:
        """Return the view of the agents of the element."""
        return KestenFirmView

    @staticmethod
    def get_growth(agent: KestenFirmView, system: SystemView) -> float:
        """Return the size multiplied by a random factor plus a random term.

        Args:
            agent:
                The firm.

            system:
                The system view of the growth.

        Returns:
            size:
                The size after the step.
        """
        factor = float(system.rng.uniform(agent.lambda_low, agent.lambda_high))
        return factor * agent.x + float(system.rng.uniform(agent.b_low, agent.b_high))


class ReflectingFloorInteraction(MapInteraction):
    """Lower bound on the sizes of a group at a fraction of their mean.

    The mean is taken before the step, `x = max(x, c <x_{t-1}>)`. It is a map of the agent form that raises every size
    below the floor to the floor, the mean of the states `previous` of the group times `c`.

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

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, state='x', function=self._get_floored, form='agent', phase=phase)
        self.c = c

    def _get_floor(self, system: SystemView) -> float:
        """Return the floor of the step.

        It is `c` times the mean size of the group before it, computed once per step.
        """
        cached = system.storage.get('floor')
        if isinstance(cached, tuple) and cached[0] == system.time:
            return float(cached[1])
        previous = [firm.previous for firm in system.get_agents('target', GibratFirmView)]
        floor = system.get_parameter('c') * float(np.mean(previous))
        system.storage['floor'] = (system.time, floor)
        return floor

    def _get_floored(self, agent: GibratFirmView, system: SystemView) -> float:
        """Return the size of a firm raised to the floor.

        The floor is `c` times the mean size of its group before the step.
        """
        return max(agent.x, self._get_floor(system))


def _get_size_observables() -> list[ObservableSpec]:
    """Return the mean and the largest size of the firms."""
    return [
        ObservableSpec('mean_size', group='firms', variable='x'),
        ObservableSpec('max_size', group='firms', reduce='max', variable='x'),
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

    Examples:
        >>> from skcomplex.systems.economics.finance import GibratProcessSystem
        >>> GibratProcessSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'economics/finance'
    topics = ('multiplicative process', 'lognormal', 'agents')
    references = ('Caiani et al. 2016', 'Thurner et al. 2018', 'Gibrat 1931')
    until = 100
    observables = tuple(_get_size_observables())

    @staticmethod
    def _get_gibrat_growth() -> list[MapInteraction]:
        """Return the maps that keep the size of every Gibrat firm and then grow it."""
        return [
            MapInteraction(
                'memory',
                target='firms',
                state='previous',
                function=lambda agent, system: agent.x,
                form='agent',
            ),
            MapInteraction('growth', target='firms', state='x', function=GibratFirmElement.get_growth, form='agent'),
        ]

    def __init__(
        self,
        n_firms: int = 200,
        *,
        x: float = 100.0,
        mu: float = 1.02,
        sigma: float = 0.05,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_firms = n_firms
        self.x = x
        self.mu = mu
        self.sigma = sigma

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        firms = GroupElement(
            'firms',
            element=GibratFirmElement,
            size=self.n_firms,
            values={'x': self.x, 'previous': self.x, 'mu': self.mu, 'sigma': self.sigma},
        )
        return CompositeSystem(
            self.label,
            elements=[firms],
            interactions=GibratProcessSystem._get_gibrat_growth(),
            time=DiscreteTime(),
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
    ) -> None:
        super().__init__(label)
        self.n_firms = n_firms
        self.x = x
        self.mu = mu
        self.sigma = sigma
        self.c = c

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        firms = GroupElement(
            'firms',
            element=GibratFirmElement,
            size=self.n_firms,
            values={'x': self.x, 'previous': self.x, 'mu': self.mu, 'sigma': self.sigma},
        )
        return CompositeSystem(
            self.label,
            elements=[firms],
            interactions=[
                *GibratProcessSystem._get_gibrat_growth(),
                ReflectingFloorInteraction('floor', target='firms', c=self.c),
            ],
            time=DiscreteTime(),
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

    Examples:
        >>> from skcomplex.systems.economics.finance import KestenProcessSystem
        >>> KestenProcessSystem(10).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.n_firms = n_firms
        self.x = x
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.b_low = b_low
        self.b_high = b_high

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        firms = GroupElement(
            'firms',
            element=KestenFirmElement,
            size=self.n_firms,
            values={
                'x': self.x,
                'lambda_low': self.lambda_low,
                'lambda_high': self.lambda_high,
                'b_low': self.b_low,
                'b_high': self.b_high,
            },
        )
        growth = MapInteraction(
            'growth',
            target='firms',
            state='x',
            function=KestenFirmElement.get_growth,
            form='agent',
        )
        return CompositeSystem(self.label, elements=[firms], interactions=[growth], time=DiscreteTime())
