"""Finance models: random multiplicative growth of the sizes of firms or fortunes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import MapInteraction
from ....core.system import System
from ....core.times import DiscreteTime
from ....engine.solvers import ModelView
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._gibrat_firm_view import GibratFirmView
from ._kesten_firm_view import KestenFirmView


def _get_gibrat_growth() -> list[MapInteraction]:
    """Return the maps that keep the size of every Gibrat firm and then grow it."""
    return [
        MapInteraction('memory', target='firms', state='previous', function=lambda agent, model: agent.x, form='agent'),
        MapInteraction('growth', target='firms', state='x', function=GibratFirmElement.get_growth, form='agent'),
    ]


def _size_observables() -> list[ObservableSpec]:
    """Return the mean and the largest size of the firms."""
    return [
        ObservableSpec('mean_size', group='firms', variable='x'),
        ObservableSpec('max_size', group='firms', reduce='max', variable='x'),
    ]


class GibratFirmElement(Element):
    """Firm whose size is multiplied every step by an independent normal growth factor, `x = lambda x`.

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
        >>> from skcomplex.models.economics.finance import GibratFirmElement
        >>> GroupElement('firms', element=GibratFirmElement, size=10, values={'mu': 1.0}).size
        10
    """

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
    def get_growth(agent: GibratFirmView, model: ModelView) -> float:
        """Return the size multiplied by a growth factor drawn from a normal distribution.

        Args:
            agent:
                The firm.

            model:
                The model view of the growth.

        Returns:
            size:
                The size after the step.
        """
        return agent.x * float(model.rng.normal(agent.mu, agent.sigma))


class KestenFirmElement(Element):
    """Firm whose size follows Kesten's process, `x = lambda x + b`, with uniform random `lambda` and `b`.

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
        >>> from skcomplex.models.economics.finance import KestenFirmElement
        >>> GroupElement('firms', element=KestenFirmElement, size=10, values={'b_high': 2.0}).size
        10
    """

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
    def get_growth(agent: KestenFirmView, model: ModelView) -> float:
        """Return the size multiplied by a random factor plus a random term.

        Args:
            agent:
                The firm.

            model:
                The model view of the growth.

        Returns:
            size:
                The size after the step.
        """
        factor = float(model.rng.uniform(agent.lambda_low, agent.lambda_high))
        return factor * agent.x + float(model.rng.uniform(agent.b_low, agent.b_high))


class ReflectingFloorInteraction(MapInteraction):
    """Lower bound on the sizes of a group at a fraction of their mean before the step, `x = max(x, c <x_{t-1}>)`.

    It is a map of the agent form that raises every size below the floor to the floor, the mean of the states
    `previous` of the group times `c`.

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
        >>> from skcomplex.models.economics.finance import ReflectingFloorInteraction
        >>> ReflectingFloorInteraction('floor', target='firms', c=0.5).check().c
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

    def _get_floored(self, agent: GibratFirmView, model: ModelView) -> float:
        """Return the size of a firm raised to the floor, `c` times the mean size of its group before the step."""
        previous = [firm.previous for firm in model.agents('target', GibratFirmView)]
        return max(agent.x, model.parameter('c') * float(np.mean(previous)))


@register_model(
    field='economics/finance',
    tags=('multiplicative process', 'lognormal', 'agents'),
    references=('Caiani et al. 2016', 'Thurner et al. 2018', 'Gibrat 1931'),
    defaults={'until': 100},
    observables=lambda system: _size_observables(),
)
def gibrat_process(
    n_firms: int = 200,
    *,
    x: float = 100.0,
    mu: float = 1.02,
    sigma: float = 0.05,
) -> System:
    """Return firms that grow by Gibrat's law of proportionate effect, which makes their sizes lognormal.

    Args:
        n_firms:
            The number of firms.

        x:
            The initial size of every firm.

        mu:
            The mean of the growth factor.

        sigma:
            The standard deviation of the growth factor.

    Returns:
        system:
            The model, with the group `firms`, the mean size `mean_size`, and the largest size `max_size`.

    Examples:
        >>> from skcomplex.models.economics.finance import gibrat_process
        >>> gibrat_process(10).elements[0].check().prepare().size_
        10
    """
    firms = GroupElement(
        'firms',
        element=GibratFirmElement,
        size=n_firms,
        values={'x': x, 'previous': x, 'mu': mu, 'sigma': sigma},
    )
    return System(elements=[firms], interactions=_get_gibrat_growth(), time=DiscreteTime())


@register_model(
    field='economics/finance',
    tags=('multiplicative process', 'power law', 'agents'),
    references=('Caiani et al. 2016', 'Thurner et al. 2018', 'Levy and Solomon 1996'),
    defaults={'until': 100},
    observables=lambda system: _size_observables(),
)
def multiplicative_process(
    n_firms: int = 200,
    *,
    x: float = 100.0,
    mu: float = 1.02,
    sigma: float = 0.05,
    c: float = 0.9,
) -> System:
    """Return Levy and Solomon's multiplicative process, whose floor at a fraction of the mean makes a power-law tail.

    Every size is multiplied by a normal growth factor and then raised to `c` times the mean size of the step before
    if it fell below it. With `c = 0` the process is Gibrat's.

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

    Returns:
        system:
            The model, with the group `firms`, the mean size `mean_size`, and the largest size `max_size`.

    Examples:
        >>> from skcomplex.models.economics.finance import multiplicative_process
        >>> multiplicative_process(c=0.5).interactions[-1].c
        0.5
    """
    firms = GroupElement(
        'firms',
        element=GibratFirmElement,
        size=n_firms,
        values={'x': x, 'previous': x, 'mu': mu, 'sigma': sigma},
    )
    return System(
        elements=[firms],
        interactions=[*_get_gibrat_growth(), ReflectingFloorInteraction('floor', target='firms', c=c)],
        time=DiscreteTime(),
    )


@register_model(
    field='economics/finance',
    tags=('multiplicative process', 'power law', 'agents'),
    references=('Caiani et al. 2016', 'Thurner et al. 2018', 'Kesten 1973'),
    defaults={'until': 100},
    observables=lambda system: _size_observables(),
)
def kesten_process(
    n_firms: int = 200,
    *,
    x: float = 100.0,
    lambda_low: float = 0.48,
    lambda_high: float = 1.48,
    b_low: float = 0.0,
    b_high: float = 1.0,
) -> System:
    """Return firms whose sizes follow Kesten's process, a random multiplicative factor plus a random positive term.

    When the factor shrinks sizes on average, the additive term reinjects them, and the stationary distribution has a
    power-law tail.

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

    Returns:
        system:
            The model, with the group `firms`, the mean size `mean_size`, and the largest size `max_size`.

    Examples:
        >>> from skcomplex.models.economics.finance import kesten_process
        >>> kesten_process(10).elements[0].check().prepare().size_
        10
    """
    firms = GroupElement(
        'firms',
        element=KestenFirmElement,
        size=n_firms,
        values={'x': x, 'lambda_low': lambda_low, 'lambda_high': lambda_high, 'b_low': b_low, 'b_high': b_high},
    )
    growth = MapInteraction('growth', target='firms', state='x', function=KestenFirmElement.get_growth, form='agent')
    return System(elements=[firms], interactions=[growth], time=DiscreteTime())
