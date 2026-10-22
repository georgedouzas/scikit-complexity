"""Finance models: random multiplicative growth of the sizes of firms or fortunes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.base import Domain, parameter, role, state
from ....core.model import Element, Interaction, Law, Observable, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView


def _size_observables() -> list[Observable]:
    """Return the mean and the largest size of the firms."""
    return [
        Observable('mean_size', group='firms', variable='x'),
        Observable('max_size', group='firms', reduce='max', variable='x'),
    ]


class GibratFirm(Element):
    """Firm whose size is multiplied every step by an independent normal growth factor, `x = lambda x`.

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

    Examples:
        >>> from skcomplex.models.economics.finance import GibratFirm
        >>> GibratFirm.group('firms', 10, mu=1.0).size
        10
    """

    x: float = state(default=100.0, domain=Domain.REAL)
    previous: float = state(default=100.0, domain=Domain.REAL)
    mu: float = parameter(default=1.02, domain=Domain.REAL)
    sigma: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the growth rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.grow)]

    def grow(self, agent: AgentView, model: ModelView) -> None:
        """Keep the size and multiply it by a growth factor drawn from a normal distribution.

        Args:
            agent:
                The firm.

            model:
                The model.
        """
        agent.previous = agent.x
        agent.x = agent.x * float(model.rng.normal(agent.mu, agent.sigma))


class KestenFirm(Element):
    """Firm whose size follows Kesten's process, `x = lambda x + b`, with uniform random `lambda` and `b`.

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

    Examples:
        >>> from skcomplex.models.economics.finance import KestenFirm
        >>> KestenFirm.group('firms', 10, b_high=2.0).size
        10
    """

    x: float = state(default=100.0, domain=Domain.NONNEGATIVE)
    lambda_low: float = parameter(default=0.48, domain=Domain.NONNEGATIVE)
    lambda_high: float = parameter(default=1.48, domain=Domain.NONNEGATIVE)
    b_low: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    b_high: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the growth rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.grow)]

    def grow(self, agent: AgentView, model: ModelView) -> None:
        """Multiply the size by a random factor and add a random term.

        Args:
            agent:
                The firm.

            model:
                The model.
        """
        factor = float(model.rng.uniform(agent.lambda_low, agent.lambda_high))
        agent.x = factor * agent.x + float(model.rng.uniform(agent.b_low, agent.b_high))


class ReflectingFloor(Interaction):
    """Lower bound on the sizes of a group at a fraction of their mean before the step, `x = max(x, c <x_{t-1}>)`.

    Args:
        label:
            The label of the interaction.

        firms:
            The path of the group of firms.

        c:
            The fraction of the previous mean size below which no size falls.

    Examples:
        >>> from skcomplex.models.economics.finance import ReflectingFloor
        >>> ReflectingFloor('floor', firms='firms', c=0.5).c
        0.5
    """

    firms: str = role()
    c: float = parameter(default=0.9, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the reflection rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.reflect)]

    def reflect(self, model: ModelView) -> None:
        """Raise every size below the floor to the floor.

        Args:
            model:
                The model.
        """
        firms = model.agents('firms')
        floor = model.parameter('c') * float(np.mean([firm.previous for firm in firms]))
        for firm in firms:
            firm.x = max(firm.x, floor)


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
        >>> gibrat_process(10).elements[0].size
        10
    """
    firms = GibratFirm.group('firms', n_firms, x=x, previous=x, mu=mu, sigma=sigma)
    return System(components=[firms])


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
        >>> multiplicative_process(c=0.5).interactions[0].c
        0.5
    """
    firms = GibratFirm.group('firms', n_firms, x=x, previous=x, mu=mu, sigma=sigma)
    return System(
        components=[firms],
        interactions=[ReflectingFloor('floor', firms='firms', c=c)],
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
        >>> kesten_process(10).elements[0].size
        10
    """
    firms = KestenFirm.group(
        'firms',
        n_firms,
        x=x,
        lambda_low=lambda_low,
        lambda_high=lambda_high,
        b_low=b_low,
        b_high=b_high,
    )
    return System(components=[firms])
