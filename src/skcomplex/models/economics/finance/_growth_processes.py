"""Finance models: random multiplicative growth of the sizes of firms or fortunes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.elements import GroupElement
from ....core.interactions import MapInteraction
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._gibrat_firm import GibratFirmElement
from ._kesten_firm import KestenFirmElement
from ._reflecting_floor import ReflectingFloorInteraction


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
