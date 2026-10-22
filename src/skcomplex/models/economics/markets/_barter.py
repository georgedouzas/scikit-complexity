"""Markets models: barter of two goods in an Edgeworth box and among prisoners with Red Cross parcels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, state
from ....core.components import Element
from ....core.laws import Law, Symbols
from ....core.runtime import register_model
from ....core.space import DiscreteTime, Phase, Phases
from ....core.system import Observable, System
from ..realizations import cobb_douglas
from ._goods_traders import Exchange

GOODS = ('chocolate', 'cigarettes')
WEIGHTS = tuple(f'{good}_weight' for good in GOODS)
EDGEWORTH_BARTERERS = 2


def _utility(agent: Any) -> float:  # noqa: ANN401
    """Return the logarithm of the Cobb-Douglas utility of a barterer."""
    return cobb_douglas(agent, GOODS, WEIGHTS)


def _barter_observables(system: System) -> list[Observable]:
    """Return the mean utility of the barterers and the total of both goods."""
    del system
    return [
        Observable('utility', _utility, group='barterers', reduce='mean'),
        *(Observable(good, group='barterers', variable=good, reduce='sum') for good in GOODS),
    ]


def contract_curve(
    chocolate: npt.ArrayLike,
    first_weight: float,
    second_weight: float,
    total_chocolate: float,
    total_cigarettes: float,
) -> npt.NDArray[np.float64]:
    """Return the cigarettes of the first of two barterers on the contract curve, given its chocolate.

    On the contract curve of two Cobb-Douglas barterers the marginal rates of substitution are equal,
    `a y / ((1 - a) x) = b (Y - y) / ((1 - b) (X - x))`, where `a` and `b` are the weights of chocolate, `x` and `y`
    the chocolate and cigarettes of the first barterer, and `X` and `Y` the totals.

    Args:
        chocolate:
            The chocolate of the first barterer.

        first_weight:
            The weight of chocolate of the first barterer.

        second_weight:
            The weight of chocolate of the second barterer.

        total_chocolate:
            The total chocolate of both.

        total_cigarettes:
            The total cigarettes of both.

    Returns:
        cigarettes:
            The cigarettes of the first barterer on the contract curve.

    Examples:
        >>> from skcomplex.models.economics.markets import contract_curve
        >>> float(contract_curve(5.0, 0.5, 0.5, 10.0, 10.0))
        5.0
    """
    x = np.asarray(chocolate, dtype=float)
    a, b = first_weight, second_weight
    return b * (1 - a) * x * total_cigarettes / (a * (1 - b) * (total_chocolate - x) + b * (1 - a) * x)


class Barterer(Element):
    """Barterer who holds chocolate and cigarettes and values them with Cobb-Douglas weights.

    Args:
        label:
            The label of the agent or group.

        chocolate:
            The holding of chocolate.

        cigarettes:
            The holding of cigarettes.

        chocolate_weight:
            The Cobb-Douglas weight of chocolate.

        cigarettes_weight:
            The Cobb-Douglas weight of cigarettes.

    Examples:
        >>> from skcomplex.models.economics.markets import Barterer
        >>> Barterer.group('barterers', 2, chocolate=[8.0, 2.0]).size
        2
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'chocolate_weight', 'color_range': (0.0, 1.0)})
    chocolate: float = state(default=5.0, domain=Domain.NONNEGATIVE)
    cigarettes: float = state(default=5.0, domain=Domain.NONNEGATIVE)
    chocolate_weight: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    cigarettes_weight: float = state(default=0.5, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return no laws, since the exchange moves the goods.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


@register_model(
    field='economics/markets',
    tags=('market', 'agents', 'barter', 'Edgeworth box', 'contract curve'),
    references=('Hamill and Gilbert 2016 ch. 5', 'Radford 1945'),
    defaults={'until': 20},
    observables=_barter_observables,
)
def barter(
    n_barterers: int = 2,
    *,
    chocolate: tuple[float, float] = (9.0, 1.0),
    cigarettes: tuple[float, float] = (2.0, 8.0),
    weights: tuple[float, float] = (0.6, 0.3),
    quantity: float = 0.02,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return barter of chocolate and cigarettes, in an Edgeworth box of two barterers or among many prisoners.

    Every step the barterers are paired at random, and every pair trades small quantities of one good for the other
    at the geometric mean of their marginal rates of substitution while both gain, so the totals of both goods are
    unchanged. Two barterers with the given endowments and weights of chocolate trade within the Edgeworth box until
    no trade improves both, which is a point of the contract curve, where their marginal rates of substitution are
    equal. More barterers are prisoners of war who receive random Red Cross parcels and random tastes, and their
    trades settle on a common price, as Radford (1945) observed (Hamill and Gilbert 2016 ch. 5).

    Args:
        n_barterers:
            The number of barterers, 2 for the Edgeworth box.

        chocolate:
            The chocolate of the two barterers of the Edgeworth box, and the range of the parcels of more barterers.

        cigarettes:
            The cigarettes of the two barterers of the Edgeworth box, and the range of the parcels of more
            barterers.

        weights:
            The weights of chocolate of the two barterers of the Edgeworth box, and the range of the weights of more
            barterers.

        quantity:
            The quantity of the bought good in every trade.

        random_state:
            The seed or generator of the parcels and the weights of more than two barterers.

    Returns:
        system:
            The model, with the group `barterers` and the exchange `exchange`, whose reporters are the price of
            cigarettes in chocolate `exchange.price`, its `exchange.dispersion`, and `exchange.volume`.

    Raises:
        ValueError:
            There are fewer than two barterers.

    Examples:
        >>> from skcomplex.models.economics.markets import barter
        >>> barter().elements[0].size
        2
    """
    if n_barterers < EDGEWORTH_BARTERERS:
        error_msg = f'Barter needs at least two barterers, but got {n_barterers}.'
        raise ValueError(error_msg)
    rng = np.random.default_rng(random_state)
    if n_barterers == EDGEWORTH_BARTERERS:
        holdings: dict[str, Any] = {'chocolate': list(chocolate), 'cigarettes': list(cigarettes)}
        shares = np.asarray(weights, dtype=float)
    else:
        holdings = {
            'chocolate': rng.uniform(min(chocolate), max(chocolate), n_barterers),
            'cigarettes': rng.uniform(min(cigarettes), max(cigarettes), n_barterers),
        }
        shares = rng.uniform(min(weights), max(weights), n_barterers)
    barterers = Barterer.group(
        'barterers',
        n_barterers,
        variable=True,
        chocolate_weight=shares,
        cigarettes_weight=1 - shares,
        **holdings,
    )
    return System(
        components=[barterers],
        interactions=[Exchange('exchange', traders='barterers', goods=GOODS, quantity=quantity)],
        time=DiscreteTime(schedule=Phases([Phase('trade')])),
    )
