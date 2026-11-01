"""Markets models: duopolists that set their output from beliefs about their rival."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.elements import GroupElement
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._cournot_adjustment import CournotAdjustmentInteraction
from ._duopolist import DuopolistElement

N_FIRMS = 2


def _price(s: SimpleNamespace) -> Expression:
    """Return the market price of the total output under the inverse demand of the interaction `market`."""
    return s.market.a - s.market.b * s.total


@register_model(
    field='economics/markets',
    tags=('market', 'oligopoly', 'agents'),
    references=('Hamill and Gilbert 2016', 'Cournot 1838'),
    defaults={'until': 30},
    observables=lambda system: [
        ObservableSpec('total', group='firms', reduce='sum', variable='output'),
        ObservableSpec('price', _price),
    ],
)
def cournot_duopoly(
    outputs: Sequence[float] = (100.0, 100.0),
    costs: Sequence[float] = (0.0, 0.0),
    *,
    a: float = 1500.0,
    b: float = 10.0,
    error: float = 0.0,
) -> System:
    """Return two Cournot duopolists that adjust their output to their beliefs about each other.

    Without errors the outputs converge to the Nash equilibrium, `q_i = (a - 2 c_i + c_j) / (3 b)`, which is 50 units
    each for the defaults. Errors in the beliefs keep the total output fluctuating around it.

    Args:
        outputs:
            The initial outputs of the two firms.

        costs:
            The marginal costs of the two firms.

        a:
            The intercept of the inverse demand.

        b:
            The slope of the inverse demand.

        error:
            The largest relative error of a firm's belief about the rival's output.

    Returns:
        system:
            The model, with the group `firms`, the total output `total`, and the price `price`.

    Examples:
        >>> from skcomplex.models.economics.markets import cournot_duopoly
        >>> cournot_duopoly(error=0.01).interactions[0].error
        0.01
    """
    firms = GroupElement(
        'firms',
        element=DuopolistElement,
        size=N_FIRMS,
        values={'output': list(outputs), 'cost': list(costs)},
    )
    return System(
        elements=[firms],
        interactions=[CournotAdjustmentInteraction('market', target='firms', a=a, b=b, error=error)],
        time=DiscreteTime(),
    )
