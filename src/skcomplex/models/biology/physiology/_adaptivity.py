"""Adaptivity: the costs of stimulus-response, adaptive response, and anticipation to recurring demands."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, ValidationError, parameter, state
from ....core.components import Element
from ....core.laws import Law, Rate, Symbols, pulse_train, where
from ....core.runtime import register_model
from ....core.system import System

STRATEGIES = {'stimulus-response': (0.0, 0.0), 'adaptive': (0.5, 0.0), 'anticipatory': (0.5, 0.5)}


class AdaptiveResponder(Element):
    """System that counters episodes of demand on a critical parameter and pays for its response in four costs.

    The excursion of the critical parameter follows `de/dt = (1 - a) s(t) - c e - k0 e`, where the stimulus `s` is a
    train of episodes of height `height` and width `width` every `period` from `start`, `c e` is the response of a
    mechanism of capacity `c`, and `a` is the share of every episode that anticipation avoids. The capacity grows with
    the demand and decays to its baseline, `dc/dt = eta e^2 - (c - c0) / tau`, and the response draws the reserves,
    which are restored, `dS/dt = -c e + kappa (1 - S)`. The four costs of Mobus accumulate as stocks: the response cost
    is the response effort `c e`, the restore cost the work `price kappa (1 - S)` of refilling the reserves, the repair
    cost `repair (e - theta)` while the excursion exceeds the damage threshold `theta`, and the preparedness cost the
    upkeep `upkeep (c - c0 + a)` of the extra capacity and of the anticipation.

    Args:
        label:
            The label of the system.

        e:
            The initial excursion of the critical parameter.

        c:
            The initial capacity of the response.

        S:
            The initial reserves, 1 when full.

        response_cost:
            The initial response cost.

        restore_cost:
            The initial restore cost.

        repair_cost:
            The initial repair cost.

        prepare_cost:
            The initial preparedness cost.

        eta:
            The rate at which demand builds capacity, 0 for a fixed stimulus-response.

        anticipation:
            The share `a` of every episode that anticipation avoids.

        c0:
            The baseline capacity.

        tau:
            The time over which unused capacity decays.

        k0:
            The homeostatic relaxation rate of the excursion.

        kappa:
            The rate of restoring the reserves.

        price:
            The cost of restoring a unit of reserves.

        theta:
            The excursion above which damage accrues.

        repair:
            The repair cost per unit of excursion above the threshold and per unit time.

        upkeep:
            The cost per unit time of a unit of extra capacity or of anticipation.

        height:
            The height of the episodes.

        width:
            The duration of the episodes.

        period:
            The time between the starts of episodes.

        start:
            The start of the first episode.

    Examples:
        >>> from skcomplex.models.biology.physiology import AdaptiveResponder
        >>> AdaptiveResponder('organism').period
        10.0
    """

    e: float = state(default=0.0)
    c: float = state(default=0.2, domain=Domain.NONNEGATIVE)
    S: float = state(default=1.0)
    response_cost: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    restore_cost: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    repair_cost: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    prepare_cost: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    eta: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    anticipation: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    c0: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    tau: float = parameter(default=50.0, domain=Domain.POSITIVE)
    k0: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    kappa: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    price: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    theta: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    repair: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    upkeep: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    height: float = 1.0
    width: float = 2.0
    period: float = 10.0
    start: float = 1.0

    def laws(self, s: Symbols) -> list[Law]:
        """Return the excursion, the capacity, the reserves, and the four costs.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        stimulus = pulse_train(s.t, self.period, self.width, self.height, start=self.start)
        effort = s.c * s.e
        return [
            Rate(s.e, (1 - s.anticipation) * stimulus - effort - s.k0 * s.e),
            Rate(s.c, s.eta * s.e**2 - (s.c - s.c0) / s.tau),
            Rate(s.S, -effort + s.kappa * (1 - s.S)),
            Rate(s.response_cost, effort),
            Rate(s.restore_cost, s.price * s.kappa * (1 - s.S)),
            Rate(s.repair_cost, s.repair * where(s.e > s.theta, s.e - s.theta, 0)),
            Rate(s.prepare_cost, s.upkeep * (s.c - s.c0 + s.anticipation)),
        ]


@register_model(
    field='biology/physiology',
    tags=('adaptation', 'anticipation', 'homeostasis', 'costs', 'stocks and flows', 'ode'),
    references=('Mobus 2022', 'Mobus 1999'),
    defaults={'until': 200.0, 'dt': 0.1},
)
def adaptive_response_costs(
    strategy: str = 'adaptive',
    *,
    height: float = 1.0,
    width: float = 2.0,
    period: float = 10.0,
) -> System:
    """Return a system that meets recurring demands by stimulus-response, adaptive response, or anticipation.

    Episodes of demand push a critical parameter away from its norm, and the system counters them, after Mobus. A
    stimulus-response keeps its baseline capacity, so large episodes overshoot the damage threshold and cost repairs.
    An adaptive response builds capacity under repeated demand and loses it when demand stops, at the price of its
    upkeep. An anticipatory response also predicts every episode and avoids half of it. Mobus's objective is the sum
    of the response, restore, and repair costs over the episodes, with the preparedness cost amortized over them:
    adaptive response lowers it below stimulus-response by avoiding repairs, and anticipation lowers it further by
    cutting the response it needs.

    Args:
        strategy:
            `'stimulus-response'`, `'adaptive'`, or `'anticipatory'`.

        height:
            The height of the episodes of demand.

        width:
            The duration of the episodes.

        period:
            The time between the starts of episodes.

    Returns:
        system:
            The model, with the excursion `organism.e`, the capacity `organism.c`, the reserves `organism.S`, and the
            costs `organism.response_cost`, `organism.restore_cost`, `organism.repair_cost`, and
            `organism.prepare_cost`.

    Raises:
        ValidationError:
            The strategy is unknown.

    Examples:
        >>> from skcomplex.models.biology.physiology import adaptive_response_costs
        >>> adaptive_response_costs('stimulus-response').elements[0].eta
        0.0
    """
    if strategy not in STRATEGIES:
        error_msg = f'Unknown strategy {strategy!r}. Use one of {", ".join(STRATEGIES)}.'
        raise ValidationError(error_msg)
    eta, anticipation = STRATEGIES[strategy]
    organism = AdaptiveResponder(
        'organism',
        eta=eta,
        anticipation=anticipation,
        height=height,
        width=width,
        period=period,
    )
    return System(components=[organism])
