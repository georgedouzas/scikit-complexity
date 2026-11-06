"""Adaptivity: the costs of stimulus-response, adaptive response, and anticipation to recurring demands."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction, ReactionInteraction, RelaxationInteraction
from ....core.components.laws.utils import pulse_train, where
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system

STRATEGIES = {'stimulus-response': (0.0, 0.0), 'adaptive': (0.5, 0.0), 'anticipatory': (0.5, 0.5)}


class AdaptiveResponderElement(Element):
    """System that counters episodes of demand on a critical parameter and pays for its response in four costs.

    The excursion of the critical parameter follows `de/dt = (1 - a) s(t) - c e - k0 e`, where the stimulus `s` is a
    train of episodes of height `height` and width `width` every `period` from `start`, `c e` is the response of a
    mechanism of capacity `c`, and `a` is the share of every episode that anticipation avoids. The capacity grows with
    the demand and decays to its baseline, `dc/dt = eta e^2 - (c - c0) / tau`, and the response draws the reserves,
    which are restored, `dS/dt = -c e + kappa (1 - S)`. The four costs of Mobus accumulate as stocks: the response cost
    is the response effort `c e`, the restore cost the work `price kappa (1 - S)` of refilling the reserves, the repair
    cost `repair (e - theta)` while the excursion exceeds the damage threshold `theta`, and the preparedness cost the
    upkeep `upkeep (c - c0 + a)` of the extra capacity and of the anticipation.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `adaptive_response_costs` read its parameters.

    Args:
        label:
            The label of the system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.physiology import AdaptiveResponderElement
        >>> AdaptiveResponderElement('organism').period
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        e: Annotated[float, StateSpec()] = 0.0,
        c: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        S: Annotated[float, StateSpec()] = 1.0,
        response_cost: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        restore_cost: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        repair_cost: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        prepare_cost: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        anticipation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        c0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 50.0,
        k0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        price: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        theta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        repair: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        upkeep: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        height: float = 1.0,
        width: float = 2.0,
        period: float = 10.0,
        start: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.e = e
        self.c = c
        self.S = S
        self.response_cost = response_cost
        self.restore_cost = restore_cost
        self.repair_cost = repair_cost
        self.prepare_cost = prepare_cost
        self.eta = eta
        self.anticipation = anticipation
        self.c0 = c0
        self.tau = tau
        self.k0 = k0
        self.kappa = kappa
        self.price = price
        self.theta = theta
        self.repair = repair
        self.upkeep = upkeep
        self.height = height
        self.width = width
        self.period = period
        self.start = start


@register_system(
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
            The system, with the excursion `organism.e`, the capacity `organism.c`, the reserves `organism.S`, and the
            costs `organism.response_cost`, `organism.restore_cost`, `organism.repair_cost`, and
            `organism.prepare_cost`.

    Raises:
        ValidationError:
            The strategy is unknown.

    Examples:
        >>> from skcomplex.systems.biology.physiology import adaptive_response_costs
        >>> adaptive_response_costs('stimulus-response').elements[0].eta
        0.0
    """
    if strategy not in STRATEGIES:
        error_msg = f'Unknown strategy {strategy!r}. Use one of {", ".join(STRATEGIES)}.'
        raise ValidationError(error_msg)
    eta, anticipation = STRATEGIES[strategy]
    organism = AdaptiveResponderElement(
        'organism',
        eta=eta,
        anticipation=anticipation,
        height=height,
        width=width,
        period=period,
    )
    interactions = [
        RateInteraction(
            'stimulus',
            target='organism',
            state='e',
            rate=lambda s: (1 - s.target.anticipation)
            * pulse_train(s.t, organism.period, organism.width, organism.height, start=organism.start),
        ),
        ReactionInteraction(
            'effort',
            target='organism',
            reactants={'e': 1, 'S': 1},
            products={'response_cost': 1},
            propensity=lambda s: s.target.c * s.target.e,
        ),
        ReactionInteraction(
            'effort_decay',
            target='organism',
            reactants={'e': 1},
            propensity=lambda s: s.target.k0 * s.target.e,
        ),
        RateInteraction('capacity_growth', target='organism', state='c', rate=lambda s: s.target.eta * s.target.e**2),
        RelaxationInteraction(
            'capacity_relaxation',
            target='organism',
            state='c',
            goal=lambda s: s.target.c0,
            time=lambda s: s.target.tau,
        ),
        RateInteraction('restoration', target='organism', state='S', rate=lambda s: s.target.kappa * (1 - s.target.S)),
        RateInteraction(
            'restoration_cost',
            target='organism',
            state='restore_cost',
            rate=lambda s: s.target.price * s.target.kappa * (1 - s.target.S),
        ),
        RateInteraction(
            'repair_cost',
            target='organism',
            state='repair_cost',
            rate=lambda s: s.target.repair * where(s.target.e > s.target.theta, s.target.e - s.target.theta, 0),
        ),
        RateInteraction(
            'preparation_cost',
            target='organism',
            state='prepare_cost',
            rate=lambda s: s.target.upkeep * (s.target.c - s.target.c0 + s.target.anticipation),
        ),
    ]
    return System(elements=[organism], interactions=interactions)
