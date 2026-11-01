"""Econophysics models: Arthur's El Farol bar problem of inductive reasoning."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import ChoiceInteraction, ReinforcementInteraction
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....specs import ObservableSpec, PhaseSpec, PhasesScheduleSpec
from ..._catalog import register_model
from ._patron import PatronElement
from ._predictor_pool import PREDICTORS, PredictorPoolElement

"""Pool of Arthur's predictors: cycle detectors, averages, trends, mirror images, and anchors on fixed levels."""


@register_model(
    field='economics/econophysics',
    tags=('el farol', 'agents', 'inductive reasoning', 'bounded rationality'),
    references=('Arthur 1994', 'Yakovenko and Rosser 2009', 'Carmichael and Hadzikadic 2019 ch. 9'),
    defaults={'until': 200},
    observables=lambda system: [ObservableSpec('attendance', group='patrons', variable='attends', reduce='sum')],
)
def el_farol(
    n_agents: int = 100,
    *,
    capacity: float = 60.0,
    n_predictors: int = 6,
    discount: float = 0.9,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Arthur's El Farol bar problem, whose attendance self-organizes around the capacity.

    Every patron holds a random set of predictors of next week's attendance, drawn from a pool of 46, and acts on
    its currently most accurate one, going when it forecasts room in the bar. The attendance is random-looking, yet
    its mean settles near the capacity. The patrons read the past attendances from the observable `attendance` of
    the run, which the catalog lists, so a run passes it in `observables`.

    Args:
        n_agents:
            The number of patrons.

        capacity:
            The attendance above which the bar is too crowded to enjoy.

        n_predictors:
            The number of predictors every patron holds.

        discount:
            The weight of past errors in the accuracy of a predictor.

        random_state:
            The seed or generator of the predictors that every patron holds.

    Returns:
        system:
            The model, with the group `patrons`, the one-member group `pool` of the forecasts and the errors of the
            predictors, the reinforcement `scoring` of the errors, and the choice `choice` to go when the forecast is
            below the capacity of the patrons.

    Examples:
        >>> from skcomplex.models.economics.econophysics import el_farol
        >>> el_farol(10).elements[0].check().prepare().size_
        10
    """
    rng = np.random.default_rng(random_state)
    masks = [
        float(sum(1 << int(index) for index in rng.choice(len(PREDICTORS), n_predictors, replace=False)))
        for _ in range(n_agents)
    ]
    patrons = GroupElement(
        'patrons',
        element=PatronElement,
        size=n_agents,
        values={'predictors': masks, 'capacity': capacity},
    )
    pool = GroupElement(
        'pool',
        element=PredictorPoolElement,
        size=1,
        values={'forecasts': np.zeros((1, len(PREDICTORS))), 'errors': np.zeros((1, len(PREDICTORS)))},
    )
    scoring = ReinforcementInteraction(
        'scoring',
        target='pool',
        scores='errors',
        payoff=PredictorPoolElement.error,
        rate=1 - discount,
        form='agent',
        phase='learn',
    )
    choice = ChoiceInteraction(
        'choice',
        target='patrons',
        state='attends',
        options=(0.0, 1.0),
        utility=lambda agent, model, option: option if agent.forecast < agent.capacity else -option,
        rule='best',
        form='agent',
        phase='decide',
    )
    return System(
        elements=[patrons, pool],
        interactions=[scoring, choice],
        time=DiscreteTime(
            schedule=PhasesScheduleSpec(
                [PhaseSpec(name) for name in ('learn', 'forecast', 'predict', 'decide')],
            ),
        ),
    )
