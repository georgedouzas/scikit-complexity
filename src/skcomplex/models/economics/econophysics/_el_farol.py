"""Econophysics models: Arthur's El Farol bar problem of inductive reasoning."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import ChoiceInteraction, ReinforcementInteraction
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import ObservableSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ..common.forecasting import heuristics
from ._patron_view import PatronView
from ._predictor_pool_view import PredictorPoolView

N_SEEDED_WEEKS = 10
MAX_ATTENDANCE = 100.0


def _lag(history: Array, weeks: int) -> float:
    """Return the attendance a number of weeks ago, which detects cycles of that period."""
    return float(history[-weeks])


def _average(history: Array, weeks: int) -> float:
    """Return the mean attendance of the last weeks."""
    return float(np.mean(history[-weeks:]))


def _fitted_trend(history: Array, weeks: int) -> float:
    """Return the next value of the least squares line through the last weeks."""
    slope, intercept = np.polyfit(np.arange(weeks), history[-weeks:], 1)
    return float(slope * weeks + intercept)


Predictor = tuple[Callable[..., float], Mapping[str, float]]

PREDICTORS: tuple[Predictor, ...] = (
    *((_lag, {'weeks': weeks}) for weeks in range(1, 7)),
    *((_average, {'weeks': weeks}) for weeks in (2, 3, 4, 5, 6, 8, 10)),
    *((_fitted_trend, {'weeks': weeks}) for weeks in (3, 4, 5, 6, 8, 10)),
    *((heuristics.trend, {'strength': strength}) for strength in (0.25, 0.5, 1.0)),
    (heuristics.anchoring, {}),
    *((heuristics.fundamental, {'fundamental': level, 'reversion': -1.0}) for level in range(40, 75, 5)),
    *((heuristics.fundamental, {'fundamental': level, 'reversion': 0.5}) for level in range(30, 95, 5)),
    *((heuristics.fundamental, {'fundamental': level, 'reversion': -0.5}) for level in (55, 60, 65)),
)
"""Pool of Arthur's predictors: cycle detectors, averages, trends, mirror images, and anchors on fixed levels."""


def forecasts(history: npt.ArrayLike) -> Array:
    """Return the forecast of next week's attendance of every predictor of the pool, between 0 and 100.

    Args:
        history:
            The past attendances, the last one last, at least ten of them.

    Returns:
        forecasts:
            One forecast per predictor of the pool.

    Examples:
        >>> from skcomplex.models.economics.econophysics import forecasts
        >>> forecasts([60.0] * 10)[:3].tolist()
        [60.0, 60.0, 60.0]
    """
    values = np.asarray(history, dtype=float)
    predicted = [function(values, **arguments) for function, arguments in PREDICTORS]
    return np.clip(predicted, 0.0, MAX_ATTENDANCE)


class PredictorPoolElement(Element):
    """Pool of Arthur's predictors, with the forecast and the accuracy of every predictor that the patrons share.

    The accuracy of a predictor depends only on the history of attendance, the observable `attendance` of the run,
    so one member holds it for every patron. Every week the pool forecasts the attendance with every predictor, from
    ten random weeks it draws at the first week followed by the observed attendances, and a
    `ReinforcementInteraction` moves the error of every predictor towards its last absolute error, a discounted mean
    of the errors.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        forecasts:
            The forecast of next week's attendance by every predictor of the pool.

        errors:
            The discounted mean absolute error of every predictor of the pool.

    Examples:
        >>> from skcomplex.models.economics.econophysics import PredictorPoolElement
        >>> GroupElement('pool', element=PredictorPoolElement, size=1).size
        1
    """

    attendance: ClassVar[str] = 'attendance'

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        forecasts: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        errors: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.forecasts = forecasts
        self.errors = errors

    @classmethod
    def _get_view(cls) -> type[PredictorPoolView]:
        """Return the view of the pool."""
        return PredictorPoolView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the forecasting rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.forecast, phase='forecast')]

    def _history(self, model: ModelView) -> Array:
        """Return the attendances, after ten random weeks drawn at the first week."""
        seeded = model.storage.get('seeded')
        if not isinstance(seeded, np.ndarray):
            seeded = model.rng.integers(0, int(MAX_ATTENDANCE) + 1, N_SEEDED_WEEKS).astype(float)
            model.storage['seeded'] = seeded
        return np.concatenate([seeded, model.history(self.attendance)])

    def forecast(self, agent: PredictorPoolView, model: ModelView) -> None:
        """Forecast next week's attendance with every predictor of the pool.

        Args:
            agent:
                The pool.

            model:
                The model.
        """
        agent.forecasts = forecasts(self._history(model))

    @staticmethod
    def error(agent: PredictorPoolView, model: ModelView, index: int) -> float:
        """Return the absolute error of a predictor last week, or its error so far before the first attendance.

        Args:
            agent:
                The pool.

            model:
                The model view of the rule that reads the error.

            index:
                The index of the predictor in the pool.

        Returns:
            error:
                The absolute error of the forecast of the predictor.
        """
        attendance = model.history(PredictorPoolElement.attendance)
        if not len(attendance):
            return float(np.ravel(agent.errors)[index])
        return float(abs(np.ravel(agent.forecasts)[index] - attendance[-1]))


class PatronElement(Element):
    """Patron of the El Farol bar, who forecasts the attendance with its most accurate predictor.

    Every patron holds a few predictors of the pool, the set bits of `predictors`, and forecasts with the one of
    smallest error in the pool. A choice then sends the patron to the bar when the forecast is below the capacity.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        predictors:
            The bit mask of the predictors of the pool that the patron holds.

        forecast:
            The last forecast of the patron's most accurate predictor.

        attends:
            Whether the patron went to the bar this week, 1 or 0.

        capacity:
            The attendance above which the bar is too crowded to enjoy.

    Examples:
        >>> from skcomplex.models.economics.econophysics import PatronElement
        >>> GroupElement('patrons', element=PatronElement, size=100, values={'predictors': 7}).size
        100
    """

    pool: ClassVar[str] = 'pool'

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        predictors: Annotated[float, StateSpec(domain=NN)] = 1,
        forecast: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        attends: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        capacity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 60.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.predictors = predictors
        self.forecast = forecast
        self.attends = attends
        self.capacity = capacity

    @classmethod
    def _get_view(cls) -> type[PatronView]:
        """Return the view of the agents of the element."""
        return PatronView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the predicting rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.predict, phase='predict')]

    def predict(self, agent: PatronView, model: ModelView) -> None:
        """Forecast the attendance with the most accurate predictor held.

        Args:
            agent:
                The patron.

            model:
                The model.
        """
        pool = model.agents(self.pool, PredictorPoolView)[0]
        errors, predicted = np.ravel(pool.errors), np.ravel(pool.forecasts)
        mask = int(agent.predictors)
        held = [index for index in range(len(PREDICTORS)) if mask >> index & 1]
        agent.forecast = float(predicted[held[int(np.argmin(errors[held]))]])


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
