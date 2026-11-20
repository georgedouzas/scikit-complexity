"""Econophysics systems: Arthur's El Farol bar problem of inductive reasoning."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, Protocol, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import BaseSolver, MesaSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ...general.agents import ChoiceInteraction
from ..common.forecasting import heuristics
from ..common.learning import PayoffLearningInteraction

N_SEEDED_WEEKS = 10
MAX_ATTENDANCE = 100.0


class PatronView(AgentViewProtocol, Protocol):
    """Patron as the rules of the bar see it.

    It reads the bit mask of its `predictors`, its `forecast`, whether it `attends`, and the `capacity` of the bar.
    """

    predictors: float
    forecast: float
    attends: float
    capacity: float


class PredictorPoolView(AgentViewProtocol, Protocol):
    """Pool of predictors as the rules of the bar see it.

    It holds the `forecasts` and the `errors` of its predictors.
    """

    forecasts: Array
    errors: Array


def _lag(history: Array, weeks: int) -> float:
    """Return the attendance a number of weeks ago.

    It detects cycles of that period.
    """
    return float(history[-weeks])


def _average(history: Array, weeks: int) -> float:
    """Return the mean attendance of the last weeks."""
    return float(np.mean(history[-weeks:]))


def _fitted_trend(history: Array, weeks: int) -> float:
    """Return the next value of the least squares line through the last weeks."""
    slope, intercept = np.polyfit(np.arange(weeks), history[-weeks:], 1)
    return float(slope * weeks + intercept)


PREDICTORS: tuple[tuple[Callable[..., float], Mapping[str, float]], ...] = (
    *((_lag, {'weeks': weeks}) for weeks in range(1, 7)),
    *((_average, {'weeks': weeks}) for weeks in (2, 3, 4, 5, 6, 8, 10)),
    *((_fitted_trend, {'weeks': weeks}) for weeks in (3, 4, 5, 6, 8, 10)),
    *((heuristics.trend, {'strength': strength}) for strength in (0.25, 0.5, 1.0)),
    (heuristics.anchoring, {}),
    *((heuristics.fundamental, {'fundamental': level, 'reversion': -1.0}) for level in range(40, 75, 5)),
    *((heuristics.fundamental, {'fundamental': level, 'reversion': 0.5}) for level in range(30, 95, 5)),
    *((heuristics.fundamental, {'fundamental': level, 'reversion': -0.5}) for level in (55, 60, 65)),
)


def forecasts(history: npt.ArrayLike) -> Array:
    """Return the forecast of next week's attendance of every predictor of the pool.

    It lies between 0 and 100.

    Args:
        history:
            The past attendances, the last one last, at least ten of them.

    Returns:
        forecasts:
            One forecast per predictor of the pool.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import forecasts
        >>> forecasts([60.0] * 10)[:3].tolist()
        [60.0, 60.0, 60.0]
    """
    values = np.asarray(history, dtype=float)
    predicted = [function(values, **arguments) for function, arguments in PREDICTORS]
    return np.clip(predicted, 0.0, MAX_ATTENDANCE)


class PredictorPoolElement(Element):
    """Pool of Arthur's predictors that the patrons share.

    It holds the forecast and the accuracy of every predictor. The accuracy of a predictor depends only on the history
    of attendance, the observable `attendance` of the run, so one member holds it for every patron. Every week
    `ForecastingInteraction` forecasts the attendance with every predictor, and a `PayoffLearningInteraction` moves the
    error of every predictor towards its last absolute error, a discounted mean of the errors.

    Args:
        label:
            The label of the agent or group.

        forecasts:
            The forecast of next week's attendance by every predictor of the pool.

        errors:
            The discounted mean absolute error of every predictor of the pool.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import PredictorPoolElement
        >>> PredictorPoolElement('pool', size=1).size
        1
    """

    def __init__(
        self,
        label: str,
        *,
        forecasts: npt.ArrayLike = 0.0,
        errors: npt.ArrayLike = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.forecasts = forecasts
        self.errors = errors


class ForecastingInteraction(Interaction):
    """Forecasts of next week's attendance by every predictor of the pool of the El Farol bar.

    Every week the pool forecasts the attendance with every predictor, from ten random weeks it draws at the first week
    followed by the attendances of the observable `attendance` of the run.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the pool.

        phase:
            The phase the pool forecasts in.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import ForecastingInteraction
        >>> ForecastingInteraction('forecasting', target='pool').phase
        'forecast'
    """

    law_classes = (UpdateLaw,)
    draws = True

    attendance: ClassVar[str] = 'attendance'

    def __init__(self, label: str, *, target: Role, phase: str | None = 'forecast') -> None:
        super().__init__(label)
        self.target = target
        self.phase = phase

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the forecast of the pool as its rule."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self.forecast, phase=self.phase)]

    def _get_history(self, system: SystemView) -> Array:
        """Return the attendances, after ten random weeks drawn at the first week."""
        seeded = system.storage.get('seeded')
        if not isinstance(seeded, np.ndarray):
            seeded = system.rng.integers(0, int(MAX_ATTENDANCE) + 1, N_SEEDED_WEEKS).astype(float)
            system.storage['seeded'] = seeded
        return np.concatenate([seeded, system.get_history(self.attendance)])

    def forecast(self, agent: PredictorPoolView, system: SystemView) -> None:
        """Forecast next week's attendance with every predictor of the pool.

        Args:
            agent:
                The pool.

            system:
                The system.
        """
        agent.forecasts = forecasts(self._get_history(system))

    @staticmethod
    def get_error(agent: PredictorPoolView, system: SystemView, index: int) -> float:
        """Return the absolute error of a predictor last week.

        Before the first attendance, it is its error so far.

        Args:
            agent:
                The pool.

            system:
                The system view of the rule that reads the error.

            index:
                The index of the predictor in the pool.

        Returns:
            error:
                The absolute error of the forecast of the predictor.
        """
        attendance = system.get_history(ForecastingInteraction.attendance)
        if not len(attendance):
            return float(np.ravel(agent.errors)[index])
        return float(abs(np.ravel(agent.forecasts)[index] - attendance[-1]))


class PatronElement(Element):
    """Patron of the El Farol bar.

    It forecasts the attendance with its most accurate predictor, by the rule of `PredictingInteraction`. Every patron
    holds a few predictors of the pool, the set bits of `predictors`. A choice then sends the patron to the bar when the
    forecast is below the capacity.

    Args:
        label:
            The label of the agent or group.

        predictors:
            The bit mask of the predictors of the pool that the patron holds.

        forecast:
            The last forecast of the patron's most accurate predictor.

        attends:
            Whether the patron went to the bar this week, 1 or 0.

        capacity:
            The attendance above which the bar is too crowded to enjoy.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import PatronElement
        >>> PatronElement('patrons', predictors=7, size=100).size
        100
    """

    def __init__(
        self,
        label: str,
        *,
        predictors: Annotated[object, NN] = 1,
        forecast: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        attends: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        capacity: Annotated[object, RealSet.closed_open(0, oo)] = 60.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.predictors = predictors
        self.forecast = forecast
        self.attends = attends
        self.capacity = capacity


class PredictingInteraction(Interaction):
    """Forecast of every patron of the El Farol bar with its most accurate predictor.

    Every patron forecasts with the predictor of smallest error in the pool among those it holds.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of patrons.

        pool:
            The path of the pool of predictors.

        phase:
            The phase the patrons predict in.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import PredictingInteraction
        >>> PredictingInteraction('predicting', target='patrons', pool='pool').pool
        'pool'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, pool: Role, phase: str | None = 'predict') -> None:
        super().__init__(label)
        self.target = target
        self.pool = pool
        self.phase = phase

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the forecast of every patron as its rule."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self.predict, phase=self.phase)]

    def predict(self, agent: PatronView, system: SystemView) -> None:
        """Forecast the attendance with the most accurate predictor held.

        Args:
            agent:
                The patron.

            system:
                The system.
        """
        pool = cast(list[PredictorPoolView], system.get_agents('pool'))[0]
        errors, predicted = np.ravel(pool.errors), np.ravel(pool.forecasts)
        mask = int(agent.predictors)
        held = [index for index in range(len(PREDICTORS)) if mask >> index & 1]
        agent.forecast = float(predicted[held[int(np.argmin(errors[held]))]])


class ElFarolSystem(BaseSystem):
    """Arthur's El Farol bar problem.

    Its attendance self-organizes around the capacity. Every patron holds a random set of predictors of next week's
    attendance, drawn from a pool of 46, and acts on its currently most accurate one, going when it forecasts room in
    the bar. The attendance is random-looking, yet its mean settles near the capacity. The patrons read the past
    attendances from the observable `attendance` of the run, which the catalog lists, so a run passes it in
    `observables`.

    The system has the group `patrons`, the one-member group `pool` of the forecasts and the errors of the predictors,
    the reinforcement `scoring` of the errors, and the choice `choice` to go when the forecast is below the capacity of
    the patrons.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import ElFarolSystem
        >>> ElFarolSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'economics/econophysics'
    topics = ('el farol', 'agents', 'inductive reasoning', 'bounded rationality')
    references = ('Arthur 1994', 'Yakovenko and Rosser 2009', 'Carmichael and Hadzikadic 2019 ch. 9')
    until = 200
    observables = (Observable('attendance', group='patrons', variable='attends', reduce='sum'),)

    def __init__(
        self,
        n_agents: int = 100,
        *,
        capacity: float = 60.0,
        n_predictors: int = 6,
        discount: float = 0.9,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.capacity = capacity
        self.n_predictors = n_predictors
        self.discount = discount
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        masks = [
            float(sum(1 << int(index) for index in rng.choice(len(PREDICTORS), self.n_predictors, replace=False)))
            for _ in range(self.n_agents)
        ]
        patrons = PatronElement('patrons', predictors=masks, capacity=self.capacity, size=self.n_agents)
        pool = PredictorPoolElement(
            'pool',
            forecasts=np.zeros((1, len(PREDICTORS))),
            errors=np.zeros((1, len(PREDICTORS))),
            size=1,
        )
        scoring = PayoffLearningInteraction(
            'scoring',
            target='pool',
            scores='errors',
            payoff=ForecastingInteraction.get_error,
            rate=1 - self.discount,
            schedule='synchronous',
            phase='learn',
        )
        choice = ChoiceInteraction(
            'choice',
            target='patrons',
            state='attends',
            options=(0.0, 1.0),
            utility=lambda agent, system, option: option if agent.forecast < agent.capacity else -option,
            rule='best',
            schedule='synchronous',
            phase='decide',
        )
        self.system_ = System(
            self.label,
            elements=[patrons, pool],
            interactions=[
                scoring,
                ForecastingInteraction('forecasting', target='pool'),
                PredictingInteraction('predicting', target='patrons', pool='pool'),
                choice,
            ],
            solver=MesaSolver(phases=[PhaseSpec(name) for name in ('learn', 'forecast', 'predict', 'decide')]),
        )
