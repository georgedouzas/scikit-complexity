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
from ....core.interactions import ChoiceInteraction, FeedbackInteraction
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ..common.forecasting import heuristics
from ._patron_view import PatronView
from ._pool_spec import _PoolSpec

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


class PatronElement(Element):
    """Patron of the El Farol bar, who forecasts the attendance with its most accurate predictor.

    Every patron holds a few predictors of the pool, the set bits of `predictors`. The accuracy of a predictor is the
    discounted mean of its absolute errors, the same for every patron that holds it, since it depends only on the
    history of attendance that the bar reports. A choice then sends the patron to the bar when the forecast is below
    the capacity.

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

        discount:
            The weight of the past errors in the accuracy of a predictor.

    Examples:
        >>> from skcomplex.models.economics.econophysics import PatronElement
        >>> GroupElement('patrons', element=PatronElement, size=100, values={'predictors': 7}).size
        100
    """

    bar: ClassVar[str] = 'bar'

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        predictors: Annotated[float, StateSpec(domain=NN)] = 1,
        forecast: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        attends: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        discount: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
    ) -> None:
        super().__init__(label, visual=visual)
        self.predictors = predictors
        self.forecast = forecast
        self.attends = attends
        self.discount = discount

    @classmethod
    def _get_view(cls) -> type[PatronView]:
        """Return the view of the agents of the element."""
        return PatronView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the forecasting rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.predict, phase='forecast')]

    def _pool(self, model: ModelView) -> _PoolSpec:
        """Return this week's forecasts and errors of the pool, shared by the patrons and updated once a week."""
        pool = model.storage.get('pool')
        if isinstance(pool, _PoolSpec) and pool.time == model.time:
            return pool
        if isinstance(pool, _PoolSpec):
            seeded = pool.seeded
        else:
            seeded = model.rng.integers(0, int(MAX_ATTENDANCE) + 1, N_SEEDED_WEEKS).astype(float)
        history = np.concatenate([seeded, model.history(f'{self.bar}.attendance')])
        if isinstance(pool, _PoolSpec):
            discount = model.parameter('discount')
            errors = discount * pool.errors + (1 - discount) * np.abs(pool.forecasts - history[-1])
        else:
            errors = np.zeros(len(PREDICTORS))
        pool = _PoolSpec(model.time, seeded, forecasts(history), errors)
        model.storage['pool'] = pool
        return pool

    def predict(self, agent: PatronView, model: ModelView) -> None:
        """Forecast the attendance with the most accurate predictor held.

        Args:
            agent:
                The patron.

            model:
                The model.
        """
        pool = self._pool(model)
        mask = int(agent.predictors)
        held = [index for index in range(len(PREDICTORS)) if mask >> index & 1]
        best = held[int(np.argmin(pool.errors[held]))]
        agent.forecast = float(pool.forecasts[best])


class BarInteraction(FeedbackInteraction[ModelView]):
    """El Farol bar, which counts its patrons every week and reports the attendance.

    Args:
        label:
            The label of the bar.

        rule:
            The function of the model view that acts, unused since the bar overrides `act`.

        phase:
            The phase in which the bar counts its patrons.

        capacity:
            The attendance above which the bar is too crowded to enjoy.

        patrons:
            The path of the group of patrons.

    Examples:
        >>> from skcomplex.models.economics.econophysics import BarInteraction
        >>> BarInteraction('bar', patrons='patrons').capacity
        60.0
    """

    reports: ClassVar[tuple[str, ...]] = ('attendance',)

    def __init__(
        self,
        label: str,
        *,
        rule: Callable[[ModelView], None] | None = None,
        phase: str | None = 'count',
        capacity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 60.0,
        patrons: str = 'patrons',
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.capacity = capacity
        self.patrons = patrons

    def act(self, model: ModelView) -> None:
        """Count the patrons in the bar and report the attendance.

        Args:
            model:
                The model view of the bar.
        """
        model.report('attendance', sum(agent.attends for agent in model.agents(self.patrons, PatronView)))


@register_model(
    field='economics/econophysics',
    tags=('el farol', 'agents', 'inductive reasoning', 'bounded rationality'),
    references=('Arthur 1994', 'Yakovenko and Rosser 2009', 'Carmichael and Hadzikadic 2019 ch. 9'),
    defaults={'until': 200},
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
    its mean settles near the capacity.

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
            The model, with the group `patrons`, the choice `choice` to go when the forecast is below the capacity,
            and the institution `bar`, which reports the attendance
            `bar.attendance`.

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
        values={'predictors': masks, 'discount': discount},
    )
    choice = ChoiceInteraction(
        'choice',
        target='patrons',
        state='attends',
        options=(0.0, 1.0),
        utility=lambda agent, model, option: option if agent.forecast < model.parameters['bar.capacity'] else -option,
        rule='best',
        form='agent',
        phase='decide',
    )
    return System(
        elements=[patrons],
        interactions=[choice, BarInteraction('bar', capacity=capacity)],
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('forecast'), PhaseSpec('decide'), PhaseSpec('count')]),
        ),
    )
