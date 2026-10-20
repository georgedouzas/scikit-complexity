"""Econophysics models: Arthur's El Farol bar problem of inductive reasoning."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, parameter, state
from ....core.model import AgentType, Institution, Law, Phase, Rule, Symbols, System, heuristics
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

type Array = npt.NDArray[np.float64]

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


class Patron(AgentType):
    """Patron of the El Farol bar, who goes when its most accurate predictor forecasts room in the bar.

    Every patron holds a few predictors of the pool, the set bits of `predictors`. The accuracy of a predictor is the
    discounted mean of its absolute errors, the same for every patron that holds it, since it depends only on the
    history of attendance that the bar reports.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        predictors:
            The bit mask of the predictors of the pool that the patron holds.

        forecast:
            The last forecast of the patron's most accurate predictor.

        attends:
            Whether the patron went to the bar this week, 1 or 0.

        discount:
            The weight of the past errors in the accuracy of a predictor.

    Examples:
        >>> from skcomplex.models.economics.econophysics import Patron
        >>> Patron.group('patrons', 100, predictors=7).size
        100
    """

    bar: ClassVar[str] = 'bar'
    predictors: float = state(default=1, domain=Domain.NONNEGATIVE_INTEGER)
    forecast: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    attends: float = state(default=0.0, domain=Domain.BOOLEAN)
    discount: float = parameter(default=0.9, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the decision rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.decide, phase='decide')]

    def _pool(self, model: ModelView) -> dict[str, Array]:
        """Return this week's forecasts and errors of the pool, shared by the patrons and updated once a week."""
        storage = model.storage
        if storage.get('time') == model.time:
            return storage
        if 'seeded' not in storage:
            storage['seeded'] = model.rng.integers(0, int(MAX_ATTENDANCE) + 1, N_SEEDED_WEEKS).astype(float)
        history = np.concatenate([storage['seeded'], model.history(f'{self.bar}.attendance')])
        if 'forecasts' in storage:
            discount = model.parameter('discount')
            errors = np.abs(storage['forecasts'] - history[-1])
            storage['errors'] = discount * storage['errors'] + (1 - discount) * errors
        else:
            storage['errors'] = np.zeros(len(PREDICTORS))
        storage['forecasts'], storage['time'] = forecasts(history), model.time
        return storage

    def decide(self, agent: AgentView, model: ModelView) -> None:
        """Forecast the attendance with the most accurate predictor held and go when it is below the capacity.

        Args:
            agent:
                The patron.

            model:
                The model.
        """
        pool = self._pool(model)
        mask = int(agent.predictors)
        held = [index for index in range(len(PREDICTORS)) if mask >> index & 1]
        best = held[int(np.argmin(pool['errors'][held]))]
        agent.forecast = float(pool['forecasts'][best])
        agent.attends = float(agent.forecast < model.parameters[f'{self.bar}.capacity'])


class Bar(Institution):
    """El Farol bar, which counts its patrons every week and reports the attendance.

    Args:
        label:
            The label of the bar.

        patrons:
            The path of the group of patrons.

        capacity:
            The attendance above which the bar is too crowded to enjoy.

        phase:
            The phase in which the bar counts its patrons.

    Examples:
        >>> from skcomplex.models.economics.econophysics import Bar
        >>> Bar('bar', patrons='patrons').capacity
        60.0
    """

    reports: ClassVar[tuple[str, ...]] = ('attendance',)
    patrons: str = 'patrons'
    capacity: float = parameter(default=60.0, domain=Domain.NONNEGATIVE)
    phase: str | None = 'count'

    def act(self, model: ModelView) -> None:
        """Count the patrons in the bar and report the attendance.

        Args:
            model:
                The model view of the bar.
        """
        model.report('attendance', sum(agent.attends for agent in model.agents(self.patrons)))


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
    random_state: int | np.random.Generator | None = None,
) -> System:
    """Return Arthur's El Farol bar problem, whose attendance self-organizes around the capacity.

    Every patron holds a random set of predictors of next week's attendance, drawn from a pool of 46, and acts on
    its currently most accurate one. The attendance is random-looking, yet its mean settles near the capacity.

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
            The model, with the group `patrons` and the institution `bar`, which reports the attendance
            `bar.attendance`.

    Examples:
        >>> from skcomplex.models.economics.econophysics import el_farol
        >>> el_farol(10).elements[0].size
        10
    """
    rng = np.random.default_rng(random_state)
    masks = [
        float(sum(1 << int(index) for index in rng.choice(len(PREDICTORS), n_predictors, replace=False)))
        for _ in range(n_agents)
    ]
    patrons = Patron.group('patrons', n_agents, predictors=masks, discount=discount)
    return System(
        elements=[patrons],
        interactions=[Bar('bar', capacity=capacity)],
        phases=[Phase('decide'), Phase('count')],
    )
