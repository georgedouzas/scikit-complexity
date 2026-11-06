"""Pool of Arthur's predictors, with the forecast and the accuracy of every predictor that the patrons share."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
import numpy.typing as npt

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import StateSpec, VisualSpec
from ....core.types import Array
from ..common.forecasting import heuristics
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


def forecasts(history: npt.ArrayLike) -> Array:
    """Return the forecast of next week's attendance of every predictor of the pool, between 0 and 100.

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.econophysics import PredictorPoolElement
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

    def _history(self, system: SystemView) -> Array:
        """Return the attendances, after ten random weeks drawn at the first week."""
        seeded = system.storage.get('seeded')
        if not isinstance(seeded, np.ndarray):
            seeded = system.rng.integers(0, int(MAX_ATTENDANCE) + 1, N_SEEDED_WEEKS).astype(float)
            system.storage['seeded'] = seeded
        return np.concatenate([seeded, system.history(self.attendance)])

    def forecast(self, agent: PredictorPoolView, system: SystemView) -> None:
        """Forecast next week's attendance with every predictor of the pool.

        Args:
            agent:
                The pool.

            system:
                The system.
        """
        agent.forecasts = forecasts(self._history(system))

    @staticmethod
    def error(agent: PredictorPoolView, system: SystemView, index: int) -> float:
        """Return the absolute error of a predictor last week, or its error so far before the first attendance.

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
        attendance = system.history(PredictorPoolElement.attendance)
        if not len(attendance):
            return float(np.ravel(agent.errors)[index])
        return float(abs(np.ravel(agent.forecasts)[index] - attendance[-1]))
